//! Keeping fragment vectors current, and searching them.
//!
//! [`embed_pending`] finds live fragments whose vectors are missing, from a
//! different model, or for older content, and embeds them in batches. The
//! model runs outside the writer thread so writes never wait on it; only
//! storing the finished batch is a write. Stale fragments are skipped
//! (recall never returns them) and picked up if they become live again.
//!
//! The model reads at most ~510 tokens and silently drops the rest, so a
//! long fragment is embedded as several overlapping parts of about
//! [`BODY_TOKENS`] tokens. Every part starts with a one-line header naming
//! the fragment's project and type: the chunk-size benchmark
//! (`research/chunk-bench`) found 400 tokens + header beat the old
//! whole-fragment vectors on every hybrid metric at about half the tokens.
//!
//! [`VectorIndex`] holds the part vectors in memory as one contiguous matrix
//! and ranks fragments by their best part (dot product = cosine, since
//! vectors are unit length). For the thousands of fragments a person
//! accumulates this takes well under a millisecond; an ANN index
//! (sqlite-vec) is only worth it far beyond that.

use std::collections::{HashMap, HashSet};

use rusqlite::{Connection, params};
use sha2::{Digest, Sha256};
use wevex_embed::Embedder;

use crate::Result;
use crate::storage::Store;
use crate::storage::fragments::LIVE;

/// Fragments embedded per write transaction.
pub const BATCH: usize = 32;

/// Target size of one part's body, in model tokens (header excluded).
pub const BODY_TOKENS: usize = 400;

/// Version of the text layout fed to the model (header + body). Bump it
/// when [`header`] or the splitting changes, so old vectors are redone.
pub const TEXT_FORMAT: &str = "h1";

/// Key stored in `fragment_vectors.model`: the embedder plus the text
/// layout, so vectors made differently are never compared.
pub fn model_key(embedder: &dyn Embedder) -> String {
    format!("{}/{TEXT_FORMAT}", embedder.id())
}

pub fn content_hash(text: &str) -> String {
    Sha256::digest(text.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

pub fn encode(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn decode(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The line put before every part, e.g. `"wevex-rs · decision"`.
pub fn header(scope_name: &str, kind: &str) -> String {
    format!("{scope_name} · {kind}")
}

/// Split `text` into overlapping byte ranges of at most `body` tokens.
/// `starts` holds the byte offset of each token (from
/// [`Embedder::token_starts`]). Consecutive parts share about 1/8 of their
/// tokens so a sentence cut at a boundary is whole in one of them. Text that
/// fits returns one range covering all of it.
pub fn split(text: &str, starts: &[usize], body: usize) -> Vec<(usize, usize)> {
    let body = body.max(1);
    if starts.len() <= body {
        return vec![(0, text.len())];
    }
    let step = (body - body / 8).max(1);
    let mut parts = Vec::new();
    let mut i = 0;
    loop {
        let from = if i == 0 { 0 } else { starts[i] };
        let end = i + body;
        if end >= starts.len() {
            parts.push((from, text.len()));
            return parts;
        }
        parts.push((from, starts[end]));
        i += step;
    }
}

pub struct Pending {
    pub id: String,
    /// Header line for every part.
    pub header: String,
    pub content: String,
    /// Hash of header + content: a rename of the project redoes the vectors.
    pub hash: String,
}

/// Up to `limit` live fragments whose vectors are missing or out of date for
/// `model` (a [`model_key`]). Most recently created first, so new work is
/// searchable soonest.
pub fn pending(conn: &Connection, model: &str, limit: usize) -> Result<Vec<Pending>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT f.id, f.content, f.type, COALESCE(s.name, f.scope_id), v.model, v.content_hash
           FROM fragments f
           LEFT JOIN scopes s ON s.id = f.scope_id
           LEFT JOIN fragment_vectors v ON v.fragment_id = f.id AND v.part = 0
          WHERE {LIVE}
          ORDER BY julianday(f.created_at) DESC"
    ))?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        let content: String = r.get(1)?;
        let header = header(&r.get::<_, String>(3)?, &r.get::<_, String>(2)?);
        let hash = content_hash(&format!("{header}\n{content}"));
        let current = r.get::<_, Option<String>>(4)?.as_deref() == Some(model)
            && r.get::<_, Option<String>>(5)?.as_deref() == Some(hash.as_str());
        if !current {
            out.push(Pending {
                id: r.get(0)?,
                header,
                content,
                hash,
            });
            if out.len() == limit {
                break;
            }
        }
    }
    Ok(out)
}

/// The texts to embed for one fragment: header + each part of the body.
pub fn part_texts(embedder: &dyn Embedder, p: &Pending) -> Result<Vec<String>> {
    let header_tokens = embedder.token_starts(&p.header)?.len() + 1;
    let body = BODY_TOKENS.min(embedder.max_tokens().saturating_sub(header_tokens));
    let starts = embedder.token_starts(&p.content)?;
    Ok(split(&p.content, &starts, body)
        .into_iter()
        .map(|(a, b)| format!("{}\n{}", p.header, p.content[a..b].trim()))
        .collect())
}

/// Embed every pending fragment. Returns how many fragments were embedded.
/// Safe to interrupt and rerun: each batch is committed on its own, and a
/// fragment's parts are replaced together.
pub fn embed_pending(store: &Store, embedder: &dyn Embedder) -> Result<usize> {
    let model = model_key(embedder);
    let mut done = 0;
    loop {
        let batch = {
            let model = model.clone();
            store.read(move |c| pending(c, &model, BATCH))?
        };
        if batch.is_empty() {
            return Ok(done);
        }
        // One model call for every part of every fragment in the batch.
        let mut texts = Vec::new();
        let mut counts = Vec::with_capacity(batch.len());
        for p in &batch {
            let parts = part_texts(embedder, p)?;
            counts.push(parts.len());
            texts.extend(parts);
        }
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let mut vectors = embedder.embed(&refs)?.into_iter();
        let rows: Vec<(String, String, Vec<Vec<u8>>)> = batch
            .into_iter()
            .zip(counts)
            .map(|(p, n)| {
                (
                    p.id,
                    p.hash,
                    vectors.by_ref().take(n).map(|v| encode(&v)).collect(),
                )
            })
            .collect();
        let n = rows.len();
        let model = model.clone();
        store.write(move |c| {
            let tx = c.transaction()?;
            {
                let mut clear =
                    tx.prepare("DELETE FROM fragment_vectors WHERE fragment_id = ?1")?;
                let mut insert = tx.prepare(
                    "INSERT INTO fragment_vectors (fragment_id, part, model, content_hash, vector)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                )?;
                for (id, hash, parts) in &rows {
                    clear.execute([id])?;
                    for (part, blob) in parts.iter().enumerate() {
                        insert.execute(params![id, part as i64, model, hash, blob])?;
                    }
                }
            }
            tx.commit()?;
            Ok(())
        })?;
        done += n;
    }
}

/// In-memory part vectors of live fragments for one model key. One row per
/// part; a fragment scores as its best part.
pub struct VectorIndex {
    model: String,
    dim: usize,
    /// Fragment id of each row.
    ids: Vec<String>,
    scopes: Vec<String>,
    data: Vec<f32>,
    /// Rows of each fragment.
    pos: HashMap<String, Vec<usize>>,
}

impl VectorIndex {
    pub fn empty(model: &str, dim: usize) -> Self {
        Self {
            model: model.to_owned(),
            dim,
            ids: Vec::new(),
            scopes: Vec::new(),
            data: Vec::new(),
            pos: HashMap::new(),
        }
    }

    /// Load the current vectors of live fragments for `model` (a
    /// [`model_key`]).
    pub fn load(conn: &Connection, model: &str, dim: usize) -> Result<Self> {
        let mut index = Self::empty(model, dim);
        let mut stmt = conn.prepare(&format!(
            "SELECT f.id, f.scope_id, v.vector
               FROM fragments f JOIN fragment_vectors v ON v.fragment_id = f.id
              WHERE v.model = ?1 AND {LIVE}
              ORDER BY f.id, v.part"
        ))?;
        let mut rows = stmt.query([model])?;
        while let Some(r) = rows.next()? {
            let v = decode(&r.get::<_, Vec<u8>>(2)?);
            if v.len() == dim {
                index.push(r.get(0)?, r.get(1)?, &v);
            }
        }
        Ok(index)
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    /// Number of fragments (not parts).
    pub fn len(&self) -> usize {
        self.pos.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pos.is_empty()
    }

    fn push(&mut self, id: String, scope_id: String, v: &[f32]) {
        self.pos.entry(id.clone()).or_default().push(self.ids.len());
        self.ids.push(id);
        self.scopes.push(scope_id);
        self.data.extend_from_slice(v);
    }

    /// Replace all of a fragment's part vectors.
    pub fn upsert(&mut self, id: &str, scope_id: &str, parts: &[Vec<f32>]) {
        assert!(
            parts.iter().all(|v| v.len() == self.dim),
            "vector dimension mismatch"
        );
        self.remove(id);
        for v in parts {
            self.push(id.to_owned(), scope_id.to_owned(), v);
        }
    }

    /// Remove a fragment. Each of its rows is filled by swapping the last
    /// row into the gap: O(parts), row order not kept.
    pub fn remove(&mut self, id: &str) {
        let Some(mut rows) = self.pos.remove(id) else {
            return;
        };
        // Highest first, so a row still to be removed is never the one
        // moved into a gap.
        rows.sort_unstable_by(|a, b| b.cmp(a));
        for i in rows {
            let last = self.ids.len() - 1;
            if i != last {
                self.ids.swap(i, last);
                self.scopes.swap(i, last);
                let (head, tail) = self.data.split_at_mut(last * self.dim);
                head[i * self.dim..(i + 1) * self.dim].copy_from_slice(&tail[..self.dim]);
                let moved = self
                    .pos
                    .get_mut(&self.ids[i])
                    .expect("moved row is indexed");
                *moved.iter_mut().find(|r| **r == last).expect("row listed") = i;
            }
            self.ids.pop();
            self.scopes.pop();
            self.data.truncate(last * self.dim);
        }
    }

    /// Top `k` fragments by their best part's cosine similarity, optionally
    /// limited to `scopes`. Highest first; ties broken by id for stable
    /// output.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        scopes: Option<&HashSet<String>>,
    ) -> Vec<(String, f32)> {
        if query.len() != self.dim || k == 0 {
            return Vec::new();
        }
        let mut best: HashMap<&str, f32> = HashMap::new();
        for (i, row) in self.data.chunks_exact(self.dim).enumerate() {
            if scopes.is_some_and(|s| !s.contains(&self.scopes[i])) {
                continue;
            }
            let score = wevex_embed::dot(row, query);
            best.entry(self.ids[i].as_str())
                .and_modify(|b| *b = b.max(score))
                .or_insert(score);
        }
        let mut scored: Vec<(&str, f32)> = best.into_iter().collect();
        let by_score =
            |a: &(&str, f32), b: &(&str, f32)| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0));
        if scored.len() > k {
            scored.select_nth_unstable_by(k - 1, by_score);
            scored.truncate(k);
        }
        scored.sort_by(by_score);
        scored
            .into_iter()
            .map(|(id, s)| (id.to_owned(), s))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_round_trip() {
        let v = vec![0.25f32, -1.5, 3.0e-7];
        assert_eq!(decode(&encode(&v)), v);
    }

    #[test]
    fn index_upsert_remove_and_search() {
        let mut ix = VectorIndex::empty("m", 2);
        ix.upsert("a", "s1", &[vec![1.0, 0.0]]);
        ix.upsert("b", "s1", &[vec![0.0, 1.0]]);
        ix.upsert("c", "s2", &[vec![0.8, 0.6]]);
        assert_eq!(ix.search(&[1.0, 0.0], 2, None)[0].0, "a");
        assert_eq!(
            ix.search(&[1.0, 0.0], 3, None)
                .iter()
                .map(|(id, _)| id.as_str())
                .collect::<Vec<_>>(),
            ["a", "c", "b"]
        );

        let s2: HashSet<String> = ["s2".to_string()].into();
        assert_eq!(ix.search(&[1.0, 0.0], 3, Some(&s2)).len(), 1);

        ix.upsert("a", "s1", &[vec![0.0, 1.0]]); // edited: now points the other way
        ix.remove("b");
        assert_eq!(ix.len(), 2);
        assert_eq!(ix.search(&[0.0, 1.0], 1, None)[0].0, "a");
        ix.remove("missing");
        ix.remove("a");
        ix.remove("c");
        assert!(ix.is_empty());
        assert!(ix.data.is_empty() && ix.ids.is_empty());
    }

    #[test]
    fn fragment_scores_as_its_best_part_and_parts_go_together() {
        let mut ix = VectorIndex::empty("m", 2);
        ix.upsert(
            "long",
            "s",
            &[vec![0.0, 1.0], vec![1.0, 0.0], vec![0.6, 0.8]],
        );
        ix.upsert("short", "s", &[vec![0.8, 0.6]]);
        assert_eq!(ix.len(), 2);
        let top = ix.search(&[1.0, 0.0], 5, None);
        assert_eq!(top.len(), 2, "one hit per fragment, not per part");
        assert_eq!(top[0], ("long".to_string(), 1.0));

        // Removing a multi-part fragment from the middle keeps the rest intact.
        ix.upsert("tail", "s", &[vec![1.0, 0.0]]);
        ix.remove("long");
        assert_eq!(ix.ids.len(), 2);
        let top = ix.search(&[1.0, 0.0], 5, None);
        assert_eq!(top[0].0, "tail");
        assert_eq!(top[1].0, "short");

        // Re-upserting with fewer parts drops the old ones.
        ix.upsert("tail", "s", &[vec![0.0, 1.0]]);
        assert_eq!(ix.ids.len(), 2);
        assert_eq!(ix.search(&[0.0, 1.0], 1, None)[0].0, "tail");
    }

    #[test]
    fn split_covers_text_with_overlap() {
        let text = (0..20)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let starts = wevex_embed::word_starts(&text);

        assert_eq!(
            split(&text, &starts, 20),
            [(0, text.len())],
            "fits: one part"
        );
        assert_eq!(split(&text, &starts, 50), [(0, text.len())]);

        let parts = split(&text, &starts, 8);
        assert_eq!(parts.first().unwrap().0, 0);
        assert_eq!(parts.last().unwrap().1, text.len());
        for w in parts.windows(2) {
            assert!(w[1].0 < w[0].1, "consecutive parts overlap");
        }
        for &(a, b) in &parts {
            assert!(wevex_embed::word_starts(&text[a..b]).len() <= 8);
        }
    }
}
