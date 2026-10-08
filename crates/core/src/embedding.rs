//! Keeping fragment vectors current, and searching them.
//!
//! [`embed_pending`] finds live fragments whose vector is missing, from a
//! different model, or for older content, and embeds them in batches. The
//! model runs outside the writer thread so writes never wait on it; only
//! storing the finished batch is a write. Stale fragments are skipped
//! (recall never returns them) and picked up if they become live again.
//!
//! [`VectorIndex`] holds the vectors in memory as one contiguous matrix and
//! ranks by dot product (cosine, since vectors are unit length). For the
//! thousands of fragments a person accumulates this takes well under a
//! millisecond; an ANN index (sqlite-vec) is only worth it far beyond that.

use std::collections::{HashMap, HashSet};

use rusqlite::{Connection, params};
use sha2::{Digest, Sha256};
use wevex_embed::Embedder;

use crate::Result;
use crate::storage::Store;
use crate::storage::fragments::LIVE;

/// Fragments embedded per model call and per write transaction.
pub const BATCH: usize = 32;

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

pub struct Pending {
    pub id: String,
    pub content: String,
    pub hash: String,
}

/// Up to `limit` live fragments whose vector is missing or out of date for
/// `model`. Most recently created first, so new work is searchable soonest.
pub fn pending(conn: &Connection, model: &str, limit: usize) -> Result<Vec<Pending>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT f.id, f.content, v.model, v.content_hash
           FROM fragments f
           LEFT JOIN fragment_vectors v ON v.fragment_id = f.id
          WHERE {LIVE}
          ORDER BY julianday(f.created_at) DESC"
    ))?;
    let mut rows = stmt.query([])?;
    let mut out = Vec::new();
    while let Some(r) = rows.next()? {
        let content: String = r.get(1)?;
        let hash = content_hash(&content);
        let current = r.get::<_, Option<String>>(2)?.as_deref() == Some(model)
            && r.get::<_, Option<String>>(3)?.as_deref() == Some(hash.as_str());
        if !current {
            out.push(Pending {
                id: r.get(0)?,
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

/// Embed every pending fragment. Returns how many were embedded. Safe to
/// interrupt and rerun: each batch is committed on its own.
pub fn embed_pending(store: &Store, embedder: &dyn Embedder) -> Result<usize> {
    let model = embedder.id().to_owned();
    let mut done = 0;
    loop {
        let batch = {
            let model = model.clone();
            store.read(move |c| pending(c, &model, BATCH))?
        };
        if batch.is_empty() {
            return Ok(done);
        }
        let texts: Vec<&str> = batch.iter().map(|p| p.content.as_str()).collect();
        let vectors = embedder.embed(&texts)?;
        let rows: Vec<(String, String, Vec<u8>)> = batch
            .into_iter()
            .zip(vectors)
            .map(|(p, v)| (p.id, p.hash, encode(&v)))
            .collect();
        let n = rows.len();
        let model = model.clone();
        store.write(move |c| {
            let tx = c.transaction()?;
            {
                let mut upsert = tx.prepare(
                    "INSERT INTO fragment_vectors (fragment_id, model, content_hash, vector)
                     VALUES (?1, ?2, ?3, ?4)
                     ON CONFLICT(fragment_id) DO UPDATE SET
                       model = excluded.model, content_hash = excluded.content_hash,
                       vector = excluded.vector, created_at = datetime('now')",
                )?;
                for (id, hash, blob) in &rows {
                    upsert.execute(params![id, model, hash, blob])?;
                }
            }
            tx.commit()?;
            Ok(())
        })?;
        done += n;
    }
}

/// In-memory vectors of live fragments for one model.
pub struct VectorIndex {
    model: String,
    dim: usize,
    ids: Vec<String>,
    scopes: Vec<String>,
    data: Vec<f32>,
    pos: HashMap<String, usize>,
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

    /// Load the current vectors of live fragments for `model`.
    pub fn load(conn: &Connection, model: &str, dim: usize) -> Result<Self> {
        let mut index = Self::empty(model, dim);
        let mut stmt = conn.prepare(&format!(
            "SELECT f.id, f.scope_id, v.vector
               FROM fragments f JOIN fragment_vectors v ON v.fragment_id = f.id
              WHERE v.model = ?1 AND {LIVE}"
        ))?;
        let mut rows = stmt.query([model])?;
        while let Some(r) = rows.next()? {
            let v = decode(&r.get::<_, Vec<u8>>(2)?);
            if v.len() == dim {
                index.upsert(&r.get::<_, String>(0)?, &r.get::<_, String>(1)?, &v);
            }
        }
        Ok(index)
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn upsert(&mut self, id: &str, scope_id: &str, v: &[f32]) {
        assert_eq!(v.len(), self.dim, "vector dimension mismatch");
        match self.pos.get(id) {
            Some(&i) => {
                self.data[i * self.dim..(i + 1) * self.dim].copy_from_slice(v);
                self.scopes[i] = scope_id.to_owned();
            }
            None => {
                self.pos.insert(id.to_owned(), self.ids.len());
                self.ids.push(id.to_owned());
                self.scopes.push(scope_id.to_owned());
                self.data.extend_from_slice(v);
            }
        }
    }

    /// Remove by swapping the last row into the gap: O(1), order not kept.
    pub fn remove(&mut self, id: &str) {
        let Some(i) = self.pos.remove(id) else {
            return;
        };
        let last = self.ids.len() - 1;
        if i != last {
            self.ids.swap(i, last);
            self.scopes.swap(i, last);
            let (head, tail) = self.data.split_at_mut(last * self.dim);
            head[i * self.dim..(i + 1) * self.dim].copy_from_slice(&tail[..self.dim]);
            self.pos.insert(self.ids[i].clone(), i);
        }
        self.ids.pop();
        self.scopes.pop();
        self.data.truncate(last * self.dim);
    }

    /// Top `k` fragments by cosine similarity, optionally limited to
    /// `scopes`. Highest first; ties broken by id for stable output.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        scopes: Option<&HashSet<String>>,
    ) -> Vec<(String, f32)> {
        if query.len() != self.dim || k == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f32)> = self
            .data
            .chunks_exact(self.dim)
            .enumerate()
            .filter(|(i, _)| scopes.is_none_or(|s| s.contains(&self.scopes[*i])))
            .map(|(i, row)| (i, wevex_embed::dot(row, query)))
            .collect();
        let by_score = |a: &(usize, f32), b: &(usize, f32)| {
            b.1.total_cmp(&a.1)
                .then_with(|| self.ids[a.0].cmp(&self.ids[b.0]))
        };
        if scored.len() > k {
            scored.select_nth_unstable_by(k - 1, by_score);
            scored.truncate(k);
        }
        scored.sort_by(by_score);
        scored
            .into_iter()
            .map(|(i, s)| (self.ids[i].clone(), s))
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
        ix.upsert("a", "s1", &[1.0, 0.0]);
        ix.upsert("b", "s1", &[0.0, 1.0]);
        ix.upsert("c", "s2", &[0.8, 0.6]);
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

        ix.upsert("a", "s1", &[0.0, 1.0]); // edited: now points the other way
        ix.remove("b");
        assert_eq!(ix.len(), 2);
        assert_eq!(ix.search(&[0.0, 1.0], 1, None)[0].0, "a");
        ix.remove("missing");
        ix.remove("a");
        ix.remove("c");
        assert!(ix.is_empty());
    }
}
