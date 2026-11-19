//! Hybrid recall: keyword (BM25 over FTS5) and meaning (vectors) ranked
//! separately, then merged with reciprocal-rank fusion.
//!
//! RRF scores each fragment by `Σ 1 / (k + rank)` over the lists it appears
//! in, so a fragment both legs agree on beats one either leg ranks first.
//! It needs no score calibration between BM25 and cosine, which live on
//! unrelated scales. The chunk-size benchmark used the same setup (top 50
//! per leg, k = 60) and found it beat either leg alone by 4–5 points.

use std::collections::{HashMap, HashSet};

use rusqlite::{Connection, params_from_iter, types::Value};
use wevex_embed::Embedder;

use crate::Result;
use crate::embedding::VectorIndex;
use crate::storage::fragments::LIVE;

/// Candidates taken from each leg before fusion.
pub const PER_LEG: usize = 50;
/// RRF damping constant; 60 is the value from the original RRF paper.
pub const RRF_K: f64 = 60.0;

/// Turn free text into an FTS5 query: each word quoted (so `AND`, `NEAR`,
/// `*` or `-` are plain words), joined with OR. FTS5's default is AND, which
/// finds nothing for most multi-word questions. `None` when there is no word
/// to search for.
pub fn fts_query(text: &str) -> Option<String> {
    let mut seen = HashSet::new();
    let terms: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .filter(|w| seen.insert(w.clone()))
        .map(|w| format!("\"{w}\""))
        .collect();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}

/// Live fragment ids matching `text`, best BM25 first. Scope filtering runs
/// inside the query so a narrow scope still gets `limit` results.
pub fn keyword(
    conn: &Connection,
    text: &str,
    scopes: Option<&HashSet<String>>,
    limit: usize,
) -> Result<Vec<String>> {
    let Some(q) = fts_query(text) else {
        return Ok(Vec::new());
    };
    let mut args: Vec<Value> = vec![Value::Text(q)];
    let scope_sql = match scopes {
        Some(s) if s.is_empty() => return Ok(Vec::new()),
        Some(s) => {
            let mut ids: Vec<&String> = s.iter().collect();
            ids.sort();
            args.extend(ids.into_iter().map(|id| Value::Text(id.clone())));
            format!(
                "AND f.scope_id IN ({})",
                vec!["?"; args.len() - 1].join(", ")
            )
        }
        None => String::new(),
    };
    args.push(Value::Integer(limit as i64));
    let mut stmt = conn.prepare(&format!(
        "SELECT f.id
           FROM fragments_fts
           JOIN fragments f ON f.id = fragments_fts.fragment_id
          WHERE fragments_fts MATCH ?1 AND {LIVE} {scope_sql}
          ORDER BY bm25(fragments_fts), f.id
          LIMIT ?{}",
        args.len()
    ))?;
    let ids = stmt
        .query_map(params_from_iter(args), |r| r.get(0))?
        .collect::<rusqlite::Result<Vec<String>>>()?;
    Ok(ids)
}

/// Reciprocal-rank fusion of ranked id lists. Highest first; ties broken by
/// id for stable output.
pub fn rrf(lists: &[Vec<String>], k: f64) -> Vec<(String, f64)> {
    let mut score: HashMap<&str, f64> = HashMap::new();
    for list in lists {
        for (rank, id) in list.iter().enumerate() {
            *score.entry(id.as_str()).or_default() += 1.0 / (k + rank as f64 + 1.0);
        }
    }
    let mut out: Vec<(String, f64)> = score
        .into_iter()
        .map(|(id, s)| (id.to_owned(), s))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    out
}

/// One recall result.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: String,
    /// RRF score; only meaningful for ordering.
    pub score: f64,
    pub by_keyword: bool,
    pub by_meaning: bool,
}

/// Top `k` live fragments for `query`, optionally limited to scope ids.
/// `index` must hold vectors made by `embedder` (see
/// [`crate::embedding::model_key`]); fragments not embedded yet are still
/// found by keyword.
///
/// The meaning leg always has a nearest neighbour, even for a query nothing
/// relates to, so matches below `min_similarity` (cosine) are dropped
/// before fusion. The right floor depends on the model; set it from the
/// labelled eval.
pub fn recall(
    conn: &Connection,
    embedder: &dyn Embedder,
    index: &VectorIndex,
    query: &str,
    scopes: Option<&HashSet<String>>,
    k: usize,
    min_similarity: f32,
) -> Result<Vec<Hit>> {
    if k == 0 || query.trim().is_empty() {
        return Ok(Vec::new());
    }
    let by_keyword = keyword(conn, query, scopes, PER_LEG)?;
    let by_meaning: Vec<String> = if index.is_empty() {
        Vec::new()
    } else {
        let q = embedder.embed(&[query])?.remove(0);
        index
            .search(&q, PER_LEG, scopes)
            .into_iter()
            .filter(|(_, sim)| *sim >= min_similarity)
            .map(|(id, _)| id)
            .collect()
    };
    let kw: HashSet<&str> = by_keyword.iter().map(String::as_str).collect();
    let mn: HashSet<&str> = by_meaning.iter().map(String::as_str).collect();
    Ok(rrf(&[by_keyword.clone(), by_meaning.clone()], RRF_K)
        .into_iter()
        .take(k)
        .map(|(id, score)| Hit {
            by_keyword: kw.contains(id.as_str()),
            by_meaning: mn.contains(id.as_str()),
            id,
            score,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_query_quotes_words_and_drops_operators() {
        assert_eq!(fts_query("  "), None);
        assert_eq!(fts_query("*** -- ()"), None);
        assert_eq!(
            fts_query("SQLite AND \"wal\" NEAR(x) -drop"),
            Some(r#""sqlite" OR "and" OR "wal" OR "near" OR "x" OR "drop""#.to_string())
        );
        assert_eq!(
            fts_query("db DB Db"),
            Some(r#""db""#.to_string()),
            "duplicates collapse"
        );
        assert_eq!(
            fts_query("эссе о соседе"),
            Some(r#""эссе" OR "о" OR "соседе""#.to_string())
        );
    }

    fn ids(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn rrf_rewards_agreement_and_handles_one_sided_and_empty_lists() {
        let fused = rrf(&[ids(&["a", "b", "c"]), ids(&["b", "d"])], RRF_K);
        let order: Vec<&str> = fused.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(order[0], "b", "in both lists");
        assert_eq!(order.len(), 4);
        assert!(order.contains(&"d"), "one-sided ids kept");

        // Equal scores: ordered by id.
        let tie = rrf(&[ids(&["y"]), ids(&["x"])], RRF_K);
        assert_eq!(tie[0].0, "x");
        assert_eq!(tie[0].1, tie[1].1);

        assert!(rrf(&[vec![], vec![]], RRF_K).is_empty());
        assert_eq!(rrf(&[ids(&["a"]), vec![]], RRF_K)[0].0, "a");
    }
}
