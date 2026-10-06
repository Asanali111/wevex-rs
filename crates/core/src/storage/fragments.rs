//! Typed access to the `fragments` table.

use std::collections::BTreeMap;

use rusqlite::{Connection, OptionalExtension, Row};
use serde::Serialize;

use crate::Result;

/// A fragment is "live" when recall should consider it. `julianday()` is
/// used because Python stored `expires_at` both as `YYYY-MM-DD HH:MM:SS`
/// and as ISO-8601 with a `T`, and plain string comparison mixes them up.
pub const LIVE: &str = "is_stale = 0
     AND superseded_by_fragment_id IS NULL
     AND (expires_at IS NULL OR julianday(expires_at) > julianday('now'))";

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Fragment {
    pub id: String,
    pub kind: String,
    pub content: String,
    pub scope_id: String,
    pub owner_id: String,
    pub confidence: Option<f64>,
    pub tags: Vec<String>,
    pub territory: Option<String>,
    pub is_stale: bool,
    pub expires_at: Option<String>,
    pub supersedes: Option<String>,
    pub superseded_by: Option<String>,
    pub extraction_method: String,
    pub value: f64,
    pub recall_hits: i64,
    pub has_embedding: bool,
    pub created_at: String,
    pub updated_at: String,
}

const COLUMNS: &str = "id, type, content, scope_id, owner_id, confidence, tags,
     territory, is_stale, expires_at, supersedes_fragment_id,
     superseded_by_fragment_id, extraction_method, value, recall_hits,
     content_embedding IS NOT NULL, created_at, updated_at";

impl Fragment {
    fn from_row(r: &Row<'_>) -> rusqlite::Result<Self> {
        let tags: String = r.get(6)?;
        Ok(Self {
            id: r.get(0)?,
            kind: r.get(1)?,
            content: r.get(2)?,
            scope_id: r.get(3)?,
            owner_id: r.get(4)?,
            confidence: r.get(5)?,
            // Tolerate malformed legacy JSON rather than failing the read.
            tags: serde_json::from_str(&tags).unwrap_or_default(),
            territory: r.get(7)?,
            is_stale: r.get(8)?,
            expires_at: r.get(9)?,
            supersedes: r.get(10)?,
            superseded_by: r.get(11)?,
            extraction_method: r.get(12)?,
            value: r.get(13)?,
            recall_hits: r.get(14)?,
            has_embedding: r.get(15)?,
            created_at: r.get(16)?,
            updated_at: r.get(17)?,
        })
    }
}

pub fn get(conn: &Connection, id: &str) -> Result<Option<Fragment>> {
    Ok(conn
        .query_row(
            &format!("SELECT {COLUMNS} FROM fragments WHERE id = ?1"),
            [id],
            Fragment::from_row,
        )
        .optional()?)
}

/// Most recently updated live fragments.
pub fn recent(conn: &Connection, limit: usize) -> Result<Vec<Fragment>> {
    let mut stmt = conn.prepare(&format!(
        "SELECT {COLUMNS} FROM fragments WHERE {LIVE}
         ORDER BY updated_at DESC LIMIT ?1"
    ))?;
    let rows = stmt
        .query_map([limit as i64], Fragment::from_row)?
        .collect::<rusqlite::Result<_>>()?;
    Ok(rows)
}

#[derive(Debug, Default, Clone, PartialEq, Serialize)]
pub struct Stats {
    pub total: i64,
    pub live: i64,
    pub stale: i64,
    pub expired: i64,
    pub superseded: i64,
    pub with_embedding: i64,
    pub live_by_type: BTreeMap<String, i64>,
}

pub fn stats(conn: &Connection) -> Result<Stats> {
    let mut s = conn.query_row(
        &format!(
            "SELECT
               COUNT(*),
               COALESCE(SUM({LIVE}), 0),
               COALESCE(SUM(is_stale = 1), 0),
               COALESCE(SUM(is_stale = 0 AND expires_at IS NOT NULL
                            AND julianday(expires_at) <= julianday('now')), 0),
               COALESCE(SUM(superseded_by_fragment_id IS NOT NULL), 0),
               COALESCE(SUM(content_embedding IS NOT NULL), 0)
             FROM fragments"
        ),
        [],
        |r| {
            Ok(Stats {
                total: r.get(0)?,
                live: r.get(1)?,
                stale: r.get(2)?,
                expired: r.get(3)?,
                superseded: r.get(4)?,
                with_embedding: r.get(5)?,
                live_by_type: BTreeMap::new(),
            })
        },
    )?;
    let mut stmt = conn.prepare(&format!(
        "SELECT type, COUNT(*) FROM fragments WHERE {LIVE} GROUP BY type"
    ))?;
    for row in stmt.query_map([], |r| Ok((r.get::<_, String>(0)?, r.get(1)?)))? {
        let (k, n) = row?;
        s.live_by_type.insert(k, n);
    }
    Ok(s)
}
