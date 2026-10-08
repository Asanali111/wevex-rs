//! Versioned schema migrations, tracked in `PRAGMA user_version`.
//!
//! Python v0.2.2 never set `user_version`, so its databases report 0. That
//! is the same value as an empty file, and both are handled the same way:
//! 0001 is all `IF NOT EXISTS`, then [`repair_legacy_columns`] adds any
//! columns an older v0.2.x added via `ALTER TABLE`.

use rusqlite::Connection;

use crate::{Error, Result, value};

pub struct Migration {
    pub version: i64,
    pub name: &'static str,
    pub sql: &'static str,
}

pub const MIGRATIONS: &[Migration] = &[Migration {
    version: 1,
    name: "baseline",
    sql: include_str!("../../migrations/0001_baseline.sql"),
}];

pub const LATEST: i64 = MIGRATIONS[MIGRATIONS.len() - 1].version;

/// Columns that v0.2.x added to existing tables after their first release.
/// `CREATE TABLE IF NOT EXISTS` skips tables that already exist, so a
/// database from an early v0.2 can lack these.
const LEGACY_COLUMNS: &[(&str, &str, &str)] = &[
    ("fragments", "created_by_tool", "TEXT"),
    ("fragments", "created_in_session_id", "TEXT"),
    ("fragments", "created_against_commit", "TEXT"),
    (
        "fragments",
        "files_open_at_creation",
        "TEXT NOT NULL DEFAULT '[]'",
    ),
    (
        "fragments",
        "supersedes_fragment_id",
        "TEXT REFERENCES fragments(id)",
    ),
    (
        "fragments",
        "superseded_by_fragment_id",
        "TEXT REFERENCES fragments(id)",
    ),
    (
        "fragments",
        "extraction_method",
        "TEXT NOT NULL DEFAULT 'explicit'",
    ),
    ("fragments", "extraction_confidence", "REAL"),
    ("fragments", "value", "REAL NOT NULL DEFAULT 0.5"),
    ("fragments", "dedupe_key", "TEXT"),
    ("fragments", "recall_hits", "INTEGER NOT NULL DEFAULT 0"),
    ("fragments", "last_recalled_at", "TEXT"),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MigrationReport {
    pub from: i64,
    pub to: i64,
    pub repaired_columns: Vec<String>,
}

impl MigrationReport {
    pub fn changed(&self) -> bool {
        self.from != self.to || !self.repaired_columns.is_empty()
    }
}

pub fn user_version(conn: &Connection) -> Result<i64> {
    Ok(conn.query_row("PRAGMA user_version", [], |r| r.get(0))?)
}

/// Bring the schema up to [`LATEST`] in a single transaction.
pub fn migrate(conn: &mut Connection) -> Result<MigrationReport> {
    let from = user_version(conn)?;
    if from > LATEST {
        return Err(Error::SchemaTooNew {
            found: from,
            supported: LATEST,
        });
    }
    let mut report = MigrationReport {
        from,
        to: from,
        repaired_columns: Vec::new(),
    };
    if from == LATEST {
        return Ok(report);
    }

    let tx = conn.transaction()?;
    for m in MIGRATIONS.iter().filter(|m| m.version > from) {
        if m.version == 1 {
            // Repair before the baseline: its indexes reference these columns.
            report.repaired_columns = repair_legacy_columns(&tx)?;
        }
        tx.execute_batch(m.sql)?;
        tx.pragma_update(None, "user_version", m.version)?;
        report.to = m.version;
    }
    tx.commit()?;
    Ok(report)
}

/// True when the database already has tables, i.e. there is data worth
/// backing up before migrating.
pub fn has_tables(conn: &Connection) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'table')",
        [],
        |r| r.get(0),
    )?)
}

fn repair_legacy_columns(conn: &Connection) -> Result<Vec<String>> {
    let mut added = Vec::new();
    for (table, column, ddl) in LEGACY_COLUMNS {
        if !table_exists(conn, table)? || column_exists(conn, table, column)? {
            continue;
        }
        conn.execute_batch(&format!("ALTER TABLE {table} ADD COLUMN {column} {ddl}"))?;
        added.push(format!("{table}.{column}"));
    }
    if added.iter().any(|c| c == "fragments.value") {
        backfill_values(conn)?;
    }
    Ok(added)
}

/// Give pre-existing fragments a real initial value instead of the column
/// default, as v0.2.2 did when it added the column. The `updated_at`
/// trigger is dropped first so the backfill does not stamp every row as
/// just-updated; the baseline that runs next recreates it.
fn backfill_values(conn: &Connection) -> Result<()> {
    conn.execute_batch("DROP TRIGGER IF EXISTS fragments_updated_at")?;
    let rows: Vec<(String, String, String, String, Option<String>, String)> = {
        let mut stmt = conn.prepare(
            "SELECT id, type, content, extraction_method, created_by_tool, metadata FROM fragments",
        )?;
        stmt.query_map([], |r| {
            Ok((
                r.get(0)?,
                r.get(1)?,
                r.get(2)?,
                r.get(3)?,
                r.get(4)?,
                r.get(5)?,
            ))
        })?
        .collect::<rusqlite::Result<_>>()?
    };
    let mut update = conn.prepare("UPDATE fragments SET value = ?1 WHERE id = ?2")?;
    for (id, kind, content, method, tool, metadata) in rows {
        let metadata = serde_json::from_str(&metadata).unwrap_or(serde_json::Value::Null);
        let v = value::compute(&value::FragmentFacts {
            kind: &kind,
            content: &content,
            extraction_method: &method,
            created_by_tool: tool.as_deref(),
            metadata: &metadata,
        });
        update.execute(rusqlite::params![v, id])?;
    }
    Ok(())
}

fn table_exists(conn: &Connection, table: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS (SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
        [table],
        |r| r.get(0),
    )?)
}

fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS (SELECT 1 FROM pragma_table_info(?1) WHERE name = ?2)",
        [table, column],
        |r| r.get(0),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn columns(conn: &Connection, table: &str) -> Vec<String> {
        let mut stmt = conn
            .prepare("SELECT name FROM pragma_table_info(?1)")
            .unwrap();
        stmt.query_map([table], |r| r.get(0))
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
    }

    #[test]
    fn fresh_database_reaches_latest() {
        let mut conn = Connection::open_in_memory().unwrap();
        let report = migrate(&mut conn).unwrap();
        assert_eq!((report.from, report.to), (0, LATEST));
        assert!(report.repaired_columns.is_empty());
        assert_eq!(user_version(&conn).unwrap(), LATEST);
        assert!(columns(&conn, "fragments").contains(&"last_recalled_at".to_string()));
    }

    #[test]
    fn migrate_is_idempotent() {
        let mut conn = Connection::open_in_memory().unwrap();
        migrate(&mut conn).unwrap();
        let again = migrate(&mut conn).unwrap();
        assert!(!again.changed());
    }

    #[test]
    fn early_v02_database_gets_missing_columns() {
        let mut conn = Connection::open_in_memory().unwrap();
        // The original v0.2 fragments table, before the ALTERs.
        conn.execute_batch(
            "CREATE TABLE fragments (
               id TEXT PRIMARY KEY, type TEXT NOT NULL, content TEXT NOT NULL,
               scope_id TEXT NOT NULL, owner_id TEXT NOT NULL, confidence REAL,
               version INTEGER NOT NULL DEFAULT 1, ttl_seconds INTEGER,
               expires_at TEXT, permanent INTEGER NOT NULL DEFAULT 0,
               is_stale INTEGER NOT NULL DEFAULT 0, stale_reason TEXT,
               tags TEXT NOT NULL DEFAULT '[]', territory TEXT,
               source_commit_id TEXT, metadata TEXT NOT NULL DEFAULT '{}',
               content_embedding BLOB,
               created_at TEXT NOT NULL DEFAULT (datetime('now')),
               updated_at TEXT NOT NULL DEFAULT (datetime('now')));
             CREATE TRIGGER fragments_updated_at AFTER UPDATE ON fragments BEGIN
               UPDATE fragments SET updated_at = datetime('now') WHERE id = new.id;
             END;
             INSERT INTO fragments (id, type, content, scope_id, owner_id, updated_at)
               VALUES ('f1', 'fact', 'kept', 's', 'o', '2026-01-01 00:00:00');",
        )
        .unwrap();

        let report = migrate(&mut conn).unwrap();
        assert_eq!(report.repaired_columns.len(), LEGACY_COLUMNS.len());

        let (content, value, updated_at): (String, f64, String) = conn
            .query_row(
                "SELECT content, value, updated_at FROM fragments WHERE id = 'f1'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap();
        assert_eq!(content, "kept");
        // Computed like v0.2.2 (explicit, no tool, short content), not the 0.5 default.
        assert!((value - 0.95).abs() < 1e-9);
        // The backfill must not look like an edit.
        assert_eq!(updated_at, "2026-01-01 00:00:00");

        // ...and the trigger is back afterwards.
        conn.execute(
            "UPDATE fragments SET content = 'edited' WHERE id = 'f1'",
            [],
        )
        .unwrap();
        let updated_at: String = conn
            .query_row(
                "SELECT updated_at FROM fragments WHERE id = 'f1'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_ne!(updated_at, "2026-01-01 00:00:00");
    }

    #[test]
    fn newer_schema_is_refused() {
        let mut conn = Connection::open_in_memory().unwrap();
        conn.pragma_update(None, "user_version", LATEST + 1)
            .unwrap();
        assert!(matches!(
            migrate(&mut conn),
            Err(Error::SchemaTooNew { .. })
        ));
    }
}
