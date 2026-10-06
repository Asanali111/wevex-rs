//! The Wevex database: one SQLite file, owned by the daemon.
//!
//! [`Store::open`] applies connection settings, snapshots the file if a
//! migration is pending, migrates, then starts the writer thread and the
//! reader pool. Only `wevexd` should open a `Store`; the CLI talks to the
//! daemon instead of the file.

pub mod backup;
pub mod fragments;
pub mod migrations;
mod pool;

use std::path::{Path, PathBuf};
use std::time::Duration;

use rusqlite::{Connection, OpenFlags};

pub use migrations::MigrationReport;
use pool::{Readers, Writer};

use crate::Result;

/// Reader connections in the pool. Recall is the hot path; four covers
/// several agents querying at once on any machine we target.
pub const DEFAULT_READERS: usize = 4;

/// Snapshots kept by [`backup::rotate`] after a migration.
const KEEP_SNAPSHOTS: usize = 5;

#[derive(Debug, Clone, Default)]
pub struct OpenReport {
    pub migration: Option<MigrationReport>,
    pub snapshot: Option<PathBuf>,
}

pub struct Store {
    path: PathBuf,
    writer: Writer,
    readers: Readers,
    report: OpenReport,
}

impl Store {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with(path, DEFAULT_READERS)
    }

    pub fn open_with(path: &Path, readers: usize) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut conn = Connection::open(path)?;
        configure(&conn)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;

        let mut report = OpenReport::default();
        if migrations::user_version(&conn)? < migrations::LATEST && migrations::has_tables(&conn)? {
            let dir = path.parent().unwrap_or(Path::new("."));
            let from = migrations::user_version(&conn)?;
            report.snapshot = Some(backup::snapshot(&conn, dir, &format!("v{from}"))?);
            backup::rotate(dir, KEEP_SNAPSHOTS)?;
        }
        let migration = migrations::migrate(&mut conn)?;
        if migration.changed() {
            report.migration = Some(migration);
        }

        let readers = (0..readers.max(1))
            .map(|_| open_reader(path))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self {
            path: path.to_path_buf(),
            writer: Writer::spawn(conn)?,
            readers: Readers::new(readers),
            report,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// What happened while opening: migration applied, snapshot taken.
    pub fn open_report(&self) -> &OpenReport {
        &self.report
    }

    /// Run a read on a pooled `query_only` connection.
    pub fn read<T>(&self, f: impl FnOnce(&Connection) -> Result<T>) -> Result<T> {
        self.readers.run(f)
    }

    /// Run a write on the single writer thread, in submission order.
    pub fn write<T, F>(&self, f: F) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
    {
        self.writer.run(f)
    }
}

/// Per-connection settings, matching Python v0.2.2 where it matters.
fn configure(conn: &Connection) -> Result<()> {
    conn.busy_timeout(Duration::from_secs(30))?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    conn.pragma_update(None, "cache_size", -16_000)?; // ~16 MB
    Ok(())
}

fn open_reader(path: &Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY
            | OpenFlags::SQLITE_OPEN_NO_MUTEX
            | OpenFlags::SQLITE_OPEN_URI,
    )?;
    configure(&conn)?;
    conn.pragma_update(None, "query_only", "ON")?;
    Ok(conn)
}
