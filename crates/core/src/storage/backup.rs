//! Online snapshots via SQLite's backup API (safe while the daemon writes,
//! unlike copying the file) and integrity checks.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use rusqlite::Connection;
use rusqlite::backup::Backup;

use crate::Result;

/// Prefix for snapshots this build creates. Rotation only ever touches files
/// with this prefix, never the `wevex.db.bak-*` files Python left behind.
pub const SNAPSHOT_PREFIX: &str = "wevex.db.snap-";

/// Copy `conn`'s main database to `<dir>/wevex.db.snap-<label>-<unix secs>`.
pub fn snapshot(conn: &Connection, dir: &Path, label: &str) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let path = dir.join(format!("{SNAPSHOT_PREFIX}{label}-{secs}"));
    let mut dst = Connection::open(&path)?;
    Backup::new(conn, &mut dst)?.run_to_completion(256, Duration::ZERO, None)?;
    Ok(path)
}

/// Keep the `keep` newest snapshots in `dir`, delete the rest.
pub fn rotate(dir: &Path, keep: usize) -> Result<Vec<PathBuf>> {
    let mut snaps: Vec<(SystemTime, PathBuf)> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().starts_with(SNAPSHOT_PREFIX))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    snaps.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
    let mut removed = Vec::new();
    for (_, path) in snaps.into_iter().skip(keep) {
        std::fs::remove_file(&path)?;
        removed.push(path);
    }
    Ok(removed)
}

/// `PRAGMA integrity_check`. Empty means healthy.
pub fn integrity_check(conn: &Connection) -> Result<Vec<String>> {
    let mut stmt = conn.prepare("PRAGMA integrity_check")?;
    let rows: Vec<String> = stmt
        .query_map([], |r| r.get(0))?
        .collect::<rusqlite::Result<_>>()?;
    Ok(if rows == ["ok"] { Vec::new() } else { rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_copies_data_and_rotation_keeps_newest() {
        let dir = tempfile::tempdir().unwrap();
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE t (x); INSERT INTO t VALUES (42);")
            .unwrap();

        let snap = snapshot(&conn, dir.path(), "test").unwrap();
        let copy = Connection::open(&snap).unwrap();
        let x: i64 = copy.query_row("SELECT x FROM t", [], |r| r.get(0)).unwrap();
        assert_eq!(x, 42);
        assert!(integrity_check(&copy).unwrap().is_empty());

        std::fs::write(dir.path().join("wevex.db.bak-python"), b"untouched").unwrap();
        for i in 0..3 {
            snapshot(&conn, dir.path(), &format!("n{i}")).unwrap();
        }
        rotate(dir.path(), 2).unwrap();
        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names
                .iter()
                .filter(|n| n.starts_with(SNAPSHOT_PREFIX))
                .count(),
            2
        );
        assert!(names.contains(&"wevex.db.bak-python".to_string()));
    }
}
