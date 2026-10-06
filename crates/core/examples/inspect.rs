//! Open a *copy* of a Wevex database through `Store` and report what the
//! Rust core sees. The source file is only read, via SQLite's backup API, so
//! this is safe to run against the live database while a daemon is up.
//!
//!     cargo run -p wevex-core --example inspect -- [path/to/wevex.db]

use std::path::PathBuf;

use rusqlite::{Connection, OpenFlags};
use wevex_core::storage::{Store, backup, fragments, migrations};

fn main() -> wevex_core::Result<()> {
    let src = match std::env::args_os().nth(1) {
        Some(p) => PathBuf::from(p),
        None => wevex_core::paths::db_path()?,
    };
    let work = tempfile::tempdir()?;

    let source = Connection::open_with_flags(&src, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let copy = backup::snapshot(&source, work.path(), "inspect")?;
    let before = migrations::user_version(&source)?;
    drop(source);

    let db = work.path().join("wevex.db");
    std::fs::rename(&copy, &db)?;

    let started = std::time::Instant::now();
    let store = Store::open(&db)?;
    let opened = started.elapsed();

    let stats = store.read(fragments::stats)?;
    let issues = store.read(backup::integrity_check)?;
    let recent = store.read(|c| fragments::recent(c, 3))?;

    println!("source          {}", src.display());
    println!("user_version    {before} -> {}", migrations::LATEST);
    println!("migration       {:?}", store.open_report().migration);
    println!("open+migrate    {opened:?}");
    println!(
        "integrity       {}",
        if issues.is_empty() {
            "ok".into()
        } else {
            issues.join("; ")
        }
    );
    println!("{}", serde_json::to_string_pretty(&stats)?);
    for f in recent {
        let preview: String = f.content.chars().take(60).collect();
        println!("recent          [{}] {preview}", f.kind);
    }
    Ok(())
}
