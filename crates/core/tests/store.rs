use std::sync::Arc;

use rusqlite::{Connection, params};
use wevex_core::storage::{Store, backup, fragments, migrations};

fn seed(conn: &Connection) {
    conn.execute_batch(
        "INSERT INTO identities (id, handle, type, name) VALUES ('u', 'me', 'user', 'Me');
         INSERT INTO scopes (id, handle, type, name, owner_id) VALUES ('s', 'proj', 'project', 'Proj', 'u');",
    )
    .unwrap();
}

fn insert(conn: &Connection, id: &str, stale: bool, expires_at: Option<&str>) {
    conn.execute(
        "INSERT INTO fragments (id, type, content, scope_id, owner_id, is_stale, expires_at, tags)
         VALUES (?1, 'fact', ?2, 's', 'u', ?3, ?4, '[\"a\",\"b\"]')",
        params![id, format!("content of {id}"), stale, expires_at],
    )
    .unwrap();
}

#[test]
fn stats_separate_live_stale_expired_across_timestamp_formats() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(&dir.path().join("wevex.db")).unwrap();

    store
        .write(|c| {
            seed(c);
            insert(c, "live", false, None);
            insert(
                c,
                "live-future-iso",
                false,
                Some("2999-01-01T00:00:00+00:00"),
            );
            insert(c, "stale", true, None);
            insert(c, "expired-sqlite", false, Some("2000-01-01 00:00:00"));
            insert(c, "expired-iso", false, Some("2000-01-01T00:00:00Z"));
            Ok(())
        })
        .unwrap();

    let s = store.read(fragments::stats).unwrap();
    assert_eq!((s.total, s.live, s.stale, s.expired), (5, 2, 1, 2));
    assert_eq!(s.live_by_type.get("fact"), Some(&2));

    let f = store
        .read(|c| fragments::get(c, "live"))
        .unwrap()
        .expect("fragment exists");
    assert_eq!(f.tags, ["a", "b"]);
    assert!(!f.has_embedding);

    let recent = store.read(|c| fragments::recent(c, 10)).unwrap();
    assert_eq!(recent.len(), 2);
}

#[test]
fn fts_trigger_indexes_new_fragments() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(&dir.path().join("wevex.db")).unwrap();
    store
        .write(|c| {
            seed(c);
            insert(c, "f", false, None);
            Ok(())
        })
        .unwrap();
    let hit: String = store
        .read(|c| {
            Ok(c.query_row(
                "SELECT fragment_id FROM fragments_fts WHERE fragments_fts MATCH 'content'",
                [],
                |r| r.get(0),
            )?)
        })
        .unwrap();
    assert_eq!(hit, "f");
}

#[test]
fn readers_cannot_write() {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(&dir.path().join("wevex.db")).unwrap();
    let err = store.read(|c| Ok(c.execute_batch("CREATE TABLE nope (x)")?));
    assert!(err.is_err());
}

#[test]
fn concurrent_reads_see_writes_from_writer_thread() {
    let dir = tempfile::tempdir().unwrap();
    let store = Arc::new(Store::open(&dir.path().join("wevex.db")).unwrap());
    store
        .write(|c| {
            seed(c);
            Ok(())
        })
        .unwrap();

    let writer = {
        let store = Arc::clone(&store);
        std::thread::spawn(move || {
            for i in 0..200 {
                store
                    .write(move |c| {
                        insert(c, &format!("f{i}"), false, None);
                        Ok(())
                    })
                    .unwrap();
            }
        })
    };
    let readers: Vec<_> = (0..8)
        .map(|_| {
            let store = Arc::clone(&store);
            std::thread::spawn(move || {
                let mut last = 0;
                for _ in 0..200 {
                    let n = store.read(fragments::stats).unwrap().total;
                    assert!(n >= last, "counts never go backwards");
                    last = n;
                }
            })
        })
        .collect();

    writer.join().unwrap();
    for r in readers {
        r.join().unwrap();
    }
    assert_eq!(store.read(fragments::stats).unwrap().total, 200);
}

#[test]
fn legacy_database_is_snapshotted_then_migrated_without_data_loss() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("wevex.db");

    // A v0.2.2 database: full schema, data, user_version still 0.
    {
        let mut conn = Connection::open(&path).unwrap();
        migrations::migrate(&mut conn).unwrap();
        seed(&conn);
        insert(&conn, "old", false, None);
        conn.pragma_update(None, "user_version", 0).unwrap();
    }

    let store = Store::open(&path).unwrap();
    let report = store.open_report();
    let snap = report
        .snapshot
        .as_ref()
        .expect("snapshot taken before migrating");
    assert_eq!(report.migration.as_ref().unwrap().to, migrations::LATEST);

    let snap_conn = Connection::open(snap).unwrap();
    assert!(backup::integrity_check(&snap_conn).unwrap().is_empty());
    assert_eq!(migrations::user_version(&snap_conn).unwrap(), 0);

    assert!(store.read(|c| fragments::get(c, "old")).unwrap().is_some());

    // Reopening an up-to-date database neither migrates nor snapshots.
    drop(store);
    let again = Store::open(&path).unwrap();
    assert!(again.open_report().snapshot.is_none());
    assert!(again.open_report().migration.is_none());
}
