use std::collections::HashSet;

use rusqlite::params;
use wevex_core::embedding::{VectorIndex, embed_pending, model_key};
use wevex_core::storage::Store;
use wevex_embed::{Embedder, HashEmbedder};

fn store_with(fragments: &[(&str, &str, &str, bool)]) -> (tempfile::TempDir, Store) {
    let dir = tempfile::tempdir().unwrap();
    let store = Store::open(&dir.path().join("wevex.db")).unwrap();
    let rows: Vec<(String, String, String, bool)> = fragments
        .iter()
        .map(|(id, scope, content, stale)| {
            (
                id.to_string(),
                scope.to_string(),
                content.to_string(),
                *stale,
            )
        })
        .collect();
    store
        .write(move |c| {
            c.execute_batch(
                "INSERT INTO identities (id, handle, type, name) VALUES ('u', 'me', 'user', 'Me');
                 INSERT INTO scopes (id, handle, type, name, owner_id) VALUES
                   ('s1', 'one', 'project', 'One', 'u'), ('s2', 'two', 'project', 'Two', 'u');",
            )?;
            for (id, scope, content, stale) in &rows {
                c.execute(
                    "INSERT INTO fragments (id, type, content, scope_id, owner_id, is_stale)
                     VALUES (?1, 'fact', ?2, ?3, 'u', ?4)",
                    params![id, content, scope, stale],
                )?;
            }
            Ok(())
        })
        .unwrap();
    (dir, store)
}

#[test]
fn embeds_live_fragments_once_and_redoes_edits_and_model_changes() {
    let (_dir, store) = store_with(&[
        ("a", "s1", "sqlite storage layer for the daemon", false),
        ("b", "s1", "college essay about my roommate", false),
        ("c", "s2", "rust borrow checker notes", false),
        ("old", "s1", "stale scanner output", true),
    ]);
    let e = HashEmbedder::new(64);

    assert_eq!(embed_pending(&store, &e).unwrap(), 3, "stale skipped");
    assert_eq!(embed_pending(&store, &e).unwrap(), 0, "nothing left");

    store
        .write(|c| {
            c.execute(
                "UPDATE fragments SET content = 'edited text' WHERE id = 'b'",
                [],
            )?;
            Ok(())
        })
        .unwrap();
    assert_eq!(embed_pending(&store, &e).unwrap(), 1, "edit re-embedded");

    // A different model id re-embeds everything live.
    struct Other(HashEmbedder);
    impl Embedder for Other {
        fn id(&self) -> &str {
            "other-model"
        }
        fn dim(&self) -> usize {
            self.0.dim()
        }
        fn embed(&self, t: &[&str]) -> wevex_embed::Result<Vec<Vec<f32>>> {
            self.0.embed(t)
        }
    }
    assert_eq!(
        embed_pending(&store, &Other(HashEmbedder::new(64))).unwrap(),
        3
    );
}

#[test]
fn index_loads_live_vectors_and_searches_by_scope() {
    let (_dir, store) = store_with(&[
        ("a", "s1", "sqlite storage layer for the daemon", false),
        ("b", "s1", "college essay about my roommate", false),
        ("c", "s2", "sqlite storage notes in another project", false),
    ]);
    let e = HashEmbedder::new(64);
    embed_pending(&store, &e).unwrap();

    let ix = store
        .read(|c| VectorIndex::load(c, &model_key(&e), 64))
        .unwrap();
    assert_eq!(ix.len(), 3);

    let q = e.embed(&["sqlite storage"]).unwrap().remove(0);
    let top = ix.search(&q, 3, None);
    assert_ne!(top[0].0, "b");
    assert_eq!(top.last().unwrap().0, "b");

    let only_s1: HashSet<String> = ["s1".to_string()].into();
    let hits: Vec<String> = ix
        .search(&q, 3, Some(&only_s1))
        .into_iter()
        .map(|(id, _)| id)
        .collect();
    assert_eq!(hits, ["a", "b"]);

    // Vectors from another model are not loaded.
    let other = store.read(|c| VectorIndex::load(c, "other", 64)).unwrap();
    assert!(other.is_empty());
}

#[test]
fn long_fragments_are_embedded_as_parts_and_found_by_any_part() {
    // 60 filler words, then the one thing that matters at the very end.
    let filler = vec!["filler"; 60].join(" ");
    let long = format!("{filler} kubernetes ingress controller");
    let (_dir, store) = store_with(&[
        ("long", "s1", long.as_str(), false),
        ("short", "s1", "college essay about my roommate", false),
    ]);
    // A model that only reads 20 tokens, like bge-small reading 510.
    let e = HashEmbedder::new(64).with_max_tokens(20);
    assert_eq!(embed_pending(&store, &e).unwrap(), 2);

    let parts: Vec<(String, i64)> = store
        .read(|c| {
            let mut st = c.prepare(
                "SELECT fragment_id, COUNT(*) FROM fragment_vectors GROUP BY 1 ORDER BY 1",
            )?;
            let rows = st.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            Ok(rows.collect::<rusqlite::Result<Vec<_>>>()?)
        })
        .unwrap();
    assert!(parts[0].1 > 1, "long fragment split: {parts:?}");
    assert_eq!(parts[1], ("short".to_string(), 1));

    // The tail would have been cut off by truncation; as its own part it is found.
    let ix = store
        .read(|c| VectorIndex::load(c, &model_key(&e), 64))
        .unwrap();
    assert_eq!(ix.len(), 2);
    let q = e
        .embed(&["kubernetes ingress controller"])
        .unwrap()
        .remove(0);
    assert_eq!(ix.search(&q, 1, None)[0].0, "long");

    // Shrinking the fragment to one part drops the old extra parts.
    store
        .write(|c| {
            c.execute(
                "UPDATE fragments SET content = 'short now' WHERE id = 'long'",
                [],
            )?;
            Ok(())
        })
        .unwrap();
    assert_eq!(embed_pending(&store, &e).unwrap(), 1);
    let n: i64 = store
        .read(|c| {
            Ok(c.query_row(
                "SELECT COUNT(*) FROM fragment_vectors WHERE fragment_id = 'long'",
                [],
                |r| r.get(0),
            )?)
        })
        .unwrap();
    assert_eq!(n, 1);
}
