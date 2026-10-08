use std::collections::HashSet;

use rusqlite::params;
use wevex_core::embedding::{VectorIndex, embed_pending};
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
        .read(|c| VectorIndex::load(c, HashEmbedder::ID, 64))
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
