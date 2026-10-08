//! Re-embed a *copy* of a Wevex database with the real model, then run a
//! few searches by meaning. The source file is only read (backup API).
//!
//!     cargo run -p wevexd --example reembed -- [db] [query]...

use std::path::PathBuf;
use std::time::Instant;

use rusqlite::{Connection, OpenFlags};
use wevex_core::embedding::{VectorIndex, embed_pending};
use wevex_core::storage::{Store, backup};
use wevex_embed::{BgeSmall, Embedder};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let src = args
        .next()
        .map(PathBuf::from)
        .unwrap_or(wevex_core::paths::db_path()?);
    let queries: Vec<String> = args.collect();

    let work = tempfile::tempdir()?;
    let source = Connection::open_with_flags(&src, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let copy = backup::snapshot(&source, work.path(), "reembed")?;
    drop(source);
    let db = work.path().join("wevex.db");
    std::fs::rename(&copy, &db)?;
    let store = Store::open(&db)?;

    let models = std::env::var_os("WEVEX_MODEL_CACHE")
        .map(PathBuf::from)
        .unwrap_or(wevex_core::paths::state_dir()?.join("models"));
    let t = Instant::now();
    let model = BgeSmall::load(&models)?;
    println!("model load      {:?}", t.elapsed());

    let t = Instant::now();
    let n = embed_pending(&store, &model)?;
    let took = t.elapsed();
    println!(
        "embedded        {n} fragments in {took:?} ({:.1} ms each)",
        took.as_secs_f64() * 1000.0 / n.max(1) as f64
    );

    let index = store.read(|c| VectorIndex::load(c, model.id(), model.dim()))?;
    for q in &queries {
        let t = Instant::now();
        let qv = model.embed(&[q])?.remove(0);
        let embed_ms = t.elapsed();
        let t = Instant::now();
        let hits = index.search(&qv, 3, None);
        let search = t.elapsed();
        println!("\n\"{q}\"  (embed {embed_ms:?}, search {search:?})");
        for (id, score) in hits {
            let content: String = store.read(move |c| {
                Ok(
                    c.query_row("SELECT content FROM fragments WHERE id = ?1", [&id], |r| {
                        r.get(0)
                    })?,
                )
            })?;
            let preview: String = content.replace('\n', " ").chars().take(90).collect();
            println!("  {score:.3}  {preview}");
        }
    }
    Ok(())
}
