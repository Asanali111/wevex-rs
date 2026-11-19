//! Re-embed a *copy* of a Wevex database with the real model, check that
//! every part fits the model, then run a few hybrid recalls. The source
//! file is only read (backup API).
//!
//!     cargo run -p wevexd --example reembed -- [db] [query]...

use std::path::PathBuf;
use std::time::Instant;

use rusqlite::{Connection, OpenFlags};
use wevex_core::embedding::{VectorIndex, embed_pending, model_key, part_texts, pending};
use wevex_core::recall::recall;
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

    // Every fragment, split as embed_pending will split it ("none" matches
    // no stored model, so everything counts as pending).
    let all = store.read(|c| pending(c, "none", usize::MAX))?;
    let (mut parts, mut multi, mut max_tok) = (0, 0, 0);
    for p in &all {
        let texts = part_texts(&model, p)?;
        parts += texts.len();
        multi += usize::from(texts.len() > 1);
        for t in &texts {
            max_tok = max_tok.max(model.token_starts(t)?.len());
        }
    }
    println!(
        "parts           {parts} for {} fragments ({multi} split); largest {max_tok} tokens (limit {})",
        all.len(),
        model.max_tokens()
    );
    assert!(
        max_tok <= model.max_tokens(),
        "a part exceeds the model limit"
    );

    let t = Instant::now();
    let n = embed_pending(&store, &model)?;
    let took = t.elapsed();
    println!(
        "embedded        {n} fragments in {took:?} ({:.1} ms each)",
        took.as_secs_f64() * 1000.0 / n.max(1) as f64
    );

    let index = store.read(|c| VectorIndex::load(c, &model_key(&model), model.dim()))?;
    for q in &queries {
        let t = Instant::now();
        let hits = store.read(|c| recall(c, &model, &index, q, None, 3, 0.0))?;
        let search = t.elapsed();
        println!("\n\"{q}\"  (hybrid recall incl. query embedding {search:?})");
        for h in hits {
            let id = h.id.clone();
            let content: String = store.read(move |c| {
                Ok(
                    c.query_row("SELECT content FROM fragments WHERE id = ?1", [&id], |r| {
                        r.get(0)
                    })?,
                )
            })?;
            let legs = match (h.by_keyword, h.by_meaning) {
                (true, true) => "kw+mean",
                (true, false) => "kw",
                _ => "mean",
            };
            let preview: String = content.replace('\n', " ").chars().take(90).collect();
            println!("  {:.4} {legs:<7} {preview}", h.score);
        }
    }
    Ok(())
}
