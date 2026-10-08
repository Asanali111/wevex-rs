//! Lazy loading, idle unloading and a query cache around a model.
//!
//! The model costs ~150 MB of memory while loaded, and the daemon runs all
//! day, so it is loaded on first use and dropped after it has been idle for
//! a while (`unload_if_idle`, driven by the daemon). Repeated queries, which
//! agents send a lot, are answered from a small cache without the model.

use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::{Embedder, Result};

type Loader = Box<dyn Fn() -> Result<Box<dyn Embedder>> + Send + Sync>;

pub struct Managed {
    id: String,
    dim: usize,
    loader: Loader,
    state: Mutex<State>,
    cache: Mutex<QueryCache>,
}

struct State {
    model: Option<Box<dyn Embedder>>,
    last_used: Instant,
}

impl Managed {
    /// `id` and `dim` must be what `loader`'s model reports; they are needed
    /// before the model is ever loaded (to decide what needs re-embedding).
    pub fn new(
        id: impl Into<String>,
        dim: usize,
        cache_size: usize,
        loader: impl Fn() -> Result<Box<dyn Embedder>> + Send + Sync + 'static,
    ) -> Self {
        Self {
            id: id.into(),
            dim,
            loader: Box::new(loader),
            state: Mutex::new(State {
                model: None,
                last_used: Instant::now(),
            }),
            cache: Mutex::new(QueryCache::new(cache_size)),
        }
    }

    pub fn is_loaded(&self) -> bool {
        self.lock_state().model.is_some()
    }

    /// Embed one query, from the cache when possible.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        if let Some(v) = self.lock_cache().get(text) {
            return Ok(v);
        }
        let v = self.embed(&[text])?.remove(0);
        self.lock_cache().put(text.to_owned(), v.clone());
        Ok(v)
    }

    /// Drop the model if it has not been used for `idle`. Returns whether it
    /// was unloaded. The query cache is kept: it is small and still valid.
    pub fn unload_if_idle(&self, idle: Duration) -> bool {
        let mut state = self.lock_state();
        if state.model.is_some() && state.last_used.elapsed() >= idle {
            state.model = None;
            return true;
        }
        false
    }

    fn lock_state(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn lock_cache(&self) -> std::sync::MutexGuard<'_, QueryCache> {
        self.cache.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Embedder for Managed {
    fn id(&self) -> &str {
        &self.id
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let mut state = self.lock_state();
        if state.model.is_none() {
            state.model = Some((self.loader)()?);
        }
        state.last_used = Instant::now();
        state.model.as_ref().expect("loaded above").embed(texts)
    }
}

/// Least-recently-inserted eviction; plenty for repeated agent queries.
struct QueryCache {
    cap: usize,
    map: HashMap<String, Vec<f32>>,
    order: VecDeque<String>,
}

impl QueryCache {
    fn new(cap: usize) -> Self {
        Self {
            cap,
            map: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&self, k: &str) -> Option<Vec<f32>> {
        self.map.get(k).cloned()
    }

    fn put(&mut self, k: String, v: Vec<f32>) {
        if self.cap == 0 || self.map.contains_key(&k) {
            return;
        }
        if self.map.len() == self.cap {
            if let Some(old) = self.order.pop_front() {
                self.map.remove(&old);
            }
        }
        self.order.push_back(k.clone());
        self.map.insert(k, v);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::HashEmbedder;

    fn counting(loads: Arc<AtomicUsize>) -> Managed {
        Managed::new(HashEmbedder::ID, 32, 2, move || {
            loads.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(HashEmbedder::new(32)) as Box<dyn Embedder>)
        })
    }

    #[test]
    fn loads_lazily_unloads_when_idle_and_reloads_on_demand() {
        let loads = Arc::new(AtomicUsize::new(0));
        let m = counting(Arc::clone(&loads));
        assert!(!m.is_loaded());

        m.embed(&["a"]).unwrap();
        m.embed(&["b"]).unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 1);

        assert!(!m.unload_if_idle(Duration::from_secs(3600)));
        assert!(m.unload_if_idle(Duration::ZERO));
        assert!(!m.is_loaded());

        m.embed(&["c"]).unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn repeated_queries_skip_the_model() {
        let loads = Arc::new(AtomicUsize::new(0));
        let m = counting(Arc::clone(&loads));
        let first = m.embed_query("uni list").unwrap();
        m.unload_if_idle(Duration::ZERO);
        assert_eq!(m.embed_query("uni list").unwrap(), first);
        assert_eq!(loads.load(Ordering::SeqCst), 1, "served from cache");

        // Capacity 2: a third query evicts the oldest.
        m.embed_query("two").unwrap();
        m.embed_query("three").unwrap();
        m.unload_if_idle(Duration::ZERO);
        m.embed_query("uni list").unwrap();
        assert_eq!(loads.load(Ordering::SeqCst), 3);
    }
}
