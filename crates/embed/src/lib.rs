//! Local text embeddings.
//!
//! [`Embedder`] is the interface the rest of Wevex uses. The real model,
//! [`BgeSmall`], is behind the `bge` feature (on by default) so crates that
//! only need the interface, like `wevex-core`, build without the ML stack.
//! [`HashEmbedder`] is a fast, deterministic stand-in for tests.

#[cfg(feature = "bge")]
mod bge;
#[cfg(feature = "bge")]
mod download;
mod hash;
mod managed;

#[cfg(feature = "bge")]
pub use bge::BgeSmall;
pub use hash::HashEmbedder;
pub use managed::Managed;

pub type Result<T, E = Error> = std::result::Result<T, E>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("model download failed: {0}")]
    Download(String),

    #[error("embedding model: {0}")]
    Model(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub trait Embedder: Send + Sync {
    /// Identifies the model *and* its weights. Stored next to every vector;
    /// a different id means old vectors are re-embedded, never mixed.
    fn id(&self) -> &str;
    fn dim(&self) -> usize;
    /// One unit-length vector of [`Embedder::dim`] values per text, in order.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Most tokens of text one input can hold, special tokens excluded.
    /// Anything longer is silently truncated by the model, so callers split
    /// long text first (see `wevex_core::embedding`).
    fn max_tokens(&self) -> usize {
        usize::MAX
    }

    /// Byte offset at which each token of `text` starts, special tokens
    /// excluded. The default treats each whitespace-separated word as one
    /// token.
    fn token_starts(&self, text: &str) -> Result<Vec<usize>> {
        Ok(word_starts(text))
    }
}

/// Byte offset of each whitespace-separated word.
pub fn word_starts(text: &str) -> Vec<usize> {
    let mut starts = Vec::new();
    let mut in_word = false;
    for (i, c) in text.char_indices() {
        if c.is_whitespace() {
            in_word = false;
        } else if !in_word {
            starts.push(i);
            in_word = true;
        }
    }
    starts
}

/// Dot product. For unit-length vectors this is the cosine similarity.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Scale `v` to unit length in place (left as is if it is all zeros).
pub fn normalize(v: &mut [f32]) {
    let norm = dot(v, v).sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn word_starts_handles_unicode_and_runs_of_space() {
        let t = "  héllo  wörld\nok";
        let s = word_starts(t);
        assert_eq!(s.len(), 3);
        assert_eq!(&t[s[0]..s[0] + 6], "héllo");
        assert_eq!(&t[s[2]..], "ok");
    }
}
