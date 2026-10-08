//! A deterministic, dependency-free embedder for tests.
//!
//! Each lowercase word is hashed into a few of `dim` buckets, so texts that
//! share words get similar vectors. It has no notion of meaning ("uni list"
//! and "target schools" share nothing), so it must never be used for real
//! recall; its id makes that visible wherever vectors are stored.

use crate::{Embedder, Result, normalize};

pub struct HashEmbedder {
    dim: usize,
}

impl HashEmbedder {
    pub const ID: &'static str = "test-hash";

    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl Embedder for HashEmbedder {
    fn id(&self) -> &str {
        Self::ID
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|text| {
                let mut v = vec![0.0f32; self.dim];
                for word in text
                    .split(|c: char| !c.is_alphanumeric())
                    .filter(|w| !w.is_empty())
                {
                    let h = fnv1a(&word.to_lowercase());
                    for k in 0..3u64 {
                        let bucket = (h.rotate_left(k as u32 * 21) % self.dim as u64) as usize;
                        v[bucket] += 1.0;
                    }
                }
                normalize(&mut v);
                v
            })
            .collect())
    }
}

/// FNV-1a: tiny and stable across Rust versions, unlike `DefaultHasher`.
fn fnv1a(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dot;

    #[test]
    fn shared_words_score_higher_and_output_is_stable() {
        let e = HashEmbedder::new(64);
        let v = e
            .embed(&[
                "sqlite storage layer",
                "Layer, storage: SQLite",
                "college essay",
            ])
            .unwrap();
        assert!(dot(&v[0], &v[1]) > 0.99);
        assert!(dot(&v[0], &v[2]) < 0.5);
        assert_eq!(v[0], e.embed(&["sqlite storage layer"]).unwrap()[0]);
    }
}
