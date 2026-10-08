//! Local text embeddings: BAAI/bge-small-en-v1.5 run with candle, in pure
//! Rust, so the same code and the same vectors on macOS (Apple Silicon and
//! Intel) and Windows. ONNX Runtime was the first choice but ships no
//! prebuilt library for Intel Macs.

mod download;

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config};
use tokenizers::{PaddingParams, Tokenizer, TruncationParams};

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

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Model(e.to_string())
    }
}

/// Pinned model revision. Changing it changes every vector, so it is part
/// of the model id stored next to each vector.
const REVISION: &str = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a";

const FILES: &[download::ModelFile] = &[
    download::ModelFile {
        name: "config.json",
        size: 743,
        sha256: "094f8e891b932f2000c92cfc663bac4c62069f5d8af5b5278c4306aef3084750",
    },
    download::ModelFile {
        name: "tokenizer.json",
        size: 711_396,
        sha256: "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
    },
    download::ModelFile {
        name: "model.safetensors",
        size: 133_466_304,
        sha256: "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
    },
];

/// BAAI/bge-small-en-v1.5: 384 dimensions, CLS pooling, unit-length output.
pub struct BgeSmall {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl BgeSmall {
    pub const ID: &'static str = "bge-small-en-v1.5@5c38ec7";
    pub const DIM: usize = 384;
    /// Longest input in tokens; longer text is truncated.
    const MAX_TOKENS: usize = 512;

    /// Load from `models_dir/bge-small-en-v1.5`, downloading (~134 MB) and
    /// verifying the files on first use.
    pub fn load(models_dir: &Path) -> Result<Self> {
        let dir = models_dir.join("bge-small-en-v1.5");
        let base = format!("https://huggingface.co/BAAI/bge-small-en-v1.5/resolve/{REVISION}");
        download::ensure(&dir, &base, FILES)?;

        let config: Config = serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)
            .map_err(|e| Error::Model(format!("config.json: {e}")))?;

        let mut tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::Model(format!("tokenizer.json: {e}")))?;
        tokenizer.with_padding(Some(PaddingParams::default()));
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: Self::MAX_TOKENS,
                ..Default::default()
            }))
            .map_err(|e| Error::Model(format!("tokenizer: {e}")))?;

        let device = Device::Cpu;
        // SAFETY: the file is memory-mapped read-only; it was size-checked
        // above and is only replaced via an atomic rename.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[dir.join("model.safetensors")],
                DType::F32,
                &device,
            )?
        };
        let model = BertModel::load(vb, &config)?;
        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Embed a batch. Each output has [`Self::DIM`] values and unit length,
    /// so cosine similarity is a plain dot product.
    pub fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| Error::Model(format!("tokenize: {e}")))?;

        let stack = |f: &dyn Fn(&tokenizers::Encoding) -> &[u32]| -> Result<Tensor> {
            let rows = encodings
                .iter()
                .map(|e| Tensor::new(f(e), &self.device))
                .collect::<candle_core::Result<Vec<_>>>()?;
            Ok(Tensor::stack(&rows, 0)?)
        };
        let ids = stack(&|e| e.get_ids())?;
        let type_ids = stack(&|e| e.get_type_ids())?;
        let mask = stack(&|e| e.get_attention_mask())?;

        let hidden = self.model.forward(&ids, &type_ids, Some(&mask))?;
        // CLS pooling: bge uses the first token's hidden state.
        let cls = hidden.narrow(1, 0, 1)?.squeeze(1)?;
        let norm = cls.sqr()?.sum_keepdim(1)?.sqrt()?;
        let unit = cls.broadcast_div(&norm)?;
        Ok(unit.to_vec2::<f32>()?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models_dir() -> std::path::PathBuf {
        std::env::var_os("WEVEX_MODEL_CACHE")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join("wevex-model-cache"))
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Downloads the model (~134 MB) on first run: `cargo test -- --ignored`.
    #[test]
    #[ignore]
    fn embeds_and_ranks_by_meaning() {
        let m = BgeSmall::load(&models_dir()).unwrap();
        let v = m
            .embed(&[
                "uni list",
                "target schools for college applications",
                "rust borrow checker error",
            ])
            .unwrap();
        assert!(v.iter().all(|x| x.len() == BgeSmall::DIM));
        assert!(v.iter().all(|x| (dot(x, x) - 1.0).abs() < 1e-4));
        let related = dot(&v[0], &v[1]);
        let unrelated = dot(&v[0], &v[2]);
        eprintln!("uni list ~ target schools {related:.3}, ~ borrow checker {unrelated:.3}");
        assert!(related > unrelated);
    }

    /// Padding a short text to a longer batch must not change its vector.
    #[test]
    #[ignore]
    fn batching_does_not_change_vectors() {
        let m = BgeSmall::load(&models_dir()).unwrap();
        let alone = m.embed(&["short"]).unwrap();
        let batched = m
            .embed(&[
                "short",
                "a much longer sentence that forces padding of the first one",
            ])
            .unwrap();
        assert!(dot(&alone[0], &batched[0]) > 0.9999);
    }
}
