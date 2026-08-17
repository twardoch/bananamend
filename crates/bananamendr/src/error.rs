//! this_file: crates/bananamendr/src/error.rs

use std::path::PathBuf;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("invalid config: {0}")]
    Config(String),
    #[error("safetensors: {0}")]
    SafeTensors(#[from] safetensors::SafeTensorError),
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    #[error("missing tensor {0}")]
    MissingTensor(String),
    #[error("tensor {name}: expected shape {expected:?}, found {found:?}")]
    Shape {
        name: String,
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    #[error("tensor {name}: expected F32, found {found:?}")]
    Dtype {
        name: String,
        found: safetensors::Dtype,
    },
    #[error(
        "context length exceeded: {requested} tokens requested, \
         max_position_embeddings is {limit}"
    )]
    ContextOverflow { requested: usize, limit: usize },
    #[error("token id {id} is outside the vocabulary of {vocab_size}")]
    TokenOutOfRange { id: u32, vocab_size: usize },
    #[error("prompt is empty")]
    EmptyPrompt,
    #[error("quantized checkpoint: {0}")]
    Quantization(String),
}

impl Error {
    #[cfg(feature = "std-fs")]
    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}
