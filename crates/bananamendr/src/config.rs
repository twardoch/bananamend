//! this_file: crates/bananamendr/src/config.rs
//!
//! `config.json` as published in the BananaMind-2 checkpoints. The Nano, Mini
//! and Pro repositories ship byte-identical modeling code, so one struct with
//! the published dimensions describes all three.

use serde::Deserialize;

use crate::error::Error;

#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    /// Read from the file rather than derived: the checkpoints set it
    /// explicitly and it is not always `hidden_size / num_attention_heads`.
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    #[serde(default = "default_true")]
    pub tie_word_embeddings: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    #[serde(default)]
    pub model_type: String,
    #[serde(default)]
    pub architectures: Vec<String>,
}

fn default_true() -> bool {
    true
}

impl Config {
    pub fn from_json(text: &str) -> Result<Self, Error> {
        let config: Config = serde_json::from_str(text)?;
        config.validate()?;
        Ok(config)
    }

    /// Queries per key/value head (grouped-query attention repeat factor).
    pub fn n_rep(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// `sqrt(hidden_size)`, the scale the reference applies to input
    /// embeddings. It cannot be folded into `wte` because `lm_head` is tied to
    /// the same table.
    pub fn embd_scale(&self) -> f32 {
        (self.hidden_size as f32).sqrt()
    }

    fn validate(&self) -> Result<(), Error> {
        if self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Error::Config(format!(
                "num_attention_heads ({}) must be a positive multiple of \
                 num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(Error::Config(format!(
                "head_dim ({}) must be even for interleaved RoPE",
                self.head_dim
            )));
        }
        Ok(())
    }
}
