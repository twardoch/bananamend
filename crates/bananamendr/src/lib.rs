//! this_file: crates/bananamendr/src/lib.rs
//!
//! CPU inference for the BananaMind-2 chat checkpoints (Nano, Mini, Pro).
//!
//! ```no_run
//! use bananamendr::{GenerateOptions, Message, Pipeline};
//!
//! let pipeline = Pipeline::from_dir("ref/BananaMind-2-Nano-Chat".as_ref())?;
//! let out = pipeline.chat(
//!     &[Message::new("user", "Hello!")],
//!     &GenerateOptions::default(),
//!     |_| {},
//! )?;
//! println!("{}", out.text);
//! # Ok::<(), bananamendr::Error>(())
//! ```

pub mod config;
pub mod error;
pub mod model;
pub mod ops;
pub mod sample;
pub mod tokenizer;
pub mod weights;

#[cfg(feature = "std-fs")]
use std::path::Path;
// `std::time::Instant` panics in a browser, because WebAssembly has no clock of
// its own. `web_time::Instant` reads the clock of the page and has the same
// interface, so the code below does not change.
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

pub use config::Config;
pub use error::Error;
pub use model::{Model, State};
pub use sample::{Rng, SamplingConfig};
pub use tokenizer::{Message, Tokenizer};
pub use weights::Weights;

#[derive(Debug, Clone)]
pub struct GenerateOptions {
    pub max_new_tokens: usize,
    pub sampling: SamplingConfig,
    /// Stop as soon as the model emits `eos_token_id`.
    pub stop_on_eos: bool,
    /// KV-cache capacity; clamped to the checkpoint's `max_position_embeddings`.
    /// `None` allocates the full context.
    pub max_seq_len: Option<usize>,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 128,
            sampling: SamplingConfig::default(),
            stop_on_eos: true,
            max_seq_len: None,
        }
    }
}

/// One streamed step: the new token and the text it added.
#[derive(Debug, Clone)]
pub struct Step {
    pub token: u32,
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct Generation {
    /// Decoded completion, special tokens removed.
    pub text: String,
    /// Generated token ids, including a trailing EOS if one was produced.
    pub tokens: Vec<u32>,
    pub prompt_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    /// True when generation stopped because EOS was emitted rather than because
    /// the token or context budget ran out.
    pub finished_by_eos: bool,
}

impl Generation {
    pub fn tokens_per_second(&self) -> f64 {
        if self.decode_seconds <= 0.0 {
            return 0.0;
        }
        self.tokens.len() as f64 / self.decode_seconds
    }
}

/// Length in bytes of the longest shared prefix of `a` and `b`, always on a
/// character boundary so the remainder can be sliced safely.
fn common_prefix_len(a: &str, b: &str) -> usize {
    let mut index = 0;
    for (ca, cb) in a.chars().zip(b.chars()) {
        if ca != cb {
            break;
        }
        index += ca.len_utf8();
    }
    index
}

/// A loaded checkpoint: weights plus tokenizer.
pub struct Pipeline {
    pub model: Model,
    pub tokenizer: Tokenizer,
}

impl Pipeline {
    /// Reads a complete checkpoint directory.
    #[cfg(feature = "std-fs")]
    pub fn from_dir(dir: &Path) -> Result<Self, Error> {
        Ok(Self {
            model: Model::from_dir(dir)?,
            tokenizer: Tokenizer::from_dir(dir)?,
        })
    }

    /// Builds a pipeline from parts that you already have in memory.
    ///
    /// A browser uses this function, because a browser has no files. Give the
    /// text of `config.json`, the text of `tokenizer.json`, the text of
    /// `tokenizer_config.json` (or `None`), and the bytes of
    /// `model.safetensors`.
    pub fn from_parts(
        config_json: &str,
        tokenizer_json: &str,
        tokenizer_config_json: Option<&str>,
        weights: &[u8],
    ) -> Result<Self, Error> {
        let config = Config::from_json(config_json)?;
        let model = Model::from_parts(config, Weights::from_bytes(weights)?)?;
        Ok(Self {
            model,
            tokenizer: Tokenizer::from_json(tokenizer_json, tokenizer_config_json)?,
        })
    }

    pub fn config(&self) -> &Config {
        &self.model.config
    }

    pub fn new_state(&self, options: &GenerateOptions) -> State {
        match options.max_seq_len {
            Some(len) => State::new(&self.model, len),
            None => State::full(&self.model),
        }
    }

    /// Logits for the next token after `tokens`, computed from a fresh state.
    pub fn logits(&self, tokens: &[u32]) -> Result<Vec<f32>, Error> {
        let mut state = State::full(&self.model);
        self.model.forward_tokens(&mut state, tokens)?;
        Ok(state.logits().to_vec())
    }

    /// Generates from raw prompt token ids, calling `on_step` for each new token.
    pub fn generate_tokens(
        &self,
        prompt: &[u32],
        options: &GenerateOptions,
        mut on_step: impl FnMut(&Step),
    ) -> Result<Generation, Error> {
        if prompt.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        let mut state = self.new_state(options);
        let eos = self.model.config.eos_token_id;
        let mut rng = Rng::new(options.sampling.seed);

        let prefill_start = Instant::now();
        self.model.forward_tokens(&mut state, prompt)?;
        let prefill_seconds = prefill_start.elapsed().as_secs_f64();

        let mut history: Vec<u32> = prompt.to_vec();
        let mut generated: Vec<u32> = Vec::new();
        let mut emitted = String::new();
        let mut finished_by_eos = false;

        let decode_start = Instant::now();
        for _ in 0..options.max_new_tokens {
            let mut logits = state.logits().to_vec();
            let token = sample::sample(&mut logits, &history, &options.sampling, &mut rng);
            generated.push(token);
            history.push(token);

            if token == eos && options.stop_on_eos {
                finished_by_eos = true;
                break;
            }

            // Decode the whole completion each step and emit the new suffix, so
            // multi-byte characters split across BPE pieces surface intact. A
            // character still missing bytes decodes to U+FFFD; hold that tail
            // back until the next token completes it.
            let decoded = self.tokenizer.decode(&generated, true)?;
            let stable = decoded.trim_end_matches(char::REPLACEMENT_CHARACTER);
            let delta = stable[common_prefix_len(&emitted, stable)..].to_string();
            emitted = stable.to_string();
            on_step(&Step { token, text: delta });

            if state.position() >= state.max_seq_len {
                break;
            }
            let pos = state.position();
            self.model.forward(&mut state, token, pos)?;
        }
        let decode_seconds = decode_start.elapsed().as_secs_f64();

        Ok(Generation {
            text: self.tokenizer.decode(&generated, true)?,
            tokens: generated,
            prompt_tokens: prompt.len(),
            prefill_seconds,
            decode_seconds,
            finished_by_eos,
        })
    }

    /// Generates a continuation of raw text (no chat template applied).
    pub fn generate(
        &self,
        prompt: &str,
        options: &GenerateOptions,
        on_step: impl FnMut(&Step),
    ) -> Result<Generation, Error> {
        let mut tokens = vec![self.model.config.bos_token_id];
        tokens.extend(self.tokenizer.encode(prompt)?);
        self.generate_tokens(&tokens, options, on_step)
    }

    /// Applies the chat template, then generates the assistant turn.
    pub fn chat(
        &self,
        messages: &[Message],
        options: &GenerateOptions,
        on_step: impl FnMut(&Step),
    ) -> Result<Generation, Error> {
        let tokens = self.tokenizer.encode_chat(messages, true)?;
        self.generate_tokens(&tokens, options, on_step)
    }
}
