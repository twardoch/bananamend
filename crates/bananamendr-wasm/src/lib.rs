//! this_file: crates/bananamendr-wasm/src/lib.rs
//!
//! WebAssembly bindings for `bananamendr`.
//!
//! A browser has no files, so the page must download the four parts of a
//! checkpoint and give them to `Model.from_parts`. The page keeps the parts; this
//! module keeps only the model.
//!
//! The decode loop runs on the thread that calls it. A browser tab has one main
//! thread, so a long generation stops the page from drawing. Put this module in a
//! Web Worker, or give a small number to `max_new_tokens`.

use bananamendr::{GenerateOptions, Message, Pipeline, SamplingConfig, Step};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Facts about a loaded checkpoint.
#[derive(Serialize)]
pub struct Info {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub tokenizer_vocab_size: usize,
    /// The form of each part: `float32`, `int8` or `ternary`, and how many
    /// matrices hold each form.
    pub storage: Vec<(String, String)>,
    /// The bytes that the weights need in memory. A quantized checkpoint keeps
    /// its codes, so this number stays near the size of the file.
    pub weight_bytes: usize,
}

/// The result of a generation.
#[derive(Serialize)]
pub struct Generation {
    pub text: String,
    pub tokens: Vec<u32>,
    pub prompt_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub tokens_per_second: f64,
    pub finished_by_eos: bool,
}

/// One turn of a conversation.
#[derive(Deserialize)]
pub struct WasmMessage {
    pub role: String,
    pub content: String,
}

/// The controls for a generation. Each field has a default.
#[derive(Deserialize)]
#[serde(default)]
pub struct Options {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
    pub seed: u64,
    pub max_seq_len: Option<usize>,
    pub stop_on_eos: bool,
}

impl Default for Options {
    fn default() -> Self {
        // The same defaults as the Python module: greedy, and a short answer.
        Self {
            max_new_tokens: 64,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 64,
            seed: 0,
            max_seq_len: None,
            stop_on_eos: true,
        }
    }
}

impl Options {
    fn to_generate_options(&self) -> GenerateOptions {
        GenerateOptions {
            max_new_tokens: self.max_new_tokens,
            sampling: SamplingConfig {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                repetition_penalty: self.repetition_penalty,
                repetition_window: self.repetition_window,
                seed: self.seed,
            },
            max_seq_len: self.max_seq_len,
            stop_on_eos: self.stop_on_eos,
        }
    }
}

fn to_js_error(error: bananamendr::Error) -> JsValue {
    JsValue::from_str(&error.to_string())
}

fn parse<T: for<'de> Deserialize<'de> + Default>(value: JsValue) -> Result<T, JsValue> {
    if value.is_undefined() || value.is_null() {
        return Ok(T::default());
    }
    serde_wasm_bindgen::from_value(value).map_err(|e| JsValue::from_str(&e.to_string()))
}

/// Gives the version of the crate that built this module.
#[wasm_bindgen]
pub fn version() -> String {
    env!("CARGO_PKG_VERSION").to_string()
}

/// A loaded BananaMind-2 checkpoint.
#[wasm_bindgen]
pub struct Model {
    pipeline: Pipeline,
}

#[wasm_bindgen]
impl Model {
    /// Builds a model from the four parts of a checkpoint.
    ///
    /// Give the text of `config.json`, the text of `tokenizer.json`, the text of
    /// `tokenizer_config.json` (or nothing), and the bytes of
    /// `model.safetensors`.
    #[wasm_bindgen(js_name = fromParts)]
    pub fn from_parts(
        config_json: &str,
        tokenizer_json: &str,
        tokenizer_config_json: Option<String>,
        weights: &[u8],
    ) -> Result<Model, JsValue> {
        let pipeline = Pipeline::from_parts(
            config_json,
            tokenizer_json,
            tokenizer_config_json.as_deref(),
            weights,
        )
        .map_err(to_js_error)?;
        Ok(Model { pipeline })
    }

    /// Gives the architecture and tokenizer facts of the checkpoint.
    pub fn info(&self) -> Result<JsValue, JsValue> {
        let c = self.pipeline.config();
        let info = Info {
            model_type: c.model_type.clone(),
            hidden_size: c.hidden_size,
            intermediate_size: c.intermediate_size,
            num_hidden_layers: c.num_hidden_layers,
            num_attention_heads: c.num_attention_heads,
            num_key_value_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
            vocab_size: c.vocab_size,
            max_position_embeddings: c.max_position_embeddings,
            bos_token_id: c.bos_token_id,
            eos_token_id: c.eos_token_id,
            tokenizer_vocab_size: self.pipeline.tokenizer.vocab_size(),
            storage: self.pipeline.model.storage(),
            weight_bytes: self.pipeline.model.weight_bytes(),
        };
        serde_wasm_bindgen::to_value(&info).map_err(|e| JsValue::from_str(&e.to_string()))
    }

    /// Gives the token ids of a text.
    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>, JsValue> {
        self.pipeline.tokenizer.encode(text).map_err(to_js_error)
    }

    /// Gives the text of token ids.
    pub fn detokenize(
        &self,
        tokens: Vec<u32>,
        skip_special_tokens: bool,
    ) -> Result<String, JsValue> {
        self.pipeline
            .tokenizer
            .decode(&tokens, skip_special_tokens)
            .map_err(to_js_error)
    }

    /// Applies the chat format of the checkpoint and gives the text.
    #[wasm_bindgen(js_name = applyChatTemplate)]
    pub fn apply_chat_template(&self, messages: JsValue) -> Result<String, JsValue> {
        let messages = to_messages(messages)?;
        Ok(self.pipeline.tokenizer.apply_chat_template(&messages, true))
    }

    /// Applies the chat format and gives the token ids.
    #[wasm_bindgen(js_name = chatTokens)]
    pub fn chat_tokens(&self, messages: JsValue) -> Result<Vec<u32>, JsValue> {
        let messages = to_messages(messages)?;
        self.pipeline
            .tokenizer
            .encode_chat(&messages, true)
            .map_err(to_js_error)
    }

    /// Continues a text. The chat format is not applied.
    ///
    /// `on_token` is optional. If you give a function, this module calls it for
    /// each new token with the text of that token and its id.
    pub fn generate(
        &self,
        prompt: &str,
        options: JsValue,
        on_token: Option<js_sys::Function>,
    ) -> Result<JsValue, JsValue> {
        let options: Options = parse(options)?;
        let mut tokens = vec![self.pipeline.config().bos_token_id];
        tokens.extend(
            self.pipeline
                .tokenizer
                .encode(prompt)
                .map_err(to_js_error)?,
        );
        self.run(tokens, options, on_token)
    }

    /// Applies the chat format, and then writes the answer of the assistant.
    pub fn chat(
        &self,
        messages: JsValue,
        options: JsValue,
        on_token: Option<js_sys::Function>,
    ) -> Result<JsValue, JsValue> {
        let options: Options = parse(options)?;
        let messages = to_messages(messages)?;
        let tokens = self
            .pipeline
            .tokenizer
            .encode_chat(&messages, true)
            .map_err(to_js_error)?;
        self.run(tokens, options, on_token)
    }

    /// Gives the scores of all of the next possible tokens for a text.
    ///
    /// The text goes to the tokenizer without a change. No token is added, which
    /// is what the Python module does as well.
    pub fn logits(&self, text: &str) -> Result<Vec<f32>, JsValue> {
        let tokens = self.pipeline.tokenizer.encode(text).map_err(to_js_error)?;
        self.pipeline.logits(&tokens).map_err(to_js_error)
    }
}

impl Model {
    fn run(
        &self,
        tokens: Vec<u32>,
        options: Options,
        on_token: Option<js_sys::Function>,
    ) -> Result<JsValue, JsValue> {
        let generate_options = options.to_generate_options();
        let generation = match on_token {
            Some(callback) => {
                let this = JsValue::null();
                self.pipeline
                    .generate_tokens(&tokens, &generate_options, |step: &Step| {
                        // A failure in the page must not stop the generation, so
                        // the result of the call is not used.
                        let _ = callback.call2(
                            &this,
                            &JsValue::from_str(&step.text),
                            &JsValue::from_f64(step.token as f64),
                        );
                    })
                    .map_err(to_js_error)?
            }
            None => self
                .pipeline
                .generate_tokens(&tokens, &generate_options, |_| {})
                .map_err(to_js_error)?,
        };

        let result = Generation {
            tokens_per_second: generation.tokens_per_second(),
            text: generation.text,
            tokens: generation.tokens,
            prompt_tokens: generation.prompt_tokens,
            prefill_seconds: generation.prefill_seconds,
            decode_seconds: generation.decode_seconds,
            finished_by_eos: generation.finished_by_eos,
        };
        serde_wasm_bindgen::to_value(&result).map_err(|e| JsValue::from_str(&e.to_string()))
    }
}

fn to_messages(value: JsValue) -> Result<Vec<Message>, JsValue> {
    let parsed: Vec<WasmMessage> = serde_wasm_bindgen::from_value(value).map_err(|e| {
        JsValue::from_str(&format!(
            "messages must be a list of objects with a role and content: {e}"
        ))
    })?;
    Ok(parsed
        .into_iter()
        .map(|m| Message::new(m.role, m.content))
        .collect())
}
