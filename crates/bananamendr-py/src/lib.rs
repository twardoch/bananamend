//! this_file: crates/bananamendr-py/src/lib.rs
//!
//! Python bindings over `bananamendr`.
//!
//! ```python
//! import bananamendr
//! model = bananamendr.Model("ref/BananaMind-2-Nano-Chat")
//! print(model.chat([{"role": "user", "content": "Hello!"}], temperature=0.0))
//! ```

use std::path::PathBuf;

use ::bananamendr::{Error, GenerateOptions, Message, Pipeline, SamplingConfig, Step};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn to_py_err(error: Error) -> PyErr {
    match error {
        Error::Config(_) | Error::TokenOutOfRange { .. } | Error::EmptyPrompt => {
            PyValueError::new_err(error.to_string())
        }
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

fn parse_messages(messages: &Bound<'_, PyList>) -> PyResult<Vec<Message>> {
    let mut out = Vec::with_capacity(messages.len());
    for item in messages.iter() {
        let dict = item.cast::<PyDict>().map_err(|_| {
            PyValueError::new_err("each message must be a dict with 'role' and 'content'")
        })?;
        let role: String = dict
            .get_item("role")?
            .ok_or_else(|| PyValueError::new_err("message is missing 'role'"))?
            .extract()?;
        let content: String = dict
            .get_item("content")?
            .ok_or_else(|| PyValueError::new_err("message is missing 'content'"))?
            .extract()?;
        out.push(Message::new(role, content));
    }
    Ok(out)
}

/// A generation result.
#[pyclass(module = "bananamendr", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct Generation {
    /// Decoded completion with special tokens removed.
    pub text: String,
    pub tokens: Vec<u32>,
    pub prompt_tokens: usize,
    pub prefill_seconds: f64,
    pub decode_seconds: f64,
    pub tokens_per_second: f64,
    pub finished_by_eos: bool,
}

#[pymethods]
impl Generation {
    fn __repr__(&self) -> String {
        format!(
            "Generation(text={:?}, tokens={}, tokens_per_second={:.1}, finished_by_eos={})",
            self.text,
            self.tokens.len(),
            self.tokens_per_second,
            self.finished_by_eos
        )
    }

    fn __str__(&self) -> String {
        self.text.clone()
    }
}

impl From<::bananamendr::Generation> for Generation {
    fn from(g: ::bananamendr::Generation) -> Self {
        Self {
            tokens_per_second: g.tokens_per_second(),
            text: g.text,
            tokens: g.tokens,
            prompt_tokens: g.prompt_tokens,
            prefill_seconds: g.prefill_seconds,
            decode_seconds: g.decode_seconds,
            finished_by_eos: g.finished_by_eos,
        }
    }
}

/// A loaded BananaMind-2 checkpoint.
#[pyclass(module = "bananamendr", frozen)]
pub struct Model {
    pipeline: Pipeline,
    path: PathBuf,
}

#[pymethods]
impl Model {
    /// Loads `config.json`, `model.safetensors` and `tokenizer.json` from `path`.
    #[new]
    #[pyo3(signature = (path))]
    fn new(path: PathBuf) -> PyResult<Self> {
        let pipeline = Pipeline::from_dir(&path).map_err(to_py_err)?;
        Ok(Self { pipeline, path })
    }

    fn __repr__(&self) -> String {
        let c = self.pipeline.config();
        format!(
            "Model({:?}, model_type={:?}, layers={}, hidden={}, vocab={})",
            self.path, c.model_type, c.num_hidden_layers, c.hidden_size, c.vocab_size
        )
    }

    /// Architecture and tokenizer facts as a dict.
    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let c = self.pipeline.config();
        let d = PyDict::new(py);
        d.set_item("model_type", &c.model_type)?;
        d.set_item("architectures", c.architectures.clone())?;
        d.set_item("hidden_size", c.hidden_size)?;
        d.set_item("intermediate_size", c.intermediate_size)?;
        d.set_item("num_hidden_layers", c.num_hidden_layers)?;
        d.set_item("num_attention_heads", c.num_attention_heads)?;
        d.set_item("num_key_value_heads", c.num_key_value_heads)?;
        d.set_item("head_dim", c.head_dim)?;
        d.set_item("vocab_size", c.vocab_size)?;
        d.set_item("max_position_embeddings", c.max_position_embeddings)?;
        d.set_item("rope_theta", c.rope_theta)?;
        d.set_item("rms_norm_eps", c.rms_norm_eps)?;
        d.set_item("tie_word_embeddings", c.tie_word_embeddings)?;
        d.set_item("bos_token_id", c.bos_token_id)?;
        d.set_item("eos_token_id", c.eos_token_id)?;
        d.set_item("pad_token_id", c.pad_token_id)?;
        Ok(d)
    }

    /// The form of each part of the model: `float32`, `int8` or `ternary`.
    #[getter]
    fn storage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (part, kind) in self.pipeline.model.storage() {
            d.set_item(part, kind)?;
        }
        Ok(d)
    }

    /// Bytes that the weights need in memory. A quantized checkpoint keeps its
    /// codes, so this number stays near the size of the file.
    #[getter]
    fn weight_bytes(&self) -> usize {
        self.pipeline.model.weight_bytes()
    }

    /// Encodes text exactly as `transformers` would, without adding BOS.
    fn tokenize(&self, text: &str) -> PyResult<Vec<u32>> {
        self.pipeline.tokenizer.encode(text).map_err(to_py_err)
    }

    #[pyo3(signature = (tokens, skip_special_tokens = true))]
    fn detokenize(&self, tokens: Vec<u32>, skip_special_tokens: bool) -> PyResult<String> {
        self.pipeline
            .tokenizer
            .decode(&tokens, skip_special_tokens)
            .map_err(to_py_err)
    }

    /// Renders the checkpoint's chat template.
    #[pyo3(signature = (messages, add_generation_prompt = true))]
    fn apply_chat_template(
        &self,
        messages: &Bound<'_, PyList>,
        add_generation_prompt: bool,
    ) -> PyResult<String> {
        let messages = parse_messages(messages)?;
        Ok(self
            .pipeline
            .tokenizer
            .apply_chat_template(&messages, add_generation_prompt))
    }

    /// Next-token logits after `tokens`, or after tokenizing `text`.
    #[pyo3(signature = (*, tokens = None, text = None))]
    fn logits(
        &self,
        py: Python<'_>,
        tokens: Option<Vec<u32>>,
        text: Option<String>,
    ) -> PyResult<Vec<f32>> {
        let ids = match (tokens, text) {
            (Some(ids), _) => ids,
            (None, Some(text)) => self.pipeline.tokenizer.encode(&text).map_err(to_py_err)?,
            (None, None) => return Err(PyValueError::new_err("pass tokens= or text=")),
        };
        py.detach(|| self.pipeline.logits(&ids)).map_err(to_py_err)
    }

    /// Continues raw text. `on_token(text, token_id)` is called per step.
    #[pyo3(signature = (
        prompt,
        *,
        max_new_tokens = 128,
        temperature = 0.0,
        top_k = 0,
        top_p = 1.0,
        repetition_penalty = 1.0,
        seed = 0,
        max_seq_len = None,
        stop_on_eos = true,
        on_token = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
        max_seq_len: Option<usize>,
        stop_on_eos: bool,
        on_token: Option<Py<PyAny>>,
    ) -> PyResult<Generation> {
        let options = build_options(
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            max_seq_len,
            stop_on_eos,
        );
        let mut tokens = vec![self.pipeline.config().bos_token_id];
        tokens.extend(self.pipeline.tokenizer.encode(prompt).map_err(to_py_err)?);
        self.run(py, tokens, options, on_token)
    }

    /// Applies the chat template, then generates the assistant turn.
    #[pyo3(signature = (
        messages,
        *,
        max_new_tokens = 128,
        temperature = 0.0,
        top_k = 0,
        top_p = 1.0,
        repetition_penalty = 1.0,
        seed = 0,
        max_seq_len = None,
        stop_on_eos = true,
        on_token = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn chat(
        &self,
        py: Python<'_>,
        messages: &Bound<'_, PyList>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
        max_seq_len: Option<usize>,
        stop_on_eos: bool,
        on_token: Option<Py<PyAny>>,
    ) -> PyResult<Generation> {
        let options = build_options(
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            max_seq_len,
            stop_on_eos,
        );
        let messages = parse_messages(messages)?;
        let tokens = self
            .pipeline
            .tokenizer
            .encode_chat(&messages, true)
            .map_err(to_py_err)?;
        self.run(py, tokens, options, on_token)
    }

    /// Generates from explicit prompt token ids.
    #[pyo3(signature = (
        tokens,
        *,
        max_new_tokens = 128,
        temperature = 0.0,
        top_k = 0,
        top_p = 1.0,
        repetition_penalty = 1.0,
        seed = 0,
        max_seq_len = None,
        stop_on_eos = true,
        on_token = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn generate_tokens(
        &self,
        py: Python<'_>,
        tokens: Vec<u32>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        seed: u64,
        max_seq_len: Option<usize>,
        stop_on_eos: bool,
        on_token: Option<Py<PyAny>>,
    ) -> PyResult<Generation> {
        let options = build_options(
            max_new_tokens,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            seed,
            max_seq_len,
            stop_on_eos,
        );
        self.run(py, tokens, options, on_token)
    }
}

impl Model {
    fn run(
        &self,
        py: Python<'_>,
        tokens: Vec<u32>,
        options: GenerateOptions,
        on_token: Option<Py<PyAny>>,
    ) -> PyResult<Generation> {
        match on_token {
            // With a callback we keep the GIL: every step re-enters Python.
            Some(callback) => {
                let mut callback_error = None;
                let generation = self
                    .pipeline
                    .generate_tokens(&tokens, &options, |step: &Step| {
                        if callback_error.is_some() {
                            return;
                        }
                        if let Err(e) = callback.call1(py, (step.text.as_str(), step.token)) {
                            callback_error = Some(e);
                        }
                    })
                    .map_err(to_py_err)?;
                if let Some(e) = callback_error {
                    return Err(e);
                }
                Ok(generation.into())
            }
            // Without one, release the GIL for the whole decode loop.
            None => py
                .detach(|| self.pipeline.generate_tokens(&tokens, &options, |_| {}))
                .map(Into::into)
                .map_err(to_py_err),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_options(
    max_new_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repetition_penalty: f32,
    seed: u64,
    max_seq_len: Option<usize>,
    stop_on_eos: bool,
) -> GenerateOptions {
    GenerateOptions {
        max_new_tokens,
        sampling: SamplingConfig {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            repetition_window: 64,
            seed,
        },
        stop_on_eos,
        max_seq_len,
    }
}

#[pymodule]
fn bananamendr(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<Model>()?;
    m.add_class::<Generation>()?;
    Ok(())
}
