//! this_file: crates/bananamendr/src/tokenizer.rs
//!
//! Thin wrapper over the Hugging Face `tokenizers` crate plus the chat format.
//!
//! `tokenizer.json` is loaded verbatim, so encoding is byte-identical to
//! `transformers`. The chat template shipped by all three checkpoints is
//! identical and simple enough to render directly instead of pulling in a Jinja
//! engine; `chat_template_parity` in the test suite pins that equivalence.

use std::path::Path;

use serde::Deserialize;
use tokenizers::Tokenizer as HfTokenizer;

use crate::error::Error;

#[derive(Debug, Clone, Deserialize)]
struct TokenizerConfig {
    #[serde(default = "default_bos")]
    bos_token: String,
    #[serde(default = "default_eos")]
    eos_token: String,
}

fn default_bos() -> String {
    "<|bos|>".to_string()
}

fn default_eos() -> String {
    "<|eos|>".to_string()
}

#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
        }
    }
}

pub struct Tokenizer {
    inner: HfTokenizer,
    bos_token: String,
    eos_token: String,
}

impl Tokenizer {
    pub fn from_dir(dir: &Path) -> Result<Self, Error> {
        let tokenizer_path = dir.join("tokenizer.json");
        let inner = HfTokenizer::from_file(&tokenizer_path)
            .map_err(|e| Error::Tokenizer(format!("{}: {e}", tokenizer_path.display())))?;

        let config_path = dir.join("tokenizer_config.json");
        let config: TokenizerConfig = match std::fs::read_to_string(&config_path) {
            Ok(text) => serde_json::from_str(&text)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => TokenizerConfig {
                bos_token: default_bos(),
                eos_token: default_eos(),
            },
            Err(e) => return Err(Error::io(&config_path, e)),
        };

        Ok(Self {
            inner,
            bos_token: config.bos_token,
            eos_token: config.eos_token,
        })
    }

    /// Encodes raw text. Special tokens present in the text (`<|bos|>` and
    /// friends) are still recognised; no extra BOS is prepended, because the
    /// chat template already emits one.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| Error::Tokenizer(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String, Error> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|e| Error::Tokenizer(e.to_string()))
    }

    /// Decodes a single token id. Byte-level BPE pieces can be partial UTF-8, so
    /// callers streaming token by token should buffer until the result is valid;
    /// `decode` over the accumulated ids is the safe path.
    pub fn decode_one(&self, id: u32, skip_special_tokens: bool) -> Result<String, Error> {
        self.decode(&[id], skip_special_tokens)
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn bos_token(&self) -> &str {
        &self.bos_token
    }

    pub fn eos_token(&self) -> &str {
        &self.eos_token
    }

    /// Renders the checkpoint's chat template.
    ///
    /// Equivalent to `transformers`' `apply_chat_template(..., tokenize=False)`
    /// for the template shipped with BananaMind-2 Nano, Mini and Pro.
    pub fn apply_chat_template(&self, messages: &[Message], add_generation_prompt: bool) -> String {
        let mut out = String::from(&self.bos_token);
        for message in messages {
            match message.role.as_str() {
                "assistant" => {
                    out.push_str("<|assistant|>\n");
                    out.push_str(&message.content);
                    out.push_str(&self.eos_token);
                    out.push('\n');
                }
                "system" => {
                    out.push_str("<|system|>\n");
                    out.push_str(&message.content);
                    out.push('\n');
                }
                "user" => {
                    out.push_str("<|user|>\n");
                    out.push_str(&message.content);
                    out.push('\n');
                }
                other => {
                    out.push_str(&format!("<|{other}|>\n"));
                    out.push_str(&message.content);
                    out.push('\n');
                }
            }
        }
        if add_generation_prompt {
            out.push_str("<|assistant|>\n");
        }
        out
    }

    pub fn encode_chat(
        &self,
        messages: &[Message],
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>, Error> {
        self.encode(&self.apply_chat_template(messages, add_generation_prompt))
    }
}
