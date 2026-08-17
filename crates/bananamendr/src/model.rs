//! this_file: crates/bananamendr/src/model.rs
//!
//! The BananaMind-2 decoder: RMSNorm pre-norm blocks, grouped-query attention
//! with per-head Q/K RMSNorm before interleaved RoPE, SwiGLU MLP, tied
//! embedding/output matrix, and an input embedding scaled by `sqrt(hidden_size)`.
//!
//! Weights are immutable and shared (`Model`); everything mutable lives in
//! `State`, so one loaded model can serve several independent sequences.

#[cfg(feature = "std-fs")]
use std::path::Path;

use crate::config::Config;
use crate::error::Error;
use crate::matrix::Matrix;
use crate::ops::{self, RopeTable};
use crate::weights::Weights;

struct Layer {
    ln_1: Vec<f32>,
    ln_2: Vec<f32>,
    // A projection can be 32-bit floats, 8-bit codes or ternary codes. The
    // `quantization` block of `config.json` decides, and the multiplication
    // works the same way for all three.
    q_proj: Matrix,
    k_proj: Matrix,
    v_proj: Matrix,
    o_proj: Matrix,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    w_gate: Matrix,
    w_up: Matrix,
    w_down: Matrix,
}

pub struct Model {
    pub config: Config,
    wte: Matrix,
    ln_f: Vec<f32>,
    /// Present only when the checkpoint unties the output matrix; the published
    /// checkpoints all tie it to `wte`.
    lm_head: Option<Matrix>,
    layers: Vec<Layer>,
    rope: RopeTable,
}

impl Model {
    /// Reads `config.json` and `model.safetensors` from a checkpoint directory.
    #[cfg(feature = "std-fs")]
    pub fn from_dir(dir: &Path) -> Result<Self, Error> {
        let config_path = dir.join("config.json");
        let text = std::fs::read_to_string(&config_path).map_err(|e| Error::io(&config_path, e))?;
        let config = Config::from_json(&text)?;
        let weights = Weights::load(&dir.join("model.safetensors"))?;
        Self::from_parts(config, weights)
    }

    pub fn from_parts(config: Config, mut weights: Weights) -> Result<Self, Error> {
        let (hidden, ffn) = (config.hidden_size, config.intermediate_size);
        let q_dim = config.num_attention_heads * config.head_dim;
        let kv_dim = config.num_key_value_heads * config.head_dim;

        let quantization = config.quantization.clone();
        let quantization = quantization.as_ref();
        let wte = weights.take_matrix(
            "transformer.wte.weight",
            &[config.vocab_size, hidden],
            quantization,
        )?;
        let ln_f = weights.take("transformer.ln_f.weight", &[hidden])?;
        let has_lm_head = weights.contains("lm_head.weight")
            || quantization
                .map(|q| q.tensors.contains_key("lm_head.weight"))
                .unwrap_or(false);
        let lm_head = if has_lm_head {
            Some(weights.take_matrix(
                "lm_head.weight",
                &[config.vocab_size, hidden],
                quantization,
            )?)
        } else {
            None
        };

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("transformer.h.{i}.");
            layers.push(Layer {
                ln_1: weights.take(&format!("{p}ln_1.weight"), &[hidden])?,
                ln_2: weights.take(&format!("{p}ln_2.weight"), &[hidden])?,
                q_proj: weights.take_matrix(
                    &format!("{p}attn.q_proj.weight"),
                    &[q_dim, hidden],
                    quantization,
                )?,
                k_proj: weights.take_matrix(
                    &format!("{p}attn.k_proj.weight"),
                    &[kv_dim, hidden],
                    quantization,
                )?,
                v_proj: weights.take_matrix(
                    &format!("{p}attn.v_proj.weight"),
                    &[kv_dim, hidden],
                    quantization,
                )?,
                o_proj: weights.take_matrix(
                    &format!("{p}attn.o_proj.weight"),
                    &[hidden, q_dim],
                    quantization,
                )?,
                q_norm: weights.take(&format!("{p}attn.q_norm.weight"), &[config.head_dim])?,
                k_norm: weights.take(&format!("{p}attn.k_norm.weight"), &[config.head_dim])?,
                w_gate: weights.take_matrix(
                    &format!("{p}mlp.w_gate.weight"),
                    &[ffn, hidden],
                    quantization,
                )?,
                w_up: weights.take_matrix(
                    &format!("{p}mlp.w_up.weight"),
                    &[ffn, hidden],
                    quantization,
                )?,
                w_down: weights.take_matrix(
                    &format!("{p}mlp.w_down.weight"),
                    &[hidden, ffn],
                    quantization,
                )?,
            });
        }

        let rope = RopeTable::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        );
        Ok(Self {
            config,
            wte,
            ln_f,
            lm_head,
            layers,
            rope,
        })
    }

    fn output_matrix(&self) -> &Matrix {
        self.lm_head.as_ref().unwrap_or(&self.wte)
    }

    /// The form of each part of the model, for a report to a user.
    pub fn storage(&self) -> Vec<(&'static str, &'static str)> {
        let mut out = vec![("embedding", self.wte.kind())];
        if let Some(head) = &self.lm_head {
            out.push(("output", head.kind()));
        }
        if let Some(layer) = self.layers.first() {
            out.push(("attention", layer.q_proj.kind()));
            out.push(("mlp", layer.w_gate.kind()));
        }
        out
    }

    /// The number of bytes that the weights hold in memory.
    pub fn weight_bytes(&self) -> usize {
        let mut total = self.wte.bytes() + self.ln_f.len() * 4;
        if let Some(head) = &self.lm_head {
            total += head.bytes();
        }
        for layer in &self.layers {
            total += layer.q_proj.bytes()
                + layer.k_proj.bytes()
                + layer.v_proj.bytes()
                + layer.o_proj.bytes()
                + layer.w_gate.bytes()
                + layer.w_up.bytes()
                + layer.w_down.bytes()
                + (layer.ln_1.len() + layer.ln_2.len() + layer.q_norm.len() + layer.k_norm.len())
                    * 4;
        }
        total
    }

    /// Runs one token at position `pos`, leaving the next-token logits in
    /// `state.logits()`. `pos` must equal the number of tokens already in `state`.
    pub fn forward(&self, state: &mut State, token: u32, pos: usize) -> Result<(), Error> {
        let cfg = &self.config;
        if token as usize >= cfg.vocab_size {
            return Err(Error::TokenOutOfRange {
                id: token,
                vocab_size: cfg.vocab_size,
            });
        }
        if pos >= state.max_seq_len {
            return Err(Error::ContextOverflow {
                requested: pos + 1,
                limit: state.max_seq_len,
            });
        }

        let hidden = cfg.hidden_size;
        let head_dim = cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let embd_scale = cfg.embd_scale();

        match &self.wte {
            Matrix::Dense { data, .. } => {
                let row = &data[token as usize * hidden..(token as usize + 1) * hidden];
                for (dst, src) in state.x.iter_mut().zip(row) {
                    *dst = src * embd_scale;
                }
            }
            other => {
                // A quantized embedding needs one row rebuilt for this token.
                let row = &mut state.embedding[..hidden];
                other.row_into(token as usize, row);
                for (dst, src) in state.x.iter_mut().zip(row.iter()) {
                    *dst = src * embd_scale;
                }
            }
        }

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            ops::rms_norm(&mut state.xb, &state.x, &layer.ln_1, cfg.rms_norm_eps);

            layer.q_proj.matvec(&mut state.q, &state.xb);
            let k_slot = layer_idx * state.max_seq_len * kv_dim + pos * kv_dim;
            let v_slot = k_slot;
            layer
                .k_proj
                .matvec(&mut state.k_cache[k_slot..k_slot + kv_dim], &state.xb);
            layer
                .v_proj
                .matvec(&mut state.v_cache[v_slot..v_slot + kv_dim], &state.xb);

            // Per-head Q/K RMSNorm, then RoPE. Order matters: the reference
            // normalises before rotating.
            for h in 0..cfg.num_attention_heads {
                let head = &mut state.q[h * head_dim..(h + 1) * head_dim];
                ops::rms_norm_in_place(head, &layer.q_norm, cfg.rms_norm_eps);
                self.rope.apply(head, pos);
            }
            for h in 0..cfg.num_key_value_heads {
                let head = &mut state.k_cache[k_slot + h * head_dim..k_slot + (h + 1) * head_dim];
                ops::rms_norm_in_place(head, &layer.k_norm, cfg.rms_norm_eps);
                self.rope.apply(head, pos);
            }

            let layer_kv = layer_idx * state.max_seq_len * kv_dim;
            for h in 0..cfg.num_attention_heads {
                let kv_head = h / cfg.n_rep();
                let q = &state.q[h * head_dim..(h + 1) * head_dim];
                let scores = &mut state.scores[..pos + 1];
                for (t, score) in scores.iter_mut().enumerate() {
                    let base = layer_kv + t * kv_dim + kv_head * head_dim;
                    *score = ops::dot(q, &state.k_cache[base..base + head_dim]) * scale;
                }
                ops::softmax(scores);

                let out = &mut state.attn_out[h * head_dim..(h + 1) * head_dim];
                out.fill(0.0);
                for (t, &weight) in scores.iter().enumerate() {
                    let base = layer_kv + t * kv_dim + kv_head * head_dim;
                    let v = &state.v_cache[base..base + head_dim];
                    for i in 0..head_dim {
                        out[i] += weight * v[i];
                    }
                }
            }

            layer.o_proj.matvec(&mut state.xb, &state.attn_out);
            for i in 0..hidden {
                state.x[i] += state.xb[i];
            }

            ops::rms_norm(&mut state.xb, &state.x, &layer.ln_2, cfg.rms_norm_eps);
            layer.w_gate.matvec(&mut state.gate, &state.xb);
            layer.w_up.matvec(&mut state.up, &state.xb);
            for i in 0..cfg.intermediate_size {
                state.gate[i] = ops::silu(state.gate[i]) * state.up[i];
            }
            layer.w_down.matvec(&mut state.xb, &state.gate);
            for i in 0..hidden {
                state.x[i] += state.xb[i];
            }
        }

        ops::rms_norm(&mut state.xb, &state.x, &self.ln_f, cfg.rms_norm_eps);
        self.output_matrix().matvec(&mut state.logits, &state.xb);
        state.position = pos + 1;
        Ok(())
    }

    /// Feeds `tokens` sequentially starting at `state.position`, leaving the
    /// logits of the final token in `state.logits()`.
    pub fn forward_tokens(&self, state: &mut State, tokens: &[u32]) -> Result<(), Error> {
        if tokens.is_empty() {
            return Err(Error::EmptyPrompt);
        }
        let start = state.position;
        for (offset, &token) in tokens.iter().enumerate() {
            self.forward(state, token, start + offset)?;
        }
        Ok(())
    }
}

/// Per-sequence mutable state: KV cache plus scratch buffers.
pub struct State {
    pub max_seq_len: usize,
    position: usize,
    x: Vec<f32>,
    xb: Vec<f32>,
    q: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    attn_out: Vec<f32>,
    scores: Vec<f32>,
    logits: Vec<f32>,
    k_cache: Vec<f32>,
    v_cache: Vec<f32>,
    /// One row of a quantized embedding, rebuilt for each token.
    embedding: Vec<f32>,
}

impl State {
    /// `max_seq_len` is clamped to the checkpoint's `max_position_embeddings`;
    /// the KV cache is allocated up front for that many positions.
    pub fn new(model: &Model, max_seq_len: usize) -> Self {
        let cfg = &model.config;
        let max_seq_len = max_seq_len.clamp(1, cfg.max_position_embeddings);
        let q_dim = cfg.num_attention_heads * cfg.head_dim;
        let kv_dim = cfg.num_key_value_heads * cfg.head_dim;
        let cache_len = cfg.num_hidden_layers * max_seq_len * kv_dim;
        Self {
            max_seq_len,
            position: 0,
            x: vec![0.0; cfg.hidden_size],
            xb: vec![0.0; cfg.hidden_size],
            q: vec![0.0; q_dim],
            gate: vec![0.0; cfg.intermediate_size],
            up: vec![0.0; cfg.intermediate_size],
            attn_out: vec![0.0; q_dim],
            scores: vec![0.0; max_seq_len],
            logits: vec![0.0; cfg.vocab_size],
            k_cache: vec![0.0; cache_len],
            v_cache: vec![0.0; cache_len],
            embedding: vec![0.0; cfg.hidden_size],
        }
    }

    /// Full-context state.
    pub fn full(model: &Model) -> Self {
        Self::new(model, model.config.max_position_embeddings)
    }

    /// Number of tokens consumed so far, i.e. the position of the next token.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Discards the sequence, keeping the allocations for reuse.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    pub fn logits(&self) -> &[f32] {
        &self.logits
    }
}
