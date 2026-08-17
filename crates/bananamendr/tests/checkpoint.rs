//! this_file: crates/bananamendr/tests/checkpoint.rs
//!
//! Integration tests against a real checkpoint. They are skipped (passing with a
//! note) when `ref/BananaMind-2-Nano-Chat` has not been fetched, so `cargo test`
//! works on a fresh clone. Numeric parity against `transformers` lives in
//! `tests/test_parity.py`.

use std::path::PathBuf;

use bananamendr::{Error, GenerateOptions, Message, Pipeline, SamplingConfig};

fn checkpoint() -> Option<PathBuf> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|root| root.join("ref").join("BananaMind-2-Nano-Chat"))?;
    if dir.join("model.safetensors").is_file() {
        Some(dir)
    } else {
        eprintln!("skipping: {} not fetched", dir.display());
        None
    }
}

fn pipeline() -> Option<Pipeline> {
    let dir = checkpoint()?;
    Some(Pipeline::from_dir(&dir).expect("checkpoint should load"))
}

#[test]
fn test_config_when_nano_loaded_then_published_dimensions() {
    let Some(pipeline) = pipeline() else { return };
    let c = pipeline.config();
    assert_eq!(c.model_type, "bananamind2_nano");
    assert_eq!(c.num_hidden_layers, 10);
    assert_eq!(c.hidden_size, 256);
    assert_eq!(c.num_attention_heads, 4);
    assert_eq!(c.num_key_value_heads, 2);
    assert_eq!(c.head_dim, 64);
    assert_eq!(c.vocab_size, 8192);
    assert_eq!(c.embd_scale(), 16.0);
}

#[test]
fn test_chat_template_when_rendered_then_matches_published_template() {
    let Some(pipeline) = pipeline() else { return };
    let rendered = pipeline.tokenizer.apply_chat_template(
        &[
            Message::new("system", "Be brief."),
            Message::new("user", "Hi"),
            Message::new("assistant", "Hello"),
            Message::new("user", "Again"),
        ],
        true,
    );
    assert_eq!(
        rendered,
        "<|bos|><|system|>\nBe brief.\n<|user|>\nHi\n<|assistant|>\nHello<|eos|>\n\
         <|user|>\nAgain\n<|assistant|>\n"
    );
}

#[test]
fn test_greedy_generation_when_repeated_then_identical() {
    let Some(pipeline) = pipeline() else { return };
    let options = GenerateOptions {
        max_new_tokens: 16,
        sampling: SamplingConfig::greedy(),
        ..GenerateOptions::default()
    };
    let messages = [Message::new("user", "What is the capital of France?")];
    let first = pipeline.chat(&messages, &options, |_| {}).unwrap();
    let second = pipeline.chat(&messages, &options, |_| {}).unwrap();
    assert_eq!(
        first.tokens, second.tokens,
        "greedy decoding must be stable"
    );
    assert!(
        first.text.contains("Paris"),
        "expected Paris in {:?}",
        first.text
    );
}

#[test]
fn test_decode_from_cache_matches_fresh_prefill() {
    // Tokens produced through the KV cache must equal those produced when the
    // same prefix is prefilled from scratch: k tokens generated, then re-fed as
    // part of the prompt, must yield the same continuation.
    let Some(pipeline) = pipeline() else { return };
    let options = GenerateOptions {
        max_new_tokens: 12,
        sampling: SamplingConfig::greedy(),
        ..GenerateOptions::default()
    };
    let prompt = pipeline
        .tokenizer
        .encode_chat(&[Message::new("user", "Name three metals.")], true)
        .unwrap();

    let cached = pipeline
        .generate_tokens(&prompt, &options, |_| {})
        .unwrap()
        .tokens;
    assert!(cached.len() > 4, "need a few tokens to compare");

    let split = 4;
    let mut extended = prompt.clone();
    extended.extend_from_slice(&cached[..split]);
    let restarted = pipeline
        .generate_tokens(
            &extended,
            &GenerateOptions {
                max_new_tokens: cached.len() - split,
                ..options.clone()
            },
            |_| {},
        )
        .unwrap()
        .tokens;
    assert_eq!(
        &cached[split..],
        &restarted[..],
        "decoding through the cache diverged from a fresh prefill"
    );

    // A shorter prompt must produce different logits — a cheap guard against a
    // position or cache bug that ignores the tail of the prompt.
    let full = pipeline.logits(&prompt).unwrap();
    let shorter = pipeline.logits(&prompt[..prompt.len() - 1]).unwrap();
    assert_ne!(full, shorter);
}

#[test]
fn test_streaming_deltas_when_joined_then_equal_final_text() {
    let Some(pipeline) = pipeline() else { return };
    let options = GenerateOptions {
        max_new_tokens: 24,
        sampling: SamplingConfig::greedy(),
        ..GenerateOptions::default()
    };
    let mut joined = String::new();
    let generation = pipeline
        .chat(
            &[Message::new("user", "List two colours.")],
            &options,
            |step| joined.push_str(&step.text),
        )
        .unwrap();
    assert_eq!(joined, generation.text);
}

#[test]
fn test_generation_when_context_exhausted_then_context_overflow_error() {
    let Some(pipeline) = pipeline() else { return };
    let options = GenerateOptions {
        max_new_tokens: 32,
        sampling: SamplingConfig::greedy(),
        max_seq_len: Some(4),
        ..GenerateOptions::default()
    };
    let error = pipeline
        .generate_tokens(&[1, 2, 3, 4, 5], &options, |_| {})
        .expect_err("a 5-token prompt must not fit a 4-token cache");
    assert!(
        matches!(error, Error::ContextOverflow { .. }),
        "unexpected error: {error}"
    );
}

#[test]
fn test_token_out_of_range_is_rejected() {
    let Some(pipeline) = pipeline() else { return };
    let error = pipeline
        .logits(&[999_999])
        .expect_err("out-of-vocabulary ids must be rejected");
    assert!(matches!(error, Error::TokenOutOfRange { .. }));
}

#[test]
fn test_empty_prompt_is_rejected() {
    let Some(pipeline) = pipeline() else { return };
    let error = pipeline
        .generate_tokens(&[], &GenerateOptions::default(), |_| {})
        .expect_err("empty prompts must be rejected");
    assert!(matches!(error, Error::EmptyPrompt));
}
