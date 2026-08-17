//! this_file: crates/bananamendr/examples/generate.rs
//!
//! Library usage from Rust:
//!
//! ```sh
//! cargo run --release --example generate -- ref/BananaMind-2-Nano-Chat "Hello!"
//! ```

use std::io::Write;

use bananamendr::{GenerateOptions, Message, Pipeline, SamplingConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .ok_or("usage: generate <checkpoint-dir> [prompt]")?;
    let prompt = args.next().unwrap_or_else(|| "Hello!".to_string());

    let pipeline = Pipeline::from_dir(dir.as_ref())?;
    let options = GenerateOptions {
        max_new_tokens: 96,
        sampling: SamplingConfig {
            temperature: 0.7,
            top_k: 40,
            top_p: 0.95,
            repetition_penalty: 1.1,
            repetition_window: 64,
            seed: 42,
        },
        ..GenerateOptions::default()
    };

    let generation = pipeline.chat(&[Message::new("user", prompt)], &options, |step| {
        print!("{}", step.text);
        let _ = std::io::stdout().flush();
    })?;
    println!();
    eprintln!(
        "[{} prompt tokens, {} generated, {:.1} tok/s]",
        generation.prompt_tokens,
        generation.tokens.len(),
        generation.tokens_per_second()
    );
    Ok(())
}
