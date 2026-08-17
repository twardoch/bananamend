//! this_file: crates/bananamendr/src/main.rs
//!
//! `bananamendr` — chat with, generate from, inspect and benchmark a
//! BananaMind-2 checkpoint directory.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use bananamendr::{GenerateOptions, Message, Pipeline, SamplingConfig};
use clap::{Args, Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "bananamendr",
    version,
    about = "Local inference for BananaMind-2 models"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Interactive chat using the checkpoint's chat template.
    Chat(ChatArgs),
    /// Continue a raw prompt without applying the chat template.
    Generate(GenerateArgs),
    /// Print the next-token logits for a prompt (parity and debugging).
    Logits(LogitsArgs),
    /// Print architecture and tokenizer facts about a checkpoint.
    Info(ModelArg),
    /// Measure prefill and decode throughput.
    Bench(BenchArgs),
}

#[derive(Args)]
struct ModelArg {
    /// Checkpoint directory containing config.json, model.safetensors and tokenizer.json.
    #[arg(short, long)]
    model: PathBuf,
}

#[derive(Args)]
struct SamplingArgs {
    /// 0 selects greedy decoding.
    #[arg(long, default_value_t = 0.8)]
    temperature: f32,
    #[arg(long, default_value_t = 40)]
    top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 1.1)]
    repetition_penalty: f32,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 256)]
    max_new_tokens: usize,
    /// KV-cache capacity; defaults to the checkpoint's full context.
    #[arg(long)]
    max_seq_len: Option<usize>,
}

impl SamplingArgs {
    fn options(&self) -> GenerateOptions {
        GenerateOptions {
            max_new_tokens: self.max_new_tokens,
            sampling: SamplingConfig {
                temperature: self.temperature,
                top_k: self.top_k,
                top_p: self.top_p,
                repetition_penalty: self.repetition_penalty,
                repetition_window: 64,
                seed: self.seed,
            },
            stop_on_eos: true,
            max_seq_len: self.max_seq_len,
        }
    }
}

#[derive(Args)]
struct ChatArgs {
    #[command(flatten)]
    model: ModelArg,
    #[command(flatten)]
    sampling: SamplingArgs,
    /// Optional system message prepended to the conversation.
    #[arg(long)]
    system: Option<String>,
    /// Answer a single message and exit instead of starting a REPL.
    #[arg(long)]
    prompt: Option<String>,
}

#[derive(Args)]
struct GenerateArgs {
    #[command(flatten)]
    model: ModelArg,
    #[command(flatten)]
    sampling: SamplingArgs,
    #[arg(long)]
    prompt: String,
}

#[derive(Args)]
struct LogitsArgs {
    #[command(flatten)]
    model: ModelArg,
    /// Raw text; tokenized as-is, without a chat template or added BOS.
    #[arg(long)]
    text: Option<String>,
    /// Explicit token ids, comma separated. Takes precedence over --text.
    #[arg(long)]
    tokens: Option<String>,
    /// Write the full logit vector as JSON to this path.
    #[arg(long)]
    out: Option<PathBuf>,
    /// How many top tokens to print.
    #[arg(long, default_value_t = 5)]
    top: usize,
}

#[derive(Args)]
struct BenchArgs {
    #[command(flatten)]
    model: ModelArg,
    #[arg(long, default_value_t = 64)]
    prompt_tokens: usize,
    #[arg(long, default_value_t = 64)]
    new_tokens: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Chat(args) => chat(args),
        Command::Generate(args) => generate(args),
        Command::Logits(args) => logits(args),
        Command::Info(args) => info(args),
        Command::Bench(args) => bench(args),
    }
}

fn load(model: &ModelArg) -> Result<Pipeline> {
    Pipeline::from_dir(&model.model)
        .with_context(|| format!("loading checkpoint {}", model.model.display()))
}

fn stream_stdout(step: &bananamendr::Step) {
    print!("{}", step.text);
    let _ = std::io::stdout().flush();
}

fn chat(args: ChatArgs) -> Result<()> {
    let pipeline = load(&args.model)?;
    let options = args.sampling.options();

    let mut messages = Vec::new();
    if let Some(system) = &args.system {
        messages.push(Message::new("system", system.clone()));
    }

    if let Some(prompt) = args.prompt {
        messages.push(Message::new("user", prompt));
        let generation = pipeline.chat(&messages, &options, stream_stdout)?;
        println!();
        eprintln!(
            "[{} prompt tokens, {} generated, {:.1} tok/s]",
            generation.prompt_tokens,
            generation.tokens.len(),
            generation.tokens_per_second()
        );
        return Ok(());
    }

    eprintln!(
        "{} loaded. Empty line or Ctrl-D exits.",
        args.model.model.display()
    );
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        print!("> ");
        std::io::stdout().flush()?;
        let Some(line) = lines.next() else { break };
        let line = line?;
        if line.trim().is_empty() {
            break;
        }
        messages.push(Message::new("user", line));
        let generation = pipeline.chat(&messages, &options, stream_stdout)?;
        println!();
        eprintln!(
            "[{} generated, {:.1} tok/s]",
            generation.tokens.len(),
            generation.tokens_per_second()
        );
        messages.push(Message::new("assistant", generation.text));
    }
    Ok(())
}

fn generate(args: GenerateArgs) -> Result<()> {
    let pipeline = load(&args.model)?;
    print!("{}", args.prompt);
    let generation = pipeline.generate(&args.prompt, &args.sampling.options(), stream_stdout)?;
    println!();
    eprintln!(
        "[{} prompt tokens, {} generated, prefill {:.3}s, {:.1} tok/s]",
        generation.prompt_tokens,
        generation.tokens.len(),
        generation.prefill_seconds,
        generation.tokens_per_second()
    );
    Ok(())
}

fn logits(args: LogitsArgs) -> Result<()> {
    let pipeline = load(&args.model)?;
    let tokens: Vec<u32> = match (&args.tokens, &args.text) {
        (Some(list), _) => list
            .split(',')
            .map(|s| s.trim().parse::<u32>())
            .collect::<Result<_, _>>()
            .context("--tokens must be a comma separated list of integers")?,
        (None, Some(text)) => pipeline.tokenizer.encode(text)?,
        (None, None) => anyhow::bail!("provide --text or --tokens"),
    };

    let values = pipeline.logits(&tokens)?;
    let mut ranked: Vec<(usize, f32)> = values.iter().copied().enumerate().collect();
    ranked.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));

    println!("tokens: {tokens:?}");
    for (id, value) in ranked.iter().take(args.top) {
        let text = pipeline.tokenizer.decode_one(*id as u32, false)?;
        println!("  {id:>6} {value:>12.6}  {text:?}");
    }
    if let Some(path) = args.out {
        let json = serde_json::json!({ "tokens": tokens, "logits": values });
        std::fs::write(&path, serde_json::to_string(&json)?)
            .with_context(|| format!("writing {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

fn info(args: ModelArg) -> Result<()> {
    let pipeline = load(&args)?;
    let c = pipeline.config();
    println!("model_type            {}", c.model_type);
    println!("architectures         {:?}", c.architectures);
    println!("hidden_size           {}", c.hidden_size);
    println!("intermediate_size     {}", c.intermediate_size);
    println!("num_hidden_layers     {}", c.num_hidden_layers);
    println!(
        "attention heads       {} query / {} kv (n_rep {})",
        c.num_attention_heads,
        c.num_key_value_heads,
        c.n_rep()
    );
    println!("head_dim              {}", c.head_dim);
    println!("vocab_size            {}", c.vocab_size);
    println!("max_position_embeddings {}", c.max_position_embeddings);
    println!("rope_theta            {}", c.rope_theta);
    println!("rms_norm_eps          {}", c.rms_norm_eps);
    println!("tie_word_embeddings   {}", c.tie_word_embeddings);
    println!("embedding scale       {:.6}", c.embd_scale());
    println!("tokenizer vocab       {}", pipeline.tokenizer.vocab_size());
    println!(
        "special tokens        bos {} ({}), eos {} ({}), pad {}",
        c.bos_token_id,
        pipeline.tokenizer.bos_token(),
        c.eos_token_id,
        pipeline.tokenizer.eos_token(),
        c.pad_token_id
    );
    Ok(())
}

fn bench(args: BenchArgs) -> Result<()> {
    let pipeline = load(&args.model)?;
    let vocab = pipeline.config().vocab_size as u32;
    // Deterministic pseudo-random prompt so runs are comparable.
    let prompt: Vec<u32> = (0..args.prompt_tokens.max(1))
        .map(|i| ((i as u32).wrapping_mul(2654435761) % vocab).max(4))
        .collect();
    let options = GenerateOptions {
        max_new_tokens: args.new_tokens,
        sampling: SamplingConfig::greedy(),
        stop_on_eos: false,
        max_seq_len: None,
    };
    let generation = pipeline.generate_tokens(&prompt, &options, |_| {})?;
    println!(
        "prefill {} tokens in {:.3}s ({:.1} tok/s)",
        generation.prompt_tokens,
        generation.prefill_seconds,
        generation.prompt_tokens as f64 / generation.prefill_seconds
    );
    println!(
        "decode  {} tokens in {:.3}s ({:.1} tok/s)",
        generation.tokens.len(),
        generation.decode_seconds,
        generation.tokens_per_second()
    );
    Ok(())
}
