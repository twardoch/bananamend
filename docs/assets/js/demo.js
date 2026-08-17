// this_file: docs/assets/js/demo.js
//
// The browser demonstration of bananamendr.
//
// The page downloads the four parts of the Nano checkpoint from Hugging Face and
// gives them to the WebAssembly build of the engine. All of the work after the
// download stays in the browser.
//
// The engine runs on the thread of the page. A generation therefore stops the page
// from drawing until it finishes. The page writes the answer in one piece after
// the generation, and the callback collects the tokens while they arrive.

import init, { Model, version } from "../wasm/bananamendr.js";

const REPO = "https://huggingface.co/BananaMind/BananaMind-2-Nano-Chat/resolve/main";
const PARTS = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"];

const EXAMPLES = [
  { mode: "chat", text: "Name one ocean." },
  { mode: "chat", text: "Why is the sky blue?" },
  { mode: "chat", text: "Write three rules for a safe kitchen." },
  { mode: "generate", text: "The capital of France is" },
];

const element = (id) => document.getElementById(id);
const ui = {
  load: element("load"),
  status: element("status"),
  progressBox: element("progressBox"),
  progress: element("progress"),
  progressText: element("progressText"),
  info: element("info"),
  modelType: element("modelType"),
  layers: element("layers"),
  hidden: element("hidden"),
  vocab: element("vocab"),
  context: element("context"),
  version: element("version"),
  controls: element("controls"),
  mode: element("mode"),
  example: element("example"),
  prompt: element("prompt"),
  maxTokens: element("maxTokens"),
  temperature: element("temperature"),
  topK: element("topK"),
  seed: element("seed"),
  run: element("run"),
  tokenize: element("tokenize"),
  clear: element("clear"),
  output: element("output"),
  stats: element("stats"),
  promptTokens: element("promptTokens"),
  newTokens: element("newTokens"),
  prefill: element("prefill"),
  decode: element("decode"),
  speed: element("speed"),
  eos: element("eos"),
};

let model = null;

function say(message, kind = "info") {
  ui.status.textContent = message;
  ui.status.dataset.kind = kind;
}

// daisyUI has no class for a disabled fieldset that also disables the buttons in
// it, so the page sets both.
function setBusy(busy) {
  ui.controls.disabled = busy;
  ui.run.disabled = busy;
  ui.tokenize.disabled = busy;
}

function megabytes(bytes) {
  return `${(bytes / 1e6).toFixed(1)} MB`;
}

// Downloads one file and reports the progress. The safetensors file is large, so
// the page reads it in pieces instead of waiting in silence.
async function download(name, onProgress) {
  const response = await fetch(`${REPO}/${name}`);
  if (!response.ok) {
    throw new Error(`${name}: the server answered ${response.status}`);
  }
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body) {
    return new Uint8Array(await response.arrayBuffer());
  }
  const reader = response.body.getReader();
  const pieces = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) {
      break;
    }
    pieces.push(value);
    received += value.length;
    onProgress(received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const piece of pieces) {
    bytes.set(piece, offset);
    offset += piece.length;
  }
  return bytes;
}

async function load() {
  ui.load.disabled = true;
  ui.progressBox.hidden = false;
  const decoder = new TextDecoder();
  const files = {};

  try {
    await init(new URL("../wasm/bananamendr_bg.wasm", import.meta.url));
    for (const name of PARTS) {
      say(`Downloading ${name}…`);
      files[name] = await download(name, (received, total) => {
        const percent = total > 0 ? (received / total) * 100 : 0;
        ui.progress.value = percent;
        ui.progressText.textContent = total
          ? `${name}: ${megabytes(received)} of ${megabytes(total)}`
          : `${name}: ${megabytes(received)}`;
      });
    }

    say("Building the model…");
    model = Model.fromParts(
      decoder.decode(files["config.json"]),
      decoder.decode(files["tokenizer.json"]),
      decoder.decode(files["tokenizer_config.json"]),
      files["model.safetensors"],
    );

    const info = model.info();
    ui.modelType.textContent = info.model_type;
    ui.layers.textContent = String(info.num_hidden_layers);
    ui.hidden.textContent = `${info.hidden_size} (heads ${info.num_attention_heads}/${info.num_key_value_heads})`;
    ui.vocab.textContent = String(info.vocab_size);
    ui.context.textContent = `${info.max_position_embeddings} tokens`;
    ui.version.textContent = version();
    ui.info.hidden = false;
    setBusy(false);
    ui.progressBox.hidden = true;
    say("The model is ready.", "ok");
  } catch (error) {
    ui.load.disabled = false;
    ui.progressBox.hidden = true;
    say(`The model did not load: ${error}`, "error");
  }
}

function options() {
  const number = (input, fallback) => {
    const value = Number.parseFloat(input.value);
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    max_new_tokens: Math.max(1, Math.round(number(ui.maxTokens, 48))),
    temperature: Math.max(0, number(ui.temperature, 0)),
    top_k: Math.max(0, Math.round(number(ui.topK, 0))),
    top_p: 1.0,
    repetition_penalty: 1.1,
    seed: Math.max(0, Math.round(number(ui.seed, 0))),
    stop_on_eos: true,
  };
}

function run() {
  if (model === null) {
    return;
  }
  const text = ui.prompt.value.trim();
  if (text === "") {
    say("Write some text first.", "error");
    return;
  }

  setBusy(true);
  ui.output.textContent = "";
  say("The model writes an answer. The page cannot draw until it finishes…");

  // The browser must paint the message above before the generation starts.
  requestAnimationFrame(() => {
    setTimeout(() => {
      const pieces = [];
      try {
        const settings = options();
        const result =
          ui.mode.value === "chat"
            ? model.chat([{ role: "user", content: text }], settings, (piece) => {
                pieces.push(piece);
              })
            : model.generate(text, settings, (piece) => {
                pieces.push(piece);
              });

        ui.output.textContent = result.text;
        ui.promptTokens.textContent = String(result.prompt_tokens);
        ui.newTokens.textContent = String(result.tokens.length);
        ui.prefill.textContent = `${result.prefill_seconds.toFixed(2)} s`;
        ui.decode.textContent = `${result.decode_seconds.toFixed(2)} s`;
        ui.speed.textContent = `${result.tokens_per_second.toFixed(1)} tokens/s`;
        ui.eos.textContent = result.finished_by_eos ? "yes" : "no, the limit stopped it";
        ui.stats.hidden = false;
        say(
          `${result.tokens.length} tokens, and the callback saw ${pieces.length}.`,
          "ok",
        );
      } catch (error) {
        say(`The generation failed: ${error}`, "error");
      } finally {
        setBusy(false);
      }
    }, 0);
  });
}

function showTokens() {
  if (model === null) {
    return;
  }
  const text = ui.prompt.value.trim();
  try {
    const tokens =
      ui.mode.value === "chat"
        ? model.chatTokens([{ role: "user", content: text }])
        : model.tokenize(text);
    const ids = Array.from(tokens);
    const lines = ids.map((id) => `${String(id).padStart(6)}  ${JSON.stringify(model.detokenize(new Uint32Array([id]), false))}`);
    ui.output.textContent = `${ids.length} tokens\n\n${lines.join("\n")}`;
    say(`${ids.length} tokens.`, "ok");
  } catch (error) {
    say(`The tokenizer failed: ${error}`, "error");
  }
}

function loadExample() {
  const example = EXAMPLES[Number(ui.example.value)] ?? EXAMPLES[0];
  ui.prompt.value = example.text;
  ui.mode.value = example.mode;
}

ui.load.addEventListener("click", load);
ui.run.addEventListener("click", run);
ui.tokenize.addEventListener("click", showTokens);
ui.clear.addEventListener("click", () => {
  ui.output.textContent = "";
  ui.stats.hidden = true;
});
ui.example.addEventListener("change", loadExample);

loadExample();
say("The model is not loaded. Use the button to download it (about 40 MB).");
