// this_file: docs/assets/js/demo.js
//
// The browser demonstration of bananamendr.
//
// The page downloads the four parts of a checkpoint from Hugging Face and gives
// them to the WebAssembly build of the engine. All of the work after the
// download stays in the browser.
//
// The engine runs on the thread of the page. A generation therefore stops the
// page from drawing until it finishes. The page writes the answer in one piece
// after the generation, and the callback counts the tokens while they arrive.

import init, { Model, version } from "../wasm/bananamendr.js";
import { HEAVY_MEGABYTES, MODELS, findModel } from "./models.js";

const HUB = "https://huggingface.co";
const PARTS = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"];
// A checkpoint needs these three. The fourth only names the special tokens, and
// the engine uses the usual names when it is absent.
const REQUIRED = new Set(["config.json", "tokenizer.json", "model.safetensors"]);

const EXAMPLES = [
  { mode: "chat", text: "Name one ocean." },
  { mode: "chat", text: "Why is the sky blue?" },
  { mode: "chat", text: "Write three rules for a safe kitchen." },
  { mode: "generate", text: "The capital of France is" },
];

const element = (id) => document.getElementById(id);
const ui = {
  model: element("model"),
  custom: element("custom"),
  modelNote: element("modelNote"),
  load: element("load"),
  status: element("status"),
  progressBox: element("progressBox"),
  progress: element("progress"),
  progressText: element("progressText"),
  info: element("info"),
  loadedName: element("loadedName"),
  modelType: element("modelType"),
  layers: element("layers"),
  weights: element("weights"),
  form: element("form"),
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
let loadedRepo = "";
let ready = false;

function say(message, kind = "info") {
  ui.status.textContent = message;
  ui.status.dataset.kind = kind;
}

function setBusy(busy) {
  ui.controls.disabled = busy;
  ui.run.disabled = busy;
  ui.tokenize.disabled = busy;
  ui.load.disabled = busy;
}

function megabytes(bytes) {
  return `${(bytes / 1e6).toFixed(1)} MB`;
}

/** The repository that the page must load: the list, or the field below it. */
function selectedRepo() {
  const custom = ui.custom.value.trim();
  return custom !== "" ? custom : ui.model.value;
}

function describeSelection() {
  const custom = ui.custom.value.trim();
  if (custom !== "") {
    ui.modelNote.textContent =
      `Your own repository: ${custom}. It needs config.json, tokenizer.json and ` +
      "model.safetensors, and the engine reads the quantized forms as well.";
    ui.load.textContent = "Download and start";
    return;
  }
  const entry = findModel(ui.model.value);
  if (!entry) {
    ui.modelNote.textContent = "";
    return;
  }
  const warning =
    entry.megabytes > HEAVY_MEGABYTES
      ? " A browser probably cannot hold this one."
      : "";
  ui.modelNote.textContent = `${entry.note} Quality: ${entry.quality}.${warning}`;
  ui.load.textContent = `Download ${entry.megabytes} MB and start`;
}

// Downloads one file and reports the progress. The weights are large, so the
// page reads them in pieces instead of waiting in silence.
async function download(repo, name, onProgress) {
  const response = await fetch(`${HUB}/${repo}/resolve/main/${name}`);
  if (!response.ok) {
    if (!REQUIRED.has(name) && response.status === 404) {
      return null; // an optional file that this repository does not hold
    }
    // Hugging Face answers 401 for a repository that does not exist and for one
    // that needs a token, so the message must cover both.
    if (response.status === 401 || response.status === 403 || response.status === 404) {
      throw new Error(
        `${repo} did not give ${name}. The repository does not exist, or it is private.`,
      );
    }
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

function showInfo(repo, info) {
  const forms = Object.fromEntries(info.storage);
  ui.loadedName.textContent = repo;
  ui.modelType.textContent = info.model_type;
  ui.layers.textContent = `${info.num_hidden_layers} × ${info.hidden_size}`;
  ui.weights.textContent = megabytes(info.weight_bytes);
  ui.form.textContent = Object.entries(forms)
    .map(([part, value]) =>
      part.startsWith("matrices_")
        ? `${value} matrices in ${part.replace("matrices_", "")}`
        : `${part} in ${value}`,
    )
    .join(", ");
  ui.context.textContent = `${info.max_position_embeddings} tokens`;
  ui.version.textContent = version();
  ui.info.hidden = false;
}

async function load() {
  const repo = selectedRepo();
  const entry = findModel(repo);
  if (repo === loadedRepo && model !== null) {
    say("That checkpoint is already loaded.", "ok");
    return;
  }

  setBusy(true);
  ui.progressBox.hidden = false;
  ui.progress.value = 0;

  // The previous model must go before the next one arrives, or the two together
  // fill the memory of the module.
  if (model !== null) {
    model.free();
    model = null;
    loadedRepo = "";
    ready = false;
    ui.info.hidden = true;
  }

  const decoder = new TextDecoder();
  const files = {};
  try {
    if (!ready) {
      await init(new URL("../wasm/bananamendr_bg.wasm", import.meta.url));
    }
    for (const name of PARTS) {
      say(`Downloading ${name} from ${repo}…`);
      files[name] = await download(repo, name, (received, total) => {
        ui.progress.value = total > 0 ? (received / total) * 100 : 0;
        ui.progressText.textContent = total
          ? `${name}: ${megabytes(received)} of ${megabytes(total)}`
          : `${name}: ${megabytes(received)}`;
      });
    }

    say("Building the model…");
    const tokenizerConfig = files["tokenizer_config.json"];
    model = Model.fromParts(
      decoder.decode(files["config.json"]),
      decoder.decode(files["tokenizer.json"]),
      tokenizerConfig ? decoder.decode(tokenizerConfig) : undefined,
      files["model.safetensors"],
    );
    loadedRepo = repo;
    ready = true;

    showInfo(repo, model.info());
    ui.progressBox.hidden = true;
    setBusy(false);
    say(
      entry?.research
        ? "Ready. This checkpoint is a measurement, and its answers are not useful."
        : "Ready.",
      entry?.research ? "error" : "ok",
    );
  } catch (error) {
    ui.progressBox.hidden = true;
    setBusy(false);
    ui.load.disabled = false;
    ui.controls.disabled = model === null;
    const hint =
      entry && entry.megabytes > HEAVY_MEGABYTES
        ? " This checkpoint is probably too large for a browser; use the 8-bit copy."
        : "";
    say(`The model did not load: ${error}.${hint}`, "error");
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
    say("Load a checkpoint first.", "error");
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

  // The browser must paint the message above before the generation starts. A
  // short timer gives it that chance. `requestAnimationFrame` would be more
  // exact, but a browser stops those calls in a tab that nobody looks at, and
  // the generation would then never start.
  setTimeout(() => {
    {
      const pieces = [];
      try {
        const settings = options();
        const collect = (piece) => pieces.push(piece);
        const result =
          ui.mode.value === "chat"
            ? model.chat([{ role: "user", content: text }], settings, collect)
            : model.generate(text, settings, collect);

        ui.output.textContent = result.text;
        ui.promptTokens.textContent = String(result.prompt_tokens);
        ui.newTokens.textContent = String(result.tokens.length);
        ui.prefill.textContent = `${result.prefill_seconds.toFixed(2)} s`;
        ui.decode.textContent = `${result.decode_seconds.toFixed(2)} s`;
        ui.speed.textContent = `${result.tokens_per_second.toFixed(1)} tokens/s`;
        ui.eos.textContent = result.finished_by_eos ? "yes" : "no, the limit stopped it";
        ui.stats.hidden = false;
        // The engine stops at the end token and does not send it to the
        // callback, so the callback sees one piece less than the token count.
        const note = result.finished_by_eos
          ? " The last token is the end marker, and the engine does not show it."
          : "";
        say(`${result.tokens.length} tokens in ${result.decode_seconds.toFixed(1)} s.${note}`, "ok");
      } catch (error) {
        say(`The generation failed: ${error}`, "error");
      } finally {
        setBusy(false);
      }
    }
  }, 16);
}

function showTokens() {
  if (model === null) {
    say("Load a checkpoint first.", "error");
    return;
  }
  const text = ui.prompt.value.trim();
  try {
    const tokens =
      ui.mode.value === "chat"
        ? model.chatTokens([{ role: "user", content: text }])
        : model.tokenize(text);
    const ids = Array.from(tokens);
    const lines = ids.map(
      (id) =>
        `${String(id).padStart(6)}  ${JSON.stringify(model.detokenize(new Uint32Array([id]), false))}`,
    );
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

function fillModels() {
  ui.model.replaceChildren(
    ...MODELS.map((entry) => {
      const option = document.createElement("option");
      option.value = entry.repo;
      option.textContent = `${entry.label} — ${entry.megabytes} MB${
        entry.research ? " (does not work)" : ""
      }`;
      option.selected = Boolean(entry.recommended);
      return option;
    }),
  );
}

ui.load.addEventListener("click", load);
ui.run.addEventListener("click", run);
ui.tokenize.addEventListener("click", showTokens);
ui.clear.addEventListener("click", () => {
  ui.output.textContent = "";
  ui.stats.hidden = true;
});
ui.example.addEventListener("change", loadExample);
ui.model.addEventListener("change", describeSelection);
ui.custom.addEventListener("input", describeSelection);

fillModels();
loadExample();
describeSelection();
say("Select a checkpoint and use the button.");
