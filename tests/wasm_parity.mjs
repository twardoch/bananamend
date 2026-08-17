// this_file: tests/wasm_parity.mjs
//
// Compares the WebAssembly build of bananamendr with the Python extension module.
//
// Both builds use the same Rust code. Greedy decoding must therefore give the
// same tokens for the same checkpoint. This test proves it: the Python side writes
// a reference file, and this script gives the same checkpoint to the WebAssembly
// module and compares every value.
//
// Usage:
//   node tests/wasm_parity.mjs <wasm-dir> <checkpoint-dir> <reference.json>

import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

const [modulePath, checkpointPath, referencePath] = process.argv.slice(2);
if (!modulePath || !checkpointPath || !referencePath) {
  console.error("usage: node tests/wasm_parity.mjs <wasm-dir> <checkpoint-dir> <reference.json>");
  process.exit(2);
}

const wasm = await import(pathToFileURL(join(modulePath, "bananamendr.js")).href);
const reference = JSON.parse(readFileSync(referencePath, "utf8"));

let checks = 0;
const failures = [];

// wasm-bindgen gives token ids as a Uint32Array. JSON.stringify writes an object
// for a typed array, so the comparison needs a plain array first.
function plain(value) {
  return ArrayBuffer.isView(value) ? Array.from(value) : value;
}

function check(name, actual, expected) {
  checks += 1;
  actual = plain(actual);
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures.push(
      `${name}\n  wasm:   ${JSON.stringify(actual)}\n  python: ${JSON.stringify(expected)}`,
    );
  }
}

check("version", wasm.version(), reference.version);

// A browser downloads these four parts. Node reads them from the disk, but the
// module receives them in exactly the same form: text, text, text and bytes.
const read = (name) => readFileSync(join(checkpointPath, name), "utf8");
const model = wasm.Model.fromParts(
  read("config.json"),
  read("tokenizer.json"),
  read("tokenizer_config.json"),
  readFileSync(join(checkpointPath, "model.safetensors")),
);

const info = model.info();
for (const [key, value] of Object.entries(reference.info)) {
  check(`info.${key}`, info[key], value);
}
check("info.tokenizer_vocab_size", info.tokenizer_vocab_size, info.vocab_size);

for (const example of reference.tokenize) {
  const label = `tokenize ${JSON.stringify(example.text)}`;
  check(label, model.tokenize(example.text), example.tokens);
  check(`${label} roundtrip`, model.detokenize(example.tokens, true), example.roundtrip);
}

for (const example of reference.chat_template) {
  const label = `chat template ${example.messages.length} message(s)`;
  check(`${label} text`, model.applyChatTemplate(example.messages), example.rendered);
  check(`${label} tokens`, model.chatTokens(example.messages), example.tokens);
}

for (const example of reference.generate) {
  const label = `generate ${JSON.stringify(example.prompt)}`;
  const result = model.generate(example.prompt, reference.options, undefined);
  check(`${label} text`, result.text, example.text);
  check(`${label} tokens`, result.tokens, example.tokens);
  check(`${label} prompt tokens`, result.prompt_tokens, example.prompt_tokens);
  check(`${label} end`, result.finished_by_eos, example.finished_by_eos);
}

for (const example of reference.chat) {
  const label = `chat ${JSON.stringify(example.messages.at(-1).content)}`;
  const result = model.chat(example.messages, reference.options, undefined);
  check(`${label} text`, result.text, example.text);
  check(`${label} tokens`, result.tokens, example.tokens);
  check(`${label} prompt tokens`, result.prompt_tokens, example.prompt_tokens);
}

for (const example of reference.logits) {
  const scores = model.logits(example.text);
  check(`logits ${JSON.stringify(example.text)} length`, scores.length, example.length);
  const top = [...scores.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)
    .map(([index, score]) => [index, Math.round(score * 1000) / 1000]);
  check(`logits ${JSON.stringify(example.text)} top 5`, top, example.top);
}

// The callback must see every token, and the parts must build the complete text.
const parts = [];
const streamed = model.generate(
  reference.generate[0].prompt,
  reference.options,
  (text, token) => {
    parts.push({ text, token });
  },
);
// The engine stops at the end token and does not send it to the callback, so the
// callback sees one piece less when the answer ended by itself.
const expectedPieces = streamed.finished_by_eos
  ? streamed.tokens.length - 1
  : streamed.tokens.length;
check("stream token count", parts.length, expectedPieces);
check(
  "stream token ids",
  parts.map((p) => p.token),
  Array.from(streamed.tokens).slice(0, expectedPieces),
);
if (!streamed.finished_by_eos) {
  check("stream text", parts.map((p) => p.text).join(""), streamed.text);
}
check("stream is the same as no stream", streamed.tokens, reference.generate[0].tokens);

if (failures.length > 0) {
  console.error(`WASM PARITY FAILURE: ${failures.length} of ${checks} checks disagree\n`);
  for (const failure of failures) {
    console.error(failure);
  }
  process.exit(1);
}

console.log(`WASM parity passed: ${checks} checks against the Python module`);
