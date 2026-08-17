---
this_file: docs/demo.md
title: Demonstration
layout: default
nav_order: 2
permalink: /demo/
---

# The model in your browser

This page runs the real engine. It is a WebAssembly build of the same Rust crate
that the command line program uses. Greedy decoding gives the same tokens as the
program on your computer, and a test in the repository proves it.

{: .warning }
> The page downloads the checkpoint that you choose from Hugging Face. The
> smallest is 6 MB and the largest is 556 MB. Your browser keeps the download in
> its cache, so a second visit to the same checkpoint is fast. Start with
> **Nano · 8 bits**, which is 11 MB.

{: .note }
> The engine has one thread here, and it runs on the thread of the page. Because
> of this the page cannot draw while the model writes an answer. Keep
> **maximum new tokens** small for a quick answer.

<!-- daisyUI comes from a CDN as plain CSS. The components below need no build
     step, and the theme of this site keeps control of the page. -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/daisyui@5/daisyui.css">
<link rel="stylesheet" href="{{ '/assets/css/demo.css' | relative_url }}">

<div id="demo" class="bananamend-demo" data-theme="dark">

  <fieldset class="fieldset bg-base-200 border-base-300 rounded-box border p-4">
    <legend class="fieldset-legend">Choose a checkpoint</legend>

    <div class="demo-row">
      <label class="demo-label" for="model">Checkpoint</label>
      <select id="model" class="select select-bordered select-sm demo-wide"></select>
    </div>

    <div class="demo-row">
      <label class="demo-label" for="custom">Or any repository on Hugging Face</label>
      <input id="custom" class="input input-bordered input-sm demo-wide" type="text"
             placeholder="owner/name — leave empty to use the list above" spellcheck="false">
    </div>

    <p id="modelNote" class="demo-note"></p>

    <div class="demo-row">
      <button id="load" type="button" class="btn btn-primary">Download and start</button>
      <span id="status" role="status" class="badge badge-ghost badge-lg demo-status">
        Select a checkpoint and use the button.
      </span>
    </div>

    <div class="demo-row" id="progressBox" hidden>
      <progress id="progress" class="progress progress-primary" max="100" value="0"></progress>
      <span id="progressText" class="demo-mono"></span>
    </div>
  </fieldset>

  <div class="stats stats-horizontal demo-stats" id="info" hidden>
    <div class="stat"><div class="stat-title">Loaded</div><div class="stat-value demo-stat-value" id="loadedName">—</div></div>
    <div class="stat"><div class="stat-title">Model</div><div class="stat-value demo-stat-value" id="modelType">—</div></div>
    <div class="stat"><div class="stat-title">Layers × hidden</div><div class="stat-value demo-stat-value" id="layers">—</div></div>
    <div class="stat"><div class="stat-title">Weights in memory</div><div class="stat-value demo-stat-value" id="weights">—</div></div>
    <div class="stat"><div class="stat-title">Form</div><div class="stat-value demo-stat-value" id="form">—</div></div>
    <div class="stat"><div class="stat-title">Context</div><div class="stat-value demo-stat-value" id="context">—</div></div>
    <div class="stat"><div class="stat-title">Engine</div><div class="stat-value demo-stat-value" id="version">—</div></div>
  </div>

  <fieldset id="controls" class="fieldset bg-base-200 border-base-300 rounded-box border p-4" disabled>
    <legend class="fieldset-legend">Ask the model</legend>

    <div class="demo-row">
      <label class="demo-label" for="mode">Mode</label>
      <select id="mode" class="select select-bordered select-sm">
        <option value="chat">chat — the chat format is applied</option>
        <option value="generate">generate — the text continues</option>
      </select>
      <label class="demo-label" for="example">Example</label>
      <select id="example" class="select select-bordered select-sm">
        <option value="0">Name one ocean.</option>
        <option value="1">Why is the sky blue?</option>
        <option value="2">Write three rules for a safe kitchen.</option>
        <option value="3">The capital of France is</option>
      </select>
    </div>

    <label class="demo-label" for="prompt">Your text</label>
    <textarea id="prompt" class="textarea textarea-bordered w-full" rows="3" spellcheck="false"></textarea>

    <div class="demo-row">
      <label class="demo-label" for="maxTokens">Maximum new tokens</label>
      <input id="maxTokens" type="number" class="input input-bordered input-sm demo-number" min="1" max="512" value="48">
      <label class="demo-label" for="temperature">Temperature</label>
      <input id="temperature" type="number" class="input input-bordered input-sm demo-number" min="0" max="2" step="0.1" value="0">
      <label class="demo-label" for="topK">Top-k</label>
      <input id="topK" type="number" class="input input-bordered input-sm demo-number" min="0" max="200" value="0">
      <label class="demo-label" for="seed">Seed</label>
      <input id="seed" type="number" class="input input-bordered input-sm demo-number" min="0" max="99999" value="0">
    </div>

    <div class="demo-row">
      <button id="run" type="button" class="btn btn-primary btn-sm">Write an answer</button>
      <button id="tokenize" type="button" class="btn btn-outline btn-sm">Show the tokens</button>
      <button id="clear" type="button" class="btn btn-ghost btn-sm">Clear</button>
    </div>
  </fieldset>

  <pre id="output" class="mockup-code demo-output" aria-live="polite"></pre>

  <div class="stats stats-horizontal demo-stats" id="stats" hidden>
    <div class="stat"><div class="stat-title">Prompt tokens</div><div class="stat-value demo-stat-value" id="promptTokens">—</div></div>
    <div class="stat"><div class="stat-title">New tokens</div><div class="stat-value demo-stat-value" id="newTokens">—</div></div>
    <div class="stat"><div class="stat-title">Prefill</div><div class="stat-value demo-stat-value" id="prefill">—</div></div>
    <div class="stat"><div class="stat-title">Decode</div><div class="stat-value demo-stat-value" id="decode">—</div></div>
    <div class="stat"><div class="stat-title">Speed</div><div class="stat-value demo-stat-value" id="speed">—</div></div>
    <div class="stat"><div class="stat-title">End token</div><div class="stat-value demo-stat-value" id="eos">—</div></div>
  </div>
</div>

<script type="module" src="{{ '/assets/js/demo.js' | relative_url }}"></script>

## What the page does

1. It downloads four files from the repository that you choose: `config.json`,
   `tokenizer.json`, `tokenizer_config.json` and `model.safetensors`. The third
   one is optional.
2. It gives the four parts to `Model.fromParts`, which builds the engine.
3. It sends your text through the same code as the command line program.

Nothing goes to a server of this site. The only network requests are the
downloads from Hugging Face.

## The checkpoints in the list

The list holds the three original checkpoints and the quantized copies of them.
A quantized copy is smaller both to download and to hold in memory, because the
engine keeps the codes and rebuilds each value inside the multiplication. The
table under [Small checkpoints](../quantization/) gives the measured quality of
each one.

The two `ternary everywhere` entries do not work. They are in the list because the
measurement is the point: a model of this size cannot carry ternary weights in
every matrix, and the page lets you see what that looks like.

**Your own repository works as well.** Write `owner/name` in the field under the
list. The repository needs `config.json`, `tokenizer.json` and
`model.safetensors`, and the architecture must be one that this engine knows.

{: .note }
> A browser gives the module 4 GB of memory, and the engine needs approximately
> three times the size of the file while it reads a checkpoint. The 556 MB Pro
> checkpoint of 32-bit floats is therefore near the limit; its 8-bit copy of
> 150 MB is not.

## Temperature 0 is greedy

Temperature 0 always selects the most probable token. The answer is then the same
on each attempt, and it is the same as the answer on your computer. A temperature
above 0 uses the random numbers, and the seed then controls the result.

## Do the same work on your computer

```bash
uv pip install bananamendy
bananamendy pull nano
bananamendy chat --prompt "Name one ocean." --temperature 0
```

Your computer has more threads, so it is faster. It can also use the larger
models: `bananamendy pull mini` and `bananamendy pull pro`.
