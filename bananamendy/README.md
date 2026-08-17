# bananamendy

CLI and persistent OpenAI-compatible server for local CPU inference on the
**BananaMind-2** chat checkpoints, on top of the Rust core
[`bananamendr`](https://pypi.org/project/bananamendr/). No PyTorch at runtime.

```bash
uv pip install bananamendy

bananamendy pull nano                      # into the Hugging Face cache
bananamendy models                         # what is cached locally
bananamendy info                           # architecture facts
bananamendy chat --prompt "Why is the sky blue?"
bananamendy chat                           # REPL
bananamendy generate --prompt "Once upon a time"
bananamendy serve                          # OpenAI-compatible on 127.0.0.1:8377
```

Point any OpenAI client at it:

```bash
curl http://127.0.0.1:8377/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "nano", "messages": [{"role": "user", "content": "Hi"}], "stream": true}'
```

Implemented: `GET /v1/models`, `POST /v1/chat/completions`,
`POST /v1/completions` (both with SSE streaming), `GET /health`. Sampling
parameters left out of a request fall back to your config, so the server and the
CLI behave identically.

Configuration is TOML in the platformdirs location (`bananamendy init_config`
writes it; `bananamendy config` shows the effective values), overridable with
`BANANAMENDY_*` environment variables.

Weights live in the ordinary Hugging Face cache — `HF_HOME` / `HF_HUB_CACHE` are
respected, and a checkpoint you already have is not downloaded twice. Aliases
`nano`, `mini` and `pro` expand to the `BananaMind/BananaMind-2-*-Chat` repos;
any repo id or local directory works too.

Full documentation: <https://github.com/twardoch/bananamend>

## License

Apache-2.0
