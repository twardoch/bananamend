---
this_file: docs/server.md
title: The server
layout: default
nav_order: 5
permalink: /server/
---

# The server

`bananamendy serve` starts a server with the OpenAI interface. Any client that
speaks to OpenAI can then speak to a model on your computer.

```bash
bananamendy serve                        # http://127.0.0.1:8377
bananamendy serve --port 9000 --name mini
bananamendy serve --preload=False        # load the model at the first request
```

The server loads the model one time, and it keeps it. `--preload=False` moves the
load to the first request, which makes the start faster.

## What the server accepts

| Method and path | Function |
|:----------------|:---------|
| `GET /health` | The state of the server, and the names of the loaded models. |
| `GET /v1/models` | The models that you can ask for. |
| `POST /v1/chat/completions` | A conversation. The chat format is applied. |
| `POST /v1/completions` | A continuation of your text. |

Both `POST` paths accept `"stream": true`. The answer is then a sequence of
server-sent events, and the last event is `data: [DONE]`.

## Examples

```bash
curl http://127.0.0.1:8377/health
curl http://127.0.0.1:8377/v1/models

curl http://127.0.0.1:8377/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "nano", "messages": [{"role": "user", "content": "Name one ocean."}]}'

curl -N http://127.0.0.1:8377/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hi"}], "stream": true}'

curl http://127.0.0.1:8377/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_tokens": 8, "temperature": 0}'
```

With the OpenAI Python library:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8377/v1", api_key="not-needed")
answer = client.chat.completions.create(
    model="nano",
    messages=[{"role": "user", "content": "Name one ocean."}],
    max_tokens=32,
    temperature=0,
)
print(answer.choices[0].message.content)
```

The server asks for no key. Do not put it on a public address.

## Which parameters the server reads

| Field in the request | Effect |
|:---------------------|:-------|
| `model` | The alias, the repository name, or the path of a model. The default comes from your configuration. |
| `messages` | The conversation. `/v1/chat/completions` only. |
| `prompt` | The text to continue. `/v1/completions` only. |
| `max_tokens` or `max_completion_tokens` | How many tokens to write. |
| `temperature` | 0 always selects the most probable token. |
| `top_p`, `top_k` | Limit the tokens that the model can select. |
| `frequency_penalty` | The server changes this to the repetition penalty, which is the nearest control that the engine has. |
| `seed` | The start value for the random numbers. |
| `stream` | With `true`, the answer arrives token by token. |

**A parameter that you do not give comes from your configuration, and not from
the defaults of OpenAI.** The server and the command line then behave in the same
way. See `bananamendy config`.

## One request at a time

The server answers one request at a time.

The reason is the Python interpreter lock. A generation in the engine holds the
lock for the complete decode, so two generations in one process cannot run
together. A pool of workers would give no more speed, and it would use more
memory for a second copy of the model.

Streaming still works while a generation runs. The generation runs on a worker
thread, and its callback puts each piece of text into a queue. The response reads
the queue and sends the pieces.

If you need to answer many requests at the same time, start more than one server
on different ports, and put a load balancer in front of them. Each server holds
its own copy of the model.

## Errors

| Code | Cause |
|:-----|:------|
| 404 | The model is not in the cache, and the server cannot download it. |
| 400 | The engine refused the request. An example is a prompt that is longer than the context. |
| 422 | The request does not have the necessary fields. |

The body of an error holds a `detail` field with the text of the error.
