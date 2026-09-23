# 05-nim / quickstart — streaming a hosted NIM through FastAPI

Week 1 · Day 3 deliverable. A FastAPI route in front of a **hosted** NIM
(`build.nvidia.com`), streaming tokens end to end, with TTFT visible in the client.

No GPU, no pod, no driver. Est. spend: **$0**.

## Layout

```
quickstart/
├── app/
│   ├── config.py        settings; key from env, never from code
│   ├── nim_client.py    OpenAI SDK pointed at the NIM + timing instrumentation
│   ├── schemas.py       request/response models
│   ├── routes/chat.py   POST /chat/stream (SSE) and POST /chat (buffered baseline)
│   └── main.py          app wiring, lifespan, /healthz, /models
├── client/stream_client.py   CLI client — prints tokens as they land, reports TTFT
├── static/index.html         browser client — same, in ~80 lines of JS
├── .env.example
└── pyproject.toml
```

## Setup

```bash
cd 05-nim/quickstart
uv sync
```

The key is picked up from, in order of precedence:

1. the process environment (`NVIDIA_API_KEY`, or `NGC_API_KEY` as fallback)
2. `05-nim/quickstart/.env` (gitignored — `cp .env.example .env`)
3. `00-environment/.env` (the shared repo env)

## Run

```bash
uv run uvicorn app.main:app --reload --port 8080
```

Check it booted with a key, and which models you can actually reach:

```bash
curl http://127.0.0.1:8080/healthz
curl http://127.0.0.1:8080/models
```

Then stream, from the CLI:

```bash
uv run python client/stream_client.py "Explain CUDA streams in two sentences"
```

…or open <http://127.0.0.1:8080/> for the browser version.

For the buffered baseline (same upstream call, held until complete):

```bash
curl -s -X POST http://127.0.0.1:8080/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

## Notes that matter later

**The endpoint is OpenAI-shaped.** The only NVIDIA-specific thing in `nim_client.py`
is `base_url` + the key. Self-hosting in Week 8 should be a config change
(`NIM_BASE_URL=http://localhost:8000/v1`), not a rewrite — that portability is the
main thing NIM sells.

**Buffering is the default failure.** FastAPI's `StreamingResponse` streams, but
anything in the path will undo it: a proxy without `X-Accel-Buffering: no`, a
compression middleware, a client that calls `.json()` instead of iterating. Both
clients print their own TTFT next to the server's so a gap between them is visible
rather than invisible. That number is what Week 5 makes you care about.

**Two TTFTs, on purpose.** `ttft_ms` from the server is measured at the OpenAI SDK
boundary; the client's is measured at the socket. Server fast + client slow = we
buffered. Both slow = the model (or the queue in front of it) is slow.

**The model catalogue lies to you in two different ways.** `GET /v1/models` returned
82 ids, but most of them 404 for this key — listed ≠ entitled. And ids from last
year's docs (`meta/llama-3.3-70b-instruct`) answer **410 Gone**, retired. So:

| Status | Meaning                                | Fix                                     |
| ------ | -------------------------------------- | --------------------------------------- |
| 410    | Model retired from the hosted gateway  | Pick a current id                       |
| 404    | Listed, but your key can't invoke it   | Different id, or check entitlements     |

Probe before trusting a list — `GET /models` gives the ids, a 4-token chat call
tells you which ones actually answer. This is an argument *for* self-hosting that
has nothing to do with latency: on your own NIM, the model doesn't move.

**Reasoning models split "first byte" from "first token".** The nemotron family
streams `reasoning_content` deltas before any visible `content`, so a naive TTFT
measured on `delta.content` counts the whole thinking phase as dead air. Measured
here on `nvidia/nemotron-3-super-120b-a12b`: first byte **453 ms**, first *visible*
token **928 ms**, 17 reasoning chunks before 2 content chunks. The client shows a
heartbeat during that window rather than looking hung. Budget both numbers separately
in Week 5.

### Hosted vs self-hosted — axes to revisit in Week 8

| Axis            | Hosted (`build.nvidia.com`)              | Self-hosted NIM                                   |
| --------------- | ---------------------------------------- | ------------------------------------------------- |
| Cost            | Free for dev; per-token beyond           | GPU-hour + NVIDIA AI Enterprise licensing          |
| Data residency  | Prompts leave your perimeter             | Stays in your VPC — the deciding axis for fincrime |
| Latency         | Public internet + shared queue           | Yours; TTFT tunable via model profile              |
| Model profiles  | Whatever NVIDIA published                | Pick quantisation / batching / TP for your GPU     |
| Ops burden      | None                                     | Drivers, containers, capacity, upgrades            |

## Verified run (2026-09-23)

Models reachable with this key: `meta/llama-3.2-11b-vision-instruct`,
`nvidia/nemotron-3-super-120b-a12b`, `nvidia/nemotron-3.5-lightning-30b-a3b`.

```
model             : meta/llama-3.2-11b-vision-instruct
TTFT (client)     : 4252 ms      # includes connect; server saw 3610 ms
TTFT (server)     : 3610.2 ms
total (client)    : 10061 ms
chunks            : 80           # 80 incremental deltas, not one blob
```

Buffered baseline for the same route: `ttft_ms 346.6, total_ms 553.7, chunks 7` —
the client waits the full `total_ms` before seeing anything, which is the UX this
day exists to avoid.

## Done when

- [x] A request round-trips through the API to the model
- [x] Streaming works
- [x] Tokens arrive incrementally in the client (80 chunks, not one buffered blob)
- [x] The NGC key lives in env/config, not in code or the frontend
- [x] Hosted-vs-self-host tradeoffs noted (table above)

## Reflection

- **What did I build or measure today?**
- **What surprised me or broke, and why?**
- **Production / financial-crime implication?**
