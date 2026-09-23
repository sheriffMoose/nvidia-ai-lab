# 05-nim / quickstart-nest — streaming a hosted NIM through NestJS

Week 1 · Day 3 deliverable, NestJS edition. Same scaffold as [`../quickstart`](../quickstart)
(FastAPI), same wire contract, different framework — so the interesting differences are
framework-shaped, not model-shaped.

A NestJS route in front of a **hosted** NIM (`build.nvidia.com`), streaming tokens end to
end, with TTFT visible in the client. No GPU, no pod, no driver. Est. spend: **$0**.

Runs on **8081** so it can sit next to the FastAPI twin on 8080 and be compared directly.

## Layout

```
quickstart-nest/
├── src/
│   ├── config/configuration.ts   settings; key from env, never from code
│   ├── nim/nim.service.ts        OpenAI SDK pointed at the NIM + timing instrumentation
│   ├── nim/nim.types.ts          event union shared by service and controller
│   ├── chat/dto/chat-request.dto.ts  class-validator request shapes
│   ├── chat/chat.controller.ts   POST /chat/stream (SSE) and POST /chat (buffered baseline)
│   ├── health/health.controller.ts   GET /healthz, GET /models
│   └── main.ts                   bootstrap, CORS, validation pipe, static assets
├── client/stream-client.mjs      CLI client — prints tokens as they land, reports TTFT
├── public/index.html             browser client — same, in ~80 lines of JS
├── .env.example
└── package.json
```

## Setup

```bash
cd 05-nim/quickstart-nest
npm install
```

The key is picked up from, in order of precedence:

1. the process environment (`NVIDIA_API_KEY`, or `NGC_API_KEY` as fallback)
2. `05-nim/quickstart-nest/.env` (gitignored — `cp .env.example .env`)
3. `00-environment/.env` (the shared repo env)

`@nestjs/config` keeps the **first** definition it sees, which is why `envFilePath` is
ordered `[local, repo]` — the opposite of pydantic-settings, where the *last* file wins.
Easy thing to get backwards when porting.

## Run

```bash
npm run start:dev      # watch mode, port 8081
# or: npm run build && npm run start:prod
```

Check it booted with a key, and which models you can actually reach:

```bash
curl http://127.0.0.1:8081/healthz
curl http://127.0.0.1:8081/models
```

Then stream, from the CLI:

```bash
node client/stream-client.mjs "Explain CUDA streams in two sentences"
```

…or open <http://127.0.0.1:8081/> for the browser version.

For the buffered baseline (same upstream call, held until complete):

```bash
curl -s -X POST http://127.0.0.1:8081/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```

Same contract on both servers, so either client can be pointed at either one:

```bash
node client/stream-client.mjs --api-url http://127.0.0.1:8080 "hi"   # hits FastAPI
```

## Notes that matter later

**The endpoint is OpenAI-shaped.** The only NVIDIA-specific thing in `nim.service.ts` is
`baseURL` + the key. Self-hosting in Week 8 should be a config change
(`NIM_BASE_URL=http://localhost:8000/v1`), not a rewrite — that portability is the main
thing NIM sells, and it is just as true from the Node SDK as from the Python one.

**Nest's streaming traps are different from FastAPI's.** Three, in the order they bite:

1. **`@Sse()` is the wrong tool here.** It wants an `Observable<MessageEvent>` and owns the
   frame format, so it can't emit the `data: [DONE]` sentinel the Python twin's clients
   expect. Taking the raw `@Res()` gives the exact contract — and, importantly, opts the
   handler out of Nest's response pipeline, so no interceptor or serializer can collect the
   stream into one object before it goes out.
2. **`flushHeaders()` is not optional.** Without it Express holds the response head until
   the first chunk, and the client can't timestamp the connection separately from the
   model's first token.
3. **`compression` middleware silently re-buffers SSE.** It isn't installed here on
   purpose; if you add it later, exclude `text/event-stream` or you undo this whole day.

Add `X-Accel-Buffering: no` on top for the proxy in front of you (nginx, a cloud LB), which
will otherwise buffer the response regardless of what the app does.

**An async generator is the cleanest bridge.** `NimService.streamChat()` is an
`AsyncGenerator<NimEvent>` — the same shape as the Python `AsyncIterator[dict]` — so the
controller is a `for await` loop plus `res.write()`, and nothing in between holds state.
Swapping in an RxJS `Observable` would buy backpressure semantics we don't need and make
the "where did it buffer?" question harder to answer.

**Two TTFTs, on purpose.** `ttft_ms` from the server is measured at the OpenAI SDK
boundary; the client's is measured at the socket. Server fast + client slow = we buffered.
Both slow = the model (or the queue in front of it) is slow.

**The model catalogue lies to you in two different ways.** `GET /models` returns ~80 ids,
most of which 404 for this key — listed ≠ entitled. Ids from last year's docs
(`meta/llama-3.3-70b-instruct`) answer **410 Gone**, retired.

| Status | Meaning                               | Fix                                 |
| ------ | ------------------------------------- | ----------------------------------- |
| 410    | Model retired from the hosted gateway | Pick a current id                   |
| 404    | Listed, but your key can't invoke it  | Different id, or check entitlements |

**Reasoning models split "first byte" from "first token".** The nemotron family streams
`reasoning_content` deltas before any visible `content`, so a naive TTFT measured on
`delta.content` counts the whole thinking phase as dead air. Note that `reasoning_content`
is *not* in the OpenAI SDK's `delta` type — it's a NIM extension, read through a narrow
cast in `nim.service.ts`. Both clients show a heartbeat during that window rather than
looking hung.

### Hosted vs self-hosted — axes to revisit in Week 8

| Axis           | Hosted (`build.nvidia.com`)    | Self-hosted NIM                                   |
| -------------- | ------------------------------ | ------------------------------------------------- |
| Cost           | Free for dev; per-token beyond | GPU-hour + NVIDIA AI Enterprise licensing          |
| Data residency | Prompts leave your perimeter   | Stays in your VPC — the deciding axis for fincrime |
| Latency        | Public internet + shared queue | Yours; TTFT tunable via model profile              |
| Model profiles | Whatever NVIDIA published      | Pick quantisation / batching / TP for your GPU     |
| Ops burden     | None                           | Drivers, containers, capacity, upgrades            |

## Verified run (2026-09-23)

Node 22.19.0 · NestJS 11 · `openai` 6.x.

```
model             : meta/llama-3.2-11b-vision-instruct
TTFT (client)     : 3954 ms      # includes connect; server saw 3335 ms
TTFT (server)     : 3334.8 ms
total (client)    : 7386 ms
chunks            : 73           # 73 incremental deltas, not one blob
```

Reasoning model, same route:

```
model             : nvidia/nemotron-3-super-120b-a12b
TTFT (client)     : 931 ms
TTFT (server)     : 884.9 ms
first byte (srv)  : 619.4 ms     # 265 ms of thinking before the first visible token
chunks            : 4
reasoning chunks  : 9
```

Frames arriving incrementally, timestamped as they hit the shell (`curl -sN`):

```
15:26:45.224 data: {"type":"open","model":"nvidia/nemotron-3-super-120b-a12b"}
15:26:45.603 data: {"type":"reasoning","text":"User"}
15:26:45.680 data: {"type":"reasoning","text":" asks: \"What is 12"}
15:26:45.763 data: {"type":"reasoning","text":"*7? One"}
```

Buffered baseline for the same route: `ttft_ms 12508.3, total_ms 12838.8, chunks 7` — the
client waits the full `total_ms` before seeing anything, which is the UX this day exists to
avoid.

## Done when

- [x] A request round-trips through the API to the model
- [x] Streaming works
- [x] Tokens arrive incrementally in the client (73 chunks, timestamps above)
- [x] The NGC key lives in env/config, not in code or the frontend
- [x] Hosted-vs-self-host tradeoffs noted (table above)

## Reflection

- **What did I build or measure today?**
- **What surprised me or broke, and why?**
- **Production / financial-crime implication?**
