/**
 * Scratch client: hits our NestJS route and prints tokens as they land.
 *
 *     node client/stream-client.mjs "Explain CUDA streams in two sentences"
 *
 * The client measures its own TTFT. If it is far worse than the server's reported
 * `ttft_ms`, something between them is buffering.
 *
 * Same wire contract as the FastAPI twin, so this also works against port 8080:
 *     node client/stream-client.mjs --api-url http://127.0.0.1:8080 "hi"
 */
import { parseArgs } from 'node:util';

const { values, positionals } = parseArgs({
  options: {
    'api-url': { type: 'string', default: 'http://127.0.0.1:8081' },
    model: { type: 'string' },
    'max-tokens': { type: 'string', default: '256' },
  },
  allowPositionals: true,
});

const prompt = positionals.join(' ') || 'Explain what a NIM is in two sentences.';
const apiUrl = values['api-url'];

const started = performance.now();
let clientTtftMs = null;
let server = {};

const response = await fetch(`${apiUrl}/chat/stream`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    messages: [{ role: 'user', content: prompt }],
    max_tokens: Number(values['max-tokens']),
    ...(values.model ? { model: values.model } : {}),
  }),
});

if (!response.ok) {
  console.error(`HTTP ${response.status}: ${await response.text()}`);
  process.exit(1);
}

const decoder = new TextDecoder();
let buffer = '';

outer: for await (const chunk of response.body) {
  buffer += decoder.decode(chunk, { stream: true });

  // SSE frames are separated by a blank line; keep the trailing partial one.
  const frames = buffer.split('\n\n');
  buffer = frames.pop();

  for (const frame of frames) {
    const line = frame.split('\n').find((l) => l.startsWith('data: '));
    if (!line) continue;
    const data = line.slice(6);
    if (data === '[DONE]') break outer;

    const event = JSON.parse(data);
    if (event.type === 'token') {
      clientTtftMs ??= performance.now() - started;
      process.stdout.write(event.text);
    } else if (event.type === 'reasoning') {
      // Reasoning models think before they speak. Show a heartbeat so the wait
      // reads as "working", not "hung".
      process.stdout.write('.');
    } else if (event.type === 'done') {
      server = event;
    } else if (event.type === 'error') {
      console.error(`\n[stream error] ${event.message}`);
      process.exit(1);
    }
  }
}

const wallMs = performance.now() - started;
console.log('\n' + '-'.repeat(60));
console.log(`model             : ${server.model ?? '?'}`);
console.log(
  clientTtftMs === null
    ? 'TTFT: no tokens'
    : `TTFT (client)     : ${clientTtftMs.toFixed(0)} ms`,
);
console.log(`TTFT (server)     : ${server.ttft_ms ?? '?'} ms`);
console.log(`first byte (srv)  : ${server.first_byte_ms ?? '?'} ms`);
console.log(`total (client)    : ${wallMs.toFixed(0)} ms`);
console.log(`chunks            : ${server.chunks ?? '?'}`);
if (server.reasoning_chunks) {
  console.log(`reasoning chunks  : ${server.reasoning_chunks}`);
}
