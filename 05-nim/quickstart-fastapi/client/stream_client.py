"""Scratch client: hits our FastAPI route and prints tokens as they land.

    uv run python client/stream_client.py "Explain CUDA streams in two sentences"

The client measures its own TTFT. If it is far worse than the server's reported
`ttft_ms`, something between them is buffering.
"""

import argparse
import asyncio
import json
import sys
import time

import httpx


async def stream(api_url: str, prompt: str, model: str | None, max_tokens: int) -> int:
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if model:
        payload["model"] = model

    started = time.perf_counter()
    client_ttft_ms: float | None = None
    server: dict = {}

    async with httpx.AsyncClient(timeout=httpx.Timeout(90.0, connect=10.0)) as http:
        async with http.stream("POST", f"{api_url}/chat/stream", json=payload) as response:
            if response.status_code != 200:
                body = (await response.aread()).decode(errors="replace")
                print(f"HTTP {response.status_code}: {body}", file=sys.stderr)
                return 1

            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break

                event = json.loads(data)
                if event["type"] == "token":
                    if client_ttft_ms is None:
                        client_ttft_ms = (time.perf_counter() - started) * 1000
                    print(event["text"], end="", flush=True)
                elif event["type"] == "reasoning":
                    # Reasoning models think before they speak. Show a heartbeat so
                    # the wait reads as "working", not "hung".
                    print(".", end="", flush=True)
                elif event["type"] == "done":
                    server = event
                elif event["type"] == "error":
                    print(f"\n[stream error] {event['message']}", file=sys.stderr)
                    return 1

    wall_ms = (time.perf_counter() - started) * 1000
    print("\n" + "-" * 60)
    print(f"model             : {server.get('model', '?')}")
    print(f"TTFT (client)     : {client_ttft_ms:.0f} ms" if client_ttft_ms else "TTFT: no tokens")
    print(f"TTFT (server)     : {server.get('ttft_ms', '?')} ms")
    print(f"first byte (srv)  : {server.get('first_byte_ms', '?')} ms")
    print(f"total (client)    : {wall_ms:.0f} ms")
    print(f"chunks            : {server.get('chunks', '?')}")
    if server.get("reasoning_chunks"):
        print(f"reasoning chunks  : {server['reasoning_chunks']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Stream a completion through the quickstart API")
    parser.add_argument("prompt", nargs="?", default="Explain what a NIM is in two sentences.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default=None, help="override the server's default model")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    return asyncio.run(stream(args.api_url, args.prompt, args.model, args.max_tokens))


if __name__ == "__main__":
    raise SystemExit(main())
