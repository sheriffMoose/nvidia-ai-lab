/** Chat routes: one streaming (SSE), one buffered — so you can feel the difference. */
import {
  Body,
  Controller,
  HttpException,
  Post,
  Req,
  Res,
  ServiceUnavailableException,
} from '@nestjs/common';
import type { Request, Response } from 'express';

import { NimService } from '../nim/nim.service';
import type { NimEvent } from '../nim/nim.types';
import { ChatRequestDto } from './dto/chat-request.dto';

// Without these, a proxy (nginx, a cloud LB) will happily buffer the whole
// response and hand the browser one blob — the exact failure this day is about.
const SSE_HEADERS: Record<string, string> = {
  'Content-Type': 'text/event-stream; charset=utf-8',
  'Cache-Control': 'no-cache, no-transform',
  Connection: 'keep-alive',
  'X-Accel-Buffering': 'no',
};

@Controller('chat')
export class ChatController {
  constructor(private readonly nim: NimService) {}

  /**
   * Server-sent events. Each `token` event is flushed as it arrives from the NIM.
   *
   * This takes the raw `@Res()` instead of Nest's `@Sse()` decorator on purpose:
   * `@Sse()` wants an Observable and owns the frame format, and we need the exact
   * `data: {...}` / `data: [DONE]` contract the FastAPI twin speaks. Taking `@Res()`
   * also opts this handler out of Nest's response pipeline — which is what keeps
   * interceptors and serializers from collecting the stream into one blob.
   */
  @Post('stream')
  async streamChat(
    @Body() payload: ChatRequestDto,
    @Req() request: Request,
    @Res() response: Response,
  ): Promise<void> {
    this.assertAvailable();

    response.writeHead(200, SSE_HEADERS);
    // Get the headers on the wire now, not when the first chunk shows up.
    response.flushHeaders();

    let clientGone = false;
    request.on('close', () => {
      clientGone = true;
    });

    const send = (event: NimEvent): void => {
      response.write(`data: ${JSON.stringify(event)}\n\n`);
    };

    // Flush something immediately so the client can timestamp the connection
    // separately from the model's first token.
    send({ type: 'open', model: payload.model ?? this.nim.defaultModel });

    try {
      for await (const event of this.nim.streamChat(payload.messages, {
        model: payload.model,
        temperature: payload.temperature,
        maxTokens: payload.max_tokens,
      })) {
        if (clientGone) break;
        send(event);
      }
    } catch (error) {
      // Headers are already out, so an error can only be reported in-band.
      send({ type: 'error', message: `stream failed: ${(error as Error).message}` });
    }

    response.write('data: [DONE]\n\n');
    response.end();
  }

  /** Same upstream call, collected before responding. Useful as the latency baseline. */
  @Post()
  async bufferedChat(@Body() payload: ChatRequestDto) {
    this.assertAvailable();

    const started = performance.now();
    const parts: string[] = [];
    let ttftMs: number | null = null;
    let chunks = 0;

    for await (const event of this.nim.streamChat(payload.messages, {
      model: payload.model,
      temperature: payload.temperature,
      maxTokens: payload.max_tokens,
    })) {
      if (event.type === 'token') {
        ttftMs ??= Math.round((performance.now() - started) * 10) / 10;
        parts.push(event.text);
      } else if (event.type === 'error') {
        throw new HttpException(event.message, event.status_code ?? 502);
      } else if (event.type === 'done') {
        chunks = event.chunks;
      }
    }

    return {
      model: payload.model ?? this.nim.defaultModel,
      content: parts.join(''),
      timings: {
        // The client measures its own TTFT to catch buffering in between.
        ttft_ms: ttftMs,
        total_ms: Math.round((performance.now() - started) * 10) / 10,
        chunks,
      },
    };
  }

  private assertAvailable(): void {
    if (!this.nim.available) {
      throw new ServiceUnavailableException('NIM client unavailable — check NVIDIA_API_KEY');
    }
  }
}
