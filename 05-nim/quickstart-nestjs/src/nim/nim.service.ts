/**
 * Thin wrapper over the NIM's OpenAI-compatible endpoint.
 *
 * The only NVIDIA-specific part is `baseURL` + the NGC key; everything else is the
 * OpenAI SDK. That is the whole point of NIM, and the reason Week 8's self-hosted
 * swap should be a one-line config change.
 */
import { Injectable, Logger } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import OpenAI from 'openai';

import type { AppConfig } from '../config/configuration';
import { LOCAL_ENV, REPO_ENV } from '../config/configuration';
import type { ChatMessage, NimEvent, StreamOptions } from './nim.types';

const round1 = (ms: number): number => Math.round(ms * 10) / 10;

@Injectable()
export class NimService {
  private readonly logger = new Logger(NimService.name);
  private readonly cfg: AppConfig;
  private readonly client: OpenAI | null;
  /** Why the client is missing, so /healthz can explain instead of the app crash-looping. */
  readonly unavailableReason: string | null;

  constructor(config: ConfigService) {
    this.cfg = config.getOrThrow<AppConfig>('app');
    const key = this.cfg.nvidiaApiKey ?? this.cfg.ngcApiKey;

    if (!key) {
      this.client = null;
      this.unavailableReason =
        'No NVIDIA_API_KEY (or NGC_API_KEY) found. Set it in the environment, ' +
        `in ${LOCAL_ENV}, or in ${REPO_ENV}.`;
      this.logger.error(this.unavailableReason);
      return;
    }

    this.client = new OpenAI({
      apiKey: key,
      baseURL: this.cfg.nimBaseUrl,
      timeout: this.cfg.requestTimeoutMs,
    });
    this.unavailableReason = null;
    this.logger.log(`NIM client ready — ${this.cfg.nimModel} @ ${this.cfg.nimBaseUrl}`);
  }

  get available(): boolean {
    return this.client !== null;
  }

  get defaultModel(): string {
    return this.cfg.nimModel;
  }

  get baseUrl(): string {
    return this.cfg.nimBaseUrl;
  }

  get keyLoaded(): boolean {
    return Boolean(this.cfg.nvidiaApiKey ?? this.cfg.ngcApiKey);
  }

  /** What this key can actually reach. The hosted catalogue changes under you. */
  async listModels(): Promise<string[]> {
    const client = this.requireClient();
    const page = await client.models.list();
    return page.data.map((model) => model.id).sort();
  }

  /**
   * Yield `{type: ...}` events: `token`/`reasoning` per delta, then one `done` or `error`.
   *
   * Timings are measured here, upstream of our own SSE layer, so comparing them
   * with the client's numbers tells you whether *we* are the ones buffering.
   *
   * Reasoning models (the nemotron family) stream `reasoning_content` deltas first
   * and only then visible `content`. Those are emitted as `reasoning` events so
   * "first byte" and "first *visible* token" stay separate numbers.
   */
  async *streamChat(
    messages: ChatMessage[],
    { model, temperature = 0.2, maxTokens = 512 }: StreamOptions = {},
  ): AsyncGenerator<NimEvent> {
    const client = this.requireClient();
    const started = performance.now();
    const resolvedModel = model ?? this.cfg.nimModel;

    let firstDeltaAt: number | null = null;
    let firstTokenAt: number | null = null;
    let chunks = 0;
    let reasoningChunks = 0;

    try {
      const stream = await client.chat.completions.create({
        model: resolvedModel,
        messages,
        temperature,
        max_tokens: maxTokens,
        stream: true,
      });

      for await (const chunk of stream) {
        const delta = chunk.choices?.[0]?.delta as
          | { content?: string | null; reasoning_content?: string | null }
          | undefined;
          console.log('####', chunk, delta);
          if (!delta) continue;

        const text = delta.content;
        const thought = delta.reasoning_content;
        if (!text && !thought) continue;

        firstDeltaAt ??= performance.now();

        if (thought) {
          reasoningChunks += 1;
          yield { type: 'reasoning', text: thought };
        }
        if (text) {
          firstTokenAt ??= performance.now();
          chunks += 1;
          yield { type: 'token', text };
        }
      }
    } catch (error) {
      // 410 usually means the model was retired — GET /models for live ids.
      if (error instanceof OpenAI.APIError) {
        const status = error.status ?? 502;
        const detail = String(error.message ?? '').trim().slice(0, 300);
        this.logger.warn(`NIM returned ${status} for model ${resolvedModel}: ${detail}`);
        yield {
          type: 'error',
          message:
            `NIM upstream error ${status} for '${resolvedModel}'` + (detail ? `: ${detail}` : ''),
          status_code: status,
        };
        return;
      }
      this.logger.error(`NIM request failed: ${String(error)}`);
      yield {
        type: 'error',
        message: `NIM request failed: ${(error as Error)?.name ?? 'Error'}`,
      };
      return;
    }

    yield {
      type: 'done',
      model: resolvedModel,
      chunks,
      reasoning_chunks: reasoningChunks,
      // First visible token vs. first anything on the wire. They differ on
      // reasoning models, and only the first one is what a user perceives.
      ttft_ms: firstTokenAt === null ? null : round1(firstTokenAt - started),
      first_byte_ms: firstDeltaAt === null ? null : round1(firstDeltaAt - started),
      total_ms: round1(performance.now() - started),
    };
  }

  private requireClient(): OpenAI {
    if (!this.client) {
      throw new NimUnavailableError(this.unavailableReason ?? 'NIM client unavailable');
    }
    return this.client;
  }
}

/** Raised when the service booted without a usable key. Mapped to 503 by the controllers. */
export class NimUnavailableError extends Error {}
