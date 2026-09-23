/** Boot diagnostics: did we get a key, and which models can this key actually reach? */
import { Controller, Get, HttpException, ServiceUnavailableException } from '@nestjs/common';
import OpenAI from 'openai';

import { NimService } from '../nim/nim.service';

@Controller()
export class HealthController {
  constructor(private readonly nim: NimService) {}

  @Get('healthz')
  healthz() {
    return {
      ok: this.nim.available,
      model: this.nim.defaultModel,
      base_url: this.nim.baseUrl,
      key_loaded: this.nim.keyLoaded,
      ...(this.nim.unavailableReason ? { detail: this.nim.unavailableReason } : {}),
    };
  }

  /** Live model ids for this key. Start here when a request comes back 410. */
  @Get('models')
  async models() {
    if (!this.nim.available) {
      throw new ServiceUnavailableException('NIM client unavailable — check NVIDIA_API_KEY');
    }
    try {
      return { models: await this.nim.listModels() };
    } catch (error) {
      if (error instanceof OpenAI.APIError) {
        throw new HttpException(`Could not list models: ${error.message}`, error.status ?? 502);
      }
      throw new HttpException(`Could not list models: ${(error as Error).name}`, 502);
    }
  }
}
