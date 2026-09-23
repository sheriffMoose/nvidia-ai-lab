/**
 * NestJS app: a thin, streaming-safe front door to a hosted NIM.
 *
 * Run: npm run start:dev   (port 8081, so the FastAPI twin can keep 8080)
 */
import { Logger, ValidationPipe } from '@nestjs/common';
import { ConfigService } from '@nestjs/config';
import { NestFactory } from '@nestjs/core';
import type { NestExpressApplication } from '@nestjs/platform-express';

import { AppModule } from './app.module';
import type { AppConfig } from './config/configuration';
import { PUBLIC_DIR } from './config/configuration';

async function bootstrap(): Promise<void> {
  const app = await NestFactory.create<NestExpressApplication>(AppModule, { bufferLogs: false });
  const cfg = app.get(ConfigService).getOrThrow<AppConfig>('app');

  app.useGlobalPipes(
    new ValidationPipe({ transform: true, whitelist: true, forbidNonWhitelisted: true }),
  );

  if (cfg.corsOrigins.length) {
    app.enableCors({
      origin: cfg.corsOrigins,
      methods: ['GET', 'POST'],
      allowedHeaders: ['Content-Type'],
    });
  }

  // Serves public/index.html at `/`. Note there is deliberately no compression
  // middleware here: gzip in front of SSE re-buffers the stream we just built.
  app.useStaticAssets(PUBLIC_DIR);

  await app.listen(cfg.port, '0.0.0.0');
  new Logger('bootstrap').log(`Listening on http://127.0.0.1:${cfg.port}`);
}

void bootstrap();
