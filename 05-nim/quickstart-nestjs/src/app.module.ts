import { Module } from '@nestjs/common';
import { ConfigModule } from '@nestjs/config';

import configuration, { ENV_FILES } from './config/configuration';
import { ChatModule } from './chat/chat.module';
import { HealthModule } from './health/health.module';
import { NimModule } from './nim/nim.module';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true,
      envFilePath: ENV_FILES,
      load: [configuration],
      cache: true,
    }),
    NimModule,
    ChatModule,
    HealthModule,
  ],
})
export class AppModule {}
