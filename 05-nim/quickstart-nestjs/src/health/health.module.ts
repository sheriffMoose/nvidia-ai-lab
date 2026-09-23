import { Module } from '@nestjs/common';

import { NimModule } from '../nim/nim.module';
import { HealthController } from './health.controller';

@Module({
  imports: [NimModule],
  controllers: [HealthController],
})
export class HealthModule {}
