import { Module } from '@nestjs/common';

import { NimService } from './nim.service';

@Module({
  providers: [NimService],
  exports: [NimService],
})
export class NimModule {}
