import { Module } from '@nestjs/common';

import { NimModule } from '../nim/nim.module';
import { ChatController } from './chat.controller';

@Module({
  imports: [NimModule],
  controllers: [ChatController],
})
export class ChatModule {}
