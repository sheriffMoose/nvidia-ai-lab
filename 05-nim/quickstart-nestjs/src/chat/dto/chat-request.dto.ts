/**
 * Request shapes for the quickstart API.
 *
 * Field names stay snake_case (`max_tokens`) to match the FastAPI twin exactly —
 * same wire contract means `client/stream_client.py` and the browser page work
 * against either server without a change.
 */
import { Type } from 'class-transformer';
import {
  ArrayMinSize,
  IsArray,
  IsIn,
  IsInt,
  IsNumber,
  IsOptional,
  IsString,
  Max,
  Min,
  ValidateNested,
} from 'class-validator';

export class MessageDto {
  @IsIn(['system', 'user', 'assistant'])
  role!: 'system' | 'user' | 'assistant';

  @IsString()
  content!: string;
}

export class ChatRequestDto {
  @IsArray()
  @ArrayMinSize(1)
  @ValidateNested({ each: true })
  @Type(() => MessageDto)
  messages!: MessageDto[];

  @IsOptional()
  @IsString()
  model?: string;

  @IsOptional()
  @IsNumber()
  @Min(0)
  @Max(2)
  temperature: number = 0.2;

  @IsOptional()
  @IsInt()
  @Min(1)
  @Max(8192)
  max_tokens: number = 512;
}
