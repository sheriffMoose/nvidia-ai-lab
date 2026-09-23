/** Wire events. Identical shape to the FastAPI twin, so both share the clients. */

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export type NimEvent =
  | { type: 'open'; model: string }
  | { type: 'token'; text: string }
  | { type: 'reasoning'; text: string }
  | {
      type: 'done';
      model: string;
      chunks: number;
      reasoning_chunks: number;
      /** First *visible* token. Null if the model never produced one. */
      ttft_ms: number | null;
      /** First anything on the wire — differs on reasoning models. */
      first_byte_ms: number | null;
      total_ms: number;
    }
  | { type: 'error'; message: string; status_code?: number };

export interface StreamOptions {
  model?: string;
  temperature?: number;
  maxTokens?: number;
}
