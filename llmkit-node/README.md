# LLMKit Node.js

Node.js/TypeScript bindings for LLMKit - a unified LLM API client library.

## Installation

```bash
npm install llmkit
# or
pnpm add llmkit
# or
yarn add llmkit
```

## Quick Start

```typescript
import {
  JsLLMKitClient as LLMKitClient,
  JsMessage as Message,
  JsCompletionRequest as CompletionRequest,
} from 'llmkit';

// Create client from environment variables
const client = LLMKitClient.fromEnv();

// Make a completion request
const response = await client.complete(
  new CompletionRequest({
    model: 'anthropic/claude-sonnet-4-20250514',
    messages: [Message.user('Hello!')],
  })
);

console.log(response.textContent());

// Streaming
const stream = client.stream(request);
for await (const chunk of stream) {
  if (chunk.text) {
    process.stdout.write(chunk.text);
  }
}
```

## Features

- Unified API for 100+ LLM providers
- Full TypeScript support
- Streaming support
- Tool/function calling
- Extended thinking (reasoning)
- Prompt caching
- Structured output (JSON schema)

## License

MIT OR Apache-2.0
