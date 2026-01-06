# ModelSuite Node.js

Node.js/TypeScript bindings for ModelSuite - a unified LLM API client library.

## Installation

```bash
npm install modelsuite
# or
pnpm add modelsuite
# or
yarn add modelsuite
```

## Quick Start

```typescript
import {
  JsModelSuiteClient as ModelSuiteClient,
  JsMessage as Message,
  JsCompletionRequest as CompletionRequest,
} from 'modelsuite';

// Create client from environment variables
const client = ModelSuiteClient.fromEnv();

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

- Unified API for 70+ LLM providers
- Full TypeScript support
- Streaming support
- Tool/function calling
- Extended thinking (reasoning)
- Prompt caching
- Structured output (JSON schema)

## License

MIT OR Apache-2.0
