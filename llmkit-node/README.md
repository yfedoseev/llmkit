# LLMKit Node.js

**The production-grade LLM client for Node.js.** Native Rust performance with full TypeScript support.

[![npm](https://img.shields.io/npm/v/llmkit-node.svg)](https://www.npmjs.com/package/llmkit-node)
[![Node](https://img.shields.io/node/v/llmkit-node.svg)](https://www.npmjs.com/package/llmkit-node)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/yfedoseev/llmkit/blob/main/LICENSE)

## Why LLMKit?

- **Rust Core** — Native performance, memory safety, true concurrency
- **100+ Providers** — OpenAI, Anthropic, Google, AWS Bedrock, Azure, Groq, and more
- **11,000+ Models** — Built-in registry with pricing and capabilities
- **Prompt Caching** — Save up to 90% on API costs with native caching support
- **Extended Thinking** — Unified reasoning API across 5 providers
- **Production Ready** — No memory leaks, scales to thousands of concurrent requests

## Installation

```bash
npm install llmkit-node
# or
pnpm add llmkit-node
# or
yarn add llmkit-node
```

## Quick Start

```typescript
import { LLMKitClient, Message, CompletionRequest } from 'llmkit-node'

// Create client from environment variables
const client = LLMKitClient.fromEnv()

// Make a completion request
const response = await client.complete(
  new CompletionRequest('anthropic/claude-sonnet-4-20250514', [
    Message.user('Hello!')
  ])
)

console.log(response.textContent())
```

## Streaming

```typescript
const stream = client.stream(request)

for await (const chunk of stream) {
  if (chunk.text) {
    process.stdout.write(chunk.text)
  }
}
```

## Tool Calling

```typescript
import { ToolBuilder } from 'llmkit-node'

// Build tools with fluent API
const weatherTool = new ToolBuilder('get_weather')
  .description('Get current weather for a location')
  .stringParam('city', 'City name', { required: true })
  .enumParam('unit', 'Temperature unit', ['celsius', 'fahrenheit'])
  .build()

const request = new CompletionRequest(
  'anthropic/claude-sonnet-4-20250514',
  [Message.user("What's the weather in Tokyo?")]
).withTools([weatherTool])

const response = await client.complete(request)

for (const toolCall of response.toolCalls()) {
  console.log(`Call ${toolCall.name} with`, toolCall.arguments)
}
```

## Prompt Caching

Save up to 90% on repeated prompts:

```typescript
// Large system prompts are automatically cached
const request = new CompletionRequest(
  'anthropic/claude-sonnet-4-20250514',
  [
    Message.system(largeSystemPrompt), // Cached after first call
    Message.user('Question 1')
  ]
).withCache()

// Subsequent calls reuse the cached system prompt
const response = await client.complete(request)
console.log(`Cache savings: ${response.usage.cacheReadTokens} tokens`)
```

## Extended Thinking

Unified reasoning across Anthropic, OpenAI, Google, DeepSeek, and OpenRouter:

```typescript
const request = new CompletionRequest(
  'anthropic/claude-sonnet-4-20250514',
  [Message.user('Solve this step by step: ...')]
).withThinking({ budgetTokens: 10000 })

const response = await client.complete(request)
console.log('Reasoning:', response.thinkingContent())
console.log('Answer:', response.textContent())
```

## Model Registry

11,000+ models with pricing and capabilities — no API calls needed:

```typescript
import { getModelInfo, getModelsByProvider, getModelsWithCapability } from 'llmkit-node'

// Get model details instantly
const info = getModelInfo('anthropic/claude-sonnet-4-20250514')
console.log(`Context: ${info.contextWindow.toLocaleString()} tokens`)
console.log(`Input: $${info.inputPrice}/1M tokens`)
console.log(`Output: $${info.outputPrice}/1M tokens`)

// Find models by provider
const anthropicModels = getModelsByProvider('anthropic')

// Find models with specific capabilities
const visionModels = getModelsWithCapability({ vision: true })
```

## TypeScript Support

Full TypeScript definitions included:

```typescript
import type {
  CompletionRequest,
  CompletionResponse,
  Message,
  ToolDefinition,
  StreamChunk,
  ModelInfo,
} from 'llmkit-node'
```

## Features

| Feature | Status |
|---------|--------|
| Chat Completions | Supported |
| Streaming | Supported |
| Tool Calling | Supported |
| Structured Output | Supported |
| Extended Thinking | Supported |
| Prompt Caching | Supported |
| Vision/Images | Supported |
| Embeddings | Supported |
| Image Generation | Supported |
| Audio STT/TTS | Supported |
| Video Generation | Supported |

## Documentation

- [Getting Started](https://github.com/yfedoseev/llmkit/blob/main/docs/getting-started-nodejs.md)
- [Full Documentation](https://github.com/yfedoseev/llmkit/tree/main/docs)
- [Examples](https://github.com/yfedoseev/llmkit/tree/main/examples)

## License

MIT OR Apache-2.0
