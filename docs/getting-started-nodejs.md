# Getting Started with ModelSuite (Node.js/TypeScript)

ModelSuite is a unified LLM API client that provides a single interface to 48+ LLM providers and 120+ models including Anthropic, OpenAI, Azure, AWS Bedrock, Google Vertex AI, and many more.

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
    JsLlmKitClient as LLMKitClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
} from 'llmkit'

// Create client from environment variables
const client = LLMKitClient.fromEnv()

// Make a completion request
const response = await client.complete(
    CompletionRequest.create('claude-sonnet-4-20250514', [
        Message.user('What is the capital of France?')
    ])
)

console.log(response.textContent())
```

## Type Aliases

For cleaner code, create type aliases:

```typescript
// types.ts
export {
    JsLlmKitClient as LLMKitClient,
    JsMessage as Message,
    JsContentBlock as ContentBlock,
    JsCompletionRequest as CompletionRequest,
    JsCompletionResponse as CompletionResponse,
    JsToolBuilder as ToolBuilder,
    JsToolDefinition as ToolDefinition,
    JsEmbeddingRequest as EmbeddingRequest,
    JsEmbeddingResponse as EmbeddingResponse,
    JsTokenCountRequest as TokenCountRequest,
    JsBatchRequest as BatchRequest,
    JsBatchJob as BatchJob,
    JsProvider as Provider,
    JsStreamChunk as StreamChunk,
} from 'llmkit'
```

## Environment Setup

Set one or more provider API keys:

```bash
# Core providers
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Cloud providers
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=gpt-4
export AWS_REGION=us-east-1  # For Bedrock
export GOOGLE_CLOUD_PROJECT=your-project  # For Vertex AI
export VERTEX_LOCATION=us-central1

# Fast inference providers
export GROQ_API_KEY=...
export MISTRAL_API_KEY=...
export CEREBRAS_API_KEY=...
export DEEPSEEK_API_KEY=...

# Other providers
export COHERE_API_KEY=...
export OPENROUTER_API_KEY=...
# ... and 30+ more
```

LLMKit automatically detects which providers are configured from environment variables.

## Explicit Configuration

Instead of environment variables, you can configure providers explicitly:

```typescript
const client = new LLMKitClient({
    providers: {
        anthropic: { apiKey: 'sk-ant-...' },
        openai: { apiKey: 'sk-...' },
        azure: {
            apiKey: '...',
            endpoint: 'https://your-resource.openai.azure.com',
            deployment: 'gpt-4'
        },
        bedrock: { region: 'us-east-1' },
    },
    defaultProvider: 'anthropic'
})
```

## Streaming

### Async Iterator (Recommended)

```typescript
const request = CompletionRequest
    .create('claude-sonnet-4-20250514', [
        Message.user('Write a haiku about programming')
    ])
    .withStreaming()

const stream = await client.stream(request)

let chunk
while ((chunk = await stream.next()) !== null) {
    if (chunk.text) {
        process.stdout.write(chunk.text)
    }
    if (chunk.isDone) break
}
console.log()
```

### Callback-based

```typescript
client.completeStream(request, (chunk, error) => {
    if (error) {
        console.error('Stream error:', error)
        return
    }
    if (chunk?.text) {
        process.stdout.write(chunk.text)
    }
    if (chunk?.isDone) {
        console.log('\nDone!')
    }
})
```

## Tool Calling (Function Calling)

Define and use tools:

```typescript
import { JsToolBuilder as ToolBuilder, JsContentBlock as ContentBlock } from 'llmkit'

// Define a tool
const weatherTool = new ToolBuilder('get_weather')
    .description('Get current weather for a city')
    .stringParam('city', 'City name', true)
    .enumParam('unit', 'Temperature unit', ['celsius', 'fahrenheit'])
    .build()

// Make request with tools
const request = CompletionRequest
    .create('claude-sonnet-4-20250514', [
        Message.user("What's the weather in Paris?")
    ])
    .withTools([weatherTool])

const response = await client.complete(request)

// Check if the model wants to use a tool
if (response.hasToolUse()) {
    for (const toolUse of response.toolUses()) {
        const info = toolUse.asToolUse()
        if (info) {
            console.log('Tool:', info.name)
            console.log('Input:', info.input)

            // Execute the tool and send results back
            const result = ContentBlock.toolResult(
                info.id,
                '{"temperature": 22, "unit": "celsius"}',
                false
            )

            // Continue the conversation
            const finalResponse = await client.complete(
                CompletionRequest.create('claude-sonnet-4-20250514', [
                    Message.user("What's the weather in Paris?"),
                    Message.assistantWithContent(response.content),
                    Message.toolResults([result])
                ])
            )
            console.log(finalResponse.textContent())
        }
    }
}
```

## Structured Output

Get JSON responses with schema validation:

```typescript
const schema = {
    type: 'object',
    properties: {
        name: { type: 'string' },
        age: { type: 'integer' },
        city: { type: 'string' }
    },
    required: ['name', 'age', 'city']
}

const request = CompletionRequest
    .create('claude-sonnet-4-20250514', [
        Message.user("Generate a fake person's info")
    ])
    .withJsonSchema('person', schema)

const response = await client.complete(request)
const data = JSON.parse(response.textContent())
console.log(data)  // { name: 'Alice', age: 30, city: 'Paris' }
```

## Extended Thinking

Enable reasoning mode for complex tasks:

```typescript
const request = CompletionRequest
    .create('claude-sonnet-4-20250514', [
        Message.user('Solve this puzzle: ...')
    ])
    .withThinking(5000)  // 5000 token budget

const response = await client.complete(request)

// Get thinking content (reasoning process)
const thinking = response.thinkingContent()
if (thinking) {
    console.log('Thinking:', thinking)
}

console.log('Answer:', response.textContent())
```

## Vision / Image Analysis

Analyze images:

```typescript
import { readFileSync } from 'fs'

// From file
const imageData = readFileSync('image.png').toString('base64')
const message = Message.userWithContent([
    ContentBlock.text("What's in this image?"),
    ContentBlock.image('image/png', imageData)
])

// Or from URL
const messageUrl = Message.userWithContent([
    ContentBlock.text('Describe this image:'),
    ContentBlock.imageUrl('https://example.com/image.png')
])

const response = await client.complete(
    CompletionRequest.create('claude-sonnet-4-20250514', [message])
)
console.log(response.textContent())
```

## Embeddings

Generate text embeddings:

```typescript
import { JsEmbeddingRequest as EmbeddingRequest } from 'llmkit'

// Single text
const request = new EmbeddingRequest('text-embedding-3-small', 'Hello, world!')
const response = await client.embed(request)

console.log('Dimensions:', response.dimensionCount)
console.log('First 5 values:', response.values()?.slice(0, 5))

// Batch embeddings
const batchRequest = EmbeddingRequest.batch(
    'text-embedding-3-small',
    ['Hello', 'World', 'How are you?']
)
const batchResponse = await client.embed(batchRequest)

for (const embedding of batchResponse.embeddings) {
    console.log(`Index ${embedding.index}: ${embedding.dimensionCount} dimensions`)
}

// Compute similarity
const emb1 = batchResponse.embeddings[0]
const emb2 = batchResponse.embeddings[1]
const similarity = emb1.cosineSimilarity(emb2)
console.log('Similarity:', similarity)
```

## Token Counting

Estimate token usage before making requests:

```typescript
import { JsTokenCountRequest as TokenCountRequest } from 'llmkit'

const countRequest = TokenCountRequest
    .create('claude-sonnet-4-20250514', [
        Message.user('Hello, how are you?')
    ])
    .withSystem('You are a helpful assistant')

const result = await client.countTokens(countRequest)
console.log('Input tokens:', result.inputTokens)
```

## Batch Processing

Process multiple requests asynchronously:

```typescript
import { JsBatchRequest as BatchRequest } from 'llmkit'

// Create batch requests
const batchRequests = [
    BatchRequest.create('request-1',
        CompletionRequest.create('claude-sonnet-4-20250514', [
            Message.user('What is 2+2?')
        ])
    ),
    BatchRequest.create('request-2',
        CompletionRequest.create('claude-sonnet-4-20250514', [
            Message.user('What is 3+3?')
        ])
    ),
]

// Submit batch
const batchJob = await client.createBatch(batchRequests)
console.log('Batch ID:', batchJob.id)
console.log('Status:', batchJob.status)

// Check status
const updatedJob = await client.getBatch('anthropic', batchJob.id)
console.log('Status:', updatedJob.status)
console.log('Succeeded:', updatedJob.requestCounts.succeeded)

// Get results when complete
if (updatedJob.isComplete()) {
    const results = await client.getBatchResults('anthropic', batchJob.id)
    for (const result of results) {
        if (result.isSuccess()) {
            console.log(`${result.customId}: ${result.response?.textContent()}`)
        } else {
            console.log(`${result.customId}: Error - ${result.error?.message}`)
        }
    }
}
```

## Model Registry

Query available models:

```typescript
import {
    getModelInfo,
    getAllModels,
    getModelsByProvider,
    getAvailableModels,
    getCheapestModel,
    JsProvider as Provider,
} from 'llmkit'

// Get info about a specific model
const info = getModelInfo('claude-sonnet-4-20250514')
if (info) {
    console.log('Name:', info.name)
    console.log('Price:', `$${info.pricing.inputPer1M}/1M input tokens`)
    console.log('Max context:', info.capabilities.maxContext)
    console.log('Supports vision:', info.capabilities.vision)
    console.log('Supports tools:', info.capabilities.tools)
}

// Get all Anthropic models
const anthropicModels = getModelsByProvider(Provider.Anthropic)
for (const model of anthropicModels) {
    console.log(`${model.name}: ${model.description}`)
}

// Get available models (with configured API keys)
const available = getAvailableModels()
console.log(`${available.length} models available`)

// Find cheapest model with specific requirements
const cheapest = getCheapestModel(100000, true, true)  // 100k context, vision, tools
if (cheapest) {
    console.log('Cheapest:', cheapest.name)
}
```

## Error Handling

Handle errors gracefully:

```typescript
try {
    const response = await client.complete(request)
} catch (error) {
    if (error instanceof Error) {
        const message = error.message

        if (message.includes('Authentication')) {
            console.log('Invalid API key')
        } else if (message.includes('RateLimit')) {
            console.log('Rate limited')
        } else if (message.includes('ContextLength')) {
            console.log('Input too long')
        } else if (message.includes('InvalidRequest')) {
            console.log('Invalid request:', message)
        } else if (message.includes('ProviderNotFound')) {
            console.log('Provider not configured')
        } else if (message.includes('Timeout')) {
            console.log('Request timed out')
        } else {
            console.log('Error:', message)
        }
    }
}
```

## Multiple Providers

Use different providers for different tasks:

```typescript
// List available providers
console.log(client.providers())  // ['anthropic', 'openai', ...]

// Use a specific provider
const response = await client.completeWithProvider(
    'openai',
    CompletionRequest.create('gpt-4o', [
        Message.user('Hello!')
    ])
)

// Check default provider
console.log(client.defaultProvider)
```

## Prompt Caching

Cache frequently used prompts (Anthropic):

```typescript
// Enable caching on system prompt
const request = CompletionRequest
    .create('claude-sonnet-4-20250514', [
        Message.user('Summarize this document: ...')
    ])
    .withSystem('You are a document summarizer.')
    .withSystemCaching()  // 5-minute cache

// Extended caching (1 hour)
const extendedRequest = request.withSystemCachingExtended()

// Check cache usage in response
const response = await client.complete(request)
console.log('Cache creation:', response.usage.cacheCreationInputTokens)
console.log('Cache read:', response.usage.cacheReadInputTokens)
```

## TypeScript Configuration

For best TypeScript experience:

```json
// tsconfig.json
{
  "compilerOptions": {
    "strict": true,
    "esModuleInterop": true,
    "moduleResolution": "node",
    "target": "ES2020"
  }
}
```

## Next Steps

- Check out the [examples](../examples/nodejs/) for more complete code samples
- See the [API Reference](./api-reference-nodejs.md) for detailed documentation
- View the [Provider Guide](./providers.md) for provider-specific configuration
