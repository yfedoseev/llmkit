# LLMKit

**Unified LLM API library** - One interface for 52+ LLM providers and specialized APIs.

Rust core with Python and Node.js/TypeScript bindings.

## Features

- **52+ Providers & Specialized APIs**: Anthropic, OpenAI, Azure, Bedrock, Vertex AI (+ 5 partner models), Groq, Mistral, xAI, Meta Llama, DataRobot, Stability AI, AWS SageMaker, Snowflake Cortex, Exa semantic search, and more
- **Unified Types**: Same message format works everywhere
- **Streaming**: First-class async streaming support
- **Tool Calling**: Abstract tool definitions with builder pattern
- **Extended Thinking**: Reasoning mode for complex tasks
- **Prompt Caching**: Cache system prompts for cost savings
- **Structured Output**: JSON schema enforcement
- **Vision/Images**: Image analysis support
- **Embeddings**: Text embedding generation
- **Batch Processing**: Async batch API
- **Token Counting**: Estimate costs before requests
- **Model Registry**: 100+ models with pricing and capabilities

## Installation

### Python

```bash
pip install llmkit
```

### Node.js/TypeScript

```bash
npm install llmkit
```

### Rust

```toml
[dependencies]
llmkit = { version = "0.1", features = ["anthropic", "openai"] }
```

## Quick Start

### Python

```python
from llmkit import LLMKitClient, Message, CompletionRequest

client = LLMKitClient.from_env()
response = client.complete(
    CompletionRequest(
        model="anthropic/claude-sonnet-4-20250514",  # provider/model format
        messages=[Message.user("Hello!")]
    )
)
print(response.text_content())
```

### Node.js/TypeScript

```typescript
import { JsLlmKitClient as LLMKitClient, JsMessage as Message, JsCompletionRequest as CompletionRequest } from 'llmkit'

const client = LLMKitClient.fromEnv()
const response = await client.complete(
    CompletionRequest.create('openai/gpt-4o', [Message.user('Hello!')])
)
console.log(response.textContent())
```

### Rust

```rust
use llmkit::{LLMKitClient, Message, CompletionRequest};

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    let client = LLMKitClient::from_env()?;

    let response = client.complete(
        CompletionRequest::new("groq/llama-3.3-70b-versatile", vec![Message::user("Hello!")])
    ).await?;

    println!("{}", response.text_content());
    Ok(())
}
```

## Model Format

LLMKit uses a unified `"provider/model"` format for explicit provider routing:

```
anthropic/claude-sonnet-4-20250514
openai/gpt-4o
groq/llama-3.3-70b-versatile
mistral/mistral-large-latest
```

This format is self-documenting and eliminates ambiguity. The provider prefix is **required** for all models.

## Streaming

### Python

```python
for chunk in client.complete_stream(request.with_streaming()):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

### Node.js

```typescript
const stream = await client.stream(request.withStreaming())
let chunk
while ((chunk = await stream.next()) !== null) {
    if (chunk.text) process.stdout.write(chunk.text)
    if (chunk.isDone) break
}
```

### Rust

```rust
use futures::StreamExt;

let mut stream = client.complete_stream(request).await?;
while let Some(chunk) = stream.next().await {
    if let Some(text) = chunk?.text() {
        print!("{}", text);
    }
}
```

## Tool Calling

### Python

```python
from llmkit import ToolBuilder

weather_tool = ToolBuilder("get_weather") \
    .description("Get current weather") \
    .string_param("city", "City name") \
    .build()

request = CompletionRequest(...).with_tools([weather_tool])
```

### Node.js

```typescript
const weatherTool = new ToolBuilder('get_weather')
    .description('Get current weather')
    .stringParam('city', 'City name')
    .build()

const request = CompletionRequest.create(...).withTools([weatherTool])
```

## Providers

LLMKit supports 49+ LLM providers:

| Category | Providers |
|----------|-----------|
| **Core** | Anthropic, OpenAI, Azure OpenAI |
| **Cloud** | AWS Bedrock, Google Vertex AI (Google, Anthropic, DeepSeek, Meta Llama, Mistral, AI21), Google AI (Gemini) |
| **Fast Inference** | Groq, Mistral, Cerebras, SambaNova, Fireworks, DeepSeek |
| **Enterprise** | Cohere, AI21 |
| **OpenAI-Compatible** | xAI (Grok), Meta Llama, Lambda Labs, Friendli, Volcengine |
| **Hosted** | Together, Perplexity, Anyscale, DeepInfra, Novita, Hyperbolic |
| **Platforms** | HuggingFace, Replicate, Baseten, RunPod |
| **Cloud ML** | Cloudflare, WatsonX, Databricks, DataRobot |
| **Local** | Ollama, LM Studio, vLLM, TGI, Llamafile |
| **Regional** | YandexGPT, GigaChat, Clova, Maritaca |
| **Specialized** | Stability AI, Voyage, Jina, Deepgram, ElevenLabs, Fal |

### Environment Variables

Each provider is auto-detected from environment variables:

```bash
# Core providers
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Azure OpenAI
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=gpt-4

# AWS Bedrock
export AWS_REGION=us-east-1

# Google
export GOOGLE_API_KEY=...  # For Gemini
export GOOGLE_CLOUD_PROJECT=...  # For Vertex AI
export VERTEX_LOCATION=us-central1

# Fast inference
export GROQ_API_KEY=...
export MISTRAL_API_KEY=...
export CEREBRAS_API_KEY=...
export DEEPSEEK_API_KEY=...

# And 30+ more...
```

## Rust Features

Enable only the providers you need:

```toml
[dependencies]
# Just Anthropic
llmkit = { version = "0.1", features = ["anthropic"] }

# Multiple providers
llmkit = { version = "0.1", features = ["anthropic", "openai", "groq"] }

# All providers
llmkit = { version = "0.1", features = ["all-providers"] }
```

| Feature | Description |
|---------|-------------|
| `anthropic` | Anthropic Claude (default) |
| `openai` | OpenAI GPT (default) |
| `azure` | Azure OpenAI |
| `bedrock` | AWS Bedrock |
| `vertex` | Google Vertex AI |
| `google` | Google AI (Gemini) |
| `groq` | Groq |
| `mistral` | Mistral AI |
| `cohere` | Cohere |
| `ollama` | Local Ollama |
| `all-providers` | Enable all |

## Advanced Features

### Extended Thinking

```python
request = CompletionRequest(...).with_thinking(budget_tokens=5000)
response = client.complete(request)
print(response.thinking_content())  # Reasoning process
print(response.text_content())       # Final answer
```

### Structured Output

```python
schema = {"type": "object", "properties": {...}}
request = CompletionRequest(...).with_json_schema("name", schema)
```

### Embeddings

```python
from llmkit import EmbeddingRequest

response = client.embed(EmbeddingRequest("openai/text-embedding-3-small", "Hello"))
print(response.values())  # [0.123, -0.456, ...]
```

### Token Counting

```python
from llmkit import TokenCountRequest

result = client.count_tokens(TokenCountRequest(model, messages))
print(result.input_tokens)
```

### Batch Processing

```python
from llmkit import BatchRequest

batch = client.create_batch([BatchRequest("id-1", request1), ...])
results = client.get_batch_results("anthropic", batch.id)
```

## Documentation

- [Getting Started (Python)](docs/getting-started-python.md)
- [Getting Started (Node.js)](docs/getting-started-nodejs.md)
- [Getting Started (Rust)](docs/getting-started-rust.md)
- [Examples](examples/)

## Examples

See the [`examples/`](examples/) directory:

- `examples/python/` - Python examples
- `examples/nodejs/` - Node.js/TypeScript examples
- `examples/` - Rust examples (run with `cargo run --example name`)

## License

MIT OR Apache-2.0
