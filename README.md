# LLMKit

**The production-grade LLM client.** One API for 100+ providers. Pure Rust core with native bindings.

**11,000+ models** · **100+ providers** · **Rust | Python | Node.js**

```
                         ┌──────────────┐
                         │  Rust Core   │
                         └──────┬───────┘
          ┌──────────┬─────────┼─────────┬──────────┐
          ▼          ▼         ▼         ▼          ▼
      ┌───────┐  ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
      │Python │  │ Node  │ │ WASM  │ │  Go   │ │  ...  │
      │  ✅   │  │  ✅   │ │ Soon  │ │ Soon  │ │       │
      └───────┘  └───────┘ └───────┘ └───────┘ └───────┘
```

[![CI](https://github.com/yfedoseev/llmkit/actions/workflows/ci.yml/badge.svg)](https://github.com/yfedoseev/llmkit/actions/workflows/ci.yml)
[![Crates.io](https://img.shields.io/crates/v/llmkit.svg)](https://crates.io/crates/llmkit)
[![PyPI](https://img.shields.io/pypi/v/llmkit-python.svg)](https://pypi.org/project/llmkit-python/)
[![npm](https://img.shields.io/npm/v/llmkit-node.svg)](https://www.npmjs.com/package/llmkit-node)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

[Documentation](docs/) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

## Why LLMKit?

### Built for Production

LLMKit is written in **pure Rust** — no Python runtime, no garbage collector, no memory leaks. Deploy with confidence knowing your LLM infrastructure won't degrade over time or crash under load.

- **Memory Safety** — Rust's ownership model eliminates memory leaks by design
- **True Concurrency** — No GIL. Handle thousands of concurrent streams efficiently
- **Minimal Footprint** — Native binary, not a 150MB Python package
- **Run Forever** — No worker restarts, no memory bloat, no surprises

### Features That Actually Work

- **Prompt Caching** — Native support for Anthropic, OpenAI, Google, DeepSeek. Save up to 90% on API costs
- **Extended Thinking** — Unified API for reasoning across 5 providers (Anthropic, OpenAI, Google, DeepSeek, OpenRouter)
- **Streaming** — Zero-copy streaming with automatic request deduplication
- **11,000+ Model Registry** — Pricing, context limits, and capabilities baked in. No external API calls

### Production Features

| Feature | Description |
|---------|-------------|
| **Smart Router** | ML-based provider selection optimizing for latency, cost, or reliability |
| **Circuit Breaker** | Automatic failure detection and recovery with anomaly detection |
| **Rate Limiting** | Lock-free, hierarchical rate limiting at scale |
| **Cost Tracking** | Multi-tenant metering with cache-aware pricing |
| **Guardrails** | PII detection, secret scanning, prompt injection prevention |
| **Observability** | OpenTelemetry integration for tracing and metrics |

## Quick Start

### Rust
```rust
use llmkit::{LLMKitClient, Message, CompletionRequest};

let client = LLMKitClient::from_env()?;
let response = client.complete(
    CompletionRequest::new("anthropic/claude-sonnet-4-20250514", vec![Message::user("Hello!")])
).await?;
println!("{}", response.text_content());
```

### Python
```python
from llmkit import LLMKitClient, Message, CompletionRequest

client = LLMKitClient.from_env()
response = client.complete(CompletionRequest(
    model="openai/gpt-4o",
    messages=[Message.user("Hello!")]
))
print(response.text_content())
```

### Node.js
```typescript
import { LLMKitClient, Message, CompletionRequest } from 'llmkit-node'

const client = LLMKitClient.fromEnv()
const response = await client.complete(
  new CompletionRequest('anthropic/claude-sonnet-4-20250514', [Message.user('Hello!')])
)
console.log(response.textContent())
```

## Installation

### Rust
```toml
[dependencies]
llmkit = { version = "0.1", features = ["anthropic", "openai"] }
```

### Python
```bash
pip install llmkit-python
```

### Node.js
```bash
npm install llmkit-node
```

## Features

| Chat | Media | Specialized |
|------|-------|-------------|
| Streaming | Image Generation | Embeddings |
| Tool Calling | Vision/Images | Token Counting |
| Structured Output | Audio STT/TTS | Batch Processing |
| Extended Thinking | Video Generation | Model Registry |
| Prompt Caching | | 11,000+ Models |

## Providers

| Category | Providers |
|----------|-----------|
| **Core** | Anthropic, OpenAI, Azure OpenAI |
| **Cloud** | AWS Bedrock, Google Vertex AI, Google AI |
| **Fast Inference** | Groq, Mistral, Cerebras, SambaNova, Fireworks, DeepSeek |
| **Enterprise** | Cohere, AI21 |
| **Hosted** | Together, Perplexity, DeepInfra, OpenRouter |
| **Local** | Ollama, LM Studio, vLLM |
| **Audio** | Deepgram, ElevenLabs |
| **Video** | Runware |

See [PROVIDERS.md](PROVIDERS.md) for the full list with environment variables.

## Examples

### Streaming
```rust
let mut stream = client.complete_stream(request).await?;
while let Some(chunk) = stream.next().await {
    if let Some(text) = chunk?.text() { print!("{}", text); }
}
```

### Tool Calling
```python
from llmkit import ToolBuilder

tool = ToolBuilder("get_weather") \
    .description("Get current weather") \
    .string_param("city", "City name", required=True) \
    .build()

request = CompletionRequest(model, messages).with_tools([tool])
```

### Prompt Caching
```python
# Cache large system prompts - save up to 90% on repeated calls
request = CompletionRequest(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[Message.system(large_prompt), Message.user("Question")]
).with_cache()
```

### Extended Thinking
```typescript
// Unified reasoning API across providers
const request = new CompletionRequest('anthropic/claude-sonnet-4-20250514', messages)
  .withThinking({ budgetTokens: 10000 })

const response = await client.complete(request)
console.log(response.thinkingContent()) // See the reasoning process
console.log(response.textContent())     // Final answer
```

### Model Registry
```python
from llmkit import get_model_info, get_models_by_provider

# Get model details - no API calls, instant lookup
info = get_model_info("anthropic/claude-sonnet-4-20250514")
print(f"Context: {info.context_window}, Price: ${info.input_price}/1M tokens")

# Find models by provider
anthropic_models = get_models_by_provider("anthropic")
```

For more examples, see [examples/](examples/).

## Documentation

- [Getting Started (Rust)](docs/getting-started-rust.md)
- [Getting Started (Python)](docs/getting-started-python.md)
- [Getting Started (Node.js)](docs/getting-started-nodejs.md)
- [Model Registry](docs/MODELS_REGISTRY.md) — 11,000+ models with pricing

## Building from Source

```bash
git clone https://github.com/yfedoseev/llmkit
cd llmkit
cargo build --release
cargo test

# Python bindings
cd llmkit-python && maturin develop

# Node.js bindings
cd llmkit-node && pnpm install && pnpm build
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).

---

**Built with Rust** · Production Ready · [GitHub](https://github.com/yfedoseev/llmkit)
