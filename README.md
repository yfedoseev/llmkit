# LLMKit

**Unified LLM API Client for Rust, Python, and Node.js**

One interface for 100+ LLM providers. Rust core with bindings for every language.

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
[![PyPI](https://img.shields.io/pypi/v/llmkit.svg)](https://pypi.org/project/llmkit/)
[![npm](https://img.shields.io/npm/v/llmkit.svg)](https://www.npmjs.com/package/llmkit)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

[📖 Documentation](docs/) | [📝 Changelog](CHANGELOG.md) | [🤝 Contributing](CONTRIBUTING.md) | [🔒 Security](SECURITY.md)

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
import { JsLLMKitClient as LLMKitClient, JsMessage as Message, JsCompletionRequest as CompletionRequest } from 'llmkit'

const client = LLMKitClient.fromEnv()
const response = await client.complete(CompletionRequest.create('groq/llama-3.3-70b-versatile', [Message.user('Hello!')]))
console.log(response.textContent())
```

## Why LLMKit?

- 🌍 **100+ Providers** - OpenAI, Anthropic, Google, AWS Bedrock, Azure, Groq, Mistral, and more
- 🔄 **Unified API** - Same interface for all providers with `provider/model` format
- ⚡ **Streaming** - First-class async streaming support
- 🛠️ **Tool Calling** - Abstract tool definitions with builder pattern
- 🧠 **Extended Thinking** - Reasoning mode across 4 providers (OpenAI, Anthropic, Google, DeepSeek)
- 🦀 **Pure Rust** - Memory-safe, high performance core

## Features

| Chat | Media | Specialized |
|------|-------|-------------|
| Streaming | Image Generation | Embeddings |
| Tool Calling | Vision/Images | Token Counting |
| Structured Output | Audio STT/TTS | Batch Processing |
| Extended Thinking | Video Generation | Model Registry |
| Prompt Caching | | 11,000+ Models |

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

See [PROVIDERS.md](PROVIDERS.md) for full list with environment variables.

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

tool = ToolBuilder("get_weather").description("Get weather").string_param("city", "City name").build()
request = CompletionRequest(...).with_tools([tool])
```

### Extended Thinking
```typescript
const request = CompletionRequest.create('deepseek/deepseek-reasoner', messages).withThinking(5000)
const response = await client.complete(request)
console.log(response.thinkingContent()) // Reasoning process
```

For more examples, see [examples/](examples/).

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

## Documentation

- **[Getting Started (Rust)](docs/getting-started-rust.md)**
- **[Getting Started (Python)](docs/getting-started-python.md)**
- **[Getting Started (Node.js)](docs/getting-started-nodejs.md)**
- **[Model Registry](docs/MODELS_REGISTRY.md)** - 11,000+ models with pricing

```bash
cargo doc --open
```

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
cargo build && cargo test && cargo fmt && cargo clippy -- -D warnings
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.

---

**Built with** 🦀 Rust | **Status**: Production Ready | **v0.1.0** | 100+ Providers
