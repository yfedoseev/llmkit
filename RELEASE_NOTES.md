# LLMKit v0.1.0 Release Notes

**Release Date:** January 3, 2026

## Overview

LLMKit v0.1.0 is the initial release of a production-grade LLM client library. Written in pure Rust with native bindings for Python and Node.js, LLMKit provides a unified interface to 100+ LLM providers and 11,000+ models.

## Highlights

- **100+ Providers** — Single API for Anthropic, OpenAI, Google, AWS Bedrock, Azure, and many more
- **11,000+ Models** — Built-in registry with pricing, capabilities, and benchmarks
- **Extended Thinking** — Unified reasoning API across 4 providers
- **Prompt Caching** — Save up to 90% on API costs
- **Production Ready** — No memory leaks, no GIL limitations, runs forever

## Key Features

### Unified Provider Interface

```rust
// Same API for any provider
let response = client.complete(
    CompletionRequest::new("anthropic/claude-sonnet-4-20250514", messages)
).await?;

let response = client.complete(
    CompletionRequest::new("openai/gpt-4o", messages)
).await?;
```

### Extended Thinking

Unified reasoning across OpenAI, Anthropic, Google, and DeepSeek:

```python
request = CompletionRequest(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[Message.user("Solve step by step...")]
).with_thinking(budget_tokens=10000)

response = client.complete(request)
print(response.thinking_content())  # See reasoning
print(response.text_content())       # Final answer
```

### Prompt Caching

Cache large system prompts for significant cost savings:

```typescript
const request = new CompletionRequest('anthropic/claude-sonnet-4-20250514', [
  Message.system(largeSystemPrompt),  // Cached after first call
  Message.user('Question')
]).withCache()
```

### Model Registry

Instant lookups for 11,000+ models — no API calls needed:

```python
info = get_model_info("anthropic/claude-sonnet-4-20250514")
print(f"Context: {info.context_window:,} tokens")
print(f"Price: ${info.input_price}/1M input tokens")
```

## Supported Providers

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

See [PROVIDERS.md](PROVIDERS.md) for the full list.

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

## Documentation

- [Getting Started (Rust)](docs/getting-started-rust.md)
- [Getting Started (Python)](docs/getting-started-python.md)
- [Getting Started (Node.js)](docs/getting-started-nodejs.md)
- [Model Registry](docs/MODELS_REGISTRY.md)
- [Examples](examples/)

## What's Next

Version 0.2.0 will include:
- Provider pooling and load balancing
- Automatic failover between providers
- Cost metering and budget controls
- Guardrails integration

## License

MIT OR Apache-2.0
