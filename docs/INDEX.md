# ModelSuite Documentation

**Version:** 0.1.0
**Provider Coverage:** 70+ providers, 1,798+ models

---

## Getting Started

Choose your preferred language:

- **[Getting Started with Rust](getting-started-rust.md)** - Native Rust API
- **[Getting Started with Python](getting-started-python.md)** - Python bindings via PyO3
- **[Getting Started with Node.js](getting-started-nodejs.md)** - Node.js bindings via NAPI-RS

---

## Core Documentation

### Library Overview
- **[ModelSuite Overview](llmkit.md)** - Architecture, features, and design philosophy

### API Reference
- **[Model Registry](MODELS_REGISTRY.md)** - Complete model catalog and capabilities
- **[Provider Sources](PROVIDER_MODEL_SOURCES.md)** - Provider API documentation links

---

## Specialized APIs

### Multimodal Capabilities
- **[Image API](image-api.md)** - Image generation and vision models
- **[Audio API](audio-api.md)** - Speech-to-text and text-to-speech
- **[Video API](video-api.md)** - Video generation models
- **[Specialized API](specialized-api.md)** - Embeddings, rerankers, and domain models

### Provider Categories
- **[Additional Providers](additional_providers.md)** - Extended provider coverage
- **[Emerging Providers](emerging_specialized_providers.md)** - New and specialized providers

---

## Reference

### Technical Reference
- **[Domain Models](domain_models.md)** - Domain-specific model details
- **[Scientific Benchmarks](scientific_benchmarks.md)** - Model evaluation metrics

---

## Quick Links

### Rust
```rust
use modelsuite::{Client, Provider};

let client = Client::new();
let response = client.chat(Provider::OpenAI, "gpt-4o", messages).await?;
```

### Python
```python
from modelsuite import Client, Provider

client = Client()
response = await client.chat(Provider.OPENAI, "gpt-4o", messages)
```

### Node.js
```typescript
import { Client, Provider } from 'modelsuite';

const client = new Client();
const response = await client.chat(Provider.OpenAI, "gpt-4o", messages);
```

---

## Support

- **GitHub Issues:** [Report bugs and request features](https://github.com/yfedoseev/modelsuite/issues)
- **Documentation:** This index and linked guides

---

**License:** MIT OR Apache-2.0
