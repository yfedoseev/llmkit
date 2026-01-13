# Changelog

All notable changes to LLMKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Cache Pricing**: Added missing cache pricing for 8 models (Gemini 3, DeepSeek Reasoner, Claude Haiku 4.5)
- **Capability Flags**: Removed incorrect structured output flag from 26 models that don't support strict JSON schema
- **Benchmark Scores**: Corrected benchmark scores for OpenAI o3, DeepSeek R1, Claude Opus/Sonnet 4.5
- **Provider Consistency**: Verified cross-provider model consistency for DeepSeek R1 and Gemini 3 variants

## [0.1.0] - 2026-01-03

Initial release of LLMKit.

### Added

#### Core Features
- Unified LLM API interface with **100+ providers**
- **11,000+ model registry** with pricing, capabilities, and benchmarks
- Streaming completions with async iterators
- Tool/function calling with fluent builder pattern (`ToolBuilder`)
- Structured output with JSON schema enforcement
- Vision/image input support (base64 and URLs)
- Comprehensive error types for all failure modes
- Feature flags for provider selection

#### Extended Thinking
- Unified `ThinkingConfig` API across 4 providers:
  - OpenAI (o3, o1-pro)
  - Anthropic (Claude with extended thinking)
  - Google Vertex AI (Gemini 2.0 Deep Thinking)
  - DeepSeek (DeepSeek-R1)

#### Prompt Caching
- Native support for Anthropic, OpenAI, Google, and DeepSeek
- 5-minute and 1-hour TTL options
- Up to 90% cost savings on repeated prompts

#### Regional Providers
- Mistral EU with GDPR-compliant endpoint
- Maritaca AI for Brazilian Portuguese
- Regional configuration via environment variables

#### Audio & Voice
- Deepgram v3 with Nova-3 models
- ElevenLabs with configurable latency modes
- Speech-to-text and text-to-speech support

#### Video Generation
- Runware aggregator supporting 5+ video models
- Runway Gen-4.5, Kling 2.0, Pika 1.0, and more

#### Embeddings & Specialized
- Voyage AI embeddings
- Jina AI embeddings and reranking
- Token counting API
- Batch processing API

#### Providers
- **Core**: Anthropic, OpenAI, Azure OpenAI
- **Cloud**: AWS Bedrock, Google Vertex AI, Google AI
- **Fast Inference**: Groq, Mistral, Cerebras, SambaNova, Fireworks, DeepSeek
- **Enterprise**: Cohere, AI21
- **Hosted**: Together, Perplexity, DeepInfra, OpenRouter
- **Local**: Ollama, LM Studio, vLLM, TGI, Llamafile
- **Regional**: YandexGPT, GigaChat, Clova, Maritaca
- **Specialized**: Voyage, Jina, Deepgram, ElevenLabs, Fal

#### Python Bindings
- Synchronous `LLMKitClient` and async `AsyncLLMKitClient`
- Full streaming support with iterators
- Type stubs for IDE completion
- Complete feature parity with Rust core

#### Node.js/TypeScript Bindings
- `LLMKitClient` with async/await API
- Streaming via async iterator and callbacks
- Full TypeScript type definitions
- Complete feature parity with Rust core

### Security
- No unsafe code in core library
- API keys not logged
- HTTPS enforced for all providers

### Testing
- 186+ tests (Rust, Python, Node.js)
- Unit, integration, and mock test coverage

### Documentation
- Getting Started guides for Rust, Python, and Node.js
- API reference documentation
- Provider configuration guide
- 27+ example files

---

## Future Plans

### [0.2.0] - Planned

- Provider pooling and load balancing
- Automatic failover between providers
- Health checking for provider availability
- Cost metering and budget controls
- Guardrails integration

---

[Unreleased]: https://github.com/yfedoseev/llmkit/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yfedoseev/llmkit/releases/tag/v0.1.0
