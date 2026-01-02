# Changelog

All notable changes to LLMKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-XX

### Added

#### Core Library (Rust)
- Initial release with unified LLM API interface
- Support for 37+ LLM providers:
  - **Core**: Anthropic, OpenAI, Azure OpenAI
  - **Cloud**: AWS Bedrock, Google Vertex AI, Google AI (Gemini)
  - **Fast Inference**: Groq, Mistral, Cerebras, SambaNova, Fireworks, DeepSeek
  - **Enterprise**: Cohere, AI21
  - **Hosted**: Together, Perplexity, Anyscale, DeepInfra, Novita, Hyperbolic
  - **Platforms**: HuggingFace, Replicate, Baseten, RunPod
  - **Cloud ML**: Cloudflare, WatsonX, Databricks
  - **Local**: Ollama, LM Studio, vLLM, TGI, Llamafile
  - **Regional**: YandexGPT, GigaChat, Clova, Maritaca
  - **Specialized**: Voyage, Jina, Deepgram, ElevenLabs, Fal
- Streaming completions with async iterators
- Tool/function calling with fluent builder pattern (`ToolBuilder`)
- Extended thinking mode for complex reasoning tasks
- Prompt caching support (5-minute and 1-hour TTL)
- Structured output with JSON schema enforcement
- Vision/image input support (base64 and URLs)
- Embeddings API for text vectors
- Batch processing API for async bulk requests
- Token counting API for cost estimation
- Model registry with 100+ models including:
  - Pricing information (input/output per 1M tokens)
  - Capability flags (vision, tools, streaming, JSON mode, thinking)
  - Benchmark scores (MMLU, HumanEval, MATH, etc.)
- Comprehensive error types for all failure modes
- Feature flags for provider selection (reduces binary size)
- Default retry logic with configurable backoff

#### Python Bindings
- Synchronous `LLMKitClient` for blocking operations
- Asynchronous `AsyncLLMKitClient` for async/await
- Full streaming support with iterators
- Type stubs (`.pyi`) for IDE completion
- All 37 providers accessible via `from_env()` or explicit config
- Complete feature parity with Rust core:
  - Embeddings API
  - Batch processing API
  - Token counting API
  - Model registry access

#### Node.js/TypeScript Bindings
- `LLMKitClient` with async/await API
- Streaming via async iterator (`stream()`) and callback (`completeStream()`)
- Full TypeScript type definitions (`.d.ts`)
- All 37 providers accessible via `fromEnv()` or explicit config
- Complete feature parity with Rust core:
  - Embeddings API
  - Batch processing API
  - Token counting API
  - Model registry access

### Security
- No unsafe code in core library
- API keys not logged
- HTTPS enforced for all providers
- Pre-compiled regex patterns using `LazyLock` for thread safety

### Testing
- 117+ Rust unit and integration tests
- 83 Python tests covering all major features
- 77 Node.js tests covering all major features
- Provider integration tests (requires API keys)

### Documentation
- Getting Started guides for Python, Node.js, and Rust
- API reference documentation
- Provider configuration guide
- 27+ example files across all platforms

---

## Future Plans

### [0.2.0] - Planned

#### Features
- Provider pooling and load balancing
- Automatic failover/fallback between providers
- Health checking for provider availability
- Guardrails integration
- Cost metering and budget controls
- Multi-tenancy support
- Caching provider
- Custom retry configuration
- Prompt templates with variable substitution

#### Improvements
- Secure string handling for API keys
- Key rotation support
- Audit logging
- Performance optimizations

---

[0.1.0]: https://github.com/yourorg/llmkit/releases/tag/v0.1.0
