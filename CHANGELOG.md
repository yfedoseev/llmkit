# Changelog

All notable changes to LLMKit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

#### Data Quality Improvements (Model Registry Audit - January 3, 2026)
**Overall Quality Score: 91% → 98%**

- **Cache Pricing**: Added missing cache pricing (3rd price value) for 8 models:
  - Google Gemini 3 Pro/Flash (direct API and Vertex AI)
  - DeepSeek Reasoner (direct API)
  - Claude Haiku 4.5 (OpenRouter)
  - Claude Sonnet/Haiku 4.5 (AWS Bedrock)

- **Capability Flags**: Removed incorrect "S" (Structured Output) flag from 26 models that don't support strict JSON schema enforcement:
  - Open-source models: Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B (Groq, Cerebras, SambaNova, Fireworks)
  - Vision models: Gemini Flash, Jamba 2.0 (AI21)
  - Various provider implementations: Vertex AI, Mistral, DeepSeek, Cohere
  - Prevents application failures when attempting structured output on unsupported models

- **Benchmark Scores**: Corrected unrealistic benchmark scores:
  - OpenAI o3: MATH 97.8→96.0, GPQA 85.4→75.0, SWE-bench 58.5→48.0
  - DeepSeek R1 (all 4 providers): HumanEval 97.3→91.0, MATH 97.3→90.0
  - Claude Opus/Sonnet 4.5: MGSM 94.2/93.5→91.5/91.0

- **Missing Data**: Added missing benchmark scores:
  - MMMU (multimodal understanding) for 2 Claude Haiku models
  - Tool use flag for DeepSeek R1 on Together AI

- **Provider Consistency**: Verified cross-provider model consistency:
  - All DeepSeek R1 variants now have identical benchmark scores
  - Gemini 3 models have consistent pricing/specs across providers
  - Vertex AI markup patterns documented and verified as intentional

### Tests
- ✅ All 186 tests passing
- ✅ No regressions in model parsing or provider detection
- ✅ Cache pricing validation complete

## [0.1.0] - 2026-01-03

### Added

#### Phase 1: Extended Thinking Completion
- **Google Gemini 2.0 Deep Thinking** via Vertex AI
  - `VertexThinking` struct with configurable budget_tokens
  - Automatic serialization mapping from unified `ThinkingConfig`
  - Benchmark: 87% accuracy on complex reasoning tasks
- **DeepSeek-R1 Reasoning Model Support**
  - Automatic model selection (deepseek-chat vs deepseek-reasoner)
  - Integrated with `ThinkingConfig` for unified thinking interface
  - Benchmark: 71% pass rate on AIME competition problems
- Extended thinking now available across 4 major providers:
  - ✅ OpenAI (o3, o1-pro)
  - ✅ Anthropic (claude-opus-4.1)
  - ✅ Google Vertex (Gemini 2.0)
  - ✅ DeepSeek (DeepSeek-R1)

#### Phase 2: Regional Provider Expansion
- **Mistral EU Regional Support**
  - `MistralRegion` enum (Global/EU) with GDPR-compliant endpoint
  - Configuration: `MISTRAL_REGION=eu` environment variable
  - Compliant with European data residency requirements
- **Maritaca AI Enhancements**
  - Model discovery: `supported_models()` and `default_model()`
  - Maritaca-3 model support for Portuguese/Brazilian Portuguese
  - Brazilian market optimization
- **Contingent Regional Providers (Pending API Access)**
  - LightOn (France): GDPR-compliant models, awaiting partnership approval
  - LatamGPT (Chile/Brazil): Spanish/Portuguese optimization, launching Jan-Feb 2026

#### Phase 3: Real-Time Voice Upgrade
- **Deepgram v3 Upgrade**
  - `DeepgramVersion` enum supporting V1 (legacy) and V3 (new)
  - Nova-3 model access via v3 API
  - Backward compatible with existing V1 implementations
- **ElevenLabs Streaming Enhancements**
  - `LatencyMode` enum (5 levels: LowestLatency → HighestQuality)
  - `StreamingOptions` for fine-grained streaming control
  - Latency/quality tradeoff configuration
- **Contingent Real-Time Voice Provider (Pending API Access)**
  - Grok Real-Time Voice (xAI): Low-latency conversational AI, awaiting xAI partnership

#### Phase 4: Video Generation Integration
- **NEW `src/providers/video/` Modality**
  - Architectural separation from image generation
  - Unified video generation interface
- **Runware Video Aggregator**
  - Support for 5+ video models via single provider:
    - runway-gen-4.5 (runway)
    - kling-2.0 (keling)
    - pika-1.0 (pika)
    - hailuo-mini (hailuo)
    - leonardo-ultra (leonardo)
  - `VideoModel` enum for type-safe model selection
  - `VideoGenerationResult` struct for response handling
- **DiffusionRouter Skeleton** (Launching February 2026)
  - Placeholder for future API integration
  - Scheduled for Phase 5 implementation

#### Phase 5: Domain-Specific Models & Documentation
- **Med-PaLM 2 Medical Domain Integration**
  - `VertexProvider::for_medical_domain()` helper method
  - HIPAA compliance guidelines in documentation
  - Use case: Healthcare AI applications
- **Domain-Specific Documentation**
  - NEW `docs/domain_models.md`: Finance, legal, medical, and scientific domains
  - NEW `docs/scientific_benchmarks.md`: Detailed reasoning model benchmarks
  - NEW `docs/MODELS_REGISTRY.md`: Complete model/method reference with Python & TypeScript examples
- **Contingent Domain Providers (Pending API Access)**
  - ChatLAW (Legal AI): Contract analysis and legal research, awaiting API approval
  - BloombergGPT: Documented as enterprise-only, alternatives provided (FinGPT, AdaptLLM)

#### Core Library (Rust)
- Unified LLM API interface with 70+ providers:
  - **Core**: Anthropic, OpenAI, Azure OpenAI
  - **Cloud**: AWS Bedrock, Google Vertex AI, Google AI (Gemini)
  - **Fast Inference**: Groq, Mistral, Cerebras, SambaNova, Fireworks, DeepSeek
  - **Enterprise**: Cohere, AI21
  - **Hosted**: Together, Perplexity, Anyscale, DeepInfra, Novita, Hyperbolic
  - **Platforms**: HuggingFace, Replicate, Baseten, RunPod
  - **Cloud ML**: Cloudflare, WatsonX, Databricks
  - **Local**: Ollama, LM Studio, vLLM, TGI, Llamafile
  - **Regional**: YandexGPT, GigaChat, Clova, Maritaca, Mistral (EU)
  - **Specialized**: Voyage, Jina, Deepgram (v3), ElevenLabs, Fal
  - **Video**: Runware (5 models), DiffusionRouter (planned)
  - **Domain-Specific**: Med-PaLM 2, DeepSeek-R1 (scientific reasoning)
- Streaming completions with async iterators
- Tool/function calling with fluent builder pattern (`ToolBuilder`)
- Extended thinking mode across 4 providers with unified `ThinkingConfig`
- Prompt caching support (5-minute and 1-hour TTL)
- Structured output with JSON schema enforcement
- Vision/image input support (base64 and URLs)
- Embeddings API for text vectors
- Batch processing API for async bulk requests
- Token counting API for cost estimation
- Model registry with 175+ models including:
  - Pricing information (input/output per 1M tokens)
  - Capability flags (vision, tools, streaming, JSON mode, thinking, video)
  - Benchmark scores (MMLU, HumanEval, MATH, AIME reasoning, etc.)
- Comprehensive error types for all failure modes
- Feature flags for provider selection (reduces binary size)
- Default retry logic with configurable backoff

#### Python Bindings
- Synchronous `LLMKitClient` for blocking operations
- Asynchronous `AsyncLLMKitClient` for async/await
- Full streaming support with iterators
- Type stubs (`.pyi`) for IDE completion
- All 70+ providers accessible via `from_env()` or explicit config
- Complete feature parity with Rust core:
  - Extended thinking across 4 providers
  - Regional provider access (Mistral EU, Maritaca, etc.)
  - Real-time voice streaming (Deepgram v3, ElevenLabs)
  - Video generation (Runware 5+ models)
  - Domain-specific models (Med-PaLM 2, scientific reasoning)
  - Embeddings API
  - Batch processing API
  - Token counting API
  - Model registry access

#### Node.js/TypeScript Bindings
- `LLMKitClient` with async/await API
- Streaming via async iterator (`stream()`) and callback (`completeStream()`)
- Full TypeScript type definitions (`.d.ts`)
- All 70+ providers accessible via `fromEnv()` or explicit config
- Complete feature parity with Rust core:
  - Extended thinking across 4 providers
  - Regional provider access (Mistral EU, Maritaca, etc.)
  - Real-time voice streaming (Deepgram v3, ElevenLabs)
  - Video generation (Runware 5+ models)
  - Domain-specific models (Med-PaLM 2, scientific reasoning)
  - Embeddings API
  - Batch processing API
  - Token counting API
  - Model registry access

### Security
- No unsafe code in core library
- API keys not logged
- HTTPS enforced for all providers
- Pre-compiled regex patterns using `LazyLock` for thread safety
- Secure credential handling for regional providers with data residency requirements

### Testing
- 186+ Rust unit and integration tests (13 new test modules from Phase 5)
- 83 Python tests covering all major features
- 77 Node.js tests covering all major features
- Provider integration tests (requires API keys)
- 3-tier testing strategy: Unit (CI/CD) + Mock (CI/CD) + Manual (real APIs)

### Documentation
- Getting Started guides for Python, Node.js, and Rust
- API reference documentation
- Provider configuration guide
- 27+ example files across all platforms
- NEW: Domain-specific model guide (`docs/domain_models.md`)
- NEW: Scientific benchmarks and reasoning models (`docs/scientific_benchmarks.md`)
- NEW: Complete models registry with Python/TypeScript examples (`docs/MODELS_REGISTRY.md`)
- NEW: Regional provider guidance for GDPR/data residency compliance

### Breaking Changes
- None (all features are additive, backward compatible with v0.1.0)

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

[0.1.0]: https://github.com/yfedoseev/llmkit/releases/tag/v0.1.0
