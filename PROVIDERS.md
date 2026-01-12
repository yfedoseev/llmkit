# LLMKit Provider Support

LLMKit supports **100+ LLM providers** with a unified interface.

## Phase 2: New Providers

### Phase 2.1: Enterprise & Gateway Providers (4)

In Phase 2.1, we added 4 new high-priority providers:
- **Clarifai**: Multimodal AI platform with vision capabilities
- **Vercel AI Gateway**: Unified gateway for multiple LLM providers
- **Poe (Quora)**: LLM aggregator platform with access to 100+ models
- **GradientAI**: LLM API by DigitalOcean App Platform

### Phase 2.2: Additional Tier 1 Providers (5)

In Phase 2.2, we added 5 more high-performance providers:
- **Reka AI**: Multimodal model with strong vision and reasoning capabilities
- **Lambda Labs**: GPU cloud platform with high-performance LLM inference
- **Nvidia NIM**: Enterprise inference microservices for on-premise deployment
- **Xinference**: Distributed inference platform for local and cloud deployment
- **PublicAI**: Sovereign model hosting platform with privacy-focused models

These providers bring coverage to 100+ total providers.

### Phase 2.3: Validated Additional Providers (8)

In Phase 2.3, we added 8 more validated OpenAI-compatible providers:
- **Bytez**: Serverless platform with access to 175,000+ HuggingFace models
- **Chutes**: Open-source decentralized compute provider with cost-effective inference
- **CometAPI**: Unified API aggregator supporting 500+ models from major providers
- **CompactifAI**: Specialized provider for compressed/optimized LLMs by Multiverse Computing
- **Synthetic**: Run-on-demand access to any HuggingFace model with flexible pricing
- **Morph**: Code-patching LLMs optimized for software engineering tasks
- **Heroku AI**: PaaS-integrated inference leveraging Amazon Bedrock
- **v0 (Vercel)**: Web development-focused LLM API specialized for Next.js projects

These providers bring coverage to 100+ total providers.

### Phase 2.3B: Specialized APIs (2)

In Phase 2.3B, we added 2 specialized media generation APIs:
- **RunwayML**: Industry-leading video generation API (text-to-video, image-to-video)
  - Base URL: `https://api.runwayml.com/v1`
  - Auth: Bearer token via `RUNWAYML_API_SECRET`
  - Models: `gen4_turbo`, `gen3a_turbo`, `veo3.1`
  - Features: Task-based async workflow with polling, 5-minute timeout
  - Type: Custom REST API (not OpenAI-compatible)

- **Recraft**: Professional-grade image generation API (#1 ranked on benchmarks)
  - Base URL: `https://external.api.recraft.ai/v1`
  - Auth: Bearer token via `RECRAFT_API_TOKEN`
  - Models: `recraft-v3`
  - Features: Professional styles, substyles, custom sizes
  - Type: Semi-OpenAI-compatible REST API

These specialized APIs expand LLMKit beyond text-based LLMs into creative media generation (video and image).

## Audio & Speech Providers

LLMKit includes three specialized audio providers with separate interfaces (not part of the unified chat provider pattern):

### Speech-to-Text (Transcription)

**AssemblyAI** - Advanced audio transcription with speaker diarization
- Feature flag: `assemblyai`
- Base URL: `https://api.assemblyai.com/v2`
- Auth: `ASSEMBLYAI_API_KEY` environment variable
- Usage: `AssemblyAIProvider::from_env()` or `AssemblyAIProvider::new(api_key)`
- Features:
  - High-accuracy speech-to-text transcription
  - Speaker diarization (multi-speaker detection)
  - Entity recognition in transcripts
  - Sentiment analysis
  - Content moderation
  - Supported formats: WAV, MP3, AAC, FLAC, OGG, M4A
  - Max audio duration: Up to 720 minutes (professional tier)

**Deepgram** - Fast, accurate speech-to-text API
- Feature flag: `deepgram`
- Base URL: `https://api.deepgram.com/v1`
- Auth: `DEEPGRAM_API_KEY` environment variable (Token header authentication)
- Usage: `DeepgramProvider::from_env()` or `DeepgramProvider::with_api_key(key)`
- Features:
  - Speech-to-text transcription
  - Real-time streaming transcription support
  - Speaker diarization
  - Language detection
  - Sentiment analysis
  - Low-latency processing

### Text-to-Speech

**ElevenLabs** - High-quality voice synthesis and cloning
- Feature flag: `elevenlabs`
- Base URL: `https://api.elevenlabs.io/v1`
- Auth: `ELEVENLABS_API_KEY` environment variable (xi-api-key header authentication)
- Usage: `ElevenLabsProvider::from_env()` or `ElevenLabsProvider::with_api_key(key)`
- Features:
  - Natural-sounding text-to-speech synthesis
  - Voice cloning capabilities
  - Multiple voice options
  - Streaming audio output support
  - Professional-grade audio quality

**Note**: Audio providers use a different interface than the unified chat provider pattern. They are accessed directly via their respective provider classes, not through the unified `LLMKitClient`.

## Embedding Providers

LLMKit includes three specialized embedding providers for semantic search, RAG, and similarity matching:

### Voyage AI - High-Quality Embeddings & Reranking

**Voyage AI** specializes in state-of-the-art embedding and reranking models.

- Feature flag: `voyage`
- Base URL: `https://api.voyageai.com/v1`
- Auth: `VOYAGE_API_KEY` environment variable (Bearer token)
- Usage: `LLMKitClient::new().with_voyage_from_env()` or `client.with_voyage(api_key)?`
- **Implements**: Both `Provider` trait (unified chat) and `EmbeddingProvider` trait (embedding-specific)

**Supported Embedding Models**:
- `voyage-3` - Latest general-purpose embeddings (1024 dimensions)
- `voyage-3-lite` - Faster variant (512 dimensions)
- `voyage-code-3` - Optimized for code embeddings (1024 dimensions)
- `voyage-finance-2` - Domain-optimized for finance (1024 dimensions)
- `voyage-law-2` - Domain-optimized for legal documents (1024 dimensions)

**Supported Reranking Models**:
- `rerank-2` - General-purpose reranking
- `rerank-2-lite` - Faster reranking variant

**Features**:
- High-quality semantic embeddings for search and RAG
- Document and query-specific embedding types
- Reranking for relevance optimization
- Batch processing (up to 128 documents per request)
- Truncation support for long texts

**Usage Example**:
```rust
use llmkit::ClientBuilder;

// Add as unified provider
let client = ClientBuilder::new()
    .with_voyage_from_env()?
    .build()?;

// Use via unified interface
let response = client.complete("voyage", "voyage-3", CompletionRequest { ... })?;

// Use via embedding interface
let provider = VoyageProvider::from_env()?;
let embeddings = provider.embed("voyage-3", vec!["text1".to_string(), "text2".to_string()]).await?;
```

### Jina AI - Multilingual Embeddings & Document Processing

**Jina AI** offers multilingual embeddings, reranking, and web reading capabilities.

- Feature flag: `jina`
- Base URL: `https://api.jina.ai/v1`
- Reader URL: `https://r.jina.ai` (for web content extraction)
- Auth: `JINA_API_KEY` environment variable (Bearer token)
- Usage: `LLMKitClient::new().with_jina_from_env()` or `client.with_jina(api_key)?`
- **Implements**: Both `Provider` trait (unified chat) and `EmbeddingProvider` trait (embedding-specific)

**Supported Embedding Models**:
- `jina-embeddings-v3` - Latest multilingual embeddings (1024 dimensions, dimension-adjustable)
- `jina-embeddings-v2-base-en` - English-optimized embeddings (768 dimensions)
- `jina-embeddings-v2-base-code` - Code-optimized embeddings (768 dimensions)
- `jina-clip-v2` - Vision-language embeddings for multimodal search (1024 dimensions)

**Supported Reranking Models**:
- `jina-reranker-v2-base-multilingual` - Multilingual reranking (supports 100+ languages)
- `jina-colbert-v2` - ColBERT-based dense retrieval reranking

**Supported Reader Models**:
- `jina-reader` - Extract and format web page content (supports markdown conversion)

**Features**:
- Multilingual support across 100+ languages
- Task-aware embeddings (query vs. document)
- Dimension customization (for v3 model)
- Batch processing (up to 2048 documents per request)
- Web content extraction and formatting
- Reranking for relevance optimization

**Usage Example**:
```rust
use llmkit::ClientBuilder;

// Add as unified provider
let client = ClientBuilder::new()
    .with_jina_from_env()?
    .build()?;

// Use via unified interface
let response = client.complete("jina", "jina-embeddings-v3", CompletionRequest { ... })?;

// Use via embedding interface
let provider = JinaProvider::from_env()?;
let embeddings = provider.embed("jina-embeddings-v3", vec!["text1".to_string()]).await?;

// Read web content
let content = provider.read_url("https://example.com").await?;
```

### Mistral Embeddings - Lightweight Embedding Service

**Mistral Embeddings** provides efficient embedding models for semantic search, RAG, and similarity matching.

- Feature flag: `mistral-embeddings`
- Auth: `MISTRAL_API_KEY` environment variable
- Usage: `MistralEmbeddingsProvider::from_env()` or `MistralEmbeddingsProvider::new(api_key)`
- **Implementation Status**: Direct embedding interface (mock implementation - production API integration needed)
- **Does NOT implement** the `Provider` trait (separate embedding-only interface)

**Supported Models**:
- `mistral-embed` - General-purpose embeddings
- `mistral-large-latest` - Can be used for embeddings with large model capability

**Features**:
- Semantic embeddings for search and RAG
- Single and batch embedding generation
- Token usage tracking
- Simple, lightweight API

**Usage Example**:
```rust
use llmkit::providers::MistralEmbeddingsProvider;

// Direct provider usage (not through LLMKitClient)
let provider = MistralEmbeddingsProvider::from_env()?;

// Single text embedding
let response = provider.embed("text to embed", "mistral-embed").await?;

// Batch embeddings
let texts = vec!["text1", "text2", "text3"];
let response = provider.embed_batch(&texts, "mistral-embed").await?;
```

**Note**: Mistral Embeddings uses a separate interface from the unified chat provider pattern. It is accessed directly via `MistralEmbeddingsProvider`, not through `LLMKitClient`.

## Regional Providers - Chinese Market

LLMKit includes two specialized providers optimized for the Chinese market with native language support:

### Baidu Wenxin - Enterprise Chinese LLM Service

**Baidu Wenxin** is China's leading enterprise LLM platform with models specifically optimized for Chinese language understanding and generation.

- Feature flag: `baidu`
- Auth: `BAIDU_API_KEY` and `BAIDU_SECRET_KEY` environment variables (dual authentication)
- Usage: `BaiduProvider::from_env()` or `BaiduProvider::new(api_key, secret_key)`
- **Implementation Status**: Complete - Full `Provider` trait implementation with HTTP client, streaming fallback, and token usage tracking
- **Implements** the `Provider` trait for seamless use with `LLMKitClient`

**Supported ERNIE-Bot Models** (Name → Context Window / Max Output):
- `ERNIE-Bot` - 2K context / 1K max output (base model)
- `ERNIE-Bot-Plus` - 8K context / 2K max output (with function calling)
- `ERNIE-Bot-Pro` - 32K context / 4K max output (with function calling)
- `ERNIE-Bot-Ultra` - 200K context / 8K max output (with function calling, recommended)

**Features**:
- Enterprise-grade reliability with SLA guarantees
- Native Chinese language optimization
- Multiple model tiers for different use cases
- Function calling support (Plus/Pro/Ultra)
- Streaming support via fallback pattern
- Stable and Beta API versions available

**Usage Example**:
```rust
use llmkit::providers::BaiduProvider;

// Create provider
let provider = BaiduProvider::from_env()?;
// or: let provider = BaiduProvider::new(api_key, secret_key);

// List available models
let models = provider.list_models().await?;

// Get model information
if let Some(info) = BaiduProvider::get_model_info("ERNIE-Bot-Pro") {
    println!("Model: {}", info.name);
    println!("Context: {} tokens", info.context_window);
    println!("Supports functions: {}", info.supports_function_call);
}

// Or use with LLMKitClient for unified interface
use llmkit::client::ClientBuilder;

let client = ClientBuilder::new()
    .with_baidu("api_key", "secret_key")?
    .build()?;
```

**Note**: Baidu Wenxin requires dual authentication (API key + secret key). Both credentials must be provided during initialization.

### Alibaba DashScope - Advanced Multilingual LLMs on DashScope Platform

**Alibaba DashScope** is a unified platform offering state-of-the-art multilingual models with strong Chinese language capabilities. It supports multiple model families including Qwen, Llama, Mistral, and Baichuan models.

- Feature flag: `alibaba`
- Base URL: `https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation`
- Auth: `ALIBABA_API_KEY` environment variable
- Usage: `AlibabaProvider::from_env()` or `AlibabaProvider::new(api_key)`
- Client integration: `builder.with_alibaba_from_env()` or `builder.with_alibaba(api_key)?`
- **Implementation Status**: Complete - Full `Provider` trait implementation with HTTP client, streaming fallback, and token usage tracking
- **Implements** the `Provider` trait for seamless use with `LLMKitClient`

**Supported Models** (14 total across multiple families):

**Qwen Models** (Name → Context Window / Max Output):
- `qwen-turbo` - 8K context / 2K max output
- `qwen-plus` - 32K context / 4K max output
- `qwen-max` - 32K context / 8K max output
- `qwen-max-longcontext` - 200K context / 8K max output (recommended for long documents)
- `qwen-vl-plus` - 16K context / 1K max output (Vision)
- `qwen-vl-max` - 32K context / 2K max output (Vision)

**Qwen Code Models**:
- `qwen-coder-turbo` - Optimized for code generation
- `qwen-coder-max` - Advanced code generation and understanding

**Open Source Models via DashScope** (Hosted compatibility):
- `llama-2-7b-chat` - Meta Llama 2 7B Chat
- `llama-2-13b-chat` - Meta Llama 2 13B Chat
- `llama-2-70b-chat` - Meta Llama 2 70B Chat
- `mistral-7b-instruct` - Mistral 7B Instruct
- `baichuan-2-7b-chat` - Baichuan 2 7B Chat
- `baichuan-2-13b-chat` - Baichuan 2 13B Chat

**Features**:
- Unified access to multiple model families on single platform
- Multilingual support with exceptional Chinese language performance
- Vision-language models (Qwen-VL) for image understanding
- Function calling and JSON output
- Strong Chinese and multilingual support (100+ languages)
- Semantic understanding across diverse domains
- Streaming support via fallback pattern

**Usage Example**:
```rust
use llmkit::providers::AlibabaProvider;
use llmkit::types::{CompletionRequest, Role, ContentBlock};

// Create provider
let provider = AlibabaProvider::from_env()?;
// or: let provider = AlibabaProvider::new(api_key);

// List available models
let models = provider.list_models().await?;

// Get model information
if let Some(info) = AlibabaProvider::get_model_info("qwen-max-longcontext") {
    println!("Model: {}", info.name);
    println!("Context: {} tokens", info.context_window);
    println!("Supports vision: {}", info.supports_vision);
    println!("Supports functions: {}", info.supports_function_call);
}

// Or use with LLMKitClient for unified interface
use llmkit::client::ClientBuilder;

let client = ClientBuilder::new()
    .with_alibaba_from_env()?
    .build()?;
```

**Model Specializations** (Qwen models):
- `General` - Default, balanced performance across tasks
- `Vision` - Specialized for image and multimodal understanding (Qwen-VL)
- `Code` - Optimized for code generation and understanding (Qwen Code models)
- `Math` - Specialized for mathematical reasoning and problem-solving

**Note**: The Alibaba DashScope platform hosts not just Qwen models but also open-source models (Llama, Mistral, Baichuan) for maximum flexibility. This provider is recommended for applications targeting Chinese users or requiring strong Chinese language support with access to diverse model families.

## Supported Providers

### Tier 1: Production Cloud Providers (10)

Enterprise-grade LLM services with strong API stability and feature coverage.

| Provider | Feature | Base URL | Default Model | Tools | Vision | Streaming |
|----------|---------|----------|--|--|--|--|
| OpenAI | `openai` | `https://api.openai.com/v1` | gpt-4o | ✓ | ✓ | ✓ |
| Anthropic | `anthropic` | N/A (native) | claude-sonnet-4 | ✓ | ✓ | ✓ |
| Together AI | `together` | `https://api.together.xyz/v1` | meta-llama/Meta-Llama-3.1-70B | ✓ | ✓ | ✓ |
| Fireworks AI | `fireworks` | `https://api.fireworks.ai/inference/v1` | accounts/fireworks/models/llama-v3p1-70b | ✓ | ✓ | ✓ |
| DeepSeek | `deepseek` | `https://api.deepseek.com/v1` | deepseek-chat | ✓ | ✗ | ✓ |
| Perplexity | `perplexity` | `https://api.perplexity.ai` | llama-3.1-sonar-large-128k-online | ✗ | ✗ | ✓ |
| Anyscale | `anyscale` | `https://api.endpoints.anyscale.com/v1` | meta-llama/Meta-Llama-3-70B-Instruct | ✓ | ✗ | ✓ |
| DeepInfra | `deepinfra` | `https://api.deepinfra.com/v1/openai` | meta-llama/Meta-Llama-3.1-70B-Instruct | ✓ | ✓ | ✓ |
| xAI (Grok) | `xai` | `https://api.x.ai/v1` | grok-2-latest | ✓ | ✓ | ✓ |
| NVIDIA NIM | `nvidia` | `https://integrate.api.nvidia.com/v1` | meta/llama-3.1-70b-instruct | ✓ | ✓ | ✓ |

### Tier 2: Cloud Platforms (24)

High-quality providers with strong feature support across multiple model types.

| Provider | Feature | Base URL | Default Model | Tools | Vision | Streaming |
|----------|---------|----------|--|--|--|--|
| Clarifai | `clarifai` | `https://api.clarifai.com/v1` | claude-3-5-sonnet | ✗ | ✓ | ✓ |
| Vercel AI Gateway | `vercel-ai` | `https://gateway.ai.cloudflare.com/v1/vercel` | gpt-4o | ✓ | ✓ | ✓ |
| Poe (Quora) | `poe` | `https://api.poe.com/v1` | claude-3-5-sonnet | ✓ | ✗ | ✓ |
| GradientAI | `gradient` | `https://api.gradient.ai/v1` | claude-3-sonnet | ✓ | ✓ | ✓ |
| Novita AI | `novita` | `https://api.novita.ai/v3/openai` | meta-llama/llama-3.1-70b-instruct | ✓ | ✓ | ✓ |
| Hyperbolic | `hyperbolic` | `https://api.hyperbolic.xyz/v1` | meta-llama/Meta-Llama-3.1-70B-Instruct | ✗ | ✗ | ✓ |
| Cerebras | `cerebras` | `https://api.cerebras.ai/v1` | llama3.1-70b | ✓ | ✗ | ✓ |
| Modal | `modal` | `https://api.modal.com/v1` | - | ✓ | ✗ | ✓ |
| Lambda Labs | `lambda` | `https://cloud.lambdalabs.com/api/v1` | - | ✓ | ✗ | ✓ |
| Friendli AI | `friendli` | `https://inference.friendli.ai/v1` | - | ✓ | ✓ | ✓ |
| OctoAI | `octoai` | `https://text.octoai.run/v1` | meta-llama-3.1-70b-instruct | ✓ | ✓ | ✓ |
| Predibase | `predibase` | `https://serving.predibase.com/v1` | - | ✓ | ✗ | ✓ |
| Nebius | `nebius` | `https://api.studio.nebius.ai/v1` | meta-llama/Meta-Llama-3.1-70B-Instruct | ✓ | ✓ | ✓ |
| SiliconFlow | `siliconflow` | `https://api.siliconflow.cn/v1` | Qwen/Qwen2.5-7B-Instruct | ✓ | ✓ | ✓ |
| Moonshot | `moonshot` | `https://api.moonshot.cn/v1` | moonshot-v1-8k | ✓ | ✗ | ✓ |
| Zhipu (GLM) | `zhipu` | `https://open.bigmodel.cn/api/paas/v4` | glm-4 | ✓ | ✓ | ✓ |
| Yi | `yi` | `https://api.lingyiwanwu.com/v1` | yi-large | ✓ | ✓ | ✓ |
| MiniMax | `minimax` | `https://api.minimax.chat/v1` | abab6-chat | ✓ | ✗ | ✓ |
| DashScope (Alibaba) | `dashscope` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | qwen-turbo | ✓ | ✓ | ✓ |
| Featherless AI | `featherless` | `https://api.featherless.ai/v1` | - | ✓ | ✗ | ✓ |
| NScale | `nscale` | `https://inference.nscale.com/v1` | - | ✓ | ✗ | ✓ |
| VolcEngine | `volcengine` | `https://ark.cn-beijing.volces.com/api/v3` | - | ✓ | ✓ | ✓ |
| OVHCloud | `ovhcloud` | `https://llama-3-1-70b-instruct...` | - | ✓ | ✗ | ✓ |
| Galadriel | `galadriel` | `https://api.galadriel.com/v1` | - | ✓ | ✗ | ✓ |

### Tier 3: Local & Self-Hosted (18)

Run LLMs locally on your hardware with full privacy and control.

| Provider | Feature | Default Port | Tools | Vision | Streaming |
|----------|---------|--|--|--|--|
| LM Studio | `lm-studio` | 1234 | ✓ | ✓ | ✓ |
| vLLM | `vllm` | 8000 | ✓ | ✓ | ✓ |
| Text Generation WebUI | `text-gen-webui` | 5000 | ✗ | ✗ | ✓ |
| TGI (HuggingFace) | `tgi` | 8080 | ✓ | ✗ | ✓ |
| Llamafile | `llamafile` | 8080 | ✗ | ✗ | ✓ |
| Xinference | `xinference` | 9997 | ✓ | ✓ | ✓ |
| FastChat | `fastchat` | 21002 | ✓ | ✗ | ✓ |
| Aphrodite Engine | `aphrodite` | 2242 | ✓ | ✓ | ✓ |
| Tabby | `tabby` | 8080 | ✗ | ✗ | ✓ |
| KoboldCpp | `koboldcpp` | 5001 | ✗ | ✗ | ✓ |
| LocalAI | `localai` | - | - | - | - |
| Jan | `jan` | - | - | - | - |
| OpenLLM | `openllm` | - | - | - | - |
| Nitro | `nitro` | - | - | - | - |
| MLC LLM | `mlc` | - | - | - | - |
| Infinity | `infinity` | 7997 | ✗ | ✗ | ✓ |
| Petals | `petals` | - | ✗ | ✗ | ✓ |
| Triton | `triton` | 8000 | ✗ | ✗ | ✓ |

### Tier 4: Regional & Specialized (15)

Providers focused on specific regions or use cases.

| Provider | Region | Feature | Tools | Vision | Streaming |
|----------|--------|---------|--|--|--|
| Baichuan | China | `baichuan` | - | - | - |
| Qwen (Alibaba) | China | `qwen` | - | - | - |
| Stepfun | China | `stepfun` | - | - | - |
| 360.AI | China | `ai360` | - | - | - |
| Spark (iFLYTEK) | China | `spark` | - | - | - |
| Ernie (Baidu) | China | `ernie` | - | - | - |
| Hunyuan (Tencent) | China | `hunyuan` | - | - | - |
| Writer AI | Enterprise | `writer` | ✓ | ✗ | ✓ |
| Reka AI | Multimodal | `reka` | ✓ | ✓ | ✓ |
| Upstage (Solar) | Korea | `upstage` | ✓ | ✗ | ✓ |
| Meta Llama API | Meta | `meta-llama` | - | - | - |
| Pangu | Regional | `pangu` | - | - | - |
| SenseNova | Regional | `sensenova` | - | - | - |
| SEA-LION | Singapore | `sea-lion` | - | - | - |
| Tiangong | Regional | `tiangong` | - | - | - |

### Tier 5: Proxy & Gateway Providers (5)

Proxy services that provide unified access to multiple providers.

| Provider | Feature | Base URL |
|----------|---------|----------|
| OpenAI Proxy | `openai_proxy` | Custom (proxy) |
| Portkey | `portkey` | Custom (gateway) |
| Helicone | `helicone` | Custom (observability) |
| Keywords AI | `keywords-ai` | `https://api.keywordsai.co/api` |
| Unify | `unify` | - |

### Tier 6: Enterprise & Commercial (17)

Enterprise-focused providers and commercial inference platforms.

| Provider | Feature | Base URL | Tools | Vision | Streaming |
|----------|---------|----------|--|--|--|
| AIML API | `aimlapi` | `https://api.aimlapi.com/v1` | ✓ | - | - |
| Prem | `prem` | - | - | - | - |
| Martian | `martian` | - | - | - | - |
| CentML | `centml` | - | - | - | - |
| Crusoe | `crusoe` | - | - | - | - |
| CoreWeave | `coreweave` | - | - | - | - |
| Lightning | `lightning` | - | - | - | - |
| Cerebrium | `cerebrium` | - | - | - | - |
| Banana | `banana` | - | - | - | - |
| Beam | `beam` | - | - | - | - |
| Mystic | `mystic` | - | - | - | - |
| Bytez | `bytez` | - | - | - | - |
| Morph | `morph` | - | - | - | - |
| Kluster | `kluster` | - | - | - | - |
| Lighton | `lighton` | - | - | - | - |
| IONOS | `ionos` | - | - | - | - |
| Scaleway | `scaleway` | - | - | - | - |

### Additional Providers (15+)

| Provider | Type | Feature |
|----------|------|---------|
| Google PaLM | API | `google` |
| Vertex AI | Google Cloud | `vertex` |
| Cohere | API | `cohere` |
| AI21 Labs | API | `ai21` |
| HuggingFace | Inference | `huggingface` |
| Replicate | Inference | `replicate` |
| Baseten | Inference | `baseten` |
| RunPod | GPU Cloud | `runpod` |
| Cloudflare AI | Cloudflare | `cloudflare` |
| IBM Watson X | Enterprise | `watsonx` |
| Databricks | Data Platform | `databricks` |
| DataRobot | AutoML | `datarobot` |
| SageMaker | AWS | `sagemaker` |
| Snowflake | Data Warehouse | `snowflake` |
| SambaNova | Inference | `sambanova` |
| Stability AI | Image/Audio | `stability` |
| OpenRouter | Gateway | `openrouter` |
| Ollama | Local | `ollama` |
| Groq | Inference | `groq` |
| Mistral | API | `mistral` |
| Azure OpenAI | Azure | `azure` |
| AWS Bedrock | AWS | `bedrock` |
| Oracle OCI | Oracle Cloud | `oracle` |
| SAP Generative AI | SAP | `sap` |
| Aleph Alpha | API | `aleph-alpha` |
| NLP Cloud | API | `nlp-cloud` |
| Voyage AI | Embeddings | `voyage` |
| Jina AI | Embeddings | `jina` |
| FAL | Inference | `fal` |
| Deepgram | Audio | `deepgram` |
| ElevenLabs | Audio | `elevenlabs` |
| Yandex | Russian | `yandex` |
| GigaChat | Russian | `gigachat` |
| Clova | Korean | `clova` |
| Maritaca | Brazilian | `maritaca` |
| Tavily | Search | `tavily` |
| Mistral Embeddings | Embeddings | `mistral-embeddings` |
| Lepton AI | Inference | `lepton` |
| GPT4All | Local | `gpt4all` |
| Alibaba Qwen | China | `alibaba` |
| Baidu Ernie | China | `baidu` |
| AssemblyAI | Transcription | `assemblyai` |
| QwQ | Reasoning | `qwq` |
| vLLM Embeddings | Local | `vllm` |
| Perplexity Search | Search | `perplexity` |

## Installation

### Use All Providers
```bash
cargo add llmkit --features all-providers
```

### Use Specific Providers
```bash
# Production cloud providers
cargo add llmkit --features openai,anthropic,together,fireworks

# Local inference
cargo add llmkit --features vllm,lm-studio,ollama

# Regional providers
cargo add llmkit --features qwen,moonshot,claude  # Note: adjust for actual available features
```

### Feature Flags by Category

**Tier 1 Production Cloud:**
```
openai, anthropic, together, fireworks, deepseek, perplexity, anyscale, deepinfra, xai, nvidia
```

**Tier 2 Cloud Platforms:**
```
novita, hyperbolic, cerebras, modal, lambda, friendli, octoai, predibase, nebius, siliconflow,
moonshot, zhipu, yi, minimax, dashscope, featherless, nscale, volcengine, ovhcloud, galadriel
```

**Tier 3 Local/Self-Hosted:**
```
lm-studio, vllm, tgi, llamafile, xinference, fastchat, aphrodite, tabby, koboldcpp,
text-gen-webui, localai, jan, openllm, nitro, mlc, infinity, petals, triton
```

**Tier 4 Regional/Specialized:**
```
baichuan, qwen, stepfun, ai360, spark, ernie, hunyuan, writer, reka, upstage,
meta-llama, pangu, sensenova, sea-lion, tiangong
```

**Tier 5 Proxy/Gateway:**
```
openai-proxy, portkey, helicone, keywords-ai, unify
```

**Tier 6 Enterprise/Commercial:**
```
aimlapi, prem, martian, centml, crusoe, coreweave, lightning, cerebrium, banana,
beam, mystic, bytez, morph, kluster, lighton, ionos, scaleway
```

## Usage Examples

### Using OpenAI-Compatible Providers
```rust
use llmkit::ClientBuilder;

// Using environment variables
let client = ClientBuilder::new()
    .with_together_from_env()
    .build()?;

// Using explicit API key
let client = ClientBuilder::new()
    .with_together("your-api-key")?
    .build()?;

// Using custom config
let config = ProviderConfig {
    api_key: Some("your-api-key".to_string()),
    base_url: Some("https://api.together.xyz/v1".to_string()),
    ..Default::default()
};
let client = ClientBuilder::new()
    .with_together_config(config)?
    .build()?;
```

### Using Local Models
```rust
use llmkit::ClientBuilder;

// LM Studio (default port 1234)
let client = ClientBuilder::new()
    .with_lm_studio_url("http://localhost:1234")?
    .build()?;

// vLLM (default port 8000)
let client = ClientBuilder::new()
    .with_vllm_url("http://localhost:8000")?
    .build()?;
```

### Multi-Provider Setup
```rust
use llmkit::ClientBuilder;

let client = ClientBuilder::new()
    .with_openai_from_env()?
    .with_anthropic_from_env()?
    .with_together_from_env()?
    .set_default_provider("openai")?
    .build()?;

// Route requests to different providers
let response = client.complete("openai/gpt-4o", "Hello")?;
let response = client.complete("anthropic/claude-sonnet-4", "Hello")?;
let response = client.complete("together/meta-llama-3.1-70b", "Hello")?;
```

## Contributing

To add support for a new provider:

1. **Implement the Provider trait** in `src/providers/provider.rs`
2. **Add feature flag** to `Cargo.toml`
3. **Add builder methods** to `src/client.rs`
4. **Update this documentation**
5. **Add tests** with mock HTTP responses

For OpenAI-compatible providers, use the generic `OpenAICompatibleProvider` instead of creating custom implementations.

## Support

- GitHub Issues: [github.com/yfedoseev/llmkit/issues](https://github.com/yfedoseev/llmkit/issues)
- Documentation: [llmkit.rs](https://llmkit.rs)
- Examples: [github.com/yfedoseev/llmkit/tree/main/examples](https://github.com/yfedoseev/llmkit/tree/main/examples)
