# LLMKit Supported Providers

A comprehensive list of all LLM providers supported by the LLMKit Rust library. LLMKit provides a unified interface to interact with 70+ LLM providers and 1,798+ LLM models through a common trait-based system.

**Repository:** https://github.com/yfedoseev/llmkit
**Codebase Location:** `/src/providers/`

---

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
# With specific providers
[dependencies]
llmkit = { version = "0.1", features = ["anthropic", "openai"] }

# With all providers
[dependencies]
llmkit = { version = "0.1", features = ["all-providers"] }
```

### Basic Usage

```rust
use llmkit::providers::{AnthropicProvider, Provider};
use llmkit::types::{Message, Role};

#[tokio::main]
async fn main() {
    let provider = AnthropicProvider::with_api_key("your-api-key")?;

    let messages = vec![
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text { text: "Hello!".to_string() }],
        }
    ];

    let request = CompletionRequest {
        model: "claude-3-sonnet".to_string(),
        messages,
        system: None,
        max_tokens: None,
    };

    let response = provider.complete(request).await?;
    println!("{:?}", response.content);
}
```

---

## Core Providers (Default)

### Anthropic
- **Feature Flag:** `anthropic` (enabled by default)
- **File:** `src/providers/anthropic.rs`
- **Models:** Claude 4.5, Claude 4, Claude 3.5, Claude 3
- **Authentication:** `ANTHROPIC_API_KEY`
- **Features:**
  - Tool calling (function calling)
  - Vision capabilities (image understanding)
  - Token counting
  - Batch processing
  - Extended thinking
  - Streaming responses
- **Supported Operations:** `/chat/completions`, tokens counting, batches

### OpenAI
- **Feature Flag:** `openai` (enabled by default)
- **File:** `src/providers/openai.rs`
- **Models:** GPT-4o, GPT-4 Turbo, GPT-4, GPT-3.5 Turbo
- **Authentication:** `OPENAI_API_KEY`
- **Features:**
  - Tool calling
  - Vision capabilities
  - Token counting
  - Batch processing
  - DALL-E image generation support
  - Embeddings
  - Streaming responses
- **Supported Operations:** `/chat/completions`, `/embeddings`, batches

---

## Cloud & Enterprise Providers

### Azure OpenAI
- **Feature Flag:** `azure`
- **File:** `src/providers/azure.rs`
- **Models:** Same as OpenAI (deployed on Azure infrastructure)
- **Authentication:**
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT` (resource URL)
  - `AZURE_OPENAI_DEPLOYMENT` (deployment ID)
- **Features:** Full OpenAI feature parity via Azure
- **Use Case:** Enterprise Azure deployments, compliance requirements

### AWS Bedrock
- **Feature Flag:** `bedrock`
- **File:** `src/providers/bedrock.rs`
- **Supported Model Families:**
  - Anthropic: Claude 4.5, Claude 4, Claude 3.5, Claude 3
  - Amazon Nova: Pro, Lite, Micro, Nova 2
  - Meta Llama: 4, 3.3, 3.2, 3.1, 3
  - Mistral: Large, Small, Mixtral
  - Cohere: Command R+, Command R
  - AI21 Labs: Jamba 1.5
  - Amazon Titan: Express, Lite
  - DeepSeek: R1, V3
  - Qwen/Alibaba: 2.5
- **Authentication:** AWS credentials (Access Key ID, Secret Access Key, Region)
- **Features:** Multi-provider access through AWS, enterprise billing
- **Use Case:** AWS ecosystem, unified model access through AWS

### Google Vertex AI
- **Feature Flag:** `vertex`
- **File:** `src/providers/vertex.rs`
- **Models:** Gemini Pro, Gemini 1.5, Gemini 2.0
- **Authentication:**
  - Google Cloud service account credentials
  - `GOOGLE_CLOUD_PROJECT` (project ID)
  - `VERTEX_LOCATION` (region, e.g., "us-central1")
- **Features:** Enterprise ML platform integration
- **Use Case:** GCP deployments, enterprise Google Cloud

### Google AI (Gemini)
- **Feature Flag:** `google`
- **File:** `src/providers/google.rs`
- **Models:** Gemini 2.0 Flash, Gemini 1.5 Pro, Gemini 1.5 Flash, Gemini 1.5 Flash-8B
- **Authentication:** `GOOGLE_API_KEY`
- **Features:**
  - Vision capabilities
  - Streaming responses
  - Tool calling
- **Use Case:** Direct access to Google's Gemini models

---

## Fast Inference Providers

### Groq
- **Feature Flag:** `groq`
- **File:** `src/providers/groq.rs`
- **Models:** Llama 3, Mixtral, Gemma
- **Authentication:** `GROQ_API_KEY`
- **Features:** Ultra-fast inference (focuses on speed)
- **Use Case:** Low-latency applications

### Mistral AI
- **Feature Flag:** `mistral`
- **File:** `src/providers/mistral.rs`
- **Models:** Mistral Large, Mistral Small, Mixtral 8x7B
- **Authentication:** `MISTRAL_API_KEY`
- **Features:**
  - OpenAI-compatible API
  - Tool calling
- **Use Case:** European-based inference, efficient models

### Cerebras
- **Feature Flag:** `cerebras`
- **File:** `src/providers/cerebras.rs`
- **Models:** Llama 3.1 (8B, 70B), Llama 3.3 70B
- **Authentication:** `CEREBRAS_API_KEY`
- **Features:** Ultra-fast inference using Cerebras systems
- **Use Case:** High-performance inference

### SambaNova
- **Feature Flag:** `sambanova`
- **File:** `src/providers/sambanova.rs`
- **Models:** Meta Llama 3.1 (8B, 70B, 405B)
- **Authentication:** `SAMBANOVA_API_KEY`
- **Features:** High-performance inference platform
- **Use Case:** Performance-critical applications

### Fireworks AI
- **Feature Flag:** `fireworks`
- **File:** `src/providers/fireworks.rs`
- **Models:** Llama v3.1, Mixtral, Qwen 2.5
- **Authentication:** `FIREWORKS_API_KEY`
- **Features:** Fast inference platform
- **Use Case:** Rapid model deployment

### DeepSeek
- **Feature Flag:** `deepseek`
- **File:** `src/providers/deepseek.rs`
- **Models:**
  - DeepSeek V3 (chat)
  - DeepSeek R1 (reasoning)
- **Authentication:** `DEEPSEEK_API_KEY`
- **Features:** Advanced reasoning capabilities
- **Use Case:** Complex reasoning tasks

---

## Enterprise & LLM Providers

### Cohere
- **Feature Flag:** `cohere`
- **File:** `src/providers/cohere.rs`
- **Models:** Command R+, Command R, Command, Command Light
- **Authentication:** `COHERE_API_KEY` or `CO_API_KEY`
- **Features:**
  - Tool calling
  - Text generation and understanding
- **Use Case:** Enterprise text generation

### AI21 Labs
- **Feature Flag:** `ai21`
- **File:** `src/providers/ai21.rs`
- **Models:** Jamba 1.5 Large, Jamba 1.5 Mini
- **Authentication:** `AI21_API_KEY`
- **Features:** OpenAI-compatible API
- **Use Case:** Alternative LLM provider

---

## Model Hosting & Inference Platforms

### HuggingFace
- **Feature Flag:** `huggingface`
- **File:** `src/providers/huggingface.rs`
- **Modes:**
  - Serverless Inference API (any public model)
  - Dedicated Endpoints (custom endpoints)
- **Models:** Any HF Hub model supporting Messages API (Llama, Mistral, Phi, etc.)
- **Authentication:** `HUGGINGFACE_API_KEY`
- **Features:** Access to Hugging Face Hub models
- **Use Case:** Open-source model hosting

### Replicate
- **Feature Flag:** `replicate`
- **File:** `src/providers/replicate.rs`
- **Models:** Meta Llama, Mistral, Mixtral
- **Authentication:** `REPLICATE_API_TOKEN`
- **Features:** Model deployment and inference
- **Use Case:** Easy model hosting

### Baseten
- **Feature Flag:** `baseten`
- **File:** `src/providers/baseten.rs`
- **Authentication:**
  - `BASETEN_API_KEY`
  - `BASETEN_MODEL_ID`
- **Features:** Model hosting platform
- **Use Case:** Custom model deployment

### RunPod
- **Feature Flag:** `runpod`
- **File:** `src/providers/runpod.rs`
- **Authentication:**
  - `RUNPOD_API_KEY`
  - `RUNPOD_ENDPOINT_ID`
- **Features:** Serverless GPU inference platform
- **Use Case:** GPU-accelerated inference

---

## Cloud ML Providers

### Cloudflare Workers AI
- **Feature Flag:** `cloudflare`
- **File:** `src/providers/cloudflare.rs`
- **Models:** Llama 3 8B, Mistral 7B, Gemma 7B
- **Authentication:**
  - `CLOUDFLARE_API_TOKEN`
  - `CLOUDFLARE_ACCOUNT_ID`
- **Features:** Edge AI inference
- **Use Case:** Cloudflare Workers integration

### IBM WatsonX
- **Feature Flag:** `watsonx`
- **File:** `src/providers/watsonx.rs`
- **Models:** Granite, Llama 3, Mixtral
- **Authentication:**
  - `WATSONX_API_KEY`
  - `WATSONX_PROJECT_ID`
  - `WATSONX_URL`
- **Features:** Enterprise IBM ML platform
- **Use Case:** IBM ecosystem integration

### Databricks
- **Feature Flag:** `databricks`
- **File:** `src/providers/databricks.rs`
- **Models:** Foundation Model APIs (Llama 3.1, DBRX, Mixtral)
- **Authentication:** Workspace URL + token
- **Features:** Databricks Model Serving
- **Use Case:** Databricks lakehouse platform

---

## Local & Self-Hosted

### Ollama
- **Feature Flag:** `ollama`
- **File:** `src/providers/ollama.rs`
- **Models:** Llama 3.2, Mistral, CodeLlama, Phi 3, Qwen 2.5
- **Configuration:** Default endpoint `http://localhost:11434`
- **Authentication:** None (local)
- **Features:** Local model inference
- **Use Case:** Development, privacy-focused local inference

### OpenAI-Compatible Gateway
- **Feature Flag:** `openai-compatible`
- **File:** `src/providers/openai_compatible.rs`
- **Supported Providers (15+ via known endpoints):**
  - Together AI (`https://api.together.xyz/v1`)
  - Perplexity (`https://api.perplexity.ai`)
  - Anyscale (`https://api.endpoints.anyscale.com/v1`)
  - DeepInfra (`https://api.deepinfra.com/v1/openai`)
  - Lepton AI (`https://llama3-1-8b.lepton.run/api/v1`)
  - Novita AI (`https://api.novita.ai/v3/openai`)
  - Hyperbolic (`https://api.hyperbolic.xyz/v1`)
  - Modal (`https://api.modal.com/v1`)
  - Lambda Labs (`https://cloud.lambdalabs.com/api/v1`)
  - Friendli (`https://inference.friendli.ai/v1`)
  - **Local Servers:**
    - LM Studio (`http://localhost:1234/v1`)
    - vLLM (`http://localhost:8000/v1`)
    - TGI (Text Generation Inference) (`http://localhost:8080/v1`)
    - Llamafile (`http://localhost:8080/v1`)
- **Features:** Universal OpenAI-compatible endpoint support
- **Use Case:** Any OpenAI-compatible service

---

## Multi-Provider Gateways

### OpenRouter
- **Feature Flag:** `openrouter`
- **File:** `src/providers/openrouter.rs`
- **Coverage:** 100+ models across multiple providers
- **Available Models:** Anthropic, OpenAI, Google, Meta, Mistral, and more
- **Authentication:** `OPENROUTER_API_KEY`
- **Features:** Unified access to multiple providers
- **Use Case:** Model diversity, fallback support

---

## Specialized Providers

### Voyage AI
- **Feature Flag:** `voyage`
- **File:** `src/providers/voyage.rs`
- **Specialization:** Embeddings and reranking
- **Models:**
  - Voyage 3 (large)
  - Voyage 3 Lite (lightweight)
  - Voyage Code 3 (code embeddings)
  - Rerank 2 (reranking)
  - Rerank 2 Lite (lightweight reranking)
- **Authentication:** `VOYAGE_API_KEY`
- **Use Case:** Semantic search, ranking

### Jina AI
- **Feature Flag:** `jina`
- **File:** `src/providers/jina.rs`
- **Specialization:** Embeddings, reranking, document processing
- **Models:**
  - Jina Embeddings v3
  - Jina Reranker v2
  - Jina Reader (document processing)
- **Authentication:** `JINA_API_KEY`
- **Use Case:** Multimodal embeddings, document understanding

### Fal AI
- **Feature Flag:** `fal`
- **File:** `src/providers/fal.rs`
- **Specialization:** Inference platform for LLMs and image generation
- **Models:** LLaVA (vision), Flux (image), Stable Diffusion
- **Authentication:** `FAL_API_KEY`
- **Use Case:** Multimodal AI tasks

---

## Audio & Media Providers

### Deepgram
- **Feature Flag:** `deepgram`
- **File:** `src/providers/deepgram.rs`
- **Specialization:** Speech-to-text and audio intelligence
- **Features:**
  - Speech transcription
  - Speaker diarization
  - Language detection
  - Sentiment analysis
- **Authentication:** `DEEPGRAM_API_KEY`
- **Use Case:** Audio processing and analysis

### ElevenLabs
- **Feature Flag:** `elevenlabs`
- **File:** `src/providers/elevenlabs.rs`
- **Specialization:** Text-to-speech and voice cloning
- **Features:**
  - Voice synthesis
  - Voice cloning
  - Streaming audio
- **Authentication:** `ELEVENLABS_API_KEY`
- **Use Case:** Voice generation, audio output

---

## Regional & Language-Specific Providers

### YandexGPT (Russian)
- **Feature Flag:** `yandex`
- **File:** `src/providers/yandex.rs`
- **Models:** YandexGPT Pro (32k context), YandexGPT Lite (8k context)
- **Authentication:**
  - `YANDEX_IAM_TOKEN`
  - `YANDEX_FOLDER_ID`
- **Focus:** Russian language optimization
- **Use Case:** Russian-language applications

### Sber GigaChat (Russian)
- **Feature Flag:** `gigachat`
- **File:** `src/providers/gigachat.rs`
- **Models:** GigaChat, GigaChat Lite, GigaChat Pro, GigaChat Max
- **Authentication:**
  - `GIGACHAT_CREDENTIALS` (OAuth 2.0)
  - `GIGACHAT_SCOPE`
- **Features:**
  - Vision support
  - OAuth 2.0 authentication
- **Focus:** Russian language
- **Use Case:** Russian market, Sber ecosystem

### Naver Clova (Korean)
- **Feature Flag:** `clova`
- **File:** `src/providers/clova.rs`
- **Full Name:** CLOVA Studio (HyperCLOVA X)
- **Models:**
  - HCX-005 (multimodal)
  - HCX-007 (reasoning)
  - HCX-DASH-002 (lightweight)
- **Authentication:**
  - `CLOVASTUDIO_API_KEY`
  - `NCP_CLOVASTUDIO_API_KEY`
- **Features:**
  - Vision support
  - Tool calling
  - AI content filtering
- **Focus:** Korean language
- **Use Case:** Korean-language applications

### Maritaca AI (Brazilian Portuguese)
- **Feature Flag:** `maritaca`
- **File:** `src/providers/maritaca.rs`
- **Models:** Sabiá 3, Sabiá 2 Small
- **Authentication:** `MARITALK_API_KEY`
- **Focus:** Portuguese language optimization
- **Use Case:** Brazilian/Portuguese-language applications

### Aleph Alpha (European)
- **Feature Flag:** `aleph-alpha`
- **File:** `src/providers/aleph_alpha.rs`
- **Models:**
  - Luminous Supreme
  - Luminous Extended
  - Luminous Base
  - Llama 3.1 70B
- **Authentication:** `ALEPH_ALPHA_API_KEY`
- **Features:** Multilingual capabilities, European focus
- **Use Case:** European market, multilingual applications

---

## Additional Enterprise Providers

### NLP Cloud
- **Feature Flag:** `nlp-cloud`
- **File:** `src/providers/nlp_cloud.rs`
- **Models:** ChatDolphin, Dolphin, Llama 3 70B, Mixtral 8x7B
- **Authentication:** `NLP_CLOUD_API_KEY`
- **Use Case:** NLP-focused tasks

### Writer (Palmyra)
- **Feature Flag:** `writer`
- **File:** `src/providers/writer.rs`
- **Models:**
  - Palmyra X5 (1M token context)
  - Palmyra X4
- **Authentication:** `WRITER_API_KEY`
- **Features:**
  - Enterprise-grade LLMs
  - Tool calling
  - Extended context windows
- **Use Case:** Long-context document processing

---

## Provider Statistics

| Category | Count | Providers |
|----------|-------|-----------|
| **Core** | 2 | Anthropic, OpenAI |
| **Cloud** | 4 | Azure OpenAI, AWS Bedrock, Google Vertex AI, Google AI |
| **Fast Inference** | 6 | Groq, Mistral, Cerebras, SambaNova, Fireworks, DeepSeek |
| **Enterprise** | 2 | Cohere, AI21 |
| **Hosting Platforms** | 4 | HuggingFace, Replicate, Baseten, RunPod |
| **Cloud ML** | 3 | Cloudflare, IBM WatsonX, Databricks |
| **Local/Self-Hosted** | 1 | Ollama |
| **Multi-Provider Gateways** | 2 | OpenAI-Compatible, OpenRouter |
| **Specialized** | 3 | Voyage, Jina, Fal |
| **Audio/Media** | 2 | Deepgram, ElevenLabs |
| **Regional** | 5 | YandexGPT, GigaChat, Clova, Maritaca, Aleph Alpha |
| **Other Enterprise** | 2 | NLP Cloud, Writer |
| **TOTAL** | **37** | |

---

## Feature Matrix

### Common Supported Operations

**Across most providers:**
- `complete()` - Chat completions
- `complete_stream()` - Streaming responses
- `count_tokens()` - Token counting (where supported)
- `create_batch()` - Batch processing (where supported)

### Vision Support

Providers with image/vision capabilities:
- Anthropic (Claude 3.5, 3)
- OpenAI (GPT-4V)
- Azure OpenAI
- AWS Bedrock (Claude models)
- Google Vertex AI
- Google AI (Gemini)
- Naver Clova
- Fal AI (LLaVA)
- And others

### Tool Calling

Providers with function calling:
- Anthropic
- OpenAI
- Mistral
- Cohere
- Naver Clova
- Writer
- And others

---

## Configuration & Feature Flags

### Enable Specific Providers

```toml
# Cargo.toml
[dependencies]
llmkit = { version = "0.1", features = [
    "anthropic",
    "openai",
    "groq",
    "mistral"
] }
```

### Enable All Providers

```toml
[dependencies]
llmkit = { version = "0.1", features = ["all-providers"] }
```

### Default Features

By default, only these are enabled:
- `anthropic`
- `openai`

### Available Feature Flags

All feature flags correspond to provider names:
- `anthropic`, `openai`, `azure`, `bedrock`, `vertex`, `google`
- `groq`, `mistral`, `cerebras`, `sambanova`, `fireworks`, `deepseek`
- `cohere`, `ai21`
- `huggingface`, `replicate`, `baseten`, `runpod`
- `cloudflare`, `watsonx`, `databricks`
- `ollama`
- `openai-compatible`, `openrouter`
- `voyage`, `jina`, `fal`
- `deepgram`, `elevenlabs`
- `yandex`, `gigachat`, `clova`, `maritaca`, `aleph-alpha`
- `nlp-cloud`, `writer`
- `all-providers` (enables all)

---

## Architecture

### Provider Trait System

All providers implement a unified `Provider` trait defined in `src/provider.rs`:

```rust
pub trait Provider {
    async fn complete(
        &self,
        model: &str,
        messages: Vec<Message>,
        options: CompletionOptions,
    ) -> Result<CompletionResponse>;

    async fn complete_stream(
        &self,
        model: &str,
        messages: Vec<Message>,
        options: CompletionOptions,
    ) -> Result<BoxStream<'_, Result<StreamChunk>>>;

    async fn count_tokens(&self, model: &str, text: &str) -> Result<usize>;

    // Additional methods...
}
```

### Module Organization

- `src/providers/mod.rs` - Module registry and re-exports
- `src/provider.rs` - Core trait definitions
- `src/providers/*.rs` - Individual provider implementations
- `Cargo.toml` - Feature flag definitions

---

## Getting Started Guide

### 1. Install with Desired Providers

```toml
[dependencies]
llmkit = { version = "0.1", features = ["anthropic", "openai", "groq"] }
tokio = { version = "1", features = ["full"] }
```

### 2. Create Provider Instance

```rust
use llmkit::providers::Anthropic;

let provider = Anthropic::new("your-api-key");
```

### 3. Make API Calls

```rust
use llmkit::types::{Message, Role, CompletionOptions};

let messages = vec![
    Message {
        role: Role::User,
        content: "Hello, how are you?".to_string(),
    }
];

let options = CompletionOptions::default();

let response = provider.complete("claude-3-sonnet", messages, options).await?;
println!("{}", response.content);
```

### 4. Switch Providers (same code)

```rust
use llmkit::providers::OpenAI;

let provider = OpenAI::new("openai-api-key");
// Same call, different provider!
let response = provider.complete("gpt-4", messages, options).await?;
```

---

## Best Practices

1. **Use feature flags** to only compile providers you need
2. **Set API keys via environment variables** for security
3. **Handle streaming for long responses** using `complete_stream()`
4. **Use token counting** to estimate API costs
5. **Implement fallbacks** by trying multiple providers
6. **Cache provider instances** to avoid recreating them

---

## Notes

- **Default Providers:** Anthropic and OpenAI are enabled by default
- **Unified Interface:** All providers follow the same `Provider` trait
- **Feature Flags:** Each provider can be enabled/disabled at compile time
- **Authentication:** Most providers use environment variables for API keys
- **Streaming Support:** Most providers support streaming responses
- **Total Providers:** 37 distinct provider implementations

---

## Resources

- **GitHub Repository:** https://github.com/yfedoseev/llmkit
- **Documentation:** Check README.md in repository
- **Cargo.toml:** Feature flags and dependencies configuration
- **Source Code:** `src/providers/` directory for provider implementations

---

## Last Updated
January 2026 - LLMKit with 125+ providers and 1,700+ models
