//! LLM Provider implementations organized by modality.
//!
//! This module contains implementations for various LLM providers,
//! organized by modality (chat, image, audio, embedding, specialized).
//! Each provider is feature-gated to minimize binary size.
//!
//! # Organization
//!
//! - `chat/` - Chat and completion providers (Anthropic, OpenAI, etc.)
//! - `image/` - Image generation providers (Stability AI, Fal, etc.)
//! - `audio/` - Audio processing providers (Deepgram, ElevenLabs, etc.)
//! - `embedding/` - Embedding providers (Voyage, Jina, etc.)
//! - `specialized/` - Specialized APIs (Realtime, Modal, QwQ)

// Modality-based submodules
pub mod audio;
pub mod chat;
pub mod embedding;
pub mod image;
pub mod specialized;

// ============================================================
// Re-exports for backward compatibility
// All existing imports continue to work at the top level
// ============================================================

// Chat providers
#[cfg(feature = "anthropic")]
pub use chat::{anthropic, AnthropicProvider};

#[cfg(feature = "openai")]
pub use chat::{openai, OpenAIProvider};

#[cfg(feature = "openrouter")]
pub use chat::{openrouter, OpenRouterProvider};

#[cfg(feature = "ollama")]
pub use chat::{ollama, OllamaProvider};

#[cfg(feature = "groq")]
pub use chat::{groq, GroqProvider};

#[cfg(feature = "mistral")]
pub use chat::{mistral, MistralProvider};

#[cfg(feature = "azure")]
pub use chat::azure::{AzureConfig, AzureOpenAIProvider};

#[cfg(feature = "bedrock")]
pub use chat::bedrock::{BedrockBuilder, BedrockConfig, BedrockProvider, ModelFamily};

#[cfg(feature = "openai-compatible")]
pub use chat::openai_compatible::{known_providers, OpenAICompatibleProvider, ProviderInfo};

#[cfg(feature = "google")]
pub use chat::{google, GoogleProvider};

#[cfg(feature = "vertex")]
pub use chat::vertex::{VertexConfig, VertexProvider};

#[cfg(feature = "cohere")]
pub use chat::{cohere, CohereProvider};

#[cfg(feature = "ai21")]
pub use chat::{ai21, AI21Provider};

#[cfg(feature = "huggingface")]
pub use chat::{huggingface, HuggingFaceProvider};

#[cfg(feature = "replicate")]
pub use chat::{replicate, ReplicateProvider};

#[cfg(feature = "baseten")]
pub use chat::{baseten, BasetenProvider};

#[cfg(feature = "runpod")]
pub use chat::{runpod, RunPodProvider};

#[cfg(feature = "cloudflare")]
pub use chat::{cloudflare, CloudflareProvider};

#[cfg(feature = "watsonx")]
pub use chat::{watsonx, WatsonxProvider};

#[cfg(feature = "databricks")]
pub use chat::{databricks, DatabricksProvider};

#[cfg(feature = "datarobot")]
pub use chat::{datarobot, DataRobotProvider};

#[cfg(feature = "cerebras")]
pub use chat::{cerebras, CerebrasProvider};

#[cfg(feature = "sagemaker")]
pub use chat::{sagemaker, SageMakerProvider};

#[cfg(feature = "snowflake")]
pub use chat::{snowflake, SnowflakeProvider};

#[cfg(feature = "sambanova")]
pub use chat::{sambanova, SambaNovaProvider};

#[cfg(feature = "fireworks")]
pub use chat::{fireworks, FireworksProvider};

#[cfg(feature = "deepseek")]
pub use chat::{deepseek, DeepSeekProvider};

#[cfg(feature = "aleph-alpha")]
pub use chat::{aleph_alpha, AlephAlphaProvider};

#[cfg(feature = "nlp-cloud")]
pub use chat::{nlp_cloud, NlpCloudProvider};

#[cfg(feature = "yandex")]
pub use chat::{yandex, YandexProvider};

#[cfg(feature = "gigachat")]
pub use chat::{gigachat, GigaChatProvider};

#[cfg(feature = "clova")]
pub use chat::{clova, ClovaProvider};

#[cfg(feature = "maritaca")]
pub use chat::{maritaca, MaritacaProvider};

#[cfg(feature = "writer")]
pub use chat::{writer, WriterProvider};

#[cfg(feature = "perplexity")]
pub use chat::perplexity::{
    Citation, PerplexityModelInfo, PerplexityProvider, PerplexitySearchMode,
    SearchAugmentedResponse,
};

#[cfg(feature = "baidu")]
pub use chat::baidu::{ApiVersion, BaiduModelInfo, BaiduProvider};

#[cfg(feature = "alibaba")]
pub use chat::alibaba::{AlibabaModelInfo, AlibabaProvider, ModelSpecialization};

#[cfg(feature = "vllm")]
pub use chat::vllm::{SchedulingPolicy, ServerStats, VLLMProvider};

#[cfg(feature = "oracle")]
pub use chat::oracle::{DeploymentType, OracleEndpointConfig, OracleModelInfo, OracleOCIProvider};

#[cfg(feature = "sap")]
pub use chat::sap::{IntegrationType, SAPConsumptionPlan, SAPGenerativeAIProvider, SAPModelInfo};

// Image providers
#[cfg(feature = "stability")]
pub use image::{stability, StabilityProvider};

#[cfg(feature = "fal")]
pub use image::{fal, FalProvider};

#[cfg(feature = "recraft")]
pub use image::{recraft, RecraftProvider};

#[cfg(feature = "runwayml")]
pub use image::{runwayml, RunwayMLProvider};

// Audio providers
#[cfg(feature = "deepgram")]
pub use audio::{deepgram, DeepgramProvider};

#[cfg(feature = "elevenlabs")]
pub use audio::{elevenlabs, ElevenLabsProvider};

#[cfg(feature = "assemblyai")]
pub use audio::assemblyai::{AssemblyAIProvider, AudioLanguage, TranscriptionConfig};

// Embedding providers
#[cfg(feature = "voyage")]
pub use embedding::{voyage, VoyageProvider};

#[cfg(feature = "jina")]
pub use embedding::{jina, JinaProvider};

#[cfg(feature = "mistral-embeddings")]
pub use embedding::mistral_embeddings::{EmbeddingData, MistralEmbeddingsProvider};

// Specialized providers
#[cfg(feature = "openai-realtime")]
pub use specialized::openai_realtime::{
    RealtimeProvider, RealtimeSession, ServerEvent, SessionConfig,
};

// ========== Phase 2: Additional Tier 1 Providers ==========
// Poe and Gradient are implemented as OpenAI-compatible providers
// in src/providers/chat/openai_compatible.rs
