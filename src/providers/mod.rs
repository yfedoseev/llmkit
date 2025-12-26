//! LLM Provider implementations.
//!
//! This module contains implementations for various LLM providers.
//! Each provider is feature-gated to minimize binary size.

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "openrouter")]
pub mod openrouter;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "groq")]
pub mod groq;

#[cfg(feature = "mistral")]
pub mod mistral;

#[cfg(feature = "azure")]
pub mod azure;

#[cfg(feature = "bedrock")]
pub mod bedrock;

#[cfg(feature = "openai-compatible")]
pub mod openai_compatible;

#[cfg(feature = "google")]
pub mod google;

#[cfg(feature = "vertex")]
pub mod vertex;

#[cfg(feature = "cohere")]
pub mod cohere;

#[cfg(feature = "ai21")]
pub mod ai21;

#[cfg(feature = "huggingface")]
pub mod huggingface;

#[cfg(feature = "replicate")]
pub mod replicate;

#[cfg(feature = "baseten")]
pub mod baseten;

#[cfg(feature = "runpod")]
pub mod runpod;

#[cfg(feature = "cloudflare")]
pub mod cloudflare;

#[cfg(feature = "watsonx")]
pub mod watsonx;

#[cfg(feature = "databricks")]
pub mod databricks;

#[cfg(feature = "cerebras")]
pub mod cerebras;

#[cfg(feature = "sambanova")]
pub mod sambanova;

#[cfg(feature = "fireworks")]
pub mod fireworks;

#[cfg(feature = "deepseek")]
pub mod deepseek;

#[cfg(feature = "aleph-alpha")]
pub mod aleph_alpha;

#[cfg(feature = "nlp-cloud")]
pub mod nlp_cloud;

#[cfg(feature = "voyage")]
pub mod voyage;

#[cfg(feature = "jina")]
pub mod jina;

#[cfg(feature = "fal")]
pub mod fal;

#[cfg(feature = "deepgram")]
pub mod deepgram;

#[cfg(feature = "elevenlabs")]
pub mod elevenlabs;

#[cfg(feature = "yandex")]
pub mod yandex;

#[cfg(feature = "gigachat")]
pub mod gigachat;

#[cfg(feature = "clova")]
pub mod clova;

#[cfg(feature = "maritaca")]
pub mod maritaca;

#[cfg(feature = "writer")]
pub mod writer;

// Re-exports for convenience
#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;

#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;

#[cfg(feature = "openrouter")]
pub use openrouter::OpenRouterProvider;

#[cfg(feature = "ollama")]
pub use ollama::OllamaProvider;

#[cfg(feature = "groq")]
pub use groq::GroqProvider;

#[cfg(feature = "mistral")]
pub use mistral::MistralProvider;

#[cfg(feature = "azure")]
pub use azure::{AzureConfig, AzureOpenAIProvider};

#[cfg(feature = "bedrock")]
pub use bedrock::{BedrockBuilder, BedrockConfig, BedrockProvider, ModelFamily};

#[cfg(feature = "openai-compatible")]
pub use openai_compatible::{known_providers, OpenAICompatibleProvider, ProviderInfo};

#[cfg(feature = "google")]
pub use google::GoogleProvider;

#[cfg(feature = "vertex")]
pub use vertex::{VertexConfig, VertexProvider};

#[cfg(feature = "cohere")]
pub use cohere::CohereProvider;

#[cfg(feature = "ai21")]
pub use ai21::AI21Provider;

#[cfg(feature = "huggingface")]
pub use huggingface::HuggingFaceProvider;

#[cfg(feature = "replicate")]
pub use replicate::ReplicateProvider;

#[cfg(feature = "baseten")]
pub use baseten::BasetenProvider;

#[cfg(feature = "runpod")]
pub use runpod::RunPodProvider;

#[cfg(feature = "cloudflare")]
pub use cloudflare::CloudflareProvider;

#[cfg(feature = "watsonx")]
pub use watsonx::WatsonxProvider;

#[cfg(feature = "databricks")]
pub use databricks::DatabricksProvider;

#[cfg(feature = "cerebras")]
pub use cerebras::CerebrasProvider;

#[cfg(feature = "sambanova")]
pub use sambanova::SambaNovaProvider;

#[cfg(feature = "fireworks")]
pub use fireworks::FireworksProvider;

#[cfg(feature = "deepseek")]
pub use deepseek::DeepSeekProvider;

#[cfg(feature = "aleph-alpha")]
pub use aleph_alpha::AlephAlphaProvider;

#[cfg(feature = "nlp-cloud")]
pub use nlp_cloud::NlpCloudProvider;

#[cfg(feature = "voyage")]
pub use voyage::VoyageProvider;

#[cfg(feature = "jina")]
pub use jina::JinaProvider;

#[cfg(feature = "fal")]
pub use fal::FalProvider;

#[cfg(feature = "deepgram")]
pub use deepgram::DeepgramProvider;

#[cfg(feature = "elevenlabs")]
pub use elevenlabs::ElevenLabsProvider;

#[cfg(feature = "yandex")]
pub use yandex::YandexProvider;

#[cfg(feature = "gigachat")]
pub use gigachat::GigaChatProvider;

#[cfg(feature = "clova")]
pub use clova::ClovaProvider;

#[cfg(feature = "maritaca")]
pub use maritaca::MaritacaProvider;

#[cfg(feature = "writer")]
pub use writer::WriterProvider;
