//! # LLMKit - Unified LLM API for Rust
//!
//! LLMKit is a unified interface for interacting with multiple LLM providers,
//! similar to LiteLLM but written in pure Rust.
//!
//! ## Features
//!
//! - **Unified API**: Single interface for Anthropic, OpenAI, and many other providers
//! - **Streaming**: First-class support for streaming responses
//! - **Tool Calling**: Consistent tool/function calling across providers
//! - **Type-Safe**: Full Rust type safety with compile-time provider selection
//! - **Feature Flags**: Only compile the providers you need
//!
//! ## Quick Start
//!
//! ```ignore
//! use llmkit::{LLMKitClient, Message, CompletionRequest};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create client with providers from environment
//!     let client = LLMKitClient::builder()
//!         .with_anthropic_from_env()
//!         .with_openai_from_env()
//!         .build()?;
//!
//!     // Make a request - provider auto-detected from model name
//!     let request = CompletionRequest::new(
//!         "claude-sonnet-4-20250514",
//!         vec![Message::user("Hello, how are you?")]
//!     );
//!
//!     let response = client.complete(request).await?;
//!     println!("{}", response.text_content());
//!
//!     Ok(())
//! }
//! ```
//!
//! ## Supported Providers
//!
//! | Provider | Feature Flag | Models |
//! |----------|--------------|--------|
//! | Anthropic | `anthropic` | Claude 3, Claude 3.5, Claude 4 |
//! | OpenAI | `openai` | GPT-4o, GPT-4, o1 |
//! | OpenRouter | `openrouter` | 100+ models |
//! | Ollama | `ollama` | Local models |
//! | Azure OpenAI | `azure` | Azure-hosted GPT models |
//! | AWS Bedrock | `bedrock` | Claude, Titan, etc. |
//! | Google Vertex AI | `vertex` | Gemini |
//! | Mistral | `mistral` | Mistral, Mixtral |
//! | Groq | `groq` | Fast inference |
//!
//! ## Streaming Example
//!
//! ```ignore
//! use futures::StreamExt;
//!
//! let request = CompletionRequest::new("gpt-4o", vec![Message::user("Write a story")])
//!     .with_streaming();
//!
//! let mut stream = client.complete_stream(request).await?;
//!
//! while let Some(chunk) = stream.next().await {
//!     if let Ok(chunk) = chunk {
//!         if let Some(ContentDelta::Text { text }) = chunk.delta {
//!             print!("{}", text);
//!         }
//!     }
//! }
//! ```
//!
//! ## Tool Calling Example
//!
//! ```ignore
//! use llmkit::{ToolDefinition, ToolBuilder};
//!
//! let tool = ToolBuilder::new("get_weather")
//!     .description("Get the current weather for a location")
//!     .string_param("location", "The city name", true)
//!     .build();
//!
//! let request = CompletionRequest::new("claude-sonnet-4-20250514", vec![Message::user("What's the weather in Paris?")])
//!     .with_tools(vec![tool]);
//!
//! let response = client.complete(request).await?;
//!
//! if response.has_tool_use() {
//!     for tool_use in response.tool_uses() {
//!         // Handle tool call
//!     }
//! }
//! ```

pub mod audio;
pub mod cache;
pub mod client;
pub mod embedding;
pub mod error;
pub mod failover;
pub mod guardrails;
pub mod health;
pub mod image;
pub mod metering;
pub mod models;
pub mod observability;
pub mod pool;
pub mod provider;
pub mod providers;
pub mod rate_limiter;
pub mod retry;
pub mod smart_router;
pub mod stream;
pub mod streaming_multiplexer;
pub mod templates;
pub mod tenant;
pub mod tools;
pub mod types;

// Re-export main types for convenience
pub use audio::{
    get_audio_model_info, AudioFormat, AudioInput, AudioModelInfo, AudioModelType, SpeechProvider,
    SpeechRequest, SpeechResponse, TimestampGranularity, TranscriptFormat, TranscriptSegment,
    TranscriptWord, TranscriptionProvider, TranscriptionRequest, TranscriptionResponse, VoiceInfo,
    AUDIO_MODELS,
};
pub use cache::{
    CacheBackend, CacheConfig, CacheKeyBuilder, CacheStats, CachedResponse, CachingProvider,
    InMemoryCache,
};
pub use client::{ClientBuilder, LLMKitClient};
pub use embedding::{
    get_embedding_model_info, get_embedding_models_by_provider, Embedding, EmbeddingInput,
    EmbeddingInputType, EmbeddingModelInfo, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse,
    EmbeddingUsage, EncodingFormat, EMBEDDING_MODELS,
};
pub use error::{Error, Result};
pub use failover::{FailoverConfig, FailoverProvider, FailoverTrigger, FallbackProvider};
pub use guardrails::{
    Finding, FindingType, GuardedProvider, Guardrails, GuardrailsBuilder, GuardrailsConfig,
    GuardrailsResult, PiiPattern, PiiType, SecretPattern, SecretType, Severity,
};
pub use health::{
    DeploymentStatus, HealthCheckResult, HealthCheckType, HealthChecker, HealthCheckerHandle,
    PoolHealthStatus,
};
pub use image::{
    get_image_model_info, AsyncImageProvider, GeneratedImage, ImageEditRequest, ImageFormat,
    ImageGenerationRequest, ImageGenerationResponse, ImageInput, ImageModelInfo, ImageProvider,
    ImageQuality, ImageSize, ImageStyle, ImageVariationRequest, JobId, JobStatus, IMAGE_MODELS,
};
pub use metering::{
    CostTracker, InMemoryMeteringSink, MeteringProvider, MeteringSink, ModelStats, TenantStats,
    UsageFilter, UsageRecord, UsageStats,
};
pub use models::{
    get_all_models, get_available_models, get_cheapest_model, get_classifier_models,
    get_current_models, get_model_info, get_models_by_provider, get_models_with_capability,
    get_registry_stats, list_providers, supports_structured_output, ModelBenchmarks,
    ModelCapabilities, ModelInfo, ModelPricing, ModelStatus, Provider as ProviderKind,
    RegistryStats,
};
pub use observability::{
    MetricsRecorder, MetricsSnapshot, Observability, ObservabilityConfig, RequestSpan,
    TracingContext,
};
pub use pool::{
    DeploymentConfig, DeploymentHealth, HealthCheckConfig, ProviderPool, ProviderPoolBuilder,
    RoutingStrategy,
};
pub use provider::{ModelInfo as ProviderModelInfo, Provider, ProviderConfig};
pub use rate_limiter::{RateLimiter, TokenBucketConfig};
pub use retry::{ProviderExt, RetryConfig, RetryingProvider};
pub use smart_router::{
    Optimization, ProviderMetrics, RouterProviderConfig, RouterStats, RoutingDecision, SmartRouter,
    SmartRouterBuilder,
};
pub use stream::{collect_stream, CollectingStream};
pub use streaming_multiplexer::{MultiplexedStream, MultiplexerStats, StreamingMultiplexer};
pub use templates::{
    patterns as template_patterns, PromptTemplate, TemplateRegistry, TemplatedRequestBuilder,
};
pub use tenant::{
    CostLimitConfig, CostLimitExceeded, CostLimitType, RateLimitConfig, RateLimitExceeded,
    RateLimitType, TenantConfig, TenantError, TenantId, TenantManager, TenantProvider,
    TenantUsageStats,
};
pub use tools::{ToolBuilder, ToolChoice, ToolDefinition};
pub use types::{
    BatchError, BatchJob, BatchRequest, BatchRequestCounts, BatchResult, BatchStatus,
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, JsonSchemaDefinition,
    Message, Role, StopReason, StreamChunk, StreamEventType, StructuredOutput,
    StructuredOutputType, TokenCountRequest, TokenCountResult, Usage,
};

// Re-export providers
#[cfg(feature = "anthropic")]
pub use providers::AnthropicProvider;

#[cfg(feature = "openai")]
pub use providers::OpenAIProvider;

#[cfg(feature = "azure")]
pub use providers::azure::{AzureConfig, AzureOpenAIProvider};

#[cfg(feature = "bedrock")]
pub use providers::bedrock::{BedrockBuilder, BedrockConfig, BedrockProvider, ModelFamily};

#[cfg(feature = "openai-compatible")]
pub use providers::openai_compatible::{known_providers, OpenAICompatibleProvider, ProviderInfo};

#[cfg(feature = "google")]
pub use providers::google::GoogleProvider;

#[cfg(feature = "vertex")]
pub use providers::vertex::{VertexConfig, VertexProvider};

#[cfg(feature = "cohere")]
pub use providers::cohere::CohereProvider;

#[cfg(feature = "ai21")]
pub use providers::ai21::AI21Provider;

#[cfg(feature = "huggingface")]
pub use providers::huggingface::HuggingFaceProvider;

#[cfg(feature = "replicate")]
pub use providers::replicate::ReplicateProvider;

#[cfg(feature = "baseten")]
pub use providers::baseten::BasetenProvider;

#[cfg(feature = "runpod")]
pub use providers::runpod::RunPodProvider;

#[cfg(feature = "cloudflare")]
pub use providers::cloudflare::CloudflareProvider;

#[cfg(feature = "watsonx")]
pub use providers::watsonx::WatsonxProvider;

#[cfg(feature = "databricks")]
pub use providers::databricks::DatabricksProvider;

#[cfg(feature = "cerebras")]
pub use providers::cerebras::CerebrasProvider;

#[cfg(feature = "sambanova")]
pub use providers::sambanova::SambaNovaProvider;

#[cfg(feature = "fireworks")]
pub use providers::fireworks::FireworksProvider;

#[cfg(feature = "deepseek")]
pub use providers::deepseek::DeepSeekProvider;
