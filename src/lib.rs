//! # ModelSuite - Unified LLM API for Rust
//!
//! ModelSuite is a unified interface for interacting with multiple LLM providers,
//! written in pure Rust with Python and TypeScript bindings.
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
//! use modelsuite::{ModelSuiteClient, Message, CompletionRequest};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Create client with providers from environment
//!     let client = ModelSuiteClient::builder()
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
//! use modelsuite::{ToolDefinition, ToolBuilder};
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
pub mod circuit_breaker;
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
pub mod specialized;
pub mod stream;
pub mod streaming_multiplexer;
pub mod templates;
pub mod tenant;
pub mod tools;
pub mod types;
pub mod video;

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
pub use circuit_breaker::{CircuitBreaker, CircuitBreakerConfig, CircuitState, HealthMetrics};
pub use client::{ClientBuilder, ModelSuiteClient};
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
pub use specialized::{
    get_moderation_model_info, get_ranking_model_info, ClassificationExample,
    ClassificationPrediction, ClassificationProvider, ClassificationRequest,
    ClassificationResponse, ModerationCategories, ModerationInput, ModerationModelInfo,
    ModerationProvider, ModerationRequest, ModerationResponse, ModerationScores, RankedDocument,
    RankingMeta, RankingModelInfo, RankingProvider, RankingRequest, RankingResponse,
    MODERATION_MODELS, RANKING_MODELS,
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
    StructuredOutputType, ThinkingConfig, ThinkingEffort, ThinkingType, TokenCountRequest,
    TokenCountResult, Usage,
};
pub use video::{
    get_video_model_info, get_video_models_by_provider, CameraMotion, VideoGenerationRequest,
    VideoGenerationResponse, VideoInput, VideoJobStatus, VideoModelInfo, VideoProvider,
    VideoResolution, VIDEO_MODELS,
};

// Re-export providers
#[cfg(feature = "anthropic")]
pub use providers::AnthropicProvider;

#[cfg(feature = "openai")]
pub use providers::OpenAIProvider;

#[cfg(feature = "azure")]
pub use providers::chat::azure::{AzureConfig, AzureOpenAIProvider};

#[cfg(feature = "bedrock")]
pub use providers::chat::bedrock::{BedrockBuilder, BedrockConfig, BedrockProvider};

#[cfg(feature = "openai-compatible")]
pub use providers::chat::openai_compatible::{
    known_providers, OpenAICompatibleProvider, ProviderInfo,
};

#[cfg(feature = "google")]
pub use providers::chat::google::GoogleProvider;

#[cfg(feature = "vertex")]
pub use providers::chat::vertex::{VertexConfig, VertexProvider};

#[cfg(feature = "cohere")]
pub use providers::chat::cohere::CohereProvider;

#[cfg(feature = "ai21")]
pub use providers::chat::ai21::AI21Provider;

#[cfg(feature = "huggingface")]
pub use providers::chat::huggingface::HuggingFaceProvider;

#[cfg(feature = "replicate")]
pub use providers::chat::replicate::ReplicateProvider;

#[cfg(feature = "baseten")]
pub use providers::chat::baseten::BasetenProvider;

#[cfg(feature = "runpod")]
pub use providers::chat::runpod::RunPodProvider;

#[cfg(feature = "cloudflare")]
pub use providers::chat::cloudflare::CloudflareProvider;

#[cfg(feature = "watsonx")]
pub use providers::chat::watsonx::WatsonxProvider;

#[cfg(feature = "databricks")]
pub use providers::chat::databricks::DatabricksProvider;

#[cfg(feature = "cerebras")]
pub use providers::chat::cerebras::CerebrasProvider;

#[cfg(feature = "sambanova")]
pub use providers::chat::sambanova::SambaNovaProvider;

#[cfg(feature = "fireworks")]
pub use providers::chat::fireworks::FireworksProvider;

#[cfg(feature = "deepseek")]
pub use providers::chat::deepseek::DeepSeekProvider;

#[cfg(feature = "mistral-embeddings")]
pub use providers::embedding::mistral_embeddings::{EmbeddingData, MistralEmbeddingsProvider};

#[cfg(feature = "vllm")]
pub use providers::chat::vllm::{SchedulingPolicy, ServerStats, VLLMProvider};

#[cfg(feature = "perplexity")]
pub use providers::chat::perplexity::{
    Citation, PerplexityModelInfo, PerplexityProvider, PerplexitySearchMode,
    SearchAugmentedResponse,
};

#[cfg(feature = "baidu")]
pub use providers::chat::baidu::{ApiVersion, BaiduModelInfo, BaiduProvider};

#[cfg(feature = "alibaba")]
pub use providers::chat::alibaba::{AlibabaModelInfo, AlibabaProvider, ModelSpecialization};

#[cfg(feature = "assemblyai")]
pub use providers::audio::assemblyai::{AssemblyAIProvider, AudioLanguage, TranscriptionConfig};

#[cfg(feature = "oracle")]
pub use providers::chat::oracle::{
    DeploymentType, OracleEndpointConfig, OracleModelInfo, OracleOCIProvider,
};

#[cfg(feature = "sap")]
pub use providers::chat::sap::{
    IntegrationType, SAPConsumptionPlan, SAPGenerativeAIProvider, SAPModelInfo,
};

// Additional providers - Tier 0 (Core)
#[cfg(feature = "openrouter")]
pub use providers::chat::openrouter::OpenRouterProvider;

#[cfg(feature = "ollama")]
pub use providers::chat::ollama::OllamaProvider;

#[cfg(feature = "groq")]
pub use providers::chat::groq::GroqProvider;

#[cfg(feature = "mistral")]
pub use providers::chat::mistral::{MistralConfig, MistralProvider, MistralRegion};

// Additional providers - Enterprise
#[cfg(feature = "datarobot")]
pub use providers::chat::datarobot::DataRobotProvider;

#[cfg(feature = "sagemaker")]
pub use providers::chat::sagemaker::SageMakerProvider;

#[cfg(feature = "snowflake")]
pub use providers::chat::snowflake::SnowflakeProvider;

#[cfg(feature = "aleph-alpha")]
pub use providers::chat::aleph_alpha::AlephAlphaProvider;

#[cfg(feature = "nlp-cloud")]
pub use providers::chat::nlp_cloud::NlpCloudProvider;

#[cfg(feature = "writer")]
pub use providers::chat::writer::WriterProvider;

// Additional providers - Regional
#[cfg(feature = "yandex")]
pub use providers::chat::yandex::YandexProvider;

#[cfg(feature = "gigachat")]
pub use providers::chat::gigachat::GigaChatProvider;

#[cfg(feature = "clova")]
pub use providers::chat::clova::ClovaProvider;

#[cfg(feature = "maritaca")]
pub use providers::chat::maritaca::MaritacaProvider;

// Additional providers - Inference
#[cfg(feature = "xai")]
pub use providers::chat::xai::XAIProvider;

#[cfg(feature = "deepinfra")]
pub use providers::chat::deepinfra::DeepInfraProvider;

#[cfg(feature = "nvidia-nim")]
pub use providers::chat::nvidia_nim::NvidiaNIMProvider;

#[cfg(feature = "anyscale")]
pub use providers::chat::anyscale::AnyscaleProvider;

#[cfg(feature = "github")]
pub use providers::chat::github_models::GitHubModelsProvider;

#[cfg(feature = "friendli")]
pub use providers::chat::friendli::FriendliProvider;

#[cfg(feature = "hyperbolic")]
pub use providers::chat::hyperbolic::HyperbolicProvider;

#[cfg(feature = "lambda")]
pub use providers::chat::lambda_ai::LambdaProvider;

#[cfg(feature = "novita")]
pub use providers::chat::novita::NovitaProvider;

#[cfg(feature = "nebius")]
pub use providers::chat::nebius::NebiusProvider;

#[cfg(feature = "lepton")]
pub use providers::chat::lepton::LeptonProvider;

#[cfg(feature = "stability")]
pub use providers::chat::stability::StabilityProvider;

#[cfg(feature = "gpt4all")]
pub use providers::chat::gpt4all::GPT4AllProvider;

// Additional providers - Chinese
#[cfg(feature = "minimax")]
pub use providers::chat::minimax::MiniMaxProvider;

#[cfg(feature = "moonshot")]
pub use providers::chat::moonshot::MoonshotProvider;

#[cfg(feature = "zhipu")]
pub use providers::chat::zhipu::ZhipuProvider;

#[cfg(feature = "volcengine")]
pub use providers::chat::volcengine::VolcengineProvider;

#[cfg(feature = "baichuan")]
pub use providers::chat::baichuan_ai::BaichuanProvider;

#[cfg(feature = "stepfun")]
pub use providers::chat::stepfun::StepfunProvider;

#[cfg(feature = "yi")]
pub use providers::chat::yi::YiProvider;

#[cfg(feature = "spark")]
pub use providers::chat::spark::SparkProvider;

// Additional providers - Local/Self-Hosted
#[cfg(feature = "lm-studio")]
pub use providers::chat::lm_studio::LMStudioProvider;

#[cfg(feature = "llamafile")]
pub use providers::chat::llamafile::LlamafileProvider;

#[cfg(feature = "xinference")]
pub use providers::chat::xinference::XinferenceProvider;

#[cfg(feature = "localai")]
pub use providers::chat::localai::LocalAIProvider;

#[cfg(feature = "jan")]
pub use providers::chat::jan::JanProvider;

#[cfg(feature = "petals")]
pub use providers::chat::petals::PetalsProvider;

#[cfg(feature = "triton")]
pub use providers::chat::triton::TritonProvider;

#[cfg(feature = "tgi")]
pub use providers::chat::tgi::TGIProvider;

// Additional providers - Enterprise/Specialized
#[cfg(feature = "predibase")]
pub use providers::chat::predibase::PredibaseProvider;

#[cfg(feature = "octoai")]
pub use providers::chat::octoai::OctoAIProvider;

#[cfg(feature = "featherless")]
pub use providers::chat::featherless::FeatherlessProvider;

#[cfg(feature = "ovhcloud")]
pub use providers::chat::ovhcloud::OVHCloudProvider;

#[cfg(feature = "scaleway")]
pub use providers::chat::scaleway::ScalewayProvider;

#[cfg(feature = "crusoe")]
pub use providers::chat::crusoe::CrusoeProvider;

#[cfg(feature = "cerebrium")]
pub use providers::chat::cerebrium::CerebriumProvider;

#[cfg(feature = "lightning")]
pub use providers::chat::lightning::LightningProvider;

#[cfg(feature = "runwayml")]
pub use providers::chat::runwayml::RunwayMLProvider;

// Additional providers - Asian Regional
#[cfg(feature = "naver")]
pub use providers::chat::naver::NaverProvider;

#[cfg(feature = "kakao")]
pub use providers::chat::kakao::KakaoProvider;

#[cfg(feature = "lg-exaone")]
pub use providers::chat::lg_exaone::LGExaoneProvider;

#[cfg(feature = "plamo")]
pub use providers::chat::plamo::PLaMoProvider;

#[cfg(feature = "sarvam")]
pub use providers::chat::sarvam::SarvamProvider;

#[cfg(feature = "krutrim")]
pub use providers::chat::krutrim::KrutrimProvider;

#[cfg(feature = "ntt")]
pub use providers::chat::ntt::NTTProvider;

#[cfg(feature = "softbank")]
pub use providers::chat::softbank::SoftBankProvider;

// Additional providers - European Sovereign AI
#[cfg(feature = "ionos")]
pub use providers::chat::ionos::IONOSProvider;

#[cfg(feature = "tilde")]
pub use providers::chat::tilde::TildeProvider;

#[cfg(feature = "silo-ai")]
pub use providers::chat::silo_ai::SiloAIProvider;

#[cfg(feature = "swiss-ai")]
pub use providers::chat::swiss_ai::SwissAIProvider;

// Additional providers - Router/Gateway/Meta
#[cfg(feature = "unify")]
pub use providers::chat::unify::UnifyProvider;

#[cfg(feature = "martian")]
pub use providers::chat::martian::MartianProvider;

#[cfg(feature = "portkey")]
pub use providers::chat::portkey::PortkeyProvider;

#[cfg(feature = "helicone")]
pub use providers::chat::helicone::HeliconeProvider;

#[cfg(feature = "siliconflow")]
pub use providers::chat::siliconflow::SiliconFlowProvider;

// Additional providers - Video AI
#[cfg(feature = "pika")]
pub use providers::chat::pika::PikaProvider;

#[cfg(feature = "luma")]
pub use providers::chat::luma::LumaProvider;

#[cfg(feature = "kling")]
pub use providers::chat::kling::KlingProvider;

#[cfg(feature = "heygen")]
pub use providers::chat::heygen::HeyGenProvider;

#[cfg(feature = "did")]
pub use providers::chat::did::DIDProvider;

#[cfg(feature = "twelve-labs")]
pub use providers::chat::twelve_labs::TwelveLabsProvider;

// Additional providers - Audio AI
#[cfg(feature = "rev")]
pub use providers::chat::rev::RevProvider;

#[cfg(feature = "speechmatics")]
pub use providers::chat::speechmatics::SpeechmaticsProvider;

#[cfg(feature = "playht")]
pub use providers::chat::playht::PlayHTProvider;

#[cfg(feature = "resemble")]
pub use providers::chat::resemble::ResembleProvider;

// Additional providers - Image AI
#[cfg(feature = "leonardo")]
pub use providers::chat::leonardo::LeonardoProvider;

#[cfg(feature = "ideogram")]
pub use providers::chat::ideogram::IdeogramProvider;

#[cfg(feature = "black-forest-labs")]
pub use providers::chat::black_forest_labs::BlackForestLabsProvider;

#[cfg(feature = "clarifai")]
pub use providers::chat::clarifai::ClarifaiProvider;

#[cfg(feature = "fal")]
pub use providers::chat::fal::FalProvider;

// Additional providers - Infrastructure
#[cfg(feature = "modal")]
pub use providers::chat::modal::ModalProvider;

#[cfg(feature = "coreweave")]
pub use providers::chat::coreweave::CoreWeaveProvider;

#[cfg(feature = "tensordock")]
pub use providers::chat::tensordock::TensorDockProvider;

#[cfg(feature = "beam")]
pub use providers::chat::beam::BeamProvider;

#[cfg(feature = "vastai")]
pub use providers::chat::vastai::VastAIProvider;

// Additional providers - Emerging Startups
#[cfg(feature = "nscale")]
pub use providers::chat::nscale::NscaleProvider;

#[cfg(feature = "runware")]
pub use providers::chat::runware::RunwareProvider;

#[cfg(feature = "ai71")]
pub use providers::chat::ai71::AI71Provider;

// Embedding providers
#[cfg(feature = "voyage")]
pub use providers::embedding::voyage::VoyageProvider;

#[cfg(feature = "jina")]
pub use providers::embedding::jina::JinaProvider;

// Audio providers
#[cfg(feature = "deepgram")]
pub use providers::audio::deepgram::DeepgramProvider;

#[cfg(feature = "elevenlabs")]
pub use providers::audio::elevenlabs::ElevenLabsProvider;

// Contingent providers (pending API access)
pub use providers::audio::GrokRealtimeProvider;
pub use providers::chat::{ChatLawProvider, LatamGPTProvider, LightOnProvider};
