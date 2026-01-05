//! Model Registry bindings for JavaScript/TypeScript.
//!
//! Provides access to the ModelSuite model registry with information about
//! pricing, capabilities, and benchmarks for all supported models.

use modelsuite::models::{
    self, ModelBenchmarks, ModelCapabilities, ModelInfo, ModelPricing, ModelStatus, Provider,
};
use napi_derive::napi;

// ============================================================================
// ENUMS
// ============================================================================

/// LLM Provider identifier.
#[napi(string_enum)]
pub enum JsProvider {
    // Core providers
    Anthropic,
    OpenAI,
    Google,
    Mistral,
    Groq,
    DeepSeek,
    Cohere,
    Bedrock,
    AzureOpenAI,
    VertexAI,
    TogetherAI,
    OpenRouter,
    Cerebras,
    SambaNova,
    Fireworks,
    AI21,
    HuggingFace,
    Replicate,
    Cloudflare,
    Databricks,
    // Regional providers
    Writer,
    Maritaca,
    Clova,
    Yandex,
    GigaChat,
    Upstage,
    SeaLion,
    // Additional providers
    Alibaba,
    AlephAlpha,
    Baidu,
    Baseten,
    ChatLaw,
    DataRobot,
    LatamGPT,
    LightOn,
    NLPCloud,
    Oracle,
    Perplexity,
    RunPod,
    Sagemaker,
    Sap,
    Snowflake,
    Vllm,
    WatsonX,
    Xai,
    DeepInfra,
    NvidiaNIM,
    // Tier 1 - High Priority Inference
    Ollama,
    Anyscale,
    GitHub,
    FriendliAI,
    Hyperbolic,
    Lambda,
    Novita,
    Nebius,
    Lepton,
    Stability,
    Voyage,
    Jina,
    Deepgram,
    ElevenLabs,
    GPT4All,
    // Tier 2 - Chinese Providers
    MiniMax,
    Moonshot,
    Zhipu,
    Volcengine,
    Baichuan,
    Stepfun,
    Yi,
    Spark,
    // Tier 3 - Local/Self-Hosted
    LMStudio,
    Llamafile,
    Xinference,
    LocalAI,
    Jan,
    Petals,
    Triton,
    Tgi,
    // Tier 4 - Enterprise/Specialized
    Predibase,
    OctoAI,
    Featherless,
    OVHCloud,
    Scaleway,
    Crusoe,
    Cerebrium,
    Lightning,
    AssemblyAI,
    RunwayML,
    // Tier 5 - Asian Regional Providers
    Naver,
    Kakao,
    LGExaone,
    PLaMo,
    Sarvam,
    Krutrim,
    Ntt,
    SoftBank,
    // Tier 6 - European Sovereign AI
    Ionos,
    Tilde,
    SiloAI,
    SwissAI,
    // Tier 7 - Router/Gateway/Meta Providers
    Unify,
    Martian,
    Portkey,
    Helicone,
    SiliconFlow,
    // Tier 8 - Video AI Providers
    Pika,
    Luma,
    Kling,
    HeyGen,
    Did,
    TwelveLabs,
    // Tier 9 - Audio AI Providers
    Rev,
    Speechmatics,
    PlayHT,
    Resemble,
    // Tier 10 - Image AI Providers
    Leonardo,
    Ideogram,
    BlackForestLabs,
    Clarifai,
    Fal,
    // Tier 11 - Infrastructure Providers
    Modal,
    CoreWeave,
    TensorDock,
    Beam,
    VastAI,
    // Tier 12 - Emerging Startups
    Nscale,
    Runware,
    AI71,
    // Local/Custom
    Local,
    Custom,
}

impl From<Provider> for JsProvider {
    fn from(p: Provider) -> Self {
        match p {
            Provider::Anthropic => JsProvider::Anthropic,
            Provider::OpenAI => JsProvider::OpenAI,
            Provider::Google => JsProvider::Google,
            Provider::Mistral => JsProvider::Mistral,
            Provider::Groq => JsProvider::Groq,
            Provider::DeepSeek => JsProvider::DeepSeek,
            Provider::Cohere => JsProvider::Cohere,
            Provider::Bedrock => JsProvider::Bedrock,
            Provider::AzureOpenAI => JsProvider::AzureOpenAI,
            Provider::VertexAI => JsProvider::VertexAI,
            Provider::TogetherAI => JsProvider::TogetherAI,
            Provider::OpenRouter => JsProvider::OpenRouter,
            Provider::Cerebras => JsProvider::Cerebras,
            Provider::SambaNova => JsProvider::SambaNova,
            Provider::Fireworks => JsProvider::Fireworks,
            Provider::AI21 => JsProvider::AI21,
            Provider::HuggingFace => JsProvider::HuggingFace,
            Provider::Replicate => JsProvider::Replicate,
            Provider::Cloudflare => JsProvider::Cloudflare,
            Provider::Databricks => JsProvider::Databricks,
            Provider::Writer => JsProvider::Writer,
            Provider::Maritaca => JsProvider::Maritaca,
            Provider::Clova => JsProvider::Clova,
            Provider::Yandex => JsProvider::Yandex,
            Provider::GigaChat => JsProvider::GigaChat,
            Provider::Upstage => JsProvider::Upstage,
            Provider::SeaLion => JsProvider::SeaLion,
            Provider::Alibaba => JsProvider::Alibaba,
            Provider::AlephAlpha => JsProvider::AlephAlpha,
            Provider::Baidu => JsProvider::Baidu,
            Provider::Baseten => JsProvider::Baseten,
            Provider::ChatLaw => JsProvider::ChatLaw,
            Provider::DataRobot => JsProvider::DataRobot,
            Provider::LatamGPT => JsProvider::LatamGPT,
            Provider::LightOn => JsProvider::LightOn,
            Provider::NLPCloud => JsProvider::NLPCloud,
            Provider::Oracle => JsProvider::Oracle,
            Provider::Perplexity => JsProvider::Perplexity,
            Provider::RunPod => JsProvider::RunPod,
            Provider::Sagemaker => JsProvider::Sagemaker,
            Provider::Sap => JsProvider::Sap,
            Provider::Snowflake => JsProvider::Snowflake,
            Provider::Vllm => JsProvider::Vllm,
            Provider::WatsonX => JsProvider::WatsonX,
            Provider::Xai => JsProvider::Xai,
            Provider::DeepInfra => JsProvider::DeepInfra,
            Provider::NvidiaNIM => JsProvider::NvidiaNIM,
            Provider::Ollama => JsProvider::Ollama,
            Provider::Anyscale => JsProvider::Anyscale,
            Provider::GitHub => JsProvider::GitHub,
            Provider::FriendliAI => JsProvider::FriendliAI,
            Provider::Hyperbolic => JsProvider::Hyperbolic,
            Provider::Lambda => JsProvider::Lambda,
            Provider::Novita => JsProvider::Novita,
            Provider::Nebius => JsProvider::Nebius,
            Provider::Lepton => JsProvider::Lepton,
            Provider::Stability => JsProvider::Stability,
            Provider::Voyage => JsProvider::Voyage,
            Provider::Jina => JsProvider::Jina,
            Provider::Deepgram => JsProvider::Deepgram,
            Provider::ElevenLabs => JsProvider::ElevenLabs,
            Provider::GPT4All => JsProvider::GPT4All,
            Provider::MiniMax => JsProvider::MiniMax,
            Provider::Moonshot => JsProvider::Moonshot,
            Provider::Zhipu => JsProvider::Zhipu,
            Provider::Volcengine => JsProvider::Volcengine,
            Provider::Baichuan => JsProvider::Baichuan,
            Provider::Stepfun => JsProvider::Stepfun,
            Provider::Yi => JsProvider::Yi,
            Provider::Spark => JsProvider::Spark,
            Provider::LMStudio => JsProvider::LMStudio,
            Provider::Llamafile => JsProvider::Llamafile,
            Provider::Xinference => JsProvider::Xinference,
            Provider::LocalAI => JsProvider::LocalAI,
            Provider::Jan => JsProvider::Jan,
            Provider::Petals => JsProvider::Petals,
            Provider::Triton => JsProvider::Triton,
            Provider::Tgi => JsProvider::Tgi,
            Provider::Predibase => JsProvider::Predibase,
            Provider::OctoAI => JsProvider::OctoAI,
            Provider::Featherless => JsProvider::Featherless,
            Provider::OVHCloud => JsProvider::OVHCloud,
            Provider::Scaleway => JsProvider::Scaleway,
            Provider::Crusoe => JsProvider::Crusoe,
            Provider::Cerebrium => JsProvider::Cerebrium,
            Provider::Lightning => JsProvider::Lightning,
            Provider::AssemblyAI => JsProvider::AssemblyAI,
            Provider::RunwayML => JsProvider::RunwayML,
            Provider::Naver => JsProvider::Naver,
            Provider::Kakao => JsProvider::Kakao,
            Provider::LGExaone => JsProvider::LGExaone,
            Provider::PLaMo => JsProvider::PLaMo,
            Provider::Sarvam => JsProvider::Sarvam,
            Provider::Krutrim => JsProvider::Krutrim,
            Provider::Ntt => JsProvider::Ntt,
            Provider::SoftBank => JsProvider::SoftBank,
            Provider::Ionos => JsProvider::Ionos,
            Provider::Tilde => JsProvider::Tilde,
            Provider::SiloAI => JsProvider::SiloAI,
            Provider::SwissAI => JsProvider::SwissAI,
            Provider::Unify => JsProvider::Unify,
            Provider::Martian => JsProvider::Martian,
            Provider::Portkey => JsProvider::Portkey,
            Provider::Helicone => JsProvider::Helicone,
            Provider::SiliconFlow => JsProvider::SiliconFlow,
            Provider::Pika => JsProvider::Pika,
            Provider::Luma => JsProvider::Luma,
            Provider::Kling => JsProvider::Kling,
            Provider::HeyGen => JsProvider::HeyGen,
            Provider::Did => JsProvider::Did,
            Provider::TwelveLabs => JsProvider::TwelveLabs,
            Provider::Rev => JsProvider::Rev,
            Provider::Speechmatics => JsProvider::Speechmatics,
            Provider::PlayHT => JsProvider::PlayHT,
            Provider::Resemble => JsProvider::Resemble,
            Provider::Leonardo => JsProvider::Leonardo,
            Provider::Ideogram => JsProvider::Ideogram,
            Provider::BlackForestLabs => JsProvider::BlackForestLabs,
            Provider::Clarifai => JsProvider::Clarifai,
            Provider::Fal => JsProvider::Fal,
            Provider::Modal => JsProvider::Modal,
            Provider::CoreWeave => JsProvider::CoreWeave,
            Provider::TensorDock => JsProvider::TensorDock,
            Provider::Beam => JsProvider::Beam,
            Provider::VastAI => JsProvider::VastAI,
            Provider::Nscale => JsProvider::Nscale,
            Provider::Runware => JsProvider::Runware,
            Provider::AI71 => JsProvider::AI71,
            Provider::Local => JsProvider::Local,
            Provider::Custom => JsProvider::Custom,
        }
    }
}

impl From<JsProvider> for Provider {
    fn from(p: JsProvider) -> Self {
        match p {
            JsProvider::Anthropic => Provider::Anthropic,
            JsProvider::OpenAI => Provider::OpenAI,
            JsProvider::Google => Provider::Google,
            JsProvider::Mistral => Provider::Mistral,
            JsProvider::Groq => Provider::Groq,
            JsProvider::DeepSeek => Provider::DeepSeek,
            JsProvider::Cohere => Provider::Cohere,
            JsProvider::Bedrock => Provider::Bedrock,
            JsProvider::AzureOpenAI => Provider::AzureOpenAI,
            JsProvider::VertexAI => Provider::VertexAI,
            JsProvider::TogetherAI => Provider::TogetherAI,
            JsProvider::OpenRouter => Provider::OpenRouter,
            JsProvider::Cerebras => Provider::Cerebras,
            JsProvider::SambaNova => Provider::SambaNova,
            JsProvider::Fireworks => Provider::Fireworks,
            JsProvider::AI21 => Provider::AI21,
            JsProvider::HuggingFace => Provider::HuggingFace,
            JsProvider::Replicate => Provider::Replicate,
            JsProvider::Cloudflare => Provider::Cloudflare,
            JsProvider::Databricks => Provider::Databricks,
            JsProvider::Writer => Provider::Writer,
            JsProvider::Maritaca => Provider::Maritaca,
            JsProvider::Clova => Provider::Clova,
            JsProvider::Yandex => Provider::Yandex,
            JsProvider::GigaChat => Provider::GigaChat,
            JsProvider::Upstage => Provider::Upstage,
            JsProvider::SeaLion => Provider::SeaLion,
            JsProvider::Alibaba => Provider::Alibaba,
            JsProvider::AlephAlpha => Provider::AlephAlpha,
            JsProvider::Baidu => Provider::Baidu,
            JsProvider::Baseten => Provider::Baseten,
            JsProvider::ChatLaw => Provider::ChatLaw,
            JsProvider::DataRobot => Provider::DataRobot,
            JsProvider::LatamGPT => Provider::LatamGPT,
            JsProvider::LightOn => Provider::LightOn,
            JsProvider::NLPCloud => Provider::NLPCloud,
            JsProvider::Oracle => Provider::Oracle,
            JsProvider::Perplexity => Provider::Perplexity,
            JsProvider::RunPod => Provider::RunPod,
            JsProvider::Sagemaker => Provider::Sagemaker,
            JsProvider::Sap => Provider::Sap,
            JsProvider::Snowflake => Provider::Snowflake,
            JsProvider::Vllm => Provider::Vllm,
            JsProvider::WatsonX => Provider::WatsonX,
            JsProvider::Xai => Provider::Xai,
            JsProvider::DeepInfra => Provider::DeepInfra,
            JsProvider::NvidiaNIM => Provider::NvidiaNIM,
            JsProvider::Ollama => Provider::Ollama,
            JsProvider::Anyscale => Provider::Anyscale,
            JsProvider::GitHub => Provider::GitHub,
            JsProvider::FriendliAI => Provider::FriendliAI,
            JsProvider::Hyperbolic => Provider::Hyperbolic,
            JsProvider::Lambda => Provider::Lambda,
            JsProvider::Novita => Provider::Novita,
            JsProvider::Nebius => Provider::Nebius,
            JsProvider::Lepton => Provider::Lepton,
            JsProvider::Stability => Provider::Stability,
            JsProvider::Voyage => Provider::Voyage,
            JsProvider::Jina => Provider::Jina,
            JsProvider::Deepgram => Provider::Deepgram,
            JsProvider::ElevenLabs => Provider::ElevenLabs,
            JsProvider::GPT4All => Provider::GPT4All,
            JsProvider::MiniMax => Provider::MiniMax,
            JsProvider::Moonshot => Provider::Moonshot,
            JsProvider::Zhipu => Provider::Zhipu,
            JsProvider::Volcengine => Provider::Volcengine,
            JsProvider::Baichuan => Provider::Baichuan,
            JsProvider::Stepfun => Provider::Stepfun,
            JsProvider::Yi => Provider::Yi,
            JsProvider::Spark => Provider::Spark,
            JsProvider::LMStudio => Provider::LMStudio,
            JsProvider::Llamafile => Provider::Llamafile,
            JsProvider::Xinference => Provider::Xinference,
            JsProvider::LocalAI => Provider::LocalAI,
            JsProvider::Jan => Provider::Jan,
            JsProvider::Petals => Provider::Petals,
            JsProvider::Triton => Provider::Triton,
            JsProvider::Tgi => Provider::Tgi,
            JsProvider::Predibase => Provider::Predibase,
            JsProvider::OctoAI => Provider::OctoAI,
            JsProvider::Featherless => Provider::Featherless,
            JsProvider::OVHCloud => Provider::OVHCloud,
            JsProvider::Scaleway => Provider::Scaleway,
            JsProvider::Crusoe => Provider::Crusoe,
            JsProvider::Cerebrium => Provider::Cerebrium,
            JsProvider::Lightning => Provider::Lightning,
            JsProvider::AssemblyAI => Provider::AssemblyAI,
            JsProvider::RunwayML => Provider::RunwayML,
            JsProvider::Naver => Provider::Naver,
            JsProvider::Kakao => Provider::Kakao,
            JsProvider::LGExaone => Provider::LGExaone,
            JsProvider::PLaMo => Provider::PLaMo,
            JsProvider::Sarvam => Provider::Sarvam,
            JsProvider::Krutrim => Provider::Krutrim,
            JsProvider::Ntt => Provider::Ntt,
            JsProvider::SoftBank => Provider::SoftBank,
            JsProvider::Ionos => Provider::Ionos,
            JsProvider::Tilde => Provider::Tilde,
            JsProvider::SiloAI => Provider::SiloAI,
            JsProvider::SwissAI => Provider::SwissAI,
            JsProvider::Unify => Provider::Unify,
            JsProvider::Martian => Provider::Martian,
            JsProvider::Portkey => Provider::Portkey,
            JsProvider::Helicone => Provider::Helicone,
            JsProvider::SiliconFlow => Provider::SiliconFlow,
            JsProvider::Pika => Provider::Pika,
            JsProvider::Luma => Provider::Luma,
            JsProvider::Kling => Provider::Kling,
            JsProvider::HeyGen => Provider::HeyGen,
            JsProvider::Did => Provider::Did,
            JsProvider::TwelveLabs => Provider::TwelveLabs,
            JsProvider::Rev => Provider::Rev,
            JsProvider::Speechmatics => Provider::Speechmatics,
            JsProvider::PlayHT => Provider::PlayHT,
            JsProvider::Resemble => Provider::Resemble,
            JsProvider::Leonardo => Provider::Leonardo,
            JsProvider::Ideogram => Provider::Ideogram,
            JsProvider::BlackForestLabs => Provider::BlackForestLabs,
            JsProvider::Clarifai => Provider::Clarifai,
            JsProvider::Fal => Provider::Fal,
            JsProvider::Modal => Provider::Modal,
            JsProvider::CoreWeave => Provider::CoreWeave,
            JsProvider::TensorDock => Provider::TensorDock,
            JsProvider::Beam => Provider::Beam,
            JsProvider::VastAI => Provider::VastAI,
            JsProvider::Nscale => Provider::Nscale,
            JsProvider::Runware => Provider::Runware,
            JsProvider::AI71 => Provider::AI71,
            JsProvider::Local => Provider::Local,
            JsProvider::Custom => Provider::Custom,
        }
    }
}

/// Model availability status.
#[napi(string_enum)]
pub enum JsModelStatus {
    /// Currently recommended model.
    Current,
    /// Still available but superseded by newer version.
    Legacy,
    /// Scheduled for removal, not recommended for new projects.
    Deprecated,
}

impl From<ModelStatus> for JsModelStatus {
    fn from(s: ModelStatus) -> Self {
        match s {
            ModelStatus::Current => JsModelStatus::Current,
            ModelStatus::Legacy => JsModelStatus::Legacy,
            ModelStatus::Deprecated => JsModelStatus::Deprecated,
        }
    }
}

// ============================================================================
// OBJECTS
// ============================================================================

/// Model pricing (per 1M tokens in USD).
#[napi(object)]
pub struct JsModelPricing {
    /// Input token price per 1M tokens.
    pub input_per_1m: f64,
    /// Output token price per 1M tokens.
    pub output_per_1m: f64,
    /// Cached input token price per 1M tokens (if supported).
    pub cached_input_per_1m: Option<f64>,
}

impl From<ModelPricing> for JsModelPricing {
    fn from(p: ModelPricing) -> Self {
        Self {
            input_per_1m: p.input_per_1m,
            output_per_1m: p.output_per_1m,
            cached_input_per_1m: p.cached_input_per_1m,
        }
    }
}

/// Model capabilities.
#[napi(object)]
pub struct JsModelCapabilities {
    /// Maximum input context size in tokens.
    pub max_context: u32,
    /// Maximum output tokens.
    pub max_output: u32,
    /// Supports vision/image input.
    pub vision: bool,
    /// Supports tool/function calling.
    pub tools: bool,
    /// Supports streaming responses.
    pub streaming: bool,
    /// Supports JSON mode.
    pub json_mode: bool,
    /// Supports structured output with JSON schema enforcement.
    pub structured_output: bool,
    /// Supports extended thinking/reasoning.
    pub thinking: bool,
    /// Supports prompt caching.
    pub caching: bool,
}

impl From<ModelCapabilities> for JsModelCapabilities {
    fn from(c: ModelCapabilities) -> Self {
        Self {
            max_context: c.max_context,
            max_output: c.max_output,
            vision: c.vision,
            tools: c.tools,
            streaming: c.streaming,
            json_mode: c.json_mode,
            structured_output: c.structured_output,
            thinking: c.thinking,
            caching: c.caching,
        }
    }
}

/// Benchmark scores (0-100 scale, higher is better).
#[napi(object)]
pub struct JsModelBenchmarks {
    /// MMLU - General knowledge.
    pub mmlu: Option<f64>,
    /// HumanEval - Code generation.
    pub humaneval: Option<f64>,
    /// MATH - Mathematical reasoning.
    pub math: Option<f64>,
    /// GPQA Diamond - Graduate-level science.
    pub gpqa: Option<f64>,
    /// SWE-bench - Software engineering.
    pub swe_bench: Option<f64>,
    /// IFEval - Instruction following.
    pub ifeval: Option<f64>,
    /// MMMU - Multimodal understanding.
    pub mmmu: Option<f64>,
    /// MGSM - Multilingual math.
    pub mgsm: Option<f64>,
    /// Time to first token (ms).
    pub ttft_ms: Option<u32>,
    /// Tokens per second.
    pub tokens_per_sec: Option<u32>,
}

impl From<ModelBenchmarks> for JsModelBenchmarks {
    fn from(b: ModelBenchmarks) -> Self {
        Self {
            mmlu: b.mmlu.map(|v| v as f64),
            humaneval: b.humaneval.map(|v| v as f64),
            math: b.math.map(|v| v as f64),
            gpqa: b.gpqa.map(|v| v as f64),
            swe_bench: b.swe_bench.map(|v| v as f64),
            ifeval: b.ifeval.map(|v| v as f64),
            mmmu: b.mmmu.map(|v| v as f64),
            mgsm: b.mgsm.map(|v| v as f64),
            ttft_ms: b.ttft_ms,
            tokens_per_sec: b.tokens_per_sec,
        }
    }
}

/// Registry statistics.
#[napi(object)]
pub struct JsRegistryStats {
    /// Total number of models in the registry.
    pub total_models: u32,
    /// Number of current (non-deprecated) models.
    pub current_models: u32,
    /// Number of providers.
    pub providers: u32,
    /// Number of models available (API key configured).
    pub available_models: u32,
}

// ============================================================================
// MODEL INFO CLASS
// ============================================================================

/// Complete model specification.
#[napi]
pub struct JsModelInfo {
    inner: &'static ModelInfo,
}

#[napi]
impl JsModelInfo {
    /// Unified model ID (e.g., "anthropic/claude-3-5-sonnet").
    #[napi(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Short alias (e.g., "claude-3-5-sonnet").
    #[napi(getter)]
    pub fn alias(&self) -> Option<String> {
        self.inner.alias.clone()
    }

    /// Human-readable name.
    #[napi(getter)]
    pub fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Provider.
    #[napi(getter)]
    pub fn provider(&self) -> JsProvider {
        self.inner.provider.into()
    }

    /// Model status.
    #[napi(getter)]
    pub fn status(&self) -> JsModelStatus {
        self.inner.status.into()
    }

    /// Pricing information.
    #[napi(getter)]
    pub fn pricing(&self) -> JsModelPricing {
        self.inner.pricing.into()
    }

    /// Model capabilities.
    #[napi(getter)]
    pub fn capabilities(&self) -> JsModelCapabilities {
        self.inner.capabilities.into()
    }

    /// Benchmark scores.
    #[napi(getter)]
    pub fn benchmarks(&self) -> JsModelBenchmarks {
        self.inner.benchmarks.into()
    }

    /// Model description.
    #[napi(getter)]
    pub fn description(&self) -> String {
        self.inner.description.clone()
    }

    /// Whether the model can be used as a classifier.
    #[napi(getter)]
    pub fn can_classify(&self) -> bool {
        self.inner.can_classify
    }

    /// Get the raw model ID without provider prefix.
    #[napi]
    pub fn raw_id(&self) -> String {
        self.inner.raw_id().to_string()
    }

    /// Calculate quality per dollar (higher is better value).
    #[napi]
    pub fn quality_per_dollar(&self) -> f64 {
        self.inner.quality_per_dollar()
    }

    /// Estimate cost for a request.
    #[napi]
    pub fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        self.inner.estimate_cost(input_tokens, output_tokens)
    }

    /// Calculate weighted quality score from benchmarks (0-100).
    #[napi]
    pub fn quality_score(&self) -> f64 {
        self.inner.benchmarks.quality_score() as f64
    }
}

impl From<&'static ModelInfo> for JsModelInfo {
    fn from(inner: &'static ModelInfo) -> Self {
        Self { inner }
    }
}

// ============================================================================
// PUBLIC API FUNCTIONS
// ============================================================================

/// Get model info by ID, alias, or raw ID.
///
/// @example
/// ```typescript
/// import { getModelInfo } from 'modelsuite';
///
/// const info = getModelInfo('claude-sonnet-4-5');
/// if (info) {
///   console.log(`${info.name}: $${info.pricing.inputPer1m}/1M input tokens`);
///   console.log(`Context: ${info.capabilities.maxContext} tokens`);
/// }
/// ```
#[napi]
pub fn get_model_info(model_id: String) -> Option<JsModelInfo> {
    models::get_model_info(&model_id).map(JsModelInfo::from)
}

/// Get all models in the registry.
///
/// @example
/// ```typescript
/// import { getAllModels } from 'modelsuite';
///
/// const models = getAllModels();
/// console.log(`Registry contains ${models.length} models`);
/// ```
#[napi]
pub fn get_all_models() -> Vec<JsModelInfo> {
    models::get_all_models()
        .iter()
        .map(JsModelInfo::from)
        .collect()
}

/// Get all models for a specific provider.
///
/// @example
/// ```typescript
/// import { getModelsByProvider, Provider } from 'modelsuite';
///
/// const anthropicModels = getModelsByProvider(Provider.Anthropic);
/// for (const model of anthropicModels) {
///   console.log(`${model.name}: ${model.description}`);
/// }
/// ```
#[napi]
pub fn get_models_by_provider(provider: JsProvider) -> Vec<JsModelInfo> {
    models::get_models_by_provider(provider.into())
        .into_iter()
        .map(JsModelInfo::from)
        .collect()
}

/// Get all current (non-deprecated) models.
///
/// @example
/// ```typescript
/// import { getCurrentModels } from 'modelsuite';
///
/// const current = getCurrentModels();
/// console.log(`${current.length} current models available`);
/// ```
#[napi]
pub fn get_current_models() -> Vec<JsModelInfo> {
    models::get_current_models()
        .into_iter()
        .map(JsModelInfo::from)
        .collect()
}

/// Get models that can be used as classifiers (fast, cheap, good instruction following).
///
/// @example
/// ```typescript
/// import { getClassifierModels } from 'modelsuite';
///
/// const classifiers = getClassifierModels();
/// for (const model of classifiers) {
///   console.log(`${model.name}: $${model.pricing.inputPer1m}/1M tokens`);
/// }
/// ```
#[napi]
pub fn get_classifier_models() -> Vec<JsModelInfo> {
    models::get_classifier_models()
        .into_iter()
        .map(JsModelInfo::from)
        .collect()
}

/// Get available models (provider API key is configured).
///
/// @example
/// ```typescript
/// import { getAvailableModels } from 'modelsuite';
///
/// const available = getAvailableModels();
/// console.log(`${available.length} models available with current credentials`);
/// ```
#[napi]
pub fn get_available_models() -> Vec<JsModelInfo> {
    models::get_available_models()
        .into_iter()
        .map(JsModelInfo::from)
        .collect()
}

/// Get models with specific capabilities.
///
/// @param vision - Filter by vision support (null to ignore).
/// @param tools - Filter by tool calling support (null to ignore).
/// @param thinking - Filter by extended thinking support (null to ignore).
///
/// @example
/// ```typescript
/// import { getModelsWithCapability } from 'modelsuite';
///
/// // Get all vision models
/// const visionModels = getModelsWithCapability(true, null, null);
///
/// // Get models with extended thinking
/// const thinkingModels = getModelsWithCapability(null, null, true);
/// ```
#[napi]
pub fn get_models_with_capability(
    vision: Option<bool>,
    tools: Option<bool>,
    thinking: Option<bool>,
) -> Vec<JsModelInfo> {
    models::get_models_with_capability(vision, tools, thinking)
        .into_iter()
        .map(JsModelInfo::from)
        .collect()
}

/// Get the cheapest model that meets requirements.
///
/// @param minContext - Minimum context window size (null for any).
/// @param needsVision - Whether vision support is required.
/// @param needsTools - Whether tool calling support is required.
///
/// @example
/// ```typescript
/// import { getCheapestModel } from 'modelsuite';
///
/// // Get cheapest model with at least 100k context
/// const cheapest = getCheapestModel(100000, false, true);
/// if (cheapest) {
///   console.log(`Cheapest: ${cheapest.name} at $${cheapest.pricing.inputPer1m}/1M`);
/// }
/// ```
#[napi]
pub fn get_cheapest_model(
    min_context: Option<u32>,
    needs_vision: bool,
    needs_tools: bool,
) -> Option<JsModelInfo> {
    models::get_cheapest_model(min_context, needs_vision, needs_tools).map(JsModelInfo::from)
}

/// Check if a model supports structured output (JSON schema enforcement).
///
/// @example
/// ```typescript
/// import { supportsStructuredOutput } from 'modelsuite';
///
/// if (supportsStructuredOutput('gpt-4o')) {
///   // Use structured output with JSON schema
/// } else {
///   // Fall back to text-based parsing
/// }
/// ```
#[napi]
pub fn supports_structured_output(model_id: String) -> bool {
    models::supports_structured_output(&model_id)
}

/// Get registry statistics.
///
/// @example
/// ```typescript
/// import { getRegistryStats } from 'modelsuite';
///
/// const stats = getRegistryStats();
/// console.log(`Registry: ${stats.totalModels} models from ${stats.providers} providers`);
/// ```
#[napi]
pub fn get_registry_stats() -> JsRegistryStats {
    let stats = models::get_registry_stats();
    JsRegistryStats {
        total_models: stats.total_models as u32,
        current_models: stats.current_models as u32,
        providers: stats.providers as u32,
        available_models: stats.available_models as u32,
    }
}

/// List all providers with at least one model.
///
/// @example
/// ```typescript
/// import { listProviders } from 'modelsuite';
///
/// const providers = listProviders();
/// console.log(`Supported providers: ${providers.join(', ')}`);
/// ```
#[napi]
pub fn list_providers() -> Vec<JsProvider> {
    models::list_providers()
        .into_iter()
        .map(JsProvider::from)
        .collect()
}
