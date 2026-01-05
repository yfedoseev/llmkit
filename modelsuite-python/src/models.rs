//! Model Registry bindings for Python.
//!
//! Provides access to the ModelSuite model registry with information about
//! pricing, capabilities, and benchmarks for all supported models.

use modelsuite::models::{
    self, ModelBenchmarks, ModelCapabilities, ModelInfo, ModelPricing, ModelStatus, Provider,
};
use pyo3::prelude::*;

// ============================================================================
// ENUMS
// ============================================================================

/// LLM Provider identifier.
#[pyclass(name = "Provider", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyProvider {
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
    // New providers
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

impl From<Provider> for PyProvider {
    fn from(p: Provider) -> Self {
        match p {
            // Core providers
            Provider::Anthropic => PyProvider::Anthropic,
            Provider::OpenAI => PyProvider::OpenAI,
            Provider::Google => PyProvider::Google,
            Provider::Mistral => PyProvider::Mistral,
            Provider::Groq => PyProvider::Groq,
            Provider::DeepSeek => PyProvider::DeepSeek,
            Provider::Cohere => PyProvider::Cohere,
            Provider::Bedrock => PyProvider::Bedrock,
            Provider::AzureOpenAI => PyProvider::AzureOpenAI,
            Provider::VertexAI => PyProvider::VertexAI,
            Provider::TogetherAI => PyProvider::TogetherAI,
            Provider::OpenRouter => PyProvider::OpenRouter,
            Provider::Cerebras => PyProvider::Cerebras,
            Provider::SambaNova => PyProvider::SambaNova,
            Provider::Fireworks => PyProvider::Fireworks,
            Provider::AI21 => PyProvider::AI21,
            Provider::HuggingFace => PyProvider::HuggingFace,
            Provider::Replicate => PyProvider::Replicate,
            Provider::Cloudflare => PyProvider::Cloudflare,
            Provider::Databricks => PyProvider::Databricks,
            // Regional providers
            Provider::Writer => PyProvider::Writer,
            Provider::Maritaca => PyProvider::Maritaca,
            Provider::Clova => PyProvider::Clova,
            Provider::Yandex => PyProvider::Yandex,
            Provider::GigaChat => PyProvider::GigaChat,
            Provider::Upstage => PyProvider::Upstage,
            Provider::SeaLion => PyProvider::SeaLion,
            // Additional providers
            Provider::Alibaba => PyProvider::Alibaba,
            Provider::AlephAlpha => PyProvider::AlephAlpha,
            Provider::Baidu => PyProvider::Baidu,
            Provider::Baseten => PyProvider::Baseten,
            Provider::ChatLaw => PyProvider::ChatLaw,
            Provider::DataRobot => PyProvider::DataRobot,
            Provider::LatamGPT => PyProvider::LatamGPT,
            Provider::LightOn => PyProvider::LightOn,
            Provider::NLPCloud => PyProvider::NLPCloud,
            Provider::Oracle => PyProvider::Oracle,
            Provider::Perplexity => PyProvider::Perplexity,
            Provider::RunPod => PyProvider::RunPod,
            Provider::Sagemaker => PyProvider::Sagemaker,
            Provider::Sap => PyProvider::Sap,
            Provider::Snowflake => PyProvider::Snowflake,
            Provider::Vllm => PyProvider::Vllm,
            Provider::WatsonX => PyProvider::WatsonX,
            // New providers
            Provider::Xai => PyProvider::Xai,
            Provider::DeepInfra => PyProvider::DeepInfra,
            Provider::NvidiaNIM => PyProvider::NvidiaNIM,
            // Tier 1 - High Priority Inference
            Provider::Ollama => PyProvider::Ollama,
            Provider::Anyscale => PyProvider::Anyscale,
            Provider::GitHub => PyProvider::GitHub,
            Provider::FriendliAI => PyProvider::FriendliAI,
            Provider::Hyperbolic => PyProvider::Hyperbolic,
            Provider::Lambda => PyProvider::Lambda,
            Provider::Novita => PyProvider::Novita,
            Provider::Nebius => PyProvider::Nebius,
            Provider::Lepton => PyProvider::Lepton,
            Provider::Stability => PyProvider::Stability,
            Provider::Voyage => PyProvider::Voyage,
            Provider::Jina => PyProvider::Jina,
            Provider::Deepgram => PyProvider::Deepgram,
            Provider::ElevenLabs => PyProvider::ElevenLabs,
            Provider::GPT4All => PyProvider::GPT4All,
            // Tier 2 - Chinese Providers
            Provider::MiniMax => PyProvider::MiniMax,
            Provider::Moonshot => PyProvider::Moonshot,
            Provider::Zhipu => PyProvider::Zhipu,
            Provider::Volcengine => PyProvider::Volcengine,
            Provider::Baichuan => PyProvider::Baichuan,
            Provider::Stepfun => PyProvider::Stepfun,
            Provider::Yi => PyProvider::Yi,
            Provider::Spark => PyProvider::Spark,
            // Tier 3 - Local/Self-Hosted
            Provider::LMStudio => PyProvider::LMStudio,
            Provider::Llamafile => PyProvider::Llamafile,
            Provider::Xinference => PyProvider::Xinference,
            Provider::LocalAI => PyProvider::LocalAI,
            Provider::Jan => PyProvider::Jan,
            Provider::Petals => PyProvider::Petals,
            Provider::Triton => PyProvider::Triton,
            Provider::Tgi => PyProvider::Tgi,
            // Tier 4 - Enterprise/Specialized
            Provider::Predibase => PyProvider::Predibase,
            Provider::OctoAI => PyProvider::OctoAI,
            Provider::Featherless => PyProvider::Featherless,
            Provider::OVHCloud => PyProvider::OVHCloud,
            Provider::Scaleway => PyProvider::Scaleway,
            Provider::Crusoe => PyProvider::Crusoe,
            Provider::Cerebrium => PyProvider::Cerebrium,
            Provider::Lightning => PyProvider::Lightning,
            Provider::AssemblyAI => PyProvider::AssemblyAI,
            Provider::RunwayML => PyProvider::RunwayML,
            // Tier 5 - Asian Regional Providers
            Provider::Naver => PyProvider::Naver,
            Provider::Kakao => PyProvider::Kakao,
            Provider::LGExaone => PyProvider::LGExaone,
            Provider::PLaMo => PyProvider::PLaMo,
            Provider::Sarvam => PyProvider::Sarvam,
            Provider::Krutrim => PyProvider::Krutrim,
            Provider::Ntt => PyProvider::Ntt,
            Provider::SoftBank => PyProvider::SoftBank,
            // Tier 6 - European Sovereign AI
            Provider::Ionos => PyProvider::Ionos,
            Provider::Tilde => PyProvider::Tilde,
            Provider::SiloAI => PyProvider::SiloAI,
            Provider::SwissAI => PyProvider::SwissAI,
            // Tier 7 - Router/Gateway/Meta Providers
            Provider::Unify => PyProvider::Unify,
            Provider::Martian => PyProvider::Martian,
            Provider::Portkey => PyProvider::Portkey,
            Provider::Helicone => PyProvider::Helicone,
            Provider::SiliconFlow => PyProvider::SiliconFlow,
            // Tier 8 - Video AI Providers
            Provider::Pika => PyProvider::Pika,
            Provider::Luma => PyProvider::Luma,
            Provider::Kling => PyProvider::Kling,
            Provider::HeyGen => PyProvider::HeyGen,
            Provider::Did => PyProvider::Did,
            Provider::TwelveLabs => PyProvider::TwelveLabs,
            // Tier 9 - Audio AI Providers
            Provider::Rev => PyProvider::Rev,
            Provider::Speechmatics => PyProvider::Speechmatics,
            Provider::PlayHT => PyProvider::PlayHT,
            Provider::Resemble => PyProvider::Resemble,
            // Tier 10 - Image AI Providers
            Provider::Leonardo => PyProvider::Leonardo,
            Provider::Ideogram => PyProvider::Ideogram,
            Provider::BlackForestLabs => PyProvider::BlackForestLabs,
            Provider::Clarifai => PyProvider::Clarifai,
            Provider::Fal => PyProvider::Fal,
            // Tier 11 - Infrastructure Providers
            Provider::Modal => PyProvider::Modal,
            Provider::CoreWeave => PyProvider::CoreWeave,
            Provider::TensorDock => PyProvider::TensorDock,
            Provider::Beam => PyProvider::Beam,
            Provider::VastAI => PyProvider::VastAI,
            // Tier 12 - Emerging Startups
            Provider::Nscale => PyProvider::Nscale,
            Provider::Runware => PyProvider::Runware,
            Provider::AI71 => PyProvider::AI71,
            // Local/Custom
            Provider::Local => PyProvider::Local,
            Provider::Custom => PyProvider::Custom,
        }
    }
}

impl From<PyProvider> for Provider {
    fn from(p: PyProvider) -> Self {
        match p {
            // Core providers
            PyProvider::Anthropic => Provider::Anthropic,
            PyProvider::OpenAI => Provider::OpenAI,
            PyProvider::Google => Provider::Google,
            PyProvider::Mistral => Provider::Mistral,
            PyProvider::Groq => Provider::Groq,
            PyProvider::DeepSeek => Provider::DeepSeek,
            PyProvider::Cohere => Provider::Cohere,
            PyProvider::Bedrock => Provider::Bedrock,
            PyProvider::AzureOpenAI => Provider::AzureOpenAI,
            PyProvider::VertexAI => Provider::VertexAI,
            PyProvider::TogetherAI => Provider::TogetherAI,
            PyProvider::OpenRouter => Provider::OpenRouter,
            PyProvider::Cerebras => Provider::Cerebras,
            PyProvider::SambaNova => Provider::SambaNova,
            PyProvider::Fireworks => Provider::Fireworks,
            PyProvider::AI21 => Provider::AI21,
            PyProvider::HuggingFace => Provider::HuggingFace,
            PyProvider::Replicate => Provider::Replicate,
            PyProvider::Cloudflare => Provider::Cloudflare,
            PyProvider::Databricks => Provider::Databricks,
            // Regional providers
            PyProvider::Writer => Provider::Writer,
            PyProvider::Maritaca => Provider::Maritaca,
            PyProvider::Clova => Provider::Clova,
            PyProvider::Yandex => Provider::Yandex,
            PyProvider::GigaChat => Provider::GigaChat,
            PyProvider::Upstage => Provider::Upstage,
            PyProvider::SeaLion => Provider::SeaLion,
            // Additional providers
            PyProvider::Alibaba => Provider::Alibaba,
            PyProvider::AlephAlpha => Provider::AlephAlpha,
            PyProvider::Baidu => Provider::Baidu,
            PyProvider::Baseten => Provider::Baseten,
            PyProvider::ChatLaw => Provider::ChatLaw,
            PyProvider::DataRobot => Provider::DataRobot,
            PyProvider::LatamGPT => Provider::LatamGPT,
            PyProvider::LightOn => Provider::LightOn,
            PyProvider::NLPCloud => Provider::NLPCloud,
            PyProvider::Oracle => Provider::Oracle,
            PyProvider::Perplexity => Provider::Perplexity,
            PyProvider::RunPod => Provider::RunPod,
            PyProvider::Sagemaker => Provider::Sagemaker,
            PyProvider::Sap => Provider::Sap,
            PyProvider::Snowflake => Provider::Snowflake,
            PyProvider::Vllm => Provider::Vllm,
            PyProvider::WatsonX => Provider::WatsonX,
            // New providers
            PyProvider::Xai => Provider::Xai,
            PyProvider::DeepInfra => Provider::DeepInfra,
            PyProvider::NvidiaNIM => Provider::NvidiaNIM,
            // Tier 1 - High Priority Inference
            PyProvider::Ollama => Provider::Ollama,
            PyProvider::Anyscale => Provider::Anyscale,
            PyProvider::GitHub => Provider::GitHub,
            PyProvider::FriendliAI => Provider::FriendliAI,
            PyProvider::Hyperbolic => Provider::Hyperbolic,
            PyProvider::Lambda => Provider::Lambda,
            PyProvider::Novita => Provider::Novita,
            PyProvider::Nebius => Provider::Nebius,
            PyProvider::Lepton => Provider::Lepton,
            PyProvider::Stability => Provider::Stability,
            PyProvider::Voyage => Provider::Voyage,
            PyProvider::Jina => Provider::Jina,
            PyProvider::Deepgram => Provider::Deepgram,
            PyProvider::ElevenLabs => Provider::ElevenLabs,
            PyProvider::GPT4All => Provider::GPT4All,
            // Tier 2 - Chinese Providers
            PyProvider::MiniMax => Provider::MiniMax,
            PyProvider::Moonshot => Provider::Moonshot,
            PyProvider::Zhipu => Provider::Zhipu,
            PyProvider::Volcengine => Provider::Volcengine,
            PyProvider::Baichuan => Provider::Baichuan,
            PyProvider::Stepfun => Provider::Stepfun,
            PyProvider::Yi => Provider::Yi,
            PyProvider::Spark => Provider::Spark,
            // Tier 3 - Local/Self-Hosted
            PyProvider::LMStudio => Provider::LMStudio,
            PyProvider::Llamafile => Provider::Llamafile,
            PyProvider::Xinference => Provider::Xinference,
            PyProvider::LocalAI => Provider::LocalAI,
            PyProvider::Jan => Provider::Jan,
            PyProvider::Petals => Provider::Petals,
            PyProvider::Triton => Provider::Triton,
            PyProvider::Tgi => Provider::Tgi,
            // Tier 4 - Enterprise/Specialized
            PyProvider::Predibase => Provider::Predibase,
            PyProvider::OctoAI => Provider::OctoAI,
            PyProvider::Featherless => Provider::Featherless,
            PyProvider::OVHCloud => Provider::OVHCloud,
            PyProvider::Scaleway => Provider::Scaleway,
            PyProvider::Crusoe => Provider::Crusoe,
            PyProvider::Cerebrium => Provider::Cerebrium,
            PyProvider::Lightning => Provider::Lightning,
            PyProvider::AssemblyAI => Provider::AssemblyAI,
            PyProvider::RunwayML => Provider::RunwayML,
            // Tier 5 - Asian Regional Providers
            PyProvider::Naver => Provider::Naver,
            PyProvider::Kakao => Provider::Kakao,
            PyProvider::LGExaone => Provider::LGExaone,
            PyProvider::PLaMo => Provider::PLaMo,
            PyProvider::Sarvam => Provider::Sarvam,
            PyProvider::Krutrim => Provider::Krutrim,
            PyProvider::Ntt => Provider::Ntt,
            PyProvider::SoftBank => Provider::SoftBank,
            // Tier 6 - European Sovereign AI
            PyProvider::Ionos => Provider::Ionos,
            PyProvider::Tilde => Provider::Tilde,
            PyProvider::SiloAI => Provider::SiloAI,
            PyProvider::SwissAI => Provider::SwissAI,
            // Tier 7 - Router/Gateway/Meta Providers
            PyProvider::Unify => Provider::Unify,
            PyProvider::Martian => Provider::Martian,
            PyProvider::Portkey => Provider::Portkey,
            PyProvider::Helicone => Provider::Helicone,
            PyProvider::SiliconFlow => Provider::SiliconFlow,
            // Tier 8 - Video AI Providers
            PyProvider::Pika => Provider::Pika,
            PyProvider::Luma => Provider::Luma,
            PyProvider::Kling => Provider::Kling,
            PyProvider::HeyGen => Provider::HeyGen,
            PyProvider::Did => Provider::Did,
            PyProvider::TwelveLabs => Provider::TwelveLabs,
            // Tier 9 - Audio AI Providers
            PyProvider::Rev => Provider::Rev,
            PyProvider::Speechmatics => Provider::Speechmatics,
            PyProvider::PlayHT => Provider::PlayHT,
            PyProvider::Resemble => Provider::Resemble,
            // Tier 10 - Image AI Providers
            PyProvider::Leonardo => Provider::Leonardo,
            PyProvider::Ideogram => Provider::Ideogram,
            PyProvider::BlackForestLabs => Provider::BlackForestLabs,
            PyProvider::Clarifai => Provider::Clarifai,
            PyProvider::Fal => Provider::Fal,
            // Tier 11 - Infrastructure Providers
            PyProvider::Modal => Provider::Modal,
            PyProvider::CoreWeave => Provider::CoreWeave,
            PyProvider::TensorDock => Provider::TensorDock,
            PyProvider::Beam => Provider::Beam,
            PyProvider::VastAI => Provider::VastAI,
            // Tier 12 - Emerging Startups
            PyProvider::Nscale => Provider::Nscale,
            PyProvider::Runware => Provider::Runware,
            PyProvider::AI71 => Provider::AI71,
            // Local/Custom
            PyProvider::Local => Provider::Local,
            PyProvider::Custom => Provider::Custom,
        }
    }
}

/// Model availability status.
#[pyclass(name = "ModelStatus", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyModelStatus {
    /// Currently recommended model.
    Current,
    /// Still available but superseded by newer version.
    Legacy,
    /// Scheduled for removal, not recommended for new projects.
    Deprecated,
}

impl From<ModelStatus> for PyModelStatus {
    fn from(s: ModelStatus) -> Self {
        match s {
            ModelStatus::Current => PyModelStatus::Current,
            ModelStatus::Legacy => PyModelStatus::Legacy,
            ModelStatus::Deprecated => PyModelStatus::Deprecated,
        }
    }
}

// ============================================================================
// DATA CLASSES
// ============================================================================

/// Model pricing (per 1M tokens in USD).
#[pyclass(name = "ModelPricing")]
#[derive(Clone)]
pub struct PyModelPricing {
    inner: ModelPricing,
}

#[pymethods]
impl PyModelPricing {
    /// Input token price per 1M tokens.
    #[getter]
    fn input_per_1m(&self) -> f64 {
        self.inner.input_per_1m
    }

    /// Output token price per 1M tokens.
    #[getter]
    fn output_per_1m(&self) -> f64 {
        self.inner.output_per_1m
    }

    /// Cached input token price per 1M tokens (if supported).
    #[getter]
    fn cached_input_per_1m(&self) -> Option<f64> {
        self.inner.cached_input_per_1m
    }

    /// Estimate cost for given token counts.
    fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        self.inner.estimate_cost(input_tokens, output_tokens)
    }
}

impl From<ModelPricing> for PyModelPricing {
    fn from(inner: ModelPricing) -> Self {
        Self { inner }
    }
}

/// Model capabilities.
#[pyclass(name = "ModelCapabilities")]
#[derive(Clone)]
pub struct PyModelCapabilities {
    inner: ModelCapabilities,
}

#[pymethods]
impl PyModelCapabilities {
    /// Maximum input context size in tokens.
    #[getter]
    fn max_context(&self) -> u32 {
        self.inner.max_context
    }

    /// Maximum output tokens.
    #[getter]
    fn max_output(&self) -> u32 {
        self.inner.max_output
    }

    /// Supports vision/image input.
    #[getter]
    fn vision(&self) -> bool {
        self.inner.vision
    }

    /// Supports tool/function calling.
    #[getter]
    fn tools(&self) -> bool {
        self.inner.tools
    }

    /// Supports streaming responses.
    #[getter]
    fn streaming(&self) -> bool {
        self.inner.streaming
    }

    /// Supports JSON mode.
    #[getter]
    fn json_mode(&self) -> bool {
        self.inner.json_mode
    }

    /// Supports structured output with JSON schema enforcement.
    #[getter]
    fn structured_output(&self) -> bool {
        self.inner.structured_output
    }

    /// Supports extended thinking/reasoning.
    #[getter]
    fn thinking(&self) -> bool {
        self.inner.thinking
    }

    /// Supports prompt caching.
    #[getter]
    fn caching(&self) -> bool {
        self.inner.caching
    }
}

impl From<ModelCapabilities> for PyModelCapabilities {
    fn from(inner: ModelCapabilities) -> Self {
        Self { inner }
    }
}

/// Benchmark scores (0-100 scale, higher is better).
#[pyclass(name = "ModelBenchmarks")]
#[derive(Clone)]
pub struct PyModelBenchmarks {
    inner: ModelBenchmarks,
}

#[pymethods]
impl PyModelBenchmarks {
    /// MMLU - General knowledge.
    #[getter]
    fn mmlu(&self) -> Option<f32> {
        self.inner.mmlu
    }

    /// HumanEval - Code generation.
    #[getter]
    fn humaneval(&self) -> Option<f32> {
        self.inner.humaneval
    }

    /// MATH - Mathematical reasoning.
    #[getter]
    fn math(&self) -> Option<f32> {
        self.inner.math
    }

    /// GPQA Diamond - Graduate-level science.
    #[getter]
    fn gpqa(&self) -> Option<f32> {
        self.inner.gpqa
    }

    /// SWE-bench - Software engineering.
    #[getter]
    fn swe_bench(&self) -> Option<f32> {
        self.inner.swe_bench
    }

    /// IFEval - Instruction following.
    #[getter]
    fn ifeval(&self) -> Option<f32> {
        self.inner.ifeval
    }

    /// MMMU - Multimodal understanding.
    #[getter]
    fn mmmu(&self) -> Option<f32> {
        self.inner.mmmu
    }

    /// MGSM - Multilingual math.
    #[getter]
    fn mgsm(&self) -> Option<f32> {
        self.inner.mgsm
    }

    /// Time to first token (ms).
    #[getter]
    fn ttft_ms(&self) -> Option<u32> {
        self.inner.ttft_ms
    }

    /// Tokens per second.
    #[getter]
    fn tokens_per_sec(&self) -> Option<u32> {
        self.inner.tokens_per_sec
    }

    /// Calculate weighted quality score (0-100).
    fn quality_score(&self) -> f32 {
        self.inner.quality_score()
    }
}

impl From<ModelBenchmarks> for PyModelBenchmarks {
    fn from(inner: ModelBenchmarks) -> Self {
        Self { inner }
    }
}

/// Registry statistics.
#[pyclass(name = "RegistryStats")]
#[derive(Clone)]
pub struct PyRegistryStats {
    /// Total number of models in the registry.
    #[pyo3(get)]
    pub total_models: usize,
    /// Number of current (non-deprecated) models.
    #[pyo3(get)]
    pub current_models: usize,
    /// Number of providers.
    #[pyo3(get)]
    pub providers: usize,
    /// Number of models available (API key configured).
    #[pyo3(get)]
    pub available_models: usize,
}

// ============================================================================
// MODEL INFO CLASS
// ============================================================================

/// Complete model specification.
#[pyclass(name = "ModelInfo")]
pub struct PyModelInfo {
    inner: &'static ModelInfo,
}

#[pymethods]
impl PyModelInfo {
    /// Unified model ID (e.g., "anthropic/claude-3-5-sonnet").
    #[getter]
    fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Short alias (e.g., "claude-3-5-sonnet").
    #[getter]
    fn alias(&self) -> Option<String> {
        self.inner.alias.clone()
    }

    /// Human-readable name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    /// Provider.
    #[getter]
    fn provider(&self) -> PyProvider {
        self.inner.provider.into()
    }

    /// Model status.
    #[getter]
    fn status(&self) -> PyModelStatus {
        self.inner.status.into()
    }

    /// Pricing information.
    #[getter]
    fn pricing(&self) -> PyModelPricing {
        self.inner.pricing.into()
    }

    /// Model capabilities.
    #[getter]
    fn capabilities(&self) -> PyModelCapabilities {
        self.inner.capabilities.into()
    }

    /// Benchmark scores.
    #[getter]
    fn benchmarks(&self) -> PyModelBenchmarks {
        self.inner.benchmarks.into()
    }

    /// Model description.
    #[getter]
    fn description(&self) -> String {
        self.inner.description.clone()
    }

    /// Whether the model can be used as a classifier.
    #[getter]
    fn can_classify(&self) -> bool {
        self.inner.can_classify
    }

    /// Get the raw model ID without provider prefix.
    fn raw_id(&self) -> String {
        self.inner.raw_id().to_string()
    }

    /// Calculate quality per dollar (higher is better value).
    fn quality_per_dollar(&self) -> f64 {
        self.inner.quality_per_dollar()
    }

    /// Estimate cost for a request.
    fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        self.inner.estimate_cost(input_tokens, output_tokens)
    }

    fn __repr__(&self) -> String {
        format!(
            "ModelInfo(id='{}', name='{}')",
            self.inner.id, self.inner.name
        )
    }
}

impl From<&'static ModelInfo> for PyModelInfo {
    fn from(inner: &'static ModelInfo) -> Self {
        Self { inner }
    }
}

// ============================================================================
// PUBLIC API FUNCTIONS
// ============================================================================

/// Get model info by ID, alias, or raw ID.
///
/// Args:
///     model_id: Model identifier (e.g., "claude-sonnet-4-5", "gpt-4o")
///
/// Returns:
///     ModelInfo if found, None otherwise.
///
/// Example:
///     >>> from modelsuite import get_model_info
///     >>> info = get_model_info("claude-sonnet-4-5")
///     >>> if info:
///     ...     print(f"{info.name}: ${info.pricing.input_per_1m}/1M tokens")
#[pyfunction]
pub fn get_model_info(model_id: &str) -> Option<PyModelInfo> {
    models::get_model_info(model_id).map(PyModelInfo::from)
}

/// Get all models in the registry.
///
/// Returns:
///     List of all ModelInfo objects.
///
/// Example:
///     >>> from modelsuite import get_all_models
///     >>> models = get_all_models()
///     >>> print(f"Registry contains {len(models)} models")
#[pyfunction]
pub fn get_all_models() -> Vec<PyModelInfo> {
    models::get_all_models()
        .iter()
        .map(PyModelInfo::from)
        .collect()
}

/// Get all models for a specific provider.
///
/// Args:
///     provider: Provider enum value.
///
/// Returns:
///     List of ModelInfo objects for the provider.
///
/// Example:
///     >>> from modelsuite import get_models_by_provider, Provider
///     >>> anthropic_models = get_models_by_provider(Provider.Anthropic)
///     >>> for model in anthropic_models:
///     ...     print(f"{model.name}: {model.description}")
#[pyfunction]
pub fn get_models_by_provider(provider: PyProvider) -> Vec<PyModelInfo> {
    models::get_models_by_provider(provider.into())
        .into_iter()
        .map(PyModelInfo::from)
        .collect()
}

/// Get all current (non-deprecated) models.
///
/// Returns:
///     List of current ModelInfo objects.
#[pyfunction]
pub fn get_current_models() -> Vec<PyModelInfo> {
    models::get_current_models()
        .into_iter()
        .map(PyModelInfo::from)
        .collect()
}

/// Get models that can be used as classifiers.
///
/// Returns:
///     List of classifier-suitable ModelInfo objects.
#[pyfunction]
pub fn get_classifier_models() -> Vec<PyModelInfo> {
    models::get_classifier_models()
        .into_iter()
        .map(PyModelInfo::from)
        .collect()
}

/// Get available models (provider API key is configured).
///
/// Returns:
///     List of available ModelInfo objects.
#[pyfunction]
pub fn get_available_models() -> Vec<PyModelInfo> {
    models::get_available_models()
        .into_iter()
        .map(PyModelInfo::from)
        .collect()
}

/// Get models with specific capabilities.
///
/// Args:
///     vision: Filter by vision support (None to ignore).
///     tools: Filter by tool calling support (None to ignore).
///     thinking: Filter by extended thinking support (None to ignore).
///
/// Returns:
///     List of matching ModelInfo objects.
#[pyfunction]
#[pyo3(signature = (vision=None, tools=None, thinking=None))]
pub fn get_models_with_capability(
    vision: Option<bool>,
    tools: Option<bool>,
    thinking: Option<bool>,
) -> Vec<PyModelInfo> {
    models::get_models_with_capability(vision, tools, thinking)
        .into_iter()
        .map(PyModelInfo::from)
        .collect()
}

/// Get the cheapest model that meets requirements.
///
/// Args:
///     min_context: Minimum context window size (None for any).
///     needs_vision: Whether vision support is required.
///     needs_tools: Whether tool calling support is required.
///
/// Returns:
///     Cheapest ModelInfo if found, None otherwise.
#[pyfunction]
#[pyo3(signature = (min_context=None, needs_vision=false, needs_tools=false))]
pub fn get_cheapest_model(
    min_context: Option<u32>,
    needs_vision: bool,
    needs_tools: bool,
) -> Option<PyModelInfo> {
    models::get_cheapest_model(min_context, needs_vision, needs_tools).map(PyModelInfo::from)
}

/// Check if a model supports structured output (JSON schema enforcement).
///
/// Args:
///     model_id: Model identifier.
///
/// Returns:
///     True if the model supports structured output.
#[pyfunction]
pub fn supports_structured_output(model_id: &str) -> bool {
    models::supports_structured_output(model_id)
}

/// Get registry statistics.
///
/// Returns:
///     RegistryStats with counts of models and providers.
#[pyfunction]
pub fn get_registry_stats() -> PyRegistryStats {
    let stats = models::get_registry_stats();
    PyRegistryStats {
        total_models: stats.total_models,
        current_models: stats.current_models,
        providers: stats.providers,
        available_models: stats.available_models,
    }
}

/// List all providers with at least one model.
///
/// Returns:
///     List of Provider enum values.
#[pyfunction]
pub fn list_providers() -> Vec<PyProvider> {
    models::list_providers()
        .into_iter()
        .map(PyProvider::from)
        .collect()
}
