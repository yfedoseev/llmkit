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
    Writer,
    Maritaca,
    Clova,
    Yandex,
    GigaChat,
    Upstage,
    SeaLion,
    Local,
    Custom,
}

impl From<Provider> for PyProvider {
    fn from(p: Provider) -> Self {
        match p {
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
            Provider::Writer => PyProvider::Writer,
            Provider::Maritaca => PyProvider::Maritaca,
            Provider::Clova => PyProvider::Clova,
            Provider::Yandex => PyProvider::Yandex,
            Provider::GigaChat => PyProvider::GigaChat,
            Provider::Upstage => PyProvider::Upstage,
            Provider::SeaLion => PyProvider::SeaLion,
            Provider::Local => PyProvider::Local,
            Provider::Custom => PyProvider::Custom,
            Provider::Alibaba => PyProvider::Custom, // Map to Custom for now
            Provider::Baidu => PyProvider::Custom,   // Map to Custom for now
            Provider::Zhipu => PyProvider::Custom,   // Map to Custom for now
            Provider::Moonshot => PyProvider::Custom, // Map to Custom for now
            Provider::Rakuten => PyProvider::Custom, // Map to Custom for now
            Provider::Sarvam => PyProvider::Custom,  // Map to Custom for now
        }
    }
}

impl From<PyProvider> for Provider {
    fn from(p: PyProvider) -> Self {
        match p {
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
            PyProvider::Writer => Provider::Writer,
            PyProvider::Maritaca => Provider::Maritaca,
            PyProvider::Clova => Provider::Clova,
            PyProvider::Yandex => Provider::Yandex,
            PyProvider::GigaChat => Provider::GigaChat,
            PyProvider::Upstage => Provider::Upstage,
            PyProvider::SeaLion => Provider::SeaLion,
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
