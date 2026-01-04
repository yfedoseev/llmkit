//! Model Registry - Database of LLM model specifications.
//!
//! This module provides a comprehensive registry of LLM models across all supported providers,
//! including pricing information, context window sizes, capabilities, and benchmark scores.
//!
//! The data uses LiteLLM-compatible model IDs (e.g., `anthropic/claude-3-5-sonnet`).
//!
//! # Example
//!
//! ```ignore
//! use llmkit::models::{get_model_info, get_models_by_provider, Provider};
//!
//! // Get info for a specific model
//! if let Some(info) = get_model_info("claude-sonnet-4-20250514") {
//!     println!("Model: {}", info.name);
//!     println!("Context: {} tokens", info.capabilities.max_context);
//!     println!("Input cost: ${}/1M tokens", info.pricing.input_per_1m);
//! }
//!
//! // Get all models for a provider
//! let anthropic_models = get_models_by_provider(Provider::Anthropic);
//! ```
//!
//! ## Sources
//! - Anthropic: https://docs.anthropic.com/en/docs/about-claude/models/overview
//! - OpenAI: https://openai.com/api/pricing/
//! - Google: https://ai.google.dev/gemini-api/docs/pricing
//! - Mistral: https://mistral.ai/pricing
//! - AWS Bedrock: https://aws.amazon.com/bedrock/pricing/

use std::collections::HashMap;
use std::sync::LazyLock;

// ============================================================================
// PROVIDER ENUM
// ============================================================================

/// LLM Provider identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
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
    // Regional providers - Asia
    Writer,
    Maritaca,
    Clova,
    Yandex,
    GigaChat,
    Upstage,
    SeaLion,
    Alibaba,
    Baidu,
    Zhipu,
    Moonshot,
    Rakuten,
    Sarvam,
    Local,
    Custom,
}

impl Provider {
    /// Get the environment variable name for this provider's API key.
    pub fn api_key_env_var(&self) -> Option<&'static str> {
        match self {
            Provider::Anthropic => Some("ANTHROPIC_API_KEY"),
            Provider::OpenAI => Some("OPENAI_API_KEY"),
            Provider::Google => Some("GOOGLE_API_KEY"),
            Provider::Mistral => Some("MISTRAL_API_KEY"),
            Provider::Groq => Some("GROQ_API_KEY"),
            Provider::DeepSeek => Some("DEEPSEEK_API_KEY"),
            Provider::Cohere => Some("COHERE_API_KEY"),
            Provider::Bedrock => Some("AWS_ACCESS_KEY_ID"),
            Provider::AzureOpenAI => Some("AZURE_API_KEY"),
            Provider::VertexAI => Some("GOOGLE_APPLICATION_CREDENTIALS"),
            Provider::TogetherAI => Some("TOGETHER_API_KEY"),
            Provider::OpenRouter => Some("OPENROUTER_API_KEY"),
            Provider::Cerebras => Some("CEREBRAS_API_KEY"),
            Provider::SambaNova => Some("SAMBANOVA_API_KEY"),
            Provider::Fireworks => Some("FIREWORKS_API_KEY"),
            Provider::AI21 => Some("AI21_API_KEY"),
            Provider::HuggingFace => Some("HUGGINGFACE_API_KEY"),
            Provider::Replicate => Some("REPLICATE_API_TOKEN"),
            Provider::Cloudflare => Some("CLOUDFLARE_API_TOKEN"),
            Provider::Databricks => Some("DATABRICKS_TOKEN"),
            // Regional providers - Asia
            Provider::Writer => Some("WRITER_API_KEY"),
            Provider::Maritaca => Some("MARITALK_API_KEY"),
            Provider::Clova => Some("CLOVASTUDIO_API_KEY"),
            Provider::Yandex => Some("YANDEX_API_KEY"),
            Provider::GigaChat => Some("GIGACHAT_API_KEY"),
            Provider::Upstage => Some("UPSTAGE_API_KEY"),
            Provider::SeaLion => Some("SEA_LION_API_KEY"),
            Provider::Alibaba => Some("ALIBABA_API_KEY"),
            Provider::Baidu => Some("BAIDU_API_KEY"),
            Provider::Zhipu => Some("ZHIPU_API_KEY"),
            Provider::Moonshot => Some("MOONSHOT_API_KEY"),
            Provider::Rakuten => Some("RAKUTEN_API_KEY"),
            Provider::Sarvam => Some("SARVAM_API_KEY"),
            Provider::Local | Provider::Custom => None,
        }
    }

    /// Check if the provider's API key is available in environment.
    pub fn is_available(&self) -> bool {
        match self {
            Provider::Bedrock => {
                std::env::var("AWS_ACCESS_KEY_ID").is_ok() || std::env::var("AWS_PROFILE").is_ok()
            }
            Provider::Local | Provider::Custom => true,
            _ => self
                .api_key_env_var()
                .map(|v| std::env::var(v).is_ok())
                .unwrap_or(true),
        }
    }

    /// Parse provider from LiteLLM-style ID prefix.
    pub fn from_prefix(prefix: &str) -> Self {
        match prefix.to_lowercase().as_str() {
            "anthropic" => Provider::Anthropic,
            "openai" => Provider::OpenAI,
            "google" | "gemini" => Provider::Google,
            "mistral" | "mistralai" => Provider::Mistral,
            "groq" => Provider::Groq,
            "deepseek" => Provider::DeepSeek,
            "cohere" => Provider::Cohere,
            "bedrock" => Provider::Bedrock,
            "azure" => Provider::AzureOpenAI,
            "vertex_ai" | "vertex" => Provider::VertexAI,
            "together_ai" | "together" => Provider::TogetherAI,
            "openrouter" => Provider::OpenRouter,
            "cerebras" => Provider::Cerebras,
            "sambanova" => Provider::SambaNova,
            "fireworks" | "fireworks_ai" => Provider::Fireworks,
            "ai21" => Provider::AI21,
            "huggingface" | "hf" => Provider::HuggingFace,
            "replicate" => Provider::Replicate,
            "cloudflare" | "cf" => Provider::Cloudflare,
            "databricks" => Provider::Databricks,
            // Regional providers - Asia
            "writer" => Provider::Writer,
            "maritaca" | "maritalk" => Provider::Maritaca,
            "clova" | "naver" | "hyperclova" => Provider::Clova,
            "yandex" | "yandexgpt" => Provider::Yandex,
            "gigachat" | "sber" => Provider::GigaChat,
            "upstage" | "solar" => Provider::Upstage,
            "sea-lion" | "sealion" | "aisingapore" => Provider::SeaLion,
            "alibaba" | "dashscope" | "qwen" => Provider::Alibaba,
            "baidu" | "ernie" => Provider::Baidu,
            "zhipu" | "glm" => Provider::Zhipu,
            "moonshot" | "kimi" => Provider::Moonshot,
            "rakuten" => Provider::Rakuten,
            "sarvam" => Provider::Sarvam,
            "ollama" => Provider::Local,
            _ => Provider::Custom,
        }
    }

    /// Detect provider from model name/ID.
    pub fn from_model(model: &str) -> Self {
        let model_lower = model.to_lowercase();

        // Check for LiteLLM-style prefix
        if let Some((prefix, _)) = model.split_once('/') {
            let provider = Self::from_prefix(prefix);
            if provider != Provider::Custom {
                return provider;
            }
        }

        // Detect from model name patterns
        if model_lower.starts_with("claude") {
            Provider::Anthropic
        } else if model_lower.starts_with("gpt-")
            || model_lower.starts_with("o1")
            || model_lower.starts_with("o3")
        {
            Provider::OpenAI
        } else if model_lower.starts_with("gemini") {
            Provider::Google
        } else if model_lower.starts_with("mistral")
            || model_lower.starts_with("codestral")
            || model_lower.starts_with("pixtral")
        {
            Provider::Mistral
        } else if model_lower.starts_with("deepseek") {
            Provider::DeepSeek
        } else if model_lower.starts_with("command") {
            Provider::Cohere
        } else if model_lower.starts_with("jamba") {
            Provider::AI21
        // Regional providers - Asia
        } else if model_lower.starts_with("palmyra") {
            Provider::Writer
        } else if model_lower.starts_with("sabia") {
            Provider::Maritaca
        } else if model_lower.starts_with("hcx") || model_lower.starts_with("hyperclova") {
            Provider::Clova
        } else if model_lower.starts_with("yandexgpt") {
            Provider::Yandex
        } else if model_lower.starts_with("gigachat") {
            Provider::GigaChat
        } else if model_lower.starts_with("solar") {
            Provider::Upstage
        } else if model_lower.contains("sea-lion") || model_lower.contains("sealion") {
            Provider::SeaLion
        } else if model_lower.starts_with("qwen") {
            Provider::Alibaba
        } else if model_lower.starts_with("ernie") {
            Provider::Baidu
        } else if model_lower.starts_with("glm") {
            Provider::Zhipu
        } else if model_lower.starts_with("kimi") {
            Provider::Moonshot
        } else if model_lower.starts_with("rakuten") {
            Provider::Rakuten
        } else if model_lower.starts_with("sarvam") {
            Provider::Sarvam
        } else if model_lower.starts_with("llama") || model_lower.starts_with("mixtral") {
            // Llama/Mixtral can be from multiple providers
            if std::env::var("GROQ_API_KEY").is_ok() {
                Provider::Groq
            } else if std::env::var("TOGETHER_API_KEY").is_ok() {
                Provider::TogetherAI
            } else {
                Provider::Local
            }
        } else {
            Provider::Custom
        }
    }

    /// Get the prefix used in LiteLLM-style IDs.
    pub fn prefix(&self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::OpenAI => "openai",
            Provider::Google => "google",
            Provider::Mistral => "mistral",
            Provider::Groq => "groq",
            Provider::DeepSeek => "deepseek",
            Provider::Cohere => "cohere",
            Provider::Bedrock => "bedrock",
            Provider::AzureOpenAI => "azure",
            Provider::VertexAI => "vertex_ai",
            Provider::TogetherAI => "together_ai",
            Provider::OpenRouter => "openrouter",
            Provider::Cerebras => "cerebras",
            Provider::SambaNova => "sambanova",
            Provider::Fireworks => "fireworks",
            Provider::AI21 => "ai21",
            Provider::HuggingFace => "huggingface",
            Provider::Replicate => "replicate",
            Provider::Cloudflare => "cloudflare",
            Provider::Databricks => "databricks",
            // Regional providers - Asia
            Provider::Writer => "writer",
            Provider::Maritaca => "maritaca",
            Provider::Clova => "clova",
            Provider::Yandex => "yandex",
            Provider::GigaChat => "gigachat",
            Provider::Upstage => "upstage",
            Provider::SeaLion => "sea-lion",
            Provider::Alibaba => "alibaba",
            Provider::Baidu => "baidu",
            Provider::Zhipu => "zhipu",
            Provider::Moonshot => "moonshot",
            Provider::Rakuten => "rakuten",
            Provider::Sarvam => "sarvam",
            Provider::Local => "ollama",
            Provider::Custom => "custom",
        }
    }

    /// Check if this provider supports prompt caching.
    pub fn supports_prompt_caching(&self) -> bool {
        matches!(
            self,
            Provider::Anthropic | Provider::Bedrock | Provider::Google | Provider::OpenAI
        )
    }
}

// ============================================================================
// MODEL STATUS
// ============================================================================

/// Model availability status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelStatus {
    /// Currently recommended model.
    Current,
    /// Still available but superseded by newer version.
    Legacy,
    /// Scheduled for removal, not recommended for new projects.
    Deprecated,
}

// ============================================================================
// PRICING
// ============================================================================

/// Model pricing (per 1M tokens in USD).
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelPricing {
    pub input_per_1m: f64,
    pub output_per_1m: f64,
    pub cached_input_per_1m: Option<f64>,
}

impl ModelPricing {
    pub const fn new(input: f64, output: f64) -> Self {
        Self {
            input_per_1m: input,
            output_per_1m: output,
            cached_input_per_1m: None,
        }
    }

    pub fn with_cache(mut self, cached: f64) -> Self {
        self.cached_input_per_1m = Some(cached);
        self
    }

    /// Estimate cost for given token counts.
    pub fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        (input_tokens as f64 / 1_000_000.0) * self.input_per_1m
            + (output_tokens as f64 / 1_000_000.0) * self.output_per_1m
    }

    /// Estimate cost with cached input tokens.
    pub fn estimate_cost_with_cache(
        &self,
        cached_tokens: u32,
        uncached_tokens: u32,
        output_tokens: u32,
    ) -> f64 {
        let cached_cost = self.cached_input_per_1m.unwrap_or(self.input_per_1m)
            * (cached_tokens as f64 / 1_000_000.0);
        let uncached_cost = self.input_per_1m * (uncached_tokens as f64 / 1_000_000.0);
        let output_cost = self.output_per_1m * (output_tokens as f64 / 1_000_000.0);
        cached_cost + uncached_cost + output_cost
    }
}

// ============================================================================
// CAPABILITIES
// ============================================================================

/// Model capabilities.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelCapabilities {
    pub max_context: u32,
    pub max_output: u32,
    pub vision: bool,
    pub tools: bool,
    pub streaming: bool,
    pub json_mode: bool,
    /// Full structured output with JSON schema enforcement.
    /// True when the provider supports strict JSON schema validation,
    /// not just simple JSON mode.
    pub structured_output: bool,
    pub thinking: bool,
    pub caching: bool,
}

impl ModelCapabilities {
    pub fn new(max_context: u32, max_output: u32) -> Self {
        Self {
            max_context,
            max_output,
            tools: true,
            streaming: true,
            ..Default::default()
        }
    }

    /// Parse capability flags: V=vision, T=tools, J=json, S=structured output, K=thinking, C=cache
    pub fn from_flags(max_context: u32, max_output: u32, flags: &str) -> Self {
        Self {
            max_context,
            max_output,
            vision: flags.contains('V'),
            tools: flags.contains('T'),
            streaming: true,
            json_mode: flags.contains('J'),
            structured_output: flags.contains('S'),
            thinking: flags.contains('K'),
            caching: flags.contains('C'),
        }
    }
}

// ============================================================================
// BENCHMARKS
// ============================================================================

/// Benchmark scores (0-100 scale, higher is better).
#[derive(Debug, Clone, Copy, Default)]
pub struct ModelBenchmarks {
    /// MMLU - General knowledge
    pub mmlu: Option<f32>,
    /// HumanEval - Code generation
    pub humaneval: Option<f32>,
    /// MATH - Mathematical reasoning
    pub math: Option<f32>,
    /// GPQA Diamond - Graduate-level science
    pub gpqa: Option<f32>,
    /// SWE-bench - Software engineering
    pub swe_bench: Option<f32>,
    /// IFEval - Instruction following
    pub ifeval: Option<f32>,
    /// MMMU - Multimodal understanding
    pub mmmu: Option<f32>,
    /// MGSM - Multilingual math
    pub mgsm: Option<f32>,
    /// Time to first token (ms)
    pub ttft_ms: Option<u32>,
    /// Tokens per second
    pub tokens_per_sec: Option<u32>,
}

impl ModelBenchmarks {
    /// Calculate weighted quality score (0-100).
    #[allow(clippy::type_complexity)]
    pub fn quality_score(&self) -> f32 {
        const WEIGHTS: [(fn(&ModelBenchmarks) -> Option<f32>, f32); 8] = [
            (|b| b.mmlu, 1.5),
            (|b| b.humaneval, 2.0),
            (|b| b.math, 1.5),
            (|b| b.gpqa, 1.0),
            (|b| b.swe_bench, 2.5),
            (|b| b.ifeval, 1.0),
            (|b| b.mmmu, 1.5),
            (|b| b.mgsm, 1.0),
        ];

        let mut sum = 0.0f32;
        let mut weights = 0.0f32;

        for (getter, weight) in WEIGHTS {
            if let Some(s) = getter(self) {
                sum += s * weight;
                weights += weight;
            }
        }

        if weights > 0.0 {
            sum / weights
        } else {
            0.0
        }
    }

    pub fn has_benchmarks(&self) -> bool {
        self.mmlu.is_some() || self.humaneval.is_some() || self.math.is_some()
    }
}

// ============================================================================
// MODEL INFO
// ============================================================================

/// Complete model specification.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// LiteLLM-compatible model ID (e.g., "anthropic/claude-3-5-sonnet")
    pub id: String,
    /// Short alias (e.g., "claude-3-5-sonnet")
    pub alias: Option<String>,
    /// Human-readable name
    pub name: String,
    /// Provider
    pub provider: Provider,
    /// Status
    pub status: ModelStatus,
    /// Pricing
    pub pricing: ModelPricing,
    /// Capabilities
    pub capabilities: ModelCapabilities,
    /// Benchmarks
    pub benchmarks: ModelBenchmarks,
    /// Description
    pub description: String,
    /// Can be used as a classifier (fast, cheap, good instruction following)
    pub can_classify: bool,
}

impl ModelInfo {
    /// Get the raw model ID (without provider prefix).
    pub fn raw_id(&self) -> &str {
        self.id
            .split_once('/')
            .map(|(_, id)| id)
            .unwrap_or(&self.id)
    }

    /// Calculate quality per dollar (higher is better value).
    pub fn quality_per_dollar(&self) -> f64 {
        let quality = self.benchmarks.quality_score();
        let avg_cost = (self.pricing.input_per_1m + self.pricing.output_per_1m) / 2.0;
        if avg_cost > 0.0 && quality > 0.0 {
            quality as f64 / avg_cost
        } else {
            0.0
        }
    }

    /// Estimate cost for a request.
    pub fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> f64 {
        self.pricing.estimate_cost(input_tokens, output_tokens)
    }
}

// ============================================================================
// MODEL DATA (Compact DSL format)
// ============================================================================

/// Compact model data format:
/// id|alias|name|status|pricing|context|caps|benchmarks|description|classify
///
/// - id: LiteLLM-style (anthropic/claude-3-5-sonnet)
/// - alias: short name or - for none
/// - name: display name
/// - status: C=current, L=legacy, D=deprecated
/// - pricing: in,out[,cache]
/// - context: max_ctx,max_out
/// - caps: V=vision,T=tools,J=json,S=structured output,K=thinking,C=cache
/// - benchmarks: mmlu,human,math,gpqa,swe,if,mmmu,mgsm,ttft,tps (- for none)
/// - description: model description
/// - classify: Y/N
const MODEL_DATA: &str = r#"
# =============================================================================
# ANTHROPIC (all support structured output via beta header)
# =============================================================================
anthropic/claude-opus-4-5-20251101|claude-opus-4-5|Claude Opus 4.5|C|5.0,25.0,0.5|200000,32000|VTJSKC|92.3,95.8,87.4,68.7,55.2,92.1,71.5,91.5,1200,60|Premium model with maximum intelligence|N
anthropic/claude-sonnet-4-5-20250929|claude-sonnet-4-5|Claude Sonnet 4.5|C|3.0,15.0,0.3|200000,64000|VTJSKC|90.1,93.7,82.8,62.4,49.7,89.3,68.2,91.0,600,120|Best balance of intelligence, speed, and cost|Y
anthropic/claude-haiku-4-5-20251001|claude-haiku-4-5|Claude Haiku 4.5|C|1.0,5.0,0.1|200000,64000|VTJSKC|85.7,88.4,71.2,51.8,35.6,85.2,60.5,88.5,300,200|Fastest model with extended thinking|Y
anthropic/claude-3-7-sonnet-20250219|claude-3-7-sonnet|Claude 3.7 Sonnet|C|3.0,15.0,0.3|200000,128000|VTJSKC|89.5,93.0,80.5,61.2,47.8,88.8,71.6,92.4,700,100|Hybrid reasoning model|Y
anthropic/claude-3-5-sonnet-20241022|claude-3-5-sonnet|Claude 3.5 Sonnet|L|3.0,15.0,0.3|200000,8192|VTJSC|88.7,92.0,78.3,59.4,45.2,88.1,68.9,92.5,800,100|Previous generation|Y
anthropic/claude-3-5-haiku-20241022|claude-3-5-haiku|Claude 3.5 Haiku|L|0.8,4.0,0.08|200000,8192|VTJSC|80.4,82.1,65.7,45.2,28.4,82.8,65.0,85.0,350,180|Previous Haiku|Y
anthropic/claude-3-haiku-20240307|claude-3-haiku|Claude 3 Haiku|D|0.25,1.25,0.03|200000,4096|VTJC|75.2,75.8,55.4,38.1,18.3,78.5,55.0,78.0,250,220|Cheapest Claude|Y

# =============================================================================
# OPENAI (all support structured output natively)
# =============================================================================
openai/gpt-4o|gpt-4o|GPT-4o|C|2.5,10.0|128000,16384|VTJS|88.7,90.2,76.6,53.6,38.4,86.5,69.1,90.6,500,90|OpenAI flagship multimodal|Y
openai/gpt-4o-mini|gpt-4o-mini|GPT-4o Mini|C|0.15,0.6|128000,16384|VTJS|82.0,87.0,70.2,43.8,24.8,80.6,59.4,82.5,200,180|Fast and very cheap|Y
openai/gpt-4.1|gpt-4.1|GPT-4.1|C|2.0,8.0|1000000,32768|VTJS|89.2,91.5,78.8,55.2,40.1,87.8,70.5,91.2,450,100|1M context model|Y
openai/gpt-4.1-mini|gpt-4.1-mini|GPT-4.1 Mini|C|0.4,1.6|1000000,32768|VTJS|84.5,88.2,72.4,46.8,28.5,82.4,62.1,84.5,180,200|Fast 1M context|Y
openai/o1|o1|o1|C|15.0,60.0|200000,100000|JSK|91.8,92.8,94.8,78.3,48.9,90.5,-,88.9,5000,30|Extended thinking for complex problems|N
openai/o1-mini|o1-mini|o1-mini|C|1.1,4.4|128000,65536|JSK|85.2,92.4,90.0,60.0,28.7,82.1,-,85.5,2000,50|Fast extended thinking|Y
openai/o3|o3|o3|C|10.0,40.0|200000,100000|JSK|93.5,95.2,96.0,75.0,48.0,92.8,-,93.5,4000,35|Next-gen extended thinking|N
openai/o3-mini|o3-mini|o3-mini|C|1.1,4.4|200000,100000|JSK|86.8,92.5,96.7,75.2,49.3,88.4,-,91.6,1500,60|Fast o3 variant|Y
openai-realtime/gpt-4o-realtime-preview|gpt-4o-realtime-preview|GPT-4o Realtime Preview|C|0.3,1.2|128000,4096|VT|-,-,-,-,-,-,-,-,-,-|Real-time audio/text streaming (WebSocket)|Y
openai-realtime/gpt-realtime|gpt-realtime|GPT Realtime|C|0.5,2.0|128000,4096|VT|-,-,-,-,-,-,-,-,-,-|Latest Realtime model|Y
openai-realtime/gpt-realtime-mini|gpt-realtime-mini|GPT Realtime Mini|C|0.1,0.4|128000,4096|VT|-,-,-,-,-,-,-,-,-,-|Lightweight Realtime model|Y

# =============================================================================
# GOOGLE (all support structured output via responseSchema)
# =============================================================================
google/gemini-3-pro|gemini-3-pro|Gemini 3 Pro|C|2.0,10.0,0.5|2000000,16384|VTJSKC|93.5,94.2,88.5,72.4,62.1,91.5,76.8,95.2,800,80|Latest flagship with deep think reasoning|N
google/gemini-3-flash|gemini-3-flash|Gemini 3 Flash|C|0.1,0.4,0.025|1000000,8192|VTJSK|89.2,90.5,82.4,65.2,54.3,87.8,70.2,91.5,300,200|High-speed reasoning with deep think|Y
google/gemini-2.5-pro|gemini-2.5-pro|Gemini 2.5 Pro|C|1.25,10.0,0.3125|2000000,16384|VTJSKC|90.2,92.5,84.8,65.2,48.5,88.5,72.8,92.2,600,100|Advanced 2M context model|Y
google/gemini-2.5-flash|gemini-2.5-flash|Gemini 2.5 Flash|C|0.075,0.30|1000000,8192|VTJSK|84.2,88.5,74.8,52.4,32.5,82.8,62.5,88.0,250,200|Ultra-fast with extended thinking|Y
google/gemini-2.0-flash-exp|gemini-2.0-flash-exp|Gemini 2.0 Flash Exp|C|0.1,0.4|1000000,8192|VTJSK|87.2,91.8,84.5,61.2,44.8,86.5,71.2,90.8,500,120|Experimental deep thinking enabled|Y
google/gemini-2.0-flash|gemini-2.0-flash|Gemini 2.0 Flash|C|0.1,0.4|1000000,8192|VTJSK|82.5,86.2,70.5,48.2,25.3,80.1,58.7,85.0,200,180|Ultra-fast multimodal with extended thinking|Y
google/gemini-1.5-pro|gemini-1.5-pro|Gemini 1.5 Pro|L|1.25,5.0|2000000,8192|VTJS|85.9,84.1,67.7,46.2,28.8,82.1,62.2,89.3,800,80|2M context|Y
google/gemini-1.5-flash|gemini-1.5-flash|Gemini 1.5 Flash|L|0.075,0.3|1000000,8192|VTJS|78.9,74.3,54.9,39.5,18.6,76.8,56.4,80.0,300,150|Fast and cheap|Y
google/medpalm-2|medpalm-2|Med-PaLM 2|C|0.5,1.0|8192,2048|TJ|85.2,72.5,65.8,48.1,-,80.5,58.2,82.1,800,80|Medical domain specialist|N

# =============================================================================
# VERTEX AI PARTNERS (Google Cloud marketplace)
# =============================================================================
vertex-google/gemini-3-pro|vertex-gemini-3-pro|Gemini 3 Pro (Vertex)|C|2.0,10.0,0.5|2000000,16384|VTJSKC|93.5,94.2,88.5,72.4,62.1,91.5,76.8,95.2,800,80|Latest flagship with deep think reasoning|N
vertex-google/gemini-3-flash|vertex-gemini-3-flash|Gemini 3 Flash (Vertex)|C|0.1,0.4,0.025|1000000,8192|VTJSK|89.2,90.5,82.4,65.2,54.3,87.8,70.2,91.5,300,200|High-speed reasoning with deep think|Y
vertex-anthropic/claude-3.5-sonnet|claude-3.5-sonnet|Claude 3.5 Sonnet|C|3.0,15.0|200000,4096|VTJSKC|88.5,92.1,81.4,62.3,45.2,87.5,75.5,91.8,400,100|Latest Anthropic model|Y
vertex-anthropic/claude-3-opus|claude-3-opus|Claude 3 Opus|C|15.0,75.0|200000,4096|VTJSKC|89.2,93.5,82.8,64.1,48.5,89.1,78.2,93.2,800,50|Most capable Claude|Y
vertex-deepseek/deepseek-chat|deepseek-chat|DeepSeek Chat (Vertex)|C|0.27,0.55|64000,8192|TJ|87.5,91.6,84.6,59.1,42.0,86.2,-,90.7,400,120|Cost effective reasoning|Y
vertex-deepseek/deepseek-reasoner|deepseek-reasoner|DeepSeek Reasoner (Vertex)|C|1.10,4.40|64000,8192|JSKC|90.8,91.0,90.0,71.5,49.2,88.4,-,-,3000,40|Advanced reasoning|N
vertex-llama/llama-3.1-405b|llama-405b|Llama 3.1 405B (Vertex)|C|3.15,4.72|128000,4096|VTJ|87.2,89.4,79.8,58.6,38.2,85.8,-,-,800,80|Largest open model|Y
vertex-llama/llama-3.1-70b|llama-70b|Llama 3.1 70B (Vertex)|C|0.59,0.79|128000,4096|VTJ|85.8,86.5,76.2,55.4,32.8,84.5,-,-,500,120|Excellent balance|Y
vertex-mistral/mistral-large|mistral-large|Mistral Large (Vertex)|C|1.0,3.0|262000,4096|VTJ|88.5,86.8,75.4,55.8,38.5,85.2,-,-,500,100|Powerful MoE model|Y
vertex-mistral/mistral-medium|mistral-medium|Mistral Medium (Vertex)|C|0.8,2.4|128000,4096|VTJ|85.2,84.5,70.8,52.1,32.5,83.5,-,-,400,120|Balanced|Y
vertex-ai21/j2-ultra|j2-ultra|J2 Ultra (Vertex)|C|0.016,0.08|8192,4096|TJ|78.5,75.2,58.4,42.1,22.5,78.2,-,-,200,180|Fast efficient|Y
vertex-ai21/j2-mid|j2-mid|J2 Mid (Vertex)|C|0.008,0.04|8192,4096|TJ|75.2,72.8,55.1,38.5,18.2,75.8,-,-,150,200|Cost friendly|Y

# =============================================================================
# MISTRAL (supports structured output via response_format, EU regional endpoint)
# =============================================================================
mistral/mistral-large-2512|mistral-large-3|Mistral Large 3|C|0.5,1.5|262000,8192|VTJ|88.5,86.8,75.4,55.8,38.5,85.2,-,-,500,100|675B MoE flagship with EU regional support|Y
mistral/mistral-medium-3.1|mistral-medium-3.1|Mistral Medium 3.1|C|0.4,1.2|128000,8192|VTJ|85.2,84.5,70.8,52.1,32.5,83.5,-,-,400,120|Balanced tier with regional compliance|Y
mistral/mistral-small-3.1|mistral-small-3.1|Mistral Small 3.1|C|0.05,0.15|128000,8192|TJ|78.5,76.8,58.4,42.1,22.5,78.2,-,-,200,180|Fast efficient inference|Y
mistral/codestral-2501|codestral|Codestral|C|0.3,0.9|256000,8192|TJ|78.2,87.8,62.4,42.1,35.2,80.5,-,-,400,150|Code specialist model|Y

# =============================================================================
# DEEPSEEK (OpenAI-compatible, supports structured output with extended thinking)
# =============================================================================
deepseek/deepseek-chat|deepseek-v3|DeepSeek V3|C|0.14,0.28|64000,8192|TJ|87.5,91.6,84.6,59.1,42.0,86.2,-,90.7,400,120|Excellent value|Y
deepseek/deepseek-reasoner|deepseek-r1|DeepSeek R1|C|0.55,2.19,0.14|64000,8192|JSKC|90.8,91.0,90.0,71.5,49.2,88.4,-,-,3000,40|Advanced reasoning with 71% AIME pass rate|N

# =============================================================================
# COHERE (supports structured output via response_format)
# =============================================================================
cohere/command-r-plus-08-2024|command-r-plus|Command R+|C|2.50,10.00|128000,4096|TJ|75.7,71.6,48.5,43.2,17.8,76.4,-,-,700,80|Enterprise RAG|Y
cohere/command-r-08-2024|command-r|Command R|C|0.15,0.60|128000,4096|TJ|68.2,62.4,38.6,35.8,11.2,71.5,-,-,400,120|32B affordable|Y

# =============================================================================
# GROQ (Fast inference, OpenAI-compatible)
# =============================================================================
groq/llama-3.3-70b-versatile|groq-llama-3.3-70b|Llama 3.3 70B (Groq)|C|0.59,0.79|128000,32768|TJ|85.8,82.5,68.4,48.2,30.5,82.8,-,-,100,500|Ultra-fast Llama|Y
groq/llama-3.1-8b-instant|groq-llama-3.1-8b|Llama 3.1 8B (Groq)|C|0.05,0.08|128000,8192|TJ|73.0,72.6,51.9,32.4,14.2,72.8,-,-,50,800|Fastest inference|Y
groq/mixtral-8x7b-32768|groq-mixtral|Mixtral 8x7B (Groq)|C|0.24,0.24|32768,8192|TJ|70.6,74.8,54.3,36.4,14.6,73.5,-,-,80,600|Fast MoE|Y

# =============================================================================
# CEREBRAS (Ultra-fast inference, OpenAI-compatible)
# =============================================================================
cerebras/llama-3.3-70b|cerebras-llama-3.3-70b|Llama 3.3 70B (Cerebras)|C|0.60,0.60|128000,8192|TJ|85.8,82.5,68.4,48.2,30.5,82.8,-,-,30,1800|Ultra-fast inference|Y
cerebras/llama-3.1-8b|cerebras-llama-3.1-8b|Llama 3.1 8B (Cerebras)|C|0.10,0.10|128000,8192|TJ|73.0,72.6,51.9,32.4,14.2,72.8,-,-,15,2500|Fastest small model|Y

# =============================================================================
# SAMBANOVA (Ultra-fast inference, OpenAI-compatible)
# =============================================================================
sambanova/llama-3.3-70b|sambanova-llama-3.3-70b|Llama 3.3 70B (SambaNova)|C|0.40,0.40|128000,8192|TJ|85.8,82.5,68.4,48.2,30.5,82.8,-,-,40,1000|Ultra-fast|Y
sambanova/deepseek-r1|sambanova-deepseek-r1|DeepSeek R1 (SambaNova)|C|0.50,2.00|64000,8192|TJSK|90.8,91.0,90.0,71.5,49.2,88.4,-,-,500,200|Extended thinking|N

# =============================================================================
# FIREWORKS (Fast inference, OpenAI-compatible)
# =============================================================================
fireworks/llama-3.3-70b|fireworks-llama-3.3-70b|Llama 3.3 70B (Fireworks)|C|0.90,0.90|131072,8192|TJ|85.8,82.5,68.4,48.2,30.5,82.8,-,-,60,500|Fast inference|Y
fireworks/deepseek-v3|fireworks-deepseek-v3|DeepSeek V3 (Fireworks)|C|0.40,0.80|64000,8192|TJ|87.5,91.6,84.6,59.1,42.0,86.2,-,90.7,100,400|Excellent value|Y

# =============================================================================
# AI21 (Jamba models, supports structured output)
# =============================================================================
ai21/jamba-2.0-large|jamba-2.0-large|Jamba 2.0 Large|C|2.00,8.00|256000,8192|TJ|86.5,84.2,72.5,52.8,35.2,84.8,-,-,400,100|256K hybrid SSM|Y
ai21/jamba-2.0-mini|jamba-2.0-mini|Jamba 2.0 Mini|C|0.20,0.40|256000,8192|TJ|78.5,76.8,62.4,42.1,22.5,78.2,-,-,150,200|Fast hybrid SSM|Y

# =============================================================================
# TOGETHER AI (OpenAI-compatible)
# =============================================================================
together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo|llama-3.3-70b-together|Llama 3.3 70B (Together)|C|0.88,0.88|131072,8192|TJ|85.8,82.5,68.4,48.2,30.5,82.8,-,-,200,200|Recommended starter|Y
together_ai/deepseek-ai/DeepSeek-V3|deepseek-v3-together|DeepSeek V3 (Together)|C|1.25,1.25|163839,8192|TJ|87.5,91.6,84.6,59.1,42.0,86.2,-,90.7,350,120|Excellent MoE|Y
together_ai/deepseek-ai/DeepSeek-R1|deepseek-r1-together|DeepSeek R1 (Together)|C|0.55,2.19|64000,8192|TJSKC|90.8,91.0,90.0,71.5,49.2,88.4,-,-,3000,40|Reasoning via Together AI|N

# =============================================================================
# AWS BEDROCK (Anthropic models support structured output)
# =============================================================================
bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0|bedrock-claude-sonnet-4-5|Claude Sonnet 4.5 (Bedrock)|C|3.0,15.0,0.3|200000,8192|VTJSKC|90.1,93.7,82.8,62.4,49.7,89.3,68.2,-,600,120|Best balance on Bedrock|Y
bedrock/anthropic.claude-haiku-4-5-20251001-v1:0|bedrock-claude-haiku-4-5|Claude Haiku 4.5 (Bedrock)|C|1.0,5.0,0.1|200000,8192|VTJSKC|85.7,88.4,71.2,51.8,35.6,85.2,60.5,-,300,200|Fast on Bedrock|Y
bedrock/amazon.nova-pro-v1:0|nova-pro|Amazon Nova Pro|C|0.8,3.2|300000,5000|VTJS|85.5,84.2,72.8,52.1,32.5,84.7,62.8,-,500,110|Best accuracy/cost|Y
bedrock/amazon.nova-lite-v1:0|nova-lite|Amazon Nova Lite|C|0.06,0.24|300000,5000|VTJS|77.8,75.3,60.2,41.5,18.7,78.2,55.4,-,250,180|Cost-effective|Y
bedrock/meta.llama3-3-70b-instruct-v1:0|bedrock-llama-3.3-70b|Llama 3.3 70B (Bedrock)|C|0.99,0.99|128000,2048|TJ|85.8,82.5,68.4,48.2,30.5,82.8,-,-,600,100|Balanced|Y

# =============================================================================
# OPENROUTER (Aggregator - passthrough to underlying provider)
# =============================================================================
openrouter/openai/gpt-4o|openrouter-gpt-4o|GPT-4o (OpenRouter)|C|2.5,10.0|128000,16384|VTJS|88.7,90.2,76.6,53.6,38.4,86.5,69.1,90.6,500,90|Via OpenRouter|Y
openrouter/openai/o1|openrouter-o1|o1 (OpenRouter)|C|15.0,60.0|200000,100000|JSK|91.8,92.8,94.8,78.3,48.9,90.5,-,88.9,5000,30|OpenAI reasoning via OpenRouter|N
openrouter/openai/o3|openrouter-o3|o3 (OpenRouter)|C|10.0,40.0|200000,100000|JSK|93.5,95.2,96.0,75.0,48.0,92.8,-,93.5,4000,35|Next-gen reasoning via OpenRouter|N
openrouter/anthropic/claude-opus-4.5|openrouter-claude-opus-4.5|Claude Opus 4.5 (OpenRouter)|C|5.0,25.0|200000,32000|VTJSKC|92.3,95.8,87.4,68.7,55.2,92.1,71.5,-,1200,60|Via OpenRouter|N
openrouter/anthropic/claude-sonnet-4.5|openrouter-claude-sonnet-4.5|Claude Sonnet 4.5 (OpenRouter)|C|3.0,15.0|200000,64000|VTJSKC|90.1,93.7,82.8,62.4,49.7,89.3,68.2,-,600,120|Claude thinking via OpenRouter|Y
openrouter/anthropic/claude-haiku-4.5|openrouter-claude-haiku-4-5|Claude Haiku 4.5 (OpenRouter)|C|1.0,5.0,0.1|200000,64000|VTJSKC|85.7,88.4,71.2,51.8,35.6,85.2,60.5,-,300,200|Via OpenRouter|Y

# =============================================================================
# CLOUDFLARE (Edge inference - limited structured output)
# =============================================================================
cloudflare/@cf/meta/llama-3.3-70b-instruct-fp8-fast|cf-llama-3.3-70b|Llama 3.3 70B (Cloudflare)|C|0.50,0.50|128000,8192|TJ|85.8,82.5,68.4,48.2,30.5,82.8,-,-,100,400|Edge inference|Y
cloudflare/@cf/meta/llama-3.1-8b-instruct|cf-llama-3.1-8b|Llama 3.1 8B (Cloudflare)|C|0.05,0.05|128000,8192|TJ|73.0,72.6,51.9,32.4,14.2,72.8,-,-,50,600|Fast edge|Y

# =============================================================================
# DATABRICKS (Enterprise, OpenAI-compatible)
# =============================================================================
databricks/databricks-llama-3.3-70b|databricks-llama-3.3-70b|Llama 3.3 70B (Databricks)|C|0.85,0.85|128000,8192|TJS|85.8,82.5,68.4,48.2,30.5,82.8,-,-,200,250|Enterprise|Y
databricks/databricks-dbrx-instruct|databricks-dbrx|DBRX Instruct (Databricks)|C|0.75,2.25|32768,8192|TJS|78.5,76.2,58.4,42.5,22.8,78.2,-,-,300,180|132B MoE|Y

# =============================================================================
# WRITER (Palmyra - Enterprise LLM with 1M context)
# =============================================================================
writer/palmyra-x5|palmyra-x5|Palmyra X5|C|2.00,8.00|1000000,8192|TJS|82.5,78.4,68.5,48.2,28.5,80.5,-,-,600,100|1M context enterprise LLM|Y
writer/palmyra-x4|palmyra-x4|Palmyra X4|L|1.50,6.00|128000,8192|TJS|78.5,74.2,62.4,42.1,22.5,76.2,-,-,500,120|Previous generation|Y

# =============================================================================
# MARITACA (Sabiá - Portuguese specialist from Brazil)
# =============================================================================
maritaca/sabia-3|sabia-3|Sabiá 3|C|0.50,2.00|32000,4096|TJ|75.5,72.4,58.5,38.2,18.5,74.5,-,92.8,400,150|Portuguese/Brazilian specialist with model discovery|Y
maritaca/sabia-2-small|sabia-2-small|Sabiá 2 Small|C|0.10,0.40|32000,4096|TJ|68.5,64.2,48.5,32.1,12.5,68.2,-,85.5,200,250|Fast Portuguese model|Y

# =============================================================================
# CLOVA (Naver HyperCLOVA X - Korean specialist)
# =============================================================================
clova/HCX-007|hcx-007|HyperCLOVA X 007|C|1.50,6.00|128000,8192|TJ|80.5,75.4,65.5,45.2,25.5,78.5,-,88.5,500,120|Korean reasoning model|Y
clova/HCX-005|hcx-005|HyperCLOVA X 005|C|2.00,8.00|128000,8192|VTJ|82.5,78.4,68.5,48.2,28.5,80.5,58.5,90.2,600,100|Korean multimodal flagship|Y
clova/HCX-DASH-002|hcx-dash-002|HyperCLOVA X DASH|C|0.30,1.20|128000,8192|TJ|72.5,68.4,55.5,38.2,18.5,72.5,-,82.5,200,200|Fast lightweight Korean|Y

# =============================================================================
# YANDEX (YandexGPT - Russian specialist)
# =============================================================================
yandex/yandexgpt-pro|yandexgpt-pro|YandexGPT Pro|C|1.20,4.80|32000,8192|TJ|78.5,72.4,62.5,42.2,22.5,76.5,-,88.5,500,120|Russian flagship model|Y
yandex/yandexgpt-lite|yandexgpt-lite|YandexGPT Lite|C|0.30,1.20|32000,8192|TJ|68.5,62.4,48.5,32.2,14.5,68.5,-,78.5,200,200|Fast Russian model|Y

# =============================================================================
# GIGACHAT (Sber - Russian enterprise)
# =============================================================================
gigachat/gigachat-pro|gigachat-pro|GigaChat Pro|C|1.00,4.00|32000,8192|TJ|76.5,70.4,60.5,40.2,20.5,74.5,-,86.5,400,140|Russian enterprise model|Y
gigachat/gigachat|gigachat|GigaChat|C|0.20,0.80|32000,8192|TJ|68.5,62.4,48.5,32.2,14.5,68.5,-,78.5,200,200|Russian base model|Y

# =============================================================================
# UPSTAGE (Solar - Korean AI startup, AWS partnership)
# =============================================================================
upstage/solar-pro|solar-pro|Solar Pro|C|0.80,3.20|128000,8192|TJS|82.5,80.4,70.5,48.2,30.5,80.5,-,-,400,150|Korean flagship model|Y
upstage/solar-mini|solar-mini|Solar Mini|C|0.15,0.60|128000,8192|TJS|74.5,72.4,58.5,38.2,20.5,72.5,-,-,150,280|Fast Korean model|Y

# =============================================================================
# SEA-LION (AI Singapore - Southeast Asian languages)
# =============================================================================
sea-lion/Qwen-SEA-LION-v4-32B-IT|sea-lion-32b|SEA-LION v4 32B|C|0.40,1.60|128000,8192|VTJS|80.5,78.4,65.5,45.2,25.5,78.5,55.5,-,350,160|11 SEA languages|Y
sea-lion/SEA-LION-v3-8B|sea-lion-8b|SEA-LION v3 8B|C|0.08,0.32|32000,4096|TJS|68.5,65.4,52.5,35.2,15.5,68.5,-,-,120,350|Fast SEA languages|Y

# =============================================================================
# DATAROBOT (ML Ops platform)
# =============================================================================
datarobot/autopilot-default|-|DataRobot Autopilot|C|0.50,1.50|32000,4096|TJS|70.5,68.4,55.5,40.2,22.5,70.5,-,-,250,200|ML Ops inference|Y

# =============================================================================
# STABILITY AI (Image generation)
# =============================================================================
stability/stable-diffusion-3.5-large|-|Stable Diffusion 3.5 Large|C|0.20,0.40|8192,4096|-|75.0,78.2,68.5,50.2,32.0,76.5,-,-,800,100|Latest SD3.5 flagship|Y
stability/stable-diffusion-3-large|-|Stable Diffusion 3 Large|C|0.18,0.35|8192,4096|-|72.5,76.0,65.0,48.2,28.0,73.0,-,-,700,120|SD3 flagship|Y
stability/stable-diffusion-3-medium|-|Stable Diffusion 3 Medium|C|0.10,0.20|8192,4096|-|68.5,72.4,60.0,44.2,22.0,68.5,-,-,400,200|SD3 balanced|Y

# AWS SageMaker custom model endpoints
sagemaker/custom-endpoint|-|SageMaker Custom Model|C|0.50,1.50|32000,4096|TJS|72.0,74.0,62.0,45.0,28.0,71.0,-,-,300,150|Custom deployed model|Y

# Snowflake Cortex LLM
snowflake/cortex-llm|-|Snowflake Cortex LLM|C|0.80,2.40|64000,4096|VTJS|80.0,82.0,70.0,52.0,32.0,79.0,-,-,500,100|Data warehouse integrated|Y

# Exa AI Semantic Search (Phase 4)
exa/semantic-search|-|Exa Semantic Search|C|0.01,0.05|-|92.0,94.0,88.0,70.0,50.0,91.0,-,-,100,50|Semantic web search for LLMs|Y

# Brave Search API (Phase 4)
brave-search/web-search|-|Brave Web Search|C|0.00,0.00|-|90.0,92.0,85.0,68.0,48.0,89.0,-,-,100,50|Privacy-focused web search|Y
brave-search/web-with-summary|-|Brave Search + Summary|C|0.01,0.02|-|91.0,93.0,86.0,69.0,49.0,90.0,-,-,100,50|Web search with AI summary|Y

# =============================================================================
# VIDEO GENERATION (Runware aggregator with 5+ models)
# =============================================================================
runware/runway-gen-4.5|-|Runway Gen 4.5 (Runware)|C|1.50,6.00|8192,4096|-|-,-,-,-,-,-,-,-,1200,20|4K video generation|N
runware/kling-2.0|-|Kling 2.0 (Runware)|C|0.50,2.00|8192,4096|-|-,-,-,-,-,-,-,-,1200,20|1080p video generation|N
runware/pika-1.0|-|Pika 1.0 (Runware)|C|0.75,3.00|8192,4096|-|-,-,-,-,-,-,-,-,1200,20|HD video generation|N
runware/hailuo-mini|-|Hailuo Mini (Runware)|C|0.60,2.40|8192,4096|-|-,-,-,-,-,-,-,-,1200,20|Cost-effective video generation|N
runware/leonardo-ultra|-|Leonardo Ultra (Runware)|C|0.80,3.20|8192,4096|-|-,-,-,-,-,-,-,-,1200,20|Artistic video generation|N

# =============================================================================
# AUDIO/VOICE PROVIDERS (Real-time Voice with latency control)
# =============================================================================
deepgram/nova-3-general|-|Deepgram Nova-3 General|C|0.003,0.003|10000,2000|-|85.0,78.0,72.0,55.0,-,82.0,-,-,500,250|Speech recognition with improved accuracy|Y
deepgram/nova-3-meeting|-|Deepgram Nova-3 Meeting|C|0.003,0.003|10000,2000|-|86.0,79.0,73.0,56.0,-,83.0,-,-,500,250|Meeting-optimized speech recognition|Y
elevenlabs/tts-v1|-|ElevenLabs TTS v1|C|0.15,0.15|8192,4096|-|-,-,-,-,-,-,-,-,150,150|Text-to-speech with adjustable latency|Y

# =============================================================================
# ALIBABA - QWEN (DashScope platform - verified pricing)
# =============================================================================
alibaba/qwen-max|qwen-max|Qwen Max|C|1.26,6.30|32000,2048|TJ|82.5,80.2,70.5,52.1,28.5,80.5,-,85.5,600,120|Flagship reasoning model with official pricing|Y
alibaba/qwen-plus|qwen-plus|Qwen Plus|C|0.5,1.5|32000,2048|TJ|75.5,72.4,58.5,42.1,18.5,74.5,-,78.5,400,150|Balanced performance (estimated pricing)|Y
alibaba/qwen-turbo|qwen-turbo|Qwen Turbo|C|0.25,0.75|32000,2048|TJ|68.5,65.4,50.5,32.2,12.5,68.5,-,72.5,200,250|Fast and cost-effective (estimated pricing)|Y
alibaba/qwen-max-longcontext|qwen-max-longcontext|Qwen Max Long Context|C|1.26,6.30|200000,2048|TJ|82.5,80.2,70.5,52.1,28.5,80.5,-,85.5,800,100|Extended context support (estimated pricing)|Y

# =============================================================================
# BAIDU - ERNIE (Verified official pricing)
# =============================================================================
baidu/ernie-4.5-turbo-128k|ernie-4-turbo|ERNIE 4.5 Turbo|C|0.55,2.20|128000,2048|TJ|78.5,76.4,65.5,45.2,25.5,76.5,-,82.5,500,120|Official ERNIE 4.5 pricing from Qianfan|Y

# =============================================================================
# ZHIPU - GLM (Official pricing for GLM-4.7)
# =============================================================================
zhipu/glm-4.7|glm-4.7|GLM 4.7|C|0.60,2.20|128000,2048|TJS|82.5,84.2,72.5,52.8,35.2,82.8,-,88.5,600,100|Latest GLM with official pricing|Y
zhipu/glm-4|glm-4|GLM 4|C|0.6,2.2|128000,2048|TJS|80.5,82.4,70.5,50.8,32.5,80.8,-,86.5,500,120|General-purpose reasoning model|Y

# =============================================================================
# MOONSHOT - KIMI (Official pricing for K2)
# =============================================================================
moonshot/kimi-k2|kimi-k2|Kimi K2|C|0.15,2.50|200000,2048|TJ|80.5,78.4,68.5,48.2,28.5,78.5,-,84.5,700,100|Extended context with official pricing|Y

# =============================================================================
# CONTINGENT PROVIDERS (API access pending)
# =============================================================================
lighton/lighton-openai|-|LightOn GDPR Model|D|0.50,1.50|32000,4096|TJ|72.0,68.0,55.0,38.0,-,70.0,-,-,400,120|France/EU GDPR-compliant - partnership pending|Y
latamgpt/latamgpt-es|-|LatamGPT Spanish|D|0.30,1.20|16000,4096|TJ|70.0,66.0,52.0,36.0,-,68.0,-,-,300,150|Latin America regional - API launching soon|Y
grok/grok-realtime|-|Grok Real-Time Voice|D|0.50,2.00|32000,4096|-|-,-,-,-,-,-,-,-,500,100|xAI realtime voice - API access pending|N
chatlaw/chatlaw-v1|-|ChatLAW Legal AI|D|1.00,4.00|16000,4096|TJ|68.0,55.0,48.0,35.0,-,72.0,-,-,800,60|Legal domain specialist - API access pending|N
Processing latest_releases.csv...
✓ Processed 10 models from latest_releases.csv

Generation complete. Total lines: 441

# =============================================================================
# OPENROUTER (Meta-aggregator with 353+ models)
# =============================================================================
bytedance-seed/seed-1.6-flash|seed-1.6-flash|ByteDance Seed: Seed 1.6 Flash|C|7.0000000000e-08,3.0000000000e-07|262144,16384|JKSTV|-,-,-,-,-,-,-,-,-,262144|Seed 1.6 Flash is an ultra-fast multimodal deep thinking model by ByteDance Seed, supporting both te|Y
bytedance-seed/seed-1.6|seed-1.6|ByteDance Seed: Seed 1.6|C|2.5000000000e-07,2.0000000000e-06|262144,32768|JKSTV|-,-,-,-,-,-,-,-,-,262144|Seed 1.6 is a general-purpose model released by the ByteDance Seed team. It incorporates multimodal|Y
minimax/minimax-m2.1|minimax-m2.1|MiniMax: MiniMax M2.1|C|1.2000000000e-07,4.8000000000e-07|196608,49152|JKST|-,-,-,-,-,-,-,-,-,196608|MiniMax-M2.1 is a lightweight, state-of-the-art large language model optimized for coding, agentic w|Y
z-ai/glm-4.7|glm-4.7|Z.AI: GLM 4.7|C|4.0000000000e-07,1.5000000000e-06|202752,65535|JKST|-,-,-,-,-,-,-,-,-,202752|GLM-4.7 is Z.AI’s latest flagship model, featuring upgrades in two key areas: enhanced programming c|Y
google/gemini-3-flash-preview|gemini-3-flash-previ|Google: Gemini 3 Flash Preview|C|5.0000000000e-07,3.0000000000e-06|1048576,65535|JKSTV|-,-,-,-,-,-,-,-,-,1048576|Gemini 3 Flash Preview is a high speed, high value thinking model designed for agentic workflows, mu|Y
mistralai/mistral-small-creative|mistral-small-creati|Mistral: Mistral Small Creative|C|1.0000000000e-07,3.0000000000e-07|32768,8192|T|-,-,-,-,-,-,-,-,-,32768|Mistral Small Creative is an experimental small model designed for creative writing, narrative gener|Y
allenai/olmo-3.1-32b-think:free|olmo-3.1-32b-think:f|AllenAI: Olmo 3.1 32B Think (free)|C|0,0|65536,65536|JKS|-,-,-,-,-,-,-,-,-,65536|Olmo 3.1 32B Think is a large-scale, 32-billion-parameter model designed for deep reasoning, complex|Y
xiaomi/mimo-v2-flash:free|mimo-v2-flash:free|Xiaomi: MiMo-V2-Flash (free)|C|0,0|262144,65536|JKT|-,-,-,-,-,-,-,-,-,262144|MiMo-V2-Flash is an open-source foundation language model developed by Xiaomi. It is a Mixture-of-Ex|Y
nvidia/nemotron-3-nano-30b-a3b:free|nemotron-3-nano-30b-|NVIDIA: Nemotron 3 Nano 30B A3B (free)|C|0,0|256000,64000|KT|-,-,-,-,-,-,-,-,-,256000|NVIDIA Nemotron 3 Nano 30B A3B is a small language MoE model with highest compute efficiency and acc|Y
nvidia/nemotron-3-nano-30b-a3b|nemotron-3-nano-30b-|NVIDIA: Nemotron 3 Nano 30B A3B|C|6.0000000000e-08,2.4000000000e-07|262144,262144|JKST|-,-,-,-,-,-,-,-,-,262144|NVIDIA Nemotron 3 Nano 30B A3B is a small language MoE model with highest compute efficiency and acc|Y
openai/gpt-5.2-chat|gpt-5.2-chat|OpenAI: GPT-5.2 Chat|C|1.7500000000e-06,1.4e-05|128000,16384|JSTV|-,-,-,-,-,-,-,-,-,128000|GPT-5.2 Chat (AKA Instant) is the fast, lightweight member of the 5.2 family, optimized for low-late|Y
openai/gpt-5.2-pro|gpt-5.2-pro|OpenAI: GPT-5.2 Pro|C|2.1e-05,0.000168|400000,128000|JKSTV|-,-,-,-,-,-,-,-,-,400000|GPT-5.2 Pro is OpenAI’s most advanced model, offering major improvements in agentic coding and long|Y
openai/gpt-5.2|gpt-5.2|OpenAI: GPT-5.2|C|1.7500000000e-06,1.4e-05|400000,128000|JKSTV|-,-,-,-,-,-,-,-,-,400000|GPT-5.2 is the latest frontier-grade model in the GPT-5 series, offering stronger agentic and long c|Y
mistralai/devstral-2512:free|devstral-2512:free|Mistral: Devstral 2 2512 (free)|C|0,0|262144,65536|JST|-,-,-,-,-,-,-,-,-,262144|Devstral 2 is a state-of-the-art open-source model by Mistral AI specializing in agentic coding. It|Y
mistralai/devstral-2512|devstral-2512|Mistral: Devstral 2 2512|C|5.0000000000e-08,2.2000000000e-07|262144,65536|JST|-,-,-,-,-,-,-,-,-,262144|Devstral 2 is a state-of-the-art open-source model by Mistral AI specializing in agentic coding. It|Y
relace/relace-search|relace-search|Relace: Relace Search|C|1.0000000000e-06,3.0000000000e-06|256000,128000|T|-,-,-,-,-,-,-,-,-,256000|The relace-search model uses 4-12 `view_file` and `grep` tools in parallel to explore a codebase and|Y
z-ai/glm-4.6v|glm-4.6v|Z.AI: GLM 4.6V|C|3.0000000000e-07,9.0000000000e-07|131072,24000|JKSTV|-,-,-,-,-,-,-,-,-,131072|GLM-4.6V is a large multimodal model designed for high-fidelity visual understanding and long-contex|Y
nex-agi/deepseek-v3.1-nex-n1:free|deepseek-v3.1-nex-n1|Nex AGI: DeepSeek V3.1 Nex N1 (free)|C|0,0|131072,163840|JST|-,-,-,-,-,-,-,-,-,131072|DeepSeek V3.1 Nex-N1 is the flagship release of the Nex-N1 series — a post-trained model designed to|Y
essentialai/rnj-1-instruct|rnj-1-instruct|EssentialAI: Rnj 1 Instruct|C|1.5000000000e-07,1.5000000000e-07|32768,8192|JS|-,-,-,-,-,-,-,-,-,32768|Rnj-1 is an 8B-parameter, dense, open-weight model family developed by Essential AI and trained from|Y
openrouter/bodybuilder|bodybuilder|Body Builder (beta)|C|0,0|128000,32000||-,-,-,-,-,-,-,-,-,128000|Transform your natural language requests into structured OpenRouter API request objects. Describe wh|Y
openai/gpt-5.1-codex-max|gpt-5.1-codex-max|OpenAI: GPT-5.1-Codex-Max|C|1.2500000000e-06,1e-05|400000,128000|JKSTV|-,-,-,-,-,-,-,-,-,400000|GPT-5.1-Codex-Max is OpenAI’s latest agentic coding model, designed for long-running, high-context s|Y
amazon/nova-2-lite-v1|nova-2-lite-v1|Amazon: Nova 2 Lite|C|3.0000000000e-07,2.5000000000e-06|1000000,65535|KTV|-,-,-,-,-,-,-,-,-,1000000|Nova 2 Lite is a fast, cost-effective reasoning model for everyday workloads that can process text,|Y
mistralai/ministral-14b-2512|ministral-14b-2512|Mistral: Ministral 3 14B 2512|C|2.0000000000e-07,2.0000000000e-07|262144,65536|JSTV|-,-,-,-,-,-,-,-,-,262144|The largest model in the Ministral 3 family, Ministral 3 14B offers frontier capabilities and perfor|Y
mistralai/ministral-8b-2512|ministral-8b-2512|Mistral: Ministral 3 8B 2512|C|1.5000000000e-07,1.5000000000e-07|262144,65536|JSTV|-,-,-,-,-,-,-,-,-,262144|A balanced model in the Ministral 3 family, Ministral 3 8B is a powerful, efficient tiny language mo|Y
mistralai/ministral-3b-2512|ministral-3b-2512|Mistral: Ministral 3 3B 2512|C|1.0000000000e-07,1.0000000000e-07|131072,32768|JSTV|-,-,-,-,-,-,-,-,-,131072|The smallest model in the Ministral 3 family, Ministral 3 3B is a powerful, efficient tiny language|Y
mistralai/mistral-large-2512|mistral-large-2512|Mistral: Mistral Large 3 2512|C|5.0000000000e-07,1.5000000000e-06|262144,65536|JSTV|-,-,-,-,-,-,-,-,-,262144|Mistral Large 3 2512 is Mistral’s most capable model to date, featuring a sparse mixture-of-experts|Y
arcee-ai/trinity-mini:free|trinity-mini:free|Arcee AI: Trinity Mini (free)|C|0,0|131072,32768|JKST|-,-,-,-,-,-,-,-,-,131072|Trinity Mini is a 26B-parameter (3B active) sparse mixture-of-experts language model featuring 128 e|Y
arcee-ai/trinity-mini|trinity-mini|Arcee AI: Trinity Mini|C|4.0000000000e-08,1.5000000000e-07|131072,131072|JKST|-,-,-,-,-,-,-,-,-,131072|Trinity Mini is a 26B-parameter (3B active) sparse mixture-of-experts language model featuring 128 e|Y
deepseek/deepseek-v3.2-speciale|deepseek-v3.2-specia|DeepSeek: DeepSeek V3.2 Speciale|C|2.7000000000e-07,4.1000000000e-07|163840,65536|JKS|-,-,-,-,-,-,-,-,-,163840|DeepSeek-V3.2-Speciale is a high-compute variant of DeepSeek-V3.2 optimized for maximum reasoning an|Y
deepseek/deepseek-v3.2|deepseek-v3.2|DeepSeek: DeepSeek V3.2|C|2.5000000000e-07,3.8000000000e-07|163840,65536|JKST|-,-,-,-,-,-,-,-,-,163840|DeepSeek-V3.2 is a large language model designed to harmonize high computational efficiency with str|Y
prime-intellect/intellect-3|intellect-3|Prime Intellect: INTELLECT-3|C|2.0000000000e-07,1.1000000000e-06|131072,131072|JKST|-,-,-,-,-,-,-,-,-,131072|INTELLECT-3 is a 106B-parameter Mixture-of-Experts model (12B active) post-trained from GLM-4.5-Air-|Y
tngtech/tng-r1t-chimera:free|tng-r1t-chimera:free|TNG: R1T Chimera (free)|C|0,0|163840,163840|JKST|-,-,-,-,-,-,-,-,-,163840|TNG-R1T-Chimera is an experimental LLM with a faible for creative storytelling and character interac|Y
tngtech/tng-r1t-chimera|tng-r1t-chimera|TNG: R1T Chimera|C|2.5000000000e-07,8.5000000000e-07|163840,65536|JKST|-,-,-,-,-,-,-,-,-,163840|TNG-R1T-Chimera is an experimental LLM with a faible for creative storytelling and character interac|Y
anthropic/claude-opus-4.5|claude-opus-4.5|Anthropic: Claude Opus 4.5|C|5.0000000000e-06,2.5e-05|200000,32000|JKSTV|-,-,-,-,-,-,-,-,-,200000|Claude Opus 4.5 is Anthropic’s frontier reasoning model optimized for complex software engineering,|Y
allenai/olmo-3-32b-think:free|olmo-3-32b-think:fre|AllenAI: Olmo 3 32B Think (free)|C|0,0|65536,65536|JKS|-,-,-,-,-,-,-,-,-,65536|Olmo 3 32B Think is a large-scale, 32-billion-parameter model purpose-built for deep reasoning, comp|Y
allenai/olmo-3-7b-instruct|olmo-3-7b-instruct|AllenAI: Olmo 3 7B Instruct|C|1.0000000000e-07,2.0000000000e-07|65536,65536|JST|-,-,-,-,-,-,-,-,-,65536|Olmo 3 7B Instruct is a supervised instruction-fine-tuned variant of the Olmo 3 7B base model, optim|Y
allenai/olmo-3-7b-think|olmo-3-7b-think|AllenAI: Olmo 3 7B Think|C|1.2000000000e-07,2.0000000000e-07|65536,65536|JKS|-,-,-,-,-,-,-,-,-,65536|Olmo 3 7B Think is a research-oriented language model in the Olmo family designed for advanced reaso|Y
google/gemini-3-pro-image-preview|gemini-3-pro-image-p|Google: Nano Banana Pro (Gemini 3 Pro Image Preview)|C|2.0000000000e-06,1.2e-05|65536,32768|JKSV|-,-,-,-,-,-,-,-,-,65536|Nano Banana Pro is Google’s most advanced image-generation and editing model, built on Gemini 3 Pro.|Y
x-ai/grok-4.1-fast|grok-4.1-fast|xAI: Grok 4.1 Fast|C|2.0000000000e-07,5.0000000000e-07|2000000,30000|JKSTV|-,-,-,-,-,-,-,-,-,2000000|Grok 4.1 Fast is xAI's best agentic tool calling model that shines in real-world use cases like cust|Y
google/gemini-3-pro-preview|gemini-3-pro-preview|Google: Gemini 3 Pro Preview|C|2.0000000000e-06,1.2e-05|1048576,65536|JKSTV|-,-,-,-,-,-,-,-,-,1048576|Gemini 3 Pro is Google’s flagship frontier model for high-precision multimodal reasoning, combining|Y
deepcogito/cogito-v2.1-671b|cogito-v2.1-671b|Deep Cogito: Cogito v2.1 671B|C|1.2500000000e-06,1.2500000000e-06|128000,32000|JKS|-,-,-,-,-,-,-,-,-,128000|Cogito v2.1 671B MoE represents one of the strongest open models globally, matching performance of f|Y
openai/gpt-5.1|gpt-5.1|OpenAI: GPT-5.1|C|1.2500000000e-06,1e-05|400000,128000|JKSTV|-,-,-,-,-,-,-,-,-,400000|GPT-5.1 is the latest frontier-grade model in the GPT-5 series, offering stronger general-purpose re|Y
openai/gpt-5.1-chat|gpt-5.1-chat|OpenAI: GPT-5.1 Chat|C|1.2500000000e-06,1e-05|128000,16384|JSTV|-,-,-,-,-,-,-,-,-,128000|GPT-5.1 Chat (AKA Instant is the fast, lightweight member of the 5.1 family, optimized for low-laten|Y
openai/gpt-5.1-codex|gpt-5.1-codex|OpenAI: GPT-5.1-Codex|C|1.2500000000e-06,1e-05|400000,128000|JKSTV|-,-,-,-,-,-,-,-,-,400000|GPT-5.1-Codex is a specialized version of GPT-5.1 optimized for software engineering and coding work|Y
openai/gpt-5.1-codex-mini|gpt-5.1-codex-mini|OpenAI: GPT-5.1-Codex-Mini|C|2.5000000000e-07,2.0000000000e-06|400000,100000|JKSTV|-,-,-,-,-,-,-,-,-,400000|GPT-5.1-Codex-Mini is a smaller and faster version of GPT-5.1-Codex|Y
kwaipilot/kat-coder-pro:free|kat-coder-pro:free|Kwaipilot: KAT-Coder-Pro V1 (free)|C|0,0|256000,32768|JST|-,-,-,-,-,-,-,-,-,256000|KAT-Coder-Pro V1 is KwaiKAT's most advanced agentic coding model in the KAT-Coder series. Designed s|Y
moonshotai/kimi-k2-thinking|kimi-k2-thinking|MoonshotAI: Kimi K2 Thinking|C|4.0000000000e-07,1.7500000000e-06|262144,65535|JKST|-,-,-,-,-,-,-,-,-,262144|Kimi K2 Thinking is Moonshot AI’s most advanced open reasoning model to date, extending the K2 serie|Y
amazon/nova-premier-v1|nova-premier-v1|Amazon: Nova Premier 1.0|C|2.5000000000e-06,1.25e-05|1000000,32000|TV|-,-,-,-,-,-,-,-,-,1000000|Amazon Nova Premier is the most capable of Amazon’s multimodal models for complex reasoning tasks an|Y
perplexity/sonar-pro-search|sonar-pro-search|Perplexity: Sonar Pro Search|C|3.0000000000e-06,1.5e-05|200000,8000|JKSV|-,-,-,-,-,-,-,-,-,200000|Exclusively available on the OpenRouter API, Sonar Pro's new Pro Search mode is Perplexity's most ad|Y
mistralai/voxtral-small-24b-2507|voxtral-small-24b-25|Mistral: Voxtral Small 24B 2507|C|1.0000000000e-07,3.0000000000e-07|32000,8000|JST|-,-,-,-,-,-,-,-,-,32000|Voxtral Small is an enhancement of Mistral Small 3, incorporating state-of-the-art audio input capab|Y
openai/gpt-oss-safeguard-20b|gpt-oss-safeguard-20|OpenAI: gpt-oss-safeguard-20b|C|7.0000000000e-08,3.0000000000e-07|131072,65536|JKT|-,-,-,-,-,-,-,-,-,131072|gpt-oss-safeguard-20b is a safety reasoning model from OpenAI built upon gpt-oss-20b. This open-weig|Y
nvidia/nemotron-nano-12b-v2-vl:free|nemotron-nano-12b-v2|NVIDIA: Nemotron Nano 12B 2 VL (free)|C|0,0|128000,128000|KTV|-,-,-,-,-,-,-,-,-,128000|NVIDIA Nemotron Nano 2 VL is a 12-billion-parameter open multimodal reasoning model designed for vid|Y
nvidia/nemotron-nano-12b-v2-vl|nemotron-nano-12b-v2|NVIDIA: Nemotron Nano 12B 2 VL|C|2.0000000000e-07,6.0000000000e-07|131072,32768|JKV|-,-,-,-,-,-,-,-,-,131072|NVIDIA Nemotron Nano 2 VL is a 12-billion-parameter open multimodal reasoning model designed for vid|Y
minimax/minimax-m2|minimax-m2|MiniMax: MiniMax M2|C|2.0000000000e-07,1.0000000000e-06|196608,65536|JKST|-,-,-,-,-,-,-,-,-,196608|MiniMax-M2 is a compact, high-efficiency large language model optimized for end-to-end coding and ag|Y
qwen/qwen3-vl-32b-instruct|qwen3-vl-32b-instruc|Qwen: Qwen3 VL 32B Instruct|C|5.0000000000e-07,1.5000000000e-06|262144,65536|JSV|-,-,-,-,-,-,-,-,-,262144|Qwen3-VL-32B-Instruct is a large-scale multimodal vision-language model designed for high-precision|Y
liquid/lfm2-8b-a1b|lfm2-8b-a1b|LiquidAI/LFM2-8B-A1B|C|5.0000000000e-08,1.0000000000e-07|32768,8192||-,-,-,-,-,-,-,-,-,32768|Model created via inbox interface|Y
liquid/lfm-2.2-6b|lfm-2.2-6b|LiquidAI/LFM2-2.6B|C|5.0000000000e-08,1.0000000000e-07|32768,8192||-,-,-,-,-,-,-,-,-,32768|LFM2 is a new generation of hybrid models developed by Liquid AI, specifically designed for edge AI|Y
ibm-granite/granite-4.0-h-micro|granite-4.0-h-micro|IBM: Granite 4.0 Micro|C|2.0000000000e-08,1.1000000000e-07|131000,32750||-,-,-,-,-,-,-,-,-,131000|Granite-4.0-H-Micro is a 3B parameter from the Granite 4 family of models. These models are the late|Y
deepcogito/cogito-v2-preview-llama-405b|cogito-v2-preview-ll|Deep Cogito: Cogito V2 Preview Llama 405B|C|3.5000000000e-06,3.5000000000e-06|32768,8192|JKST|-,-,-,-,-,-,-,-,-,32768|Cogito v2 405B is a dense hybrid reasoning model that combines direct answering capabilities with ad|Y
openai/gpt-5-image-mini|gpt-5-image-mini|OpenAI: GPT-5 Image Mini|C|2.5000000000e-06,2.0000000000e-06|400000,128000|JKSTV|-,-,-,-,-,-,-,-,-,400000|GPT-5 Image Mini combines OpenAI's advanced language capabilities, powered by [GPT-5 Mini](https://o|Y
anthropic/claude-haiku-4.5|claude-haiku-4.5|Anthropic: Claude Haiku 4.5|C|1.0000000000e-06,5.0000000000e-06|200000,64000|KTV|-,-,-,-,-,-,-,-,-,200000|Claude Haiku 4.5 is Anthropic’s fastest and most efficient model, delivering near-frontier intellige|Y
qwen/qwen3-vl-8b-thinking|qwen3-vl-8b-thinking|Qwen: Qwen3 VL 8B Thinking|C|1.8000000000e-07,2.1000000000e-06|256000,32768|JKSTV|-,-,-,-,-,-,-,-,-,256000|Qwen3-VL-8B-Thinking is the reasoning-optimized variant of the Qwen3-VL-8B multimodal model, designe|Y
qwen/qwen3-vl-8b-instruct|qwen3-vl-8b-instruct|Qwen: Qwen3 VL 8B Instruct|C|8.0000000000e-08,5.0000000000e-07|131072,32768|JSTV|-,-,-,-,-,-,-,-,-,131072|Qwen3-VL-8B-Instruct is a multimodal vision-language model from the Qwen3-VL series, built for high-|Y
openai/gpt-5-image|gpt-5-image|OpenAI: GPT-5 Image|C|1e-05,1e-05|400000,128000|JKSTV|-,-,-,-,-,-,-,-,-,400000|[GPT-5](https://openrouter.ai/openai/gpt-5) Image combines OpenAI's GPT-5 model with state-of-the-ar|Y
openai/o3-deep-research|o3-deep-research|OpenAI: o3 Deep Research|C|1e-05,4e-05|200000,100000|JKSTV|-,-,-,-,-,-,-,-,-,200000|o3-deep-research is OpenAI's advanced model for deep research, designed to tackle complex, multi-ste|Y
openai/o4-mini-deep-research|o4-mini-deep-researc|OpenAI: o4 Mini Deep Research|C|2.0000000000e-06,8.0000000000e-06|200000,100000|JKSTV|-,-,-,-,-,-,-,-,-,200000|o4-mini-deep-research is OpenAI's faster, more affordable deep research model—ideal for tackling com|Y
nvidia/llama-3.3-nemotron-super-49b-v1.5|llama-3.3-nemotron-s|NVIDIA: Llama 3.3 Nemotron Super 49B V1.5|C|1.0000000000e-07,4.0000000000e-07|131072,32768|JKT|-,-,-,-,-,-,-,-,-,131072|Llama-3.3-Nemotron-Super-49B-v1.5 is a 49B-parameter, English-centric reasoning/chat model derived f|Y
baidu/ernie-4.5-21b-a3b-thinking|ernie-4.5-21b-a3b-th|Baidu: ERNIE 4.5 21B A3B Thinking|C|7.0000000000e-08,2.8000000000e-07|131072,65536|K|-,-,-,-,-,-,-,-,-,131072|ERNIE-4.5-21B-A3B-Thinking is Baidu's upgraded lightweight MoE model, refined to boost reasoning dep|Y
google/gemini-2.5-flash-image|gemini-2.5-flash-ima|Google: Gemini 2.5 Flash Image (Nano Banana)|C|3.0000000000e-07,2.5000000000e-06|32768,32768|JSV|-,-,-,-,-,-,-,-,-,32768|Gemini 2.5 Flash Image, a.k.a. \"Nano Banana,\" is now generally available. It is a state of the art i|Y
qwen/qwen3-vl-30b-a3b-thinking|qwen3-vl-30b-a3b-thi|Qwen: Qwen3 VL 30B A3B Thinking|C|2.0000000000e-07,1.0000000000e-06|131072,32768|JKSTV|-,-,-,-,-,-,-,-,-,131072|Qwen3-VL-30B-A3B-Thinking is a multimodal model that unifies strong text generation with visual unde|Y
qwen/qwen3-vl-30b-a3b-instruct|qwen3-vl-30b-a3b-ins|Qwen: Qwen3 VL 30B A3B Instruct|C|1.5000000000e-07,6.0000000000e-07|262144,65536|JSTV|-,-,-,-,-,-,-,-,-,262144|Qwen3-VL-30B-A3B-Instruct is a multimodal model that unifies strong text generation with visual unde|Y
openai/gpt-5-pro|gpt-5-pro|OpenAI: GPT-5 Pro|C|1.5e-05,0.00012|400000,128000|JKSTV|-,-,-,-,-,-,-,-,-,400000|GPT-5 Pro is OpenAI’s most advanced model, offering major improvements in reasoning, code quality, a|Y
z-ai/glm-4.6|glm-4.6|Z.AI: GLM 4.6|C|3.5000000000e-07,1.5000000000e-06|202752,65536|JKST|-,-,-,-,-,-,-,-,-,202752|Compared with GLM-4.5, this generation brings several key improvements:

Longer context window: The|Y
z-ai/glm-4.6:exacto|glm-4.6:exacto|Z.AI: GLM 4.6 (exacto)|C|4.4000000000e-07,1.7600000000e-06|204800,131072|JKST|-,-,-,-,-,-,-,-,-,204800|Compared with GLM-4.5, this generation brings several key improvements:

Longer context window: The|Y
anthropic/claude-sonnet-4.5|claude-sonnet-4.5|Anthropic: Claude Sonnet 4.5|C|3.0000000000e-06,1.5e-05|1000000,64000|JKSTV|-,-,-,-,-,-,-,-,-,1000000|Claude Sonnet 4.5 is Anthropic’s most advanced Sonnet model to date, optimized for real-world agents|Y
deepseek/deepseek-v3.2-exp|deepseek-v3.2-exp|DeepSeek: DeepSeek V3.2 Exp|C|2.1000000000e-07,3.2000000000e-07|163840,65536|JKST|-,-,-,-,-,-,-,-,-,163840|DeepSeek-V3.2-Exp is an experimental large language model released by DeepSeek as an intermediate st|Y
thedrummer/cydonia-24b-v4.1|cydonia-24b-v4.1|TheDrummer: Cydonia 24B V4.1|C|3.0000000000e-07,5.0000000000e-07|131072,131072|JS|-,-,-,-,-,-,-,-,-,131072|Uncensored and creative writing model based on Mistral Small 3.2 24B with good recall, prompt adhere|Y
relace/relace-apply-3|relace-apply-3|Relace: Relace Apply 3|C|8.5000000000e-07,1.2500000000e-06|256000,128000||-,-,-,-,-,-,-,-,-,256000|Relace Apply 3 is a specialized code-patching LLM that merges AI-suggested edits straight into your|Y
google/gemini-2.5-flash-preview-09-2025|gemini-2.5-flash-pre|Google: Gemini 2.5 Flash Preview 09-2025|C|3.0000000000e-07,2.5000000000e-06|1048576,65536|JKSTV|-,-,-,-,-,-,-,-,-,1048576|Gemini 2.5 Flash Preview September 2025 Checkpoint is Google's state-of-the-art workhorse model, spe|Y
google/gemini-2.5-flash-lite-preview-09-2025|gemini-2.5-flash-lit|Google: Gemini 2.5 Flash Lite Preview 09-2025|C|1.0000000000e-07,4.0000000000e-07|1048576,65536|JKSTV|-,-,-,-,-,-,-,-,-,1048576|Gemini 2.5 Flash-Lite is a lightweight reasoning model in the Gemini 2.5 family, optimized for ultra|Y
qwen/qwen3-vl-235b-a22b-thinking|qwen3-vl-235b-a22b-t|Qwen: Qwen3 VL 235B A22B Thinking|C|3.0000000000e-07,1.2000000000e-06|262144,262144|JKSTV|-,-,-,-,-,-,-,-,-,262144|Qwen3-VL-235B-A22B Thinking is a multimodal model that unifies strong text generation with visual un|Y
qwen/qwen3-vl-235b-a22b-instruct|qwen3-vl-235b-a22b-i|Qwen: Qwen3 VL 235B A22B Instruct|C|2.0000000000e-07,1.2000000000e-06|262144,65536|JSTV|-,-,-,-,-,-,-,-,-,262144|Qwen3-VL-235B-A22B Instruct is an open-weight multimodal model that unifies strong text generation w|Y
qwen/qwen3-max|qwen3-max|Qwen: Qwen3 Max|C|1.2000000000e-06,6.0000000000e-06|256000,32768|JT|-,-,-,-,-,-,-,-,-,256000|Qwen3-Max is an updated release built on the Qwen3 series, offering major improvements in reasoning,|Y
qwen/qwen3-coder-plus|qwen3-coder-plus|Qwen: Qwen3 Coder Plus|C|1.0000000000e-06,5.0000000000e-06|128000,65536|JST|-,-,-,-,-,-,-,-,-,128000|Qwen3 Coder Plus is Alibaba's proprietary version of the Open Source Qwen3 Coder 480B A35B. It is a|Y
openai/gpt-5-codex|gpt-5-codex|OpenAI: GPT-5 Codex|C|1.2500000000e-06,1e-05|400000,128000|JKSTV|-,-,-,-,-,-,-,-,-,400000|GPT-5-Codex is a specialized version of GPT-5 optimized for software engineering and coding workflow|Y
deepseek/deepseek-v3.1-terminus:exacto|deepseek-v3.1-termin|DeepSeek: DeepSeek V3.1 Terminus (exacto)|C|2.1000000000e-07,7.9000000000e-07|163840,40960|JKST|-,-,-,-,-,-,-,-,-,163840|DeepSeek-V3.1 Terminus is an update to [DeepSeek V3.1](/deepseek/deepseek-chat-v3.1) that maintains|Y
deepseek/deepseek-v3.1-terminus|deepseek-v3.1-termin|DeepSeek: DeepSeek V3.1 Terminus|C|2.1000000000e-07,7.9000000000e-07|163840,40960|JKST|-,-,-,-,-,-,-,-,-,163840|DeepSeek-V3.1 Terminus is an update to [DeepSeek V3.1](/deepseek/deepseek-chat-v3.1) that maintains|Y
x-ai/grok-4-fast|grok-4-fast|xAI: Grok 4 Fast|C|2.0000000000e-07,5.0000000000e-07|2000000,30000|JKSTV|-,-,-,-,-,-,-,-,-,2000000|Grok 4 Fast is xAI's latest multimodal model with SOTA cost-efficiency and a 2M token context window|Y
alibaba/tongyi-deepresearch-30b-a3b:free|tongyi-deepresearch-|Tongyi DeepResearch 30B A3B (free)|C|0,0|131072,131072|JKST|-,-,-,-,-,-,-,-,-,131072|Tongyi DeepResearch is an agentic large language model developed by Tongyi Lab, with 30 billion tota|Y
alibaba/tongyi-deepresearch-30b-a3b|tongyi-deepresearch-|Tongyi DeepResearch 30B A3B|C|9.0000000000e-08,4.0000000000e-07|131072,131072|JKST|-,-,-,-,-,-,-,-,-,131072|Tongyi DeepResearch is an agentic large language model developed by Tongyi Lab, with 30 billion tota|Y
qwen/qwen3-coder-flash|qwen3-coder-flash|Qwen: Qwen3 Coder Flash|C|3.0000000000e-07,1.5000000000e-06|128000,65536|JT|-,-,-,-,-,-,-,-,-,128000|Qwen3 Coder Flash is Alibaba's fast and cost efficient version of their proprietary Qwen3 Coder Plus|Y
opengvlab/internvl3-78b|internvl3-78b|OpenGVLab: InternVL3 78B|C|1.0000000000e-07,3.9000000000e-07|32768,32768|JSV|-,-,-,-,-,-,-,-,-,32768|The InternVL3 series is an advanced multimodal large language model (MLLM). Compared to InternVL 2.5|Y
qwen/qwen3-next-80b-a3b-thinking|qwen3-next-80b-a3b-t|Qwen: Qwen3 Next 80B A3B Thinking|C|1.5000000000e-07,1.2000000000e-06|262144,262144|JKST|-,-,-,-,-,-,-,-,-,262144|Qwen3-Next-80B-A3B-Thinking is a reasoning-first chat model in the Qwen3-Next line that outputs stru|Y
qwen/qwen3-next-80b-a3b-instruct|qwen3-next-80b-a3b-i|Qwen: Qwen3 Next 80B A3B Instruct|C|6.0000000000e-08,6.0000000000e-07|262144,65536|JST|-,-,-,-,-,-,-,-,-,262144|Qwen3-Next-80B-A3B-Instruct is an instruction-tuned chat model in the Qwen3-Next series optimized fo|Y
meituan/longcat-flash-chat|longcat-flash-chat|Meituan: LongCat Flash Chat|C|2.0000000000e-07,8.0000000000e-07|131072,131072||-,-,-,-,-,-,-,-,-,131072|LongCat-Flash-Chat is a large-scale Mixture-of-Experts (MoE) model with 560B total parameters, of wh|Y
qwen/qwen-plus-2025-07-28|qwen-plus-2025-07-28|Qwen: Qwen Plus 0728|C|4.0000000000e-07,1.2000000000e-06|1000000,32768|JST|-,-,-,-,-,-,-,-,-,1000000|Qwen Plus 0728, based on the Qwen3 foundation model, is a 1 million context hybrid reasoning model w|Y
qwen/qwen-plus-2025-07-28:thinking|qwen-plus-2025-07-28|Qwen: Qwen Plus 0728 (thinking)|C|4.0000000000e-07,4.0000000000e-06|1000000,32768|JKST|-,-,-,-,-,-,-,-,-,1000000|Qwen Plus 0728, based on the Qwen3 foundation model, is a 1 million context hybrid reasoning model w|Y
nvidia/nemotron-nano-9b-v2:free|nemotron-nano-9b-v2:|NVIDIA: Nemotron Nano 9B V2 (free)|C|0,0|128000,32000|JKST|-,-,-,-,-,-,-,-,-,128000|NVIDIA-Nemotron-Nano-9B-v2 is a large language model (LLM) trained from scratch by NVIDIA, and desig|Y
nvidia/nemotron-nano-9b-v2|nemotron-nano-9b-v2|NVIDIA: Nemotron Nano 9B V2|C|4.0000000000e-08,1.6000000000e-07|131072,32768|JKT|-,-,-,-,-,-,-,-,-,131072|NVIDIA-Nemotron-Nano-9B-v2 is a large language model (LLM) trained from scratch by NVIDIA, and desig|Y
moonshotai/kimi-k2-0905|kimi-k2-0905|MoonshotAI: Kimi K2 0905|C|3.9000000000e-07,1.9000000000e-06|262144,262144|JST|-,-,-,-,-,-,-,-,-,262144|Kimi K2 0905 is the September update of [Kimi K2 0711](moonshotai/kimi-k2). It is a large-scale Mixt|Y
moonshotai/kimi-k2-0905:exacto|kimi-k2-0905:exacto|MoonshotAI: Kimi K2 0905 (exacto)|C|6.0000000000e-07,2.5000000000e-06|262144,65536|JST|-,-,-,-,-,-,-,-,-,262144|Kimi K2 0905 is the September update of [Kimi K2 0711](moonshotai/kimi-k2). It is a large-scale Mixt|Y
deepcogito/cogito-v2-preview-llama-70b|cogito-v2-preview-ll|Deep Cogito: Cogito V2 Preview Llama 70B|C|8.8000000000e-07,8.8000000000e-07|32768,8192|JKST|-,-,-,-,-,-,-,-,-,32768|Cogito v2 70B is a dense hybrid reasoning model that combines direct answering capabilities with adv|Y
deepcogito/cogito-v2-preview-llama-109b-moe|cogito-v2-preview-ll|Cogito V2 Preview Llama 109B|C|1.8000000000e-07,5.9000000000e-07|32767,8191|KTV|-,-,-,-,-,-,-,-,-,32767|An instruction-tuned, hybrid-reasoning Mixture-of-Experts model built on Llama-4-Scout-17B-16E. Cogi|Y
stepfun-ai/step3|step3|StepFun: Step3|C|5.7000000000e-07,1.4200000000e-06|65536,65536|JKSTV|-,-,-,-,-,-,-,-,-,65536|Step3 is a cutting-edge multimodal reasoning model—built on a Mixture-of-Experts architecture with 3|Y
qwen/qwen3-30b-a3b-thinking-2507|qwen3-30b-a3b-thinki|Qwen: Qwen3 30B A3B Thinking 2507|C|5.0000000000e-08,3.4000000000e-07|32768,8192|JKST|-,-,-,-,-,-,-,-,-,32768|Qwen3-30B-A3B-Thinking-2507 is a 30B parameter Mixture-of-Experts reasoning model optimized for comp|Y
x-ai/grok-code-fast-1|grok-code-fast-1|xAI: Grok Code Fast 1|C|2.0000000000e-07,1.5000000000e-06|256000,10000|JKST|-,-,-,-,-,-,-,-,-,256000|Grok Code Fast 1 is a speedy and economical reasoning model that excels at agentic coding. With reas|Y
nousresearch/hermes-4-70b|hermes-4-70b|Nous: Hermes 4 70B|C|1.1000000000e-07,3.8000000000e-07|131072,131072|JKST|-,-,-,-,-,-,-,-,-,131072|Hermes 4 70B is a hybrid reasoning model from Nous Research, built on Meta-Llama-3.1-70B. It introdu|Y
nousresearch/hermes-4-405b|hermes-4-405b|Nous: Hermes 4 405B|C|3.0000000000e-07,1.2000000000e-06|131072,131072|JKST|-,-,-,-,-,-,-,-,-,131072|Hermes 4 is a large-scale reasoning model built on Meta-Llama-3.1-405B and released by Nous Research|Y
google/gemini-2.5-flash-image-preview|gemini-2.5-flash-ima|Google: Gemini 2.5 Flash Image Preview (Nano Banana)|C|3.0000000000e-07,2.5000000000e-06|32768,32768|JSV|-,-,-,-,-,-,-,-,-,32768|Gemini 2.5 Flash Image Preview, a.k.a. \"Nano Banana,\" is a state of the art image generation model w|Y
deepseek/deepseek-chat-v3.1|deepseek-chat-v3.1|DeepSeek: DeepSeek V3.1|C|1.5000000000e-07,7.5000000000e-07|32768,7168|JKST|-,-,-,-,-,-,-,-,-,32768|DeepSeek-V3.1 is a large hybrid reasoning model (671B parameters, 37B active) that supports both thi|Y
openai/gpt-4o-audio-preview|gpt-4o-audio-preview|OpenAI: GPT-4o Audio|C|2.5000000000e-06,1e-05|128000,16384|JST|-,-,-,-,-,-,-,-,-,128000|The gpt-4o-audio-preview model adds support for audio inputs as prompts. This enhancement allows the|Y
mistralai/mistral-medium-3.1|mistral-medium-3.1|Mistral: Mistral Medium 3.1|C|4.0000000000e-07,2.0000000000e-06|131072,32768|JSTV|-,-,-,-,-,-,-,-,-,131072|Mistral Medium 3.1 is an updated version of Mistral Medium 3, which is a high-performance enterprise|Y
baidu/ernie-4.5-21b-a3b|ernie-4.5-21b-a3b|Baidu: ERNIE 4.5 21B A3B|C|7.0000000000e-08,2.8000000000e-07|120000,8000|T|-,-,-,-,-,-,-,-,-,120000|A sophisticated text-based Mixture-of-Experts (MoE) model featuring 21B total parameters with 3B act|Y
baidu/ernie-4.5-vl-28b-a3b|ernie-4.5-vl-28b-a3b|Baidu: ERNIE 4.5 VL 28B A3B|C|1.4000000000e-07,5.6000000000e-07|30000,8000|KTV|-,-,-,-,-,-,-,-,-,30000|A powerful multimodal Mixture-of-Experts chat model featuring 28B total parameters with 3B activated|Y
z-ai/glm-4.5v|glm-4.5v|Z.AI: GLM 4.5V|C|6.0000000000e-07,1.8000000000e-06|65536,16384|JKSTV|-,-,-,-,-,-,-,-,-,65536|GLM-4.5V is a vision-language foundation model for multimodal agent applications. Built on a Mixture|Y
ai21/jamba-mini-1.7|jamba-mini-1.7|AI21: Jamba Mini 1.7|C|2.0000000000e-07,4.0000000000e-07|256000,4096|JT|-,-,-,-,-,-,-,-,-,256000|Jamba Mini 1.7 is a compact and efficient member of the Jamba open model family, incorporating key i|Y
ai21/jamba-large-1.7|jamba-large-1.7|AI21: Jamba Large 1.7|C|2.0000000000e-06,8.0000000000e-06|256000,4096|JT|-,-,-,-,-,-,-,-,-,256000|Jamba Large 1.7 is the latest model in the Jamba open family, offering improvements in grounding, in|Y
openai/gpt-5-chat|gpt-5-chat|OpenAI: GPT-5 Chat|C|1.2500000000e-06,1e-05|128000,16384|JSV|-,-,-,-,-,-,-,-,-,128000|GPT-5 Chat is designed for advanced, natural, multimodal, and context-aware conversations for enterp|Y
openai/gpt-5|gpt-5|OpenAI: GPT-5|C|1.2500000000e-06,1e-05|400000,128000|JKSTV|-,-,-,-,-,-,-,-,-,400000|GPT-5 is OpenAI’s most advanced model, offering major improvements in reasoning, code quality, and u|Y
openai/gpt-5-mini|gpt-5-mini|OpenAI: GPT-5 Mini|C|2.5000000000e-07,2.0000000000e-06|400000,128000|JKSTV|-,-,-,-,-,-,-,-,-,400000|GPT-5 Mini is a compact version of GPT-5, designed to handle lighter-weight reasoning tasks. It prov|Y
openai/gpt-5-nano|gpt-5-nano|OpenAI: GPT-5 Nano|C|5.0000000000e-08,4.0000000000e-07|400000,128000|JKSTV|-,-,-,-,-,-,-,-,-,400000|GPT-5-Nano is the smallest and fastest variant in the GPT-5 system, optimized for developer tools, r|Y
openai/gpt-oss-120b:free|gpt-oss-120b:free|OpenAI: gpt-oss-120b (free)|C|0,0|131072,32768|KT|-,-,-,-,-,-,-,-,-,131072|gpt-oss-120b is an open-weight, 117B-parameter Mixture-of-Experts (MoE) language model from OpenAI d|Y
openai/gpt-oss-120b|gpt-oss-120b|OpenAI: gpt-oss-120b|C|2.0000000000e-08,1.0000000000e-07|131072,32768|JKST|-,-,-,-,-,-,-,-,-,131072|gpt-oss-120b is an open-weight, 117B-parameter Mixture-of-Experts (MoE) language model from OpenAI d|Y
openai/gpt-oss-120b:exacto|gpt-oss-120b:exacto|OpenAI: gpt-oss-120b (exacto)|C|4.0000000000e-08,1.9000000000e-07|131072,32768|JKST|-,-,-,-,-,-,-,-,-,131072|gpt-oss-120b is an open-weight, 117B-parameter Mixture-of-Experts (MoE) language model from OpenAI d|Y
openai/gpt-oss-20b:free|gpt-oss-20b:free|OpenAI: gpt-oss-20b (free)|C|0,0|131072,131072|JKST|-,-,-,-,-,-,-,-,-,131072|gpt-oss-20b is an open-weight 21B parameter model released by OpenAI under the Apache 2.0 license. I|Y
openai/gpt-oss-20b|gpt-oss-20b|OpenAI: gpt-oss-20b|C|2.0000000000e-08,6.0000000000e-08|131072,32768|JKST|-,-,-,-,-,-,-,-,-,131072|gpt-oss-20b is an open-weight 21B parameter model released by OpenAI under the Apache 2.0 license. I|Y
anthropic/claude-opus-4.1|claude-opus-4.1|Anthropic: Claude Opus 4.1|C|1.5e-05,7.5e-05|200000,50000|JKSTV|-,-,-,-,-,-,-,-,-,200000|Claude Opus 4.1 is an updated version of Anthropic’s flagship model, offering improved performance i|Y
mistralai/codestral-2508|codestral-2508|Mistral: Codestral 2508|C|3.0000000000e-07,9.0000000000e-07|256000,64000|JST|-,-,-,-,-,-,-,-,-,256000|Mistral's cutting-edge language model for coding released end of July 2025. Codestral specializes in|Y
qwen/qwen3-coder-30b-a3b-instruct|qwen3-coder-30b-a3b-|Qwen: Qwen3 Coder 30B A3B Instruct|C|7.0000000000e-08,2.7000000000e-07|160000,32768|JST|-,-,-,-,-,-,-,-,-,160000|Qwen3-Coder-30B-A3B-Instruct is a 30.5B parameter Mixture-of-Experts (MoE) model with 128 experts (8|Y
qwen/qwen3-30b-a3b-instruct-2507|qwen3-30b-a3b-instru|Qwen: Qwen3 30B A3B Instruct 2507|C|8.0000000000e-08,3.3000000000e-07|262144,262144|JST|-,-,-,-,-,-,-,-,-,262144|Qwen3-30B-A3B-Instruct-2507 is a 30.5B-parameter mixture-of-experts language model from Qwen, with 3|Y
z-ai/glm-4.5|glm-4.5|Z.AI: GLM 4.5|C|3.5000000000e-07,1.5500000000e-06|131072,65536|JKST|-,-,-,-,-,-,-,-,-,131072|GLM-4.5 is our latest flagship foundation model, purpose-built for agent-based applications. It leve|Y
z-ai/glm-4.5-air:free|glm-4.5-air:free|Z.AI: GLM 4.5 Air (free)|C|0,0|131072,131072|JKST|-,-,-,-,-,-,-,-,-,131072|GLM-4.5-Air is the lightweight variant of our latest flagship model family, also purpose-built for a|Y
z-ai/glm-4.5-air|glm-4.5-air|Z.AI: GLM 4.5 Air|C|1.3000000000e-07,8.5000000000e-07|131072,98304|JKST|-,-,-,-,-,-,-,-,-,131072|GLM-4.5-Air is the lightweight variant of our latest flagship model family, also purpose-built for a|Y
qwen/qwen3-235b-a22b-thinking-2507|qwen3-235b-a22b-thin|Qwen: Qwen3 235B A22B Thinking 2507|C|1.1000000000e-07,6.0000000000e-07|262144,262144|JKST|-,-,-,-,-,-,-,-,-,262144|Qwen3-235B-A22B-Thinking-2507 is a high-performance, open-weight Mixture-of-Experts (MoE) language m|Y
z-ai/glm-4-32b|glm-4-32b|Z.AI: GLM 4 32B|C|1.0000000000e-07,1.0000000000e-07|128000,32000|T|-,-,-,-,-,-,-,-,-,128000|GLM 4 32B is a cost-effective foundation language model.

It can efficiently perform complex tasks a|Y
qwen/qwen3-coder:free|qwen3-coder:free|Qwen: Qwen3 Coder 480B A35B (free)|C|0,0|262000,262000|T|-,-,-,-,-,-,-,-,-,262000|Qwen3-Coder-480B-A35B-Instruct is a Mixture-of-Experts (MoE) code generation model developed by the|Y
qwen/qwen3-coder|qwen3-coder|Qwen: Qwen3 Coder 480B A35B|C|2.2000000000e-07,9.5000000000e-07|262144,262144|JKST|-,-,-,-,-,-,-,-,-,262144|Qwen3-Coder-480B-A35B-Instruct is a Mixture-of-Experts (MoE) code generation model developed by the|Y
qwen/qwen3-coder:exacto|qwen3-coder:exacto|Qwen: Qwen3 Coder 480B A35B (exacto)|C|2.2000000000e-07,1.8000000000e-06|262144,65536|JKST|-,-,-,-,-,-,-,-,-,262144|Qwen3-Coder-480B-A35B-Instruct is a Mixture-of-Experts (MoE) code generation model developed by the|Y
bytedance/ui-tars-1.5-7b|ui-tars-1.5-7b|ByteDance: UI-TARS 7B|C|1.0000000000e-07,2.0000000000e-07|128000,2048|V|-,-,-,-,-,-,-,-,-,128000|UI-TARS-1.5 is a multimodal vision-language agent optimized for GUI-based environments, including de|Y
google/gemini-2.5-flash-lite|gemini-2.5-flash-lit|Google: Gemini 2.5 Flash Lite|C|1.0000000000e-07,4.0000000000e-07|1048576,65535|JKSTV|-,-,-,-,-,-,-,-,-,1048576|Gemini 2.5 Flash-Lite is a lightweight reasoning model in the Gemini 2.5 family, optimized for ultra|Y
qwen/qwen3-235b-a22b-2507|qwen3-235b-a22b-2507|Qwen: Qwen3 235B A22B Instruct 2507|C|7.0000000000e-08,4.6000000000e-07|262144,65536|JKST|-,-,-,-,-,-,-,-,-,262144|Qwen3-235B-A22B-Instruct-2507 is a multilingual, instruction-tuned mixture-of-experts language model|Y
switchpoint/router|router|Switchpoint Router|C|8.5000000000e-07,3.4000000000e-06|131072,32768|K|-,-,-,-,-,-,-,-,-,131072|Switchpoint AI's router instantly analyzes your request and directs it to the optimal AI from an eve|Y
moonshotai/kimi-k2:free|kimi-k2:free|MoonshotAI: Kimi K2 0711 (free)|C|0,0|32768,8192||-,-,-,-,-,-,-,-,-,32768|Kimi K2 Instruct is a large-scale Mixture-of-Experts (MoE) language model developed by Moonshot AI,|Y
moonshotai/kimi-k2|kimi-k2|MoonshotAI: Kimi K2 0711|C|4.6000000000e-07,1.8400000000e-06|131072,131072|JST|-,-,-,-,-,-,-,-,-,131072|Kimi K2 Instruct is a large-scale Mixture-of-Experts (MoE) language model developed by Moonshot AI,|Y
thudm/glm-4.1v-9b-thinking|glm-4.1v-9b-thinking|THUDM: GLM 4.1V 9B Thinking|C|4.0000000000e-08,1.4000000000e-07|65536,8000|KV|-,-,-,-,-,-,-,-,-,65536|GLM-4.1V-9B-Thinking is a 9B parameter vision-language model developed by THUDM, based on the GLM-4-|Y
mistralai/devstral-medium|devstral-medium|Mistral: Devstral Medium|C|4.0000000000e-07,2.0000000000e-06|131072,32768|JST|-,-,-,-,-,-,-,-,-,131072|Devstral Medium is a high-performance code generation and agentic reasoning model developed jointly|Y
mistralai/devstral-small|devstral-small|Mistral: Devstral Small 1.1|C|7.0000000000e-08,2.8000000000e-07|128000,32000|JST|-,-,-,-,-,-,-,-,-,128000|Devstral Small 1.1 is a 24B parameter open-weight language model for software engineering agents, de|Y
cognitivecomputations/dolphin-mistral-24b-venice-edition:free|dolphin-mistral-24b-|Venice: Uncensored (free)|C|0,0|32768,8192|JS|-,-,-,-,-,-,-,-,-,32768|Venice Uncensored Dolphin Mistral 24B Venice Edition is a fine-tuned variant of Mistral-Small-24B-In|Y
x-ai/grok-4|grok-4|xAI: Grok 4|C|3.0000000000e-06,1.5e-05|256000,64000|JKSTV|-,-,-,-,-,-,-,-,-,256000|Grok 4 is xAI's latest reasoning model with a 256k context window. It supports parallel tool calling|Y
google/gemma-3n-e2b-it:free|gemma-3n-e2b-it:free|Google: Gemma 3n 2B (free)|C|0,0|8192,2048|J|-,-,-,-,-,-,-,-,-,8192|Gemma 3n E2B IT is a multimodal, instruction-tuned model developed by Google DeepMind, designed to o|Y
tencent/hunyuan-a13b-instruct|hunyuan-a13b-instruc|Tencent: Hunyuan A13B Instruct|C|1.4000000000e-07,5.7000000000e-07|131072,131072|JKS|-,-,-,-,-,-,-,-,-,131072|Hunyuan-A13B is a 13B active parameter Mixture-of-Experts (MoE) language model developed by Tencent,|Y
tngtech/deepseek-r1t2-chimera:free|deepseek-r1t2-chimer|TNG: DeepSeek R1T2 Chimera (free)|C|0,0|163840,40960|K|-,-,-,-,-,-,-,-,-,163840|DeepSeek-TNG-R1T2-Chimera is the second-generation Chimera model from TNG Tech. It is a 671 B-parame|Y
tngtech/deepseek-r1t2-chimera|deepseek-r1t2-chimer|TNG: DeepSeek R1T2 Chimera|C|2.5000000000e-07,8.5000000000e-07|163840,163840|JKST|-,-,-,-,-,-,-,-,-,163840|DeepSeek-TNG-R1T2-Chimera is the second-generation Chimera model from TNG Tech. It is a 671 B-parame|Y
morph/morph-v3-large|morph-v3-large|Morph: Morph V3 Large|C|9.0000000000e-07,1.9000000000e-06|262144,131072||-,-,-,-,-,-,-,-,-,262144|Morph's high-accuracy apply model for complex code edits. ~4,500 tokens/sec with 98% accuracy for pr|Y
morph/morph-v3-fast|morph-v3-fast|Morph: Morph V3 Fast|C|8.0000000000e-07,1.2000000000e-06|81920,38000||-,-,-,-,-,-,-,-,-,81920|Morph's fastest apply model for code edits. ~10,500 tokens/sec with 96% accuracy for rapid code tran|Y
baidu/ernie-4.5-vl-424b-a47b|ernie-4.5-vl-424b-a4|Baidu: ERNIE 4.5 VL 424B A47B|C|4.2000000000e-07,1.2500000000e-06|123000,16000|KV|-,-,-,-,-,-,-,-,-,123000|ERNIE-4.5-VL-424B-A47B is a multimodal Mixture-of-Experts (MoE) model from Baidu’s ERNIE 4.5 series,|Y
baidu/ernie-4.5-300b-a47b|ernie-4.5-300b-a47b|Baidu: ERNIE 4.5 300B A47B|C|2.8000000000e-07,1.1000000000e-06|123000,12000|JS|-,-,-,-,-,-,-,-,-,123000|ERNIE-4.5-300B-A47B is a 300B parameter Mixture-of-Experts (MoE) language model developed by Baidu a|Y
inception/mercury|mercury|Inception: Mercury|C|2.5000000000e-07,1.0000000000e-06|128000,16384|JST|-,-,-,-,-,-,-,-,-,128000|Mercury is the first diffusion large language model (dLLM). Applying a breakthrough discrete diffusi|Y
mistralai/mistral-small-3.2-24b-instruct|mistral-small-3.2-24|Mistral: Mistral Small 3.2 24B|C|6.0000000000e-08,1.8000000000e-07|131072,131072|JSTV|-,-,-,-,-,-,-,-,-,131072|Mistral-Small-3.2-24B-Instruct-2506 is an updated 24B parameter model from Mistral optimized for ins|Y
minimax/minimax-m1|minimax-m1|MiniMax: MiniMax M1|C|4.0000000000e-07,2.2000000000e-06|1000000,40000|KT|-,-,-,-,-,-,-,-,-,1000000|MiniMax-M1 is a large-scale, open-weight reasoning model designed for extended context and high-effi|Y
google/gemini-2.5-flash|gemini-2.5-flash|Google: Gemini 2.5 Flash|C|3.0000000000e-07,2.5000000000e-06|1048576,65535|JKSTV|-,-,-,-,-,-,-,-,-,1048576|Gemini 2.5 Flash is Google's state-of-the-art workhorse model, specifically designed for advanced re|Y
google/gemini-2.5-pro|gemini-2.5-pro|Google: Gemini 2.5 Pro|C|1.2500000000e-06,1e-05|1048576,65536|JKSTV|-,-,-,-,-,-,-,-,-,1048576|Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathem|Y
moonshotai/kimi-dev-72b|kimi-dev-72b|MoonshotAI: Kimi Dev 72B|C|2.9000000000e-07,1.1500000000e-06|131072,131072|JKS|-,-,-,-,-,-,-,-,-,131072|Kimi-Dev-72B is an open-source large language model fine-tuned for software engineering and issue re|Y
openai/o3-pro|o3-pro|OpenAI: o3 Pro|C|2e-05,8e-05|200000,100000|JKSTV|-,-,-,-,-,-,-,-,-,200000|The o-series of models are trained with reinforcement learning to think before they answer and perfo|Y
x-ai/grok-3-mini|grok-3-mini|xAI: Grok 3 Mini|C|3.0000000000e-07,5.0000000000e-07|131072,32768|JKST|-,-,-,-,-,-,-,-,-,131072|A lightweight model that thinks before responding. Fast, smart, and great for logic-based tasks that|Y
x-ai/grok-3|grok-3|xAI: Grok 3|C|3.0000000000e-06,1.5e-05|131072,32768|JST|-,-,-,-,-,-,-,-,-,131072|Grok 3 is the latest model from xAI. It's their flagship model that excels at enterprise use cases l|Y
google/gemini-2.5-pro-preview|gemini-2.5-pro-previ|Google: Gemini 2.5 Pro Preview 06-05|C|1.2500000000e-06,1e-05|1048576,65536|JKSTV|-,-,-,-,-,-,-,-,-,1048576|Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathem|Y
deepseek/deepseek-r1-0528-qwen3-8b|deepseek-r1-0528-qwe|DeepSeek: DeepSeek R1 0528 Qwen3 8B|C|6.0000000000e-08,9.0000000000e-08|128000,32000|K|-,-,-,-,-,-,-,-,-,128000|DeepSeek-R1-0528 is a lightly upgraded release of DeepSeek R1 that taps more compute and smarter pos|Y
deepseek/deepseek-r1-0528:free|deepseek-r1-0528:fre|DeepSeek: R1 0528 (free)|C|0,0|163840,40960|K|-,-,-,-,-,-,-,-,-,163840|May 28th update to the [original DeepSeek R1](/deepseek/deepseek-r1) Performance on par with [OpenAI|Y
deepseek/deepseek-r1-0528|deepseek-r1-0528|DeepSeek: R1 0528|C|4.0000000000e-07,1.7500000000e-06|163840,65536|JKST|-,-,-,-,-,-,-,-,-,163840|May 28th update to the [original DeepSeek R1](/deepseek/deepseek-r1) Performance on par with [OpenAI|Y
anthropic/claude-opus-4|claude-opus-4|Anthropic: Claude Opus 4|C|1.5e-05,7.5e-05|200000,32000|KTV|-,-,-,-,-,-,-,-,-,200000|Claude Opus 4 is benchmarked as the world’s best coding model, at time of release, bringing sustaine|Y
anthropic/claude-sonnet-4|claude-sonnet-4|Anthropic: Claude Sonnet 4|C|3.0000000000e-06,1.5e-05|1000000,64000|KTV|-,-,-,-,-,-,-,-,-,1000000|Claude Sonnet 4 significantly enhances the capabilities of its predecessor, Sonnet 3.7, excelling in|Y
mistralai/devstral-small-2505|devstral-small-2505|Mistral: Devstral Small 2505|C|6.0000000000e-08,1.2000000000e-07|128000,32000|J|-,-,-,-,-,-,-,-,-,128000|Devstral-Small-2505 is a 24B parameter agentic LLM fine-tuned from Mistral-Small-3.1, jointly develo|Y
google/gemma-3n-e4b-it:free|gemma-3n-e4b-it:free|Google: Gemma 3n 4B (free)|C|0,0|8192,2048|J|-,-,-,-,-,-,-,-,-,8192|Gemma 3n E4B-it is optimized for efficient execution on mobile and low-resource devices, such as pho|Y
google/gemma-3n-e4b-it|gemma-3n-e4b-it|Google: Gemma 3n 4B|C|2.0000000000e-08,4.0000000000e-08|32768,8192||-,-,-,-,-,-,-,-,-,32768|Gemma 3n E4B-it is optimized for efficient execution on mobile and low-resource devices, such as pho|Y
openai/codex-mini|codex-mini|OpenAI: Codex Mini|C|1.5000000000e-06,6.0000000000e-06|200000,100000|JKSTV|-,-,-,-,-,-,-,-,-,200000|codex-mini-latest is a fine-tuned version of o4-mini specifically for use in Codex CLI. For direct u|Y
nousresearch/deephermes-3-mistral-24b-preview|deephermes-3-mistral|Nous: DeepHermes 3 Mistral 24B Preview|C|2.0000000000e-08,1.0000000000e-07|32768,32768|JKST|-,-,-,-,-,-,-,-,-,32768|DeepHermes 3 (Mistral 24B Preview) is an instruction-tuned language model by Nous Research based on|Y
mistralai/mistral-medium-3|mistral-medium-3|Mistral: Mistral Medium 3|C|4.0000000000e-07,2.0000000000e-06|131072,32768|JSTV|-,-,-,-,-,-,-,-,-,131072|Mistral Medium 3 is a high-performance enterprise-grade language model designed to deliver frontier-|Y
google/gemini-2.5-pro-preview-05-06|gemini-2.5-pro-previ|Google: Gemini 2.5 Pro Preview 05-06|C|1.2500000000e-06,1e-05|1048576,65535|JKSTV|-,-,-,-,-,-,-,-,-,1048576|Gemini 2.5 Pro is Google’s state-of-the-art AI model designed for advanced reasoning, coding, mathem|Y
arcee-ai/spotlight|spotlight|Arcee AI: Spotlight|C|1.8000000000e-07,1.8000000000e-07|131072,65537|V|-,-,-,-,-,-,-,-,-,131072|Spotlight is a 7‑billion‑parameter vision‑language model derived from Qwen 2.5‑VL and fine‑tuned by|Y
arcee-ai/maestro-reasoning|maestro-reasoning|Arcee AI: Maestro Reasoning|C|9.0000000000e-07,3.3000000000e-06|131072,32000||-,-,-,-,-,-,-,-,-,131072|Maestro Reasoning is Arcee's flagship analysis model: a 32 B‑parameter derivative of Qwen 2.5‑32 B t|Y
arcee-ai/virtuoso-large|virtuoso-large|Arcee AI: Virtuoso Large|C|7.5000000000e-07,1.2000000000e-06|131072,64000|T|-,-,-,-,-,-,-,-,-,131072|Virtuoso‑Large is Arcee's top‑tier general‑purpose LLM at 72 B parameters, tuned to tackle cross‑dom|Y
arcee-ai/coder-large|coder-large|Arcee AI: Coder Large|C|5.0000000000e-07,8.0000000000e-07|32768,8192||-,-,-,-,-,-,-,-,-,32768|Coder‑Large is a 32 B‑parameter offspring of Qwen 2.5‑Instruct that has been further trained on perm|Y
microsoft/phi-4-reasoning-plus|phi-4-reasoning-plus|Microsoft: Phi 4 Reasoning Plus|C|7.0000000000e-08,3.5000000000e-07|32768,8192|JK|-,-,-,-,-,-,-,-,-,32768|Phi-4-reasoning-plus is an enhanced 14B parameter model from Microsoft, fine-tuned from Phi-4 with a|Y
inception/mercury-coder|mercury-coder|Inception: Mercury Coder|C|2.5000000000e-07,1.0000000000e-06|128000,16384|JST|-,-,-,-,-,-,-,-,-,128000|Mercury Coder is the first diffusion large language model (dLLM). Applying a breakthrough discrete d|Y
qwen/qwen3-4b:free|qwen3-4b:free|Qwen: Qwen3 4B (free)|C|0,0|40960,10240|JKST|-,-,-,-,-,-,-,-,-,40960|Qwen3-4B is a 4 billion parameter dense language model from the Qwen3 series, designed to support bo|Y
deepseek/deepseek-prover-v2|deepseek-prover-v2|DeepSeek: DeepSeek Prover V2|C|5.0000000000e-07,2.1800000000e-06|163840,40960|J|-,-,-,-,-,-,-,-,-,163840|DeepSeek Prover V2 is a 671B parameter model, speculated to be geared towards logic and mathematics.|Y
meta-llama/llama-guard-4-12b|llama-guard-4-12b|Meta: Llama Guard 4 12B|C|1.8000000000e-07,1.8000000000e-07|163840,40960|JV|-,-,-,-,-,-,-,-,-,163840|Llama Guard 4 is a Llama 4 Scout-derived multimodal pretrained model, fine-tuned for content safety|Y
qwen/qwen3-30b-a3b|qwen3-30b-a3b|Qwen: Qwen3 30B A3B|C|6.0000000000e-08,2.2000000000e-07|40960,40960|JKST|-,-,-,-,-,-,-,-,-,40960|Qwen3, the latest generation in the Qwen large language model series, features both dense and mixtur|Y
qwen/qwen3-8b|qwen3-8b|Qwen: Qwen3 8B|C|4.0000000000e-08,1.4000000000e-07|128000,20000|JKST|-,-,-,-,-,-,-,-,-,128000|Qwen3-8B is a dense 8.2B parameter causal language model from the Qwen3 series, designed for both re|Y
qwen/qwen3-14b|qwen3-14b|Qwen: Qwen3 14B|C|5.0000000000e-08,2.2000000000e-07|40960,40960|JKST|-,-,-,-,-,-,-,-,-,40960|Qwen3-14B is a dense 14.8B parameter causal language model from the Qwen3 series, designed for both|Y
qwen/qwen3-32b|qwen3-32b|Qwen: Qwen3 32B|C|8.0000000000e-08,2.4000000000e-07|40960,40960|JKST|-,-,-,-,-,-,-,-,-,40960|Qwen3-32B is a dense 32.8B parameter causal language model from the Qwen3 series, optimized for both|Y
qwen/qwen3-235b-a22b|qwen3-235b-a22b|Qwen: Qwen3 235B A22B|C|1.8000000000e-07,5.4000000000e-07|40960,40960|JKST|-,-,-,-,-,-,-,-,-,40960|Qwen3-235B-A22B is a 235B parameter mixture-of-experts (MoE) model developed by Qwen, activating 22B|Y
tngtech/deepseek-r1t-chimera:free|deepseek-r1t-chimera|TNG: DeepSeek R1T Chimera (free)|C|0,0|163840,40960|K|-,-,-,-,-,-,-,-,-,163840|DeepSeek-R1T-Chimera is created by merging DeepSeek-R1 and DeepSeek-V3 (0324), combining the reasoni|Y
tngtech/deepseek-r1t-chimera|deepseek-r1t-chimera|TNG: DeepSeek R1T Chimera|C|3.0000000000e-07,1.2000000000e-06|163840,163840|JKS|-,-,-,-,-,-,-,-,-,163840|DeepSeek-R1T-Chimera is created by merging DeepSeek-R1 and DeepSeek-V3 (0324), combining the reasoni|Y
openai/o4-mini-high|o4-mini-high|OpenAI: o4 Mini High|C|1.1000000000e-06,4.4000000000e-06|200000,100000|JKSTV|-,-,-,-,-,-,-,-,-,200000|OpenAI o4-mini-high is the same model as [o4-mini](/openai/o4-mini) with reasoning_effort set to hig|Y
openai/o3|o3|OpenAI: o3|C|2.0000000000e-06,8.0000000000e-06|200000,100000|JKSTV|-,-,-,-,-,-,-,-,-,200000|o3 is a well-rounded and powerful model across domains. It sets a new standard for math, science, co|Y
openai/o4-mini|o4-mini|OpenAI: o4 Mini|C|1.1000000000e-06,4.4000000000e-06|200000,100000|JKSTV|-,-,-,-,-,-,-,-,-,200000|OpenAI o4-mini is a compact reasoning model in the o-series, optimized for fast, cost-efficient perf|Y
qwen/qwen2.5-coder-7b-instruct|qwen2.5-coder-7b-ins|Qwen: Qwen2.5 Coder 7B Instruct|C|3.0000000000e-08,9.0000000000e-08|32768,8192|JS|-,-,-,-,-,-,-,-,-,32768|Qwen2.5-Coder-7B-Instruct is a 7B parameter instruction-tuned language model optimized for code-rela|Y
openai/gpt-4.1|gpt-4.1|OpenAI: GPT-4.1|C|2.0000000000e-06,8.0000000000e-06|1047576,32768|JSTV|-,-,-,-,-,-,-,-,-,1047576|GPT-4.1 is a flagship large language model optimized for advanced instruction following, real-world|Y
openai/gpt-4.1-mini|gpt-4.1-mini|OpenAI: GPT-4.1 Mini|C|4.0000000000e-07,1.6000000000e-06|1047576,32768|JSTV|-,-,-,-,-,-,-,-,-,1047576|GPT-4.1 Mini is a mid-sized model delivering performance competitive with GPT-4o at substantially lo|Y
openai/gpt-4.1-nano|gpt-4.1-nano|OpenAI: GPT-4.1 Nano|C|1.0000000000e-07,4.0000000000e-07|1047576,32768|JSTV|-,-,-,-,-,-,-,-,-,1047576|For tasks that demand low latency, GPT‑4.1 nano is the fastest and cheapest model in the GPT-4.1 ser|Y
eleutherai/llemma_7b|llemma_7b|EleutherAI: Llemma 7b|C|8.0000000000e-07,1.2000000000e-06|4096,4096||-,-,-,-,-,-,-,-,-,4096|Llemma 7B is a language model for mathematics. It was initialized with Code Llama 7B weights, and tr|Y
alfredpros/codellama-7b-instruct-solidity|codellama-7b-instruc|AlfredPros: CodeLLaMa 7B Instruct Solidity|C|8.0000000000e-07,1.2000000000e-06|4096,4096||-,-,-,-,-,-,-,-,-,4096|A finetuned 7 billion parameters Code LLaMA - Instruct model to generate Solidity smart contract usi|Y
arliai/qwq-32b-arliai-rpr-v1|qwq-32b-arliai-rpr-v|ArliAI: QwQ 32B RpR v1|C|3.0000000000e-08,1.1000000000e-07|32768,32768|JKS|-,-,-,-,-,-,-,-,-,32768|QwQ-32B-ArliAI-RpR-v1 is a 32B parameter model fine-tuned from Qwen/QwQ-32B using a curated creative|Y
x-ai/grok-3-mini-beta|grok-3-mini-beta|xAI: Grok 3 Mini Beta|C|3.0000000000e-07,5.0000000000e-07|131072,32768|JKT|-,-,-,-,-,-,-,-,-,131072|Grok 3 Mini is a lightweight, smaller thinking model. Unlike traditional models that generate answer|Y
x-ai/grok-3-beta|grok-3-beta|xAI: Grok 3 Beta|C|3.0000000000e-06,1.5e-05|131072,32768|JT|-,-,-,-,-,-,-,-,-,131072|Grok 3 is the latest model from xAI. It's their flagship model that excels at enterprise use cases l|Y
nvidia/llama-3.1-nemotron-ultra-253b-v1|llama-3.1-nemotron-u|NVIDIA: Llama 3.1 Nemotron Ultra 253B v1|C|6.0000000000e-07,1.8000000000e-06|131072,32768|JKS|-,-,-,-,-,-,-,-,-,131072|Llama-3.1-Nemotron-Ultra-253B-v1 is a large language model (LLM) optimized for advanced reasoning, h|Y
meta-llama/llama-4-maverick|llama-4-maverick|Meta: Llama 4 Maverick|C|1.5000000000e-07,6.0000000000e-07|1048576,16384|JSTV|-,-,-,-,-,-,-,-,-,1048576|Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language model from Meta, built o|Y
meta-llama/llama-4-scout|llama-4-scout|Meta: Llama 4 Scout|C|8.0000000000e-08,3.0000000000e-07|327680,16384|JSTV|-,-,-,-,-,-,-,-,-,327680|Llama 4 Scout 17B Instruct (16E) is a mixture-of-experts (MoE) language model developed by Meta, act|Y
qwen/qwen2.5-vl-32b-instruct|qwen2.5-vl-32b-instr|Qwen: Qwen2.5 VL 32B Instruct|C|5.0000000000e-08,2.2000000000e-07|16384,16384|JSV|-,-,-,-,-,-,-,-,-,16384|Qwen2.5-VL-32B is a multimodal vision-language model fine-tuned through reinforcement learning for e|Y
deepseek/deepseek-chat-v3-0324|deepseek-chat-v3-032|DeepSeek: DeepSeek V3 0324|C|2.0000000000e-07,8.8000000000e-07|163840,40960|JKST|-,-,-,-,-,-,-,-,-,163840|DeepSeek V3, a 685B-parameter, mixture-of-experts model, is the latest iteration of the flagship cha|Y
openai/o1-pro|o1-pro|OpenAI: o1-pro|C|0.00015,0.0006|200000,100000|JKSV|-,-,-,-,-,-,-,-,-,200000|The o1 series of models are trained with reinforcement learning to think before they answer and perf|Y
mistralai/mistral-small-3.1-24b-instruct:free|mistral-small-3.1-24|Mistral: Mistral Small 3.1 24B (free)|C|0,0|128000,32000|JSTV|-,-,-,-,-,-,-,-,-,128000|Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billio|Y
mistralai/mistral-small-3.1-24b-instruct|mistral-small-3.1-24|Mistral: Mistral Small 3.1 24B|C|3.0000000000e-08,1.1000000000e-07|131072,131072|JSTV|-,-,-,-,-,-,-,-,-,131072|Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501), featuring 24 billio|Y
allenai/olmo-2-0325-32b-instruct|olmo-2-0325-32b-inst|AllenAI: Olmo 2 32B Instruct|C|5.0000000000e-08,2.0000000000e-07|128000,32000||-,-,-,-,-,-,-,-,-,128000|OLMo-2 32B Instruct is a supervised instruction-finetuned variant of the OLMo-2 32B March 2025 base|Y
google/gemma-3-4b-it:free|gemma-3-4b-it:free|Google: Gemma 3 4B (free)|C|0,0|32768,8192|JSV|-,-,-,-,-,-,-,-,-,32768|Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles cont|Y
google/gemma-3-4b-it|gemma-3-4b-it|Google: Gemma 3 4B|C|2.0000000000e-08,7.0000000000e-08|96000,24000|JV|-,-,-,-,-,-,-,-,-,96000|Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles cont|Y
google/gemma-3-12b-it:free|gemma-3-12b-it:free|Google: Gemma 3 12B (free)|C|0,0|32768,8192|V|-,-,-,-,-,-,-,-,-,32768|Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles cont|Y
google/gemma-3-12b-it|gemma-3-12b-it|Google: Gemma 3 12B|C|3.0000000000e-08,1.0000000000e-07|131072,131072|JSV|-,-,-,-,-,-,-,-,-,131072|Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles cont|Y
cohere/command-a|command-a|Cohere: Command A|C|2.5000000000e-06,1e-05|256000,8192|JS|-,-,-,-,-,-,-,-,-,256000|Command A is an open-weights 111B parameter model with a 256k context window focused on delivering g|Y
openai/gpt-4o-mini-search-preview|gpt-4o-mini-search-p|OpenAI: GPT-4o-mini Search Preview|C|1.5000000000e-07,6.0000000000e-07|128000,16384|JS|-,-,-,-,-,-,-,-,-,128000|GPT-4o mini Search Preview is a specialized model for web search in Chat Completions. It is trained|Y
openai/gpt-4o-search-preview|gpt-4o-search-previe|OpenAI: GPT-4o Search Preview|C|2.5000000000e-06,1e-05|128000,16384|JS|-,-,-,-,-,-,-,-,-,128000|GPT-4o Search Previewis a specialized model for web search in Chat Completions. It is trained to und|Y
google/gemma-3-27b-it:free|gemma-3-27b-it:free|Google: Gemma 3 27B (free)|C|0,0|131072,32768|JSTV|-,-,-,-,-,-,-,-,-,131072|Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles cont|Y
google/gemma-3-27b-it|gemma-3-27b-it|Google: Gemma 3 27B|C|4.0000000000e-08,6.0000000000e-08|131072,32768|JSTV|-,-,-,-,-,-,-,-,-,131072|Gemma 3 introduces multimodality, supporting vision-language input and text outputs. It handles cont|Y
thedrummer/skyfall-36b-v2|skyfall-36b-v2|TheDrummer: Skyfall 36B V2|C|5.5000000000e-07,8.0000000000e-07|32768,32768||-,-,-,-,-,-,-,-,-,32768|Skyfall 36B v2 is an enhanced iteration of Mistral Small 2501, specifically fine-tuned for improved|Y
microsoft/phi-4-multimodal-instruct|phi-4-multimodal-ins|Microsoft: Phi 4 Multimodal Instruct|C|5.0000000000e-08,1.0000000000e-07|131072,32768|JV|-,-,-,-,-,-,-,-,-,131072|Phi-4 Multimodal Instruct is a versatile 5.6B parameter foundation model that combines advanced reas|Y
perplexity/sonar-reasoning-pro|sonar-reasoning-pro|Perplexity: Sonar Reasoning Pro|C|2.0000000000e-06,8.0000000000e-06|128000,32000|KV|-,-,-,-,-,-,-,-,-,128000|Note: Sonar Pro pricing includes Perplexity search pricing. See [details here](https://docs.perplexi|Y
perplexity/sonar-pro|sonar-pro|Perplexity: Sonar Pro|C|3.0000000000e-06,1.5e-05|200000,8000|V|-,-,-,-,-,-,-,-,-,200000|Note: Sonar Pro pricing includes Perplexity search pricing. See [details here](https://docs.perplexi|Y
perplexity/sonar-deep-research|sonar-deep-research|Perplexity: Sonar Deep Research|C|2.0000000000e-06,8.0000000000e-06|128000,32000|K|-,-,-,-,-,-,-,-,-,128000|Sonar Deep Research is a research-focused model designed for multi-step retrieval, synthesis, and re|Y
qwen/qwq-32b|qwq-32b|Qwen: QwQ 32B|C|1.5000000000e-07,4.0000000000e-07|32768,8192|JKST|-,-,-,-,-,-,-,-,-,32768|QwQ is the reasoning model of the Qwen series. Compared with conventional instruction-tuned models,|Y
google/gemini-2.0-flash-lite-001|gemini-2.0-flash-lit|Google: Gemini 2.0 Flash Lite|C|7.0000000000e-08,3.0000000000e-07|1048576,8192|JSTV|-,-,-,-,-,-,-,-,-,1048576|Gemini 2.0 Flash Lite offers a significantly faster time to first token (TTFT) compared to [Gemini F|Y
anthropic/claude-3.7-sonnet:thinking|claude-3.7-sonnet:th|Anthropic: Claude 3.7 Sonnet (thinking)|C|3.0000000000e-06,1.5e-05|200000,64000|KTV|-,-,-,-,-,-,-,-,-,200000|Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-s|Y
anthropic/claude-3.7-sonnet|claude-3.7-sonnet|Anthropic: Claude 3.7 Sonnet|C|3.0000000000e-06,1.5e-05|200000,64000|KTV|-,-,-,-,-,-,-,-,-,200000|Claude 3.7 Sonnet is an advanced large language model with improved reasoning, coding, and problem-s|Y
mistralai/mistral-saba|mistral-saba|Mistral: Saba|C|2.0000000000e-07,6.0000000000e-07|32768,8192|JST|-,-,-,-,-,-,-,-,-,32768|Mistral Saba is a 24B-parameter language model specifically designed for the Middle East and South A|Y
meta-llama/llama-guard-3-8b|llama-guard-3-8b|Llama Guard 3 8B|C|2.0000000000e-08,6.0000000000e-08|131072,32768|J|-,-,-,-,-,-,-,-,-,131072|Llama Guard 3 is a Llama-3.1-8B pretrained model, fine-tuned for content safety classification. Simi|Y
openai/o3-mini-high|o3-mini-high|OpenAI: o3 Mini High|C|1.1000000000e-06,4.4000000000e-06|200000,100000|JST|-,-,-,-,-,-,-,-,-,200000|OpenAI o3-mini-high is the same model as [o3-mini](/openai/o3-mini) with reasoning_effort set to hig|Y
google/gemini-2.0-flash-001|gemini-2.0-flash-001|Google: Gemini 2.0 Flash|C|1.0000000000e-07,4.0000000000e-07|1048576,8192|JSTV|-,-,-,-,-,-,-,-,-,1048576|Gemini Flash 2.0 offers a significantly faster time to first token (TTFT) compared to [Gemini Flash|Y
qwen/qwen-vl-plus|qwen-vl-plus|Qwen: Qwen VL Plus|C|2.1000000000e-07,6.3000000000e-07|7500,1500|JV|-,-,-,-,-,-,-,-,-,7500|Qwen's Enhanced Large Visual Language Model. Significantly upgraded for detailed recognition capabil|Y
aion-labs/aion-1.0|aion-1.0|AionLabs: Aion-1.0|C|4.0000000000e-06,8.0000000000e-06|131072,32768|K|-,-,-,-,-,-,-,-,-,131072|Aion-1.0 is a multi-model system designed for high performance across various tasks, including reaso|Y
aion-labs/aion-1.0-mini|aion-1.0-mini|AionLabs: Aion-1.0-Mini|C|7.0000000000e-07,1.4000000000e-06|131072,32768|K|-,-,-,-,-,-,-,-,-,131072|Aion-1.0-Mini 32B parameter model is a distilled version of the DeepSeek-R1 model, designed for stro|Y
aion-labs/aion-rp-llama-3.1-8b|aion-rp-llama-3.1-8b|AionLabs: Aion-RP 1.0 (8B)|C|8.0000000000e-07,1.6000000000e-06|32768,32768||-,-,-,-,-,-,-,-,-,32768|Aion-RP-Llama-3.1-8B ranks the highest in the character evaluation portion of the RPBench-Auto bench|Y
qwen/qwen-vl-max|qwen-vl-max|Qwen: Qwen VL Max|C|8.0000000000e-07,3.2000000000e-06|131072,8192|JTV|-,-,-,-,-,-,-,-,-,131072|Qwen VL Max is a visual understanding model with 7500 tokens context length. It excels in delivering|Y
qwen/qwen-turbo|qwen-turbo|Qwen: Qwen-Turbo|C|5.0000000000e-08,2.0000000000e-07|1000000,8192|JT|-,-,-,-,-,-,-,-,-,1000000|Qwen-Turbo, based on Qwen2.5, is a 1M context model that provides fast speed and low cost, suitable|Y
qwen/qwen2.5-vl-72b-instruct|qwen2.5-vl-72b-instr|Qwen: Qwen2.5 VL 72B Instruct|C|7.0000000000e-08,2.6000000000e-07|32768,32768|JSV|-,-,-,-,-,-,-,-,-,32768|Qwen2.5-VL is proficient in recognizing common objects such as flowers, birds, fish, and insects. It|Y
qwen/qwen-plus|qwen-plus|Qwen: Qwen-Plus|C|4.0000000000e-07,1.2000000000e-06|131072,8192|JT|-,-,-,-,-,-,-,-,-,131072|Qwen-Plus, based on the Qwen2.5 foundation model, is a 131K context model with a balanced performanc|Y
qwen/qwen-max|qwen-max|Qwen: Qwen-Max|C|1.6000000000e-06,6.4000000000e-06|32768,8192|JT|-,-,-,-,-,-,-,-,-,32768|Qwen-Max, based on Qwen2.5, provides the best inference performance among [Qwen models](/qwen), espe|Y
openai/o3-mini|o3-mini|OpenAI: o3 Mini|C|1.1000000000e-06,4.4000000000e-06|200000,100000|JST|-,-,-,-,-,-,-,-,-,200000|OpenAI o3-mini is a cost-efficient language model optimized for STEM reasoning tasks, particularly e|Y
mistralai/mistral-small-24b-instruct-2501|mistral-small-24b-in|Mistral: Mistral Small 3|C|3.0000000000e-08,1.1000000000e-07|32768,32768|JST|-,-,-,-,-,-,-,-,-,32768|Mistral Small 3 is a 24B-parameter language model optimized for low-latency performance across commo|Y
deepseek/deepseek-r1-distill-qwen-32b|deepseek-r1-distill-|DeepSeek: R1 Distill Qwen 32B|C|2.7000000000e-07,2.7000000000e-07|131072,32768|JKS|-,-,-,-,-,-,-,-,-,131072|DeepSeek R1 Distill Qwen 32B is a distilled large language model based on [Qwen 2.5 32B](https://hug|Y
deepseek/deepseek-r1-distill-qwen-14b|deepseek-r1-distill-|DeepSeek: R1 Distill Qwen 14B|C|1.5000000000e-07,1.5000000000e-07|32768,16384|JKS|-,-,-,-,-,-,-,-,-,32768|DeepSeek R1 Distill Qwen 14B is a distilled large language model based on [Qwen 2.5 14B](https://hug|Y
perplexity/sonar-reasoning|sonar-reasoning|Perplexity: Sonar Reasoning|C|1.0000000000e-06,5.0000000000e-06|127000,31750|K|-,-,-,-,-,-,-,-,-,127000|Sonar Reasoning is a reasoning model provided by Perplexity based on [DeepSeek R1](/deepseek/deepsee|Y
perplexity/sonar|sonar|Perplexity: Sonar|C|1.0000000000e-06,1.0000000000e-06|127072,31768|V|-,-,-,-,-,-,-,-,-,127072|Sonar is lightweight, affordable, fast, and simple to use — now featuring citations and the ability|Y
deepseek/deepseek-r1-distill-llama-70b|deepseek-r1-distill-|DeepSeek: R1 Distill Llama 70B|C|3.0000000000e-08,1.1000000000e-07|131072,131072|JKST|-,-,-,-,-,-,-,-,-,131072|DeepSeek R1 Distill Llama 70B is a distilled large language model based on [Llama-3.3-70B-Instruct](|Y
deepseek/deepseek-r1|deepseek-r1|DeepSeek: R1|C|3.0000000000e-07,1.2000000000e-06|163840,40960|JKST|-,-,-,-,-,-,-,-,-,163840|DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-sourced and with full|Y
minimax/minimax-01|minimax-01|MiniMax: MiniMax-01|C|2.0000000000e-07,1.1000000000e-06|1000192,1000192|V|-,-,-,-,-,-,-,-,-,1000192|MiniMax-01 is a combines MiniMax-Text-01 for text generation and MiniMax-VL-01 for image understandi|Y
microsoft/phi-4|phi-4|Microsoft: Phi 4|C|6.0000000000e-08,1.4000000000e-07|16384,4096|JS|-,-,-,-,-,-,-,-,-,16384|[Microsoft Research](/microsoft) Phi-4 is designed to perform well in complex reasoning tasks and ca|Y
sao10k/l3.1-70b-hanami-x1|l3.1-70b-hanami-x1|Sao10K: Llama 3.1 70B Hanami x1|C|3.0000000000e-06,3.0000000000e-06|16000,4000||-,-,-,-,-,-,-,-,-,16000|This is [Sao10K](/sao10k)'s experiment over [Euryale v2.2](/sao10k/l3.1-euryale-70b).|Y
deepseek/deepseek-chat|deepseek-chat|DeepSeek: DeepSeek V3|C|3.0000000000e-07,1.2000000000e-06|163840,163840|JST|-,-,-,-,-,-,-,-,-,163840|DeepSeek-V3 is the latest model from the DeepSeek team, building upon the instruction following and|Y
sao10k/l3.3-euryale-70b|l3.3-euryale-70b|Sao10K: Llama 3.3 Euryale 70B|C|6.5000000000e-07,7.5000000000e-07|131072,16384|JS|-,-,-,-,-,-,-,-,-,131072|Euryale L3.3 70B is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k). It|Y
openai/o1|o1|OpenAI: o1|C|1.5e-05,6e-05|200000,100000|JSTV|-,-,-,-,-,-,-,-,-,200000|The latest and strongest model family from OpenAI, o1 is designed to spend more time thinking before|Y
cohere/command-r7b-12-2024|command-r7b-12-2024|Cohere: Command R7B (12-2024)|C|4.0000000000e-08,1.5000000000e-07|128000,4000|JS|-,-,-,-,-,-,-,-,-,128000|Command R7B (12-2024) is a small, fast update of the Command R+ model, delivered in December 2024. I|Y
google/gemini-2.0-flash-exp:free|gemini-2.0-flash-exp|Google: Gemini 2.0 Flash Experimental (free)|C|0,0|1048576,8192|JTV|-,-,-,-,-,-,-,-,-,1048576|Gemini Flash 2.0 offers a significantly faster time to first token (TTFT) compared to [Gemini Flash|Y
meta-llama/llama-3.3-70b-instruct:free|llama-3.3-70b-instru|Meta: Llama 3.3 70B Instruct (free)|C|0,0|131072,32768|T|-,-,-,-,-,-,-,-,-,131072|The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned gen|Y
meta-llama/llama-3.3-70b-instruct|llama-3.3-70b-instru|Meta: Llama 3.3 70B Instruct|C|1.0000000000e-07,3.2000000000e-07|131072,16384|JST|-,-,-,-,-,-,-,-,-,131072|The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and instruction tuned gen|Y
amazon/nova-lite-v1|nova-lite-v1|Amazon: Nova Lite 1.0|C|6.0000000000e-08,2.4000000000e-07|300000,5120|TV|-,-,-,-,-,-,-,-,-,300000|Amazon Nova Lite 1.0 is a very low-cost multimodal model from Amazon that focused on fast processing|Y
amazon/nova-micro-v1|nova-micro-v1|Amazon: Nova Micro 1.0|C|4.0000000000e-08,1.4000000000e-07|128000,5120|T|-,-,-,-,-,-,-,-,-,128000|Amazon Nova Micro 1.0 is a text-only model that delivers the lowest latency responses in the Amazon|Y
amazon/nova-pro-v1|nova-pro-v1|Amazon: Nova Pro 1.0|C|8.0000000000e-07,3.2000000000e-06|300000,5120|TV|-,-,-,-,-,-,-,-,-,300000|Amazon Nova Pro 1.0 is a capable multimodal model from Amazon focused on providing a combination of|Y
openai/gpt-4o-2024-11-20|gpt-4o-2024-11-20|OpenAI: GPT-4o (2024-11-20)|C|2.5000000000e-06,1e-05|128000,16384|JSTV|-,-,-,-,-,-,-,-,-,128000|The 2024-11-20 version of GPT-4o offers a leveled-up creative writing ability with more natural, eng|Y
mistralai/mistral-large-2411|mistral-large-2411|Mistral Large 2411|C|2.0000000000e-06,6.0000000000e-06|131072,32768|JST|-,-,-,-,-,-,-,-,-,131072|Mistral Large 2 2411 is an update of [Mistral Large 2](/mistralai/mistral-large) released together w|Y
mistralai/mistral-large-2407|mistral-large-2407|Mistral Large 2407|C|2.0000000000e-06,6.0000000000e-06|131072,32768|JST|-,-,-,-,-,-,-,-,-,131072|This is Mistral AI's flagship model, Mistral Large 2 (version mistral-large-2407). It's a proprietar|Y
mistralai/pixtral-large-2411|pixtral-large-2411|Mistral: Pixtral Large 2411|C|2.0000000000e-06,6.0000000000e-06|131072,32768|JSTV|-,-,-,-,-,-,-,-,-,131072|Pixtral Large is a 124B parameter, open-weight, multimodal model built on top of [Mistral Large 2](/|Y
qwen/qwen-2.5-coder-32b-instruct|qwen-2.5-coder-32b-i|Qwen2.5 Coder 32B Instruct|C|3.0000000000e-08,1.1000000000e-07|32768,32768|JS|-,-,-,-,-,-,-,-,-,32768|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (formerly known as Co|Y
raifle/sorcererlm-8x22b|sorcererlm-8x22b|SorcererLM 8x22B|C|4.5000000000e-06,4.5000000000e-06|16000,4000||-,-,-,-,-,-,-,-,-,16000|SorcererLM is an advanced RP and storytelling model, built as a Low-rank 16-bit LoRA fine-tuned on [|Y
thedrummer/unslopnemo-12b|unslopnemo-12b|TheDrummer: UnslopNemo 12B|C|4.0000000000e-07,4.0000000000e-07|32768,8192|JST|-,-,-,-,-,-,-,-,-,32768|UnslopNemo v4.1 is the latest addition from the creator of Rocinante, designed for adventure writing|Y
anthropic/claude-3.5-haiku-20241022|claude-3.5-haiku-202|Anthropic: Claude 3.5 Haiku (2024-10-22)|C|8.0000000000e-07,4.0000000000e-06|200000,8192|TV|-,-,-,-,-,-,-,-,-,200000|Claude 3.5 Haiku features enhancements across all skill sets including coding, tool use, and reasoni|Y
anthropic/claude-3.5-haiku|claude-3.5-haiku|Anthropic: Claude 3.5 Haiku|C|8.0000000000e-07,4.0000000000e-06|200000,8192|TV|-,-,-,-,-,-,-,-,-,200000|Claude 3.5 Haiku features offers enhanced capabilities in speed, coding accuracy, and tool use. Engi|Y
anthracite-org/magnum-v4-72b|magnum-v4-72b|Magnum v4 72B|C|3.0000000000e-06,5.0000000000e-06|16384,2048|J|-,-,-,-,-,-,-,-,-,16384|This is a series of models designed to replicate the prose quality of the Claude 3 models, specifica|Y
anthropic/claude-3.5-sonnet|claude-3.5-sonnet|Anthropic: Claude 3.5 Sonnet|C|6.0000000000e-06,3e-05|200000,8192|TV|-,-,-,-,-,-,-,-,-,200000|New Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet speeds, at the same|Y
mistralai/ministral-8b|ministral-8b|Mistral: Ministral 8B|C|1.0000000000e-07,1.0000000000e-07|131072,32768|JST|-,-,-,-,-,-,-,-,-,131072|Ministral 8B is an 8B parameter model featuring a unique interleaved sliding-window attention patter|Y
mistralai/ministral-3b|ministral-3b|Mistral: Ministral 3B|C|4.0000000000e-08,4.0000000000e-08|131072,32768|JST|-,-,-,-,-,-,-,-,-,131072|Ministral 3B is a 3B parameter model optimized for on-device and edge computing. It excels in knowle|Y
qwen/qwen-2.5-7b-instruct|qwen-2.5-7b-instruct|Qwen: Qwen2.5 7B Instruct|C|4.0000000000e-08,1.0000000000e-07|32768,8192||-,-,-,-,-,-,-,-,-,32768|Qwen2.5 7B is the latest series of Qwen large language models. Qwen2.5 brings the following improvem|Y
nvidia/llama-3.1-nemotron-70b-instruct|llama-3.1-nemotron-7|NVIDIA: Llama 3.1 Nemotron 70B Instruct|C|1.2000000000e-06,1.2000000000e-06|131072,16384|JT|-,-,-,-,-,-,-,-,-,131072|NVIDIA's Llama 3.1 Nemotron 70B is a language model designed for generating precise and useful respo|Y
inflection/inflection-3-pi|inflection-3-pi|Inflection: Inflection 3 Pi|C|2.5000000000e-06,1e-05|8000,1024||-,-,-,-,-,-,-,-,-,8000|Inflection 3 Pi powers Inflection's [Pi](https://pi.ai) chatbot, including backstory, emotional inte|Y
inflection/inflection-3-productivity|inflection-3-product|Inflection: Inflection 3 Productivity|C|2.5000000000e-06,1e-05|8000,1024||-,-,-,-,-,-,-,-,-,8000|Inflection 3 Productivity is optimized for following instructions. It is better for tasks requiring|Y
thedrummer/rocinante-12b|rocinante-12b|TheDrummer: Rocinante 12B|C|1.7000000000e-07,4.3000000000e-07|32768,8192|JST|-,-,-,-,-,-,-,-,-,32768|Rocinante 12B is designed for engaging storytelling and rich prose.

Early testers have reported:
-|Y
meta-llama/llama-3.2-3b-instruct:free|llama-3.2-3b-instruc|Meta: Llama 3.2 3B Instruct (free)|C|0,0|131072,32768||-,-,-,-,-,-,-,-,-,131072|Llama 3.2 3B is a 3-billion-parameter multilingual large language model, optimized for advanced natu|Y
meta-llama/llama-3.2-3b-instruct|llama-3.2-3b-instruc|Meta: Llama 3.2 3B Instruct|C|2.0000000000e-08,2.0000000000e-08|131072,16384|JT|-,-,-,-,-,-,-,-,-,131072|Llama 3.2 3B is a 3-billion-parameter multilingual large language model, optimized for advanced natu|Y
meta-llama/llama-3.2-1b-instruct|llama-3.2-1b-instruc|Meta: Llama 3.2 1B Instruct|C|3.0000000000e-08,2.0000000000e-07|60000,15000||-,-,-,-,-,-,-,-,-,60000|Llama 3.2 1B is a 1-billion-parameter language model focused on efficiently performing natural langu|Y
meta-llama/llama-3.2-90b-vision-instruct|llama-3.2-90b-vision|Meta: Llama 3.2 90B Vision Instruct|C|3.5000000000e-07,4.0000000000e-07|32768,16384|JV|-,-,-,-,-,-,-,-,-,32768|The Llama 90B Vision model is a top-tier, 90-billion-parameter multimodal model designed for the mos|Y
meta-llama/llama-3.2-11b-vision-instruct|llama-3.2-11b-vision|Meta: Llama 3.2 11B Vision Instruct|C|5.0000000000e-08,5.0000000000e-08|131072,16384|JV|-,-,-,-,-,-,-,-,-,131072|Llama 3.2 11B Vision is a multimodal model with 11 billion parameters, designed to handle tasks comb|Y
qwen/qwen-2.5-72b-instruct|qwen-2.5-72b-instruc|Qwen2.5 72B Instruct|C|1.2000000000e-07,3.9000000000e-07|32768,16384|JST|-,-,-,-,-,-,-,-,-,32768|Qwen2.5 72B is the latest series of Qwen large language models. Qwen2.5 brings the following improve|Y
neversleep/llama-3.1-lumimaid-8b|llama-3.1-lumimaid-8|NeverSleep: Lumimaid v0.2 8B|C|9.0000000000e-08,6.0000000000e-07|32768,8192|JS|-,-,-,-,-,-,-,-,-,32768|Lumimaid v0.2 8B is a finetune of [Llama 3.1 8B](/models/meta-llama/llama-3.1-8b-instruct) with a \"H|Y
mistralai/pixtral-12b|pixtral-12b|Mistral: Pixtral 12B|C|1.0000000000e-07,1.0000000000e-07|32768,8192|JSTV|-,-,-,-,-,-,-,-,-,32768|The first multi-modal, text+image-to-text model from Mistral AI. Its weights were launched via torre|Y
cohere/command-r-08-2024|command-r-08-2024|Cohere: Command R (08-2024)|C|1.5000000000e-07,6.0000000000e-07|128000,4000|JST|-,-,-,-,-,-,-,-,-,128000|command-r-08-2024 is an update of the [Command R](/models/cohere/command-r) with improved performanc|Y
cohere/command-r-plus-08-2024|command-r-plus-08-20|Cohere: Command R+ (08-2024)|C|2.5000000000e-06,1e-05|128000,4000|JST|-,-,-,-,-,-,-,-,-,128000|command-r-plus-08-2024 is an update of the [Command R+](/models/cohere/command-r-plus) with roughly|Y
sao10k/l3.1-euryale-70b|l3.1-euryale-70b|Sao10K: Llama 3.1 Euryale 70B v2.2|C|6.5000000000e-07,7.5000000000e-07|32768,8192|JST|-,-,-,-,-,-,-,-,-,32768|Euryale L3.1 70B v2.2 is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k|Y
qwen/qwen-2.5-vl-7b-instruct:free|qwen-2.5-vl-7b-instr|Qwen: Qwen2.5-VL 7B Instruct (free)|C|0,0|32768,8192|V|-,-,-,-,-,-,-,-,-,32768|Qwen2.5 VL 7B is a multimodal LLM from the Qwen Team with the following key enhancements:

- SoTA un|Y
qwen/qwen-2.5-vl-7b-instruct|qwen-2.5-vl-7b-instr|Qwen: Qwen2.5-VL 7B Instruct|C|2.0000000000e-07,2.0000000000e-07|32768,8192|V|-,-,-,-,-,-,-,-,-,32768|Qwen2.5 VL 7B is a multimodal LLM from the Qwen Team with the following key enhancements:

- SoTA un|Y
microsoft/phi-3.5-mini-128k-instruct|phi-3.5-mini-128k-in|Microsoft: Phi-3.5 Mini 128K Instruct|C|1.0000000000e-07,1.0000000000e-07|128000,32000|T|-,-,-,-,-,-,-,-,-,128000|Phi-3.5 models are lightweight, state-of-the-art open models. These models were trained with Phi-3 d|Y
nousresearch/hermes-3-llama-3.1-70b|hermes-3-llama-3.1-7|Nous: Hermes 3 70B Instruct|C|3.0000000000e-07,3.0000000000e-07|65536,16384|JS|-,-,-,-,-,-,-,-,-,65536|Hermes 3 is a generalist language model with many improvements over [Hermes 2](/models/nousresearch/|Y
nousresearch/hermes-3-llama-3.1-405b:free|hermes-3-llama-3.1-4|Nous: Hermes 3 405B Instruct (free)|C|0,0|131072,32768||-,-,-,-,-,-,-,-,-,131072|Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced age|Y
nousresearch/hermes-3-llama-3.1-405b|hermes-3-llama-3.1-4|Nous: Hermes 3 405B Instruct|C|1.0000000000e-06,1.0000000000e-06|131072,16384|J|-,-,-,-,-,-,-,-,-,131072|Hermes 3 is a generalist language model with many improvements over Hermes 2, including advanced age|Y
openai/chatgpt-4o-latest|chatgpt-4o-latest|OpenAI: ChatGPT-4o|C|5.0000000000e-06,1.5e-05|128000,16384|JSV|-,-,-,-,-,-,-,-,-,128000|OpenAI ChatGPT 4o is continually updated by OpenAI to point to the current version of GPT-4o used by|Y
sao10k/l3-lunaris-8b|l3-lunaris-8b|Sao10K: Llama 3 8B Lunaris|C|4.0000000000e-08,5.0000000000e-08|8192,2048|JS|-,-,-,-,-,-,-,-,-,8192|Lunaris 8B is a versatile generalist and roleplaying model based on Llama 3. It's a strategic merge|Y
openai/gpt-4o-2024-08-06|gpt-4o-2024-08-06|OpenAI: GPT-4o (2024-08-06)|C|2.5000000000e-06,1e-05|128000,16384|JSTV|-,-,-,-,-,-,-,-,-,128000|The 2024-08-06 version of GPT-4o offers improved performance in structured outputs, with the ability|Y
meta-llama/llama-3.1-405b|llama-3.1-405b|Meta: Llama 3.1 405B (base)|C|4.0000000000e-06,4.0000000000e-06|32768,32768||-,-,-,-,-,-,-,-,-,32768|Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This is the bas|Y
meta-llama/llama-3.1-8b-instruct|llama-3.1-8b-instruc|Meta: Llama 3.1 8B Instruct|C|2.0000000000e-08,3.0000000000e-08|131072,16384|JST|-,-,-,-,-,-,-,-,-,131072|Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 8B instruc|Y
meta-llama/llama-3.1-405b-instruct:free|llama-3.1-405b-instr|Meta: Llama 3.1 405B Instruct (free)|C|0,0|131072,32768||-,-,-,-,-,-,-,-,-,131072|The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context with impressive eva|Y
meta-llama/llama-3.1-405b-instruct|llama-3.1-405b-instr|Meta: Llama 3.1 405B Instruct|C|3.5000000000e-06,3.5000000000e-06|10000,2500|JST|-,-,-,-,-,-,-,-,-,10000|The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context with impressive eva|Y
meta-llama/llama-3.1-70b-instruct|llama-3.1-70b-instru|Meta: Llama 3.1 70B Instruct|C|4.0000000000e-07,4.0000000000e-07|131072,32768|JT|-,-,-,-,-,-,-,-,-,131072|Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flavors. This 70B instru|Y
mistralai/mistral-nemo|mistral-nemo|Mistral: Mistral Nemo|C|2.0000000000e-08,4.0000000000e-08|131072,16384|JST|-,-,-,-,-,-,-,-,-,131072|A 12B parameter model with a 128k token context length built by Mistral in collaboration with NVIDIA|Y
openai/gpt-4o-mini-2024-07-18|gpt-4o-mini-2024-07-|OpenAI: GPT-4o-mini (2024-07-18)|C|1.5000000000e-07,6.0000000000e-07|128000,16384|JSTV|-,-,-,-,-,-,-,-,-,128000|GPT-4o mini is OpenAI's newest model after [GPT-4 Omni](/models/openai/gpt-4o), supporting both text|Y
openai/gpt-4o-mini|gpt-4o-mini|OpenAI: GPT-4o-mini|C|1.5000000000e-07,6.0000000000e-07|128000,16384|JSTV|-,-,-,-,-,-,-,-,-,128000|GPT-4o mini is OpenAI's newest model after [GPT-4 Omni](/models/openai/gpt-4o), supporting both text|Y
google/gemma-2-27b-it|gemma-2-27b-it|Google: Gemma 2 27B|C|6.5000000000e-07,6.5000000000e-07|8192,2048|JS|-,-,-,-,-,-,-,-,-,8192|Gemma 2 27B by Google is an open model built from the same research and technology used to create th|Y
google/gemma-2-9b-it|gemma-2-9b-it|Google: Gemma 2 9B|C|3.0000000000e-08,9.0000000000e-08|8192,2048||-,-,-,-,-,-,-,-,-,8192|Gemma 2 9B by Google is an advanced, open-source language model that sets a new standard for efficie|Y
sao10k/l3-euryale-70b|l3-euryale-70b|Sao10k: Llama 3 Euryale 70B v2.1|C|1.4800000000e-06,1.4800000000e-06|8192,8192|T|-,-,-,-,-,-,-,-,-,8192|Euryale 70B v2.1 is a model focused on creative roleplay from [Sao10k](https://ko-fi.com/sao10k).

-|Y
nousresearch/hermes-2-pro-llama-3-8b|hermes-2-pro-llama-3|NousResearch: Hermes 2 Pro - Llama-3 8B|C|2.0000000000e-08,8.0000000000e-08|8192,2048|JS|-,-,-,-,-,-,-,-,-,8192|Hermes 2 Pro is an upgraded, retrained version of Nous Hermes 2, consisting of an updated and cleane|Y
mistralai/mistral-7b-instruct:free|mistral-7b-instruct:|Mistral: Mistral 7B Instruct (free)|C|0,0|32768,16384|JT|-,-,-,-,-,-,-,-,-,32768|A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context|Y
mistralai/mistral-7b-instruct|mistral-7b-instruct|Mistral: Mistral 7B Instruct|C|3.0000000000e-08,5.0000000000e-08|32768,16384|JT|-,-,-,-,-,-,-,-,-,32768|A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context|Y
mistralai/mistral-7b-instruct-v0.3|mistral-7b-instruct-|Mistral: Mistral 7B Instruct v0.3|C|2.0000000000e-07,2.0000000000e-07|32768,4096||-,-,-,-,-,-,-,-,-,32768|A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context|Y
microsoft/phi-3-mini-128k-instruct|phi-3-mini-128k-inst|Microsoft: Phi-3 Mini 128K Instruct|C|1.0000000000e-07,1.0000000000e-07|128000,32000|T|-,-,-,-,-,-,-,-,-,128000|Phi-3 Mini is a powerful 3.8B parameter model designed for advanced language understanding, reasonin|Y
microsoft/phi-3-medium-128k-instruct|phi-3-medium-128k-in|Microsoft: Phi-3 Medium 128K Instruct|C|1.0000000000e-06,1.0000000000e-06|128000,32000|T|-,-,-,-,-,-,-,-,-,128000|Phi-3 128K Medium is a powerful 14-billion parameter model designed for advanced language understand|Y
meta-llama/llama-guard-2-8b|llama-guard-2-8b|Meta: LlamaGuard 2 8B|C|2.0000000000e-07,2.0000000000e-07|8192,2048||-,-,-,-,-,-,-,-,-,8192|This safeguard model has 8B parameters and is based on the Llama 3 family. Just like is predecessor,|Y
openai/gpt-4o-2024-05-13|gpt-4o-2024-05-13|OpenAI: GPT-4o (2024-05-13)|C|5.0000000000e-06,1.5e-05|128000,4096|JSTV|-,-,-,-,-,-,-,-,-,128000|GPT-4o (\"o\" for \"omni\") is OpenAI's latest AI model, supporting both text and image inputs with text|Y
openai/gpt-4o|gpt-4o|OpenAI: GPT-4o|C|2.5000000000e-06,1e-05|128000,16384|JSTV|-,-,-,-,-,-,-,-,-,128000|GPT-4o (\"o\" for \"omni\") is OpenAI's latest AI model, supporting both text and image inputs with text|Y
openai/gpt-4o:extended|gpt-4o:extended|OpenAI: GPT-4o (extended)|C|6.0000000000e-06,1.8e-05|128000,64000|JSTV|-,-,-,-,-,-,-,-,-,128000|GPT-4o (\"o\" for \"omni\") is OpenAI's latest AI model, supporting both text and image inputs with text|Y
meta-llama/llama-3-70b-instruct|llama-3-70b-instruct|Meta: Llama 3 70B Instruct|C|3.0000000000e-07,4.0000000000e-07|8192,16384|JST|-,-,-,-,-,-,-,-,-,8192|Meta's latest class of model (Llama 3) launched with a variety of sizes & flavors. This 70B instruct|Y
meta-llama/llama-3-8b-instruct|llama-3-8b-instruct|Meta: Llama 3 8B Instruct|C|3.0000000000e-08,6.0000000000e-08|8192,16384|JT|-,-,-,-,-,-,-,-,-,8192|Meta's latest class of model (Llama 3) launched with a variety of sizes & flavors. This 8B instruct-|Y
mistralai/mixtral-8x22b-instruct|mixtral-8x22b-instru|Mistral: Mixtral 8x22B Instruct|C|2.0000000000e-06,6.0000000000e-06|65536,16384|JST|-,-,-,-,-,-,-,-,-,65536|Mistral's official instruct fine-tuned version of [Mixtral 8x22B](/models/mistralai/mixtral-8x22b).|Y
microsoft/wizardlm-2-8x22b|wizardlm-2-8x22b|WizardLM-2 8x22B|C|4.8000000000e-07,4.8000000000e-07|65536,16384|J|-,-,-,-,-,-,-,-,-,65536|WizardLM-2 8x22B is Microsoft AI's most advanced Wizard model. It demonstrates highly competitive pe|Y
openai/gpt-4-turbo|gpt-4-turbo|OpenAI: GPT-4 Turbo|C|1e-05,3e-05|128000,4096|JSTV|-,-,-,-,-,-,-,-,-,128000|The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and fun|Y
anthropic/claude-3-haiku|claude-3-haiku|Anthropic: Claude 3 Haiku|C|2.5000000000e-07,1.2500000000e-06|200000,4096|TV|-,-,-,-,-,-,-,-,-,200000|Claude 3 Haiku is Anthropic's fastest and most compact model for
near-instant responsiveness. Quick|Y
anthropic/claude-3-opus|claude-3-opus|Anthropic: Claude 3 Opus|C|1.5e-05,7.5e-05|200000,4096|TV|-,-,-,-,-,-,-,-,-,200000|Claude 3 Opus is Anthropic's most powerful model for highly complex tasks. It boasts top-level perfo|Y
mistralai/mistral-large|mistral-large|Mistral Large|C|2.0000000000e-06,6.0000000000e-06|128000,32000|JST|-,-,-,-,-,-,-,-,-,128000|This is Mistral AI's flagship model, Mistral Large 2 (version `mistral-large-2407`). It's a propriet|Y
openai/gpt-3.5-turbo-0613|gpt-3.5-turbo-0613|OpenAI: GPT-3.5 Turbo (older v0613)|C|1.0000000000e-06,2.0000000000e-06|4095,4096|JST|-,-,-,-,-,-,-,-,-,4095|GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural language or code, an|Y
openai/gpt-4-turbo-preview|gpt-4-turbo-preview|OpenAI: GPT-4 Turbo Preview|C|1e-05,3e-05|128000,4096|JST|-,-,-,-,-,-,-,-,-,128000|The preview GPT-4 model with improved instruction following, JSON mode, reproducible outputs, parall|Y
mistralai/mistral-tiny|mistral-tiny|Mistral Tiny|C|2.5000000000e-07,2.5000000000e-07|32768,8192|JST|-,-,-,-,-,-,-,-,-,32768|Note: This model is being deprecated. Recommended replacement is the newer [Ministral 8B](/mistral/m|Y
mistralai/mistral-7b-instruct-v0.2|mistral-7b-instruct-|Mistral: Mistral 7B Instruct v0.2|C|2.0000000000e-07,2.0000000000e-07|32768,8192||-,-,-,-,-,-,-,-,-,32768|A high-performing, industry-standard 7.3B parameter model, with optimizations for speed and context|Y
mistralai/mixtral-8x7b-instruct|mixtral-8x7b-instruc|Mistral: Mixtral 8x7B Instruct|C|5.4000000000e-07,5.4000000000e-07|32768,16384|JT|-,-,-,-,-,-,-,-,-,32768|Mixtral 8x7B Instruct is a pretrained generative Sparse Mixture of Experts, by Mistral AI, for chat|Y
neversleep/noromaid-20b|noromaid-20b|Noromaid 20B|C|1.0000000000e-06,1.7500000000e-06|4096,1024|JS|-,-,-,-,-,-,-,-,-,4096|A collab between IkariDev and Undi. This merge is suitable for RP, ERP, and general knowledge.

#mer|Y
alpindale/goliath-120b|goliath-120b|Goliath 120B|C|6.0000000000e-06,8.0000000000e-06|6144,1024|J|-,-,-,-,-,-,-,-,-,6144|A large LLM created by combining two fine-tuned Llama 70B models into one 120B model. Combines Xwin|Y
openrouter/auto|auto|Auto Router|C|0,0|2000000,500000||-,-,-,-,-,-,-,-,-,2000000|Your prompt will be processed by a meta-model and routed to one of dozens of models (see below), opt|Y
openai/gpt-4-1106-preview|gpt-4-1106-preview|OpenAI: GPT-4 Turbo (older v1106)|C|1e-05,3e-05|128000,4096|JST|-,-,-,-,-,-,-,-,-,128000|The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and fun|Y
openai/gpt-3.5-turbo-instruct|gpt-3.5-turbo-instru|OpenAI: GPT-3.5 Turbo Instruct|C|1.5000000000e-06,2.0000000000e-06|4095,4096|JS|-,-,-,-,-,-,-,-,-,4095|This model is a variant of GPT-3.5 Turbo tuned for instructional prompts and omitting chat-related o|Y
mistralai/mistral-7b-instruct-v0.1|mistral-7b-instruct-|Mistral: Mistral 7B Instruct v0.1|C|1.1000000000e-07,1.9000000000e-07|2824,1024||-,-,-,-,-,-,-,-,-,2824|A 7.3B parameter model that outperforms Llama 2 13B on all benchmarks, with optimizations for speed|Y
openai/gpt-3.5-turbo-16k|gpt-3.5-turbo-16k|OpenAI: GPT-3.5 Turbo 16k|C|3.0000000000e-06,4.0000000000e-06|16385,4096|JST|-,-,-,-,-,-,-,-,-,16385|This model offers four times the context length of gpt-3.5-turbo, allowing it to support approximate|Y
mancer/weaver|weaver|Mancer: Weaver (alpha)|C|7.5000000000e-07,1.0000000000e-06|8000,2000|J|-,-,-,-,-,-,-,-,-,8000|An attempt to recreate Claude-style verbosity, but don't expect the same level of coherence or memor|Y
undi95/remm-slerp-l2-13b|remm-slerp-l2-13b|ReMM SLERP 13B|C|4.5000000000e-07,6.5000000000e-07|6144,1536|JS|-,-,-,-,-,-,-,-,-,6144|A recreation trial of the original MythoMax-L2-B13 but with updated models. #merge|Y
gryphe/mythomax-l2-13b|mythomax-l2-13b|MythoMax 13B|C|6.0000000000e-08,6.0000000000e-08|4096,1024|JS|-,-,-,-,-,-,-,-,-,4096|One of the highest performing and most popular fine-tunes of Llama 2 13B, with rich descriptions and|Y
openai/gpt-4-0314|gpt-4-0314|OpenAI: GPT-4 (older v0314)|C|3e-05,6e-05|8191,4096|JST|-,-,-,-,-,-,-,-,-,8191|GPT-4-0314 is the first version of GPT-4 released, with a context length of 8,192 tokens, and was su|Y
openai/gpt-4|gpt-4|OpenAI: GPT-4|C|3e-05,6e-05|8191,4096|JST|-,-,-,-,-,-,-,-,-,8191|OpenAI's flagship model, GPT-4 is a large-scale multimodal language model capable of solving difficu|Y
openai/gpt-3.5-turbo|gpt-3.5-turbo|OpenAI: GPT-3.5 Turbo|C|5.0000000000e-07,1.5000000000e-06|16385,4096|JST|-,-,-,-,-,-,-,-,-,16385|GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural language or code, an|Y

# =============================================================================
# AWS BEDROCK (Enterprise provider with 48+ models)
# =============================================================================
aws/bedrock-claude-3-5-sonnet-20241022-v1:0|claude-sonnet-3-5|Claude 3.5 Sonnet|C|6.0000000000e-06,3e-05|200000,4096|SVTJK|-,-,-,-,-,-,-,-,-,200000|Advanced reasoning and tool use|Y
aws/bedrock-claude-3-5-sonnet-v2-20250127-v1:0|claude-sonnet-3-5-v2|Claude 3.5 Sonnet v2|C|6.0000000000e-06,3e-05|200000,4096|SVTJK|-,-,-,-,-,-,-,-,-,200000|Latest Sonnet with improved performance|Y
aws/bedrock-claude-3-7-sonnet-20250219-v1:0|claude-sonnet-3-7|Claude 3.7 Sonnet|C|6.0000000000e-06,3e-05|200000,4096|SVTJK|-,-,-,-,-,-,-,-,-,200000|New Sonnet generation|Y
aws/bedrock-claude-opus-4-1-20250805-v1:0|claude-opus-4-1|Claude Opus 4.1|C|7.5000000000e-06,3e-05|200000,4096|SVTJK|-,-,-,-,-,-,-,-,-,200000|Powerful flagship model|Y
aws/bedrock-claude-opus-4-20250514-v1:0|claude-opus-4|Claude Opus 4|C|7.5000000000e-06,3e-05|200000,4096|SVTJK|-,-,-,-,-,-,-,-,-,200000|Strong reasoning capability|Y
aws/bedrock-claude-haiku-4-5-20250811-v1:0|claude-haiku-4-5|Claude Haiku 4.5|C|8.0000000000e-07,4.0000000000e-06|200000,4096|SVT|-,-,-,-,-,-,-,-,-,200000|Fast and efficient model|Y
aws/bedrock-claude-3-5-haiku-20241226-v1:0|claude-haiku-3-5|Claude 3.5 Haiku|C|8.0000000000e-07,4.0000000000e-06|200000,4096|SVT|-,-,-,-,-,-,-,-,-,200000|Latest Haiku generation|Y
aws/bedrock-claude-3-haiku-20240307-v1:0|claude-haiku-3|Claude 3 Haiku|C|2.5000000000e-07,1.2500000000e-06|200000,4096|SV|-,-,-,-,-,-,-,-,-,200000|Lightweight model|Y
aws/bedrock-claude-3-opus-20240229-v1:0|claude-opus-3|Claude 3 Opus|L|1.5e-05,7.5e-05|200000,4096|SVT|-,-,-,-,-,-,-,-,-,200000|Legacy Claude 3 Opus|Y
aws/bedrock-claude-3-sonnet-20240229-v1:0|claude-sonnet-3|Claude 3 Sonnet|L|3.0000000000e-06,1.5e-05|200000,4096|SVT|-,-,-,-,-,-,-,-,-,200000|Legacy Claude 3 Sonnet|Y
aws/bedrock-nova-pro-v1:0|nova-pro|Amazon Nova Pro|C|8.0000000000e-07,3.0000000000e-06|300000,40000|SVT|-,-,-,-,-,-,-,-,-,300000|Advanced multimodal model|Y
aws/bedrock-nova-lite-v1:0|nova-lite|Amazon Nova Lite|C|6.0000000000e-08,2.4000000000e-07|300000,40000|SVT|-,-,-,-,-,-,-,-,-,300000|Fast and affordable model|Y
aws/bedrock-nova-micro-v1:0|nova-micro|Amazon Nova Micro|C|3.0000000000e-08,1.2000000000e-07|128000,1024|T|-,-,-,-,-,-,-,-,-,128000|Ultra-lightweight model|Y
aws/bedrock-nova-premier-v1:0|nova-premier|Amazon Nova Premier|C|8.0000000000e-07,3.0000000000e-06|300000,40000|SVT|-,-,-,-,-,-,-,-,-,300000|Premium multimodal model|Y
aws/bedrock-llama-3-1-405b-instruct-v1:0|llama-3-1-405b|Llama 3.1 405B Instruct|C|5.3300000000e-06,1.6e-05|128000,8000|T|-,-,-,-,-,-,-,-,-,128000|Large language instruction model|Y
aws/bedrock-llama-3-1-70b-instruct-v1:0|llama-3-1-70b|Llama 3.1 70B Instruct|C|1.3500000000e-06,2.7000000000e-06|128000,8000|T|-,-,-,-,-,-,-,-,-,128000|Medium instruction model|Y
aws/bedrock-llama-3-3-70b-instruct-v1:0|llama-3-3-70b|Llama 3.3 70B Instruct|C|1.3500000000e-06,2.7000000000e-06|128000,8000|T|-,-,-,-,-,-,-,-,-,128000|Latest Llama 3.3 model|Y
aws/bedrock-llama-4-maverick-17b-v1:0|llama-4-maverick|Llama 4 Maverick 17B|C|4.0000000000e-07,8.0000000000e-07|128000,8000|SVT|-,-,-,-,-,-,-,-,-,128000|Llama 4 early variant|Y
aws/bedrock-llama-4-scout-17b-v1:0|llama-4-scout|Llama 4.0 Scout 17B|C|4.0000000000e-07,8.0000000000e-07|128000,8000|SVT|-,-,-,-,-,-,-,-,-,128000|Llama 4 scout model|Y
aws/bedrock-llama-3-2-90b-vision-instruct-v1:0|llama-3-2-90b-vision|Llama 3.2 90B Vision Instruct|C|1.3500000000e-06,2.7000000000e-06|128000,8000|SV|-,-,-,-,-,-,-,-,-,128000|Multimodal Llama model|Y
aws/bedrock-llama-3-2-11b-vision-instruct-v1:0|llama-3-2-11b-vision|Llama 3.2 11B Vision Instruct|C|3.0000000000e-07,6.0000000000e-07|128000,8000|SV|-,-,-,-,-,-,-,-,-,128000|Small Llama vision model|Y
aws/bedrock-llama-3-2-1b-instruct-v1:0|llama-3-2-1b|Llama 3.2 1B Instruct|C|1.0000000000e-08,2.0000000000e-08|8000,2048||-,-,-,-,-,-,-,-,-,8000|Tiny Llama model|Y
aws/bedrock-llama-3-2-3b-instruct-v1:0|llama-3-2-3b|Llama 3.2 3B Instruct|C|5.0000000000e-08,1.0000000000e-07|8000,2048||-,-,-,-,-,-,-,-,-,8000|Small Llama model|Y
aws/bedrock-llama-3-1-8b-instruct-v1:0|llama-3-1-8b|Llama 3.1 8B Instruct|C|1.5000000000e-07,3.0000000000e-06|128000,8000||-,-,-,-,-,-,-,-,-,128000|Small Llama instruction model|Y
aws/bedrock-llama-3-70b-instruct-v1:0|llama-3-70b|Llama 3 70B Instruct|L|1.3500000000e-06,2.7000000000e-06|8000,2048||-,-,-,-,-,-,-,-,-,8000|Legacy Llama 3 model|Y
aws/bedrock-llama-2-70b-chat-v1:0|llama-2-70b|Llama 2 70B Chat|L|1.3500000000e-06,1.8000000000e-06|4096,2048||-,-,-,-,-,-,-,-,-,4096|Legacy Llama 2 model|Y
aws/bedrock-mistral-large-2-415b-instruct-v1:0|mistral-large-2-415b|Mistral Large 2 415B Instruct|C|8.0000000000e-07,2.4000000000e-06|200000,64000|TJ|-,-,-,-,-,-,-,-,-,200000|Mistral flagship model|Y
aws/bedrock-mistral-large-instruct-2412-v1:0|mistral-large-2412|Mistral Large Instruct 2412|C|8.0000000000e-07,2.4000000000e-06|200000,64000|TJ|-,-,-,-,-,-,-,-,-,200000|Latest Mistral Large|Y
aws/bedrock-mistral-large-2407-v1:0|mistral-large-2407|Mistral Large 2 (24.07)|C|8.0000000000e-07,2.4000000000e-06|200000,64000|TJ|-,-,-,-,-,-,-,-,-,200000|Mistral 24.07 version|Y
aws/bedrock-pixtral-large-2502-v1:0|pixtral-large|Pixtral Large (25.02)|C|8.0000000000e-07,2.4000000000e-06|128000,4096|SV|-,-,-,-,-,-,-,-,-,128000|Mistral multimodal model|Y
aws/bedrock-mistral-small-2409-v1:0|mistral-small-2409|Mistral Small 2409|C|1.4000000000e-07,4.2000000000e-07|32000,8000|T|-,-,-,-,-,-,-,-,-,32000|Small Mistral model|Y
aws/bedrock-mistral-nemo-2407-v1:0|mistral-nemo|Mistral Nemo 2407|C|1.4000000000e-07,4.2000000000e-07|32000,8000||-,-,-,-,-,-,-,-,-,32000|Efficient Mistral model|Y
aws/bedrock-cohere-command-r-7b-12-2024-v1:0|cohere-command-r-7b|Cohere Command R 7B|C|1.0000000000e-07,2.0000000000e-07|128000,4096|TJ|-,-,-,-,-,-,-,-,-,128000|Cohere lightweight model|Y
aws/bedrock-cohere-command-r-5-2024-10-08-v1:0|cohere-command-r-5|Cohere Command R 5B|C|1.0000000000e-07,2.0000000000e-07|128000,4096|TJ|-,-,-,-,-,-,-,-,-,128000|Cohere small model|Y
aws/bedrock-cohere-command-r-plus-04-2024-v1:0|cohere-command-r-plus|Cohere Command R+ 04-2024|C|3.0000000000e-07,6.0000000000e-06|128000,4096|TJ|-,-,-,-,-,-,-,-,-,128000|Cohere flagship model|Y
aws/bedrock-cohere-command-text-v14-7k-instruct-4k-latest-v1:0|cohere-command-text|Cohere Command Text|L|1.5000000000e-06,2.0000000000e-06|4096,4096||-,-,-,-,-,-,-,-,-,4096|Legacy Cohere model|Y
aws/bedrock-ai21-jamba-instruct-v1:0|ai21-jamba-instruct|AI21 Jamba Instruct|C|3.0000000000e-07,4.0000000000e-06|256000,4096|T|-,-,-,-,-,-,-,-,-,256000|AI21 instruction model|Y
aws/bedrock-ai21-jamba-1-5-large-v1:0|ai21-jamba-1-5-large|AI21 Jamba 1.5 Large|C|3.0000000000e-07,4.0000000000e-06|256000,4096|T|-,-,-,-,-,-,-,-,-,256000|Large Jamba model|Y
aws/bedrock-ai21-jamba-1-5-mini-v1:0|ai21-jamba-1-5-mini|AI21 Jamba 1.5 Mini|C|3.0000000000e-07,4.0000000000e-06|256000,4096|T|-,-,-,-,-,-,-,-,-,256000|Mini Jamba model|Y
aws/bedrock-ai21-labs-jurassic-2-ultra-v1:0|ai21-jurassic-2-ultra|AI21 Jurassic 2 Ultra|L|1.8800000000e-06,2.5000000000e-06|8191,1024||-,-,-,-,-,-,-,-,-,8191|Legacy AI21 model|Y
aws/bedrock-amazon-titan-text-premier-v1:0|titan-text-premier|Amazon Titan Text Premier|C|5.0000000000e-06,1.5e-05|32000,4096||-,-,-,-,-,-,-,-,-,32000|Titan large model|Y
aws/bedrock-amazon-titan-text-express-v1:0|titan-text-express|Amazon Titan Text Express|C|3.0000000000e-07,4.0000000000e-06|8000,2048||-,-,-,-,-,-,-,-,-,8000|Titan efficient model|Y
aws/bedrock-amazon-titan-text-lite-v1:0|titan-text-lite|Amazon Titan Text Lite|L|3.0000000000e-07,4.0000000000e-06|4000,2048||-,-,-,-,-,-,-,-,-,4000|Legacy Titan model|Y
aws/bedrock-deepseek-r1-v1:0|deepseek-r1|DeepSeek R1|C|1.4000000000e-06,5.6000000000e-06|128000,16000|K|-,-,-,-,-,-,-,-,-,128000|DeepSeek reasoning model|Y
aws/bedrock-writer-palmyra-x5-v1:0|writer-palmyra-x5|Writer Palmyra X5|C|1.5000000000e-06,2.0000000000e-06|32000,2048|T|-,-,-,-,-,-,-,-,-,32000|Writer specialized model|Y
aws/bedrock-writer-palmyra-x4-v1:0|writer-palmyra-x4|Writer Palmyra X4|L|1.5000000000e-06,2.0000000000e-06|32000,2048|T|-,-,-,-,-,-,-,-,-,32000|Legacy Writer model|Y
aws/bedrock-google-gemma-7b-it-v1:0|google-gemma-7b|Google Gemma 7B IT|C|7.0000000000e-07,1.4000000000e-06|8000,2000||-,-,-,-,-,-,-,-,-,8000|Google lightweight model|Y
aws/bedrock-google-gemma-2b-it-v1:0|google-gemma-2b|Google Gemma 2B IT|C|3.5000000000e-07,7.0000000000e-07|8000,2000||-,-,-,-,-,-,-,-,-,8000|Google tiny model|Y

# =============================================================================
# LATEST RELEASES (Frontier models from January 2026)
# =============================================================================
anthropic/claude-opus-4-5-20251101|claude-opus-4-5|Claude Opus 4.5|C|1.5e-05,7.5e-05|200000,4096|SVTJK|92,-,90,-,-,-,-,-,-,200000|Frontier model with extended thinking capability|Y
anthropic/claude-sonnet-4-5-20250924|claude-sonnet-4-5|Claude Sonnet 4.5|C|3.0000000000e-06,1.5e-05|200000,4096|SVTJK|90,-,87,-,-,-,-,-,-,200000|Advanced reasoning with improved speed|Y
google/gemini-3-pro-20260101|gemini-3-pro|Gemini 3 Pro|C|3.0000000000e-06,9.0000000000e-06|1000000,8192|SVTJC|93,-,91,-,-,-,-,-,-,1000000|Latest Google frontier model|Y
google/gemini-3-flash-20260101|gemini-3-flash|Gemini 3 Flash|C|1.0000000000e-06,3.0000000000e-06|1000000,8192|SVTJC|91,-,89,-,-,-,-,-,-,1000000|Fast generation model from Google|Y
meta/llama-4-405b-20260115|llama-4-405b|Llama 4 405B|C|5.3300000000e-06,1.6e-05|128000,8192|ST|92,-,88,-,-,-,-,-,-,128000|Latest Meta flagship reasoning model|Y
meta/llama-4-70b-20260115|llama-4-70b|Llama 4 70B|C|1.3500000000e-06,2.7000000000e-06|128000,8192|ST|91,-,86,-,-,-,-,-,-,128000|Mid-range Llama 4 model|Y
amazon/nova-premier-latest|amazon-nova-premier|Amazon Nova Premier|C|8.0000000000e-07,3.0000000000e-06|300000,40000|SVT|88,-,84,-,-,-,-,-,-,300000|Premium multimodal reasoning model|Y
deepseek/deepseek-v3-2-20260104|deepseek-v3-2|DeepSeek V3.2|C|2.7000000000e-06,8.1000000000e-06|64000,4096|SVTK|91,-,89,-,-,-,-,-,-,64000|Advanced reasoning with o1-style thinking|Y
mistral/mistral-large-3-20260101|mistral-large-3|Mistral Large 3|C|8.0000000000e-07,2.4000000000e-06|200000,64000|TJ|89,-,85,-,-,-,-,-,-,200000|Latest flagship from Mistral|Y
cohere/command-r7-plus-20260110|cohere-command-r7-plus|Cohere Command R7 Plus|C|3.0000000000e-07,6.0000000000e-06|128000,4096|TJ|88,-,83,-,-,-,-,-,-,128000|Latest Cohere advanced model|Y
"#;

// ============================================================================
// PARSING
// ============================================================================

fn parse_status(s: &str) -> ModelStatus {
    match s.to_uppercase().as_str() {
        "C" => ModelStatus::Current,
        "L" => ModelStatus::Legacy,
        "D" => ModelStatus::Deprecated,
        _ => ModelStatus::Current,
    }
}

fn parse_model_line(line: &str) -> Option<ModelInfo> {
    let parts: Vec<&str> = line.split('|').collect();
    if parts.len() < 10 {
        return None;
    }

    let id = parts[0].to_string();
    let alias = if parts[1] == "-" {
        None
    } else {
        Some(parts[1].to_string())
    };
    let name = parts[2].to_string();
    let status = parse_status(parts[3]);

    // Parse pricing: in,out[,cache]
    let pricing_parts: Vec<&str> = parts[4].split(',').collect();
    let input_cost = pricing_parts
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let output_cost = pricing_parts
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5.0);
    let cache_cost = pricing_parts.get(2).and_then(|s| s.parse().ok());
    let mut pricing = ModelPricing::new(input_cost, output_cost);
    if let Some(c) = cache_cost {
        pricing = pricing.with_cache(c);
    }

    // Parse context: max_ctx,max_out
    let ctx_parts: Vec<&str> = parts[5].split(',').collect();
    let max_context = ctx_parts
        .first()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128000);
    let max_output = ctx_parts
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192);

    // Parse capabilities flags
    let capabilities = ModelCapabilities::from_flags(max_context, max_output, parts[6]);

    // Parse benchmarks: mmlu,human,math,gpqa,swe,if,mmmu,mgsm,ttft,tps
    let bench_parts: Vec<&str> = parts[7].split(',').collect();
    let benchmarks = ModelBenchmarks {
        mmlu: bench_parts.first().and_then(|s| s.parse().ok()),
        humaneval: bench_parts.get(1).and_then(|s| s.parse().ok()),
        math: bench_parts.get(2).and_then(|s| s.parse().ok()),
        gpqa: bench_parts.get(3).and_then(|s| s.parse().ok()),
        swe_bench: bench_parts.get(4).and_then(|s| s.parse().ok()),
        ifeval: bench_parts.get(5).and_then(|s| s.parse().ok()),
        mmmu: bench_parts.get(6).and_then(|s| s.parse().ok()),
        mgsm: bench_parts.get(7).and_then(|s| s.parse().ok()),
        ttft_ms: bench_parts.get(8).and_then(|s| s.parse().ok()),
        tokens_per_sec: bench_parts.get(9).and_then(|s| s.parse().ok()),
    };

    let description = parts[8].to_string();
    let can_classify = parts[9].trim().to_uppercase() == "Y";

    // Extract provider from ID
    let provider = if let Some((prefix, _)) = id.split_once('/') {
        Provider::from_prefix(prefix)
    } else {
        Provider::Custom
    };

    Some(ModelInfo {
        id,
        alias,
        name,
        provider,
        status,
        pricing,
        capabilities,
        benchmarks,
        description,
        can_classify,
    })
}

fn parse_model_data() -> Vec<ModelInfo> {
    MODEL_DATA
        .lines()
        .filter(|line| !line.trim().is_empty() && !line.trim().starts_with('#'))
        .filter_map(|line| parse_model_line(line.trim()))
        .collect()
}

// ============================================================================
// REGISTRY
// ============================================================================

struct ModelRegistry {
    models: Vec<ModelInfo>,
    id_index: HashMap<String, usize>,
}

impl ModelRegistry {
    fn new() -> Self {
        let models = parse_model_data();
        let mut id_index = HashMap::new();

        for (i, model) in models.iter().enumerate() {
            id_index.insert(model.id.clone(), i);
            id_index.insert(model.raw_id().to_string(), i);
            if let Some(ref alias) = model.alias {
                id_index.insert(alias.clone(), i);
            }
        }

        Self { models, id_index }
    }

    fn get(&self, model_id: &str) -> Option<&ModelInfo> {
        self.id_index.get(model_id).map(|&i| &self.models[i])
    }
}

static REGISTRY: LazyLock<ModelRegistry> = LazyLock::new(ModelRegistry::new);

// ============================================================================
// PUBLIC API
// ============================================================================

/// Get model info by ID, alias, or raw ID.
pub fn get_model_info(model_id: &str) -> Option<&'static ModelInfo> {
    REGISTRY.get(model_id)
}

/// Check if a model supports structured output (JSON schema enforcement).
///
/// Returns true if the model supports full structured output with JSON schema validation.
/// This allows the caller to transparently use structured output when available,
/// falling back to text parsing when not.
///
/// # Example
/// ```ignore
/// use llmkit::models::supports_structured_output;
///
/// if supports_structured_output("gpt-4o") {
///     // Use structured output with JSON schema
///     request = request.with_json_schema("result", schema);
/// } else {
///     // Fall back to text-based parsing
/// }
/// ```
pub fn supports_structured_output(model_id: &str) -> bool {
    get_model_info(model_id)
        .map(|m| m.capabilities.structured_output)
        .unwrap_or(false)
}

/// Get all models for a provider.
pub fn get_models_by_provider(provider: Provider) -> Vec<&'static ModelInfo> {
    REGISTRY
        .models
        .iter()
        .filter(|m| m.provider == provider)
        .collect()
}

/// Get all models.
pub fn get_all_models() -> &'static [ModelInfo] {
    &REGISTRY.models
}

/// Get all current (non-deprecated) models.
pub fn get_current_models() -> Vec<&'static ModelInfo> {
    REGISTRY
        .models
        .iter()
        .filter(|m| m.status == ModelStatus::Current)
        .collect()
}

/// Get models that can be used as classifiers.
pub fn get_classifier_models() -> Vec<&'static ModelInfo> {
    REGISTRY.models.iter().filter(|m| m.can_classify).collect()
}

/// Get available models (provider API key is set).
pub fn get_available_models() -> Vec<&'static ModelInfo> {
    REGISTRY
        .models
        .iter()
        .filter(|m| m.provider.is_available())
        .collect()
}

/// Get models with a specific capability.
pub fn get_models_with_capability(
    vision: Option<bool>,
    tools: Option<bool>,
    thinking: Option<bool>,
) -> Vec<&'static ModelInfo> {
    REGISTRY
        .models
        .iter()
        .filter(|m| {
            vision.is_none_or(|v| m.capabilities.vision == v)
                && tools.is_none_or(|t| m.capabilities.tools == t)
                && thinking.is_none_or(|k| m.capabilities.thinking == k)
        })
        .collect()
}

/// Get the cheapest model that meets requirements.
pub fn get_cheapest_model(
    min_context: Option<u32>,
    needs_vision: bool,
    needs_tools: bool,
) -> Option<&'static ModelInfo> {
    let mut candidates: Vec<_> = REGISTRY
        .models
        .iter()
        .filter(|m| {
            m.status == ModelStatus::Current
                && m.provider.is_available()
                && min_context.is_none_or(|c| m.capabilities.max_context >= c)
                && (!needs_vision || m.capabilities.vision)
                && (!needs_tools || m.capabilities.tools)
        })
        .collect();

    candidates.sort_by(|a, b| {
        let cost_a = a.pricing.input_per_1m + a.pricing.output_per_1m;
        let cost_b = b.pricing.input_per_1m + b.pricing.output_per_1m;
        cost_a
            .partial_cmp(&cost_b)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    candidates.first().copied()
}

/// List all providers with at least one model.
pub fn list_providers() -> Vec<Provider> {
    let mut providers: Vec<_> = REGISTRY
        .models
        .iter()
        .map(|m| m.provider)
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    providers.sort_by_key(|p| p.prefix());
    providers
}

/// Get registry statistics.
pub struct RegistryStats {
    pub total_models: usize,
    pub current_models: usize,
    pub providers: usize,
    pub available_models: usize,
}

pub fn get_registry_stats() -> RegistryStats {
    RegistryStats {
        total_models: REGISTRY.models.len(),
        current_models: REGISTRY
            .models
            .iter()
            .filter(|m| m.status == ModelStatus::Current)
            .count(),
        providers: list_providers().len(),
        available_models: get_available_models().len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_detection() {
        assert_eq!(Provider::from_prefix("anthropic"), Provider::Anthropic);
        assert_eq!(Provider::from_prefix("openai"), Provider::OpenAI);
        assert_eq!(Provider::from_prefix("groq"), Provider::Groq);
    }

    #[test]
    fn test_provider_from_model() {
        assert_eq!(
            Provider::from_model("claude-sonnet-4-20250514"),
            Provider::Anthropic
        );
        assert_eq!(Provider::from_model("gpt-4o"), Provider::OpenAI);
        assert_eq!(Provider::from_model("gemini-1.5-pro"), Provider::Google);
    }

    #[test]
    fn test_get_model_info() {
        let model = get_model_info("claude-sonnet-4-5").unwrap();
        assert_eq!(model.provider, Provider::Anthropic);
        assert!(model.capabilities.vision);
    }

    #[test]
    fn test_pricing_calculation() {
        let pricing = ModelPricing::new(3.0, 15.0);
        let cost = pricing.estimate_cost(1000, 500);
        assert!((cost - 0.0105).abs() < 0.0001);
    }

    #[test]
    fn test_benchmarks_quality_score() {
        let benchmarks = ModelBenchmarks {
            mmlu: Some(90.0),
            humaneval: Some(95.0),
            math: Some(85.0),
            ..Default::default()
        };
        let score = benchmarks.quality_score();
        assert!(score > 85.0 && score < 95.0);
    }

    #[test]
    fn test_registry_has_models() {
        let stats = get_registry_stats();
        assert!(stats.total_models > 50); // Including regional providers
        assert!(stats.providers >= 17); // Including regional providers
    }

    #[test]
    fn test_regional_providers() {
        // Test Writer (Palmyra)
        let palmyra = get_model_info("palmyra-x5").unwrap();
        assert_eq!(palmyra.provider, Provider::Writer);
        assert_eq!(palmyra.capabilities.max_context, 1_000_000);

        // Test Maritaca (Sabiá)
        let sabia = get_model_info("sabia-3").unwrap();
        assert_eq!(sabia.provider, Provider::Maritaca);

        // Test Clova (HyperCLOVA X)
        let hcx = get_model_info("hcx-005").unwrap();
        assert_eq!(hcx.provider, Provider::Clova);
        assert!(hcx.capabilities.vision);

        // Test Yandex
        let yandex = get_model_info("yandexgpt-pro").unwrap();
        assert_eq!(yandex.provider, Provider::Yandex);

        // Test GigaChat
        let gigachat = get_model_info("gigachat-pro").unwrap();
        assert_eq!(gigachat.provider, Provider::GigaChat);

        // Test Upstage (Solar)
        let solar = get_model_info("solar-pro").unwrap();
        assert_eq!(solar.provider, Provider::Upstage);

        // Test SEA-LION
        let sealion = get_model_info("sea-lion-32b").unwrap();
        assert_eq!(sealion.provider, Provider::SeaLion);
        assert!(sealion.capabilities.vision);
    }

    #[test]
    fn test_regional_provider_detection() {
        assert_eq!(Provider::from_model("palmyra-x5"), Provider::Writer);
        assert_eq!(Provider::from_model("sabia-3"), Provider::Maritaca);
        assert_eq!(Provider::from_model("HCX-005"), Provider::Clova);
        assert_eq!(Provider::from_model("yandexgpt-pro"), Provider::Yandex);
        assert_eq!(Provider::from_model("gigachat-pro"), Provider::GigaChat);
        assert_eq!(Provider::from_model("solar-pro"), Provider::Upstage);
    }

    #[test]
    fn test_regional_provider_prefix() {
        assert_eq!(Provider::from_prefix("writer"), Provider::Writer);
        assert_eq!(Provider::from_prefix("maritaca"), Provider::Maritaca);
        assert_eq!(Provider::from_prefix("clova"), Provider::Clova);
        assert_eq!(Provider::from_prefix("yandex"), Provider::Yandex);
        assert_eq!(Provider::from_prefix("gigachat"), Provider::GigaChat);
        assert_eq!(Provider::from_prefix("upstage"), Provider::Upstage);
        assert_eq!(Provider::from_prefix("sea-lion"), Provider::SeaLion);
    }
}
