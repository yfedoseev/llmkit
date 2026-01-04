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
together_ai/deepseek-ai/DeepSeek-R1|deepseek-r1-together|DeepSeek R1 (Together)|C|0.55,2.19|64000,8192|JSKC|90.8,91.0,90.0,71.5,49.2,88.4,-,-,3000,40|Reasoning via Together AI|N

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
openrouter/anthropic/claude-haiku-4.5|openrouter-claude-haiku-4.5|Claude Haiku 4.5 (OpenRouter)|C|1.0,5.0,0.1|200000,64000|VTJSKC|85.7,88.4,71.2,51.8,35.6,85.2,60.5,-,300,200|Via OpenRouter|Y

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
