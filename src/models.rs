//! Model Registry - Database of LLM model specifications.
//!
//! This module provides a comprehensive registry of LLM models across all supported providers,
//! including pricing information, context window sizes, capabilities, and benchmark scores.
//!
//! The data uses unified model IDs (e.g., `anthropic/claude-3-5-sonnet`).
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::models::{get_model_info, get_models_by_provider, Provider};
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
    // Regional providers
    Writer,
    Maritaca,
    Clova,
    Yandex,
    GigaChat,
    Upstage,
    SeaLion,
    // Additional providers (already implemented in providers/chat/)
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
    // New providers (implemented)
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
            // Regional providers
            Provider::Writer => Some("WRITER_API_KEY"),
            Provider::Maritaca => Some("MARITALK_API_KEY"),
            Provider::Clova => Some("CLOVASTUDIO_API_KEY"),
            Provider::Yandex => Some("YANDEX_API_KEY"),
            Provider::GigaChat => Some("GIGACHAT_API_KEY"),
            Provider::Upstage => Some("UPSTAGE_API_KEY"),
            Provider::SeaLion => Some("SEA_LION_API_KEY"),
            // Additional providers
            Provider::Alibaba => Some("DASHSCOPE_API_KEY"),
            Provider::AlephAlpha => Some("ALEPH_ALPHA_API_KEY"),
            Provider::Baidu => Some("BAIDU_API_KEY"),
            Provider::Baseten => Some("BASETEN_API_KEY"),
            Provider::ChatLaw => Some("CHATLAW_API_KEY"),
            Provider::DataRobot => Some("DATAROBOT_API_KEY"),
            Provider::LatamGPT => Some("LATAMGPT_API_KEY"),
            Provider::LightOn => Some("LIGHTON_API_KEY"),
            Provider::NLPCloud => Some("NLP_CLOUD_API_KEY"),
            Provider::Oracle => Some("OCI_API_KEY"),
            Provider::Perplexity => Some("PERPLEXITY_API_KEY"),
            Provider::RunPod => Some("RUNPOD_API_KEY"),
            Provider::Sagemaker => Some("AWS_ACCESS_KEY_ID"),
            Provider::Sap => Some("SAP_API_KEY"),
            Provider::Snowflake => Some("SNOWFLAKE_API_KEY"),
            Provider::WatsonX => Some("WATSONX_API_KEY"),
            // New providers
            Provider::Xai => Some("XAI_API_KEY"),
            Provider::DeepInfra => Some("DEEPINFRA_API_KEY"),
            Provider::NvidiaNIM => Some("NVIDIA_API_KEY"),
            // Tier 1 - High Priority Inference
            Provider::Ollama => None, // Local, no API key needed
            Provider::Anyscale => Some("ANYSCALE_API_KEY"),
            Provider::GitHub => Some("GITHUB_TOKEN"),
            Provider::FriendliAI => Some("FRIENDLI_TOKEN"),
            Provider::Hyperbolic => Some("HYPERBOLIC_API_KEY"),
            Provider::Lambda => Some("LAMBDA_API_KEY"),
            Provider::Novita => Some("NOVITA_API_KEY"),
            Provider::Nebius => Some("NEBIUS_API_KEY"),
            Provider::Lepton => Some("LEPTON_API_KEY"),
            Provider::Stability => Some("STABILITY_API_KEY"),
            Provider::Voyage => Some("VOYAGE_API_KEY"),
            Provider::Jina => Some("JINA_API_KEY"),
            Provider::Deepgram => Some("DEEPGRAM_API_KEY"),
            Provider::ElevenLabs => Some("ELEVENLABS_API_KEY"),
            Provider::GPT4All => None, // Local, no API key needed
            // Tier 2 - Chinese Providers
            Provider::MiniMax => Some("MINIMAX_API_KEY"),
            Provider::Moonshot => Some("MOONSHOT_API_KEY"),
            Provider::Zhipu => Some("ZHIPU_API_KEY"),
            Provider::Volcengine => Some("VOLC_ACCESSKEY"),
            Provider::Baichuan => Some("BAICHUAN_API_KEY"),
            Provider::Stepfun => Some("STEPFUN_API_KEY"),
            Provider::Yi => Some("YI_API_KEY"),
            Provider::Spark => Some("SPARK_API_KEY"),
            // Tier 3 - Local/Self-Hosted (no API keys needed)
            Provider::LMStudio
            | Provider::Llamafile
            | Provider::Xinference
            | Provider::LocalAI
            | Provider::Jan
            | Provider::Petals
            | Provider::Triton
            | Provider::Tgi => None,
            // Tier 4 - Enterprise/Specialized
            Provider::Predibase => Some("PREDIBASE_API_KEY"),
            Provider::OctoAI => Some("OCTOAI_API_KEY"),
            Provider::Featherless => Some("FEATHERLESS_API_KEY"),
            Provider::OVHCloud => Some("OVH_API_KEY"),
            Provider::Scaleway => Some("SCALEWAY_API_KEY"),
            Provider::Crusoe => Some("CRUSOE_API_KEY"),
            Provider::Cerebrium => Some("CEREBRIUM_API_KEY"),
            Provider::Lightning => Some("LIGHTNING_API_KEY"),
            Provider::AssemblyAI => Some("ASSEMBLYAI_API_KEY"),
            Provider::RunwayML => Some("RUNWAYML_API_KEY"),
            // Tier 5 - Asian Regional Providers
            Provider::Naver => Some("NAVER_API_KEY"),
            Provider::Kakao => Some("KAKAO_API_KEY"),
            Provider::LGExaone => Some("LG_EXAONE_API_KEY"),
            Provider::PLaMo => Some("PLAMO_API_KEY"),
            Provider::Sarvam => Some("SARVAM_API_KEY"),
            Provider::Krutrim => Some("KRUTRIM_API_KEY"),
            Provider::Ntt => Some("NTT_API_KEY"),
            Provider::SoftBank => Some("SOFTBANK_API_KEY"),
            // Tier 6 - European Sovereign AI
            Provider::Ionos => Some("IONOS_API_KEY"),
            Provider::Tilde => Some("TILDE_API_KEY"),
            Provider::SiloAI => Some("SILOAI_API_KEY"),
            Provider::SwissAI => Some("SWISSAI_API_KEY"),
            // Tier 7 - Router/Gateway/Meta Providers
            Provider::Unify => Some("UNIFY_API_KEY"),
            Provider::Martian => Some("MARTIAN_API_KEY"),
            Provider::Portkey => Some("PORTKEY_API_KEY"),
            Provider::Helicone => Some("HELICONE_API_KEY"),
            Provider::SiliconFlow => Some("SILICONFLOW_API_KEY"),
            // Tier 8 - Video AI Providers
            Provider::Pika => Some("PIKA_API_KEY"),
            Provider::Luma => Some("LUMA_API_KEY"),
            Provider::Kling => Some("KLING_API_KEY"),
            Provider::HeyGen => Some("HEYGEN_API_KEY"),
            Provider::Did => Some("DID_API_KEY"),
            Provider::TwelveLabs => Some("TWELVE_LABS_API_KEY"),
            // Tier 9 - Audio AI Providers
            Provider::Rev => Some("REV_API_KEY"),
            Provider::Speechmatics => Some("SPEECHMATICS_API_KEY"),
            Provider::PlayHT => Some("PLAYHT_API_KEY"),
            Provider::Resemble => Some("RESEMBLE_API_KEY"),
            // Tier 10 - Image AI Providers
            Provider::Leonardo => Some("LEONARDO_API_KEY"),
            Provider::Ideogram => Some("IDEOGRAM_API_KEY"),
            Provider::BlackForestLabs => Some("BFL_API_KEY"),
            Provider::Clarifai => Some("CLARIFAI_API_KEY"),
            Provider::Fal => Some("FAL_API_KEY"),
            // Tier 11 - Infrastructure Providers
            Provider::Modal => Some("MODAL_TOKEN"),
            Provider::CoreWeave => Some("COREWEAVE_API_KEY"),
            Provider::TensorDock => Some("TENSORDOCK_API_KEY"),
            Provider::Beam => Some("BEAM_API_KEY"),
            Provider::VastAI => Some("VASTAI_API_KEY"),
            // Tier 12 - Emerging Startups
            Provider::Nscale => Some("NSCALE_API_KEY"),
            Provider::Runware => Some("RUNWARE_API_KEY"),
            Provider::AI71 => Some("AI71_API_KEY"),
            // Local
            Provider::Vllm | Provider::Local | Provider::Custom => None,
        }
    }

    /// Check if the provider's API key is available in environment.
    pub fn is_available(&self) -> bool {
        match self {
            Provider::Bedrock | Provider::Sagemaker => {
                std::env::var("AWS_ACCESS_KEY_ID").is_ok() || std::env::var("AWS_PROFILE").is_ok()
            }
            // Local providers - always available
            Provider::Vllm
            | Provider::Local
            | Provider::Custom
            | Provider::Ollama
            | Provider::GPT4All
            | Provider::LMStudio
            | Provider::Llamafile
            | Provider::Xinference
            | Provider::LocalAI
            | Provider::Jan
            | Provider::Petals
            | Provider::Triton
            | Provider::Tgi => true,
            _ => self
                .api_key_env_var()
                .map(|v| std::env::var(v).is_ok())
                .unwrap_or(true),
        }
    }

    /// Parse provider from unified ID prefix.
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
            // Regional providers
            "writer" => Provider::Writer,
            "maritaca" | "maritalk" => Provider::Maritaca,
            "clova" | "hyperclova" => Provider::Clova,
            "yandex" | "yandexgpt" => Provider::Yandex,
            "gigachat" | "sber" => Provider::GigaChat,
            "upstage" | "solar" => Provider::Upstage,
            "sea-lion" | "sealion" | "aisingapore" => Provider::SeaLion,
            // Additional providers
            "alibaba" | "dashscope" | "qwen" => Provider::Alibaba,
            "aleph_alpha" | "alephalpha" | "luminous" => Provider::AlephAlpha,
            "baidu" | "ernie" | "wenxin" => Provider::Baidu,
            "baseten" => Provider::Baseten,
            "chatlaw" => Provider::ChatLaw,
            "datarobot" => Provider::DataRobot,
            "latamgpt" => Provider::LatamGPT,
            "lighton" => Provider::LightOn,
            "nlp_cloud" | "nlpcloud" => Provider::NLPCloud,
            "oracle" | "oci" => Provider::Oracle,
            "perplexity" | "pplx" => Provider::Perplexity,
            "runpod" => Provider::RunPod,
            "sagemaker" | "aws_sagemaker" => Provider::Sagemaker,
            "sap" => Provider::Sap,
            "snowflake" | "cortex" => Provider::Snowflake,
            "vllm" => Provider::Vllm,
            "watsonx" | "ibm" => Provider::WatsonX,
            // New providers
            "xai" | "x-ai" | "grok" => Provider::Xai,
            "deepinfra" | "deep_infra" => Provider::DeepInfra,
            "nvidia" | "nim" | "nvidia_nim" => Provider::NvidiaNIM,
            // Tier 1 - High Priority Inference
            "ollama" => Provider::Ollama,
            "anyscale" => Provider::Anyscale,
            "github" | "github_models" => Provider::GitHub,
            "friendli" | "friendliai" => Provider::FriendliAI,
            "hyperbolic" => Provider::Hyperbolic,
            "lambda" | "lambdalabs" | "lambda_labs" => Provider::Lambda,
            "novita" | "novita_ai" => Provider::Novita,
            "nebius" => Provider::Nebius,
            "lepton" | "lepton_ai" => Provider::Lepton,
            "stability" | "stability_ai" => Provider::Stability,
            "voyage" | "voyage_ai" | "voyageai" => Provider::Voyage,
            "jina" | "jina_ai" => Provider::Jina,
            "deepgram" => Provider::Deepgram,
            "elevenlabs" | "eleven_labs" => Provider::ElevenLabs,
            "gpt4all" => Provider::GPT4All,
            // Tier 2 - Chinese Providers
            "minimax" => Provider::MiniMax,
            "moonshot" | "kimi" => Provider::Moonshot,
            "zhipu" | "glm" | "chatglm" => Provider::Zhipu,
            "volcengine" | "volc" | "bytedance" => Provider::Volcengine,
            "baichuan" => Provider::Baichuan,
            "stepfun" => Provider::Stepfun,
            "yi" | "lingyiwanwu" => Provider::Yi,
            "spark" | "iflytek" | "xunfei" => Provider::Spark,
            // Tier 3 - Local/Self-Hosted
            "lm_studio" | "lmstudio" | "lm-studio" => Provider::LMStudio,
            "llamafile" => Provider::Llamafile,
            "xinference" | "xorbits" => Provider::Xinference,
            "localai" | "local_ai" => Provider::LocalAI,
            "jan" => Provider::Jan,
            "petals" => Provider::Petals,
            "triton" => Provider::Triton,
            "tgi" | "text_generation_inference" => Provider::Tgi,
            // Tier 4 - Enterprise/Specialized
            "predibase" => Provider::Predibase,
            "octoai" | "octo_ai" => Provider::OctoAI,
            "featherless" => Provider::Featherless,
            "ovhcloud" | "ovh" => Provider::OVHCloud,
            "scaleway" => Provider::Scaleway,
            "crusoe" => Provider::Crusoe,
            "cerebrium" => Provider::Cerebrium,
            "lightning" | "lightning_ai" => Provider::Lightning,
            "assemblyai" | "assembly_ai" => Provider::AssemblyAI,
            "runwayml" | "runway" => Provider::RunwayML,
            // Tier 5 - Asian Regional Providers
            "naver" | "hyperclova_x" => Provider::Naver,
            "kakao" | "kogpt" => Provider::Kakao,
            "lg" | "exaone" | "lg_exaone" => Provider::LGExaone,
            "plamo" | "preferred_networks" | "pfn" => Provider::PLaMo,
            "sarvam" | "sarvam_ai" => Provider::Sarvam,
            "krutrim" | "ola" => Provider::Krutrim,
            "ntt" | "tsuzumi" => Provider::Ntt,
            "softbank" => Provider::SoftBank,
            // Tier 6 - European Sovereign AI
            "ionos" => Provider::Ionos,
            "tilde" => Provider::Tilde,
            "silo" | "silo_ai" | "siloai" | "viking" => Provider::SiloAI,
            "swiss" | "swiss_ai" | "swissai" | "apertus" => Provider::SwissAI,
            // Tier 7 - Router/Gateway/Meta Providers
            "unify" | "unify_ai" => Provider::Unify,
            "martian" | "withmartian" => Provider::Martian,
            "portkey" => Provider::Portkey,
            "helicone" => Provider::Helicone,
            "siliconflow" | "silicon_flow" => Provider::SiliconFlow,
            // Tier 8 - Video AI Providers
            "pika" | "pika_labs" => Provider::Pika,
            "luma" | "lumalabs" | "dream_machine" => Provider::Luma,
            "kling" | "kuaishou" => Provider::Kling,
            "heygen" => Provider::HeyGen,
            "d-id" | "did" => Provider::Did,
            "twelve_labs" | "twelvelabs" => Provider::TwelveLabs,
            // Tier 9 - Audio AI Providers
            "rev" | "rev_ai" => Provider::Rev,
            "speechmatics" => Provider::Speechmatics,
            "playht" | "play_ht" | "play.ht" => Provider::PlayHT,
            "resemble" | "resemble_ai" => Provider::Resemble,
            // Tier 10 - Image AI Providers
            "leonardo" | "leonardo_ai" => Provider::Leonardo,
            "ideogram" => Provider::Ideogram,
            "bfl" | "blackforestlabs" | "black_forest_labs" | "flux" => Provider::BlackForestLabs,
            "clarifai" => Provider::Clarifai,
            "fal" | "fal_ai" => Provider::Fal,
            // Tier 11 - Infrastructure Providers
            "modal" => Provider::Modal,
            "coreweave" => Provider::CoreWeave,
            "tensordock" => Provider::TensorDock,
            "beam" | "beam_cloud" => Provider::Beam,
            "vast" | "vast_ai" | "vastai" => Provider::VastAI,
            // Tier 12 - Emerging Startups
            "nscale" => Provider::Nscale,
            "runware" => Provider::Runware,
            "ai71" => Provider::AI71,
            // Local (legacy mapping)
            "local" => Provider::Local,
            _ => Provider::Custom,
        }
    }

    /// Detect provider from model name/ID.
    pub fn from_model(model: &str) -> Self {
        let model_lower = model.to_lowercase();

        // Check for unified prefix
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
        // Regional providers
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

    /// Get the prefix used in unified IDs.
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
            // Regional providers
            Provider::Writer => "writer",
            Provider::Maritaca => "maritaca",
            Provider::Clova => "clova",
            Provider::Yandex => "yandex",
            Provider::GigaChat => "gigachat",
            Provider::Upstage => "upstage",
            Provider::SeaLion => "sea-lion",
            // Additional providers
            Provider::Alibaba => "alibaba",
            Provider::AlephAlpha => "aleph_alpha",
            Provider::Baidu => "baidu",
            Provider::Baseten => "baseten",
            Provider::ChatLaw => "chatlaw",
            Provider::DataRobot => "datarobot",
            Provider::LatamGPT => "latamgpt",
            Provider::LightOn => "lighton",
            Provider::NLPCloud => "nlp_cloud",
            Provider::Oracle => "oracle",
            Provider::Perplexity => "perplexity",
            Provider::RunPod => "runpod",
            Provider::Sagemaker => "sagemaker",
            Provider::Sap => "sap",
            Provider::Snowflake => "snowflake",
            Provider::Vllm => "vllm",
            Provider::WatsonX => "watsonx",
            // New providers
            Provider::Xai => "xai",
            Provider::DeepInfra => "deepinfra",
            Provider::NvidiaNIM => "nvidia",
            // Tier 1 - High Priority Inference
            Provider::Ollama => "ollama",
            Provider::Anyscale => "anyscale",
            Provider::GitHub => "github",
            Provider::FriendliAI => "friendli",
            Provider::Hyperbolic => "hyperbolic",
            Provider::Lambda => "lambda",
            Provider::Novita => "novita",
            Provider::Nebius => "nebius",
            Provider::Lepton => "lepton",
            Provider::Stability => "stability",
            Provider::Voyage => "voyage",
            Provider::Jina => "jina",
            Provider::Deepgram => "deepgram",
            Provider::ElevenLabs => "elevenlabs",
            Provider::GPT4All => "gpt4all",
            // Tier 2 - Chinese Providers
            Provider::MiniMax => "minimax",
            Provider::Moonshot => "moonshot",
            Provider::Zhipu => "zhipu",
            Provider::Volcengine => "volcengine",
            Provider::Baichuan => "baichuan",
            Provider::Stepfun => "stepfun",
            Provider::Yi => "yi",
            Provider::Spark => "spark",
            // Tier 3 - Local/Self-Hosted
            Provider::LMStudio => "lm_studio",
            Provider::Llamafile => "llamafile",
            Provider::Xinference => "xinference",
            Provider::LocalAI => "localai",
            Provider::Jan => "jan",
            Provider::Petals => "petals",
            Provider::Triton => "triton",
            Provider::Tgi => "tgi",
            // Tier 4 - Enterprise/Specialized
            Provider::Predibase => "predibase",
            Provider::OctoAI => "octoai",
            Provider::Featherless => "featherless",
            Provider::OVHCloud => "ovhcloud",
            Provider::Scaleway => "scaleway",
            Provider::Crusoe => "crusoe",
            Provider::Cerebrium => "cerebrium",
            Provider::Lightning => "lightning",
            Provider::AssemblyAI => "assemblyai",
            Provider::RunwayML => "runwayml",
            // Tier 5 - Asian Regional Providers
            Provider::Naver => "naver",
            Provider::Kakao => "kakao",
            Provider::LGExaone => "lg_exaone",
            Provider::PLaMo => "plamo",
            Provider::Sarvam => "sarvam",
            Provider::Krutrim => "krutrim",
            Provider::Ntt => "ntt",
            Provider::SoftBank => "softbank",
            // Tier 6 - European Sovereign AI
            Provider::Ionos => "ionos",
            Provider::Tilde => "tilde",
            Provider::SiloAI => "silo_ai",
            Provider::SwissAI => "swiss_ai",
            // Tier 7 - Router/Gateway/Meta Providers
            Provider::Unify => "unify",
            Provider::Martian => "martian",
            Provider::Portkey => "portkey",
            Provider::Helicone => "helicone",
            Provider::SiliconFlow => "siliconflow",
            // Tier 8 - Video AI Providers
            Provider::Pika => "pika",
            Provider::Luma => "luma",
            Provider::Kling => "kling",
            Provider::HeyGen => "heygen",
            Provider::Did => "d-id",
            Provider::TwelveLabs => "twelve_labs",
            // Tier 9 - Audio AI Providers
            Provider::Rev => "rev",
            Provider::Speechmatics => "speechmatics",
            Provider::PlayHT => "playht",
            Provider::Resemble => "resemble",
            // Tier 10 - Image AI Providers
            Provider::Leonardo => "leonardo",
            Provider::Ideogram => "ideogram",
            Provider::BlackForestLabs => "bfl",
            Provider::Clarifai => "clarifai",
            Provider::Fal => "fal",
            // Tier 11 - Infrastructure Providers
            Provider::Modal => "modal",
            Provider::CoreWeave => "coreweave",
            Provider::TensorDock => "tensordock",
            Provider::Beam => "beam",
            Provider::VastAI => "vastai",
            // Tier 12 - Emerging Startups
            Provider::Nscale => "nscale",
            Provider::Runware => "runware",
            Provider::AI71 => "ai71",
            // Local (legacy)
            Provider::Local => "local",
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
    /// unified model ID (e.g., "anthropic/claude-3-5-sonnet")
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
/// - id: unified (anthropic/claude-3-5-sonnet)
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
# MODEL REGISTRY - Auto-generated from CSV files
# Format: id|alias|name|status|pricing|context|caps|benchmarks|description|classify
# Generated: 2026-01-04 21:20
# Total models: 1969
# =============================================================================

# =============================================================================
# ANTHROPIC - Direct API (35 models)
# =============================================================================
anthropic/claude-3-5-haiku-20241022|claude-3.5-haiku|Claude 3.5 Haiku|C|0.80,4.0,0.08|200000,8192|VTJSC|-|Fast and affordable for high-volume tasks|Y
anthropic/claude-3-5-sonnet-20241022|claude-3.5-sonnet|Claude 3.5 Sonnet|C|3.0,15.0,0.30|200000,8192|VTJSC|-|Best for coding, analysis, and complex reasoning tasks|Y
anthropic/claude-3-7-sonnet-20250219|claude-3.7-sonnet|Claude 3.7 Sonnet|C|3.0,15.0,0.30|200000,128000|VTJSKC|-|Extended thinking with 128K output for complex reasoning|Y
anthropic/claude-3-haiku|claude-3-haiku|Anthropic: Claude 3 Haiku|C|0.25,1.25|200000,4096|TV|-|Claude 3 Haiku is Anthropic's fastest and most compact model for near-instant re|Y
anthropic/claude-3-haiku-20240307|claude-3-haiku|Claude 3 Haiku|C|0.25,1.2,0.03|200000,4096|VTJ|-|Fastest and most compact Claude 3 model|Y
anthropic/claude-3-haiku-vision|claude-3-haiku-vision|Anthropic: Claude 3 Haiku Vision|L|0.25,1.25|200000,1024|VSTJKC|-|Lightweight Claude 3 with vision support|Y
anthropic/claude-3-opus|claude-3-opus|Anthropic: Claude 3 Opus|C|15.0,75.0|200000,4096|TV|-|Claude 3 Opus is Anthropic's most powerful model for highly complex tasks. It bo|Y
anthropic/claude-3-opus-20240229|claude-3-opus|Claude 3 Opus|C|15.0,75.0,1.5|200000,4096|VTJC|-|Powerful model for complex tasks requiring deep understanding|Y
anthropic/claude-3-opus-vision|claude-3-opus-vision|Anthropic: Claude 3 Opus Vision|L|15.0,75.0|200000,4096|VSTJKC|-|Original Claude 3 Opus with vision|Y
anthropic/claude-3-sonnet-20240229|claude-3-sonnet|Claude 3 Sonnet|C|3.0,15.0,0.30|200000,4096|VTJ|-|Balanced performance for wide range of tasks|Y
anthropic/claude-3-sonnet-vision|claude-3-sonnet-vision|Anthropic: Claude 3 Sonnet Vision|L|3.0,15.0|200000,4096|VSTJKC|-|Claude 3 Sonnet with vision capabilities|Y
anthropic/claude-3.5-haiku|claude-3.5-haiku|Anthropic: Claude 3.5 Haiku|C|1.0,5.0,0.10|200000,8192|TV|-|Claude 3.5 Haiku features offers enhanced capabilities in speed, coding accuracy|Y
anthropic/claude-3.5-haiku-20241022|claude-3.5-haiku-202|Anthropic: Claude 3.5 Haiku (2024-10-22)|C|1.0,5.0,0.10|200000,8192|TV|-|Claude 3.5 Haiku features enhancements across all skill sets including coding, t|Y
anthropic/claude-3.5-sonnet|claude-3.5-sonnet|Anthropic: Claude 3.5 Sonnet|C|3.0,15.0,0.30|200000,8192|TV|-|New Claude 3.5 Sonnet delivers better-than-Opus capabilities, faster-than-Sonnet|Y
anthropic/claude-3.7-sonnet|claude-3.7-sonnet|Anthropic: Claude 3.7 Sonnet|C|3.0,15.0,0.30|200000,64000|KTV|-|Claude 3.7 Sonnet is an advanced large language model with improved reasoning, c|N
anthropic/claude-3.7-sonnet:thinking|claude-3.7-sonnet:th|Anthropic: Claude 3.7 Sonnet (thinking)|C|3.0,15.0,0.30|200000,64000|KTV|-|Claude 3.7 Sonnet is an advanced large language model with improved reasoning, c|N
anthropic/claude-haiku-4-5-20251015|claude-4.5-haiku|Claude 4.5 Haiku|C|1.0,5.0,0.10|200000,8192|VTJSC|-|Fast model optimized for low latency and cost|Y
anthropic/claude-haiku-4.5|claude-haiku-4.5|Anthropic: Claude Haiku 4.5|C|1.0,5.0,0.10|200000,64000|KTV|-|Claude Haiku 4.5 is Anthropic's fastest and most efficient model, delivering nea|N
anthropic/claude-opus-4|claude-opus-4|Anthropic: Claude Opus 4|C|15.0,75.0,1.5|200000,32000|KTV|-|Claude Opus 4 is benchmarked as the world's best coding model, at time of releas|N
anthropic/claude-opus-4-1-20250805|claude-4.1-opus|Claude 4.1 Opus|C|15.0,75.0,1.5|200000,32000|VTJSKC|-|Most powerful Claude for agentic tasks, coding, and reasoning|Y
anthropic/claude-opus-4-20250514|claude-4-opus|Claude 4 Opus|C|15.0,75.0,1.5|200000,32000|VTJSKC|-|Claude 4 flagship for complex tasks, research, and analysis|Y
anthropic/claude-opus-4-5-20251101|claude-opus-4-5|Claude Opus 4.5|C|5.0,25.0,0.50|200000,4096|SVTJK|-|Frontier model with extended thinking capability|Y
anthropic/claude-opus-4-finetuned-code|claude-opus-code|claude-code|Anthropic: Claude Opus Code FT|C|6.0,30.0|200000,32000|VSTJKC|-|Claude Opus fine-tuned for code generation and analysis|Y
anthropic/claude-opus-4-finetuned-financial|claude-opus-finance|claude-finance|Anthropic: Claude Opus Finance FT|C|6.0,30.0|200000,32000|VSTJKC|-|Claude Opus fine-tuned for financial analysis|Y
anthropic/claude-opus-4-finetuned-legal|claude-opus-legal|claude-legal|Anthropic: Claude Opus Legal FT|C|6.0,30.0|200000,32000|VSTJKC|-|Claude Opus fine-tuned for legal contract review|Y
anthropic/claude-opus-4-finetuned-medical|claude-opus-medical|claude-medical|Anthropic: Claude Opus Medical FT|C|6.0,30.0|200000,32000|VSTJKC|-|Claude Opus fine-tuned for medical document analysis|Y
anthropic/claude-opus-4.1|claude-opus-4.1|Anthropic: Claude Opus 4.1|C|15.0,75.0,1.5|200000,50000|JKSTV|-|Claude Opus 4.1 is an updated version of Anthropic's flagship model, offering im|Y
anthropic/claude-opus-4.5|claude-opus-4.5|Anthropic: Claude Opus 4.5|C|5.0,25.0,0.50|200000,32000|JKSTV|-|Claude Opus 4.5 is Anthropic's frontier reasoning model optimized for complex so|Y
anthropic/claude-sonnet-4|claude-sonnet-4|Anthropic: Claude Sonnet 4|C|3.0,15.0,0.30|1000000,64000|KTV|-|Claude Sonnet 4 significantly enhances the capabilities of its predecessor, Sonn|N
anthropic/claude-sonnet-4-20250514|claude-4-sonnet|Claude 4 Sonnet|C|3.0,15.0,0.30|200000,64000|VTJSKC|-|Balanced intelligence and speed for everyday tasks|Y
anthropic/claude-sonnet-4-5-20250924|claude-sonnet-4-5|Claude Sonnet 4.5|C|3.0,15.0,0.30|200000,4096|SVTJK|-|Advanced reasoning with improved speed|Y
anthropic/claude-sonnet-4-5-20250929|claude-4.5-sonnet|Claude 4.5 Sonnet|C|3.0,15.0,0.30|200000,16384|VTJSKC|-|Best for complex coding and analysis, supports 1M context with beta header|Y
anthropic/claude-sonnet-4-finetuned-chat|claude-sonnet-chat|claude-chat-ft|Anthropic: Claude Sonnet Chat FT|C|3.0,15.0|200000,4096|VSTJKC|-|Claude Sonnet fine-tuned for conversational AI|Y
anthropic/claude-sonnet-4.5|claude-sonnet-4.5|Anthropic: Claude Sonnet 4.5|C|3.0,15.0,0.30|1000000,64000|JKSTV|-|Claude Sonnet 4.5 is Anthropic's most advanced Sonnet model to date, optimized f|Y
anthropic/claude3opus|anthropic-claude3-opus|Anthropic: Claude 3 Opus|L|0.01,0.07|200000,4096|VSTJKC|-|Original Claude 3 Opus release|Y

# =============================================================================
# OPENAI - Direct API (70 models)
# =============================================================================
openai/chatgpt-4o-latest|chatgpt-4o-latest|OpenAI: ChatGPT-4o|C|0.0000,0.0000|128000,16384|JSV|-|OpenAI ChatGPT 4o is continually updated by OpenAI to point to the current versi|Y
openai/codex-mini|codex-mini|OpenAI: Codex Mini|C|0.0000,0.0000|200000,100000|JKSTV|-|codex-mini-latest is a fine-tuned version of o4-mini specifically for use in Cod|Y
openai/dall-e-3|dall-e-3|DALL-E 3|C|0.04,|4000,0|I|-|Image generation, $0.04-0.12/image|N
openai/gpt-3.5-turbo|gpt-3.5-turbo|GPT-3.5 Turbo|C|0.50,1.5|16385,4096|TJ|-|Legacy model, fast and affordable|Y
openai/gpt-3.5-turbo-0613|gpt-3.5-turbo-0613|OpenAI: GPT-3.5 Turbo (older v0613)|C|0.0000,0.0000|4095,4096|JST|-|GPT-3.5 Turbo is OpenAI's fastest model. It can understand and generate natural|Y
openai/gpt-3.5-turbo-16k|gpt-3.5-turbo-16k|OpenAI: GPT-3.5 Turbo 16k|C|0.0000,0.0000|16385,4096|JST|-|This model offers four times the context length of gpt-3.5-turbo, allowing it to|Y
openai/gpt-3.5-turbo-instruct|gpt-3.5-turbo-instru|OpenAI: GPT-3.5 Turbo Instruct|C|0.0000,0.0000|4095,4096|JS|-|This model is a variant of GPT-3.5 Turbo tuned for instructional prompts and omi|Y
openai/gpt-4|gpt-4|GPT-4|C|30.0,60.0|8192,8192|TJS|-|Original GPT-4 model|Y
openai/gpt-4-0314|gpt-4-0314|OpenAI: GPT-4 (older v0314)|C|0.0000,0.0001|8191,4096|JST|-|GPT-4-0314 is the first version of GPT-4 released, with a context length of 8,19|Y
openai/gpt-4-1106-preview|gpt-4-1106-preview|OpenAI: GPT-4 Turbo (older v1106)|C|0.0000,0.0000|128000,4096|JST|-|The latest GPT-4 Turbo model with vision capabilities. Vision requests can now u|Y
openai/gpt-4-32k-vision|gpt-4-32k-vision|OpenAI: GPT-4 32K Vision|L|0.01,0.03|32000,4096|VST|-|GPT-4 with limited context for vision|Y
openai/gpt-4-turbo|gpt-4-turbo|GPT-4 Turbo|C|10.0,30.0|128000,4096|VTJS|-|Previous generation GPT-4 with vision|Y
openai/gpt-4-turbo-finetuned-code|gpt-4-code|gpt-4-code-ft|OpenAI: GPT-4 Turbo Code FT|C|3.0,12.0|128000,4096|VSTJS|-|GPT-4 Turbo fine-tuned for advanced code tasks|Y
openai/gpt-4-turbo-finetuned-medical|gpt-4-medical|gpt-4-medical-ft|OpenAI: GPT-4 Turbo Medical FT|C|3.0,12.0|128000,4096|VSTJS|-|GPT-4 Turbo fine-tuned for medical applications|Y
openai/gpt-4-turbo-preview|gpt-4-turbo-preview|OpenAI: GPT-4 Turbo Preview|C|0.0000,0.0000|128000,4096|JST|-|The preview GPT-4 model with improved instruction following, JSON mode, reproduc|Y
openai/gpt-4-turbo-vision|gpt-4-turbo-vision|OpenAI: GPT-4 Turbo Vision|L|0.0000,0.0000|128000,4096|VST|-|GPT-4 Turbo with vision support|Y
openai/gpt-4-vision|gpt-4-vision|OpenAI: GPT-4 Vision|L|0.0000,0.0000|128000,4096|VST|-|Legacy GPT-4 with vision capability|Y
openai/gpt-4.1|gpt-4.1|GPT-4.1|C|2.0,8.0,0.50|1047576,32768|VTJSC|-|Most capable GPT-4 with 1M context, flagship model|Y
openai/gpt-4.1-mini|gpt-4.1-mini|GPT-4.1 Mini|C|0.40,1.6,0.10|1047576,32768|VTJSC|-|Affordable 1M context model for high volume tasks|Y
openai/gpt-4.1-nano|gpt-4.1-nano|GPT-4.1 Nano|C|0.10,0.40,0.03|1047576,32768|VTJS|-|Fastest and cheapest GPT-4.1 variant|Y
openai/gpt-4o|gpt-4o|GPT-4o|C|2.5,10.0,1.2|128000,16384|VTJSC|-|Flagship multimodal model for text, vision, and audio|Y
openai/gpt-4o-2024-05-13|gpt-4o-2024-05-13|OpenAI: GPT-4o (2024-05-13)|C|0.0000,0.0000|128000,4096|JSTV|-|GPT-4o ("o" for "omni") is OpenAI's latest AI model, supporting both text and im|Y
openai/gpt-4o-2024-08-06|gpt-4o-2024-08-06|OpenAI: GPT-4o (2024-08-06)|C|0.0000,0.0000|128000,16384|JSTV|-|The 2024-08-06 version of GPT-4o offers improved performance in structured outpu|Y
openai/gpt-4o-2024-11-20|gpt-4o-2024-11-20|OpenAI: GPT-4o (2024-11-20)|C|0.0000,0.0000|128000,16384|JSTV|-|The 2024-11-20 version of GPT-4o offers a leveled-up creative writing ability wi|Y
openai/gpt-4o-audio-preview|gpt-4o-audio|GPT-4o Audio Preview|C|2.5,10.0|128000,16384|VTJS|-|GPT-4o with native audio understanding|Y
openai/gpt-4o-finetuned-instructions|gpt-4o-inst|gpt-4o-inst-ft|OpenAI: GPT-4o Instructions FT|C|3.0,12.0|128000,16384|VSTJS|-|GPT-4o fine-tuned for instruction following|Y
openai/gpt-4o-mini|gpt-4o-mini|GPT-4o Mini|C|0.15,0.60,0.07|128000,16384|VTJSC|-|Affordable multimodal model for lightweight tasks|Y
openai/gpt-4o-mini-2024-07-18|gpt-4o-mini-2024-07-|OpenAI: GPT-4o-mini (2024-07-18)|C|0.0000,0.0000|128000,16384|JSTV|-|GPT-4o mini is OpenAI's newest model after [GPT-4 Omni](/models/openai/gpt-4o),|Y
openai/gpt-4o-mini-search-preview|gpt-4o-mini-search-p|OpenAI: GPT-4o-mini Search Preview|C|0.0000,0.0000|128000,16384|JS|-|GPT-4o mini Search Preview is a specialized model for web search in Chat Complet|Y
openai/gpt-4o-search-preview|gpt-4o-search-previe|OpenAI: GPT-4o Search Preview|C|0.0000,0.0000|128000,16384|JS|-|GPT-4o Search Previewis a specialized model for web search in Chat Completions.|Y
openai/gpt-4o:extended|gpt-4o:extended|OpenAI: GPT-4o (extended)|C|0.0000,0.0000|128000,64000|JSTV|-|GPT-4o ("o" for "omni") is OpenAI's latest AI model, supporting both text and im|Y
openai/gpt-5|gpt-5|GPT-5|C|1.2,10.0,0.30|1000000,100000|VTJSKC|-|Most capable OpenAI model, unified reasoning and language|Y
openai/gpt-5-chat|gpt-5-chat|OpenAI: GPT-5 Chat|C|0.0000,0.0000|128000,16384|JSV|-|GPT-5 Chat is designed for advanced, natural, multimodal, and context-aware conv|Y
openai/gpt-5-codex|gpt-5-codex|OpenAI: GPT-5 Codex|C|0.0000,0.0000|400000,128000|JKSTV|-|GPT-5-Codex is a specialized version of GPT-5 optimized for software engineering|Y
openai/gpt-5-image|gpt-5-image|OpenAI: GPT-5 Image|C|0.0000,0.0000|400000,128000|JKSTV|-|[GPT-5](https://openrouter.ai/openai/gpt-5) Image combines OpenAI's GPT-5 model|Y
openai/gpt-5-image-mini|gpt-5-image-mini|OpenAI: GPT-5 Image Mini|C|0.0000,0.0000|400000,128000|JKSTV|-|GPT-5 Image Mini combines OpenAI's advanced language capabilities, powered by [G|Y
openai/gpt-5-mini|gpt-5-mini|GPT-5 Mini|C|0.30,1.2,0.07|1000000,100000|VTJSKC|-|Efficient GPT-5 for high-volume tasks|Y
openai/gpt-5-nano|gpt-5-nano|OpenAI: GPT-5 Nano|C|0.0000,0.0000|400000,128000|JKSTV|-|GPT-5-Nano is the smallest and fastest variant in the GPT-5 system, optimized fo|Y
openai/gpt-5-pro|gpt-5-pro|OpenAI: GPT-5 Pro|C|0.0000,0.0001|400000,128000|JKSTV|-|GPT-5 Pro is OpenAI's most advanced model, offering major improvements in reason|Y
openai/gpt-5.1|gpt-5.1|OpenAI: GPT-5.1|C|0.0000,0.0000|400000,128000|JKSTV|-|GPT-5.1 is the latest frontier-grade model in the GPT-5 series, offering stronge|Y
openai/gpt-5.1-chat|gpt-5.1-chat|OpenAI: GPT-5.1 Chat|C|0.0000,0.0000|128000,16384|JSTV|-|GPT-5.1 Chat (AKA Instant is the fast, lightweight member of the 5.1 family, opt|Y
openai/gpt-5.1-codex|gpt-5.1-codex|OpenAI: GPT-5.1-Codex|C|0.0000,0.0000|400000,128000|JKSTV|-|GPT-5.1-Codex is a specialized version of GPT-5.1 optimized for software enginee|Y
openai/gpt-5.1-codex-max|gpt-5.1-codex-max|OpenAI: GPT-5.1-Codex-Max|C|0.0000,0.0000|400000,128000|JKSTV|-|GPT-5.1-Codex-Max is OpenAI's latest agentic coding model, designed for long-run|Y
openai/gpt-5.1-codex-mini|gpt-5.1-codex-mini|OpenAI: GPT-5.1-Codex-Mini|C|0.0000,0.0000|400000,100000|JKSTV|-|GPT-5.1-Codex-Mini is a smaller and faster version of GPT-5.1-Codex|Y
openai/gpt-5.2|gpt-5.2|OpenAI: GPT-5.2|C|0.0000,0.0000|400000,128000|JKSTV|-|GPT-5.2 is the latest frontier-grade model in the GPT-5 series, offering stronge|Y
openai/gpt-5.2-chat|gpt-5.2-chat|OpenAI: GPT-5.2 Chat|C|0.0000,0.0000|128000,16384|JSTV|-|GPT-5.2 Chat (AKA Instant) is the fast, lightweight member of the 5.2 family, op|Y
openai/gpt-5.2-pro|gpt-5.2-pro|OpenAI: GPT-5.2 Pro|C|0.0000,0.0002|400000,128000|JKSTV|-|GPT-5.2 Pro is OpenAI's most advanced model, offering major improvements in agen|Y
openai/gpt-image-1|gpt-image-1|GPT Image 1|C|5.0,|32000,0|I|-|Advanced image generation with text input|N
openai/gpt-oss-120b|gpt-oss-120b|OpenAI: gpt-oss-120b|C|0.0000,0.0000|131072,32768|JKST|-|gpt-oss-120b is an open-weight, 117B-parameter Mixture-of-Experts (MoE) language|Y
openai/gpt-oss-120b:exacto|gpt-oss-120b:exacto|OpenAI: gpt-oss-120b (exacto)|C|0.0000,0.0000|131072,32768|JKST|-|gpt-oss-120b is an open-weight, 117B-parameter Mixture-of-Experts (MoE) language|Y
openai/gpt-oss-120b:free|gpt-oss-120b:free|OpenAI: gpt-oss-120b (free)|C|-|131072,32768|KT|-|gpt-oss-120b is an open-weight, 117B-parameter Mixture-of-Experts (MoE) language|N
openai/gpt-oss-20b|gpt-oss-20b|OpenAI: gpt-oss-20b|C|0.0000,0.0000|131072,32768|JKST|-|gpt-oss-20b is an open-weight 21B parameter model released by OpenAI under the A|Y
openai/gpt-oss-20b:free|gpt-oss-20b:free|OpenAI: gpt-oss-20b (free)|C|-|131072,131072|JKST|-|gpt-oss-20b is an open-weight 21B parameter model released by OpenAI under the A|Y
openai/gpt-oss-safeguard-20b|gpt-oss-safeguard-20|OpenAI: gpt-oss-safeguard-20b|C|0.0000,0.0000|131072,65536|JKT|-|gpt-oss-safeguard-20b is a safety reasoning model from OpenAI built upon gpt-oss|N
openai/o1|o1|o1|C|15.0,60.0,3.8|200000,100000|VTJSK|-|Deep reasoning for math, science, and coding|Y
openai/o1-mini|o1-mini|o1 Mini|C|1.1,4.4,0.55|128000,65536|VTJSK|-|Fast reasoning for STEM tasks|Y
openai/o1-pro|o1-pro|o1 Pro|C|150.0,600.0,37.5|200000,100000|VTJSK|-|Extended compute for hardest problems|Y
openai/o3|o3|o3|C|2.0,8.0,0.5|200000,100000|VTJSK|-|Most powerful reasoning model for complex problems|Y
openai/o3-deep-research|o3-deep-research|OpenAI: o3 Deep Research|C|0.0000,0.0000|200000,100000|JKSTV|-|o3-deep-research is OpenAI's advanced model for deep research, designed to tackl|Y
openai/o3-mini|o3-mini|o3 Mini|C|1.1,4.4,0.28|200000,100000|VTJSK|-|Fast and affordable reasoning model|Y
openai/o3-mini-high|o3-mini-high|OpenAI: o3 Mini High|C|0.0000,0.0000|200000,100000|JST|-|OpenAI o3-mini-high is the same model as [o3-mini](/openai/o3-mini) with reasoni|Y
openai/o3-pro|o3-pro|o3 Pro|C|20.0,80.0,5.0|200000,100000|VTJSK|-|Extended compute for hard reasoning problems|Y
openai/o4-mini|o4-mini|o4 Mini|C|1.1,4.4,0.28|200000,100000|VTJSK|-|Latest reasoning model, balanced speed and capability|Y
openai/o4-mini-deep-research|o4-mini-deep-researc|OpenAI: o4 Mini Deep Research|C|0.0000,0.0000|200000,100000|JKSTV|-|o4-mini-deep-research is OpenAI's faster, more affordable deep research model-id|Y
openai/o4-mini-high|o4-mini-high|OpenAI: o4 Mini High|C|0.0000,0.0000|200000,100000|JKSTV|-|OpenAI o4-mini-high is the same model as [o4-mini](/openai/o4-mini) with reasoni|Y
openai/text-embedding-3-large|embed-3-large|Text Embedding 3 Large|C|0.13,|8191,3072|E|-|Most capable embedding model, 3072 dimensions|N
openai/text-embedding-3-small|embed-3-small|Text Embedding 3 Small|C|0.02,|8191,1536|E|-|Efficient embedding model, 1536 dimensions|N
openai/tts-1|tts-1|TTS-1|C|15.0,|4096,0|A|-|Text-to-speech, $15/1M characters|N
openai/tts-1-hd|tts-1-hd|TTS-1 HD|C|30.0,|4096,0|A|-|High-quality text-to-speech, $30/1M characters|N
openai/whisper-1|whisper-1|Whisper|C|0.0060,|0,0|A|-|Speech-to-text model, $0.006/minute|N

# =============================================================================
# GOOGLE - Direct API (45 models)
# =============================================================================
google/gemini-1.0-pro|gemini-1.0-pro|Gemini 1.0 Pro|C|0.50,1.5|32760,8192|TJ|-|Original Gemini model, legacy support|Y
google/gemini-1.5-flash|gemini-1.5-flash|Gemini 1.5 Flash|C|0.07,0.30,0.02|1048576,8192|VTJSC|-|Fast and efficient with 1M context|Y
google/gemini-1.5-flash-8b|gemini-1.5-flash-8b|Gemini 1.5 Flash 8B|C|0.04,0.15,0.01|1048576,8192|VTJS|-|Smallest Flash variant, highly efficient|Y
google/gemini-1.5-pro|gemini-1.5-pro|Gemini 1.5 Pro|C|1.2,5.0,0.31|2097152,8192|VTJSC|-|2M context for complex reasoning and analysis|Y
google/gemini-1.5-pro-finetuned-rag|gemini-pro-rag|gemini-rag-ft|Google: Gemini 1.5 Pro RAG FT|C|1.2,5.0|1000000,8192|VSTJK|-|Gemini Pro fine-tuned for RAG and retrieval|Y
google/gemini-1.5-pro-finetuned-translation|gemini-pro-trans|gemini-trans-ft|Google: Gemini 1.5 Pro Translation FT|C|1.2,5.0|1000000,8192|VSTJK|-|Gemini Pro fine-tuned for multilingual translation|Y
google/gemini-1.5-vision|gemini-1.5-vision|Google: Gemini 1.5 Vision|L|0.0000,0.0000|1000000,4096|VSTJ|-|Previous-generation Gemini vision model|Y
google/gemini-2.0-flash|gemini-2.0-flash|Gemini 2.0 Flash|C|0.10,0.40,0.03|1048576,8192|VTJS|-|Fast multimodal model with tool use|Y
google/gemini-2.0-flash-001|gemini-2.0-flash-001|Google: Gemini 2.0 Flash|C|0.0000,0.0000|1048576,8192|JSTV|-|Gemini Flash 2.0 offers a significantly faster time to first token (TTFT) compar|Y
google/gemini-2.0-flash-exp:free|gemini-2.0-flash-exp|Google: Gemini 2.0 Flash Experimental (free)|C|-|1048576,8192|JTV|-|Gemini Flash 2.0 offers a significantly faster time to first token (TTFT) compar|Y
google/gemini-2.0-flash-lite|gemini-2.0-flash-lite|Gemini 2.0 Flash Lite|C|0.07,0.30|1048576,8192|VTJ|-|Lightweight and cost-effective Flash variant|Y
google/gemini-2.0-flash-lite-001|gemini-2.0-flash-lit|Google: Gemini 2.0 Flash Lite|C|0.0000,0.0000|1048576,8192|JSTV|-|Gemini 2.0 Flash Lite offers a significantly faster time to first token (TTFT) c|Y
google/gemini-2.5-flash|gemini-2.5-flash|Google: Gemini 2.5 Flash|C|0.0000,0.0000|1048576,65535|JKSTV|-|Gemini 2.5 Flash is Google's state-of-the-art workhorse model, specifically desi|Y
google/gemini-2.5-flash-image|gemini-2.5-flash-ima|Google: Gemini 2.5 Flash Image (Nano Banana)|C|0.0000,0.0000|32768,32768|JSV|-|Gemini 2.5 Flash Image, a.k.a. "Nano Banana," is now generally available. It is|Y
google/gemini-2.5-flash-image-preview|gemini-2.5-flash-ima|Google: Gemini 2.5 Flash Image Preview (Nano Banana)|C|0.0000,0.0000|32768,32768|JSV|-|Gemini 2.5 Flash Image Preview, a.k.a. "Nano Banana," is a state of the art imag|Y
google/gemini-2.5-flash-lite|gemini-2.5-flash-lit|Google: Gemini 2.5 Flash Lite|C|0.0000,0.0000|1048576,65535|JKSTV|-|Gemini 2.5 Flash-Lite is a lightweight reasoning model in the Gemini 2.5 family,|Y
google/gemini-2.5-flash-lite-preview-09-2025|gemini-2.5-flash-lit|Google: Gemini 2.5 Flash Lite Preview 09-2025|C|0.0000,0.0000|1048576,65536|JKSTV|-|Gemini 2.5 Flash-Lite is a lightweight reasoning model in the Gemini 2.5 family,|Y
google/gemini-2.5-flash-preview-05-20|gemini-2.5-flash|Gemini 2.5 Flash|C|0.15,0.60,0.04|1048576,65536|VTJSK|-|Fast thinking model with 1M context|Y
google/gemini-2.5-flash-preview-09-2025|gemini-2.5-flash-pre|Google: Gemini 2.5 Flash Preview 09-2025|C|0.0000,0.0000|1048576,65536|JKSTV|-|Gemini 2.5 Flash Preview September 2025 Checkpoint is Google's state-of-the-art|Y
google/gemini-2.5-pro|gemini-2.5-pro|Google: Gemini 2.5 Pro|C|0.0000,0.0000|1048576,65536|JKSTV|-|Gemini 2.5 Pro is Google's state-of-the-art AI model designed for advanced reaso|Y
google/gemini-2.5-pro-preview|gemini-2.5-pro-previ|Google: Gemini 2.5 Pro Preview 06-05|C|0.0000,0.0000|1048576,65536|JKSTV|-|Gemini 2.5 Pro is Google's state-of-the-art AI model designed for advanced reaso|Y
google/gemini-2.5-pro-preview-05-06|gemini-2.5-pro-previ|Google: Gemini 2.5 Pro Preview 05-06|C|0.0000,0.0000|1048576,65535|JKSTV|-|Gemini 2.5 Pro is Google's state-of-the-art AI model designed for advanced reaso|Y
google/gemini-2.5-pro-preview-06-05|gemini-2.5-pro|Gemini 2.5 Pro|C|1.2,10.0,0.31|1048576,65536|VTJSK|-|Latest flagship model with thinking, 1M context|Y
google/gemini-2.5-vision|gemini-2.5-vision|Google: Gemini 2.5 Vision|C|0.0000,0.0000|1000000,8192|VSTJK|-|Gemini 2.5 with advanced visual understanding and reasoning|Y
google/gemini-3-flash-20260101|gemini-3-flash|Gemini 3 Flash|C|0.0000,0.0000|1000000,8192|SVTJC|-|Fast generation model from Google|Y
google/gemini-3-flash-preview|gemini-3-flash|Gemini 3 Flash|C|0.20,0.80,0.05|1048576,65536|VTJSK|-|Best for complex multimodal, agentic problems with strong reasoning|Y
google/gemini-3-pro-20260101|gemini-3-pro|Gemini 3 Pro|C|0.0000,0.0000|1000000,8192|SVTJC|-|Latest Google frontier model|Y
google/gemini-3-pro-image-preview|gemini-3-pro-image-p|Google: Nano Banana Pro (Gemini 3 Pro Image Preview)|C|0.0000,0.0000|65536,32768|JKSV|-|Nano Banana Pro is Google's most advanced image-generation and editing model, bu|Y
google/gemini-3-pro-preview|gemini-3-pro|Gemini 3 Pro|C|1.5,12.0,0.38|1048576,65536|VTJSK|-|Latest reasoning-first model for complex agentic workflows and coding|Y
google/gemma-2-27b-it|gemma-2-27b-it|Google: Gemma 2 27B|C|0.0000,0.0000|8192,2048|JS|-|Gemma 2 27B by Google is an open model built from the same research and technolo|Y
google/gemma-2-9b-it|gemma-2-9b-it|Google: Gemma 2 9B|C|0.0000,0.0000|8192,2048|-|-|Gemma 2 9B by Google is an advanced, open-source language model that sets a new|Y
google/gemma-3-12b-it|gemma-3-12b-it|Google: Gemma 3 12B|C|0.0000,0.0000|131072,131072|JSV|-|Gemma 3 introduces multimodality, supporting vision-language input and text outp|Y
google/gemma-3-12b-it:free|gemma-3-12b-it:free|Google: Gemma 3 12B (free)|C|-|32768,8192|V|-|Gemma 3 introduces multimodality, supporting vision-language input and text outp|Y
google/gemma-3-27b-it|gemma-3-27b-it|Google: Gemma 3 27B|C|0.0000,0.0000|131072,32768|JSTV|-|Gemma 3 introduces multimodality, supporting vision-language input and text outp|Y
google/gemma-3-27b-it:free|gemma-3-27b-it:free|Google: Gemma 3 27B (free)|C|-|131072,32768|JSTV|-|Gemma 3 introduces multimodality, supporting vision-language input and text outp|Y
google/gemma-3-4b-it|gemma-3-4b-it|Google: Gemma 3 4B|C|0.0000,0.0000|96000,24000|JV|-|Gemma 3 introduces multimodality, supporting vision-language input and text outp|Y
google/gemma-3-4b-it:free|gemma-3-4b-it:free|Google: Gemma 3 4B (free)|C|-|32768,8192|JSV|-|Gemma 3 introduces multimodality, supporting vision-language input and text outp|Y
google/gemma-3n-e2b-it:free|gemma-3n-e2b-it:free|Google: Gemma 3n 2B (free)|C|-|8192,2048|J|-|Gemma 3n E2B IT is a multimodal, instruction-tuned model developed by Google Dee|Y
google/gemma-3n-e4b-it|gemma-3n-e4b-it|Google: Gemma 3n 4B|C|0.0000,0.0000|32768,8192|-|-|Gemma 3n E4B-it is optimized for efficient execution on mobile and low-resource|Y
google/gemma-3n-e4b-it:free|gemma-3n-e4b-it:free|Google: Gemma 3n 4B (free)|C|-|8192,2048|J|-|Gemma 3n E4B-it is optimized for efficient execution on mobile and low-resource|Y
google/imagen-3.0-fast-generate-001|imagen-3-fast|Imagen 3 Fast|C|0.02,|480,0|I|-|Fast image generation, $0.02/image|N
google/imagen-3.0-generate-002|imagen-3|Imagen 3|C|0.04,|480,0|I|-|Image generation, $0.04/image|N
google/text-embedding-004|text-embedding-004|Text Embedding 004|C|0.0000,|2048,768|E|-|Text embeddings, 768 dimensions|N
google/text-multilingual-embedding-002|multilingual-embed|Multilingual Embedding 002|C|0.0000,|2048,768|E|-|Multilingual text embeddings|N
google/veo-2.0-generate-001|veo-2|Veo 2|C|0.35,|480,0|D|-|Video generation, $0.35/second|N

# =============================================================================
# MISTRAL - Direct API (34 models)
# =============================================================================
mistral/codestral-2405|codestral-2405|Codestral 24.05|C|0.20,0.60|32768,32768|TJ|-|Previous Codestral version|Y
mistral/codestral-2501|codestral|Codestral 25.01|C|0.30,0.90|262144,262144|TJ|-|Specialized code generation model|Y
mistral/codestral-embed-2505|codestral-embed|Codestral Embed|C|0.15,|8192,1024|E|-|Code embeddings model|N
mistral/codestral-mamba-2407|codestral-mamba|Codestral Mamba|C|0.20,0.60|262144,262144|TJ|-|Mamba architecture for code|Y
mistral/devstral-medium-latest|devstral-2|Devstral 2|C|0.40,2.0|131072,131072|TJS|-|Agentic coding model 123B|Y
mistral/devstral-small-latest|devstral-small-2|Devstral Small 2|C|0.10,0.30|131072,131072|TJS|-|Agentic coding model 24B|Y
mistral/magistral-medium-latest|magistral-medium|Magistral Medium|C|2.0,5.0|131072,131072|TJK|-|Reasoning-focused medium model|N
mistral/magistral-small-latest|magistral-small|Magistral Small|C|0.50,1.5|131072,131072|TJK|-|Reasoning-focused small model|N
mistral/ministral-3b-2410|ministral-3b|Ministral 3B|C|0.04,0.04|131072,131072|TJ|-|Smallest Mistral model|Y
mistral/ministral-3b-latest|ministral-3b-3|Ministral 3B|C|0.10,0.10|131072,131072|VTJ|-|Smallest Ministral with vision|Y
mistral/ministral-8b-2410|ministral-8b|Ministral 8B|C|0.10,0.10|131072,131072|TJ|-|Efficient 8B model|Y
mistral/ministral-8b-latest|ministral-8b-3|Ministral 8B|C|0.15,0.15|131072,131072|VTJ|-|Efficient Ministral with vision|Y
mistral/mistral-embed|mistral-embed|Mistral Embed|C|0.10,|8192,1024|E|-|Text embeddings model|N
mistral/mistral-large-2407|mistral-large-2407|Mistral Large 24.07|C|2.0,6.0|131072,131072|VTJS|-|Previous Mistral Large version|Y
mistral/mistral-large-2411|mistral-large|Mistral Large 24.11|C|2.0,6.0|131072,131072|VTJS|-|Flagship model for complex tasks and reasoning|Y
mistral/mistral-large-3-20260101|mistral-large-3|Mistral Large 3|C|0.0000,0.0000|200000,64000|TJ|-|Latest flagship from Mistral|Y
mistral/mistral-large-finetuned-technical|mistral-technical|mistral-tech-ft|Mistral: Large Technical FT|C|1.5,4.5|128000,8192|VSTJ|-|Mistral Large fine-tuned for technical documentation|Y
mistral/mistral-large-latest|mistral-large-3|Mistral Large 3|C|0.50,1.5|131072,131072|VTJS|-|675B MoE flagship model|Y
mistral/mistral-medium-2505|mistral-medium|Mistral Medium 25.05|C|0.40,2.0|131072,131072|VTJS|-|Balanced performance and cost|Y
mistral/mistral-medium-finetuned-customer-service|mistral-service|mistral-service-ft|Mistral: Medium Service FT|C|0.70,2.1|32000,4096|VSTJ|-|Mistral Medium fine-tuned for customer service|Y
mistral/mistral-medium-latest|mistral-medium-3|Mistral Medium 3|C|0.40,2.0|131072,131072|VTJS|-|Balanced medium tier model|Y
mistral/mistral-moderation-2411|mistral-moderation|Mistral Moderation|C|0.10,|8192,0|M|-|Content moderation model|N
mistral/mistral-nemo-2407|mistral-nemo|Mistral Nemo|C|0.15,0.15|131072,131072|TJ|-|12B parameter free tier model|Y
mistral/mistral-small-2409|mistral-small-2409|Mistral Small 24.09|C|0.10,0.30|32768,32768|VTJS|-|Previous Mistral Small version|Y
mistral/mistral-small-2503|mistral-small|Mistral Small 25.03|C|0.10,0.30|32768,32768|VTJS|-|Fast and efficient for most tasks|Y
mistral/mistral-small-latest|mistral-small-3.2|Mistral Small 3.2|C|0.10,0.30|32768,32768|VTJS|-|Latest small model with vision|Y
mistral/open-mistral-7b|mistral-7b|Mistral 7B|C|0.25,0.25|32768,32768|TJ|-|Open-weight 7B model|Y
mistral/open-mixtral-8x22b|mixtral-8x22b|Mixtral 8x22B|C|2.0,6.0|65536,65536|TJ|-|Large open-weight MoE model|Y
mistral/open-mixtral-8x7b|mixtral-8x7b|Mixtral 8x7B|C|0.70,0.70|32768,32768|TJ|-|Open-weight MoE model|Y
mistral/pixtral-12b|pixtral-12b|Mistral: Pixtral 12B|C|0.0000,0.0000|8192,1024|VST|-|Efficient 12B vision model from Mistral|Y
mistral/pixtral-12b-2409|pixtral-12b|Pixtral 12B|C|0.15,0.15|131072,131072|VTJ|-|Vision model with 12B parameters|Y
mistral/pixtral-large|pixtral-large|Mistral: Pixtral Large|C|0.0000,0.0000|64000,2048|VST|-|Mistral native multimodal model for advanced vision tasks|Y
mistral/pixtral-large-2411|pixtral-large|Pixtral Large|C|2.0,6.0|131072,131072|VTJS|-|Vision-enabled large model|Y
mistral/voxtral-mini-latest|voxtral-mini|Voxtral Mini|C|0.12,|0,0|A|-|Speech transcription, $0.002/minute|N

# =============================================================================
# DEEPSEEK - Direct API (23 models)
# =============================================================================
deepseek/deepseek-chat|deepseek-v3|DeepSeek V3|C|0.27,1.1,0.07|128000,8192|VTJSC|-|671B MoE model, best performance at lowest cost|Y
deepseek/deepseek-chat-v2|deepseek-v2|DeepSeek V2|C|0.14,0.28|32768,4096|TJS|-|Previous generation model|Y
deepseek/deepseek-chat-v2.5|deepseek-v2.5|DeepSeek V2.5|C|0.14,0.28|32768,8192|TJS|-|Combined chat and code capabilities|Y
deepseek/deepseek-chat-v3-0324|deepseek-v3-0324|DeepSeek V3 0324|C|0.27,1.1,0.07|65536,8192|VTJSC|-|DeepSeek V3 March 2024 version|Y
deepseek/deepseek-chat-v3.1|deepseek-chat-v3.1|DeepSeek: DeepSeek V3.1|C|0.0000,0.0000|32768,7168|JKST|-|DeepSeek-V3.1 is a large hybrid reasoning model (671B parameters, 37B active) th|Y
deepseek/deepseek-coder|deepseek-coder|DeepSeek Coder|C|0.14,0.28,0.04|65536,8192|TJS|-|Specialized code generation model|Y
deepseek/deepseek-prover-v2|deepseek-prover-v2|DeepSeek: DeepSeek Prover V2|C|0.0000,0.0000|163840,40960|J|-|DeepSeek Prover V2 is a 671B parameter model, speculated to be geared towards lo|Y
deepseek/deepseek-r1|deepseek-r1|DeepSeek: R1|C|0.0000,0.0000|163840,40960|JKST|-|DeepSeek R1 is here: Performance on par with [OpenAI o1](/openai/o1), but open-s|Y
deepseek/deepseek-r1-0528|deepseek-r1-0528|DeepSeek: R1 0528|C|0.0000,0.0000|163840,65536|JKST|-|May 28th update to the [original DeepSeek R1](/deepseek/deepseek-r1) Performance|Y
deepseek/deepseek-r1-0528-qwen3-8b|deepseek-r1-0528-qwe|DeepSeek: DeepSeek R1 0528 Qwen3 8B|C|0.0000,0.0000|128000,32000|K|-|DeepSeek-R1-0528 is a lightly upgraded release of DeepSeek R1 that taps more com|N
deepseek/deepseek-r1-0528:free|deepseek-r1-0528:fre|DeepSeek: R1 0528 (free)|C|-|163840,40960|K|-|May 28th update to the [original DeepSeek R1](/deepseek/deepseek-r1) Performance|N
deepseek/deepseek-r1-distill-llama-70b|deepseek-r1-distill-|DeepSeek: R1 Distill Llama 70B|C|0.0000,0.0000|131072,131072|JKST|-|DeepSeek R1 Distill Llama 70B is a distilled large language model based on [Llam|Y
deepseek/deepseek-r1-distill-qwen-14b|deepseek-r1-distill-|DeepSeek: R1 Distill Qwen 14B|C|0.0000,0.0000|32768,16384|JKS|-|DeepSeek R1 Distill Qwen 14B is a distilled large language model based on [Qwen|N
deepseek/deepseek-r1-distill-qwen-32b|deepseek-r1-distill-|DeepSeek: R1 Distill Qwen 32B|C|0.0000,0.0000|131072,32768|JKS|-|DeepSeek R1 Distill Qwen 32B is a distilled large language model based on [Qwen|N
deepseek/deepseek-reasoner|deepseek-r1|DeepSeek R1|C|0.55,2.2,0.14|128000,8192|VTJSK|-|Reasoning model with chain-of-thought|Y
deepseek/deepseek-reasoner-0528|deepseek-r1-0528|DeepSeek R1 0528|C|0.55,2.2,0.14|128000,8192|VTJSK|-|DeepSeek R1 May 2025 version|Y
deepseek/deepseek-v3-2-20260104|deepseek-v3-2|DeepSeek V3.2|C|0.0000,0.0000|64000,4096|SVTK|-|Advanced reasoning with o1-style thinking|Y
deepseek/deepseek-v3.1-terminus|deepseek-v3.1-termin|DeepSeek: DeepSeek V3.1 Terminus|C|0.0000,0.0000|163840,40960|JKST|-|DeepSeek-V3.1 Terminus is an update to [DeepSeek V3.1](/deepseek/deepseek-chat-v|Y
deepseek/deepseek-v3.1-terminus:exacto|deepseek-v3.1-termin|DeepSeek: DeepSeek V3.1 Terminus (exacto)|C|0.0000,0.0000|163840,40960|JKST|-|DeepSeek-V3.1 Terminus is an update to [DeepSeek V3.1](/deepseek/deepseek-chat-v|Y
deepseek/deepseek-v3.2|deepseek-v3.2|DeepSeek: DeepSeek V3.2|C|0.0000,0.0000|163840,65536|JKST|-|DeepSeek-V3.2 is a large language model designed to harmonize high computational|Y
deepseek/deepseek-v3.2-exp|deepseek-v3.2-exp|DeepSeek: DeepSeek V3.2 Exp|C|0.0000,0.0000|163840,65536|JKST|-|DeepSeek-V3.2-Exp is an experimental large language model released by DeepSeek a|Y
deepseek/deepseek-v3.2-speciale|deepseek-v3.2-specia|DeepSeek: DeepSeek V3.2 Speciale|C|0.0000,0.0000|163840,65536|JKS|-|DeepSeek-V3.2-Speciale is a high-compute variant of DeepSeek-V3.2 optimized for|N
deepseek/deepseek-vl2|deepseek-vl2|DeepSeek: DeepSeek-VL2|C|0.0000,0.0000|8192,2048|VST|-|DeepSeek vision language model with multi-image support|Y

# =============================================================================
# COHERE - Direct API (17 models)
# =============================================================================
cohere/aya-expanse-32b|aya-expanse-32b|Aya Expanse 32B|C|0.50,1.5|128000,4096|TJ|-|Multilingual model supporting 23 languages|Y
cohere/aya-expanse-8b|aya-expanse-8b|Aya Expanse 8B|C|0.05,0.10|8192,4096|TJ|-|Efficient multilingual model|Y
cohere/command|command|Command|C|1.0,2.0|4096,4096|TJ|-|Legacy command model|Y
cohere/command-a|command-a|Cohere: Command A|C|0.0000,0.0000|256000,8192|JS|-|Command A is an open-weights 111B parameter model with a 256k context window foc|Y
cohere/command-a-03-2025|command-a|Command A|C|2.5,10.0|256000,8192|VTJS|-|Agent-focused model with 256K context|Y
cohere/command-light|command-light|Command Light|C|0.30,0.60|4096,4096|TJ|-|Legacy lightweight command model|Y
cohere/command-r-08-2024|command-r|Command R|C|0.15,0.60|128000,4096|VTJS|-|Balanced performance for RAG and tool use|Y
cohere/command-r-plus-08-2024|command-r-plus|Command R+|C|2.5,10.0|128000,4096|VTJS|-|Most capable Command model for complex tasks|Y
cohere/command-r7-plus-20260110|cohere-command-r7-plus|Cohere Command R7 Plus|C|0.0000,0.0000|128000,4096|TJ|-|Latest Cohere advanced model|Y
cohere/command-r7b-12-2024|command-r7b|Command R7B|C|0.04,0.15|128000,4096|TJS|-|Smallest Command model, highly efficient|Y
cohere/embed-english-light-v3.0|embed-english-light|Embed English Light v3|C|0.10,|512,384|E|-|Lightweight English embeddings|N
cohere/embed-english-v3.0|embed-english-v3|Embed English v3|C|0.10,|512,1024|E|-|English text embeddings|N
cohere/embed-multilingual-light-v3.0|embed-multilingual-light|Embed Multilingual Light v3|C|0.10,|512,384|E|-|Lightweight multilingual embeddings|N
cohere/embed-multilingual-v3.0|embed-multilingual-v3|Embed Multilingual v3|C|0.10,|512,1024|E|-|Multilingual text embeddings|N
cohere/rerank-english-v3.0|rerank-english-v3|Rerank English v3|C|2.0,|4096,0|R|-|English document reranking|Y
cohere/rerank-multilingual-v3.0|rerank-multilingual-v3|Rerank Multilingual v3|C|2.0,|4096,0|R|-|Multilingual document reranking|Y
cohere/rerank-v3.5|rerank-v3.5|Rerank v3.5|C|2.0,|4096,0|R|-|Document reranking model, $2/1K searches|Y

# =============================================================================
# GROQ - Fast Inference (33 models)
# =============================================================================
groq/deepseek-r1-distill-llama-70b|deepseek-r1-70b|deepseek-r1-llama-groq|Groq: DeepSeek-R1 Distill Llama 70B|C|0.0000,0.0000|64000,8000|KJT|-|DeepSeek R1 distilled to Llama 70B - reasoning capabilities on Groq LPU|N
groq/deepseek-r1-distill-qwen-32b|deepseek-r1-32b|DeepSeek R1 Distill 32B|C|0.69,0.69|131072,16384|TJK|-|DeepSeek R1 distilled to Qwen 32B|N
groq/distil-whisper-large-v3-en|distil-whisper|distil-whisper-v3-groq|Groq: Distil Whisper Large v3 EN|C|0,0|448,2048|V|-|Distilled Whisper model for English speech - ultra-lightweight on Groq LPU|Y
groq/gemma-7b-it|gemma-7b|gemma-7b-it-groq|Groq: Gemma 7B Instruct|C|0.0000,0.0000|8192,2048|JT|-|Google Gemma 7B instruction-tuned model - lightweight and fast on Groq LPU|Y
groq/gemma2-9b-it|gemma2-9b|Gemma 2 9B|C|0.20,0.20|8192,8192|TJ|-|Google's Gemma 2 9B on Groq|Y
groq/groq/compound|compound|Groq Compound|C|0,0|131072,8192|TJS|-|Compound AI with built-in tools|Y
groq/groq/compound-mini|compound-mini|Groq Compound Mini|C|0,0|131072,8192|TJS|-|Lightweight compound AI|Y
groq/llama-3.1-70b-versatile|llama-3.1-70b|Llama 3.1 70B|C|0.59,0.79|131072,8192|TJS|-|Llama 3.1 70B on Groq hardware|Y
groq/llama-3.1-8b-instant|llama-3.1-8b|Llama 3.1 8B|C|0.05,0.08|131072,8192|TJS|-|Fast and efficient Llama 3.1 8B|Y
groq/llama-3.2-11b-vision-preview|llama-3.2-11b-vision|Llama 3.2 11B Vision|C|0.18,0.18|131072,8192|VTJ|-|Compact vision model|Y
groq/llama-3.2-1b-preview|llama-3.2-1b|Llama 3.2 1B|C|0.04,0.04|131072,8192|TJ|-|Tiny Llama model for edge devices|Y
groq/llama-3.2-3b-preview|llama-3.2-3b|Llama 3.2 3B|C|0.06,0.06|131072,8192|TJ|-|Smallest Llama 3.2 model|Y
groq/llama-3.2-90b-vision-preview|llama-3.2-90b-vision|Llama 3.2 90B Vision|C|0.90,0.90|131072,8192|VTJ|-|Vision-enabled Llama 3.2 90B|Y
groq/llama-3.3-70b-specdec|llama-3.3-70b-spec|Llama 3.3 70B SpecDec|C|0.59,0.99|8192,8192|TJS|-|Llama 3.3 70B with speculative decoding|Y
groq/llama-3.3-70b-versatile|llama-3.3-70b|Llama 3.3 70B|C|0.59,0.79|131072,32768|TJS|-|Meta's Llama 3.3 70B, ultra-fast inference|Y
groq/llama-4-scout-17b-16e-instruct|llama-4-scout|Llama 4 Scout 17B|C|0.11,0.34|131072,8192|VTJS|-|Meta's Llama 4 Scout 17B on Groq hardware|Y
groq/llama2-70b-4096|llama-70b|llama2-70b-groq|Groq: Llama 2 70B|C|0.0000,0.0000|4096,2048|JT|-|Meta Llama 2 70B parameter model optimized for Groq LPU fast inference|Y
groq/llama2-70b-chat|llama-70b-chat|llama2-70b-chat-groq|Groq: Llama 2 70B Chat|C|0.0000,0.0000|4096,2048|JT|-|Instruction and chat-optimized Llama 2 70B - excellent for conversational AI on|Y
groq/meta-llama/llama-4-maverick-17b-128e-instruct|llama-4-maverick|Llama 4 Maverick 17B|C|0.20,0.60|131072,8192|VTJS|-|Meta's Llama 4 Maverick 17B on Groq|Y
groq/meta-llama/llama-guard-4-12b|llama-guard-4|Llama Guard 4 12B|C|0.20,0.20|131072,8192|M|-|Content moderation model|N
groq/mixtral-8x22b-32768|mixtral-8x22b|mixtral-8x22b-groq|Groq: Mixtral 8x22b|C|0.0000,0.0000|32768,4096|JT|-|Larger Mixtral 8x22b with increased model depth - exceptional performance on Gro|Y
groq/mixtral-8x22b-instruct|mixtral-8x22b-instruct|mixtral-8x22b-inst|Groq: Mixtral 8x22b Instruct|C|0.0000,0.0000|32768,4096|JT|-|Instruction-tuned Mixtral 8x22b - optimal for complex reasoning on Groq LPU|Y
groq/mixtral-8x7b-32768|mixtral-8x7b|mixtral-8x7b-groq|Groq: Mixtral 8x7b|C|0.0000,0.0000|32768,4096|JT|-|Mixtral 8x7b mixture of experts model optimized for Groq LPU - ultra-fast infere|Y
groq/mixtral-8x7b-instruct|mixtral-8x7b-instruct|mixtral-8x7b-inst|Groq: Mixtral 8x7b Instruct|C|0.0000,0.0000|32768,4096|JT|-|Instruction-tuned Mixtral 8x7b for Groq LPU - optimized for chat and instruction|Y
groq/moonshotai/kimi-k2-instruct-0905|kimi-k2|Kimi K2 Instruct|C|0.35,1.4|262144,16384|TJS|-|Moonshot Kimi K2 on Groq|Y
groq/openai/gpt-oss-120b|gpt-oss-120b|GPT-OSS 120B|C|0.59,0.79|131072,16384|TJS|-|OpenAI open-weight 120B MoE model|Y
groq/openai/gpt-oss-20b|gpt-oss-20b|GPT-OSS 20B|C|0.40,0.40|131072,16384|TJS|-|OpenAI open-weight 20B model|Y
groq/openai/gpt-oss-safeguard-20b|gpt-oss-safeguard|GPT-OSS Safeguard 20B|C|0.40,0.40|131072,16384|TJS|-|Safety-focused 20B model|Y
groq/qwen-qwq-32b|qwq-32b|Qwen QWQ 32B|C|0.29,0.39|131072,16384|TJK|-|Alibaba's Qwen QWQ reasoning model|N
groq/qwen/qwen3-32b|qwen3-32b|Qwen3 32B|C|0.29,0.59|131072,16384|TJS|-|Alibaba Qwen3 32B on Groq|Y
groq/t5-base|t5-base|t5-base-groq|Groq: T5 Base|C|0.0000,0.0000|512,768|JT|-|Google T5 base text-to-text transfer transformer - lightweight sequence tasks|Y
groq/whisper-large-v3|whisper-large|whisper-v3-groq|Groq: Whisper Large v3|C|0.0000,0.0000|448,2048|V|-|OpenAI Whisper large v3 speech recognition - optimized for Groq LPU|Y
groq/whisper-large-v3-turbo|whisper-v3-turbo|Whisper Large v3 Turbo|C|0.04,|0,0|A|-|Fast speech-to-text, $0.04/hour|N

# =============================================================================
# CEREBRAS - Ultra-fast Inference (7 models)
# =============================================================================
cerebras/deepseek-r1-distill-llama-70b|deepseek-r1-70b|DeepSeek R1 Distill 70B|C|0.85,1.2|8192,8192|TJK|-|DeepSeek R1 reasoning distilled to Llama|N
cerebras/llama-3.3-70b|llama-3.3-70b|Llama 3.3 70B|C|0.85,1.2|8192,8192|TJS|-|Latest Llama 3.3 on Cerebras WSE|Y
cerebras/llama-4-scout-17b-16e-instruct|llama-4-scout|Llama 4 Scout 17B|C|0.15,0.60|131072,8192|VTJS|-|Meta Llama 4 Scout on Cerebras WSE|Y
cerebras/llama3.1-70b|llama-3.1-70b|Llama 3.1 70B|C|0.60,0.60|8192,8192|TJS|-|Meta Llama 3.1 70B on Cerebras WSE|Y
cerebras/llama3.1-8b|llama-3.1-8b|Llama 3.1 8B|C|0.10,0.10|8192,8192|TJS|-|Meta Llama 3.1 8B on Cerebras WSE, ultra-fast|Y
cerebras/qwen-2.5-32b|qwen-2.5-32b|Qwen 2.5 32B|C|0.20,0.20|8192,8192|TJS|-|Alibaba Qwen 2.5 32B on Cerebras|Y
cerebras/qwen-2.5-coder-32b|qwen-coder-32b|Qwen 2.5 Coder 32B|C|0.20,0.20|8192,8192|TJS|-|Qwen 2.5 Coder for code generation|Y

# =============================================================================
# SAMBANOVA - Fast Inference (20 models)
# =============================================================================
sambanova/ALLaM-7B-Instruct-preview|allam-7b|ALLaM 7B Arabic|C|0.10,0.20|4096,4096|TJ|-|Arabic language model preview|Y
sambanova/DeepSeek-R1|deepseek-r1|DeepSeek R1|C|0.60,1.2|4096,4096|TJK|-|DeepSeek R1 reasoning model|N
sambanova/DeepSeek-R1-Distill-Llama-70B|deepseek-r1-70b|DeepSeek R1 Distill 70B|C|0.40,0.80|4096,4096|TJK|-|DeepSeek R1 distilled to Llama 70B|N
sambanova/DeepSeek-V3-0324|deepseek-v3|DeepSeek V3 0324|C|0.40,0.80|65536,16384|TJS|-|DeepSeek V3 with function calling|Y
sambanova/E5-Mistral-7B-Instruct|e5-mistral-7b|E5 Mistral 7B Embed|C|0.10,|4096,4096|E|-|Embedding model|N
sambanova/Llama-3.2-11B-Vision-Instruct|llama-3.2-11b-vision|Llama 3.2 11B Vision|C|0.15,0.30|4096,4096|VTJS|-|Compact Llama 3.2 with vision|Y
sambanova/Llama-3.2-1B-Instruct|llama-3.2-1b|Llama 3.2 1B|C|0.05,0.10|131072,4096|TJ|-|Smallest Llama 3.2 model|Y
sambanova/Llama-3.2-3B-Instruct|llama-3.2-3b|Llama 3.2 3B|C|0.08,0.16|131072,4096|TJ|-|Compact Llama 3.2 model|Y
sambanova/Llama-3.2-90B-Vision-Instruct|llama-3.2-90b-vision|Llama 3.2 90B Vision|C|0.60,1.2|4096,4096|VTJS|-|Llama 3.2 90B with vision capabilities|Y
sambanova/Llama-4-Maverick-17B-128E-Instruct|llama-4-maverick|Llama 4 Maverick 17B|C|0.20,0.60|131072,16384|VTJS|-|Meta Llama 4 Maverick 400B MoE|Y
sambanova/Llama-4-Scout-17B-16E-Instruct|llama-4-scout|Llama 4 Scout 17B|C|0.15,0.40|131072,16384|VTJS|-|Meta Llama 4 Scout 109B MoE|Y
sambanova/Meta-Llama-3.1-405B-Instruct|llama-3.1-405b|Llama 3.1 405B Instruct|C|5.0,10.0|4096,4096|TJS|-|Largest Llama model, 405B parameters|Y
sambanova/Meta-Llama-3.1-70B-Instruct|llama-3.1-70b|Llama 3.1 70B Instruct|C|0.40,0.80|4096,4096|TJS|-|Meta Llama 3.1 70B on SambaNova|Y
sambanova/Meta-Llama-3.1-8B-Instruct|llama-3.1-8b|Llama 3.1 8B Instruct|C|0.10,0.20|4096,4096|TJS|-|Meta Llama 3.1 8B on SambaNova|Y
sambanova/Meta-Llama-3.3-70B-Instruct|llama-3.3-70b|Llama 3.3 70B Instruct|C|0.40,0.80|4096,4096|TJS|-|Meta Llama 3.3 70B on SambaNova RDU|Y
sambanova/QwQ-32B|qwq-32b|QwQ 32B|C|0.20,0.40|4096,4096|TJK|-|Alibaba QwQ reasoning model|N
sambanova/Qwen2.5-72B-Instruct|qwen-2.5-72b|Qwen 2.5 72B Instruct|C|0.40,0.80|4096,4096|TJS|-|Alibaba Qwen 2.5 72B on SambaNova|Y
sambanova/Qwen2.5-Coder-32B-Instruct|qwen-coder-32b|Qwen 2.5 Coder 32B|C|0.20,0.40|4096,4096|TJS|-|Qwen 2.5 specialized for code|Y
sambanova/Qwen3-32B|qwen3-32b|Qwen3 32B|C|0.20,0.40|131072,16384|TJS|-|Alibaba Qwen3 32B multilingual|Y
sambanova/gpt-oss-120b|gpt-oss-120b|GPT-OSS 120B|C|0.59,0.79|131072,16384|TJS|-|OpenAI open-weight 120B MoE|Y

# =============================================================================
# FIREWORKS - Fast Inference (250 models)
# =============================================================================
fireworks/chronos-hermes-13b-v2|chronos-hermes-13b-v2|Chronos Hermes 13B v2|C|0.20,0.20|4096,4096|J|-|(chronos-13b-v2 + Nous-Hermes-Llama2-13b) 75/25 merge. This offers the imaginati|Y
fireworks/code-llama-13b|code-llama-13b|Code Llama 13B|C|0.20,0.20|16384,16384|J|-|Code Llama is a collection of pretrained and fine-tuned Large Language Models ra|Y
fireworks/code-llama-13b-instruct|code-llama-13b|Code Llama 13B Instruct|C|0.20,0.20|16384,16384|J|-|Code Llama is a collection of pretrained and fine-tuned Large Language Models ra|Y
fireworks/code-llama-13b-python|code-llama-13b-python|Code Llama 13B Python|C|0.20,0.20|16384,16384|J|-|Code Llama is a collection of pretrained and fine-tuned Large Language Models ra|Y
fireworks/code-llama-34b|code-llama-34b|Code Llama 34B|C|0.10,0.10|16384,16384|J|-|Code Llama is a collection of pretrained and fine-tuned Large Language Models ra|Y
fireworks/code-llama-34b-instruct|code-llama-34b|Code Llama 34B Instruct|C|0.10,0.10|16384,16384|J|-|Code Llama is a collection of pretrained and fine-tuned Large Language Models ra|Y
fireworks/code-llama-34b-python|code-llama-34b-python|Code Llama 34B Python|C|0.10,0.10|16384,16384|J|-|Code Llama is a collection of pretrained and fine-tuned Large Language Models ra|Y
fireworks/code-llama-70b|code-llama-70b|Code Llama 70B|C|0.90,0.90|4096,4096|J|-|Code Llama is a collection of pretrained and fine-tuned Large Language Models ra|Y
fireworks/code-llama-70b-instruct|code-llama-70b|Code Llama 70B Instruct|C|0.90,0.90|4096,4096|J|-|Code Llama is a collection of pretrained and fine-tuned Large Language Models ra|Y
fireworks/code-llama-70b-python|code-llama-70b-python|Code Llama 70B Python|C|0.90,0.90|4096,4096|J|-|Code Llama is a collection of pretrained and fine-tuned Large Language Models ra|Y
fireworks/code-llama-7b|code-llama-7b|Code Llama 7B|C|0.20,0.20|16384,16384|J|-|Code Llama is a collection of pretrained and fine-tuned Large Language Models ra|Y
fireworks/code-llama-7b-instruct|code-llama-7b|Code Llama 7B Instruct|C|0.20,0.20|16384,16384|J|-|Code Llama is a collection of pretrained and fine-tuned Large Language Models ra|Y
fireworks/code-llama-7b-python|code-llama-7b-python|Code Llama 7B Python|C|0.20,0.20|16384,16384|J|-|Code Llama is a collection of pretrained and fine-tuned Large Language Models ra|Y
fireworks/code-qwen-1p5-7b|code-qwen-1p5-7b|CodeQwen 1.5 7B|C|0.20,0.20|65536,16384|J|-|CodeQwen1.5 is based on Qwen1.5, a language model series including decoder langu|Y
fireworks/codegemma-2b|codegemma-2b|CodeGemma 2B|C|0.20,0.20|8192,8192|J|-|CodeGemma is a collection of lightweight open code models built on top of Gemma.|Y
fireworks/codegemma-7b|codegemma-7b|CodeGemma 7B|C|0.20,0.20|8192,8192|J|-|CodeGemma is a collection of lightweight open code models built on top of Gemma.|Y
fireworks/cogito-v1-preview-llama-3b|cogito-v1-preview-llama-3b|Cogito v1 Preview Llama 3B|C|0.10,0.10|131072,16384|TJ|-|The Cogito LLMs are instruction tuned generative models that are also hybrid rea|Y
fireworks/cogito-v1-preview-llama-70b|cogito-v1-preview-llama-70b|Cogito v1 Preview Llama 70B|C|0.90,0.90|131072,16384|TJ|-|The Cogito LLMs are instruction tuned generative models that are also hybrid rea|Y
fireworks/cogito-v1-preview-llama-8b|cogito-v1-preview-llama-8b|Cogito v1 Preview Llama 8B|C|0.20,0.20|131072,16384|TJ|-|The Cogito LLMs are instruction tuned generative models that are also hybrid rea|Y
fireworks/cogito-v1-preview-qwen-14b|cogito-v1-preview-qwen-14b|Cogito v1 Preview Qwen 14B|C|0.10,0.10|131072,16384|TJ|-|The Cogito LLMs are instruction tuned generative models that are also hybrid rea|Y
fireworks/cogito-v1-preview-qwen-32b|cogito-v1-preview-qwen-32b|Cogito v1 Preview Qwen 32B|C|0.90,0.90|131072,16384|TJ|-|The Cogito LLMs are instruction tuned generative models that are also hybrid rea|Y
fireworks/deepseek-coder-1b-base|deepseek-coder-1b-base|DeepSeek Coder 1.3B Base|C|0.90,0.90|16384,16384|J|-|DeepSeek Coder is composed of a series of code language models, each trained fro|Y
fireworks/deepseek-coder-33b-instruct|deepseek-coder-33b|DeepSeek Coder 33B Instruct|C|0.10,0.10|16384,16384|J|-|Deepseek Coder is composed of a series of code language models, each trained fro|Y
fireworks/deepseek-coder-7b-base|deepseek-coder-7b-base|DeepSeek Coder 7B Base|C|0.20,0.20|4096,4096|J|-|Deepseek Coder is composed of a series of code language models, each trained fro|Y
fireworks/deepseek-coder-7b-base-v1p5|deepseek-coder-7b-base-v1p5|DeepSeek Coder 7B Base v1.5|C|0.20,0.20|4096,4096|J|-|The Deepseek Coder 7B Base v1.5 LLM is pre-trained from Deepseek 7B on 2T tokens|Y
fireworks/deepseek-coder-7b-instruct-v1p5|deepseek-coder-7b-v1p5|DeepSeek Coder 7B Instruct v1.5|C|0.20,0.20|4096,4096|J|-|Deepseek-Coder-7B-Instruct-v1.5 is pre-trained from Deepseek-LLM 7B on 2T tokens|Y
fireworks/deepseek-coder-v2-instruct|deepseek-coder-v2|DeepSeek Coder V2 Instruct|C|0.90,0.90|32768,16384|J|-|DeepSeek Coder V2 Instruct is a 236-billion-parameter open-source Mixture-of-Exp|Y
fireworks/deepseek-coder-v2-lite-base|deepseek-coder-v2-lite-base|DeepSeek Coder V2 Lite Base|C|0.90,0.90|163840,16384|J|-|DeepSeek-Coder-V2 is an open-source Mixture-of-Experts (MoE) code language model|Y
fireworks/deepseek-coder-v2-lite-instruct|deepseek-coder-v2-lite|DeepSeek Coder V2 Lite Instruct|C|0.90,0.90|163840,16384|J|-|DeepSeek Coder V2 Lite Instruct is a 16-billion-parameter open-source Mixture-of|Y
fireworks/deepseek-prover-v2|deepseek-prover-v2|DeepSeek Prover V2|C|0.90,0.90|163840,16384|J|-|DeepSeek-Prover-V2, an open-source large language model designed for formal theo|Y
fireworks/deepseek-r1|deepseek-r1|DeepSeek R1 (Fast)|C|3.0,8.0|163840,16384|J|-|DeepSeek R1 (Fast) is the speed-optimized serverless deployment of DeepSeek-R1.|Y
fireworks/deepseek-r1-0528|deepseek-r1-0528|Deepseek R1 05/28|C|3.0,8.0|163840,16384|TJ|-|05/28 updated checkpoint of Deepseek R1. Its overall performance is now approach|Y
fireworks/deepseek-r1-0528-distill-qwen3-8b|deepseek-r1-0528-distill-qwen-3-8b|DeepSeek R1 0528 Distill Qwen3 8B|C|0.20,0.20|131072,16384|TJ|-|We distilled the chain-of-thought from DeepSeek-R1-0528 to post-train Qwen3 8B B|Y
fireworks/deepseek-r1-basic|deepseek-r1|DeepSeek R1 (Basic)|C|3.0,8.0|163840,16384|J|-|DeepSeek R1 (Basic) is the cost-optimized serverless deployment of DeepSeek-R1.|Y
fireworks/deepseek-r1-distill-llama-70b|deepseek-r1-distill-llama-70b|DeepSeek R1 Distill Llama 70B|C|0.90,0.90|131072,16384|J|-|Llama 70B distilled with reasoning from Deepseek R1|Y
fireworks/deepseek-r1-distill-llama-8b|deepseek-r1-distill-llama-8b|DeepSeek R1 Distill Llama 8B|C|0.20,0.20|131072,16384|J|-|Llama 8B distilled with reasoning from Deepseek R1|Y
fireworks/deepseek-r1-distill-qwen-14b|deepseek-r1-distill-qwen-14b|DeepSeek R1 Distill Qwen 14B|C|0.10,0.10|131072,16384|J|-|Qwen 14B distilled with reasoning from Deepseek R1|Y
fireworks/deepseek-r1-distill-qwen-1p5b|deepseek-r1-distill-qwen-1p5b|DeepSeek R1 Distill Qwen 1.5B|C|3.0,8.0|131072,16384|J|-|Qwen 1.5B distilled with reasoning from Deepseek R1|Y
fireworks/deepseek-r1-distill-qwen-32b|deepseek-r1-distill-qwen-32b|DeepSeek R1 Distill Qwen 32B|C|0.90,0.90|131072,16384|J|-|Qwen 32B distilled with reasoning from Deepseek R1|Y
fireworks/deepseek-r1-distill-qwen-7b|deepseek-r1-distill-qwen-7b|DeepSeek R1 Distill Qwen 7B|C|0.20,0.20|131072,16384|J|-|Qwen 7B distilled with reasoning from Deepseek R1|Y
fireworks/deepseek-v2-lite-chat|deepseek-v2-lite|DeepSeek V2 Lite Chat|C|0.90,0.90|163840,16384|J|-|DeepSeek-V2, a strong Mixture-of-Experts (MoE) language model characterized by e|Y
fireworks/deepseek-v2p5|deepseek-v2p5|DeepSeek V2.5|C|0.90,0.90|32768,16384|J|-|DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek|Y
fireworks/deepseek-v3|deepseek-v3|DeepSeek V3|C|0.90,0.90|131072,16384|TJ|-|A a strong Mixture-of-Experts (MoE) language model with 671B total parameters wi|Y
fireworks/deepseek-v3-0324|deepseek-v3-0324|Deepseek V3 03-24|C|0.90,0.90|163840,16384|TJ|-|A strong Mixture-of-Experts (MoE) language model with 671B total parameters with|Y
fireworks/deepseek-v3p1|deepseek-v3p1|DeepSeek V3.1|C|0.90,0.90|163840,16384|TJ|-|DeepSeek-V3.1 is post-trained on the top of DeepSeek-V3.1-Base, which is built u|Y
fireworks/deepseek-v3p1-terminus|deepseek-v3p1-terminus|DeepSeek V3.1 Terminus|C|0.90,0.90|163840,16384|TJ|-|DeepSeek-V3.1-Terminus is an updated version of DeepSeek-V3.1 with enhanced lang|Y
fireworks/deepseek-v3p2|deepseek-v3p2|Deepseek v3.2|C|0.90,0.90|163840,16384|TJ|-|Model from Deepseek that harmonizes high computational efficiency with superior|Y
fireworks/devstral-small-2-24b-instruct-2512|devstral-small-2-24b-2512|Devstral Small 2 24B Instruct 2512|C|0.10,0.10|4096,4096|VT|-|Devstral is an agentic LLM for software engineering tasks. Devstral Small 2 exce|Y
fireworks/devstral-small-2505|devstral-small-2505|Devstral-Small-2505|C|0.20,0.20|131072,16384|J|-|Devstral is an agentic LLM for software engineering tasks built under a collabor|Y
fireworks/dolphin-2-9-2-qwen2-72b|dolphin-2-9-2-qwen2-72b|Dolphin 2.9.2 Qwen2 72B|C|0.90,0.90|131072,16384|J|-|Dolphin 2.9.2 Qwen2 72B is a fine-tuned version of the Qwen2 72B Large Language|Y
fireworks/dolphin-2p6-mixtral-8x7b|dolphin-2p6-mixtral-8x7b|Dolphin 2.6 Mixtral 8x7b|C|0.20,0.20|32768,16384|J|-|Dolphin 2.6 Mixtral 8x7b is a fine-tuned version of the Mixtral-8x7b Large Langu|Y
fireworks/eagle-llama-v3-8b-instruct-v1|eagle-llama-v3-8b-v1|EAGLE Llama 3 8B V1|C|0.20,0.20|4096,4096|-|-|EAGLE draft model for Llama 3.x 8B instruct models|Y
fireworks/eagle-llama-v3-8b-instruct-v2|eagle-llama-v3-8b-v2|EAGLE Llama 3 8B V2|C|0.20,0.20|4096,4096|-|-|EAGLE draft model for Llama 3.x 8B instruct models|Y
fireworks/eagle-qwen-v2p5-3b-instruct-v2|eagle-qwen-v2p5-3b-v2|EAGLE Qwen 2.5 3B Instruct V2|C|0.10,0.10|4096,4096|-|-|EAGLE draft model for Qwen 2.5 3B instruct models|Y
fireworks/eagle1-kimi-k2-instruct-0905-v0|eagle1-kimi-k2-0905-v0|EAGLE1 kimi-k2-instruct-0905 v0|C|0.35,1.4|4096,4096|-|-|EAGLE1 for kimi-k2-instruct-0905 v0|Y
fireworks/eagle3-kimi-k2-instruct-0905-v0|eagle3-kimi-k2-0905-v0|EAGLE3 kimi-k2-instruct-0905 v0|C|0.35,1.4|4096,4096|-|-|EAGLE3 for kimi-k2-instruct-0905 v0|Y
fireworks/ernie-4p5-21b-a3b-pt|ernie-4p5-21b-a3b-pt|ERNIE-4.5-21B-A3B-PT|C|0.10,0.10|131072,16384|J|-|ERNIE-4.5-21B-A3B is a text MoE Post-trained model, with 21B total parameters an|Y
fireworks/ernie-4p5-300b-a47b-pt|ernie-4p5-300b-a47b-pt|ERNIE-4.5-300B-A47B-PT|C|0.20,0.20|131072,16384|J|-|ERNIE-4.5-300B-A47B is a text MoE Post-trained model, with 300B total parameters|Y
fireworks/fare-20b|fare-20b|FARE-20B|C|0.20,0.20|131072,16384|J|-|FARE-20B is a multi-task evaluator fine-tuned from gpt-oss-20B. It’s trained on|Y
fireworks/firefunction-v1|firefunction-v1|FireFunction V1|C|0.20,0.20|32768,16384|T|-|Fireworks' open-source function calling model.|Y
fireworks/firefunction-v2|firefunction-v2|FireFunction V2|C|0.20,0.20|4096,4096|T|-|Fireworks' latest and most performant function-calling model. Firefunction-v2 is|Y
fireworks/firesearch-ocr-v6|firesearch-ocr-v6|Firesearch OCR V6|C|0.20,0.20|8192,8192|VJ|-|OCR model provided by Fireworks|Y
fireworks/flux-1-dev-controlnet-union|flux-1-dev-controlnet-union|FLUX.1 [dev] ControlNet|C|0.20,0.20|4096,4096|-|-|Unified ControlNet for FLUX.1-dev model jointly released by researchers from Ins|Y
fireworks/flux-1-dev-fp8|flux-1-dev-fp8|FLUX.1 [dev] FP8|C|0.03,|4096,0|I|-|FLUX.1 [dev] is a 12 billion parameter rectified flow transformer capable of gen|N
fireworks/flux-1-schnell|flux-1-schnell|FLUX.1 [schnell]|C|0.03,|4096,0|I|-|FLUX.1 [schnell] is a 12 billion parameter rectified flow transformer capable of|N
fireworks/flux-1-schnell-fp8|flux-1-schnell-fp8|FLUX.1 [schnell] FP8|C|0.03,|4096,0|I|-|FLUX.1 [schnell] is a 12 billion parameter rectified flow transformer capable of|N
fireworks/flux-kontext-max|flux-kontext-max|Flux Kontext Max|C|0.05,|4096,0|I|-|FLUX Kontext Max is Black Forest Labs' new premium model that brings maximum per|N
fireworks/flux-kontext-pro|flux-kontext-pro|Flux Kontext Pro|C|0.05,|4096,0|I|-|FLUX Kontext Pro is a specialized model for generating contextually-aware images|N
fireworks/full-llama-v3p1-8b-instruct-8b-fp8|full-llama-1-8b-8b-fp8|Llama 3.1 8B Instruct FP8 [Full]|C|0.20,0.20|4096,4096|-|-|The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a|Y
fireworks/full-llama-v3p1-8b-instruct-8b-fp8-amd|full-llama-1-8b-8b-fp8-amd|Llama 3.1 8B Instruct FP8 AMD [Full]|C|0.20,0.20|4096,4096|-|-|The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a|Y
fireworks/gemma-2b-it|gemma-2b-it|Gemma 2B Instruct|C|0.20,0.20|8192,8192|J|-|Gemma is a family of lightweight, state-of-the-art open models from Google, buil|Y
fireworks/gemma-3-12b-it|gemma-3-12b-it|Gemma 3 12B Instruct|C|0.20,0.20|131072,16384|J|-|Gemma is a family of lightweight, state-of-the-art open models from Google, buil|Y
fireworks/gemma-3-27b-it|gemma-3-27b-it|Gemma 3 27B Instruct|C|0.20,0.20|131072,16384|J|-|Gemma 3 27B Instruct|Y
fireworks/gemma-3-4b-it|gemma-3-4b-it|Gemma 3 4B Instruct|C|0.10,0.10|131072,16384|J|-|Gemma is a family of lightweight, state-of-the-art open models from Google, buil|Y
fireworks/gemma-7b|gemma-7b|Gemma 7B|C|0.20,0.20|8192,8192|J|-|Gemma is a family of lightweight, state-of-the-art open models from Google, buil|Y
fireworks/gemma-7b-it|gemma-7b-it|Gemma 7B Instruct|C|0.20,0.20|8192,8192|J|-|Gemma is a family of lightweight, state-of-the-art open models from Google, buil|Y
fireworks/gemma2-9b-it|gemma2-9b-it|Gemma 2 9B Instruct|C|0.20,0.20|8192,8192|J|-|Gemma is a family of lightweight, state-of-the-art open models from Google, buil|Y
fireworks/glm-4p5|glm-4p5|GLM-4.5|C|0.35,0.35|131072,16384|TJ|-|The GLM-4.5 series models are foundation models designed for intelligent agents.|Y
fireworks/glm-4p5-air|glm-4p5-air|GLM-4.5-Air|C|0.35,0.35|131072,16384|TJ|-|The GLM-4.5 series models are foundation models designed for intelligent agents.|Y
fireworks/glm-4p5v|glm-4p5v|GLM-4.5V|C|0.35,0.35|131072,16384|VTJ|-|GLM-4.5V is based on ZhipuAI’s next-generation flagship text foundation model GL|Y
fireworks/glm-4p6|glm-4p6|GLM-4.6|C|0.35,0.35|202752,16384|TJ|-|As the latest iteration in the GLM series, GLM-4.6 achieves comprehensive enhanc|Y
fireworks/glm-4p7|glm-4p7|GLM-4.7|C|0.35,0.35|202752,16384|TJ|-|GLM-4.7 is a next-generation general-purpose model optimized for coding, reasoni|Y
fireworks/gpt-oss-120b|gpt-oss-120b|OpenAI gpt-oss-120b|C|0.50,0.50|131072,16384|TJ|-|Welcome to the gpt-oss series, OpenAI's open-weight models designed for powerful|Y
fireworks/gpt-oss-20b|gpt-oss-20b|OpenAI gpt-oss-20b|C|0.50,0.50|131072,16384|J|-|Welcome to the gpt-oss series, OpenAI's open-weight models designed for powerful|Y
fireworks/gpt-oss-20b-eagle3-v1|gpt-oss-20b-eagle3-v1|gpt-oss-20b-drafter|C|0.50,0.50|4096,4096|-|-|gpt-oss-20b drafter|Y
fireworks/gpt-oss-safeguard-120b|gpt-oss-safeguard-120b|OpenAI gpt-oss-safeguard-120b|C|0.50,0.50|131072,16384|TJ|-|gpt-oss-safeguard-120b is a safety-focused language model with 117B total parame|Y
fireworks/gpt-oss-safeguard-20b|gpt-oss-safeguard-20b|OpenAI gpt-oss-safeguard-20b|C|0.50,0.50|131072,16384|TJ|-|gpt-oss-safeguard-20b is a safety-focused language model with 21B total paramete|Y
fireworks/hermes-2-pro-mistral-7b|hermes-2-pro-mistral-7b|Hermes 2 Pro Mistral 7B|C|0.20,0.20|32768,16384|TJ|-|Latest version of Nous Research's Hermes series of models, using an updated and|Y
fireworks/internvl3-38b|internvl3-38b|InternVL3 38B|C|0.20,0.20|16384,16384|VJ|-|The InternVL3 collection of models are advanced multimodal large language models|Y
fireworks/internvl3-78b|internvl3-78b|InternVL3 78B|C|0.20,0.20|16384,16384|VJ|-|The InternVL3 collection of models are advanced multimodal large language models|Y
fireworks/internvl3-8b|internvl3-8b|InternVL3 8B|C|0.20,0.20|16384,16384|VJ|-|The InternVL3 collection of models are advanced multimodal large language models|Y
fireworks/kat-coder|kat-coder|KAT Coder|C|0.20,0.20|262144,16384|J|-|KAT-Coder-Pro V1 is KwaiKAT's most advanced agentic coding model in the KwaiKAT|Y
fireworks/kat-dev-32b|kat-dev-32b|KAT Dev 32B|C|0.90,0.90|131072,16384|J|-|KAT-Dev-32B is an open-source 32B-parameter model for software engineering tasks|Y
fireworks/kat-dev-72b-exp|kat-dev-72b-exp|KAT Dev 72B Exp|C|0.90,0.90|131072,16384|TJ|-|KAT-Dev-72B-Exp is an open-source 72B-parameter model for software engineering t|Y
fireworks/kimi-k2-instruct|kimi-k2|Kimi K2 Instruct|C|0.35,1.4|131072,16384|TJ|-|Kimi K2 is a state-of-the-art mixture-of-experts (MoE) language model with 32 bi|Y
fireworks/kimi-k2-instruct-0905|kimi-k2-0905|Kimi K2 Instruct 0905|C|0.35,1.4|262144,16384|TJ|-|Kimi K2 0905 is an updated version of Kimi K2, a state-of-the-art mixture-of-exp|Y
fireworks/kimi-k2-thinking|kimi-k2-thinking|Kimi K2 Thinking|C|0.35,1.4|4096,4096|TK|-|Kimi K2 Thinking is the latest, most capable version of open-source thinking mod|N
fireworks/llama-guard-2-8b|llama-guard-2-8b|Llama Guard v2 8B|C|0.20,0.20|8192,8192|J|-|Meta Llama Guard 2 is an 8B parameter Llama 3-based LLM safeguard model. Similar|Y
fireworks/llama-guard-3-1b|llama-guard-3-1b|Llama Guard v3 1B|C|0.20,0.20|131072,16384|J|-|Llama Guard 3-1B is a fine-tuned Llama-3.2-1B pretrained model for content safet|Y
fireworks/llama-guard-3-8b|llama-guard-3-8b|Llama Guard 3 8B|C|0.20,0.20|131072,16384|J|-|Llama Guard 3 is a Llama-3.1-8B pretrained model, fine-tuned for content safety|Y
fireworks/llama-v2-13b|llama-v2-13b|Llama 2 13B|C|0.20,0.20|4096,4096|J|-|Llama 2 is a collection of pretrained and fine-tuned generative text models rang|Y
fireworks/llama-v2-13b-chat|llama-v2-13b|Llama 2 13B Chat|C|0.20,0.20|4096,4096|J|-|Llama 2 is a collection of pretrained and fine-tuned generative text models rang|Y
fireworks/llama-v2-70b|llama-v2-70b|Llama 2 70B|C|0.90,0.90|4096,4096|J|-|Meta developed and publicly released the Llama 2 family of large language models|Y
fireworks/llama-v2-7b|llama-v2-7b|Llama 2 7B|C|0.20,0.20|4096,4096|J|-|Meta's Llama 2 model family is a collection of pretrained and fine-tuned generat|Y
fireworks/llama-v2-7b-chat|llama-v2-7b|Llama 2 7B Chat|C|0.20,0.20|4096,4096|J|-|Llama 2 is a collection of pretrained and fine-tuned generative text models rang|Y
fireworks/llama-v3-70b-instruct|llama-v3-70b|Llama 3 70B Instruct|C|0.90,0.90|8192,8192|J|-|Meta developed and released the Meta Llama 3 family of large language models (LL|Y
fireworks/llama-v3-70b-instruct-hf|llama-v3-70b-hf|Llama 3 70B Instruct (HF version)|C|0.90,0.90|8192,8192|J|-|Meta’s Llama 3 instruction tuned models are optimized for dialogue use cases and|Y
fireworks/llama-v3-70b-instruct-v2|llama-v3-70b-v2|Llama v3 70B Instruct V2 Draft Model|C|0.90,0.90|4096,4096|-|-|Meta developed and released the Meta Llama 3 family of large language models (LL|Y
fireworks/llama-v3-8b|llama-v3-8b|Llama 3 8B|C|0.20,0.20|8192,8192|J|-|Llama 3 is an auto-regressive language model that uses an optimized transformer|Y
fireworks/llama-v3-8b-instruct|llama-v3-8b|Llama 3 8B Instruct|C|0.20,0.20|8192,8192|J|-|Meta developed and released the Meta Llama 3 family of large language models (LL|Y
fireworks/llama-v3-8b-instruct-hf|llama-v3-8b-hf|Llama 3 8B Instruct (HF version)|C|0.20,0.20|8192,8192|J|-|Meta's Llama 3 instruction tuned models are optimized for dialogue use cases and|Y
fireworks/llama-v3-8b-instruct-v0|llama-v3-8b-v0|Llama v3 8B Instruct V0 Draft Model|C|0.20,0.20|4096,4096|-|-|Meta developed and released the Meta Llama 3 family of large language models (LL|Y
fireworks/llama-v3p1-405b-instruct|llama-1-405b|Llama 3.1 405B Instruct|C|3.0,3.0|131072,16384|TJ|-|The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a|Y
fireworks/llama-v3p1-405b-instruct-long|llama-1-405b-long|Llama 3.1 405B Instruct Long|C|3.0,3.0|4096,4096|-|-|The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a|Y
fireworks/llama-v3p1-70b-instruct|llama-1-70b|Llama 3.1 70B Instruct|C|0.90,0.90|131072,16384|TJ|-|The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a|Y
fireworks/llama-v3p1-70b-instruct-1b|llama-1-70b-1b|Llama 3.1 70B Instruct 1B|C|0.90,0.90|4096,4096|-|-|The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a|Y
fireworks/llama-v3p1-8b-instruct|llama-1-8b|Llama 3.1 8B Instruct|C|0.20,0.20|131072,16384|J|-|The Meta Llama 3.1 collection of multilingual large language models (LLMs) is a|Y
fireworks/llama-v3p1-nemotron-70b-instruct|llama-1-nemotron-70b|Llama 3.1 Nemotron 70B|C|0.90,0.90|131072,16384|J|-|Llama-3.1-Nemotron-70B-Instruct is a large language model customized by NVIDIA t|Y
fireworks/llama-v3p2-11b-vision-instruct|llama-2-11b-vision|Llama 3.2 11B Vision Instruct|C|0.20,0.20|131072,16384|VJ|-|Instruction-tuned image reasoning model from Meta with 11B parameters. Optimized|Y
fireworks/llama-v3p2-1b|llama-2-1b|Llama 3.2 1B|C|0.20,0.20|131072,16384|J|-|The Llama 3.2 collection of multilingual large language models (LLMs) is a colle|Y
fireworks/llama-v3p2-1b-instruct|llama-2-1b|Llama 3.2 1B Instruct|C|0.20,0.20|131072,16384|J|-|The Llama 3.2 collection of multilingual large language models (LLMs) is a colle|Y
fireworks/llama-v3p2-3b|llama-2-3b|Llama 3.2 3B|C|0.10,0.10|131072,16384|J|-|The Llama 3.2 collection of multilingual large language models (LLMs) is a colle|Y
fireworks/llama-v3p2-3b-instruct|llama-2-3b|Llama 3.2 3B Instruct|C|0.10,0.10|131072,16384|J|-|The Llama 3.2 collection of multilingual large language models (LLMs) is a colle|Y
fireworks/llama-v3p2-90b-vision-instruct|llama-2-90b-vision|Llama 3.2 90B Vision Instruct|C|0.20,0.20|131072,16384|VJ|-|Instruction-tuned image reasoning model with 90B parameters from Meta. Optimized|Y
fireworks/llama-v3p3-70b-instruct|llama-3-70b|Llama 3.3 70B Instruct|C|0.90,0.90|131072,16384|J|-|Llama 3.3 70B Instruct is the December update of Llama 3.1 70B. The model improv|Y
fireworks/llama4-maverick-instruct-basic|llama4-maverick|Llama 4 Maverick Instruct (Basic)|C|0.22,0.88|1048576,16384|VTJ|-|The Llama 4 collection of models are natively multimodal AI models that enable t|Y
fireworks/llama4-scout-instruct-basic|llama4-scout|Llama 4 Scout Instruct (Basic)|C|0.15,0.60|1048576,16384|VTJ|-|The Llama 4 collection of models are natively multimodal AI models that enable t|Y
fireworks/llamaguard-7b|llamaguard-7b|Llama Guard 7B|C|0.20,0.20|4096,4096|J|-|Llama-Guard is a 7B parameter Llama 2-based input-output safeguard model. It can|Y
fireworks/minimax-m1-80k|minimax-m1-80k|MiniMax-M1-80k|C|0.20,0.20|4096,4096|-|-|We introduce MiniMax-M1, the world's first open-weight, large-scale hybrid-atten|Y
fireworks/minimax-m2|minimax-m2|MiniMax-M2|C|0.20,0.20|196608,16384|TJ|-|**Preview:** This model is currently in preview. Full production support coming|Y
fireworks/minimax-m2p1|minimax-m2p1|MiniMax-M2.1|C|0.20,0.20|204800,16384|TJ|-|MiniMax M2.1 is built for strong real-world performance across complex, multi-la|Y
fireworks/ministral-3-14b-instruct-2512|ministral-3-14b-2512|Ministral 3 14B Instruct 2512|C|0.10,0.10|256000,16384|VTJ|-|Mistral's Ministral 3 14B dense model with vision encoder. The largest model in|Y
fireworks/ministral-3-3b-instruct-2512|ministral-3-3b-2512|Ministral 3 3B Instruct 2512|C|0.10,0.10|256000,16384|VTJ|-|Mistral's Ministral 3 3B dense model with vision encoder. The smallest model in|Y
fireworks/ministral-3-8b-instruct-2512|ministral-3-8b-2512|Ministral 3 8B Instruct 2512|C|0.20,0.20|256000,16384|VTJ|-|Mistral's Ministral 3 8B dense model with vision encoder. A balanced model in th|Y
fireworks/mistral-7b|mistral-7b|Mistral 7B|C|0.20,0.20|32768,16384|J|-|The Mistral-7B-v0.1 Large Language Model (LLM) is a pretrained generative text m|Y
fireworks/mistral-7b-instruct-4k|mistral-7b-4k|Mistal 7B Instruct V0.1|C|0.20,0.20|32768,16384|J|-|The Mistral-7B-Instruct-v0.1 Large Language Model (LLM) is a instruct fine-tuned|Y
fireworks/mistral-7b-instruct-v0p2|mistral-7b-v0p2|Mistral 7B Instruct v0.2|C|0.20,0.20|32768,16384|J|-|Mistral 7B Instruct v0.2 is an instruction fine-tuned version of the Mistral 7B|Y
fireworks/mistral-7b-instruct-v3|mistral-7b-v3|Mistral 7B Instruct v0.3|C|0.20,0.20|32768,16384|TJ|-|Mistral 7B Instruct v0.3 is an instruction fine-tuned version of the Mistral 7B|Y
fireworks/mistral-7b-v0p2|mistral-7b-v0p2|Mistral 7B v0.2|C|0.20,0.20|32768,16384|J|-|The Mistral-7B-v0.2 Large Language Model (LLM) is the successor to the Mistral-7|Y
fireworks/mistral-large-3-fp8|mistral-large-3-fp8|Mistral Large 3 675B Instruct 2512|C|0.20,0.20|256000,16384|VTJ|-|Mistral Large 3 is a state-of-the-art general-purpose Multimodal granular Mixtur|Y
fireworks/mistral-nemo-base-2407|mistral-nemo-base-2407|Mistral Nemo Base 2407|C|0.20,0.20|128000,16384|J|-|The Mistral-Nemo-Base-2407 Large Language Model (LLM) is a pretrained generative|Y
fireworks/mistral-nemo-instruct-2407|mistral-nemo-2407|Mistral Nemo Instruct 2407|C|0.20,0.20|128000,16384|J|-|The Mistral-Nemo-Instruct-2407 Large Language Model (LLM) is the instruction-tun|Y
fireworks/mistral-small-24b-instruct-2501|mistral-small-24b-2501|Mistral Small 24B Instruct 2501|C|0.10,0.10|32768,16384|J|-|Mistral Small 3 ( 2501 ) sets a new benchmark in the "small" Large Language Mode|Y
fireworks/mixtral-8x22b|mixtral-8x22b|Mixtral Moe 8x22B|C|1.2,1.2|65536,16384|J|-|The Mixtral MoE 8x22B v0.1 Large Language Model (LLM) is a pretrained generative|Y
fireworks/mixtral-8x22b-instruct|mixtral-8x22b|Mixtral MoE 8x22B Instruct|C|1.2,1.2|65536,16384|TJ|-|Mixtral MoE 8x22B Instruct v0.1 is the instruction-tuned version of Mixtral MoE|Y
fireworks/mixtral-8x7b|mixtral-8x7b|Mixtral 8x7B v0.1|C|0.20,0.20|32768,16384|J|-|Mixtral 8x7B v0.1 is a sparse mixture-of-experts (SMoE) large language model dev|Y
fireworks/mixtral-8x7b-instruct|mixtral-8x7b|Mixtral MoE 8x7B Instruct|C|0.20,0.20|32768,16384|J|-|Mixtral MoE 8x7B Instruct is the instruction-tuned version of Mixtral MoE 8x7B a|Y
fireworks/mixtral-8x7b-instruct-hf|mixtral-8x7b-hf|Mixtral MoE 8x7B Instruct (HF version)|C|0.20,0.20|32768,16384|J|-|Mixtral MoE 8x7B Instruct (HF Version) is the original, FP16 version of Mixtral|Y
fireworks/mixtral-8x7b-instruct-v0-oss|mixtral-8x7b-v0-oss|Mixtral 8x7b Instruct V0 Draft Model|C|0.20,0.20|4096,4096|-|-|Mixtral MoE 8x7B Instruct is the instruction-tuned version of Mixtral MoE 8x7B a|Y
fireworks/mythomax-l2-13b|mythomax-l2-13b|MythoMax L2 13B|C|0.20,0.20|4096,4096|J|-|An improved, potentially even perfected variant of MythoMix, a MythoLogic-L2 and|Y
fireworks/nemotron-nano-3-30b-a3b|nemotron-nano-3-30b-a3b|NVIDIA Nemotron Nano 3 30B A3B|C|0.90,0.90|262144,16384|TJ|-|Nemotron-Nano-3-30B-A3B is a  large language model trained by NVIDIA, designed|Y
fireworks/nemotron-nano-v2-12b-vl|nemotron-nano-v2-12b-vl|NVIDIA Nemotron Nano 2 VL|C|0.20,0.20|131072,16384|VJ|-|NVIDIA Nemotron Nano 2 VL is an open 12B multimodal reasoning model for document|Y
fireworks/nous-capybara-7b-v1p9|nous-capybara-7b-v1p9|Nous Capybara 7B V1.9|C|0.20,0.20|32768,16384|J|-|Nous-Capybara 7B V1.9 is a new model trained for multiple epochs on a dataset of|Y
fireworks/nous-hermes-2-mixtral-8x7b-dpo|nous-hermes-2-mixtral-8x7b-dpo|Nouse Hermes 2 Mixtral 8x7B DPO|C|0.20,0.20|32768,16384|J|-|Nous Hermes 2 Mixtral 8x7B DPO is the new flagship Nous Research model trained o|Y
fireworks/nous-hermes-llama2-13b|nous-hermes-llama2-13b|Nous Hermes Llama2 13B|C|0.20,0.20|4096,4096|J|-|Nous-Hermes-Llama2-13b is a state-of-the-art language model fine-tuned on over 3|Y
fireworks/nous-hermes-llama2-70b|nous-hermes-llama2-70b|Nous Hermes Llama2 70B|C|0.90,0.90|4096,4096|J|-|Nous-Hermes-Llama2-70b is a state-of-the-art language model fine-tuned on over 3|Y
fireworks/nous-hermes-llama2-7b|nous-hermes-llama2-7b|Nous Hermes Llama2 7B|C|0.20,0.20|4096,4096|J|-|Nous-Hermes-Llama2-7b is a state-of-the-art language model fine-tuned on over 30|Y
fireworks/nvidia-nemotron-nano-12b-v2|nvidia-nemotron-nano-12b-v2|NVIDIA Nemotron Nano 12B v2|C|0.20,0.20|128000,16384|TJ|-|NVIDIA-Nemotron-Nano-12B-v2 is a large language model (LLM) trained from scratch|Y
fireworks/nvidia-nemotron-nano-9b-v2|nvidia-nemotron-nano-9b-v2|NVIDIA Nemotron Nano 9B v2|C|0.20,0.20|128000,16384|TJ|-|NVIDIA-Nemotron-Nano-9B-v2 is a large language model (LLM) trained from scratch|Y
fireworks/openchat-3p5-0106-7b|openchat-3p5-0106-7b|OpenChat 3.5 0106|C|0.20,0.20|8192,8192|J|-|OpenChat is an innovative library of open-source language models, fine-tuned wit|Y
fireworks/openhermes-2-mistral-7b|openhermes-2-mistral-7b|OpenHermes 2 Mistral 7B|C|0.20,0.20|32768,16384|J|-|OpenHermes 2 Mistral 7B is a state of the art Mistral Fine-tune. OpenHermes was|Y
fireworks/openhermes-2p5-mistral-7b|openhermes-2p5-mistral-7b|OpenHermes 2.5 Mistral 7B|C|0.20,0.20|32768,16384|J|-|OpenHermes 2.5 Mistral 7B is a state of the art Mistral Fine-tune, a continuatio|Y
fireworks/openorca-7b|openorca-7b|Mistral 7B OpenOrca|C|0.20,0.20|32768,16384|J|-|A fine-tuned version of Mistral-7B trained on the OpenOrca dataset, based on the|Y
fireworks/phi-3-mini-128k-instruct|phi-3-mini-128k|Phi-3 Mini 128k Instruct|C|0.20,0.20|131072,16384|J|-|Phi-3-Mini-128K-Instruct is a 3.8 billion-parameter, lightweight, state-of-the-a|Y
fireworks/phi-3-vision-128k-instruct|phi-3-vision-128k|Phi-3.5 Vision Instruct|C|0.20,0.20|32064,16384|VJ|-|Phi-3-Vision-128K-Instruct is a lightweight, state-of-the-art open multimodal mo|Y
fireworks/phi4-eagle|phi4-eagle|Phi-4 Eagle|C|0.20,0.20|4096,4096|-|-|EAGLE draft model for Phi4|Y
fireworks/phind-code-llama-34b-python-v1|phind-code-llama-34b-python-v1|Phind CodeLlama 34B Python v1|C|0.10,0.10|16384,16384|J|-|Phind CodeLlama 34B Python V1 is a fine-tuned version of the CodeLlama 34B Pytho|Y
fireworks/phind-code-llama-34b-v1|phind-code-llama-34b-v1|Phind CodeLlama 34B v1|C|0.10,0.10|16384,16384|J|-|Phind CodeLlama 34B V1 is a fine-tuned version of the Code-Llama 34B LLM using a|Y
fireworks/phind-code-llama-34b-v2|phind-code-llama-34b-v2|Phind CodeLlama 34B v2|C|0.10,0.10|16384,16384|J|-|This model is fine-tuned from Phind-CodeLlama-34B-v1 and achieves 73.8% pass@1 o|Y
fireworks/pythia-12b|pythia-12b|Pythia 12B|C|0.20,0.20|2048,2048|J|-|The Pythia model suite was deliberately designed to promote scientific research|Y
fireworks/qwen-qwq-32b-preview|qwen-qwq-32b-preview|Qwen QWQ 32B Preview|C|0.90,0.90|32768,16384|JK|-|Qwen QwQ model focuses on advancing AI reasoning, and showcases the power of ope|N
fireworks/qwen-v2p5-14b-instruct|qwen-v2p5-14b|Qwen2.5 14B Instruct|C|0.10,0.10|32768,16384|J|-|Qwen2.5 are a series of decoder-only language models developed by Qwen team, Ali|Y
fireworks/qwen-v2p5-7b|qwen-v2p5-7b|Qwen2.5 7B|C|0.20,0.20|131072,16384|J|-|Qwen2.5 are a series of decoder-only language models developed by Qwen team, Ali|Y
fireworks/qwen1p5-72b-chat|qwen1p5-72b|Qwen1.5 72B Chat|C|0.90,0.90|32768,16384|J|-|Qwen1.5 is the beta version of Qwen2, a transformer-based decoder-only language|Y
fireworks/qwen2-72b-instruct|qwen2-72b|Qwen2 72B Instruct|C|0.90,0.90|32768,16384|J|-|Qwen2 72B Instruct is a 72 billion parameter model developed by Alibaba for inst|Y
fireworks/qwen2-7b-instruct|qwen2-7b|Qwen2 7B Instruct|C|0.20,0.20|32768,16384|J|-|Qwen2 7B Instruct is a 7-billion-parameter instruction-tuned language model deve|Y
fireworks/qwen2-vl-2b-instruct|qwen2-vl-2b|Qwen2-VL 2B Instruct|C|0.20,0.20|32768,16384|VJ|-|Qwen2-VL is a multimodal large language model series developed by Qwen team, Ali|Y
fireworks/qwen2-vl-72b-instruct|qwen2-vl-72b|Qwen2-VL 72B Instruct|C|0.90,0.90|32768,16384|VJ|-|Qwen2-VL is a multimodal large language model series developed by Qwen team, Ali|Y
fireworks/qwen2-vl-7b-instruct|qwen2-vl-7b|Qwen2-VL 7B Instruct|C|0.20,0.20|32768,16384|VJ|-|Qwen2-VL is a multimodal large language model series developed by Qwen team, Ali|Y
fireworks/qwen2p5-0p5b-instruct|qwen-2.5-0p5b|Qwen2.5 0.5B Instruct|C|0.20,0.20|32768,16384|J|-|Qwen2.5 are a series of decoder-only language models developed by Qwen team, Ali|Y
fireworks/qwen2p5-14b|qwen-2.5-14b|Qwen2.5 14B|C|0.10,0.10|131072,16384|J|-|Qwen2.5 are a series of decoder-only language models developed by Qwen team, Ali|Y
fireworks/qwen2p5-14b-instruct|qwen-2.5-14b|Qwen2.5 14B Instruct|C|0.10,0.10|32768,16384|J|-|Qwen2.5 are a series of decoder-only language models developed by Qwen team, Ali|Y
fireworks/qwen2p5-1p5b-instruct|qwen-2.5-1p5b|Qwen2.5 1.5B Instruct|C|0.20,0.20|32768,16384|J|-|Qwen2.5 are a series of decoder-only language models developed by Qwen team, Ali|Y
fireworks/qwen2p5-32b|qwen-2.5-32b|Qwen2.5 32B|C|0.90,0.90|131072,16384|J|-|Qwen2.5 are a series of decoder-only language models developed by Qwen team, Ali|Y
fireworks/qwen2p5-32b-instruct|qwen-2.5-32b|Qwen2.5 32B Instruct|C|0.90,0.90|32768,16384|J|-|Qwen2.5 are a series of decoder-only language models developed by Qwen team, Ali|Y
fireworks/qwen2p5-72b|qwen-2.5-72b|Qwen2.5 72B|C|0.90,0.90|131072,16384|J|-|Qwen2.5 are a series of decoder-only language models developed by Qwen team, Ali|Y
fireworks/qwen2p5-72b-instruct|qwen-2.5-72b|Qwen2.5 72B Instruct|C|0.90,0.90|32768,16384|TJ|-|Qwen2.5 are a series of decoder-only language models developed by Qwen team, Ali|Y
fireworks/qwen2p5-7b|qwen-2.5-7b|Qwen2.5 7B|C|0.20,0.20|131072,16384|J|-|Qwen2.5 are a series of decoder-only language models developed by Qwen team, Ali|Y
fireworks/qwen2p5-7b-instruct|qwen-2.5-7b|Qwen2.5 7B Instruct|C|0.20,0.20|32768,16384|J|-|Qwen2.5 are a series of decoder-only language models developed by Qwen team, Ali|Y
fireworks/qwen2p5-coder-0p5b|qwen-2.5-coder-0p5b|Qwen2.5-Coder 0.5B|C|0.20,0.20|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-0p5b-instruct|qwen-2.5-coder-0p5b|Qwen2.5-Coder 0.5B Instruct|C|0.20,0.20|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-14b|qwen-2.5-coder-14b|Qwen2.5-Coder 14B|C|0.10,0.10|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-14b-instruct|qwen-2.5-coder-14b|Qwen2.5-Coder 14B Instruct|C|0.10,0.10|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-1p5b|qwen-2.5-coder-1p5b|Qwen2.5-Coder 1.5B|C|0.20,0.20|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-1p5b-instruct|qwen-2.5-coder-1p5b|Qwen2.5-Coder 1.5B Instruct|C|0.20,0.20|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-32b|qwen-2.5-coder-32b|Qwen2.5-Coder 32B|C|0.90,0.90|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-32b-instruct|qwen-2.5-coder-32b|Qwen2.5-Coder 32B Instruct|C|0.90,0.90|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-32b-instruct-128k|qwen-2.5-coder-32b-128k|Qwen2.5-Coder 32B Instruct 128K|C|0.90,0.90|131072,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-32b-instruct-32k-rope|qwen-2.5-coder-32b-32k-rope|Qwen2.5-Coder 32B Instruct 32K RoPE|C|0.90,0.90|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-32b-instruct-64k|qwen-2.5-coder-32b-64k|Qwen2.5-Coder 32B Instruct 64k|C|0.90,0.90|65536,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-3b|qwen-2.5-coder-3b|Qwen2.5-Coder 3B|C|0.10,0.10|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-3b-instruct|qwen-2.5-coder-3b|Qwen2.5-Coder 3B Instruct|C|0.10,0.10|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-7b|qwen-2.5-coder-7b|Qwen2.5-Coder 7B|C|0.20,0.20|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-coder-7b-instruct|qwen-2.5-coder-7b|Qwen2.5-Coder 7B Instruct|C|0.20,0.20|32768,16384|J|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
fireworks/qwen2p5-math-72b-instruct|qwen-2.5-math-72b|Qwen2.5-Math 72B Instruct|C|0.90,0.90|4096,4096|J|-|Qwen2.5-Math series is expanded to support using both CoT and Tool-integrated Re|Y
fireworks/qwen2p5-vl-32b-instruct|qwen-2.5-vl-32b|Qwen2.5-VL 32B Instruct|C|0.90,0.90|128000,16384|VJ|-|Qwen2.5-VL is a multimodal large language model series developed by Qwen team, A|Y
fireworks/qwen2p5-vl-3b-instruct|qwen-2.5-vl-3b|Qwen2.5-VL 3B Instruct|C|0.10,0.10|128000,16384|VJ|-|Qwen2.5-VL is a multimodal large language model series developed by Qwen team, A|Y
fireworks/qwen2p5-vl-72b-instruct|qwen-2.5-vl-72b|Qwen2.5-VL 72B Instruct|C|0.90,0.90|128000,16384|VJ|-|Qwen2.5-VL is a multimodal large language model series developed by Qwen team, A|Y
fireworks/qwen2p5-vl-7b-instruct|qwen-2.5-vl-7b|Qwen2.5-VL 7B Instruct|C|0.20,0.20|128000,16384|VJ|-|Qwen2.5-VL is a multimodal large language model series developed by Qwen team, A|Y
fireworks/qwen3-0p6b|qwen-3-0p6b|Qwen3 0.6B|C|0.20,0.20|40960,16384|TJ|-|Qwen3 0.6B model developed by Qwen team, Alibaba Cloud,|Y
fireworks/qwen3-14b|qwen-3-14b|Qwen3 14B|C|0.10,0.10|40960,16384|TJ|-|Qwen3 14B model developed by Qwen team, Alibaba Cloud,|Y
fireworks/qwen3-1p7b|qwen-3-1p7b|Qwen3 1.7B|C|0.20,0.20|131072,16384|TJ|-|Qwen 1.7B Model developed by Qwen team, Alibaba Cloud,|Y
fireworks/qwen3-1p7b-fp8-draft|qwen-3-1p7b-fp8-draft|Qwen3 1.7B fp8 model used for drafting|C|0.20,0.20|262144,16384|J|-|qwen 1.7b fp8 used as draft model|Y
fireworks/qwen3-1p7b-fp8-draft-131072|qwen-3-1p7b-fp8-draft-131072|Qwen3 1.7B fp8 model used for drafting for 131072 context|C|0.20,0.20|131072,16384|J|-|qwen 1.7b fp8 used as draft model for 131072 context length|Y
fireworks/qwen3-1p7b-fp8-draft-40960|qwen-3-1p7b-fp8-draft-40960|Qwen3 1.7B fp8 model used for drafting for 40960 context len|C|0.20,0.20|40960,16384|J|-|qwen 1.7b fp8 used as draft model for 40960 context length|Y
fireworks/qwen3-235b-a22b|qwen-3-235b-a22b|Qwen3 235B A22B|C|0.90,0.90|131072,16384|TJ|-|Latest Qwen3 state of the art model, 235B with 22B active parameter model|Y
fireworks/qwen3-235b-a22b-instruct-2507|qwen-3-235b-a22b-2507|Qwen3 235B A22B Instruct 2507|C|0.90,0.90|262144,16384|TJ|-|Updated FP8 version of Qwen3-235B-A22B non-thinking mode, with better tool use,|Y
fireworks/qwen3-235b-a22b-thinking-2507|qwen-3-235b-a22b-thinking-2507|Qwen3 235B A22B Thinking 2507|C|0.90,0.90|262144,16384|JK|-|Latest Qwen3 thinking model, competitive against the best close source models in|N
fireworks/qwen3-30b-a3b|qwen-3-30b-a3b|Qwen3 30B-A3B|C|0.90,0.90|131072,16384|TJ|-|Latest Qwen3 state of the art model, 30B with 3B active parameter model|Y
fireworks/qwen3-30b-a3b-instruct-2507|qwen-3-30b-a3b-2507|Qwen3 30B A3B Instruct 2507|C|0.90,0.90|262144,16384|J|-|Updated FP8 version of Qwen3-30B-A3B non-thinking mode, with better tool use, co|Y
fireworks/qwen3-30b-a3b-thinking-2507|qwen-3-30b-a3b-thinking-2507|Qwen3 30B A3B Thinking 2507|C|0.90,0.90|262144,16384|TJK|-|Updated FP8 version of Qwen3-30B-A3B thinking mode, with better tool use, coding|N
fireworks/qwen3-32b|qwen-3-32b|Qwen3 32B|C|0.90,0.90|131072,16384|TJ|-|Latest Qwen3 state of the art model, 32B model|Y
fireworks/qwen3-32b-eagle3-v2|qwen-3-32b-eagle3-v2|qwen3-32b-eagle3-drafter|C|0.90,0.90|4096,4096|-|-|qwen3 32b eagle3|Y
fireworks/qwen3-4b|qwen-3-4b|Qwen3 4B|C|0.10,0.10|40960,16384|TJ|-|Latest Qwen3 state of the art model, 4B model|Y
fireworks/qwen3-4b-instruct-2507|qwen-3-4b-2507|Qwen 3 4B Instruct 2507|C|0.10,0.10|262144,16384|J|-|Introducing Qwen3-4B-Instruct-2507, with improved instruction following, reasoni|Y
fireworks/qwen3-8b|qwen-3-8b|Qwen3 8B|C|0.20,0.20|40960,16384|TJ|-|Latest Qwen3 state of the art model, FP8 version 8B Model|Y
fireworks/qwen3-coder-30b-a3b-instruct|qwen-3-coder-30b-a3b|Qwen3 Coder 30B A3B Instruct|C|0.90,0.90|262144,16384|J|-|Latest Qwen3 coder model, 30B with 3B active parameter model|Y
fireworks/qwen3-coder-480b-a35b-instruct|qwen-3-coder-480b-a35b|Qwen3 Coder 480B A35B Instruct|C|0.90,0.90|262144,16384|TJ|-|Qwen3's most agentic code model to date|Y
fireworks/qwen3-coder-480b-instruct-bf16|qwen-3-coder-480b-bf16|Qwen3 Coder 480B Instruct BF16|C|0.90,0.90|262144,16384|J|-|The BF16 version of the 480B coder model|Y
fireworks/qwen3-embedding-0p6b|qwen-3-embedding-0p6b|Qwen3 Embedding 0.6B|C|0.0080,|32768,32768|E|-|significant advancements in multiple text embedding and ranking tasks, including|N
fireworks/qwen3-embedding-4b|qwen-3-embedding-4b|Qwen3 Embedding 4B|C|0.0080,|40960,40960|E|-|significant advancements in multiple text embedding and ranking tasks, including|N
fireworks/qwen3-embedding-8b|qwen-3-embedding-8b|Qwen3 Embedding 8B|C|0.0080,|40960,40960|E|-|The Qwen3 Embedding 8B model is the latest proprietary model of the Qwen family,|N
fireworks/qwen3-next-80b-a3b-instruct|qwen-3-next-80b-a3b|Qwen3 Next 80B A3B Instruct|C|0.10,0.10|4096,4096|-|-|Qwen3 Next 80B A3B Instruct is a state-of-the-art mixture-of-experts (MoE) langu|Y
fireworks/qwen3-next-80b-a3b-thinking|qwen-3-next-80b-a3b-thinking|Qwen3 Next 80B A3B Thinking|C|0.10,0.10|4096,4096|K|-|Qwen3 Next 80B A3B Thinking is a state-of-the-art mixture-of-experts (MoE) langu|N
fireworks/qwen3-omni-30b-a3b-instruct|qwen-3-omni-30b-a3b|Qwen3 Omni 30B A3B Instruct|C|0.90,0.90|65536,16384|VTJ|-|Qwen3-Omni is a natively end-to-end multilingual omni-modal foundation model. It|Y
fireworks/qwen3-reranker-0p6b|qwen-3-reranker-0p6b|Qwen3 Reranker 0.6B|C|0.0080,|40960,40960|E|-|significant advancements in multiple text embedding and ranking tasks, including|N
fireworks/qwen3-reranker-4b|qwen-3-reranker-4b|Qwen3 Reranker 4B|C|0.0080,|40960,40960|E|-|significant advancements in multiple text embedding and ranking tasks, including|N
fireworks/qwen3-reranker-8b|qwen-3-reranker-8b|Qwen3 Reranker 8B|C|0.0080,|40960,40960|E|-|significant advancements in multiple text embedding and ranking tasks, including|N
fireworks/qwen3-vl-235b-a22b-instruct|qwen-3-vl-235b-a22b|Qwen3 VL 235B A22B Instruct|C|0.90,0.90|262144,16384|VTJ|-|Qwen3 VL 235B A22B Instruct is a state-of-the-art vision-language model with 22|Y
fireworks/qwen3-vl-235b-a22b-thinking|qwen-3-vl-235b-a22b-thinking|Qwen3 VL 235B A22B Thinking|C|0.90,0.90|262144,16384|VTJK|-|Qwen3 VL 235B A22B Thinking is a state-of-the-art vision-language model with 22|Y
fireworks/qwen3-vl-30b-a3b-instruct|qwen-3-vl-30b-a3b|Qwen3 VL 30B A3B Instruct|C|0.90,0.90|262144,16384|VTJ|-|Qwen3-VL series delivers superior text understanding & generation, deeper visual|Y
fireworks/qwen3-vl-30b-a3b-thinking|qwen-3-vl-30b-a3b-thinking|Qwen3 VL 30B A3B Thinking|C|0.90,0.90|262144,16384|VTJK|-|Qwen3-VL series delivers superior text understanding & generation, deeper visual|Y
fireworks/qwen3-vl-32b-instruct|qwen-3-vl-32b|Qwen3 VL 32B Instruct|C|0.90,0.90|4096,4096|V|-|The Qwen3-VL-32B-Instruct model is an advanced vision-language model that signif|Y
fireworks/qwen3-vl-8b-instruct|qwen-3-vl-8b|Qwen3-VL-8B-Instruct|C|0.20,0.20|4096,4096|V|-|The Qwen3-VL-8B-Instruct model is an advanced vision-language model that signifi|Y
fireworks/qwq-32b|qwq-32b|QWQ 32B|C|0.90,0.90|131072,16384|JK|-|Medium-sized reasoning model from Qwen.|N
fireworks/rolm-ocr|rolm-ocr|Rolm OCR|C|0.20,0.20|128000,16384|VJ|-|RolmOCR is an open-source document OCR model developed by Reducto AI as a drop-i|Y
fireworks/seed-oss-36b-instruct|seed-oss-36b|Seed OSS 36B Instruct|C|0.20,0.20|524288,16384|TJ|-|Seed-OSS is a series of open-source large language models developed by ByteDance|Y
fireworks/snorkel-mistral-7b-pairrm-dpo|snorkel-mistral-7b-pairrm-dpo|Snorkel Mistral PairRM DPO|C|0.20,0.20|32768,16384|J|-|A fine-tuned version of the Mistral-7B model developed by Snorkel using PairRM f|Y
fireworks/toppy-m-7b|toppy-m-7b|Toppy M 7B|C|0.20,0.20|32768,16384|J|-|A wild 7B parameter model that merges several models using the new task_arithmet|Y
fireworks/zephyr-7b-beta|zephyr-7b-beta|Zephyr 7B Beta|C|0.20,0.20|32768,16384|J|-|Zephyr is a series of language models that are trained to act as helpful assista|Y

# =============================================================================
# PERPLEXITY - Search AI (7 models)
# =============================================================================
perplexity/r1-1776|r1-1776|R1 1776|C|2.0,8.0|127072,8000|TJK|-|Post-trained reasoning model|N
perplexity/sonar|sonar|Perplexity: Sonar|C|0.0000,0.0000|127072,31768|V|-|Sonar is lightweight, affordable, fast, and simple to use - now featuring citati|Y
perplexity/sonar-deep-research|sonar-deep-research|Perplexity: Sonar Deep Research|C|0.0000,0.0000|128000,32000|K|-|Sonar Deep Research is a research-focused model designed for multi-step retrieva|N
perplexity/sonar-pro|sonar-pro|Perplexity: Sonar Pro|C|0.0000,0.0000|200000,8000|V|-|Note: Sonar Pro pricing includes Perplexity search pricing. See [details here](h|Y
perplexity/sonar-pro-search|sonar-pro-search|Perplexity: Sonar Pro Search|C|0.0000,0.0000|200000,8000|JKSV|-|Exclusively available on the OpenRouter API, Sonar Pro's new Pro Search mode is|Y
perplexity/sonar-reasoning|sonar-reasoning|Perplexity: Sonar Reasoning|C|0.0000,0.0000|127000,31750|K|-|Sonar Reasoning is a reasoning model provided by Perplexity based on [DeepSeek R|N
perplexity/sonar-reasoning-pro|sonar-reasoning-pro|Perplexity: Sonar Reasoning Pro|C|0.0000,0.0000|128000,32000|KV|-|Note: Sonar Pro pricing includes Perplexity search pricing. See [details here](h|N

# =============================================================================
# TOGETHER AI - Aggregator (139 models)
# =============================================================================
together/Alibaba-NLP/gte-modernbert-base|gte-modernbert-base|Gte Modernbert Base|C|0.08,0.08|8192,2048|E|-|Gte Modernbert Base on Together AI|N
together/BAAI/bge-base-en-v1.5|bge-base-en-v1.5|BAAI-Bge-Base-1.5|C|8000.0,8000.0|512,128|E|-|BAAI-Bge-Base-1.5 on Together AI|N
together/BAAI/bge-large-en-v1.5|bge-large-en-v1.5|BAAI-Bge-Large-1.5|C|0.02,0.02|4096,1024|E|-|BAAI-Bge-Large-1.5 on Together AI|N
together/ByteDance-Seed/Seedream-3.0|seedream-3.0|ByteDance Seedream 3.0|C|-|4096,1024|I|-|ByteDance Seedream 3.0 on Together AI|N
together/ByteDance-Seed/Seedream-4.0|seedream-4.0|ByteDance Seedream 4.0|C|-|4096,1024|I|-|ByteDance Seedream 4.0 on Together AI|N
together/ByteDance/Seedance-1.0-lite|seedance-1.0-lite|ByteDance Seedance 1.0 Lite|C|-|4096,1024|D|-|ByteDance Seedance 1.0 Lite on Together AI|N
together/ByteDance/Seedance-1.0-pro|seedance-1.0-pro|ByteDance Seedance 1.0 Pro|C|-|4096,1024|D|-|ByteDance Seedance 1.0 Pro on Together AI|N
together/HiDream-ai/HiDream-I1-Dev|hidream-i1-dev|HiDream-I1-Dev|C|-|4096,1024|I|-|HiDream-I1-Dev on Together AI|N
together/HiDream-ai/HiDream-I1-Fast|hidream-i1-fast|HiDream-I1-Fast|C|-|4096,1024|I|-|HiDream-I1-Fast on Together AI|N
together/HiDream-ai/HiDream-I1-Full|hidream-i1-full|HiDream-I1-Full|C|-|4096,1024|I|-|HiDream-I1-Full on Together AI|N
together/Lykon/DreamShaper|dreamshaper|Dreamshaper|C|-|4096,1024|I|-|Dreamshaper on Together AI|N
together/Meta-Llama/Llama-Guard-7b|llama-guard-7b|Llama Guard (7B)|C|0.20,0.20|4096,1024|M|-|Llama Guard (7B) on Together AI|N
together/Qwen/Qwen-Image|qwen-image|Qwen Image|C|-|4096,1024|I|-|Qwen Image on Together AI|N
together/Qwen/Qwen2.5-14B-Instruct|qwen2.5-14b-instruct|Qwen 2.5 14B Instruct|C|0.80,0.80|32768,4096|TJS|-|Qwen 2.5 14B Instruct on Together AI|Y
together/Qwen/Qwen2.5-72B-Instruct|qwen2.5-72b-instruct|Qwen2.5 72B Instruct|C|1.2,1.2|32768,4096|TJS|-|Qwen2.5 72B Instruct on Together AI|Y
together/Qwen/Qwen2.5-72B-Instruct-Turbo|qwen2.5-72b-instruct-turbo|Qwen2.5 72B Instruct Turbo|C|1.2,1.2|131072,4096|TJS|-|Qwen2.5 72B Instruct Turbo on Together AI|Y
together/Qwen/Qwen2.5-7B-Instruct-Turbo|qwen2.5-7b-instruct-turbo|Qwen2.5 7B Instruct Turbo|C|0.30,0.30|32768,4096|TJS|-|Qwen2.5 7B Instruct Turbo on Together AI|Y
together/Qwen/Qwen2.5-VL-72B-Instruct|qwen2.5-vl-72b-instruct|Qwen2.5-VL (72B) Instruct|C|1.9,8.0|32768,4096|VTJS|-|Qwen2.5-VL (72B) Instruct on Together AI|Y
together/Qwen/Qwen3-235B-A22B-Instruct-2507-tput|qwen3-235b-a22b-instruct-2507-tput|Qwen3 235B A22B Instruct 2507 FP8 Throughput|C|0.20,0.60|262144,4096|TJ|-|Qwen3 235B A22B Instruct 2507 FP8 Throughput on Together AI|Y
together/Qwen/Qwen3-235B-A22B-Thinking-2507|qwen3-235b-a22b-thinking-2507|Qwen3 235B A22B Thinking 2507 FP8|C|0.65,3.0|262144,4096|TJ|-|Qwen3 235B A22B Thinking 2507 FP8 on Together AI|Y
together/Qwen/Qwen3-235B-A22B-fp8-tput|qwen3-235b-a22b-fp8-tput|Qwen3 235B A22B FP8 Throughput|C|0.20,0.60|40960,4096|TJ|-|Qwen3 235B A22B FP8 Throughput on Together AI|Y
together/Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8|qwen3-coder-480b-a35b-instruct-fp8|Qwen3 Coder 480B A35B Instruct Fp8|C|2.0,2.0|262144,4096|TJ|-|Qwen3 Coder 480B A35B Instruct Fp8 on Together AI|Y
together/Qwen/Qwen3-Next-80B-A3B-Instruct|qwen3-next-80b-a3b-instruct|Qwen3 Next 80B A3b Instruct|C|0.15,1.5|262144,4096|TJ|-|Qwen3 Next 80B A3b Instruct on Together AI|Y
together/Qwen/Qwen3-Next-80B-A3B-Thinking|qwen3-next-80b-a3b-thinking|Qwen3 Next 80B A3b Thinking|C|0.15,1.5|262144,4096|TJ|-|Qwen3 Next 80B A3b Thinking on Together AI|Y
together/Qwen/Qwen3-VL-32B-Instruct|qwen3-vl-32b-instruct|Qwen3-VL-32B-Instruct|C|0.50,1.5|262144,4096|VTJ|-|Qwen3-VL-32B-Instruct on Together AI|Y
together/Qwen/Qwen3-VL-8B-Instruct|qwen3-vl-8b-instruct|Qwen3-VL-8B-Instruct|C|0.18,0.68|262144,4096|VTJ|-|Qwen3-VL-8B-Instruct on Together AI|Y
together/RunDiffusion/Juggernaut-pro-flux|juggernaut-pro-flux|Juggernaut Pro Flux by RunDiffusion 1.0.0|C|-|4096,1024|I|-|Juggernaut Pro Flux by RunDiffusion 1.0.0 on Together AI|N
together/Rundiffusion/Juggernaut-Lightning-Flux|juggernaut-lightning-flux|Juggernaut Lightning Flux by RunDiffusion|C|-|4096,1024|I|-|Juggernaut Lightning Flux by RunDiffusion on Together AI|N
together/Salesforce/Llama-Rank-V1|llama-rank-v1|Salesforce Llama Rank V1 (8B)|C|0.10,0.10|8192,2048|R|-|Salesforce Llama Rank V1 (8B) on Together AI|Y
together/ServiceNow-AI/Apriel-1.5-15b-Thinker|apriel-1.5-15b-thinker|Apriel 1.5 15B Thinker|C|-|131072,4096|T|-|Apriel 1.5 15B Thinker on Together AI|Y
together/ServiceNow-AI/Apriel-1.6-15b-Thinker|apriel-1.6-15b-thinker|Apriel 1.6 15B Thinker|C|-|131072,4096|T|-|Apriel 1.6 15B Thinker on Together AI|Y
together/Virtue-AI/VirtueGuard-Text-Lite|virtueguard-text-lite|Virtueguard Text Lite|C|0.20,0.20|32768,4096|M|-|Virtueguard Text Lite on Together AI|N
together/Wan-AI/Wan2.2-I2V-A14B|wan2.2-i2v-a14b|Wan 2.2 I2V|C|-|4096,1024|D|-|Wan 2.2 I2V on Together AI|N
together/Wan-AI/Wan2.2-T2V-A14B|wan2.2-t2v-a14b|Wan 2.2 T2V|C|-|4096,1024|D|-|Wan 2.2 T2V on Together AI|N
together/arcee-ai/trinity-mini|trinity-mini|Trinity Mini|C|0.04,0.15|128000,4096|T|-|Trinity Mini on Together AI|Y
together/arize-ai/qwen-2-1.5b-instruct|qwen-2-1.5b-instruct|Arize AI Qwen 2 1.5B Instruct|C|0.10,0.10|32768,4096|TJ|-|Arize AI Qwen 2 1.5B Instruct on Together AI|Y
together/black-forest-labs/FLUX.1-dev|flux.1-dev|FLUX.1 [dev]|C|-|4096,1024|I|-|FLUX.1 [dev] on Together AI|N
together/black-forest-labs/FLUX.1-dev-lora|flux.1-dev-lora|FLUX.1 [dev] LoRA|C|-|4096,1024|I|-|FLUX.1 [dev] LoRA on Together AI|N
together/black-forest-labs/FLUX.1-kontext-dev|flux.1-kontext-dev|FLUX.1 Kontext [dev]|C|-|4096,1024|I|-|FLUX.1 Kontext [dev] on Together AI|N
together/black-forest-labs/FLUX.1-kontext-max|flux.1-kontext-max|FLUX.1 Kontext [max]|C|-|0,0|I|-|FLUX.1 Kontext [max] on Together AI|N
together/black-forest-labs/FLUX.1-kontext-pro|flux.1-kontext-pro|FLUX.1 Kontext [pro]|C|-|0,0|I|-|FLUX.1 Kontext [pro] on Together AI|N
together/black-forest-labs/FLUX.1-krea-dev|flux.1-krea-dev|FLUX.1 Krea [dev]|C|-|4096,1024|I|-|FLUX.1 Krea [dev] on Together AI|N
together/black-forest-labs/FLUX.1-pro|flux.1-pro|FLUX.1 [pro]|C|-|4096,1024|I|-|FLUX.1 [pro] on Together AI|N
together/black-forest-labs/FLUX.1-schnell|flux.1-schnell|FLUX.1 Schnell|C|-|4096,1024|I|-|FLUX.1 Schnell on Together AI|N
together/black-forest-labs/FLUX.1.1-pro|flux.1.1-pro|FLUX1.1 [pro]|C|-|4096,1024|I|-|FLUX1.1 [pro] on Together AI|N
together/black-forest-labs/FLUX.2-dev|flux.2-dev|FLUX.2 [dev]|C|-|4096,1024|I|-|FLUX.2 [dev] on Together AI|N
together/black-forest-labs/FLUX.2-flex|flux.2-flex|FLUX.2 [flex]|C|-|4096,1024|I|-|FLUX.2 [flex] on Together AI|N
together/black-forest-labs/FLUX.2-pro|flux.2-pro|FLUX.2 [pro]|C|-|4096,1024|I|-|FLUX.2 [pro] on Together AI|N
together/canopylabs/orpheus-3b-0.1-ft|orpheus-3b-0.1-ft|Orpheus 3B 0.1 FT|C|0.27,0.85|4096,1024|A|-|Orpheus 3B 0.1 FT on Together AI|N
together/cartesia/sonic|sonic|Cartesia Sonic|C|65.0,|0,0|A|-|Cartesia Sonic on Together AI|N
together/cartesia/sonic-2|sonic-2|Cartesia Sonic 2|C|65.0,|0,0|A|-|Cartesia Sonic 2 on Together AI|N
together/dbrx-instruct|dbrx-instruct-t|Databricks: DBRX (Together)|C|0.0006,0.0006|32768,2048|VSTJ|-|Databricks DBRX via Together|Y
together/deepcogito/cogito-v2-1-671b|cogito-v2-1-671b|Cogito v2.1 671B|C|1.2,1.2|163840,4096|T|-|Cogito v2.1 671B on Together AI|Y
together/deepcogito/cogito-v2-preview-llama-109B-MoE|cogito-v2-preview-llama-109b-moe|Cogito V2 Preview Llama 109B MoE|C|0.18,0.59|32767,4096|T|-|Cogito V2 Preview Llama 109B MoE on Together AI|Y
together/deepcogito/cogito-v2-preview-llama-405B|cogito-v2-preview-llama-405b|Deepcogito Cogito V2 Preview Llama 405B|C|3.5,3.5|32768,4096|T|-|Deepcogito Cogito V2 Preview Llama 405B on Together AI|Y
together/deepcogito/cogito-v2-preview-llama-70B|cogito-v2-preview-llama-70b|Deepcogito Cogito V2 Preview Llama 70B|C|0.88,0.88|32768,4096|T|-|Deepcogito Cogito V2 Preview Llama 70B on Together AI|Y
together/deepseek-ai/DeepSeek-R1|deepseek-r1|DeepSeek R1-0528|C|3.0,7.0|163840,4096|TJK|-|DeepSeek R1-0528 on Together AI|N
together/deepseek-ai/DeepSeek-R1-0528-tput|deepseek-r1-0528-tput|DeepSeek R1 0528 Throughput|C|0.55,2.2|163840,4096|TJK|-|DeepSeek R1 0528 Throughput on Together AI|N
together/deepseek-ai/DeepSeek-R1-Distill-Llama-70B|deepseek-r1-distill-llama-70b|DeepSeek R1 Distill Llama 70B|C|2.0,2.0|131072,4096|TJK|-|DeepSeek R1 Distill Llama 70B on Together AI|N
together/deepseek-ai/DeepSeek-V3|deepseek-v3|DeepSeek V3-0324|C|1.2,1.2|131072,4096|TJS|-|DeepSeek V3-0324 on Together AI|Y
together/deepseek-ai/DeepSeek-V3.1|deepseek-v3.1|Deepseek V3.1|C|0.60,1.7|131072,4096|TJS|-|Deepseek V3.1 on Together AI|Y
together/essentialai/rnj-1-instruct|rnj-1-instruct|EssentialAI Rnj-1 Instruct|C|0.15,0.15|32768,4096|TJ|-|EssentialAI Rnj-1 Instruct on Together AI|Y
together/google/flash-image-2.5|flash-image-2.5|Gemini Flash Image 2.5 (Nano Banana)|C|-|4096,1024|I|-|Gemini Flash Image 2.5 (Nano Banana) on Together AI|N
together/google/gemini-3-pro-image|gemini-3-pro-image|Gemini 3 (Nano Banana 2 Pro)|C|-|4096,1024|I|-|Gemini 3 (Nano Banana 2 Pro) on Together AI|N
together/google/gemma-3n-E4B-it|gemma-3n-e4b-it|Gemma 3N E4B Instruct|C|0.02,0.04|32768,4096|T|-|Gemma 3N E4B Instruct on Together AI|Y
together/google/imagen-4.0-fast|imagen-4.0-fast|Google Imagen 4.0 Fast|C|-|4096,1024|I|-|Google Imagen 4.0 Fast on Together AI|N
together/google/imagen-4.0-preview|imagen-4.0-preview|Google Imagen 4.0 Preview|C|-|4096,1024|I|-|Google Imagen 4.0 Preview on Together AI|N
together/google/imagen-4.0-ultra|imagen-4.0-ultra|Google Imagen 4.0 Ultra|C|-|4096,1024|I|-|Google Imagen 4.0 Ultra on Together AI|N
together/google/veo-2.0|veo-2.0|Google Veo 2.0|C|-|4096,1024|D|-|Google Veo 2.0 on Together AI|N
together/google/veo-3.0|veo-3.0|Google Veo 3.0|C|-|4096,1024|D|-|Google Veo 3.0 on Together AI|N
together/google/veo-3.0-audio|veo-3.0-audio|Google Veo 3.0 + Audio|C|-|4096,1024|D|-|Google Veo 3.0 + Audio on Together AI|N
together/google/veo-3.0-fast|veo-3.0-fast|Google Veo 3.0 Fast|C|-|4096,1024|D|-|Google Veo 3.0 Fast on Together AI|N
together/google/veo-3.0-fast-audio|veo-3.0-fast-audio|Google Veo 3.0 Fast + Audio|C|-|4096,1024|D|-|Google Veo 3.0 Fast + Audio on Together AI|N
together/hexgrad/Kokoro-82M|kokoro-82m|Kokoro 82M|C|4.0,|4096,1024|A|-|Kokoro 82M on Together AI|N
together/ideogram/ideogram-3.0|ideogram-3.0|Ideogram 3.0|C|-|4096,1024|I|-|Ideogram 3.0 on Together AI|N
together/intfloat/multilingual-e5-large-instruct|multilingual-e5-large-instruct|Multilingual E5 Large Instruct|C|0.02,0.02|514,128|E|-|Multilingual E5 Large Instruct on Together AI|N
together/kwaivgI/kling-1.6-pro|kling-1.6-pro|Kling 1.6 Pro|C|-|4096,1024|D|-|Kling 1.6 Pro on Together AI|N
together/kwaivgI/kling-1.6-standard|kling-1.6-standard|Kling 1.6 Standard|C|-|4096,1024|D|-|Kling 1.6 Standard on Together AI|N
together/kwaivgI/kling-2.0-master|kling-2.0-master|Kling 2.0 Master|C|-|4096,1024|D|-|Kling 2.0 Master on Together AI|N
together/kwaivgI/kling-2.1-master|kling-2.1-master|Kling 2.1 Master|C|-|4096,1024|D|-|Kling 2.1 Master on Together AI|N
together/kwaivgI/kling-2.1-pro|kling-2.1-pro|Kling 2.1 Pro|C|-|4096,1024|D|-|Kling 2.1 Pro on Together AI|N
together/kwaivgI/kling-2.1-standard|kling-2.1-standard|Kling 2.1 Standard|C|-|4096,1024|D|-|Kling 2.1 Standard on Together AI|N
together/llama-3-70b-instruct|llama-3-70b-t|Meta: Llama 3 70B (Together)|C|0.0009,0.0009|8192,2048|VSTJ|-|Llama 3 70B via Together AI|Y
together/llava-1.6-13b|llava-1.6-13b|Together: LLaVA 1.6 13B|C|0.0000,0.0000|4096,2048|VST|-|Efficient LLaVA vision model with 13B parameters|Y
together/llava-1.6-34b|llava-1.6-34b|Together: LLaVA 1.6 34B|C|0.0000,0.0000|4096,2048|VST|-|Open-source LLaVA vision model with 34B parameters|Y
together/llava-onevision-72b|llava-onevision-72b|Together: LLaVA OneVision 72B|C|0.0000,0.0000|4096,4096|VST|-|Large-scale LLaVA OneVision model|Y
together/llava-onevision-7b|llava-onevision-7b|Together: LLaVA OneVision 7B|C|0.0000,0.0000|4096,1024|VST|-|Latest LLaVA OneVision compact model|Y
together/marin-community/marin-8b-instruct|marin-8b-instruct|Marin 8B Instruct|C|0.18,0.18|4096,1024|TJ|-|Marin 8B Instruct on Together AI|Y
together/meta-llama/Llama-3-70b-chat-hf|llama-3-70b-chat-hf|Meta Llama 3 70B Instruct Reference|C|0.88,0.88|8192,2048|TJ|-|Meta Llama 3 70B Instruct Reference on Together AI|Y
together/meta-llama/Llama-3-70b-hf|llama-3-70b-hf|Meta Llama 3 70B HF|C|0.90,0.90|8192,2048|T|-|Meta Llama 3 70B HF on Together AI|Y
together/meta-llama/Llama-3.1-405B-Instruct|llama-3.1-405b-instruct|Meta Llama 3.1 405B Instruct|C|3.5,3.5|4096,1024|TJS|-|Meta Llama 3.1 405B Instruct on Together AI|Y
together/meta-llama/Llama-3.2-1B-Instruct|llama-3.2-1b-instruct|Meta Llama 3.2 1B Instruct|C|0.06,0.06|131072,4096|TJS|-|Meta Llama 3.2 1B Instruct on Together AI|Y
together/meta-llama/Llama-3.2-3B-Instruct-Turbo|llama-3.2-3b-instruct-turbo|Meta Llama 3.2 3B Instruct Turbo|C|0.06,0.06|131072,4096|TJS|-|Meta Llama 3.2 3B Instruct Turbo on Together AI|Y
together/meta-llama/Llama-3.3-70B-Instruct-Turbo|llama-3.3-70b-instruct-turbo|Meta Llama 3.3 70B Instruct Turbo|C|0.88,0.88|131072,4096|TJS|-|Meta Llama 3.3 70B Instruct Turbo on Together AI|Y
together/meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8|llama-4-maverick-17b-128e-instruct-fp8|Llama 4 Maverick Instruct (17Bx128E)|C|0.27,0.85|1048576,4096|TJ|-|Llama 4 Maverick Instruct (17Bx128E) on Together AI|Y
together/meta-llama/Llama-4-Scout-17B-16E-Instruct|llama-4-scout-17b-16e-instruct|Llama 4 Scout Instruct (17Bx16E)|C|0.18,0.59|1048576,4096|TJ|-|Llama 4 Scout Instruct (17Bx16E) on Together AI|Y
together/meta-llama/Llama-Guard-3-11B-Vision-Turbo|llama-guard-3-11b-vision-turbo|Meta Llama Guard 3 11B Vision Turbo|C|0.18,0.18|131072,4096|M|-|Meta Llama Guard 3 11B Vision Turbo on Together AI|N
together/meta-llama/Llama-Guard-4-12B|llama-guard-4-12b|Llama Guard 4 12B|C|0.20,0.20|1048576,4096|M|-|Llama Guard 4 12B on Together AI|N
together/meta-llama/LlamaGuard-2-8b|llamaguard-2-8b|Meta Llama Guard 2 8B|C|0.20,0.20|8192,2048|M|-|Meta Llama Guard 2 8B on Together AI|N
together/meta-llama/Meta-Llama-3-8B-Instruct|meta-llama-3-8b-instruct|Meta Llama 3 8B Instruct|C|0.20,0.20|8192,2048|TJ|-|Meta Llama 3 8B Instruct on Together AI|Y
together/meta-llama/Meta-Llama-3-8B-Instruct-Lite|meta-llama-3-8b-instruct-lite|Meta Llama 3 8B Instruct Lite|C|0.10,0.10|8192,2048|TJ|-|Meta Llama 3 8B Instruct Lite on Together AI|Y
together/meta-llama/Meta-Llama-3.1-405B-Instruct-Lite-Pro|meta-llama-3.1-405b-instruct-lite-pro|Meta Llama 3.1 405B Instruct Turbo|C|-|4096,1024|TJS|-|Meta Llama 3.1 405B Instruct Turbo on Together AI|Y
together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo|meta-llama-3.1-405b-instruct-turbo|Meta Llama 3.1 405B Instruct Turbo|C|3.5,3.5|10000,2500|TJS|-|Meta Llama 3.1 405B Instruct Turbo on Together AI|Y
together/meta-llama/Meta-Llama-3.1-70B-Instruct-Reference|meta-llama-3.1-70b-instruct-reference|Meta Llama 3.1 70B Instruct|C|0.90,0.90|8192,2048|TJS|-|Meta Llama 3.1 70B Instruct on Together AI|Y
together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo|meta-llama-3.1-70b-instruct-turbo|Meta Llama 3.1 70B Instruct Turbo|C|0.88,0.88|131072,4096|TJS|-|Meta Llama 3.1 70B Instruct Turbo on Together AI|Y
together/meta-llama/Meta-Llama-3.1-8B-Instruct-Reference|meta-llama-3.1-8b-instruct-reference|Meta Llama 3.1 8B Instruct|C|0.20,0.20|16384,4096|TJS|-|Meta Llama 3.1 8B Instruct on Together AI|Y
together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo|meta-llama-3.1-8b-instruct-turbo|Meta Llama 3.1 8B Instruct Turbo|C|0.18,0.18|131072,4096|TJS|-|Meta Llama 3.1 8B Instruct Turbo on Together AI|Y
together/minimax/hailuo-02|hailuo-02|MiniMax Hailuo 02|C|-|4096,1024|D|-|MiniMax Hailuo 02 on Together AI|N
together/minimax/video-01-director|video-01-director|MiniMax 01 Director|C|-|4096,1024|D|-|MiniMax 01 Director on Together AI|N
together/mistralai/Ministral-3-14B-Instruct-2512|ministral-3-14b-instruct-2512|Ministral 3 14B Instruct 2512|C|0.20,0.20|262144,4096|TJ|-|Ministral 3 14B Instruct 2512 on Together AI|Y
together/mistralai/Mistral-7B-Instruct-v0.2|mistral-7b-instruct-v0.2|Mistral (7B) Instruct v0.2|C|0.20,0.20|32768,4096|TJ|-|Mistral (7B) Instruct v0.2 on Together AI|Y
together/mistralai/Mistral-7B-Instruct-v0.3|mistral-7b-instruct-v0.3|Mistral (7B) Instruct v0.3|C|0.20,0.20|32768,4096|TJ|-|Mistral (7B) Instruct v0.3 on Together AI|Y
together/mistralai/Mistral-Small-24B-Instruct-2501|mistral-small-24b-instruct-2501|Mistral Small (24B) Instruct 25.01|C|0.10,0.30|32768,4096|TJ|-|Mistral Small (24B) Instruct 25.01 on Together AI|Y
together/mistralai/Mixtral-8x7B-Instruct-v0.1|mixtral-8x7b-instruct-v0.1|Mixtral-8x7B Instruct v0.1|C|0.60,0.60|32768,4096|TJ|-|Mixtral-8x7B Instruct v0.1 on Together AI|Y
together/mixedbread-ai/Mxbai-Rerank-Large-V2|mxbai-rerank-large-v2|Mxbai Rerank Large V2|C|0.10,0.10|32768,4096|R|-|Mxbai Rerank Large V2 on Together AI|Y
together/mixtral-8x22b-instruct|mixtral-8x22b-t|Mistral: Mixtral 8x22B (Together)|C|0.0009,0.0009|65536,2048|VSTJ|-|Mixtral 8x22B sparse mixture via Together|Y
together/moonshotai/Kimi-K2-Instruct-0905|kimi-k2-instruct-0905|Kimi K2-Instruct 0905|C|1.0,3.0|262144,4096|TJ|-|Kimi K2-Instruct 0905 on Together AI|Y
together/moonshotai/Kimi-K2-Thinking|kimi-k2-thinking|Kimi K2 Thinking|C|1.2,4.0|262144,4096|T|-|Kimi K2 Thinking on Together AI|Y
together/nvidia/NVIDIA-Nemotron-Nano-9B-v2|nvidia-nemotron-nano-9b-v2|Nvidia Nemotron Nano 9B V2|C|0.06,0.25|131072,4096|T|-|Nvidia Nemotron Nano 9B V2 on Together AI|Y
together/openai/gpt-oss-120b|gpt-oss-120b|OpenAI GPT-OSS 120B|C|0.15,0.60|131072,4096|T|-|OpenAI GPT-OSS 120B on Together AI|Y
together/openai/gpt-oss-20b|gpt-oss-20b|OpenAI GPT-OSS 20B|C|0.05,0.20|131072,4096|T|-|OpenAI GPT-OSS 20B on Together AI|Y
together/openai/sora-2|sora-2|Sora 2|C|-|4096,1024|D|-|Sora 2 on Together AI|N
together/openai/sora-2-pro|sora-2-pro|Sora 2 Pro|C|-|4096,1024|D|-|Sora 2 Pro on Together AI|N
together/openai/whisper-large-v3|whisper-large-v3|Whisper large-v3|C|0.27,0.85|4096,1024|A|-|Whisper large-v3 on Together AI|N
together/phi-3-mini-instruct|phi-3-mini-t|Microsoft: Phi-3 Mini (Together)|C|0.0001,0.0001|131072,2048|VSTJ|-|Lightweight Phi-3 Mini via Together|Y
together/pixverse/pixverse-v5|pixverse-v5|PixVerse v5|C|-|4096,1024|D|-|PixVerse v5 on Together AI|N
together/scb10x/scb10x-typhoon-2-1-gemma3-12b|scb10x-typhoon-2-1-gemma3-12b|Typhoon 2.1 12B|C|0.20,0.20|131072,4096|T|-|Typhoon 2.1 12B on Together AI|Y
together/stabilityai/stable-diffusion-3-medium|stable-diffusion-3-medium|Stable Diffusion 3|C|-|4096,1024|I|-|Stable Diffusion 3 on Together AI|N
together/stabilityai/stable-diffusion-xl-base-1.0|stable-diffusion-xl-base-1.0|SD XL|C|-|4096,1024|I|-|SD XL on Together AI|N
together/togethercomputer/MoA-1|moa-1|Together AI MoA-1|C|-|32768,4096|T|-|Together AI MoA-1 on Together AI|Y
together/togethercomputer/MoA-1-Turbo|moa-1-turbo|Together AI MoA-1-Turbo|C|-|32768,4096|T|-|Together AI MoA-1-Turbo on Together AI|Y
together/togethercomputer/Refuel-Llm-V2|refuel-llm-v2|Refuel LLM V2|C|0.60,0.60|16384,4096|T|-|Refuel LLM V2 on Together AI|Y
together/togethercomputer/Refuel-Llm-V2-Small|refuel-llm-v2-small|Refuel LLM V2 Small|C|0.20,0.20|8192,2048|T|-|Refuel LLM V2 Small on Together AI|Y
together/togethercomputer/m2-bert-80M-32k-retrieval|m2-bert-80m-32k-retrieval|M2-BERT-Retrieval-32k|C|8000.0,8000.0|32768,4096|E|-|M2-BERT-Retrieval-32k on Together AI|N
together/vidu/vidu-2.0|vidu-2.0|Vidu 2.0|C|-|4096,1024|D|-|Vidu 2.0 on Together AI|N
together/vidu/vidu-q1|vidu-q1|Vidu Q1|C|-|4096,1024|D|-|Vidu Q1 on Together AI|N
together/yi-large-turbo|yi-large-turbo-t|01.AI: Yi Large Turbo|C|0.0009,0.0009|200000,2048|VSTJ|-|Yi Large Turbo via Together|Y
together/zai-org/GLM-4.5-Air-FP8|glm-4.5-air-fp8|Glm 4.5 Air Fp8|C|0.20,1.1|131072,4096|T|-|Glm 4.5 Air Fp8 on Together AI|Y
together/zai-org/GLM-4.6|glm-4.6|Glm 4.6 Fp8|C|0.60,2.2|202752,4096|T|-|Glm 4.6 Fp8 on Together AI|Y

# =============================================================================
# OPENROUTER - Aggregator (2 models)
# =============================================================================
openrouter/auto|auto|Auto Router|C|-|2000000,500000|-|-|Your prompt will be processed by a meta-model and routed to one of dozens of mod|Y
openrouter/bodybuilder|bodybuilder|Body Builder (beta)|C|-|128000,32000|-|-|Transform your natural language requests into structured OpenRouter API request|Y

# =============================================================================
# AWS BEDROCK - Cloud (106 models)
# =============================================================================
bedrock/ai21.j2-mid-v1|jurassic-2-mid|AI21 Jurassic-2 Mid|L|12.5,12.5|8191,8191|-|-|Legacy AI21 mid-tier model|Y
bedrock/ai21.j2-ultra-v1|jurassic-2-ultra|AI21 Jurassic-2 Ultra|L|18.8,18.8|8191,8191|-|-|Legacy AI21 flagship model|Y
bedrock/ai21.jamba-1-5-large-v1:0|jamba-1-5-large|AI21 Jamba 1.5 Large|C|2.0,8.0|256000,4096|TJ|-|AI21 large hybrid SSM-Transformer model|Y
bedrock/ai21.jamba-1-5-mini-v1:0|jamba-1-5-mini|AI21 Jamba 1.5 Mini|C|0.20,0.40|256000,4096|TJ|-|AI21 efficient hybrid model|Y
bedrock/ai21.jamba-instruct-v1:0|jamba-instruct|AI21 Jamba Instruct|C|0.50,0.70|256000,4096|TJ|-|AI21 instruction-tuned Jamba|Y
bedrock/amazon-nova-lite-vision|bedrock-nova-lite-vision|AWS: Nova Lite Vision|C|0.0001,0.0002|300000,5000|VSTJ|-|Lightweight Nova vision model|Y
bedrock/amazon-nova-pro-vision|bedrock-nova-pro-vision|AWS: Nova Pro Vision|C|0.0008,0.0032|300000,5000|VSTJ|-|Amazon Nova Pro multimodal model via Bedrock|Y
bedrock/amazon.nova-2-lite-v1:0|nova-2-lite|Amazon Nova 2 Lite|C|0.08,0.32|300000,5000|SVTJ|-|Next-gen efficient multimodal|Y
bedrock/amazon.nova-2-pro-v1:0|nova-2-pro|Amazon Nova 2 Pro|C|1.0,4.0|300000,5000|SVTJK|-|Next-gen advanced multimodal|Y
bedrock/amazon.nova-canvas-v1:0|nova-canvas|Amazon Nova Canvas|C|0.04,|0,0|I|-|State-of-art image generation|N
bedrock/amazon.nova-lite-v1:0|nova-lite|Amazon Nova Lite|C|0.06,0.24|300000,5000|SVTJ|-|Fast and cost-effective multimodal|Y
bedrock/amazon.nova-micro-v1:0|nova-micro|Amazon Nova Micro|C|0.04,0.14|128000,5000|TJ|-|Text-only fastest and lowest cost|Y
bedrock/amazon.nova-premier-v1:0|nova-premier|Amazon Nova Premier|C|2.5,10.0|1000000,5000|SVTJK|-|Most capable Nova for complex tasks|Y
bedrock/amazon.nova-pro-v1:0|nova-pro|Amazon Nova Pro|C|0.80,3.2|300000,5000|SVTJ|-|Advanced multimodal understanding and generation|Y
bedrock/amazon.nova-reel-v1:0|nova-reel|Amazon Nova Reel|C|0.80,|0,0|D|-|Studio-quality video generation|N
bedrock/amazon.nova-sonic-v1:0|nova-sonic|Amazon Nova Sonic|C|4.9,7.5|0,0|A|-|Streaming speech-to-speech|N
bedrock/amazon.rerank-v1:0|amazon-rerank|Amazon Rerank|C|1.0,|32000,0|R|-|Semantic reranking model|Y
bedrock/amazon.titan-embed-image-v1|titan-embed-image|Amazon Titan Multimodal Embeddings|C|0.80,|128,1024|VE|-|Image+text embeddings|N
bedrock/amazon.titan-embed-text-v1|titan-embed-v1|Amazon Titan Text Embeddings V1|C|0.10,|8192,1536|E|-|Titan embeddings v1|N
bedrock/amazon.titan-embed-text-v2:0|titan-embed-v2|Amazon Titan Text Embeddings V2|C|0.02,|8192,1024|E|-|Latest Titan embeddings|N
bedrock/amazon.titan-image-generator-v1|titan-image-v1|Amazon Titan Image Generator V1|C|0.01,|0,0|I|-|Image generation|N
bedrock/amazon.titan-image-generator-v2:0|titan-image-v2|Amazon Titan Image Generator V2|C|0.0080,|0,0|I|-|Advanced image generation|N
bedrock/amazon.titan-text-express-v1|titan-text-express|Amazon Titan Text Express|C|0.20,0.60|8000,4096|-|-|Balanced Titan text model|Y
bedrock/amazon.titan-text-lite-v1|titan-text-lite|Amazon Titan Text Lite|C|0.15,0.20|4000,4000|-|-|Lightweight Titan text|Y
bedrock/amazon.titan-text-premier-v1:0|titan-text-premier|Amazon Titan Text Premier|C|0.50,1.5|32000,3072|TJ|-|Titan flagship text model|Y
bedrock/anthropic.claude-3-5-haiku-v1:0|claude-3-5-haiku|Claude 3.5 Haiku|C|0.80,4.0|200000,8192|SVTJC|-|Fast Claude 3.5 model|Y
bedrock/anthropic.claude-3-5-sonnet-v1:0|claude-3-5-sonnet-v1|Claude 3.5 Sonnet V1|C|3.0,15.0|200000,8192|SVTJC|-|Original Claude 3.5 Sonnet|Y
bedrock/anthropic.claude-3-5-sonnet-v2:0|claude-3-5-sonnet-v2|Claude 3.5 Sonnet V2|C|3.0,15.0|200000,8192|SVTJKC|-|Latest 3.5 Sonnet with computer use|Y
bedrock/anthropic.claude-3-7-sonnet-v1:0|claude-3-7-sonnet|Claude 3.7 Sonnet|C|3.0,15.0|200000,8192|SVTJKC|-|Enhanced Claude 3.5 successor|Y
bedrock/anthropic.claude-3-haiku-v1:0|claude-3-haiku|Claude 3 Haiku|C|0.25,1.2|200000,4096|SVTJC|-|Claude 3 fastest|Y
bedrock/anthropic.claude-3-opus-v1:0|claude-3-opus|Claude 3 Opus|C|15.0,75.0|200000,4096|SVTJC|-|Claude 3 most capable|Y
bedrock/anthropic.claude-3-sonnet-v1:0|claude-3-sonnet|Claude 3 Sonnet|L|3.0,15.0|200000,4096|SVTJC|-|Claude 3 balanced|Y
bedrock/anthropic.claude-instant-v1|claude-instant|Claude Instant|L|0.80,2.4|100000,4096|-|-|Legacy fast Claude|Y
bedrock/anthropic.claude-opus-4-1-v1:0|claude-4-1-opus|Claude 4.1 Opus|C|15.0,75.0|200000,32000|SVTJKC|-|Most powerful Claude model|Y
bedrock/anthropic.claude-opus-4-v1:0|claude-4-opus|Claude 4 Opus|C|15.0,75.0|200000,32000|SVTJKC|-|Claude 4 flagship|Y
bedrock/anthropic.claude-sonnet-4-5-v1:0|claude-4-5-sonnet|Claude 4.5 Sonnet|C|3.0,15.0|200000,8192|SVTJKC|-|Best for complex coding and analysis|Y
bedrock/anthropic.claude-sonnet-4-v1:0|claude-4-sonnet|Claude 4 Sonnet|C|3.0,15.0|200000,8192|SVTJKC|-|Claude 4 balanced model|Y
bedrock/anthropic.claude-v2|claude-2|Claude 2|L|8.0,24.0|100000,4096|-|-|Legacy Claude 2|Y
bedrock/anthropic.claude-v2:1|claude-2-1|Claude 2.1|L|8.0,24.0|200000,4096|T|-|Legacy Claude 2.1|Y
bedrock/claude-3.5-sonnet-vision|bedrock-claude-3.5-vision|AWS: Claude 3.5 Sonnet Vision|C|0.0030,0.01|200000,4096|VSTJKC|-|Claude 3.5 Sonnet deployed via AWS Bedrock|Y
bedrock/cohere.command-light-text-v14|command-light|Cohere Command Light|L|0.30,0.60|4096,4096|-|-|Legacy Cohere light model|Y
bedrock/cohere.command-r-plus-v1:0|command-r-plus|Cohere Command R+|C|2.5,10.0|128000,4000|TJS|-|Cohere flagship RAG model|Y
bedrock/cohere.command-r-v1:0|command-r|Cohere Command R|C|0.50,1.5|128000,4000|TJS|-|Cohere efficient RAG model|Y
bedrock/cohere.command-text-v14|command-text|Cohere Command|L|1.5,2.0|4096,4096|-|-|Legacy Cohere command|Y
bedrock/cohere.embed-english-v3|cohere-embed-en-v3|Cohere Embed English V3|C|0.10,|512,1024|E|-|English embeddings|N
bedrock/cohere.embed-multilingual-v3|cohere-embed-multi-v3|Cohere Embed Multilingual V3|C|0.10,|512,1024|E|-|Multilingual embeddings|N
bedrock/cohere.rerank-multilingual-v3:0|cohere-rerank-multi|Cohere Rerank Multilingual|C|2.0,|4096,0|R|-|Multilingual reranking|Y
bedrock/cohere.rerank-v3-5:0|cohere-rerank-v3-5|Cohere Rerank 3.5|C|2.0,|4096,0|R|-|Latest Cohere reranker|Y
bedrock/deepseek.deepseek-r1-v1:0|deepseek-r1|DeepSeek R1|C|1.4,5.6|128000,16384|TJK|-|DeepSeek reasoning model|N
bedrock/deepseek.deepseek-v3-1-v1:0|deepseek-v3-1|DeepSeek V3.1|C|0.27,1.1|128000,16384|TJS|-|DeepSeek efficient MoE|Y
bedrock/google.gemma-2-27b-it-v1:0|gemma-2-27b|Google Gemma 2 27B IT|C|0.30,0.35|8192,8192|-|-|Gemma 2 large model|Y
bedrock/google.gemma-3-12b-it-v1:0|gemma-3-12b|Google Gemma 3 12B IT|C|0.10,0.15|128000,8192|VT|-|Gemma 3 medium model|Y
bedrock/google.gemma-3-27b-it-v1:0|gemma-3-27b|Google Gemma 3 27B IT|C|0.30,0.35|128000,8192|VT|-|Gemma 3 large model|Y
bedrock/google.gemma-3-4b-it-v1:0|gemma-3-4b|Google Gemma 3 4B IT|C|0.06,0.08|128000,8192|-|-|Gemma 3 small model|Y
bedrock/google.gemma-7b-it-v1:0|gemma-7b|Google Gemma 7B IT|L|0.07,0.14|8192,8192|-|-|Legacy Gemma instruction|Y
bedrock/luma.ray-v2:0|luma-ray-v2|Luma Ray V2|C|0.65,|0,0|D|-|Fast video generation|N
bedrock/meta.llama2-13b-chat-v1|llama-2-13b|Llama 2 13B Chat|L|0.75,1.0|4096,2048|-|-|Legacy Llama 2 small|Y
bedrock/meta.llama2-70b-chat-v1|llama-2-70b|Llama 2 70B Chat|L|1.9,2.6|4096,2048|-|-|Legacy Llama 2|Y
bedrock/meta.llama3-1-405b-instruct-v1:0|llama-3-1-405b|Llama 3.1 405B Instruct|C|5.3,16.0|128000,8192|TJS|-|Llama 3.1 largest|Y
bedrock/meta.llama3-1-70b-instruct-v1:0|llama-3-1-70b|Llama 3.1 70B Instruct|C|0.72,0.72|128000,8192|TJS|-|Llama 3.1 large|Y
bedrock/meta.llama3-1-8b-instruct-v1:0|llama-3-1-8b|Llama 3.1 8B Instruct|C|0.22,0.22|128000,8192|TJ|-|Llama 3.1 small|Y
bedrock/meta.llama3-2-11b-instruct-v1:0|llama-3-2-11b-vision|Llama 3.2 11B Vision|C|0.16,0.16|128000,4096|SVTJ|-|Llama 3.2 small vision|Y
bedrock/meta.llama3-2-1b-instruct-v1:0|llama-3-2-1b|Llama 3.2 1B Instruct|C|0.10,0.10|128000,4096|-|-|Llama 3.2 smallest|Y
bedrock/meta.llama3-2-3b-instruct-v1:0|llama-3-2-3b|Llama 3.2 3B Instruct|C|0.15,0.15|128000,4096|TJ|-|Llama 3.2 tiny|Y
bedrock/meta.llama3-2-90b-instruct-v1:0|llama-3-2-90b-vision|Llama 3.2 90B Vision|C|2.0,2.0|128000,4096|SVTJ|-|Llama 3.2 large vision|Y
bedrock/meta.llama3-3-70b-instruct-v1:0|llama-3-3-70b|Llama 3.3 70B Instruct|C|0.72,0.72|128000,8192|TJS|-|Llama 3.3 flagship|Y
bedrock/meta.llama3-70b-instruct-v1:0|llama-3-70b|Llama 3 70B Instruct|L|2.6,3.5|8192,2048|TJ|-|Legacy Llama 3 large|Y
bedrock/meta.llama3-8b-instruct-v1:0|llama-3-8b|Llama 3 8B Instruct|L|0.30,0.60|8192,2048|-|-|Legacy Llama 3 small|Y
bedrock/meta.llama4-maverick-17b-instruct-v1:0|llama-4-maverick|Llama 4 Maverick 17B|C|0.22,0.88|128000,8192|SVTJ|-|Llama 4 multimodal specialist|Y
bedrock/meta.llama4-scout-17b-instruct-v1:0|llama-4-scout|Llama 4 Scout 17B|C|0.17,0.68|1000000,8192|SVTJ|-|Llama 4 long context|Y
bedrock/minimax.minimax-m2-v1:0|minimax-m2|MiniMax M2|C|0.80,3.2|128000,8192|TJ|-|MiniMax flagship model|Y
bedrock/mistral.codestral-2501-v1:0|codestral-2501|Codestral 25.01|C|0.30,0.90|256000,16384|TJ|-|Code-specialized Mistral|Y
bedrock/mistral.ministral-3b-2410-v1:0|ministral-3b|Ministral 3B|C|0.04,0.04|128000,8192|-|-|Mistral tiny edge model|Y
bedrock/mistral.ministral-8b-2410-v1:0|ministral-8b|Ministral 8B|C|0.10,0.10|128000,8192|TJ|-|Mistral efficient edge model|Y
bedrock/mistral.mistral-7b-instruct-v0:2|mistral-7b|Mistral 7B Instruct|L|0.15,0.20|32000,8192|-|-|Legacy Mistral 7B|Y
bedrock/mistral.mistral-large-2407-v1:0|mistral-large-2407|Mistral Large 24.07|C|2.0,6.0|128000,8192|TJS|-|Mistral Large previous|Y
bedrock/mistral.mistral-large-2411-v1:0|mistral-large-2411|Mistral Large 24.11|C|2.0,6.0|128000,8192|VTJS|-|Latest Mistral Large|Y
bedrock/mistral.mistral-small-2409-v1:0|mistral-small-2409|Mistral Small 24.09|C|0.10,0.30|32000,8192|TJ|-|Cost-effective Mistral|Y
bedrock/mistral.mixtral-8x7b-instruct-v0:1|mixtral-8x7b|Mixtral 8x7B Instruct|L|0.45,0.70|32000,8192|-|-|Legacy Mixtral MoE|Y
bedrock/mistral.pixtral-12b-2409-v1:0|pixtral-12b|Pixtral 12B|C|0.15,0.15|128000,8192|SVT|-|Mistral small multimodal|Y
bedrock/mistral.pixtral-large-2502-v1:0|pixtral-large|Pixtral Large 25.02|C|2.0,6.0|128000,8192|SVTJS|-|Mistral multimodal flagship|Y
bedrock/mistral.voxtral-mini-3b-v1:0|voxtral-mini|Voxtral Mini 3B|C|0.05,0.05|32000,8192|A|-|Mistral speech model|N
bedrock/moonshot.kimi-k2-thinking-v1:0|kimi-k2-thinking|Moonshot Kimi K2 Thinking|C|0.60,2.4|131072,16384|TJK|-|Kimi reasoning model|N
bedrock/nvidia.nemotron-nano-8b-v1:0|nemotron-nano|NVIDIA Nemotron Nano 8B|C|0.10,0.10|128000,4096|TJ|-|NVIDIA efficient model|Y
bedrock/qwen.qwen3-14b-instruct-v1:0|qwen3-14b|Qwen3 14B Instruct|C|0.10,0.20|131072,8192|TJ|-|Qwen3 small model|Y
bedrock/qwen.qwen3-235b-instruct-v1:0|qwen3-235b|Qwen3 235B Instruct|C|1.5,4.0|131072,8192|TJK|-|Qwen3 largest model|N
bedrock/qwen.qwen3-32b-instruct-v1:0|qwen3-32b|Qwen3 32B Instruct|C|0.20,0.40|131072,8192|TJK|-|Qwen3 medium model|N
bedrock/qwen.qwen3-4b-instruct-v1:0|qwen3-4b|Qwen3 4B Instruct|C|0.04,0.08|131072,8192|-|-|Qwen3 tiny model|Y
bedrock/qwen.qwen3-72b-instruct-v1:0|qwen3-72b|Qwen3 72B Instruct|C|0.40,0.80|131072,8192|TJK|-|Qwen3 large model|N
bedrock/qwen.qwen3-8b-instruct-v1:0|qwen3-8b|Qwen3 8B Instruct|C|0.06,0.12|131072,8192|TJ|-|Qwen3 efficient model|Y
bedrock/stability.sd3-5-large-turbo-v1:0|sd3-5-turbo|Stable Diffusion 3.5 Turbo|C|0.04,|0,0|I|-|SD3.5 fast generation|N
bedrock/stability.sd3-5-large-v1:0|sd3-5-large|Stable Diffusion 3.5 Large|C|0.07,|0,0|I|-|SD3.5 large image generation|N
bedrock/stability.sd3-5-medium-v1:0|sd3-5-medium|Stable Diffusion 3.5 Medium|C|0.03,|0,0|I|-|SD3.5 medium generation|N
bedrock/stability.sd3-large-v1:0|sd3-large|Stable Diffusion 3 Large|C|0.08,|0,0|I|-|SD3 large generation|N
bedrock/stability.sd3-medium-v1:0|sd3-medium|Stable Diffusion 3 Medium|C|0.04,|0,0|I|-|SD3 medium generation|N
bedrock/stability.stable-diffusion-xl-v1|sdxl|Stable Diffusion XL|C|0.04,|0,0|I|-|SDXL image generation|N
bedrock/stability.stable-image-background-v1:0|stable-image-bg|Stable Image Background|C|0.04,|0,0|I|-|Background removal/replace|N
bedrock/stability.stable-image-control-v1:0|stable-image-control|Stable Image Control|C|0.04,|0,0|I|-|Sketch and structure to image|N
bedrock/stability.stable-image-core-v1:0|stable-image-core|Stable Image Core|C|0.04,|0,0|I|-|Fast image generation|N
bedrock/stability.stable-image-edit-v1:0|stable-image-edit|Stable Image Edit|C|0.04,|0,0|I|-|Inpaint and outpaint|N
bedrock/stability.stable-image-style-v1:0|stable-image-style|Stable Image Style|C|0.04,|0,0|I|-|Style transfer|N
bedrock/stability.stable-image-ultra-v1:0|stable-image-ultra|Stable Image Ultra|C|0.14,|0,0|I|-|Highest quality images|N
bedrock/twelvelabs.marengo-embed-v1:0|marengo-embed|TwelveLabs Marengo Embed|C|0.03,|0,1024|VE|-|Video embeddings|N
bedrock/twelvelabs.pegasus-1-2-v1:0|pegasus-1-2|TwelveLabs Pegasus 1.2|C|0.50,1.5|0,4096|D|-|Video understanding|N
bedrock/writer.palmyra-x4-v1:0|palmyra-x4|Writer Palmyra X4|C|2.0,6.0|128000,8192|TJS|-|Writer previous flagship|Y
bedrock/writer.palmyra-x5-v1:0|palmyra-x5|Writer Palmyra X5|C|4.0,12.0|128000,8192|TJS|-|Writer latest flagship|Y

# =============================================================================
# AI21 - Jamba (2 models)
# =============================================================================
ai21/jamba-large-1.7|jamba-large-1.7|AI21: Jamba Large 1.7|C|0.0000,0.0000|256000,4096|JT|-|Jamba Large 1.7 is the latest model in the Jamba open family, offering improveme|Y
ai21/jamba-mini-1.7|jamba-mini-1.7|AI21: Jamba Mini 1.7|C|0.0000,0.0000|256000,4096|JT|-|Jamba Mini 1.7 is a compact and efficient member of the Jamba open model family,|Y

# =============================================================================
# REPLICATE - Model Hub (3 models)
# =============================================================================
replicate/flan-t5-xl|flan-t5-xl-r|Google: FLAN-T5 XL (Replicate)|C|0.0001,0.0001|512,512|VTJ|-|FLAN-T5 XL on Replicate|Y
replicate/openhermes-2.5|openhermes-r|NousResearch: OpenHermes 2.5|C|0.0001,0.0003|4096,2048|VTJ|-|OpenHermes 2.5 via Replicate|Y
replicate/orca-mini-8b|orca-mini-r|Microsoft: Orca Mini 8B (Replicate)|C|0.0001,0.0003|8192,2048|VTJ|-|Orca Mini 8B on Replicate|Y

# =============================================================================
# HUGGINGFACE - Inference API (67 models)
# =============================================================================
huggingface/albert-base-v2|albert-base|albert-v2|Google: ALBERT Base|C|0.0000,0.0000|512,256|S|-|ALBERT lightweight model|Y
huggingface/bart-large-finetuned-scientific-abstractive-summarization|bart-sci-sum-ft|bart-sci-ft|Hugging Face: BART Scientific Sum FT|C|0.0000,0.0000|1024,512|T|-|BART fine-tuned for scientific paper summarization|Y
huggingface/bert-base-chinese-finetuned-nlcke|bert-chinese-ft|bert-zh-ft|Hugging Face: BERT Chinese NLCKE FT|C|0.0000,0.0000|512,256|S|-|BERT fine-tuned for Chinese entity extraction|Y
huggingface/biollm-7b|biollm-7b|biollm|DNABERT: BioLLM 7B|C|0.0000,0.0000|4096,2048|VS|-|Specialized model for biomedical NLP tasks|Y
huggingface/bloom-1b1|bloom-1b1|bloom-1b|BigScience: BLOOM 1.1B|C|0.0000,0.0000|2048,512|J|-|Smaller BLOOM variant|Y
huggingface/bloom-3b|bloom-3b|bloom|BigScience: BLOOM 3B|C|0.0000,0.0000|2048,1024|J|-|3B BLOOM for instruction tasks|Y
huggingface/bloom-560m|bloom-560m|bloom-small|BigScience: BLOOM 560M|C|0.0000,0.0000|2048,512|J|-|Lightweight BLOOM model for research|Y
huggingface/bloom-7b1|bloom-7b1|bloom-7b|BigScience: BLOOM 7B|C|0.0000,0.0000|2048,1024|JT|-|7B BLOOM multilingual model|Y
huggingface/codegemma-7b-it|codegemma-7b|codegemma|Google: CodeGemma 7B|C|0.0000,0.0000|8192,2048|JT|-|Google Gemma variant specialized for code|Y
huggingface/codellama-7b-instruct|codellama-7b|codellama|Meta: Code Llama 7B Instruct|C|0.0000,0.0000|8192,2048|JT|-|Specialized Llama variant for code generation|Y
huggingface/deberta-v3-base|deberta-base|deberta|Microsoft: DeBERTa v3 Base|C|0.0000,0.0000|512,256|S|-|DeBERTa disentangled attention|Y
huggingface/distilbert-base-multilingual-cased-finetuned-ner|distilbert-ner-ft|distil-ner|Hugging Face: DistilBERT NER FT|C|0.0000,0.0000|512,256|S|-|DistilBERT fine-tuned for multilingual NER|Y
huggingface/distilbert-base-uncased|distilbert|distilbert-base|HuggingFace: DistilBERT Base|C|0.0000,0.0000|512,256|S|-|Lightweight DistilBERT 66M parameters|Y
huggingface/distilgpt2-finetuned-wikitext|distilgpt2-wiki|distil-gpt2-ft|Hugging Face: DistilGPT2 WikiText FT|C|0.0000,0.0000|1024,512|T|-|DistilGPT2 fine-tuned on WikiText|Y
huggingface/distilroberta-base-finetuned-sst2|distilroberta-sst2|distil-robin-ft|Hugging Face: DistilRoBERTa SST2 FT|C|0.0000,0.0000|512,256|S|-|DistilRoBERTa fine-tuned for sentiment|Y
huggingface/dolly-v2-12b|dolly-v2-12b|dolly-12b|Databricks: Dolly v2 12B|C|0.0000,0.0000|4096,2048|JT|-|Databricks Dolly v2 12B instruction-tuned|Y
huggingface/dolly-v2-3b|dolly-v2-3b|dolly-3b|Databricks: Dolly v2 3B|C|0.0000,0.0000|2048,1024|J|-|Lightweight Dolly instruction model|Y
huggingface/electra-base-discriminator|electra-base|electra|Google: ELECTRA Base|C|0.0000,0.0000|512,256|S|-|ELECTRA replaced token detection|Y
huggingface/finbert-base-uncased|finbert|finbert-uncased|FinBERT: Base Uncased|C|0.0000,0.0000|512,256|S|-|BERT model for financial sentiment analysis|Y
huggingface/flan-t5-base|flan-t5-base|flan-base|Google: FLAN-T5 Base|C|0.0000,0.0000|512,256|T|-|FLAN instruction-tuned T5 base model|Y
huggingface/flan-t5-large|flan-t5-large|flan-large|Google: FLAN-T5 Large|C|0.0000,0.0000|512,512|T|-|Larger FLAN-T5 for complex instruction tasks|Y
huggingface/gpt-neo-2.7b|gpt-neo-2.7b|gpt-neo|EleutherAI: GPT-Neo 2.7B|C|0.0000,0.0000|2048,1024|J|-|EleutherAI GPT-Neo 2.7B model|Y
huggingface/gpt2-xl|gpt2-xl|gpt2-large|OpenAI: GPT-2 XL|C|0.0000,0.0000|1024,512|J|-|OpenAI GPT-2 XL 1.5B parameter model|Y
huggingface/idefics2-27b|idefics2-27b|Hugging Face: Idefics2 27B|C|0.0000,0.0000|4096,2048|VS|-|Larger Idefics2 for advanced vision tasks|Y
huggingface/idefics2-8b|idefics2-8b|Hugging Face: Idefics2 8B|C|0.0000,0.0000|4096,1024|VS|-|French-centric multimodal LLM with vision|Y
huggingface/instrublip-flan-t5-xl|instructblip-xl|Hugging Face: InstructBLIP XL|C|0.0000,0.0000|2048,512|VS|-|Open-source InstructBLIP model with vision-language understanding|Y
huggingface/japanese-bert-finetuned-dep-parser|bert-japanese-ft|bert-ja-ft|Hugging Face: Japanese BERT Dep FT|C|0.0000,0.0000|512,256|S|-|Japanese BERT fine-tuned for dependency parsing|Y
huggingface/lawbert-base-uncased|lawbert|lawbert-uncased|LawBERT: Base Uncased|C|0.0000,0.0000|512,256|S|-|BERT model specialized for legal document analysis|Y
huggingface/llama-2-13b-hf|llama2-13b|llama2-13b-hf|Meta: Llama 2 13B HF|C|0.0000,0.0000|4096,2048|JT|-|Llama 2 13B base model from HuggingFace|Y
huggingface/llama-2-7b-hf|llama2-7b|llama2-7b-hf|Meta: Llama 2 7B HF|C|0.0000,0.0000|4096,2048|JT|-|Llama 2 7B from HuggingFace collection|Y
huggingface/medalpaca-7b|medalpaca|medalpaca-7b|MedAlpaca: 7B|C|0.0000,0.0000|4096,2048|J|-|Specialized medical domain instruction model|Y
huggingface/mobilenet-v2-finetuned-imagenet|mobilenet-v2-ft|mobile-net-ft|Hugging Face: MobileNet v2 ImageNet FT|C|0.0000,0.0000|224,256|V|-|MobileNet v2 fine-tuned on ImageNet|Y
huggingface/moondream2|moondream2|Moondream: Moondream2|C|0.0000,0.0000|2048,512|VS|-|Ultra-lightweight vision model optimized for edge|Y
huggingface/mpt-30b-instruct|mpt-30b|mpt-30b-instruct|MosaicML: MPT 30B Instruct|C|0.0000,0.0000|8192,4096|JT|-|Larger MosaicML model for complex tasks|Y
huggingface/mpt-7b-instruct|mpt-7b-instruct|mpt-7b|MosaicML: MPT 7B Instruct|C|0.0000,0.0000|8192,2048|JT|-|MosaicML open foundation model with 8K context|Y
huggingface/neox-20b|neox-20b|gpt-neox-20b|EleutherAI: GPT-NeoX 20B|C|0.0000,0.0000|8192,2048|JT|-|EleutherAI 20B parameter autoregressive language model|Y
huggingface/neural-7b-chat|neural-7b|Intel: Neural 7B Chat|C|0.0000,0.0000|4096,2048|JT|-|Intel Neural 7B optimized for conversational AI|Y
huggingface/neural-chat-7b|neural-chat-7b|Intel: Neural Chat 7B|C|0.0000,0.0000|8192,2048|JT|-|Lightweight Intel chat model optimized for edge inference|Y
huggingface/nous-hermes-3-70b|nous-hermes-3-70b|Nous: Hermes 3 70B|C|0.0000,0.0000|4096,2048|JT|-|Nous Research 70B instruct model with strong reasoning capabilities|Y
huggingface/openchat-3.5-0106|openchat-3.5|openchat|OpenChat: 3.5|C|0.0000,0.0000|8192,2048|JT|-|Community-driven chat model optimized for instruction following|Y
huggingface/openhermes-2.5-mistral-7b|openhermes-2.5|OpenHermes: Mistral 7B|C|0.0000,0.0000|8192,2048|JT|-|OpenHermes variant of Mistral optimized for multi-turn conversations|Y
huggingface/opt-1.3b|opt-1.3b|opt-1b|Meta: OPT 1.3B|C|0.0000,0.0000|2048,512|J|-|Meta OPT 1.3B autoregressive model|Y
huggingface/opt-125m|opt-125m|opt-small|Meta: OPT 125M|C|0.0000,0.0000|2048,512|J|-|Meta OPT smallest variant|Y
huggingface/opt-2.7b|opt-2.7b|opt-2.7b|Meta: OPT 2.7B|C|0.0000,0.0000|2048,1024|J|-|Meta OPT 2.7B for various tasks|Y
huggingface/opt-350m|opt-350m|opt-350|Meta: OPT 350M|C|0.0000,0.0000|2048,512|J|-|Meta OPT 350M model|Y
huggingface/opt-6.7b|opt-6.7b|opt-6.7b|Meta: OPT 6.7B|C|0.0000,0.0000|2048,1024|JT|-|Meta OPT 6.7B with improved performance|Y
huggingface/orca-2-13b|orca-2-13b|Microsoft: Orca 2 13B|C|0.0000,0.0000|4096,2048|JT|-|Microsoft Orca 2 for complex reasoning and instruction following|Y
huggingface/orca-2-7b|orca-2-7b|orca-mini|Microsoft: Orca 2 7B|C|0.0000,0.0000|4096,2048|JT|-|Compact Orca 2 variant for efficient instruction understanding|Y
huggingface/peft-adapter-mistral-7b|peft-mistral|peft-adapter|HuggingFace: PEFT Mistral 7B|C|0.0000,0.0000|4096,2048|JT|-|Mistral 7B with PEFT adapter architecture|Y
huggingface/phi-3-vision-128k|phi-3-vision-128k|Microsoft: Phi 3 Vision|C|0.0000,0.0000|128000,4096|VST|-|Compact vision model from Microsoft with 128K context|Y
huggingface/pythia-12b-deduped|pythia-12b|pythia-12b-dedup|EleutherAI: Pythia 12B|C|0.0000,0.0000|4096,2048|JT|-|EleutherAI 12B model for advanced tasks|Y
huggingface/pythia-1b-deduped|pythia-1b|pythia-1b-dedup|EleutherAI: Pythia 1B|C|0.0000,0.0000|2048,512|J|-|EleutherAI small model for research|Y
huggingface/pythia-6.9b-deduped|pythia-7b|pythia-7b-dedup|EleutherAI: Pythia 7B|C|0.0000,0.0000|4096,2048|JT|-|EleutherAI 7B model with deduplication|Y
huggingface/roberta-base|roberta|roberta-base|Facebook: RoBERTa Base|C|0.0000,0.0000|512,256|S|-|RoBERTa robust BERT pretraining|Y
huggingface/sciBERT-base|scibert|scibert-base|AllenAI: SciBERT Base|C|0.0000,0.0000|512,256|S|-|BERT model for scientific text understanding|Y
huggingface/solar-10.7b|solar-10.7b|solar|Upstage: Solar 10.7B|C|0.0000,0.0000|4096,2048|JT|-|Upstage Solar base model with efficient inference|Y
huggingface/stablelm-3b|stablelm-3b|stable-3b|Stability: StableLM 3B|C|0.0000,0.0000|4096,1024|J|-|Lightweight StableLM model for mobile and edge devices|Y
huggingface/stablelm-base-alpha-7b|stablelm-7b|stablelm-base|Stability: StableLM 7B Alpha|C|0.0000,0.0000|4096,2048|JT|-|Base StableLM 7B for instruct-tuning|Y
huggingface/starling-7b|starling-7b|starling-lm|Starling: 7B|C|0.0000,0.0000|4096,2048|JT|-|Open LM foundation model with strong chat capabilities|Y
huggingface/t5-small-finetuned-question-generation|t5-qa-gen-ft|t5-qg-ft|Hugging Face: T5 Small QA Generation FT|C|0.0000,0.0000|512,256|T|-|T5 Small fine-tuned for question generation|Y
huggingface/tinyllama-1.1b|tinyllama-1.1b|tinyllama|TinyLlama: 1.1B|C|0.0000,0.0000|2048,512|J|-|Ultra-lightweight 1.1B model for resource-constrained devices|Y
huggingface/umtf-llama2-7b-medical|umtf-medical|medical-llama2|UMTF: Medical Llama2 7B|C|0.0000,0.0000|4096,2048|J|-|Llama 2 7B fine-tuned for medical domain|Y
huggingface/wizard-lm-1.3b|wizard-lm-1.3b|wizard-mini|WizardLM: 1.3B|C|0.0000,0.0000|2048,512|J|-|Lightweight WizardLM variant for instruction following|Y
huggingface/wizard-lm-13b|wizard-lm-13b|wizard|WizardLM: 13B|C|0.0000,0.0000|4096,2048|JT|-|WizardLM 13B optimized for instruction following|Y
huggingface/xlm-roberta-base-finetuned-cross-lingual-sentiment|xlm-roberta-sent-ft|xlm-sent-ft|Hugging Face: XLM-RoBERTa Sentiment FT|C|0.0000,0.0000|512,256|S|-|XLM-RoBERTa fine-tuned for cross-lingual sentiment|Y
huggingface/xlnet-base-cased|xlnet-base|xlnet|Google: XLNet Base|C|0.0000,0.0000|512,256|S|-|XLNet base bidirectional transformer|Y
huggingface/zephyr-7b-beta|zephyr-7b|zephyr|HuggingFace: Zephyr 7B|C|0.0000,0.0000|4096,2048|JT|-|Community chat model from HuggingFace with strong performance|Y

# =============================================================================
# QWEN/DASHSCOPE - Alibaba (59 models)
# =============================================================================
qwen/qwen-2.5-72b-instruct|qwen-2.5-72b-instruc|Qwen2.5 72B Instruct|C|0.0000,0.0000|32768,16384|JST|-|Qwen2.5 72B is the latest series of Qwen large language models. Qwen2.5 brings t|Y
qwen/qwen-2.5-7b-instruct|qwen-2.5-7b-instruct|Qwen: Qwen2.5 7B Instruct|C|0.0000,0.0000|32768,8192|-|-|Qwen2.5 7B is the latest series of Qwen large language models. Qwen2.5 brings th|Y
qwen/qwen-2.5-coder-32b-instruct|qwen-2.5-coder-32b-i|Qwen2.5 Coder 32B Instruct|C|0.0000,0.0000|32768,32768|JS|-|Qwen2.5-Coder is the latest series of Code-Specific Qwen large language models (|Y
qwen/qwen-2.5-vl-7b-instruct|qwen-2.5-vl-7b-instr|Qwen: Qwen2.5-VL 7B Instruct|C|0.0000,0.0000|32768,8192|V|-|Qwen2.5 VL 7B is a multimodal LLM from the Qwen Team with the following key enha|Y
qwen/qwen-2.5-vl-7b-instruct:free|qwen-2.5-vl-7b-instr|Qwen: Qwen2.5-VL 7B Instruct (free)|C|-|32768,8192|V|-|Qwen2.5 VL 7B is a multimodal LLM from the Qwen Team with the following key enha|Y
qwen/qwen-long|qwen-long|Qwen Long|C|0.0007,0.0028|10000000,8192|TJS|-|10M context for ultra-long documents|Y
qwen/qwen-max|qwen-max|Qwen: Qwen-Max|C|0.0000,0.0000|32768,8192|JT|-|Qwen-Max, based on Qwen2.5, provides the best inference performance among [Qwen|Y
qwen/qwen-plus|qwen-plus|Qwen: Qwen-Plus|C|0.0000,0.0000|131072,8192|JT|-|Qwen-Plus, based on the Qwen2.5 foundation model, is a 131K context model with a|Y
qwen/qwen-plus-2025-07-28|qwen-plus-2025-07-28|Qwen: Qwen Plus 0728|C|0.0000,0.0000|1000000,32768|JST|-|Qwen Plus 0728, based on the Qwen3 foundation model, is a 1 million context hybr|Y
qwen/qwen-plus-2025-07-28:thinking|qwen-plus-2025-07-28|Qwen: Qwen Plus 0728 (thinking)|C|0.0000,0.0000|1000000,32768|JKST|-|Qwen Plus 0728, based on the Qwen3 foundation model, is a 1 million context hybr|Y
qwen/qwen-turbo|qwen-turbo|Qwen: Qwen-Turbo|C|0.0000,0.0000|1000000,8192|JT|-|Qwen-Turbo, based on Qwen2.5, is a 1M context model that provides fast speed and|Y
qwen/qwen-vl-max|qwen-vl-max|Qwen: Qwen VL Max|C|0.0000,0.0000|131072,8192|JTV|-|Qwen VL Max is a visual understanding model with 7500 tokens context length. It|Y
qwen/qwen-vl-max-0809|qwen-vl-max|Qwen VL Max|C|0.28,0.82|32768,8192|VTJS|-|Best vision-language model|Y
qwen/qwen-vl-plus|qwen-vl-plus|Qwen: Qwen VL Plus|C|0.0000,0.0000|7500,1500|JV|-|Qwen's Enhanced Large Visual Language Model. Significantly upgraded for detailed|Y
qwen/qwen-vl-plus-0809|qwen-vl-plus|Qwen VL Plus|C|0.11,0.33|32768,8192|VTJS|-|Enhanced vision-language model|Y
qwen/qwen2-audio-instruct|qwen2-audio|Qwen2 Audio|C|0.03,0.08|32768,8192|ATJ|-|Audio understanding model|Y
qwen/qwen2-vl-72b-instruct|qwen2-vl-72b|Qwen2 VL 72B|C|0.55,1.6|131072,8192|VTJS|-|Large vision-language model|Y
qwen/qwen2-vl-7b-instruct|qwen2-vl-7b|Qwen2 VL 7B|C|0.01,0.04|32768,8192|VTJS|-|Compact vision-language model|Y
qwen/qwen2.5-14b-instruct|qwen2.5-14b|Qwen 2.5 14B|C|0.06,0.11|131072,8192|TJS|-|Efficient 14B model|Y
qwen/qwen2.5-32b-instruct|qwen2.5-32b|Qwen 2.5 32B|C|0.28,0.55|131072,8192|VTJS|-|Balanced 32B model|Y
qwen/qwen2.5-3b-instruct|qwen2.5-3b|Qwen 2.5 3B|C|0.0042,0.0083|32768,8192|TJ|-|Tiny 3B model|Y
qwen/qwen2.5-72b-instruct|qwen2.5-72b|Qwen 2.5 72B|C|0.55,1.6|131072,8192|VTJS|-|Flagship 72B instruct model|Y
qwen/qwen2.5-7b-instruct|qwen2.5-7b|Qwen 2.5 7B|C|0.01,0.04|131072,8192|TJS|-|Compact 7B model|Y
qwen/qwen2.5-coder-14b-instruct|qwen-coder-14b|Qwen 2.5 Coder 14B|C|0.06,0.11|131072,8192|TJS|-|Code-specialized 14B model|Y
qwen/qwen2.5-coder-32b-instruct|qwen-coder-32b|Qwen 2.5 Coder 32B|C|0.28,0.55|131072,8192|TJS|-|Code-specialized 32B model|Y
qwen/qwen2.5-coder-7b-instruct|qwen2.5-coder-7b-ins|Qwen: Qwen2.5 Coder 7B Instruct|C|0.0000,0.0000|32768,8192|JS|-|Qwen2.5-Coder-7B-Instruct is a 7B parameter instruction-tuned language model opt|Y
qwen/qwen2.5-math-72b-instruct|qwen-math-72b|Qwen 2.5 Math 72B|C|0.55,1.6|4096,4096|TJS|-|Math-specialized 72B model|Y
qwen/qwen2.5-vl-32b-instruct|qwen2.5-vl-32b-instr|Qwen: Qwen2.5 VL 32B Instruct|C|0.0000,0.0000|16384,16384|JSV|-|Qwen2.5-VL-32B is a multimodal vision-language model fine-tuned through reinforc|Y
qwen/qwen2.5-vl-72b-instruct|qwen2.5-vl-72b-instr|Qwen: Qwen2.5 VL 72B Instruct|C|0.0000,0.0000|32768,32768|JSV|-|Qwen2.5-VL is proficient in recognizing common objects such as flowers, birds, f|Y
qwen/qwen2.5-vl-7b-instruct|qwen2.5-vl-7b|Alibaba: Qwen2.5-VL 7B|C|0.0000,0.0000|8192,2048|VST|-|Qwen 2.5 Vision Language 7B model with OCR capabilities|Y
qwen/qwen3-14b|qwen3-14b|Qwen: Qwen3 14B|C|0.0000,0.0000|40960,40960|JKST|-|Qwen3-14B is a dense 14.8B parameter causal language model from the Qwen3 series|Y
qwen/qwen3-235b-a22b|qwen3-235b-a22b|Qwen: Qwen3 235B A22B|C|0.0000,0.0000|40960,40960|JKST|-|Qwen3-235B-A22B is a 235B parameter mixture-of-experts (MoE) model developed by|Y
qwen/qwen3-235b-a22b-2507|qwen3-235b-a22b-2507|Qwen: Qwen3 235B A22B Instruct 2507|C|0.0000,0.0000|262144,65536|JKST|-|Qwen3-235B-A22B-Instruct-2507 is a multilingual, instruction-tuned mixture-of-ex|Y
qwen/qwen3-235b-a22b-thinking-2507|qwen3-235b-a22b-thin|Qwen: Qwen3 235B A22B Thinking 2507|C|0.0000,0.0000|262144,262144|JKST|-|Qwen3-235B-A22B-Thinking-2507 is a high-performance, open-weight Mixture-of-Expe|Y
qwen/qwen3-30b-a3b|qwen3-30b-a3b|Qwen: Qwen3 30B A3B|C|0.0000,0.0000|40960,40960|JKST|-|Qwen3, the latest generation in the Qwen large language model series, features b|Y
qwen/qwen3-30b-a3b-instruct-2507|qwen3-30b-a3b-instru|Qwen: Qwen3 30B A3B Instruct 2507|C|0.0000,0.0000|262144,262144|JST|-|Qwen3-30B-A3B-Instruct-2507 is a 30.5B-parameter mixture-of-experts language mod|Y
qwen/qwen3-30b-a3b-thinking-2507|qwen3-30b-a3b-thinki|Qwen: Qwen3 30B A3B Thinking 2507|C|0.0000,0.0000|32768,8192|JKST|-|Qwen3-30B-A3B-Thinking-2507 is a 30B parameter Mixture-of-Experts reasoning mode|Y
qwen/qwen3-32b|qwen3-32b|Qwen: Qwen3 32B|C|0.0000,0.0000|40960,40960|JKST|-|Qwen3-32B is a dense 32.8B parameter causal language model from the Qwen3 series|Y
qwen/qwen3-4b:free|qwen3-4b:free|Qwen: Qwen3 4B (free)|C|-|40960,10240|JKST|-|Qwen3-4B is a 4 billion parameter dense language model from the Qwen3 series, de|Y
qwen/qwen3-8b|qwen3-8b|Qwen: Qwen3 8B|C|0.0000,0.0000|128000,20000|JKST|-|Qwen3-8B is a dense 8.2B parameter causal language model from the Qwen3 series,|Y
qwen/qwen3-coder|qwen3-coder|Qwen: Qwen3 Coder 480B A35B|C|0.0000,0.0000|262144,262144|JKST|-|Qwen3-Coder-480B-A35B-Instruct is a Mixture-of-Experts (MoE) code generation mod|Y
qwen/qwen3-coder-30b-a3b-instruct|qwen3-coder-30b-a3b-|Qwen: Qwen3 Coder 30B A3B Instruct|C|0.0000,0.0000|160000,32768|JST|-|Qwen3-Coder-30B-A3B-Instruct is a 30.5B parameter Mixture-of-Experts (MoE) model|Y
qwen/qwen3-coder-flash|qwen3-coder-flash|Qwen: Qwen3 Coder Flash|C|0.0000,0.0000|128000,65536|JT|-|Qwen3 Coder Flash is Alibaba's fast and cost efficient version of their propriet|Y
qwen/qwen3-coder-plus|qwen3-coder-plus|Qwen: Qwen3 Coder Plus|C|0.0000,0.0000|128000,65536|JST|-|Qwen3 Coder Plus is Alibaba's proprietary version of the Open Source Qwen3 Coder|Y
qwen/qwen3-coder:exacto|qwen3-coder:exacto|Qwen: Qwen3 Coder 480B A35B (exacto)|C|0.0000,0.0000|262144,65536|JKST|-|Qwen3-Coder-480B-A35B-Instruct is a Mixture-of-Experts (MoE) code generation mod|Y
qwen/qwen3-coder:free|qwen3-coder:free|Qwen: Qwen3 Coder 480B A35B (free)|C|-|262000,262000|T|-|Qwen3-Coder-480B-A35B-Instruct is a Mixture-of-Experts (MoE) code generation mod|Y
qwen/qwen3-max|qwen3-max|Qwen: Qwen3 Max|C|0.0000,0.0000|256000,32768|JT|-|Qwen3-Max is an updated release built on the Qwen3 series, offering major improv|Y
qwen/qwen3-next-80b-a3b-instruct|qwen3-next-80b-a3b-i|Qwen: Qwen3 Next 80B A3B Instruct|C|0.0000,0.0000|262144,65536|JST|-|Qwen3-Next-80B-A3B-Instruct is an instruction-tuned chat model in the Qwen3-Next|Y
qwen/qwen3-next-80b-a3b-thinking|qwen3-next-80b-a3b-t|Qwen: Qwen3 Next 80B A3B Thinking|C|0.0000,0.0000|262144,262144|JKST|-|Qwen3-Next-80B-A3B-Thinking is a reasoning-first chat model in the Qwen3-Next li|Y
qwen/qwen3-vl-235b-a22b-instruct|qwen3-vl-235b-a22b-i|Qwen: Qwen3 VL 235B A22B Instruct|C|0.0000,0.0000|262144,65536|JSTV|-|Qwen3-VL-235B-A22B Instruct is an open-weight multimodal model that unifies stro|Y
qwen/qwen3-vl-235b-a22b-thinking|qwen3-vl-235b-a22b-t|Qwen: Qwen3 VL 235B A22B Thinking|C|0.0000,0.0000|262144,262144|JKSTV|-|Qwen3-VL-235B-A22B Thinking is a multimodal model that unifies strong text gener|Y
qwen/qwen3-vl-30b-a3b-instruct|qwen3-vl-30b-a3b-ins|Qwen: Qwen3 VL 30B A3B Instruct|C|0.0000,0.0000|262144,65536|JSTV|-|Qwen3-VL-30B-A3B-Instruct is a multimodal model that unifies strong text generat|Y
qwen/qwen3-vl-30b-a3b-thinking|qwen3-vl-30b-a3b-thi|Qwen: Qwen3 VL 30B A3B Thinking|C|0.0000,0.0000|131072,32768|JKSTV|-|Qwen3-VL-30B-A3B-Thinking is a multimodal model that unifies strong text generat|Y
qwen/qwen3-vl-32b-instruct|qwen3-vl-32b-instruc|Qwen: Qwen3 VL 32B Instruct|C|0.0000,0.0000|262144,65536|JSV|-|Qwen3-VL-32B-Instruct is a large-scale multimodal vision-language model designed|Y
qwen/qwen3-vl-8b-instruct|qwen3-vl-8b-instruct|Qwen: Qwen3 VL 8B Instruct|C|0.0000,0.0000|131072,32768|JSTV|-|Qwen3-VL-8B-Instruct is a multimodal vision-language model from the Qwen3-VL ser|Y
qwen/qwen3-vl-8b-thinking|qwen3-vl-8b-thinking|Qwen: Qwen3 VL 8B Thinking|C|0.0000,0.0000|256000,32768|JKSTV|-|Qwen3-VL-8B-Thinking is the reasoning-optimized variant of the Qwen3-VL-8B multi|Y
qwen/qwq-32b|qwq-32b|Qwen: QwQ 32B|C|0.0000,0.0000|32768,8192|JKST|-|QwQ is the reasoning model of the Qwen series. Compared with conventional instru|Y
qwen/qwq-32b-preview|qwq-32b|QwQ 32B|C|0.28,0.82|32768,32768|TJK|-|Chain-of-thought reasoning model|N
qwen/text-embedding-v3|qwen-embed-v3|Text Embedding V3|C|0.0001,|8192,1024|E|-|Text embeddings|N

# =============================================================================
# ZHIPU/GLM - Chinese AI (17 models)
# =============================================================================
zhipu/codegeex-4|codegeex-4|CodeGeeX 4|C|0.0014,0.0014|131072,8192|TJS|-|Specialized code generation|Y
zhipu/embedding-3|embedding-3|Embedding 3|C|0.0007,|8192,2048|E|-|Text embeddings|N
zhipu/glm-3.5-turbo|glm-3.5|glm-turbo|Zhipu: GLM-3.5 Turbo|C|0.0000,0.0000|8192,2048|VT|-|Fast GLM-3.5 Turbo for real-time applications|Y
zhipu/glm-4|glm-4|glm-4-turbo|Zhipu: GLM-4|C|0.0000,0.0000|8192,2048|VSTJ|-|Zhipu GLM-4 with multimodal capabilities|Y
zhipu/glm-4-0520|glm-4|GLM-4|C|0.14,0.14|131072,8192|VTJS|-|Standard GLM-4 model|Y
zhipu/glm-4-air|glm-4-air|GLM-4 Air|C|0.0014,0.0014|131072,8192|TJS|-|Lightweight GLM-4 variant|Y
zhipu/glm-4-airx|glm-4-airx|GLM-4 AirX|C|0.01,0.01|8192,8192|TJS|-|Fast inference GLM-4|Y
zhipu/glm-4-flash|glm-4-flash|GLM-4 Flash|C|0,0|131072,8192|TJS|-|Free tier GLM-4|Y
zhipu/glm-4-flashx|glm-4-flashx|GLM-4 FlashX|C|0,0|131072,8192|TJS|-|Free tier fast GLM-4|Y
zhipu/glm-4-long|glm-4-long|GLM-4 Long|C|0.0014,0.0014|1000000,8192|TJS|-|1M context for ultra-long documents|Y
zhipu/glm-4-plus|glm-4-plus|GLM-4 Plus|C|0.69,0.69|131072,8192|VTJS|-|Enhanced GLM-4 for complex tasks|Y
zhipu/glm-4v|glm-4v|GLM-4V|C|0.07,0.07|8192,4096|VTJ|-|Standard vision model|Y
zhipu/glm-4v-flash|glm-4v-flash|GLM-4V Flash|C|0,0|8192,4096|VTJ|-|Free tier vision model|Y
zhipu/glm-4v-plus|glm-4v-plus|GLM-4V Plus|C|0.14,0.14|8192,4096|VTJS|-|Enhanced vision model|Y
zhipu/glm-z1-air-preview|glm-z1-air|GLM-Z1 Air Preview|C|0.07,0.07|16384,16384|TJK|-|Lightweight reasoning model|N
zhipu/glm-z1-flash-preview|glm-z1-flash|GLM-Z1 Flash Preview|C|0,0|16384,16384|TJK|-|Free tier reasoning model|N
zhipu/glm-z1-preview|glm-z1|GLM-Z1 Preview|C|0.69,0.69|16384,16384|TJK|-|Reasoning model with thinking|N

# =============================================================================
# MINIMAX - Chinese AI (17 models)
# =============================================================================
minimax/abab5.5-chat|abab5.5|ABAB 5.5|C|0.21,0.21|16384,8192|VTJ|-|Legacy vision model|Y
minimax/abab5.5s-chat|abab5.5s|ABAB 5.5s|C|0.07,0.07|16384,8192|TJ|-|Legacy efficient model|Y
minimax/abab6-chat|abab6|ABAB 6|C|0.21,0.21|32768,8192|VTJ|-|Previous generation ABAB|Y
minimax/abab6.5g-chat|abab6.5g|ABAB 6.5g|C|0.14,0.14|8192,8192|TJS|-|General purpose model|Y
minimax/abab6.5s-chat|abab6.5s|ABAB 6.5s|C|0.14,0.14|245760,16384|VTJS|-|Fast 245K context model|Y
minimax/abab6.5t-chat|abab6.5t|ABAB 6.5t|C|0.07,0.07|8192,8192|TJS|-|Efficient text model|Y
minimax/abab7-chat-preview|abab7|ABAB 7|C|0.70,0.70|4000000,16384|VTJS|-|Flagship 4M context model for complex tasks|Y
minimax/embo-01|embo-01|Embo 01|C|0.0001,|4096,1536|E|-|Text embeddings|N
minimax/minimax-01|minimax-01|MiniMax: MiniMax-01|C|0.0000,0.0000|1000192,1000192|V|-|MiniMax-01 is a combines MiniMax-Text-01 for text generation and MiniMax-VL-01 f|Y
minimax/minimax-m1|minimax-m1|MiniMax: MiniMax M1|C|0.0000,0.0000|1000000,40000|KT|-|MiniMax-M1 is a large-scale, open-weight reasoning model designed for extended c|N
minimax/minimax-m2|minimax-m2|MiniMax: MiniMax M2|C|0.0000,0.0000|196608,65536|JKST|-|MiniMax-M2 is a compact, high-efficiency large language model optimized for end-|Y
minimax/minimax-m2.1|minimax-m2.1|MiniMax: MiniMax M2.1|C|0.0000,0.0000|196608,49152|JKST|-|MiniMax-M2.1 is a lightweight, state-of-the-art large language model optimized f|Y
minimax/music-01|music-01|Music 01|C|0.01,|0,0|A|-|Music generation|N
minimax/speech-01-hd|speech-hd|Speech 01 HD|C|0.10,|0,0|A|-|High-quality TTS|N
minimax/speech-01-turbo|speech-turbo|Speech 01 Turbo|C|0.0070,|0,0|A|-|Fast TTS model|N
minimax/speech-02-turbo|speech-02|Speech 02 Turbo|C|0.0070,|0,0|A|-|Next-gen TTS|N
minimax/video-01|video-01|Video 01|C|0.30,|0,0|D|-|Video generation|N

# =============================================================================
# MOONSHOT/KIMI - Chinese AI (6 models)
# =============================================================================
moonshot/kimi-k2-0711-preview|kimi-k2|Kimi K2|C|0.55,2.0|131072,8192|VTJSK|-|1T MoE model with thinking capabilities|Y
moonshot/moonshot-v1-128k|moonshot-128k|moonshot-long|Moonshot: v1 128K|C|0.0000,0.0000|128000,8192|VSTJ|-|Ultra-long context Moonshot v1 128K|Y
moonshot/moonshot-v1-256k|moonshot-256k|Moonshot V1 256K|C|0.84,0.84|262144,8192|TJS|-|256K context for long document processing|Y
moonshot/moonshot-v1-32k|moonshot-32k|moonshot-extended|Moonshot: v1 32K|C|0.0000,0.0000|32000,8192|VSTJ|-|Extended context Moonshot v1 32K model|Y
moonshot/moonshot-v1-8k|moonshot-8k|moonshot-standard|Moonshot: v1 8K|C|0.0000,0.0000|8192,2048|VSTJ|-|Moonshot v1 8K context model|Y
moonshot/moonshot-v1-8k-web|moonshot-web|Moonshot V1 Web|C|0.08,0.08|8192,8192|TJS|-|8K with web search capability|Y

# =============================================================================
# BAICHUAN - Chinese AI (7 models)
# =============================================================================
baichuan/Baichuan-Text-Embedding|baichuan-embed|Baichuan Text Embedding|C|0.0007,|512,1024|E|-|Text embeddings|N
baichuan/Baichuan2-Turbo|baichuan2-turbo|Baichuan2 Turbo|C|0.01,0.01|32768,4096|TJ|-|Legacy turbo model|Y
baichuan/Baichuan2-Turbo-192k|baichuan2-192k|Baichuan2 Turbo 192K|C|0.02,0.02|196608,4096|TJ|-|192K context legacy model|Y
baichuan/Baichuan3-Turbo|baichuan3-turbo|Baichuan3 Turbo|C|0.02,0.02|131072,8192|TJS|-|Previous gen turbo model|Y
baichuan/Baichuan3-Turbo-128k|baichuan3-128k|Baichuan3 Turbo 128K|C|0.03,0.03|131072,8192|TJS|-|128K context model|Y
baichuan/Baichuan4-Air|baichuan4-air|Baichuan4 Air|C|0.01,0.01|131072,8192|TJS|-|Lightweight efficient model|Y
baichuan/Baichuan4-Turbo|baichuan4-turbo|Baichuan4 Turbo|C|0.14,0.14|131072,8192|VTJS|-|Flagship turbo model|Y

# =============================================================================
# YI/01.AI - Chinese AI (10 models)
# =============================================================================
yi/yi-large|yi-large|Yi Large|C|0.42,0.42|32768,8192|TJS|-|Flagship large model|Y
yi/yi-large-fc|yi-large-fc|Yi Large FC|C|0.42,0.42|32768,8192|TJS|-|Function calling model|Y
yi/yi-large-rag|yi-large-rag|Yi Large RAG|C|0.35,0.35|16384,8192|TJS|-|RAG-optimized model|Y
yi/yi-large-turbo|yi-large-turbo|Yi Large Turbo|C|0.17,0.17|16384,8192|TJS|-|Fast large model|Y
yi/yi-lightning|yi-lightning|Yi Lightning|C|0.14,0.14|16384,16384|TJS|-|Fast and affordable model|Y
yi/yi-medium|yi-medium|Yi Medium|C|0.04,0.04|16384,8192|TJ|-|Balanced medium model|Y
yi/yi-medium-200k|yi-medium-200k|Yi Medium 200K|C|0.17,0.17|204800,8192|TJ|-|200K context model|Y
yi/yi-spark|yi-spark|Yi Spark|C|0.0014,0.0014|16384,8192|TJ|-|Lightweight affordable model|Y
yi/yi-vision|yi-vision|Yi Vision|C|0.08,0.08|16384,8192|VTJ|-|Vision-language model|Y
yi/yi-vl-34b|yi-vl-34b|01.AI: Yi Vision 34B|C|0.0000,0.0000|8192,2048|VST|-|Yi Vision Language model with strong OCR capabilities|Y

# =============================================================================
# VOLCENGINE/DOUBAO - ByteDance (13 models)
# =============================================================================
volcengine/doubao-1.5-thinking-pro|doubao-thinking-pro|Doubao 1.5 Thinking Pro|C|0.55,2.2|131072,16384|VTJSK|-|Chain-of-thought reasoning model|Y
volcengine/doubao-1.5-thinking-pro-m|doubao-thinking-m|Doubao 1.5 Thinking Pro M|C|0.28,0.82|131072,16384|VTJSK|-|Medium thinking model|Y
volcengine/doubao-embedding|doubao-embed|Doubao Embedding|C|0.0007,|4096,2048|E|-|Text embeddings|N
volcengine/doubao-embedding-large|doubao-embed-large|Doubao Embedding Large|C|0.0007,|4096,2048|E|-|Large text embeddings|N
volcengine/doubao-lite-128k|doubao-lite-128k|Doubao Lite 128K|C|0.04,0.12|131072,8192|TJS|-|Lightweight 128K model|Y
volcengine/doubao-lite-32k|doubao-lite-32k|Doubao Lite 32K|C|0.04,0.12|32768,8192|TJS|-|Lightweight 32K model|Y
volcengine/doubao-lite-4k|doubao-lite-4k|Doubao Lite 4K|C|0.04,0.12|4096,4096|TJS|-|Lightweight 4K model|Y
volcengine/doubao-pro-128k|doubao-pro-128k|Doubao Pro 128K|C|0.69,1.2|131072,8192|VTJS|-|High-performance 128K model|Y
volcengine/doubao-pro-256k|doubao-pro-256k|Doubao Pro 256K|C|0.69,1.2|262144,8192|VTJS|-|Flagship 256K context model|Y
volcengine/doubao-pro-32k|doubao-pro-32k|Doubao Pro 32K|C|0.11,0.24|32768,8192|VTJS|-|Balanced 32K model|Y
volcengine/doubao-pro-4k|doubao-pro-4k|Doubao Pro 4K|C|0.11,0.24|4096,4096|VTJS|-|Fast 4K model|Y
volcengine/doubao-vision-lite-32k|doubao-vision-lite|Doubao Vision Lite 32K|C|0.11,0.11|32768,8192|VTJ|-|Lightweight vision model|Y
volcengine/doubao-vision-pro-32k|doubao-vision-pro|Doubao Vision Pro 32K|C|0.27,0.27|32768,8192|VTJS|-|Vision-language model|Y

# =============================================================================
# SPARK/IFLYTEK - Chinese AI (10 models)
# =============================================================================
spark/spark-3.5-max|spark-3.5-max|Spark 3.5 Max|C|0.42,0.42|131072,8192|TJS|-|Previous gen 128K model|Y
spark/spark-3.5-pro|spark-3.5-pro|Spark 3.5 Pro|C|0.14,0.14|32768,8192|TJS|-|Balanced 32K model|Y
spark/spark-4.0-max-32k|spark-4-max|Spark 4.0 Max 32K|C|0.42,0.42|32768,8192|VTJS|-|32K context model|Y
spark/spark-4.0-ultra|spark-4-ultra|Spark 4.0 Ultra|C|0.69,0.69|131072,8192|VTJS|-|Flagship 128K context model|Y
spark/spark-asr-v2|spark-asr|Spark ASR v2|C|0.03,|0,0|A|-|Speech recognition|N
spark/spark-code|spark-code|Spark Code|C|0.21,0.21|32768,8192|TJS|-|Code generation model|Y
spark/spark-embedding-v1|spark-embed|Spark Embedding v1|C|0.0003,|512,768|E|-|Text embeddings|N
spark/spark-lite|spark-lite|Spark Lite|C|0,0|8192,4096|TJ|-|Free lightweight model|Y
spark/spark-tts-v2|spark-tts|Spark TTS v2|C|0.01,|0,0|A|-|Text-to-speech|N
spark/spark-vision-4.0|spark-vision-4|Spark Vision 4.0|C|0.42,0.42|32768,8192|VTJ|-|Vision-language model|Y

# =============================================================================
# STABILITY AI - Image Generation (16 models)
# =============================================================================
stability/sd3-large|sd3-large|SD 3 Large|C|0.07,|10000,0|I|-|SD 3 Large, $0.065/image|N
stability/sd3-large-turbo|sd3-turbo|SD 3 Large Turbo|C|0.04,|10000,0|I|-|Fast SD 3, $0.04/image|N
stability/sd3-medium|sd3-medium|SD 3 Medium|C|0.04,|10000,0|I|-|SD 3 Medium, $0.035/image|N
stability/sd3.5-large|sd3.5-large|SD 3.5 Large|C|0.07,|10000,0|I|-|SD 3.5 Large, $0.065/image|N
stability/sd3.5-large-turbo|sd3.5-turbo|SD 3.5 Large Turbo|C|0.04,|10000,0|I|-|Fast SD 3.5, $0.04/image|N
stability/sd3.5-medium|sd3.5-medium|SD 3.5 Medium|C|0.04,|10000,0|I|-|SD 3.5 Medium, $0.035/image|N
stability/stable-diffusion-inpaint|sd-inpaint|SD Inpaint|C|0.0020,|77,0|I|-|Image inpainting|N
stability/stable-diffusion-v1-6|sd-1.6|SD 1.6|C|0.0020,|77,0|I|-|Legacy SD 1.6|N
stability/stable-diffusion-xl-1024-v0-9|sdxl-0.9|SDXL 0.9|C|0.0020,|77,0|I|-|SDXL 0.9 beta|N
stability/stable-diffusion-xl-1024-v1-0|sdxl-1.0|SDXL 1.0|C|0.0020,|77,0|I|-|SDXL, $0.002-0.006 depending on steps|N
stability/stable-fast-upscale|fast-upscale|Fast Upscale|C|0.01,|0,0|I|-|Fast upscaling, $0.01/image|N
stability/stable-image-control|stable-control|Stable Image Control|C|0.04,|10000,0|I|-|Controlled image generation|N
stability/stable-image-core|stable-core|Stable Image Core|C|0.03,|10000,0|I|-|Efficient core model, $0.03/image|N
stability/stable-image-ultra|stable-ultra|Stable Image Ultra|C|0.08,|10000,0|I|-|Highest quality image generation, $0.08/image|N
stability/stable-image-upscale|stable-upscale|Stable Image Upscale|C|0.02,|0,0|I|-|Image upscaling, $0.02/image|N
stability/stable-video-diffusion|svd|Stable Video Diffusion|C|0.20,|0,0|D|-|Video generation|N

# =============================================================================
# ELEVENLABS - Voice AI (10 models)
# =============================================================================
elevenlabs/eleven_conversational_v1|conversational-v1|Conversational v1|C|0.18,|5000,0|A|-|Optimized for dialogue, $0.18/1K chars|N
elevenlabs/eleven_flash_v2|flash-v2|Flash v2|C|0.09,|5000,0|A|-|Fast English, $0.09/1K chars|N
elevenlabs/eleven_flash_v2_5|flash-v2.5|Flash v2.5|C|0.09,|5000,0|A|-|Ultra-low latency, $0.09/1K chars|N
elevenlabs/eleven_monolingual_v1|english-v1|English v1|C|0.18,|5000,0|A|-|English only, legacy model, $0.18/1K chars|N
elevenlabs/eleven_multilingual_v2|multilingual-v2|Multilingual v2|C|0.18,|5000,0|A|-|29 languages, most natural, $0.18/1K chars|N
elevenlabs/eleven_sound_effects|sound-effects|Sound Effects|C|0.18,|450,0|A|-|Sound effect generation, $0.18/1K chars|N
elevenlabs/eleven_speech_to_speech_v2|speech-to-speech|Speech to Speech v2|C|0.25,|0,0|A|-|Voice conversion, $0.25/1K chars|N
elevenlabs/eleven_turbo_v2|turbo-v2|Turbo v2|C|0.18,|5000,0|A|-|English only, fastest, $0.18/1K chars|N
elevenlabs/eleven_turbo_v2_5|turbo-v2.5|Turbo v2.5|C|0.18,|5000,0|A|-|32 languages, low latency, $0.18/1K chars|N
elevenlabs/eleven_voice_isolation|voice-isolation|Voice Isolation|C|0.50,|0,0|A|-|Isolate voices from audio, $0.50/min|N

# =============================================================================
# DEEPGRAM - Speech-to-Text (20 models)
# =============================================================================
deepgram/aura-asteria-en|aura-asteria|Aura Asteria|C|0.01,|0,0|A|-|TTS Asteria voice, $0.015/1K chars|N
deepgram/aura-luna-en|aura-luna|Aura Luna|C|0.01,|0,0|A|-|TTS Luna voice, $0.015/1K chars|N
deepgram/aura-orion-en|aura-orion|Aura Orion|C|0.01,|0,0|A|-|TTS Orion voice, $0.015/1K chars|N
deepgram/aura-stella-en|aura-stella|Aura Stella|C|0.01,|0,0|A|-|TTS Stella voice, $0.015/1K chars|N
deepgram/base|base|Base|C|0.01,|0,0|A|-|Base model, $0.0125/min|N
deepgram/enhanced|enhanced|Enhanced|C|0.01,|0,0|A|-|Enhanced accuracy, $0.0145/min|N
deepgram/nova-2|nova-2|Nova 2|C|0.0043,|0,0|A|-|Previous gen STT, $0.0043/min|N
deepgram/nova-2-atc|nova-2-atc|Nova 2 ATC|C|0.0043,|0,0|A|-|Air traffic control, $0.0043/min|N
deepgram/nova-2-conversational-ai|nova-2-conv|Nova 2 Conversational|C|0.0043,|0,0|A|-|Conversational AI, $0.0043/min|N
deepgram/nova-2-drivethru|nova-2-drive|Nova 2 Drive-thru|C|0.0043,|0,0|A|-|Drive-thru ordering, $0.0043/min|N
deepgram/nova-2-finance|nova-2-finance|Nova 2 Finance|C|0.0043,|0,0|A|-|Financial transcription, $0.0043/min|N
deepgram/nova-2-general|nova-2-general|Nova 2 General|C|0.0043,|0,0|A|-|General purpose, $0.0043/min|N
deepgram/nova-2-meeting|nova-2-meeting|Nova 2 Meeting|C|0.0043,|0,0|A|-|Meeting transcription, $0.0043/min|N
deepgram/nova-2-phonecall|nova-2-phone|Nova 2 Phone|C|0.0043,|0,0|A|-|Phone call transcription, $0.0043/min|N
deepgram/nova-2-voicemail|nova-2-voicemail|Nova 2 Voicemail|C|0.0043,|0,0|A|-|Voicemail transcription, $0.0043/min|N
deepgram/nova-3|nova-3|Nova 3|C|0.0043,|0,0|A|-|Most accurate STT, $0.0043/min|N
deepgram/nova-3-medical|nova-3-medical|Nova 3 Medical|C|0.0050,|0,0|A|-|Medical transcription, $0.005/min|N
deepgram/whisper-cloud|whisper-cloud|Whisper Cloud|C|0.0048,|0,0|A|-|OpenAI Whisper on Deepgram, $0.0048/min|N
deepgram/whisper-cloud-large|whisper-large|Whisper Large|C|0.0048,|0,0|A|-|Whisper Large, $0.0048/min|N
deepgram/whisper-cloud-medium|whisper-medium|Whisper Medium|C|0.0048,|0,0|A|-|Whisper Medium, $0.0048/min|N

# =============================================================================
# VOYAGE AI - Embeddings (12 models)
# =============================================================================
voyage/rerank-2|rerank-2|Rerank 2|C|0.05,|32000,0|R|-|Document reranking, $0.05/1M tokens|Y
voyage/rerank-2-lite|rerank-2-lite|Rerank 2 Lite|C|0.02,|32000,0|R|-|Lightweight reranking, $0.02/1M tokens|Y
voyage/voyage-02|voyage-2|Voyage 2|C|0.10,|16000,1024|E|-|Previous generation embeddings|N
voyage/voyage-3|voyage-3|Voyage 3|C|0.06,|32000,1024|E|-|General purpose embeddings|N
voyage/voyage-3-lite|voyage-3-lite|Voyage 3 Lite|C|0.02,|32000,512|E|-|Lightweight embeddings, 512 dimensions|N
voyage/voyage-3.5|voyage-3.5|Voyage 3.5|C|0.06,|32000,1024|E|-|Latest embedding model, 1024 dimensions|N
voyage/voyage-code-3|voyage-code-3|Voyage Code 3|C|0.18,|32000,1024|E|-|Code-optimized embeddings|N
voyage/voyage-finance-2|voyage-finance|Voyage Finance 2|C|0.12,|32000,1024|E|-|Finance-optimized embeddings|N
voyage/voyage-large-2|voyage-large-2|Voyage Large 2|C|0.12,|16000,1536|E|-|Large embeddings, 1536 dimensions|N
voyage/voyage-large-2-instruct|voyage-large-instruct|Voyage Large 2 Instruct|C|0.12,|16000,1024|E|-|Instruction-tuned embeddings|N
voyage/voyage-law-2|voyage-law|Voyage Law 2|C|0.12,|16000,1024|E|-|Legal document embeddings|N
voyage/voyage-multilingual-2|voyage-multilingual|Voyage Multilingual 2|C|0.12,|32000,1024|E|-|Multilingual embeddings|N

# =============================================================================
# JINA AI - Embeddings (16 models)
# =============================================================================
jina/jina-clip-v1|jina-clip-v1|Jina CLIP v1|C|0.02,|8192,768|VE|-|Image-text embeddings|N
jina/jina-clip-v2|jina-clip-v2|Jina CLIP v2|C|0.02,|8192,1024|VE|-|Multimodal image-text embeddings|N
jina/jina-colbert-v1-en|jina-colbert-v1|Jina ColBERT v1 EN|C|0.02,|8192,128|E|-|English ColBERT|N
jina/jina-colbert-v2|jina-colbert-v2|Jina ColBERT v2|C|0.02,|8192,128|E|-|Late interaction retrieval|N
jina/jina-embeddings-v2-base-code|jina-embed-v2-code|Jina Embeddings v2 Code|C|0.02,|8192,768|E|-|Code embeddings|N
jina/jina-embeddings-v2-base-de|jina-embed-v2-de|Jina Embeddings v2 Base DE|C|0.02,|8192,768|E|-|German embeddings|N
jina/jina-embeddings-v2-base-en|jina-embed-v2-en|Jina Embeddings v2 Base EN|C|0.02,|8192,768|E|-|English embeddings, 768 dimensions|N
jina/jina-embeddings-v2-base-es|jina-embed-v2-es|Jina Embeddings v2 Base ES|C|0.02,|8192,768|E|-|Spanish embeddings|N
jina/jina-embeddings-v2-base-zh|jina-embed-v2-zh|Jina Embeddings v2 Base ZH|C|0.02,|8192,768|E|-|Chinese embeddings|N
jina/jina-embeddings-v2-small-en|jina-embed-v2-small|Jina Embeddings v2 Small EN|C|0.01,|8192,512|E|-|Small English embeddings|N
jina/jina-embeddings-v3|jina-embed-v3|Jina Embeddings v3|C|0.02,|8192,1024|E|-|Latest multilingual embeddings|N
jina/jina-reader-v1|jina-reader|Jina Reader v1|C|0,|0,0|-|-|URL/PDF to markdown|Y
jina/jina-reranker-v1-base-en|jina-rerank-v1|Jina Reranker v1 EN|C|0.02,|8192,0|R|-|English reranking|Y
jina/jina-reranker-v1-tiny-en|jina-rerank-tiny|Jina Reranker v1 Tiny|C|0.0050,|8192,0|R|-|Tiny reranking model|Y
jina/jina-reranker-v1-turbo-en|jina-rerank-turbo|Jina Reranker v1 Turbo|C|0.01,|8192,0|R|-|Fast English reranking|Y
jina/jina-reranker-v2-base-multilingual|jina-rerank-v2|Jina Reranker v2 Multilingual|C|0.02,|8192,0|R|-|Multilingual reranking|Y

# =============================================================================
# NOVITA AI - Inference (19 models)
# =============================================================================
novita/all-MiniLM-L6-v2|minilm-l6|MiniLM L6 v2|C|0.01,|512,384|E|-|Lightweight embeddings|N
novita/bge-large-en-v1.5|bge-large|BGE Large EN|C|0.01,|512,1024|E|-|BGE embeddings on Novita|N
novita/codestral-2501|codestral|Codestral|C|0.10,0.30|262144,8192|TJS|-|Codestral on Novita|Y
novita/deepseek-r1|deepseek-r1|DeepSeek R1|C|0.29,1.2|65536,8192|TJK|-|DeepSeek R1 reasoning on Novita|N
novita/deepseek-r1-distill-llama-70b|deepseek-r1-70b|DeepSeek R1 Distill 70B|C|0.19,0.19|65536,8192|TJK|-|DeepSeek R1 distilled|N
novita/deepseek-v3|deepseek-v3|DeepSeek V3|C|0.29,0.87|65536,8192|TJS|-|DeepSeek V3 on Novita|Y
novita/flux-1-dev|flux-dev|FLUX.1 Dev|C|0.03,|77,0|I|-|FLUX Dev image generation|N
novita/flux-1-schnell|flux-schnell|FLUX.1 Schnell|C|0.0030,|77,0|I|-|Fast FLUX image generation|N
novita/llama-3.1-405b-instruct|llama-3.1-405b|Llama 3.1 405B|C|1.0,1.0|131072,8192|TJS|-|Llama 3.1 405B on Novita|Y
novita/llama-3.1-70b-instruct|llama-3.1-70b|Llama 3.1 70B|C|0.19,0.19|131072,8192|TJS|-|Llama 3.1 70B on Novita|Y
novita/llama-3.1-8b-instruct|llama-3.1-8b|Llama 3.1 8B|C|0.03,0.03|131072,8192|TJS|-|Llama 3.1 8B on Novita|Y
novita/llama-3.3-70b-instruct|llama-3.3-70b|Llama 3.3 70B|C|0.19,0.19|131072,8192|TJS|-|Llama 3.3 70B on Novita|Y
novita/mistral-large-2411|mistral-large|Mistral Large|C|0.80,2.4|131072,8192|TJS|-|Mistral Large on Novita|Y
novita/mixtral-8x22b-instruct|mixtral-8x22b|Mixtral 8x22B|C|0.29,0.29|65536,8192|TJS|-|Mixtral 8x22B on Novita|Y
novita/qwen-2.5-72b-instruct|qwen-2.5-72b|Qwen 2.5 72B|C|0.29,0.29|131072,8192|TJS|-|Qwen 2.5 72B on Novita|Y
novita/qwen-2.5-coder-32b-instruct|qwen-coder-32b|Qwen 2.5 Coder 32B|C|0.10,0.10|131072,8192|TJS|-|Qwen Coder on Novita|Y
novita/qwq-32b-preview|qwq-32b|QwQ 32B|C|0.10,0.10|32768,32768|TJK|-|QwQ reasoning on Novita|N
novita/sd3-medium|sd3-medium|SD3 Medium|C|0.03,|77,0|I|-|SD3 Medium on Novita|N
novita/sdxl|sdxl|SDXL|C|0.0020,|77,0|I|-|SDXL on Novita|N

# =============================================================================
# NEBIUS - Inference (15 models)
# =============================================================================
nebius/DeepSeek-R1|deepseek-r1|DeepSeek R1|C|0.35,2.1|131072,16384|TJK|-|DeepSeek R1 reasoning|N
nebius/DeepSeek-V3|deepseek-v3|DeepSeek V3|C|0.35,0.90|131072,16384|TJS|-|DeepSeek V3 on Nebius|Y
nebius/Llama-3.2-11B-Vision-Instruct|llama-3.2-11b-vision|Llama 3.2 11B Vision|C|0.06,0.06|131072,16384|VTJS|-|Compact vision model|Y
nebius/Llama-3.2-90B-Vision-Instruct|llama-3.2-90b-vision|Llama 3.2 90B Vision|C|0.35,0.40|131072,16384|VTJS|-|Vision model on Nebius|Y
nebius/Llama-3.3-70B-Instruct|llama-3.3-70b|Llama 3.3 70B|C|0.35,0.40|131072,16384|TJS|-|Llama 3.3 70B on Nebius|Y
nebius/Meta-Llama-3.1-405B-Instruct|llama-3.1-405b|Llama 3.1 405B|C|2.4,2.4|131072,16384|TJS|-|Largest Llama on Nebius|Y
nebius/Meta-Llama-3.1-70B-Instruct|llama-3.1-70b|Llama 3.1 70B|C|0.35,0.40|131072,16384|TJS|-|Llama 3.1 70B on Nebius|Y
nebius/Meta-Llama-3.1-8B-Instruct|llama-3.1-8b|Llama 3.1 8B|C|0.06,0.06|131072,16384|TJS|-|Llama 3.1 8B on Nebius|Y
nebius/Mistral-Large-Instruct-2411|mistral-large|Mistral Large|C|0.90,2.7|131072,16384|TJS|-|Mistral Large on Nebius|Y
nebius/Mistral-Nemo-Instruct-2407|mistral-nemo|Mistral Nemo|C|0.06,0.06|131072,16384|TJS|-|Mistral Nemo on Nebius|Y
nebius/Phi-3.5-MoE-instruct|phi-3.5-moe|Phi 3.5 MoE|C|0.12,0.12|131072,16384|TJS|-|Microsoft Phi 3.5 MoE|Y
nebius/QwQ-32B-Preview|qwq-32b|QwQ 32B|C|0.15,0.15|32768,32768|TJK|-|QwQ reasoning model|N
nebius/Qwen2.5-72B-Instruct|qwen-2.5-72b|Qwen 2.5 72B|C|0.40,0.40|131072,16384|TJS|-|Qwen 2.5 72B on Nebius|Y
nebius/bge-m3|bge-m3|BGE M3|C|0.02,|8192,1024|E|-|BGE M3 embeddings|N
nebius/bge-multilingual-gemma2|bge-gemma2|BGE Multilingual Gemma2|C|0.02,|8192,1024|E|-|Multilingual embeddings|N

# =============================================================================
# HYPERBOLIC - Inference (13 models)
# =============================================================================
hyperbolic/DeepSeek-R1|deepseek-r1|DeepSeek R1|C|0.50,2.0|65536,8192|TJK|-|DeepSeek R1 reasoning|N
hyperbolic/DeepSeek-R1-Zero|deepseek-r1-zero|DeepSeek R1 Zero|C|0.50,2.0|65536,8192|TJK|-|DeepSeek R1 Zero-shot|N
hyperbolic/DeepSeek-V3|deepseek-v3|DeepSeek V3|C|0.50,1.0|131072,8192|TJS|-|DeepSeek V3 on Hyperbolic|Y
hyperbolic/FLUX.1-dev|flux-dev|FLUX.1 Dev|C|0.0030,|77,0|I|-|FLUX image generation|N
hyperbolic/Hermes-3-Llama-3.1-70B|hermes-3-70b|Hermes 3 70B|C|0.40,0.40|131072,8192|TJS|-|Hermes 3 Llama 70B|Y
hyperbolic/Llama-3.3-70B-Instruct|llama-3.3-70b|Llama 3.3 70B|C|0.40,0.40|131072,8192|TJS|-|Llama 3.3 70B on Hyperbolic|Y
hyperbolic/Meta-Llama-3.1-405B-Instruct|llama-3.1-405b|Llama 3.1 405B|C|3.0,3.0|131072,8192|TJS|-|Llama 3.1 405B on Hyperbolic|Y
hyperbolic/Meta-Llama-3.1-70B-Instruct|llama-3.1-70b|Llama 3.1 70B|C|0.40,0.40|131072,8192|TJS|-|Llama 3.1 70B on Hyperbolic|Y
hyperbolic/Meta-Llama-3.1-8B-Instruct|llama-3.1-8b|Llama 3.1 8B|C|0.04,0.04|131072,8192|TJS|-|Llama 3.1 8B on Hyperbolic|Y
hyperbolic/QwQ-32B-Preview|qwq-32b|QwQ 32B|C|0.20,0.20|32768,32768|TJK|-|QwQ reasoning model|N
hyperbolic/Qwen2.5-72B-Instruct|qwen-2.5-72b|Qwen 2.5 72B|C|0.40,0.40|131072,8192|TJS|-|Qwen 2.5 72B on Hyperbolic|Y
hyperbolic/Qwen2.5-Coder-32B-Instruct|qwen-coder-32b|Qwen 2.5 Coder 32B|C|0.20,0.20|131072,8192|TJS|-|Qwen Coder on Hyperbolic|Y
hyperbolic/stable-diffusion-xl-base-1.0|sdxl|SDXL|C|0.0020,|77,0|I|-|SDXL image generation|N

# =============================================================================
# GRAPH (1 models)
# =============================================================================
graph/graphormer-base|graphormer|graph-transformer|Meta: Graphormer|C|0.0000,0.0000|512,256|S|-|Transformer for graph-structured data|Y

# =============================================================================
# # NOISE REDUCTION  (1 models)
# =============================================================================
# NOISE REDUCTION / ENHANCEMENT|||C|-|0,0|-|-||Y

# =============================================================================
# AMAZON (6 models)
# =============================================================================
amazon/nova-2-lite-v1|nova-2-lite-v1|Amazon: Nova 2 Lite|C|0.0000,0.0000|1000000,65535|KTV|-|Nova 2 Lite is a fast, cost-effective reasoning model for everyday workloads tha|N
amazon/nova-lite-v1|nova-lite-v1|Amazon: Nova Lite 1.0|C|0.0000,0.0000|300000,5120|TV|-|Amazon Nova Lite 1.0 is a very low-cost multimodal model from Amazon that focuse|Y
amazon/nova-micro-v1|nova-micro-v1|Amazon: Nova Micro 1.0|C|0.0000,0.0000|128000,5120|T|-|Amazon Nova Micro 1.0 is a text-only model that delivers the lowest latency resp|Y
amazon/nova-premier-latest|amazon-nova-premier|Amazon Nova Premier|C|0.0000,0.0000|300000,40000|SVT|-|Premium multimodal reasoning model|Y
amazon/nova-premier-v1|nova-premier-v1|Amazon: Nova Premier 1.0|C|0.0000,0.0000|1000000,32000|TV|-|Amazon Nova Premier is the most capable of Amazon's multimodal models for comple|Y
amazon/nova-pro-v1|nova-pro-v1|Amazon: Nova Pro 1.0|C|0.0000,0.0000|300000,5120|TV|-|Amazon Nova Pro 1.0 is a capable multimodal model from Amazon focused on providi|Y

# =============================================================================
# INCONTEXT (1 models)
# =============================================================================
incontext/incontext-learner|incontext-f|In-Context Learner|C|0.0003,0.0009|4096,1024|VSTJ|-|In-context learning model|Y

# =============================================================================
# MICROSOFT (7 models)
# =============================================================================
microsoft/phi-3-medium-128k-instruct|phi-3-medium-128k-in|Microsoft: Phi-3 Medium 128K Instruct|C|0.0000,0.0000|128000,32000|T|-|Phi-3 128K Medium is a powerful 14-billion parameter model designed for advanced|Y
microsoft/phi-3-mini-128k-instruct|phi-3-mini-128k-inst|Microsoft: Phi-3 Mini 128K Instruct|C|0.0000,0.0000|128000,32000|T|-|Phi-3 Mini is a powerful 3.8B parameter model designed for advanced language und|Y
microsoft/phi-3.5-mini-128k-instruct|phi-3.5-mini-128k-in|Microsoft: Phi-3.5 Mini 128K Instruct|C|0.0000,0.0000|128000,32000|T|-|Phi-3.5 models are lightweight, state-of-the-art open models. These models were|Y
microsoft/phi-4|phi-4|Microsoft: Phi 4|C|0.0000,0.0000|16384,4096|JS|-|[Microsoft Research](/microsoft) Phi-4 is designed to perform well in complex re|Y
microsoft/phi-4-multimodal-instruct|phi-4-multimodal-ins|Microsoft: Phi 4 Multimodal Instruct|C|0.0000,0.0000|131072,32768|JV|-|Phi-4 Multimodal Instruct is a versatile 5.6B parameter foundation model that co|Y
microsoft/phi-4-reasoning-plus|phi-4-reasoning-plus|Microsoft: Phi 4 Reasoning Plus|C|0.0000,0.0000|32768,8192|JK|-|Phi-4-reasoning-plus is an enhanced 14B parameter model from Microsoft, fine-tun|N
microsoft/wizardlm-2-8x22b|wizardlm-2-8x22b|WizardLM-2 8x22B|C|0.0000,0.0000|65536,16384|J|-|WizardLM-2 8x22B is Microsoft AI's most advanced Wizard model. It demonstrates h|Y

# =============================================================================
# AZURE (2 models)
# =============================================================================
azure/gpt-4-turbo-vision|azure-gpt-4-turbo-vision|Azure: GPT-4 Turbo Vision|C|0.0000,0.0000|128000,4096|VST|-|GPT-4 Turbo with vision via Azure OpenAI|Y
azure/gpt-4o-vision|azure-gpt-4o-vision|Azure: GPT-4o Vision|C|0.0025,0.01|128000,16384|VSTJS|-|GPT-4o deployed via Azure with vision|Y

# =============================================================================
# IR (2 models)
# =============================================================================
ir/cross-encoder-ms-marco-MiniLM-L-6-v2|cross-encoder-msmarco|ir-ranker|Sentence Transformers: Cross-Encoder MARCO|C|0.0000,0.0000|512,256|S|-|Cross-encoder for passage ranking|Y
ir/cross-encoder-qnli-distilroberta-base|cross-encoder-qnli|ir-classifier|Sentence Transformers: Cross-Encoder QNLI|C|0.0000,0.0000|512,256|S|-|Cross-encoder for query-document classification|Y

# =============================================================================
# PROMPT (1 models)
# =============================================================================
prompt/prompt-optimizer|prompt-opt-f|Prompt Optimizer|C|0.0002,0.0006|4096,1024|VSTJ|-|Prompt optimization model|Y

# =============================================================================
# THUDM (1 models)
# =============================================================================
thudm/glm-4.1v-9b-thinking|glm-4.1v-9b-thinking|THUDM: GLM 4.1V 9B Thinking|C|0.0000,0.0000|65536,8000|KV|-|GLM-4.1V-9B-Thinking is a 9B parameter vision-language model developed by THUDM,|N

# =============================================================================
# # TABULAR DATA (1 models)
# =============================================================================
# TABULAR DATA/STRUCTURED|||C|-|0,0|-|-||Y

# =============================================================================
# ALPINDALE (1 models)
# =============================================================================
alpindale/goliath-120b|goliath-120b|Goliath 120B|C|0.0000,0.0000|6144,1024|J|-|A large LLM created by combining two fine-tuned Llama 70B models into one 120B m|Y

# =============================================================================
# NER (6 models)
# =============================================================================
ner/bert-base-multilingual-cased-ner|bert-multilingual-ner|ner-multilingual|Hugging Face: Multilingual NER|C|0.0000,0.0000|512,256|S|-|Multilingual BERT for entity recognition|Y
ner/bert-large-uncased-ner|bert-ner-f|BERT Large NER|C|0.0001,0.0003|512,512|VSTJ|-|BERT NER for entities|Y
ner/biobert-base-cased-v1.1-ner|biobert-ner|bioner|BioBERT: NER|C|0.0000,0.0000|512,256|S|-|BioBERT for biomedical entity extraction|Y
ner/conllpp-ner|conllpp-ner|ner-conll|CoNLL++ NER|C|0.0000,0.0000|512,256|S|-|State-of-the-art CoNLL++ NER model|Y
ner/distilbert-ner|distilbert-ner-f|DistilBERT NER|C|0.0001,0.0003|512,512|VSTJ|-|DistilBERT NER token classification|Y
ner/xlm-roberta-large-finetuned-conll03-english|xlm-conll|ner-xlm|XLM-RoBERTa: CoNLL03 English|C|0.0000,0.0000|512,256|S|-|XLM for multilingual NER|Y

# =============================================================================
# STRUCTURED (2 models)
# =============================================================================
structured/ditto|ditto|data-matching|University of Wisconsin: DITTO|C|0.0000,0.0000|512,256|S|-|Deep learning for entity matching|Y
structured/ta-bert|tabert|tabular-bert|IBM: TabBERT|C|0.0000,0.0000|512,256|S|-|TabBERT for tabular data understanding|Y

# =============================================================================
# MEITUAN (1 models)
# =============================================================================
meituan/longcat-flash-chat|longcat-flash-chat|Meituan: LongCat Flash Chat|C|0.0000,0.0000|131072,131072|-|-|LongCat-Flash-Chat is a large-scale Mixture-of-Experts (MoE) model with 560B tot|Y

# =============================================================================
# MORPH (2 models)
# =============================================================================
morph/morph-v3-fast|morph-v3-fast|Morph: Morph V3 Fast|C|0.0000,0.0000|81920,38000|-|-|Morph's fastest apply model for code edits. ~10,500 tokens/sec with 96% accuracy|Y
morph/morph-v3-large|morph-v3-large|Morph: Morph V3 Large|C|0.0000,0.0000|262144,131072|-|-|Morph's high-accuracy apply model for complex code edits. ~4,500 tokens/sec with|Y

# =============================================================================
# ASIA (12 models)
# =============================================================================
asia/baichuan-13b|baichuan-13b-a|Baichuan: 13B|C|0.0004,0.0004|4096,2048|VSTJ|-|Baichuan Chinese optimized 13B|Y
asia/baichuan-7b|baichuan-7b-a|Baichuan: 7B|C|0.0002,0.0002|4096,2048|VSTJ|-|Baichuan 7B model|Y
asia/chatglm3|chatglm3-a|Zhipu: ChatGLM3|C|0.0005,0.0005|8192,2048|VSTJ|-|ChatGLM3 Chinese advanced|Y
asia/chatglm4|chatglm4-a|Zhipu: ChatGLM4|C|0.0010,0.0010|128000,2048|VSTJK|-|ChatGLM4 extended context|Y
asia/minicpm-2b|minicpm-2b-a|MiniCPM: 2B|C|0.0000,0.0000|4096,1024|VST|-|MiniCPM ultra-small|Y
asia/minicpm-v|minicpm-v-a|MiniCPM-V|C|0.0001,0.0001|1024,1024|VT|-|MiniCPM vision model|Y
asia/moonshot-128k|moonshot-128k-a|Moonshot: 128K|C|0.01,0.01|128000,2048|VSTJ|-|Moonshot 128K long context|Y
asia/moonshot-32k|moonshot-32k-a|Moonshot: 32K|C|0.0030,0.0030|32768,2048|VSTJ|-|Moonshot 32K extended|Y
asia/moonshot-8k|moonshot-8k-a|Moonshot: 8K|C|0.0010,0.0010|8192,2048|VSTJ|-|Moonshot 8K context model|Y
asia/xverse-65b|xverse-65b-a|XVERSE: 65B|C|0.0009,0.0009|4096,2048|VSTJ|-|XVERSE 65B multilingual|Y
asia/yi-1.5-34b|yi-34b-a|01.AI: Yi 1.5 34B|C|0.0006,0.0006|200000,2048|VSTJ|-|Yi 1.5 34B extended context|Y
asia/yi-1.5-9b|yi-9b-a|01.AI: Yi 1.5 9B|C|0.0001,0.0001|200000,2048|VSTJ|-|Yi 1.5 9B lightweight|Y

# =============================================================================
# ESSENTIALAI (1 models)
# =============================================================================
essentialai/rnj-1-instruct|rnj-1-instruct|EssentialAI: Rnj 1 Instruct|C|0.0000,0.0000|32768,8192|JS|-|Rnj-1 is an 8B-parameter, dense, open-weight model family developed by Essential|Y

# =============================================================================
# ADAPTER (2 models)
# =============================================================================
adapter/llama-medical-lora|llama-medical-lora-a|Meta: Llama Medical LoRA|C|0.0006,0.0006|8192,2048|VSTJ|-|Llama 3 with medical LoRA|Y
adapter/mistral-legal-lora|mistral-legal-lora-a|Mistral: Legal LoRA|C|0.0004,0.0004|8192,2048|VSTJ|-|Mistral with legal LoRA|Y

# =============================================================================
# PREMIUM (43 models)
# =============================================================================
premium/claude-haiku-research|claude-haiku-research-p|Anthropic: Claude Haiku Research|C|0.0001,0.0004|200000,4096|VSTJKC|-|Haiku research variant|Y
premium/claude-legal-pro|claude-legal-pro-p|Anthropic: Claude Legal Pro|C|7.0,35.0,0.70|200000,32000|VSTJKC|-|Claude legal expertise|Y
premium/claude-opus-4-enterprise-ultra|claude-opus-ultra-p|Anthropic: Claude Opus 4 Enterprise Ultra|C|10.0,50.0,1.0|200000,32000|VSTJSKC|-|Claude Opus 4 maximum tier|Y
premium/claude-opus-code|claude-code-p|Anthropic: Claude Opus Code|C|6.0,30.0,0.60|200000,32000|VSTJKC|-|Claude optimized for coding|Y
premium/claude-opus-compliant|claude-compliant-p|Anthropic: Claude Opus Compliant|C|7.0,35.0,0.70|200000,32000|VSTJKC|-|Claude compliance-ready|Y
premium/claude-opus-context-max|claude-context-max-p|Anthropic: Claude Opus Context Max|C|6.0,30.0,0.60|400000,32000|VSTJKC|-|Claude 400K context|Y
premium/claude-opus-multilingual|claude-multi-p|Anthropic: Claude Opus Multilingual|C|5.5,27.5,0.55|200000,32000|VSTJKC|-|Claude multilingual expert|Y
premium/claude-opus-precise|claude-precise-p|Anthropic: Claude Opus Precise|C|7.0,35.0,0.70|200000,32000|VSTJKC|-|Claude maximum precision|Y
premium/claude-opus-reasoning|claude-reasoning-p|Anthropic: Claude Opus Reasoning|C|5.0,25.0,0.50|200000,32000|VSTJSKC|-|Claude with enhanced reasoning|Y
premium/claude-opus-safe|claude-safe-p|Anthropic: Claude Opus Safe|C|6.5,32.5,0.65|200000,32000|VSTJKC|-|Claude with safety focus|Y
premium/claude-opus-turbo|claude-turbo-p|Anthropic: Claude Opus Turbo|C|6.0,30.0,0.60|200000,32000|VSTJKC|-|Claude optimized for speed|Y
premium/claude-opus-vision|claude-opus-vision-p|Anthropic: Claude Opus Vision|C|5.0,25.0,0.50|200000,4096|VSTJSKC|-|Claude with vision capabilities|Y
premium/claude-tuning-pro|claude-tuning-pro-p|Anthropic: Claude Tuning Pro|C|5.5,27.5,0.55|200000,32000|VSTJKC|-|Claude tuning-optimized|Y
premium/codegemini-2|codegemini-2-p|Google: Code Gemini 2|C|0.0075,0.03|1000000,8192|VSTJK|-|Gemini 2 code optimization|Y
premium/gemini-2-ultra|gemini-2-ultra-p|Google: Gemini 2 Ultra|C|0.02,0.08|1000000,8192|VSTJK|-|Gemini 2 ultra performance|Y
premium/gemini-compliance-plus|gemini-compliance-plus-p|Google: Gemini Compliance Plus|C|0.01,0.04|1000000,8192|VSTJ|-|Gemini compliance assured|Y
premium/gemini-context-giant|gemini-context-giant-p|Google: Gemini Context Giant|C|0.01,0.06|2000000,8192|VSTJ|-|Gemini 2M context|Y
premium/gemini-express|gemini-express-p|Google: Gemini Express|C|0.0050,0.02|1000000,8192|VSTJ|-|Gemini rapid response|Y
premium/gemini-financial-pro|gemini-fin-pro-p|Google: Gemini Financial Pro|C|0.01,0.04|1000000,8192|VSTJ|-|Gemini finance expert|Y
premium/gemini-flash-advanced|gemini-flash-adv-p|Google: Gemini Flash Advanced|C|0.0008,0.0030|1000000,8192|VSTJK|-|Gemini Flash advanced|Y
premium/gemini-multilingual-pro|gemini-multi-pro-p|Google: Gemini Multilingual Pro|C|0.0090,0.04|1000000,8192|VSTJ|-|Gemini language specialist|Y
premium/gemini-precise-pro|gemini-precise-pro-p|Google: Gemini Precise Pro|C|0.01,0.05|1000000,8192|VSTJ|-|Gemini precision mode|Y
premium/gemini-reasoning-pro|gemini-reason-pro-p|Google: Gemini Reasoning Pro|C|0.01,0.04|1000000,8192|VSTJK|-|Gemini with reasoning|Y
premium/gemini-safety-pro|gemini-safety-pro-p|Google: Gemini Safety Pro|C|0.0090,0.04|1000000,8192|VSTJ|-|Gemini safety certified|Y
premium/gemini-tuning-elite|gemini-tuning-elite-p|Google: Gemini Tuning Elite|C|0.0085,0.03|1000000,8192|VSTJ|-|Gemini tuning specialist|Y
premium/gemini-vision-ultra|gemini-vision-ultra-p|Google: Gemini Vision Ultra|C|0.0075,0.03|1000000,8192|VSTJK|-|Gemini Vision maximum|Y
premium/gpt-4-accurate|gpt-4-accurate-p|OpenAI: GPT-4 Accurate|C|0.0040,0.01|128000,16384|VSTJK|-|GPT-4 high accuracy|Y
premium/gpt-4-code-interpreter|gpt-4-code-p|OpenAI: GPT-4 Code Interpreter|C|0.0030,0.0090|128000,16384|VSTJK|-|GPT-4 with code execution|Y
premium/gpt-4-context-ultra|gpt-4-context-ultra-p|OpenAI: GPT-4 Context Ultra|C|0.0030,0.0090|256000,16384|VSTJK|-|GPT-4 ultra long context|Y
premium/gpt-4-fast|gpt-4-fast-p|OpenAI: GPT-4 Fast|C|0.0015,0.0045|128000,16384|VSTJ|-|GPT-4 low latency|Y
premium/gpt-4-guardrails|gpt-4-guardrails-p|OpenAI: GPT-4 Guardrails|C|0.0030,0.0090|128000,16384|VSTJK|-|GPT-4 with guardrails|Y
premium/gpt-4-medical-pro|gpt-4-medical-pro-p|OpenAI: GPT-4 Medical Pro|C|0.0040,0.01|128000,16384|VSTJK|-|GPT-4 medical specialist|Y
premium/gpt-4-mini-advanced|gpt-4-mini-adv-p|OpenAI: GPT-4 Mini Advanced|C|0.0001,0.0003|128000,16384|VSTJK|-|GPT-4 Mini advanced version|Y
premium/gpt-4-multilingual|gpt-4-multi-p|OpenAI: GPT-4 Multilingual|C|0.0020,0.0060|128000,16384|VSTJ|-|GPT-4 language expert|Y
premium/gpt-4-reasoning|gpt-4-reasoning-p|OpenAI: GPT-4 Reasoning|C|0.0030,0.0090|128000,16384|VSTJK|-|GPT-4 reasoning enhanced|Y
premium/gpt-4-regulatory|gpt-4-regulatory-p|OpenAI: GPT-4 Regulatory|C|0.0040,0.01|128000,16384|VSTJK|-|GPT-4 regulatory compliant|Y
premium/gpt-4-tuning-pro|gpt-4-tuning-pro-p|OpenAI: GPT-4 Tuning Pro|C|0.0025,0.0075|128000,16384|VSTJK|-|GPT-4 tuning enhanced|Y
premium/gpt-4-turbo-enterprise|gpt-4-turbo-ent-p|OpenAI: GPT-4 Turbo Enterprise|C|0.0020,0.0060,0.0010|128000,16384|VSTJK|-|GPT-4 Turbo enterprise edition|Y
premium/gpt-4-vision-ultra|gpt-4-vision-ultra-p|OpenAI: GPT-4 Vision Ultra|C|0.0025,0.0075|128000,4096|VSTJK|-|GPT-4 Vision ultra resolution|Y
premium/mistral-code-large|mistral-code-large-p|Mistral: Code Large|C|0.0010,0.0030|128000,8192|VSTJK|-|Mistral specialized code|Y
premium/mistral-large-2.5|mistral-large-25-p|Mistral: Large 2.5|C|0.0020,0.0060|128000,8192|VSTJ|-|Mistral Large 2.5 advanced|Y
premium/multimodal-fusion-ultra|fusion-ultra-p|ModelSuite: Multimodal Fusion Ultra|C|0.01,0.03|128000,8192|VSTJKC|-|Ultimate multimodal integration|Y
premium/vision-language-pro|vlm-pro-p|ModelSuite: Vision-Language Pro|C|0.0080,0.02|100000,4096|VSTJK|-|Pro multimodal synthesis|Y

# =============================================================================
# JAPANESE (3 models)
# =============================================================================
japanese/llama-2-13b-jp|llama-jp|llama-japanese|Meta: Llama 2 13B Japanese|C|0.0000,0.0000|4096,2048|VT|-|Llama 2 13B fine-tuned for Japanese|Y
japanese/mistral-large-jp|mistral-jp|mistral-japanese|Mistral: Large Japanese|C|0.0000,0.0000|8192,2048|VSTJ|-|Mistral Large optimized for Japanese language|Y
japanese/rinna-3.6b-instruction|rinna-3.6b|rinna|Rinna: 3.6B Instruction|C|0.0000,0.0000|2048,1024|T|-|Rinna 3.6B Japanese instruction model|Y

# =============================================================================
# XIAOMI (1 models)
# =============================================================================
xiaomi/mimo-v2-flash:free|mimo-v2-flash:free|Xiaomi: MiMo-V2-Flash (free)|C|-|262144,65536|JKT|-|MiMo-V2-Flash is an open-source foundation language model developed by Xiaomi. I|N

# =============================================================================
# # ACTION RECOGNITION  (1 models)
# =============================================================================
# ACTION RECOGNITION / VIDEO|||C|-|0,0|-|-||Y

# =============================================================================
# NOUSRESEARCH (7 models)
# =============================================================================
nousresearch/deephermes-3-mistral-24b-preview|deephermes-3-mistral|Nous: DeepHermes 3 Mistral 24B Preview|C|0.0000,0.0000|32768,32768|JKST|-|DeepHermes 3 (Mistral 24B Preview) is an instruction-tuned language model by Nou|Y
nousresearch/hermes-2-pro-llama-3-8b|hermes-2-pro-llama-3|NousResearch: Hermes 2 Pro - Llama-3 8B|C|0.0000,0.0000|8192,2048|JS|-|Hermes 2 Pro is an upgraded, retrained version of Nous Hermes 2, consisting of a|Y
nousresearch/hermes-3-llama-3.1-405b|hermes-3-llama-3.1-4|Nous: Hermes 3 405B Instruct|C|0.0000,0.0000|131072,16384|J|-|Hermes 3 is a generalist language model with many improvements over Hermes 2, in|Y
nousresearch/hermes-3-llama-3.1-405b:free|hermes-3-llama-3.1-4|Nous: Hermes 3 405B Instruct (free)|C|-|131072,32768|-|-|Hermes 3 is a generalist language model with many improvements over Hermes 2, in|Y
nousresearch/hermes-3-llama-3.1-70b|hermes-3-llama-3.1-7|Nous: Hermes 3 70B Instruct|C|0.0000,0.0000|65536,16384|JS|-|Hermes 3 is a generalist language model with many improvements over [Hermes 2](/|Y
nousresearch/hermes-4-405b|hermes-4-405b|Nous: Hermes 4 405B|C|0.0000,0.0000|131072,131072|JKST|-|Hermes 4 is a large-scale reasoning model built on Meta-Llama-3.1-405B and relea|Y
nousresearch/hermes-4-70b|hermes-4-70b|Nous: Hermes 4 70B|C|0.0000,0.0000|131072,131072|JKST|-|Hermes 4 70B is a hybrid reasoning model from Nous Research, built on Meta-Llama|Y

# =============================================================================
# UNKNOWN (161 models)
# =============================================================================
# 3D VISION|||C|-|0,0|-|-||Y
# ACCURACY FOCUSED|||C|-|0,0|-|-||Y
# ADAPTER & LORA VARIANTS|||C|-|0,0|-|-||Y
# ADDITIONAL ASIAN MODELS|||C|-|0,0|-|-||Y
# ADDITIONAL OPEN SOURCE|||C|-|0,0|-|-||Y
# ADDITIONAL PROVIDER MODELS|||C|-|0,0|-|-||Y
# ANOMALY DETECTION|||C|-|0,0|-|-||Y
# AUDIO & SPEECH MODELS|||C|-|0,0|-|-||Y
# AUDIO FINGERPRINTING|||C|-|0,0|-|-||Y
# AUDIO SOURCE SEPARATION|||C|-|0,0|-|-||Y
# AUDIO-SPECIALIZED FINE-TUNED|||C|-|0,0|-|-||Y
# AUTOMOTIVE|||C|-|0,0|-|-||Y
# AUTONOMOUS DRIVING (PERCEPTION)|||C|-|0,0|-|-||Y
# CALIBRATION & UNCERTAINTY|||C|-|0,0|-|-||Y
# CHAIN-OF-THOUGHT|||C|-|0,0|-|-||Y
# CHAT-OPTIMIZED VARIANTS|||C|-|0,0|-|-||Y
# CHINESE REGION MODELS|||C|-|0,0|-|-||Y
# CLASSIFICATION MODELS|||C|-|0,0|-|-||Y
# CLAUDE FINE-TUNED VARIANTS|||C|-|0,0|-|-||Y
# CODE GENERATION MODELS|||C|-|0,0|-|-||Y
# CODE-SPECIFIC MODELS|||C|-|0,0|-|-||Y
# COMPLIANCE READY|||C|-|0,0|-|-||Y
# CONSTITUENCY PARSING|||C|-|0,0|-|-||Y
# CONTENT GENERATION & MARKETING|||C|-|0,0|-|-||Y
# CONTEXT EXTENDED|||C|-|0,0|-|-||Y
# COREFERENCE RESOLUTION|||C|-|0,0|-|-||Y
# CROSS-LINGUAL TRANSFER|||C|-|0,0|-|-||Y
# CUSTOMER SERVICE & SUPPORT|||C|-|0,0|-|-||Y
# DATA LABELING & ANNOTATION|||C|-|0,0|-|-||Y
# DATA SCIENCE & ANALYTICS|||C|-|0,0|-|-||Y
# DEPENDENCY PARSING|||C|-|0,0|-|-||Y
# DIALOGUE & CONVERSATIONAL MODELS|||C|-|0,0|-|-||Y
# DISTILLED VARIANTS|||C|-|0,0|-|-||Y
# DOCUMENT UNDERSTANDING|||C|-|0,0|-|-||Y
# DOMAIN EXPERT PREMIUM|||C|-|0,0|-|-||Y
# DOMAIN SPECIFIC LANGUAGE|||C|-|0,0|-|-||Y
# DOMAIN-SPECIFIC EMERGING|||C|-|0,0|-|-||Y
# E-COMMERCE & RETAIL|||C|-|0,0|-|-||Y
# EDGE & MOBILE MODELS|||C|-|0,0|-|-||Y
# EDUCATION & LEARNING|||C|-|0,0|-|-||Y
# EMBEDDING MODELS|||C|-|0,0|-|-||Y
# EMERGING CHINESE MODELS|||C|-|0,0|-|-||Y
# EMERGING JAPANESE MODELS|||C|-|0,0|-|-||Y
# EMOTION RECOGNITION (SPEECH & AUDIO)|||C|-|0,0|-|-||Y
# ENERGY & UTILITIES|||C|-|0,0|-|-||Y
# ENTERPRISE LLM VARIANTS|||C|-|0,0|-|-||Y
# ENTERPRISE PREMIUM MODELS|||C|-|0,0|-|-||Y
# FACE DETECTION & RECOGNITION|||C|-|0,0|-|-||Y
# FINAL 1200+ MILESTONE MODELS|||C|-|0,0|-|-||Y
# FINANCIAL DOMAIN MODELS|||C|-|0,0|-|-||Y
# FINANCIAL DOMAIN SPECIFIC|||C|-|0,0|-|-||Y
# FINE-TUNING OPTIMIZED|||C|-|0,0|-|-||Y
# GEMINI FINE-TUNED VARIANTS|||C|-|0,0|-|-||Y
# GOVERNMENT & PUBLIC SECTOR|||C|-|0,0|-|-||Y
# GPT-4 FINE-TUNED VARIANTS|||C|-|0,0|-|-||Y
# GRAPH NEURAL NETWORKS|||C|-|0,0|-|-||Y
# HEALTHCARE SPECIFIC MODELS|||C|-|0,0|-|-||Y
# HOSPITALITY & TOURISM|||C|-|0,0|-|-||Y
# HUGGING FACE INFERENCE API|||C|-|0,0|-|-||Y
# HUMAN RESOURCES|||C|-|0,0|-|-||Y
# IMAGE CLASSIFICATION|||C|-|0,0|-|-||Y
# IN-CONTEXT LEARNING|||C|-|0,0|-|-||Y
# INDIAN REGION MODELS|||C|-|0,0|-|-||Y
# INFERENCE PLATFORMS (ANYSCALE|RUNPOD|VAST AI)|C|-|0,0|-|-|VAST AI)|Y
# INFORMATION RETRIEVAL & RANKING|||C|-|0,0|-|-||Y
# INSTANCE SEGMENTATION|||C|-|0,0|-|-||Y
# INSTRUCTION TUNED VARIANTS|||C|-|0,0|-|-||Y
# INSTRUCTION-FOLLOWING VARIANTS|||C|-|0,0|-|-||Y
# INSTRUCTION-TUNED VARIANTS|||C|-|0,0|-|-||Y
# JAPANESE REGION MODELS|||C|-|0,0|-|-||Y
# KNOWLEDGE DISTILLATION|||C|-|0,0|-|-||Y
# KOREAN REGION MODELS|||C|-|0,0|-|-||Y
# LANGUAGE DETECTION|||C|-|0,0|-|-||Y
# LATIN AMERICAN REGION MODELS|||C|-|0,0|-|-||Y
# LEGAL DOMAIN MODELS|||C|-|0,0|-|-||Y
# LEGAL DOMAIN SPECIFIC|||C|-|0,0|-|-||Y
# LIGHTWEIGHT EDGE MODELS|||C|-|0,0|-|-||Y
# LLAMA FINE-TUNED VARIANTS|||C|-|0,0|-|-||Y
# LONG CONTEXT MODELS|||C|-|0,0|-|-||Y
# MANUFACTURING & QUALITY|||C|-|0,0|-|-||Y
# MEDICAL DOMAIN MODELS|||C|-|0,0|-|-||Y
# MEDICAL IMAGING|||C|-|0,0|-|-||Y
# MIDDLE EAST & AFRICA MODELS|||C|-|0,0|-|-||Y
# MISTRAL FINE-TUNED VARIANTS|||C|-|0,0|-|-||Y
# MULTI-TASK LEARNING|||C|-|0,0|-|-||Y
# MULTILINGUAL MODELS|||C|-|0,0|-|-||Y
# MULTILINGUAL PREMIUM|||C|-|0,0|-|-||Y
# MULTIMODAL AUDIO|||C|-|0,0|-|-||Y
# MULTIMODAL INTEGRATION (AUDIO-VISUAL)|||C|-|0,0|-|-||Y
# MULTIMODAL OPEN SOURCE|||C|-|0,0|-|-||Y
# MULTIMODAL PREMIUM|||C|-|0,0|-|-||Y
# MULTIMODEL ENSEMBLE|||C|-|0,0|-|-||Y
# MUSIC UNDERSTANDING & GENERATION|||C|-|0,0|-|-||Y
# NAMED ENTITY LINKING|||C|-|0,0|-|-||Y
# NAMED ENTITY RECOGNITION|||C|-|0,0|-|-||Y
# NER & SEQUENCE LABELING|||C|-|0,0|-|-||Y
# NEW OPEN SOURCE RELEASES|||C|-|0,0|-|-||Y
# OBJECT DETECTION|||C|-|0,0|-|-||Y
# OBJECT DETECTION & LOCALIZATION|||C|-|0,0|-|-||Y
# OPTICAL CHARACTER RECOGNITION (OCR)|||C|-|0,0|-|-||Y
# PANOPTIC SEGMENTATION|||C|-|0,0|-|-||Y
# PARAPHRASE & SEMANTIC SIMILARITY|||C|-|0,0|-|-||Y
# PARAPHRASE MODELS|||C|-|0,0|-|-||Y
# POSE ESTIMATION|||C|-|0,0|-|-||Y
# PREFERENCE-ALIGNED VARIANTS|||C|-|0,0|-|-||Y
# PROMPT OPTIMIZATION|||C|-|0,0|-|-||Y
# PROPRIETARY PROVIDER MODELS|||C|-|0,0|-|-||Y
# QUANTIZED INFERENCE|||C|-|0,0|-|-||Y
# QUESTION ANSWERING|||C|-|0,0|-|-||Y
# QUESTION ANSWERING SPECIALIZED|||C|-|0,0|-|-||Y
# RANKING & LEARNING-TO-RANK|||C|-|0,0|-|-||Y
# REAL ESTATE|||C|-|0,0|-|-||Y
# REASONING & MATH MODELS|||C|-|0,0|-|-||Y
# REASONING & MATHS MODELS|||C|-|0,0|-|-||Y
# REASONING OPTIMIZED|||C|-|0,0|-|-||Y
# RECOMMENDATION SYSTEMS|||C|-|0,0|-|-||Y
# REINFORCEMENT LEARNING MODELS|||C|-|0,0|-|-||Y
# REPLICATE EDGE & LOCAL MODELS|||C|-|0,0|-|-||Y
# RERANKER MODELS|||C|-|0,0|-|-||Y
# RESEARCH & ADVANCED MODELS|||C|-|0,0|-|-||Y
# RESEARCH LAB MODELS|||C|-|0,0|-|-||Y
# RETRIEVAL & EMBEDDING MODELS|||C|-|0,0|-|-||Y
# RETRIEVAL AUGMENTED GENERATION (RAG)|||C|-|0,0|-|-||Y
# RETRIEVAL-SPECIALIZED FINE-TUNED|||C|-|0,0|-|-||Y
# SAFETY CERTIFIED|||C|-|0,0|-|-||Y
# SATELLITE & AERIAL IMAGING|||C|-|0,0|-|-||Y
# SCIENTIFIC RESEARCH MODELS|||C|-|0,0|-|-||Y
# SEMANTIC ROLE LABELING|||C|-|0,0|-|-||Y
# SEMANTIC SEARCH|||C|-|0,0|-|-||Y
# SEMANTIC SEGMENTATION|||C|-|0,0|-|-||Y
# SEMANTIC TEXTUAL SIMILARITY|||C|-|0,0|-|-||Y
# SENTIMENT & EMOTION ANALYSIS|||C|-|0,0|-|-||Y
# SENTIMENT ANALYSIS|||C|-|0,0|-|-||Y
# SOFTWARE DEVELOPMENT|||C|-|0,0|-|-||Y
# SOUND CLASSIFICATION & TAGGING|||C|-|0,0|-|-||Y
# SOUTHEAST ASIAN MODELS|||C|-|0,0|-|-||Y
# SPARSE MODELS|||C|-|0,0|-|-||Y
# SPEAKER RECOGNITION|||C|-|0,0|-|-||Y
# SPECIALIZED CODING MODELS|||C|-|0,0|-|-||Y
# SPECIALIZED FINE-TUNED MODELS|||C|-|0,0|-|-||Y
# SPECIALIZED MULTIMODAL|||C|-|0,0|-|-||Y
# SPECIALIZED REASONING MODELS|||C|-|0,0|-|-||Y
# SPECIALIZED TRANSLATION MODELS|||C|-|0,0|-|-||Y
# SPEECH RECOGNITION (ASR)|||C|-|0,0|-|-||Y
# SPEED OPTIMIZED|||C|-|0,0|-|-||Y
# SUMMARIZATION MODELS|||C|-|0,0|-|-||Y
# SUMMARIZATION SPECIALIZED|||C|-|0,0|-|-||Y
# SYNTHETIC DATA GENERATION|||C|-|0,0|-|-||Y
# TELECOMMUNICATIONS|||C|-|0,0|-|-||Y
# TEXT CLASSIFICATION VARIANTS|||C|-|0,0|-|-||Y
# TEXT-TO-SPEECH (TTS)|||C|-|0,0|-|-||Y
# TOGETHER AI ADDITIONAL MODELS|||C|-|0,0|-|-||Y
# TOKENIZATION MODELS|||C|-|0,0|-|-||Y
# TRANSLATION MODELS|||C|-|0,0|-|-||Y
# TREE-OF-THOUGHTS|||C|-|0,0|-|-||Y
# VIDEO UNDERSTANDING|||C|-|0,0|-|-||Y
# VISION PREMIUM|||C|-|0,0|-|-||Y
# VISION SPECIALIST|||C|-|0,0|-|-||Y
# VISION-SPECIALIZED FINE-TUNED|||C|-|0,0|-|-||Y
# ZERO-SHOT & FEW-SHOT|||C|-|0,0|-|-||Y
# ZERO-SHOT CLASSIFICATION|||C|-|0,0|-|-||Y

# =============================================================================
# # MUSIC GENERATION  (1 models)
# =============================================================================
# MUSIC GENERATION / ANALYSIS|||C|-|0,0|-|-||Y

# =============================================================================
# NEL (1 models)
# =============================================================================
nel/luke-large-finetuned-conll-2003|luke-ner|entity-linker|Studio Ousia: LUKE NER|C|0.0000,0.0000|512,256|S|-|LUKE for entity typing and linking|Y

# =============================================================================
# OPENSOURCE (10 models)
# =============================================================================
opensource/llama-3.1-405b|llama-31-405b-f|Meta: Llama 3.1 405B|C|0.0027,0.0081|131072,4096|VSTJK|-|Llama 3.1 405B ultra|Y
opensource/llama-3.1-70b|llama-31-70b-f|Meta: Llama 3.1 70B|C|0.0006,0.0008|131072,4096|VSTJK|-|Llama 3.1 70B large|Y
opensource/llama-3.1-8b|llama-31-8b-f|Meta: Llama 3.1 8B|C|0.0000,0.0001|131072,4096|VSTJK|-|Llama 3.1 8B small|Y
opensource/llama-3.2-1b|llama-32-1b-f|Meta: Llama 3.2 1B|C|0.0000,0.0000|8192,4096|VSTJK|-|Llama 3.2 1B tiny|Y
opensource/llama-3.2-90b|llama-32-90b-f|Meta: Llama 3.2 90B|C|0.0008,0.0008|8192,4096|VSTJK|-|Llama 3.2 90B multimodal|Y
opensource/mistral-7b-v0.3|mistral-v03-f|Mistral: 7B v0.3|C|0.0001,0.0001|32768,2048|VSTJ|-|Mistral 7B v0.3 latest|Y
opensource/mistral-nemo|nemo-f|Mistral: Nemo|C|0.0001,0.0001|8192,2048|VSTJ|-|Mistral Nemo 12B|Y
opensource/nous-hermes-3-405b|hermes-3-405b-f|NousResearch: Hermes 3 405B|C|0.0030,0.0090|8192,2048|VSTJ|-|Hermes 3 405B ultra|Y
opensource/nous-hermes-3-70b|hermes-3-70b-f|NousResearch: Hermes 3 70B|C|0.0006,0.0008|8192,2048|VSTJ|-|Hermes 3 70B advanced|Y
opensource/solstice-7b|solstice-7b-f|Solstice: 7B|C|0.0001,0.0001|4096,2048|VSTJ|-|Solstice 7B optimized|Y

# =============================================================================
# # QUANTIZED (1 models)
# =============================================================================
# QUANTIZED/COMPRESSED VARIANTS|||C|-|0,0|-|-||Y

# =============================================================================
# CLASSIFICATION (3 models)
# =============================================================================
classification/deberta-v3-large-mnli|deberta-mnli|text-inference|Microsoft: DeBERTa MNLI|C|0.0000,0.0000|512,256|S|-|DeBERTa for textual entailment|Y
classification/electra-large-discriminator-finetuned-rte|electra-rte|text-entailment|Google: ELECTRA RTE|C|0.0000,0.0000|512,256|S|-|ELECTRA for RTE entailment classification|Y
classification/xlnet-base-cased-imdb|xlnet-classification|text-classify|XLNet: IMDB Classification|C|0.0000,0.0000|512,256|S|-|XLNet for document classification|Y

# =============================================================================
# RERANKER (3 models)
# =============================================================================
reranker/bge-reranker-large|bge-reranker-f|BAAI: BGE Reranker Large|C|0.0001,0.0004|512,256|VSTJ|-|BGE semantic reranking|Y
reranker/jina-reranker-v1|jina-reranker-f|Jina: Reranker v1|C|0.0001,0.0003|8192,256|VSTJ|-|Jina reranking model|Y
reranker/rankgpt|rankgpt-f|RankGPT|C|0.0005,0.0015|4096,256|VSTJ|-|GPT-based reranking|Y

# =============================================================================
# ENTERPRISE (4 models)
# =============================================================================
enterprise/claude-opus-4.5-20251101|claude-4.5-prod|opus-enterprise|Anthropic: Claude Opus 4.5 (Enterprise)|C|5.0,25.0|200000,32000|VTJSKC|-|Claude Opus 4.5 optimized for enterprise deployments|Y
enterprise/gemini-2.5-flash-prod|gemini-2.5-flash-ent|gemini-enterprise|Google: Gemini 2.5 Flash (Enterprise)|C|0.38,1.5|1000000,8192|VSTJK|-|Gemini 2.5 Flash for enterprise-scale inference|Y
enterprise/gpt-4o-20250101|gpt-4o-prod|gpt-4o-enterprise|OpenAI: GPT-4o (Enterprise)|C|2.5,10.0|128000,16384|VSTJS|-|GPT-4o optimized for enterprise production|Y
enterprise/mistral-large-2-prod|mistral-large-ent|mistral-enterprise|Mistral: Large 2 (Enterprise)|C|1.2,3.6|128000,8192|VSTJ|-|Mistral Large 2 for production deployments|Y

# =============================================================================
# KWAIPILOT (1 models)
# =============================================================================
kwaipilot/kat-coder-pro:free|kat-coder-pro:free|Kwaipilot: KAT-Coder-Pro V1 (free)|C|-|256000,32768|JST|-|KAT-Coder-Pro V1 is KwaiKAT's most advanced agentic coding model in the KAT-Code|Y

# =============================================================================
# MULTITASK (1 models)
# =============================================================================
multitask/multitask-nlp|multitask-f|MultiTask NLP|C|0.0002,0.0006|512,512|VSTJ|-|Multi-task NLP model|Y

# =============================================================================
# MULTIMODAL (6 models)
# =============================================================================
multimodal/albef|albef-m|ALBEF Contrastive|C|0.0002,0.0006|256,256|VSTJ|-|Align before fusing multimodal|Y
multimodal/av-hubert|av-hubert-m|Audio-Visual HuBERT|C|0.0002,0.0004|16000,512|VSTJ|-|Audio-visual pre-training|Y
multimodal/clip-audio|clip-audio-m|CLIP with Audio|C|0.0002,0.0005|16000,512|VSTJ|-|Audio-visual CLIP|Y
multimodal/layoutlm-v3|layoutlm-v3-m|LayoutLMv3|C|0.0001,0.0004|512,512|VSTJ|-|Document layout understanding|Y
multimodal/uniter|uniter-m|UNITER|C|0.0002,0.0006|512,512|VSTJ|-|Universal image-text representation|Y
multimodal/vision-audio-fusion|fusion-va-m|Vision-Audio Fusion|C|0.0003,0.0006|16000,512|VSTJ|-|Unified audio-visual fusion|Y

# =============================================================================
# NEVERSLEEP (2 models)
# =============================================================================
neversleep/llama-3.1-lumimaid-8b|llama-3.1-lumimaid-8|NeverSleep: Lumimaid v0.2 8B|C|0.0000,0.0000|32768,8192|JS|-|Lumimaid v0.2 8B is a finetune of [Llama 3.1 8B](/models/meta-llama/llama-3.1-8b|Y
neversleep/noromaid-20b|noromaid-20b|Noromaid 20B|C|0.0000,0.0000|4096,1024|JS|-|A collab between IkariDev and Undi. This merge is suitable for RP, ERP, and gene|Y

# =============================================================================
# BETA (3 models)
# =============================================================================
beta/anthropic-research-opus|claude-research-b|Anthropic: Research Opus|C|3.0,12.0|200000,32000|VSTJKC|-|Claude Opus research edition|Y
beta/gemini-2-preview|gemini-2-prev|Google: Gemini 2 Preview|C|0.01,0.04|1000000,8192|VSTJK|-|Gemini 2 preview edition|Y
beta/gpt-4.5-preview|gpt-4.5-prev|OpenAI: GPT-4.5 Preview|C|0.0020,0.0060|128000,16384|VSTJK|-|GPT-4.5 early preview|Y

# =============================================================================
# CHAT (2 models)
# =============================================================================
chat/neural-chat-7b-v3-conversation|neural-chat-conv|neural-conv|Intel: Neural Chat 7B Conversation|C|0.0000,0.0000|8192,2048|VSTJ|-|Neural Chat optimized for natural conversation flow|Y
chat/zephyr-7b-beta-chat-optimized|zephyr-chat|zephyr-conv|HuggingFace: Zephyr 7B Chat Optimized|C|0.0000,0.0000|4096,2048|VSTJ|-|Zephyr 7B optimized for chat interactions|Y

# =============================================================================
# MOBILE (7 models)
# =============================================================================
mobile/distilbert-mobile|distilbert-mobile|distilbert-tiny|Hugging Face: DistilBERT Mobile|C|0.0000,0.0000|512,256|S|-|Lightweight DistilBERT for edge devices|Y
mobile/mobilebert|mobilebert|mobile-bert|Google: MobileBERT|C|0.0000,0.0000|512,256|S|-|MobileBERT for mobile NLP deployment|Y
mobile/mobilenetv3-small|mobilenetv3|mobilenet-small|Google: MobileNetV3 Small|C|0.0000,0.0000|224,256|V|-|MobileNetV3 Small for image classification|Y
mobile/mobilevibert|mobilevit|mobilevit-small|Apple: MobileViT|C|0.0000,0.0000|512,256|VS|-|Vision transformer for mobile devices|Y
mobile/squeezebert|squeezebert|squeeze-bert|SqueezeBERT: Base|C|0.0000,0.0000|512,256|S|-|SqueezeBERT for efficient inference|Y
mobile/tinybert-6l-768d|tinybert|tinybert-6l|Huawei: TinyBERT 6L|C|0.0000,0.0000|512,256|S|-|TinyBERT 6L distilled for edge inference|Y
mobile/xlnet-tiny|xlnet-tiny|xlnet-mobile|Google: XLNet Tiny|C|0.0000,0.0000|512,256|S|-|Tiny XLNet for lightweight deployment|Y

# =============================================================================
# XAI (1 models)
# =============================================================================
xai/grok-vision|grok-vision|xAI: Grok Vision|C|0.0000,0.0000|128000,8192|VST|-|xAI Grok multimodal model with real-time data|Y

# =============================================================================
# CLASSIFY (3 models)
# =============================================================================
classify/deberta-v3-large|deberta-f|DeBERTa v3 Large|C|0.0001,0.0004|512,256|VSTJ|-|DeBERTa classification|Y
classify/distilroberta-base|distilroberta-f|DistilRoBERTa Base|C|0.0001,0.0003|512,256|VSTJ|-|RoBERTa classification|Y
classify/electra-large|electra-f|ELECTRA Large|C|0.0001,0.0003|512,256|VSTJ|-|ELECTRA classification|Y

# =============================================================================
# LIQUID (2 models)
# =============================================================================
liquid/lfm-2.2-6b|lfm-2.2-6b|LiquidAI/LFM2-2.6B|C|0.0000,0.0000|32768,8192|-|-|LFM2 is a new generation of hybrid models developed by Liquid AI, specifically d|Y
liquid/lfm2-8b-a1b|lfm2-8b-a1b|LiquidAI/LFM2-8B-A1B|C|0.0000,0.0000|32768,8192|-|-|Model created via inbox interface|Y

# =============================================================================
# INCEPTION (2 models)
# =============================================================================
inception/mercury|mercury|Inception: Mercury|C|0.0000,0.0000|128000,16384|JST|-|Mercury is the first diffusion large language model (dLLM). Applying a breakthro|Y
inception/mercury-coder|mercury-coder|Inception: Mercury Coder|C|0.0000,0.0000|128000,16384|JST|-|Mercury Coder is the first diffusion large language model (dLLM). Applying a bre|Y

# =============================================================================
# QUANTIZED (2 models)
# =============================================================================
quantized/llama-3-70b-q4|llama-70b-q4|Meta: Llama 3 70B Q4|C|0.0004,0.0004|8192,2048|VST|-|Llama 3 70B quantized to 4-bit|Y
quantized/mistral-8x22b-q5|mixtral-22b-q5|Mistral: Mixtral 8x22B Q5|C|0.0005,0.0005|32768,2048|VST|-|Mixtral 8x22B quantized to 5-bit|Y

# =============================================================================
# VISION (51 models)
# =============================================================================
vision/3dmax|3dmax-v|Kaolin: 3D Max Pool|C|0.0001,0.0001|1024,1024|VT|-|3D object detection|Y
vision/aerial-detection|aerial-detection-v|Aerial Object Detection|C|0.0001,0.0002|640,640|VT|-|High-resolution aerial detection|Y
vision/arcface|arcface-v|ArcFace Recognition|C|0.0001,0.0001|512,512|VT|-|Large margin cosine loss|Y
vision/bev-perception|bev-perception-v|BEV Perception|C|0.0003,0.0003|1024,1024|VT|-|Bird's eye view perception|Y
vision/blip-2|blip-2-v|Salesforce: BLIP-2|C|0.0001,0.0003|4096,2048|VT|-|Multimodal foundation model|Y
vision/change-detection|change-detection-v|Multi-temporal Change|C|0.0001,0.0003|512,512|VT|-|SAR change detection|Y
vision/clip-vit-base-finetuned-multilingual|clip-multilingual|clip-ft|OpenAI: CLIP ViT Multilingual FT|C|0.0000,0.0000|512,256|VS|-|CLIP fine-tuned for multilingual image-text understanding|Y
vision/convnext-large|convnext-large-v|ConvNeXt Large|C|0.0001,0.0001|224,224|VT|-|Vision backbone modernization|Y
vision/cutpaste|cutpaste-v|CutPaste Anomaly|C|0.0001,0.0001|256,256|VT|-|Self-supervised anomaly detection|Y
vision/deeplab-v3+|deeplab-v3plus-v|DeepLab v3+ ResNet101|C|0.0001,0.0001|1024,1024|VT|-|Semantic segmentation with atrous convolution|Y
vision/deeplabv3-resnet50|deeplabv3|segment-deeplab|DeepLabV3 ResNet50|C|0.0000,0.0000|512,512|V|-|DeepLabV3 for semantic segmentation|Y
vision/detr|detr-v|Facebook: DETR|C|0.0001,0.0001|800,800|VT|-|Detection Transformer architecture|Y
vision/dinov2-base-finetuned-object-detection|dinov2-det|dinov2-ft|Meta: DINOv2 Object Detection FT|C|0.0000,0.0000|512,256|VS|-|DINOv2 fine-tuned for object detection|Y
vision/dinov2-large|dinov2-large-v|Meta: DINOv2 Large|C|0.0001,0.0001|2048,2048|VT|-|Vision backbone without labels|Y
vision/easyocr|easyocr-v|EasyOCR|C|0.0001,0.0002|2048,2048|VT|-|Easy-to-use OCR toolkit|Y
vision/efficientnet-b7|efficientnet-b7-v|EfficientNet-B7|C|0.0001,0.0001|600,600|VT|-|Efficient image classification|Y
vision/faster-rcnn|faster-rcnn-v|Faster R-CNN ResNet50|C|0.0001,0.0001|800,800|VT|-|Regional CNN with FPN|Y
vision/faster-rcnn-resnet101-fpn|faster-rcnn|object-detect-rcnn|Faster R-CNN ResNet101 FPN|C|0.0000,0.0000|800,512|V|-|Faster R-CNN with FPN backbone|Y
vision/hrnet|hrnet-v|HRNet Pose|C|0.0001,0.0001|256,256|VT|-|High resolution pose network|Y
vision/industrial-anomaly-detection|industrial-ad-v|Industrial Anomaly Detector|C|0.0001,0.0002|512,512|VT|-|Manufacturing defect detection|Y
vision/lidar-3d|lidar-3d-v|LiDAR 3D Detection|C|0.0003,0.0003|2048,2048|VT|-|LiDAR-based 3D object detection|Y
vision/mask-rcnn|mask-rcnn-v|Mask R-CNN FPN|C|0.0001,0.0001|1024,1024|VT|-|Instance segmentation with masks|Y
vision/mask-rcnn-resnet50-fpn|mask-rcnn|segment-instance|Mask R-CNN ResNet50 FPN|C|0.0000,0.0000|800,512|V|-|Mask R-CNN for instance segmentation|Y
vision/medsam|medsam-v|Meta: Medical SAM|C|0.0002,0.0002|1024,1024|VT|-|Segment Anything for medical|Y
vision/monai-segmentation|monai-seg-v|MONAI Segmentation|C|0.0002,0.0002|1024,1024|VT|-|Medical image segmentation|Y
vision/nerf|nerf-v|NeRF Multi-View|C|0.0002,0.0002|512,512|VT|-|Neural radiance fields|Y
vision/openpose|openpose-v|CMU: OpenPose|C|0.0001,0.0001|640,480|VT|-|Multi-person pose estimation|Y
vision/paddleocr|paddleocr-v|PaddleOCR v4|C|0.0001,0.0001|2048,2048|VT|-|Multilingual OCR system|Y
vision/padim|padim-v|PaDiM Anomaly Detection|C|0.0001,0.0001|256,256|VT|-|Patch-level anomaly detection|Y
vision/panoptic-fpn|panoptic-fpn-v|Panoptic FPN|C|0.0001,0.0001|1024,1024|VT|-|Panoptic segmentation FPN|Y
vision/panoptic-fpn-resnet101|panoptic-fpn|segment-panoptic|Panoptic FPN ResNet101|C|0.0000,0.0000|800,512|V|-|Panoptic FPN combining semantic and instance|Y
vision/radiomics|radiomics-v|Radiomics Feature Extractor|C|0.0001,0.0001|512,512|VT|-|Quantitative imaging analysis|Y
vision/resnet50-finetuned-medical-imaging|resnet50-medical|resnet-medical-ft|ResNet50 Medical FT|C|0.0000,0.0000|224,256|VS|-|ResNet50 fine-tuned for medical imaging classification|Y
vision/retinaface|retinaface-v|RetinaFace|C|0.0001,0.0001|640,640|VT|-|Multi-task face detection|Y
vision/segformer|segformer-v|NVIDIA: SegFormer|C|0.0001,0.0001|1024,1024|VT|-|Efficient semantic segmentation|Y
vision/segformer-b0|segformer-small|segment-efficient|NVIDIA: SegFormer-B0|C|0.0000,0.0000|512,512|V|-|Efficient SegFormer for segmentation|Y
vision/sentinel-classification|sentinel-class-v|Sentinel 2 Classification|C|0.0001,0.0001|512,512|VT|-|Land cover classification|Y
vision/surya-ocr|surya-ocr-v|Surya OCR|C|0.0001,0.0003|4096,4096|VT|-|Multilingual reading comprehension OCR|Y
vision/swin-large|swin-large-v|Swin Transformer Large|C|0.0001,0.0001|224,224|VT|-|Shifted window transformer|Y
vision/tesseract-5|tesseract-5-v|Tesseract 5.0|C|0.0000,0.0001|2048,2048|VT|-|Open-source OCR engine|Y
vision/vggface2|vggface2-v|VGGFace2|C|0.0001,0.0001|224,224|VT|-|Large-scale face recognition|Y
vision/vit-large-21k|vit-large-21k-v|Vision Transformer Large|C|0.0001,0.0001|224,224|VT|-|Large ViT pre-trained on 21K|Y
vision/voxel-rcnn|voxel-rcnn-v|Voxel R-CNN|C|0.0001,0.0001|1024,1024|VT|-|3D point cloud detection|Y
vision/yolov5-medium|yolov5-medium|object-detect-med|YOLOv5 Medium|C|0.0000,0.0000|640,512|V|-|YOLOv5 Medium balanced detector|Y
vision/yolov5-small|yolov5-small|object-detect-small|YOLOv5 Small|C|0.0000,0.0000|640,512|V|-|YOLOv5 Small for object detection|Y
vision/yolov8-pose|yolov8-pose-v|Ultralytics: YOLOv8 Pose|C|0.0001,0.0001|640,640|VT|-|Real-time pose detection|Y
vision/yolov8-traffic|yolov8-traffic-v|YOLOv8 Traffic|C|0.0002,0.0002|640,640|VT|-|Traffic sign and light detection|Y
vision/yolov8m|yolov8m-v|Ultralytics: YOLOv8 Medium|C|0.0001,0.0001|640,640|VT|-|Medium object detection model|Y
vision/yolov8n|yolov8n-v|Ultralytics: YOLOv8 Nano|C|0.0000,0.0000|640,640|VT|-|Real-time object detection nano model|Y
vision/yolov8s|yolov8s-v|Ultralytics: YOLOv8 Small|C|0.0000,0.0000|640,640|VT|-|Small object detection model|Y
vision/yolov9-large|yolov9-large-v|Ultralytics: YOLOv9 Large|C|0.0001,0.0001|640,640|VT|-|Large YOLOv9 with attention|Y

# =============================================================================
# DEEPCOGITO (4 models)
# =============================================================================
deepcogito/cogito-v2-preview-llama-109b-moe|cogito-v2-preview-ll|Cogito V2 Preview Llama 109B|C|0.0000,0.0000|32767,8191|KTV|-|An instruction-tuned, hybrid-reasoning Mixture-of-Experts model built on Llama-4|N
deepcogito/cogito-v2-preview-llama-405b|cogito-v2-preview-ll|Deep Cogito: Cogito V2 Preview Llama 405B|C|0.0000,0.0000|32768,8192|JKST|-|Cogito v2 405B is a dense hybrid reasoning model that combines direct answering|Y
deepcogito/cogito-v2-preview-llama-70b|cogito-v2-preview-ll|Deep Cogito: Cogito V2 Preview Llama 70B|C|0.0000,0.0000|32768,8192|JKST|-|Cogito v2 70B is a dense hybrid reasoning model that combines direct answering c|Y
deepcogito/cogito-v2.1-671b|cogito-v2.1-671b|Deep Cogito: Cogito v2.1 671B|C|0.0000,0.0000|128000,32000|JKS|-|Cogito v2.1 671B MoE represents one of the strongest open models globally, match|N

# =============================================================================
# MANCER (1 models)
# =============================================================================
mancer/weaver|weaver|Mancer: Weaver (alpha)|C|0.0000,0.0000|8000,2000|J|-|An attempt to recreate Claude-style verbosity, but don't expect the same level o|Y

# =============================================================================
# OPENGVLAB (1 models)
# =============================================================================
opengvlab/internvl3-78b|internvl3-78b|OpenGVLab: InternVL3 78B|C|0.0000,0.0000|32768,32768|JSV|-|The InternVL3 series is an advanced multimodal large language model (MLLM). Comp|Y

# =============================================================================
# LABELING (2 models)
# =============================================================================
labeling/label-studio|labelstudio-f|Label Studio Model|C|0.0001,0.0004|4096,512|VSTJ|-|Active learning labeling|Y
labeling/prodigy-model|prodigy-f|Prodigy Annotation|C|0.0002,0.0006|4096,512|VSTJ|-|Active learning annotation|Y

# =============================================================================
# DOCUMENT (2 models)
# =============================================================================
document/layoutlm-base-uncased|layoutlm|doc-understanding|Microsoft: LayoutLM Base|C|0.0000,0.0000|512,256|VS|-|LayoutLM for document layout understanding|Y
document/layoutlmv2-base-uncased|layoutlmv2|doc-layout-v2|Microsoft: LayoutLMv2 Base|C|0.0000,0.0000|512,256|VS|-|LayoutLMv2 improved document understanding|Y

# =============================================================================
# RUNPOD (1 models)
# =============================================================================
runpod/mistral-7b-instruct|mistral-7b-rp|Mistral 7B Instruct (RunPod)|C|0.0001,0.0001|32768,2048|VSTJ|-|Mistral 7B on RunPod serverless|Y

# =============================================================================
# TIMESERIES (2 models)
# =============================================================================
timeseries/n-beats|nbeats|forecast-nbeats|Element AI: N-BEATS|C|0.0000,0.0000|512,256|S|-|Neural basis expansion for forecasting|Y
timeseries/temporal-transformer|temporal-trans|forecast-temporal|Google: Temporal Transformer|C|0.0000,0.0000|512,256|S|-|Transformer for temporal sequence modeling|Y

# =============================================================================
# VIDEO (5 models)
# =============================================================================
video/actionclip|actionclip-v|ActionCLIP Video|C|0.0001,0.0001|8,8|VT|-|Contrastive learning for action|Y
video/slowfast-101-r101|slowfusion|video-slowfast|SlowFusion 101|C|0.0000,0.0000|8,2048|V|-|SlowFast network for video understanding|Y
video/slowfusion|slowfusion-v|SlowFusion 3D CNN|C|0.0002,0.0002|16,16|VT|-|Spatiotemporal 3D CNN|Y
video/timesformer|timesformer-v|TimeSformer Video|C|0.0003,0.0003|8,8|VT|-|Transformer for video understanding|Y
video/timesformer-base-finetuned-kinetics-400|timesformer|video-action|Meta: TimeSformer Kinetics|C|0.0000,0.0000|8,2048|V|-|Vision transformer for video action recognition|Y

# =============================================================================
# SRL (1 models)
# =============================================================================
srl/diegogarciar-electra-base-srl-english|electra-srl|semantic-roles|ELECTRA: Semantic Role Labeling|C|0.0000,0.0000|512,256|S|-|ELECTRA for semantic role labeling|Y

# =============================================================================
# # REINFORCEMENT LEARNING  (1 models)
# =============================================================================
# REINFORCEMENT LEARNING / POLICY MODELS|||C|-|0,0|-|-||Y

# =============================================================================
# ANYSCALE (1 models)
# =============================================================================
anyscale/meta-llama/Llama-2-13b|llama-2-13b-any|Meta: Llama 2 13B (Anyscale)|C|0.0001,0.0002|4096,2048|VSTJ|-|Llama 2 13B via Anyscale|Y

# =============================================================================
# MANUFACTURING (2 models)
# =============================================================================
manufacturing/predictivemaint|predictmaint|mfg-maintenance|PredictiveMaintGPT|C|0.0000,0.0000|8192,2048|VSTJ|-|Predictive maintenance and equipment optimization|Y
manufacturing/qualitygpt|qualitygpt|mfg-quality|QualityGPT: Inspection|C|0.0000,0.0000|8192,2048|VSTJ|-|Quality control and defect detection|Y

# =============================================================================
# EDUCATION (3 models)
# =============================================================================
education/assessgpt|assessgpt|edu-assess|AssessGPT: Evaluation|C|0.0000,0.0000|4096,2048|VSTJ|-|Educational assessment and grading automation|Y
education/currigpt|currigpt|edu-curriculum|CurricGPT: Planning|C|0.0000,0.0000|8192,2048|VSTJ|-|Curriculum design and educational planning|Y
education/tutorgpt|tutorgpt|edu-tutor|TutorGPT: Educational|C|0.0000,0.0000|8192,2048|VSTJ|-|Personalized tutoring and learning assistance|Y

# =============================================================================
# NVIDIA (9 models)
# =============================================================================
nvidia/llama-3.1-nemotron-70b-instruct|llama-3.1-nemotron-7|NVIDIA: Llama 3.1 Nemotron 70B Instruct|C|0.0000,0.0000|131072,16384|JT|-|NVIDIA's Llama 3.1 Nemotron 70B is a language model designed for generating prec|Y
nvidia/llama-3.1-nemotron-ultra-253b-v1|llama-3.1-nemotron-u|NVIDIA: Llama 3.1 Nemotron Ultra 253B v1|C|0.0000,0.0000|131072,32768|JKS|-|Llama-3.1-Nemotron-Ultra-253B-v1 is a large language model (LLM) optimized for a|N
nvidia/llama-3.3-nemotron-super-49b-v1.5|llama-3.3-nemotron-s|NVIDIA: Llama 3.3 Nemotron Super 49B V1.5|C|0.0000,0.0000|131072,32768|JKT|-|Llama-3.3-Nemotron-Super-49B-v1.5 is a 49B-parameter, English-centric reasoning/|N
nvidia/nemotron-3-nano-30b-a3b|nemotron-3-nano-30b-|NVIDIA: Nemotron 3 Nano 30B A3B|C|0.0000,0.0000|262144,262144|JKST|-|NVIDIA Nemotron 3 Nano 30B A3B is a small language MoE model with highest comput|Y
nvidia/nemotron-3-nano-30b-a3b:free|nemotron-3-nano-30b-|NVIDIA: Nemotron 3 Nano 30B A3B (free)|C|-|256000,64000|KT|-|NVIDIA Nemotron 3 Nano 30B A3B is a small language MoE model with highest comput|N
nvidia/nemotron-nano-12b-v2-vl|nemotron-nano-12b-v2|NVIDIA: Nemotron Nano 12B 2 VL|C|0.0000,0.0000|131072,32768|JKV|-|NVIDIA Nemotron Nano 2 VL is a 12-billion-parameter open multimodal reasoning mo|N
nvidia/nemotron-nano-12b-v2-vl:free|nemotron-nano-12b-v2|NVIDIA: Nemotron Nano 12B 2 VL (free)|C|-|128000,128000|KTV|-|NVIDIA Nemotron Nano 2 VL is a 12-billion-parameter open multimodal reasoning mo|N
nvidia/nemotron-nano-9b-v2|nemotron-nano-9b-v2|NVIDIA: Nemotron Nano 9B V2|C|0.0000,0.0000|131072,32768|JKT|-|NVIDIA-Nemotron-Nano-9B-v2 is a large language model (LLM) trained from scratch|N
nvidia/nemotron-nano-9b-v2:free|nemotron-nano-9b-v2:|NVIDIA: Nemotron Nano 9B V2 (free)|C|-|128000,32000|JKST|-|NVIDIA-Nemotron-Nano-9B-v2 is a large language model (LLM) trained from scratch|Y

# =============================================================================
# DETECT (1 models)
# =============================================================================
detect/langdetect|langdetect-f|LangDetect|C|0.0000,0.0000|1024,32|VST|-|Language identification|Y

# =============================================================================
# TRANSFER (1 models)
# =============================================================================
transfer/xlm-cross-lingual|xlm-transfer-f|XLM Cross-Lingual|C|0.0001,0.0003|512,512|VSTJ|-|Cross-lingual transfer|Y

# =============================================================================
# ZEROSHOT (3 models)
# =============================================================================
zeroshot/bart-large-mnli|bart-mnli-f|BART Large MNLI|C|0.0001,0.0004|1024,256|VSTJ|-|BART zero-shot classification|Y
zeroshot/distiluse-base-multilingual-cased-v2|distiluse-zs|zeroshot-sentence|Sentence Transformers: DistilUSE|C|0.0000,0.0000|512,256|VS|-|DistilUSE for zero-shot classification|Y
zeroshot/mDeBERTa-large-zero-shot|mdeberta-zs|zeroshot-large|Microsoft: mDeBERTa Zero-Shot|C|0.0000,0.0000|512,256|S|-|Multilingual DeBERTa for zero-shot|Y

# =============================================================================
# SUPPORT (3 models)
# =============================================================================
support/intentgpt|intentgpt|support-intent|IntentGPT: Understanding|C|0.0000,0.0000|4096,2048|VSTJ|-|Intent classification for support chatbots|Y
support/multilingual-support|multisupp|support-multi|MultilingualSupport LLM|C|0.0000,0.0000|8192,2048|VSTJ|-|Multilingual customer support automation|Y
support/supportgpt|supportgpt|support-bot|SupportGPT: Ticketing|C|0.0000,0.0000|8192,2048|VSTJ|-|Customer support automation and ticket routing|Y

# =============================================================================
# JAPAN (1 models)
# =============================================================================
japan/cyberagent-llama-70b|cyberagent-llama-j|CyberAgent: Llama 70B JP|C|0.0009,0.0009|8192,2048|VSTJ|-|Llama 70B Japanese optimized|Y

# =============================================================================
# FINANCE (7 models)
# =============================================================================
finance/blipbert-finance|blipbert-fin|finance-bert|Bloomberg: BLIPBert Finance|C|0.0000,0.0000|4096,2048|VSTJ|-|Fine-tuned for financial document analysis|Y
finance/econombert|econombert|finance-econ|EconBERT: Economic|C|0.0000,0.0000|4096,2048|VSTJ|-|BERT model for economic and financial texts|Y
finance/financialbert|financialbert|financial-domain|FinancialBERT: Domain|C|0.0000,0.0000|512,256|S|-|FinancialBERT for financial text|Y
finance/secbert|secbert|sec-bert|SecBERT: SEC Filings|C|0.0000,0.0000|512,256|S|-|SecBERT trained on SEC filings|Y
finance/stockbert|stockbert|stock-bert|StockBERT: Market|C|0.0000,0.0000|512,256|S|-|StockBERT for stock market analysis|Y
finance/stockllama|stockllama|finance-stock|StockLlama: Trading|C|0.0000,0.0000|8192,2048|VSTJ|-|Llama variant for stock market analysis and trading|Y
finance/tradingllm|tradingllm|finance-trading|TradingLLM: Quantitative|C|0.0000,0.0000|8192,2048|VSTJ|-|Specialized LLM for quantitative trading strategies|Y

# =============================================================================
# OLLAMA (4 models)
# =============================================================================
ollama/llama2-7b-chat-q4|llama2-7b-q4|llama-q4|Ollama: Llama 2 7B Q4|C|0.0000,0.0000|4096,2048|JT|-|Llama 2 7B chat quantized to 4-bit|Y
ollama/mistral-7b-instruct-q4|mistral-7b-q4|mistral-q4|Ollama: Mistral 7B Q4|C|0.0000,0.0000|32768,4096|JT|-|Mistral 7B quantized to 4-bit for efficient inference|Y
ollama/neural-chat-7b-v3-3-q5|neural-chat-q5|neural-q5|Ollama: Neural Chat Q5|C|0.0000,0.0000|8192,2048|JT|-|Neural Chat 7B quantized to 5-bit|Y
ollama/openchat-3.5-q4|openchat-q4|openchat-q4|Ollama: OpenChat 3.5 Q4|C|0.0000,0.0000|8192,2048|JT|-|OpenChat 3.5 quantized to 4-bit|Y

# =============================================================================
# SUMMARY (3 models)
# =============================================================================
summary/bart-large-cnn|bart-cnn-f|BART Large CNN|C|0.0001,0.0003|1024,512|VSTJ|-|BART CNN news summarization|Y
summary/pegasus-xsum|pegasus-xsum-f|PEGASUS XSum|C|0.0001,0.0003|1024,512|VSTJ|-|PEGASUS extreme summarization|Y
summary/t5-large|t5-large-f|T5 Large|C|0.0001,0.0003|512,512|VSTJ|-|T5 text-to-text large|Y

# =============================================================================
# SPARSE (2 models)
# =============================================================================
sparse/llm-pruned-70b|llm-pruned-70b-s|ModelSuite: Pruned LLM 70B|C|0.0005,0.0005|8192,2048|VSTJ|-|Sparsely pruned 70B model|Y
sparse/vision-pruned-large|vision-pruned-l-s|ModelSuite: Pruned Vision|C|0.0001,0.0002|2048,2048|VT|-|Sparsely pruned vision model|Y

# =============================================================================
# META (5 models)
# =============================================================================
meta/llama-2-70b-finetuned-code-llama|llama-code|llama-code-ft|Meta: Llama 2 70B Code FT|C|0.0000,0.0000|8192,2048|VSTJ|-|Llama 70B fine-tuned for code generation|Y
meta/llama-2-70b-finetuned-instruct-medical|llama-medical|llama-med-ft|Meta: Llama 2 70B Medical FT|C|0.0000,0.0000|8192,2048|VSTJ|-|Llama 70B fine-tuned for medical domain|Y
meta/llama-3-70b-finetuned-chat|llama-3-chat|llama-3-chat-ft|Meta: Llama 3 70B Chat FT|C|0.0000,0.0000|8192,2048|VSTJ|-|Llama 3 70B fine-tuned for chat|Y
meta/llama-4-405b-20260115|llama-4-405b|Llama 4 405B|C|0.0000,0.0000|128000,8192|ST|-|Latest Meta flagship reasoning model|Y
meta/llama-4-70b-20260115|llama-4-70b|Llama 4 70B|C|0.0000,0.0000|128000,8192|ST|-|Mid-range Llama 4 model|Y

# =============================================================================
# SEA (3 models)
# =============================================================================
sea/multilingual-t5-large|mT5-large|mT5-sea|Google: mT5 Large|C|0.0000,0.0000|512,512|T|-|Multilingual T5 Large for SE Asian tasks|Y
sea/ph-llama-13b|ph-llama|manila-llama|Philippine: Llama 13B|C|0.0000,0.0000|4096,2048|T|-|Llama 13B optimized for Philippine English and Tagalog|Y
sea/xlm-roberta-large|xlm-roberta|xlm-sea|Facebook: XLM-RoBERTa Large|C|0.0000,0.0000|512,256|S|-|XLM-RoBERTa for Southeast Asian languages|Y

# =============================================================================
# ENSEMBLE (2 models)
# =============================================================================
ensemble/llm-fusion|llm-fusion-e|ModelSuite: LLM Fusion|C|0.0050,0.01|8192,2048|VSTJK|-|Ensemble combining multiple models|Y
ensemble/vision-fusion|vision-fusion-e|ModelSuite: Vision Fusion|C|0.0010,0.0020|4096,2048|VSTJK|-|Ensemble combining vision models|Y

# =============================================================================
# SYNTHETIC (1 models)
# =============================================================================
synthetic/llm-synthetic|synthetic-llm-f|Synthetic Data LLM|C|0.0003,0.0009|4096,2048|VSTJ|-|Synthetic data generation|Y

# =============================================================================
# MISTRALAI (35 models)
# =============================================================================
mistralai/codestral-2508|codestral-2508|Mistral: Codestral 2508|C|0.0000,0.0000|256000,64000|JST|-|Mistral's cutting-edge language model for coding released end of July 2025. Code|Y
mistralai/devstral-2512|devstral-2512|Mistral: Devstral 2 2512|C|0.0000,0.0000|262144,65536|JST|-|Devstral 2 is a state-of-the-art open-source model by Mistral AI specializing in|Y
mistralai/devstral-2512:free|devstral-2512:free|Mistral: Devstral 2 2512 (free)|C|-|262144,65536|JST|-|Devstral 2 is a state-of-the-art open-source model by Mistral AI specializing in|Y
mistralai/devstral-medium|devstral-medium|Mistral: Devstral Medium|C|0.0000,0.0000|131072,32768|JST|-|Devstral Medium is a high-performance code generation and agentic reasoning mode|Y
mistralai/devstral-small|devstral-small|Mistral: Devstral Small 1.1|C|0.0000,0.0000|128000,32000|JST|-|Devstral Small 1.1 is a 24B parameter open-weight language model for software en|Y
mistralai/devstral-small-2505|devstral-small-2505|Mistral: Devstral Small 2505|C|0.0000,0.0000|128000,32000|J|-|Devstral-Small-2505 is a 24B parameter agentic LLM fine-tuned from Mistral-Small|Y
mistralai/ministral-14b-2512|ministral-14b-2512|Mistral: Ministral 3 14B 2512|C|0.0000,0.0000|262144,65536|JSTV|-|The largest model in the Ministral 3 family, Ministral 3 14B offers frontier cap|Y
mistralai/ministral-3b|ministral-3b|Mistral: Ministral 3B|C|0.0000,0.0000|131072,32768|JST|-|Ministral 3B is a 3B parameter model optimized for on-device and edge computing.|Y
mistralai/ministral-3b-2512|ministral-3b-2512|Mistral: Ministral 3 3B 2512|C|0.0000,0.0000|131072,32768|JSTV|-|The smallest model in the Ministral 3 family, Ministral 3 3B is a powerful, effi|Y
mistralai/ministral-8b|ministral-8b|Mistral: Ministral 8B|C|0.0000,0.0000|131072,32768|JST|-|Ministral 8B is an 8B parameter model featuring a unique interleaved sliding-win|Y
mistralai/ministral-8b-2512|ministral-8b-2512|Mistral: Ministral 3 8B 2512|C|0.0000,0.0000|262144,65536|JSTV|-|A balanced model in the Ministral 3 family, Ministral 3 8B is a powerful, effici|Y
mistralai/mistral-7b-instruct|mistral-7b-instruct|Mistral: Mistral 7B Instruct|C|0.0000,0.0000|32768,16384|JT|-|A high-performing, industry-standard 7.3B parameter model, with optimizations fo|Y
mistralai/mistral-7b-instruct-v0.1|mistral-7b-instruct-|Mistral: Mistral 7B Instruct v0.1|C|0.0000,0.0000|2824,1024|-|-|A 7.3B parameter model that outperforms Llama 2 13B on all benchmarks, with opti|Y
mistralai/mistral-7b-instruct-v0.2|mistral-7b-instruct-|Mistral: Mistral 7B Instruct v0.2|C|0.0000,0.0000|32768,8192|-|-|A high-performing, industry-standard 7.3B parameter model, with optimizations fo|Y
mistralai/mistral-7b-instruct-v0.3|mistral-7b-instruct-|Mistral: Mistral 7B Instruct v0.3|C|0.0000,0.0000|32768,4096|-|-|A high-performing, industry-standard 7.3B parameter model, with optimizations fo|Y
mistralai/mistral-7b-instruct:free|mistral-7b-instruct:|Mistral: Mistral 7B Instruct (free)|C|-|32768,16384|JT|-|A high-performing, industry-standard 7.3B parameter model, with optimizations fo|Y
mistralai/mistral-large|mistral-large|Mistral Large|C|0.0000,0.0000|128000,32000|JST|-|This is Mistral AI's flagship model, Mistral Large 2 (version 'mistral-large-240|Y
mistralai/mistral-large-2407|mistral-large-2407|Mistral Large 2407|C|0.0000,0.0000|131072,32768|JST|-|This is Mistral AI's flagship model, Mistral Large 2 (version mistral-large-2407|Y
mistralai/mistral-large-2411|mistral-large-2411|Mistral Large 2411|C|0.0000,0.0000|131072,32768|JST|-|Mistral Large 2 2411 is an update of [Mistral Large 2](/mistralai/mistral-large)|Y
mistralai/mistral-large-2512|mistral-large-2512|Mistral: Mistral Large 3 2512|C|0.0000,0.0000|262144,65536|JSTV|-|Mistral Large 3 2512 is Mistral's most capable model to date, featuring a sparse|Y
mistralai/mistral-medium-3|mistral-medium-3|Mistral: Mistral Medium 3|C|0.0000,0.0000|131072,32768|JSTV|-|Mistral Medium 3 is a high-performance enterprise-grade language model designed|Y
mistralai/mistral-medium-3.1|mistral-medium-3.1|Mistral: Mistral Medium 3.1|C|0.0000,0.0000|131072,32768|JSTV|-|Mistral Medium 3.1 is an updated version of Mistral Medium 3, which is a high-pe|Y
mistralai/mistral-nemo|mistral-nemo|Mistral: Mistral Nemo|C|0.0000,0.0000|131072,16384|JST|-|A 12B parameter model with a 128k token context length built by Mistral in colla|Y
mistralai/mistral-saba|mistral-saba|Mistral: Saba|C|0.0000,0.0000|32768,8192|JST|-|Mistral Saba is a 24B-parameter language model specifically designed for the Mid|Y
mistralai/mistral-small-24b-instruct-2501|mistral-small-24b-in|Mistral: Mistral Small 3|C|0.0000,0.0000|32768,32768|JST|-|Mistral Small 3 is a 24B-parameter language model optimized for low-latency perf|Y
mistralai/mistral-small-3.1-24b-instruct|mistral-small-3.1-24|Mistral: Mistral Small 3.1 24B|C|0.0000,0.0000|131072,131072|JSTV|-|Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501),|Y
mistralai/mistral-small-3.1-24b-instruct:free|mistral-small-3.1-24|Mistral: Mistral Small 3.1 24B (free)|C|-|128000,32000|JSTV|-|Mistral Small 3.1 24B Instruct is an upgraded variant of Mistral Small 3 (2501),|Y
mistralai/mistral-small-3.2-24b-instruct|mistral-small-3.2-24|Mistral: Mistral Small 3.2 24B|C|0.0000,0.0000|131072,131072|JSTV|-|Mistral-Small-3.2-24B-Instruct-2506 is an updated 24B parameter model from Mistr|Y
mistralai/mistral-small-creative|mistral-small-creati|Mistral: Mistral Small Creative|C|0.0000,0.0000|32768,8192|T|-|Mistral Small Creative is an experimental small model designed for creative writ|Y
mistralai/mistral-tiny|mistral-tiny|Mistral Tiny|C|0.0000,0.0000|32768,8192|JST|-|Note: This model is being deprecated. Recommended replacement is the newer [Mini|Y
mistralai/mixtral-8x22b-instruct|mixtral-8x22b-instru|Mistral: Mixtral 8x22B Instruct|C|0.0000,0.0000|65536,16384|JST|-|Mistral's official instruct fine-tuned version of [Mixtral 8x22B](/models/mistra|Y
mistralai/mixtral-8x7b-instruct|mixtral-8x7b-instruc|Mistral: Mixtral 8x7B Instruct|C|0.0000,0.0000|32768,16384|JT|-|Mixtral 8x7B Instruct is a pretrained generative Sparse Mixture of Experts, by M|Y
mistralai/pixtral-12b|pixtral-12b|Mistral: Pixtral 12B|C|0.0000,0.0000|32768,8192|JSTV|-|The first multi-modal, text+image-to-text model from Mistral AI. Its weights wer|Y
mistralai/pixtral-large-2411|pixtral-large-2411|Mistral: Pixtral Large 2411|C|0.0000,0.0000|131072,32768|JSTV|-|Pixtral Large is a 124B parameter, open-weight, multimodal model built on top of|Y
mistralai/voxtral-small-24b-2507|voxtral-small-24b-25|Mistral: Voxtral Small 24B 2507|C|0.0000,0.0000|32000,8000|JST|-|Voxtral Small is an enhancement of Mistral Small 3, incorporating state-of-the-a|Y

# =============================================================================
# MIDDLE_EAST (2 models)
# =============================================================================
middle_east/arabert-base|arabert|arabert-large|AraGPT: AraBoRT Base|C|0.0000,0.0000|512,256|S|-|AraBoRT for Arabic language understanding|Y
middle_east/gpt2-arabic|gpt2-ar|gpt2-arabic|GPT2: Arabic|C|0.0000,0.0000|1024,512|T|-|GPT-2 Arabic variant for text generation|Y

# =============================================================================
# PARAPHRASE (1 models)
# =============================================================================
paraphrase/paraphrase-multilingual|paraphrase-f|Paraphrase Multilingual|C|0.0001,0.0003|512,512|VSTJ|-|Multilingual paraphrasing|Y

# =============================================================================
# TRANSLATION (7 models)
# =============================================================================
translation/m2m-100|m2m-100-t|Facebook: M2M-100|C|0.0001,0.0002|512,512|VTJ|-|Many-to-Many translation model|Y
translation/m2m-100-1.2b|m2m-1.2b|m2m-medium|Facebook: M2M 1.2B|C|0.0000,0.0000|512,256|T|-|M2M 100 1.2B enhanced translation|Y
translation/m2m-100-418m|m2m-418m|m2m-small|Facebook: M2M 418M|C|0.0000,0.0000|512,256|T|-|M2M 100 418M many-to-many translation|Y
translation/nllb-200|nllb-200-t|Meta: NLLB-200|C|0.0001,0.0003|1024,512|VTJ|-|Meta's No Language Left Behind 200|Y
translation/nllb-200-1.3b|nllb-1.3b|nllb-small|Meta: NLLB 1.3B|C|0.0000,0.0000|512,256|T|-|NLLB 1.3B for multilingual translation|Y
translation/nllb-200-3.3b|nllb-3.3b|nllb-medium|Meta: NLLB 3.3B|C|0.0000,0.0000|512,512|T|-|NLLB 3.3B medium multilingual model|Y
translation/nllb-200-distilled|nllb-200|nllb-dist|Meta: NLLB 200M|C|0.0000,0.0000|512,256|T|-|NLLB 200M distilled for 200 languages|Y

# =============================================================================
# INFLECTION (2 models)
# =============================================================================
inflection/inflection-3-pi|inflection-3-pi|Inflection: Inflection 3 Pi|C|0.0000,0.0000|8000,1024|-|-|Inflection 3 Pi powers Inflection's [Pi](https://pi.ai) chatbot, including backs|Y
inflection/inflection-3-productivity|inflection-3-product|Inflection: Inflection 3 Productivity|C|0.0000,0.0000|8000,1024|-|-|Inflection 3 Productivity is optimized for following instructions. It is better|Y

# =============================================================================
# QA (5 models)
# =============================================================================
qa/deberta-large-qa|deberta-qa-f|DeBERTa Large QA|C|0.0001,0.0003|512,256|VSTJ|-|DeBERTa QA extraction|Y
qa/deepset-bert-base-uncased-squad2|deepset-squad|qa-bert|Deepset: BERT Base SQuAD2|C|0.0000,0.0000|512,256|S|-|BERT fine-tuned for SQuAD 2.0 QA|Y
qa/electra-base-discriminator-qa|electra-qa|electra-squad|Google: ELECTRA Base QA|C|0.0000,0.0000|512,256|S|-|ELECTRA tuned for extractive QA|Y
qa/electra-large-qa|electra-qa-f|ELECTRA Large QA|C|0.0001,0.0003|512,256|VSTJ|-|ELECTRA question answering|Y
qa/mrqa-small|mrqa-small|qa-mrqa|IBM: MRQA Small|C|0.0000,0.0000|512,256|S|-|MRQA multi-dataset question answering|Y

# =============================================================================
# # EDGE (1 models)
# =============================================================================
# EDGE/MOBILE/LIGHTWEIGHT MODELS (sub-2B parameters)|||C|-|0,0|-|-||Y

# =============================================================================
# TOKENIZE (2 models)
# =============================================================================
tokenize/gpt2-tokenizer|gpt2-tokenizer-f|GPT-2 Tokenizer|C|0.0000,0.0000|8192,256|VST|-|BPE tokenization|Y
tokenize/sentencepiece|sentencepiece-f|SentencePiece|C|0.0000,0.0000|8192,256|VST|-|Subword tokenization|Y

# =============================================================================
# META-LLAMA (19 models)
# =============================================================================
meta-llama/llama-3-70b-instruct|llama-3-70b-instruct|Meta: Llama 3 70B Instruct|C|0.0000,0.0000|8192,16384|JST|-|Meta's latest class of model (Llama 3) launched with a variety of sizes & flavor|Y
meta-llama/llama-3-8b-instruct|llama-3-8b-instruct|Meta: Llama 3 8B Instruct|C|0.0000,0.0000|8192,16384|JT|-|Meta's latest class of model (Llama 3) launched with a variety of sizes & flavor|Y
meta-llama/llama-3.1-405b|llama-3.1-405b|Meta: Llama 3.1 405B (base)|C|0.0000,0.0000|32768,32768|-|-|Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flav|Y
meta-llama/llama-3.1-405b-instruct|llama-3.1-405b-instr|Meta: Llama 3.1 405B Instruct|C|0.0000,0.0000|10000,2500|JST|-|The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context|Y
meta-llama/llama-3.1-405b-instruct:free|llama-3.1-405b-instr|Meta: Llama 3.1 405B Instruct (free)|C|-|131072,32768|-|-|The highly anticipated 400B class of Llama3 is here! Clocking in at 128k context|Y
meta-llama/llama-3.1-70b-instruct|llama-3.1-70b-instru|Meta: Llama 3.1 70B Instruct|C|0.0000,0.0000|131072,32768|JT|-|Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flav|Y
meta-llama/llama-3.1-8b-instruct|llama-3.1-8b-instruc|Meta: Llama 3.1 8B Instruct|C|0.0000,0.0000|131072,16384|JST|-|Meta's latest class of model (Llama 3.1) launched with a variety of sizes & flav|Y
meta-llama/llama-3.2-11b-vision-instruct|llama-3.2-11b-vision|Meta: Llama 3.2 11B Vision Instruct|C|0.0000,0.0000|131072,16384|JV|-|Llama 3.2 11B Vision is a multimodal model with 11 billion parameters, designed|Y
meta-llama/llama-3.2-1b-instruct|llama-3.2-1b-instruc|Meta: Llama 3.2 1B Instruct|C|0.0000,0.0000|60000,15000|-|-|Llama 3.2 1B is a 1-billion-parameter language model focused on efficiently perf|Y
meta-llama/llama-3.2-3b-instruct|llama-3.2-3b-instruc|Meta: Llama 3.2 3B Instruct|C|0.0000,0.0000|131072,16384|JT|-|Llama 3.2 3B is a 3-billion-parameter multilingual large language model, optimiz|Y
meta-llama/llama-3.2-3b-instruct:free|llama-3.2-3b-instruc|Meta: Llama 3.2 3B Instruct (free)|C|-|131072,32768|-|-|Llama 3.2 3B is a 3-billion-parameter multilingual large language model, optimiz|Y
meta-llama/llama-3.2-90b-vision-instruct|llama-3.2-90b-vision|Meta: Llama 3.2 90B Vision Instruct|C|0.0000,0.0000|32768,16384|JV|-|The Llama 90B Vision model is a top-tier, 90-billion-parameter multimodal model|Y
meta-llama/llama-3.3-70b-instruct|llama-3.3-70b-instru|Meta: Llama 3.3 70B Instruct|C|0.0000,0.0000|131072,16384|JST|-|The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and i|Y
meta-llama/llama-3.3-70b-instruct:free|llama-3.3-70b-instru|Meta: Llama 3.3 70B Instruct (free)|C|-|131072,32768|T|-|The Meta Llama 3.3 multilingual large language model (LLM) is a pretrained and i|Y
meta-llama/llama-4-maverick|llama-4-maverick|Meta: Llama 4 Maverick|C|0.0000,0.0000|1048576,16384|JSTV|-|Llama 4 Maverick 17B Instruct (128E) is a high-capacity multimodal language mode|Y
meta-llama/llama-4-scout|llama-4-scout|Meta: Llama 4 Scout|C|0.0000,0.0000|327680,16384|JSTV|-|Llama 4 Scout 17B Instruct (16E) is a mixture-of-experts (MoE) language model de|Y
meta-llama/llama-guard-2-8b|llama-guard-2-8b|Meta: LlamaGuard 2 8B|C|0.0000,0.0000|8192,2048|-|-|This safeguard model has 8B parameters and is based on the Llama 3 family. Just|Y
meta-llama/llama-guard-3-8b|llama-guard-3-8b|Llama Guard 3 8B|C|0.0000,0.0000|131072,32768|J|-|Llama Guard 3 is a Llama-3.1-8B pretrained model, fine-tuned for content safety|Y
meta-llama/llama-guard-4-12b|llama-guard-4-12b|Meta: Llama Guard 4 12B|C|0.0000,0.0000|163840,40960|JV|-|Llama Guard 4 is a Llama 4 Scout-derived multimodal pretrained model, fine-tuned|Y

# =============================================================================
# HF (2 models)
# =============================================================================
hf/meta-llama/Llama-2-70b-chat|llama-2-70b-hf|Meta: Llama 2 70B Chat (HF)|C|0.0010,0.0020|4096,2048|VSTJ|-|Llama 2 70B Chat via HF Inference|Y
hf/mistralai/Mistral-7B-Instruct-v0.2|mistral-7b-v2-hf|Mistral: 7B Instruct v0.2 (HF)|C|0.0001,0.0001|32768,2048|VSTJ|-|Mistral 7B v0.2 via HF Inference|Y

# =============================================================================
# LEGAL (6 models)
# =============================================================================
legal/contractgpt|contractgpt|legal-contract|ContractGPT: Agreements|C|0.0000,0.0000|8192,2048|VSTJ|-|GPT variant specialized for contract analysis|Y
legal/contractnorm-legal|contractnorm|contract-legal|ContractNorm: Legal|C|0.0000,0.0000|512,256|S|-|ContractNorm for contract analysis|Y
legal/legalai-llama|legalai-llama|legal-llama|LegalAI: Llama Legal|C|0.0000,0.0000|8192,2048|VSTJ|-|Llama fine-tuned for legal document analysis|Y
legal/legalbert|legalbert|legal-domain|LegalBERT: Domain|C|0.0000,0.0000|512,256|S|-|LegalBERT specialized for legal documents|Y
legal/legalberta|legalberta|legal-albert|LegalBERTa: ALBERTa|C|0.0000,0.0000|512,256|S|-|LegalBERTa efficient legal model|Y
legal/regulationgpt|regulationgpt|legal-reg|RegulationGPT: Compliance|C|0.0000,0.0000|8192,2048|VSTJ|-|Model for regulatory compliance and legal research|Y

# =============================================================================
# RAIFLE (1 models)
# =============================================================================
raifle/sorcererlm-8x22b|sorcererlm-8x22b|SorcererLM 8x22B|C|0.0000,0.0000|16000,4000|-|-|SorcererLM is an advanced RP and storytelling model, built as a Low-rank 16-bit|Y

# =============================================================================
# SENTIMENT (6 models)
# =============================================================================
sentiment/bert-base-uncased-finetuned-emotion|bert-emotion|emotion-analysis|Hugging Face: BERT Emotion|C|0.0000,0.0000|512,256|S|-|BERT trained for emotion detection|Y
sentiment/bert-large-uncased-finetuned|sentiment-bert-f|BERT Large Sentiment|C|0.0001,0.0003|512,256|VSTJ|-|BERT large sentiment|Y
sentiment/distilbert-base-uncased-finetuned|sentiment-f|DistilBERT Sentiment|C|0.0001,0.0001|512,256|VSTJ|-|Fine-tuned sentiment analysis|Y
sentiment/distilbert-base-uncased-finetuned-sst-2|distilbert-sst2|sentiment-small|Hugging Face: DistilBERT SST-2|C|0.0000,0.0000|512,256|S|-|DistilBERT fine-tuned for sentiment|Y
sentiment/roberta-base-openai-detector|roberta-openai|text-classifier|RoBERTa: OpenAI Detector|C|0.0000,0.0000|512,256|S|-|RoBERTa for synthetic text detection|Y
sentiment/xlnet-base-cased-finetuned-imdb|xlnet-imdb|sentiment-xlnet|XLNet: IMDB Sentiment|C|0.0000,0.0000|512,256|S|-|XLNet fine-tuned on IMDB reviews|Y

# =============================================================================
# ARCEE-AI (6 models)
# =============================================================================
arcee-ai/coder-large|coder-large|Arcee AI: Coder Large|C|0.0000,0.0000|32768,8192|-|-|Coder-Large is a 32'B-parameter offspring of Qwen'2.5-Instruct that has been fur|Y
arcee-ai/maestro-reasoning|maestro-reasoning|Arcee AI: Maestro Reasoning|C|0.0000,0.0000|131072,32000|-|-|Maestro Reasoning is Arcee's flagship analysis model: a 32'B-parameter derivativ|Y
arcee-ai/spotlight|spotlight|Arcee AI: Spotlight|C|0.0000,0.0000|131072,65537|V|-|Spotlight is a 7-billion-parameter vision-language model derived from Qwen'2.5-V|Y
arcee-ai/trinity-mini|trinity-mini|Arcee AI: Trinity Mini|C|0.0000,0.0000|131072,131072|JKST|-|Trinity Mini is a 26B-parameter (3B active) sparse mixture-of-experts language m|Y
arcee-ai/trinity-mini:free|trinity-mini:free|Arcee AI: Trinity Mini (free)|C|-|131072,32768|JKST|-|Trinity Mini is a 26B-parameter (3B active) sparse mixture-of-experts language m|Y
arcee-ai/virtuoso-large|virtuoso-large|Arcee AI: Virtuoso Large|C|0.0000,0.0000|131072,64000|T|-|Virtuoso-Large is Arcee's top-tier general-purpose LLM at 72'B parameters, tuned|Y

# =============================================================================
# SCIENCE (4 models)
# =============================================================================
science/bioinformatics-llama|bioinformatics|science-bio|BioInformatics Llama|C|0.0000,0.0000|8192,2048|VSTJ|-|Llama for genomics and biological sequence analysis|Y
science/cheminformatics-bert|cheminformatics|science-chem|Cheminformatics BERT|C|0.0000,0.0000|4096,2048|VSTJ|-|Model for chemical structure and property prediction|Y
science/mathandgpt|mathandgpt|science-math|MathAndGPT: Theorem|C|0.0000,0.0000|8192,2048|VSTJK|-|Specialized for mathematical theorem proving|Y
science/scibertagpt|scibertagpt|science-bert|SciGPT: SciPDF|C|0.0000,0.0000|8192,2048|VSTJ|-|Model for scientific paper understanding|Y

# =============================================================================
# ALIBABA (5 models)
# =============================================================================
alibaba/qwen-max|qwen-max|qwen-pro|Alibaba: Qwen Max|C|0.0000,0.0000|32000,8192|VSTJ|-|Alibaba flagship Qwen Max model with extended reasoning|Y
alibaba/qwen-plus|qwen-plus|qwen-standard|Alibaba: Qwen Plus|C|0.0000,0.0000|32000,4096|VSTJ|-|Balanced Qwen Plus model for production use|Y
alibaba/qwen-turbo|qwen-turbo|qwen-fast|Alibaba: Qwen Turbo|C|0.0000,0.0000|8192,2048|VT|-|Fast and efficient Qwen Turbo variant|Y
alibaba/tongyi-deepresearch-30b-a3b|tongyi-deepresearch-|Tongyi DeepResearch 30B A3B|C|0.0000,0.0000|131072,131072|JKST|-|Tongyi DeepResearch is an agentic large language model developed by Tongyi Lab,|Y
alibaba/tongyi-deepresearch-30b-a3b:free|tongyi-deepresearch-|Tongyi DeepResearch 30B A3B (free)|C|-|131072,131072|JKST|-|Tongyi DeepResearch is an agentic large language model developed by Tongyi Lab,|Y

# =============================================================================
# MILESTONE (11 models)
# =============================================================================
milestone/compliance-monitor|compliance-m|ModelSuite: Compliance Monitor|C|0.0006,0.0018|32000,2048|VSTJ|-|Regulatory compliance enforcer|Y
milestone/cost-efficiency-engine|cost-engine-m|ModelSuite: Cost Efficiency Engine|C|0.0005,0.0015|32000,2048|VSTJ|-|Optimal cost-performance selection|Y
milestone/ecosystem-connector|ecosystem-m|ModelSuite: Ecosystem Connector|C|0.0008,0.0024|64000,2048|VSTJK|-|Universal AI ecosystem integration|Y
milestone/expert-routing-system|expert-routing-m|ModelSuite: Expert Routing System|C|0.0030,0.0090|128000,4096|VSTJKC|-|Intelligent model selection engine|Y
milestone/knowledge-synthesis|knowledge-synthesis-m|ModelSuite: Knowledge Synthesis|C|0.0020,0.0060|100000,4096|VSTJ|-|Cross-model knowledge synthesis|Y
milestone/latency-reducer|latency-m|ModelSuite: Latency Reducer|C|0.0003,0.0009|16000,1024|VST|-|Ultra-low latency optimization|Y
milestone/modelsuite-comprehensive-registry|registry-final|ModelSuite: Comprehensive Registry 1200+|C|0,0|1000000,32000|VSTJSKC|-|Complete 1200+ unified model registry|Y
milestone/performance-optimizer|perf-optim-m|ModelSuite: Performance Optimizer|C|0.0010,0.0030|64000,2048|VSTJ|-|Model performance optimization|Y
milestone/reliability-orchestrator|reliability-m|ModelSuite: Reliability Orchestrator|C|0.0002,0.0006|8192,512|VSTJ|-|High-reliability model orchestration|Y
milestone/security-guardian|security-m|ModelSuite: Security Guardian|C|0.0004,0.0012|16000,1024|VSTJ|-|Security-focused model wrapper|Y
milestone/universal-agent|universal-agent-m|ModelSuite: Universal Agent|C|0.0050,0.01|200000,8192|VSTJSKC|-|Unified multi-modal agent|Y

# =============================================================================
# CODE (12 models)
# =============================================================================
code/codellama-34b|codellama-34b|codellama-large|Meta: Code Llama 34B|C|0.0000,0.0000|16384,4096|JT|-|Code Llama 34B for advanced programming|Y
code/codellama-70b|codellama-70b|codellama-xl|Meta: Code Llama 70B|C|0.0000,0.0000|16384,4096|JT|-|Code Llama 70B with extended capabilities|Y
code/codestral|codestral-f|Mistral: Codestral|C|0.0005,0.0015|32768,2048|VSTJK|-|Mistral Codestral code model|Y
code/deepseek-coder-7b|deepseek-coder-7b-f|DeepSeek: Coder 7B|C|0.0001,0.0001|4096,2048|VSTJK|-|DeepSeek Code 7B|Y
code/deepseek-coder-97b|deepseek-coder-97b-f|DeepSeek: Coder 97B|C|0.0009,0.0009|4096,2048|VSTJK|-|DeepSeek Code 97B|Y
code/granite-code-20b|granite-20b-f|IBM: Granite Code 20B|C|0.0003,0.0003|4096,1024|VSTJK|-|Granite Code 20B|Y
code/granite-code-3b|granite-3b-f|IBM: Granite Code 3B|C|0.0001,0.0001|2048,1024|VSTJK|-|Granite Code 3B lightweight|Y
code/phi-1-code|phi-1-code|phi-code|Microsoft: Phi 1 Code|C|0.0000,0.0000|2048,1024|JT|-|Phi 1 1.3B specialized for coding|Y
code/replit-code-v1.5|replit-code-f|Replit: Code v1.5|C|0.0002,0.0002|4096,1024|VSTJK|-|Replit Code generation|Y
code/starcoder|starcoder|star-coder|BigCode: StarCoder|C|0.0000,0.0000|8192,2048|JT|-|StarCoder 15B for code generation|Y
code/starcoder2|starcoder2|star-coder-2|BigCode: StarCoder2|C|0.0000,0.0000|16384,4096|JT|-|StarCoder2 15B improved code model|Y
code/wizardcoder-15b|wizardcoder|wizard-code|WizardCoder: 15B|C|0.0000,0.0000|4096,2048|JT|-|WizardCoder 15B for complex code tasks|Y

# =============================================================================
# VERTEX (2 models)
# =============================================================================
vertex/claude-3.5-sonnet-vision|vertex-claude-vision|GCP: Claude 3.5 Sonnet Vision|C|0.0030,0.01|200000,4096|VSTJKC|-|Claude 3.5 Sonnet via Google Cloud Vertex AI|Y
vertex/gemini-2.5-vision-exp|vertex-gemini-vision-exp|GCP: Gemini 2.5 Vision Exp|C|0.0000,0.0000|1000000,8192|VSTJK|-|Experimental Gemini 2.5 vision via Vertex|Y

# =============================================================================
# ALLENAI (5 models)
# =============================================================================
allenai/olmo-2-0325-32b-instruct|olmo-2-0325-32b-inst|AllenAI: Olmo 2 32B Instruct|C|0.0000,0.0000|128000,32000|-|-|OLMo-2 32B Instruct is a supervised instruction-finetuned variant of the OLMo-2|Y
allenai/olmo-3-32b-think:free|olmo-3-32b-think:fre|AllenAI: Olmo 3 32B Think (free)|C|-|65536,65536|JKS|-|Olmo 3 32B Think is a large-scale, 32-billion-parameter model purpose-built for|N
allenai/olmo-3-7b-instruct|olmo-3-7b-instruct|AllenAI: Olmo 3 7B Instruct|C|0.0000,0.0000|65536,65536|JST|-|Olmo 3 7B Instruct is a supervised instruction-fine-tuned variant of the Olmo 3|Y
allenai/olmo-3-7b-think|olmo-3-7b-think|AllenAI: Olmo 3 7B Think|C|0.0000,0.0000|65536,65536|JKS|-|Olmo 3 7B Think is a research-oriented language model in the Olmo family designe|N
allenai/olmo-3.1-32b-think:free|olmo-3.1-32b-think:f|AllenAI: Olmo 3.1 32B Think (free)|C|-|65536,65536|JKS|-|Olmo 3.1 32B Think is a large-scale, 32-billion-parameter model designed for dee|N

# =============================================================================
# VISION3D (2 models)
# =============================================================================
vision3d/pointnet++-segmentation|pointnet++|3d-segment|PointNet++ Segmentation|C|0.0000,0.0000|1024,256|V|-|PointNet++ for 3D segmentation|Y
vision3d/pointnet-classification|pointnet|3d-classify|PointNet Classification|C|0.0000,0.0000|1024,256|V|-|PointNet for 3D point cloud classification|Y

# =============================================================================
# BAIDU (10 models)
# =============================================================================
baidu/ernie-4.5-21b-a3b|ernie-4.5-21b-a3b|Baidu: ERNIE 4.5 21B A3B|C|0.0000,0.0000|120000,8000|T|-|A sophisticated text-based Mixture-of-Experts (MoE) model featuring 21B total pa|Y
baidu/ernie-4.5-21b-a3b-thinking|ernie-4.5-21b-a3b-th|Baidu: ERNIE 4.5 21B A3B Thinking|C|0.0000,0.0000|131072,65536|K|-|ERNIE-4.5-21B-A3B-Thinking is Baidu's upgraded lightweight MoE model, refined to|N
baidu/ernie-4.5-300b-a47b|ernie-4.5-300b-a47b|Baidu: ERNIE 4.5 300B A47B|C|0.0000,0.0000|123000,12000|JS|-|ERNIE-4.5-300B-A47B is a 300B parameter Mixture-of-Experts (MoE) language model|Y
baidu/ernie-4.5-vl-28b-a3b|ernie-4.5-vl-28b-a3b|Baidu: ERNIE 4.5 VL 28B A3B|C|0.0000,0.0000|30000,8000|KTV|-|A powerful multimodal Mixture-of-Experts chat model featuring 28B total paramete|N
baidu/ernie-4.5-vl-424b-a47b|ernie-4.5-vl-424b-a4|Baidu: ERNIE 4.5 VL 424B A47B|C|0.0000,0.0000|123000,16000|KV|-|ERNIE-4.5-VL-424B-A47B is a multimodal Mixture-of-Experts (MoE) model from Baidu|N
baidu/ernie-4v-32k|ernie-4v-32k|Baidu: ERNIE 4.0 Vision 32K|C|0.0000,0.0000|32000,4096|VST|-|Extended context Baidu vision model|Y
baidu/ernie-4v-8k|ernie-4v-8k|Baidu: ERNIE 4.0 Vision|C|0.0000,0.0000|8192,2048|VST|-|Baidu multimodal model for Chinese and English vision tasks|Y
baidu/ernie-bot-3.5|ernie-3.5|ernie-turbo|Baidu: ERNIE Bot 3.5|C|0.0000,0.0000|8192,2048|VSTJ|-|Baidu ERNIE Bot 3.5 Turbo for faster inference|Y
baidu/ernie-bot-4|ernie-4|ernie-bot-4|Baidu: ERNIE Bot 4|C|0.0000,0.0000|8192,2048|VSTJ|-|Baidu ERNIE Bot 4 with strong Chinese understanding|Y
baidu/ernie-bot-8k|ernie-8k|ernie-extended|Baidu: ERNIE Bot 8K|C|0.0000,0.0000|8192,2048|VSTJ|-|Extended context ERNIE Bot for long documents|Y

# =============================================================================
# BYTEDANCE (1 models)
# =============================================================================
bytedance/ui-tars-1.5-7b|ui-tars-1.5-7b|ByteDance: UI-TARS 7B|C|0.0000,0.0000|128000,2048|V|-|UI-TARS-1.5 is a multimodal vision-language agent optimized for GUI-based enviro|Y

# =============================================================================
# HOSPITALITY (2 models)
# =============================================================================
hospitality/bookingbot|bookingbot|hospitality-book|BookingBot: Reservations|C|0.0000,0.0000|8192,2048|VSTJ|-|Hotel and travel booking optimization|Y
hospitality/concierge|concierge|hospitality-concierge|ConciergeGPT: Service|C|0.0000,0.0000|8192,2048|VSTJ|-|Personalized concierge and travel recommendations|Y

# =============================================================================
# ARLIAI (1 models)
# =============================================================================
arliai/qwq-32b-arliai-rpr-v1|qwq-32b-arliai-rpr-v|ArliAI: QwQ 32B RpR v1|C|0.0000,0.0000|32768,32768|JKS|-|QwQ-32B-ArliAI-RpR-v1 is a 32B parameter model fine-tuned from Qwen/QwQ-32B usin|N

# =============================================================================
# MOONSHOTAI (6 models)
# =============================================================================
moonshotai/kimi-dev-72b|kimi-dev-72b|MoonshotAI: Kimi Dev 72B|C|0.0000,0.0000|131072,131072|JKS|-|Kimi-Dev-72B is an open-source large language model fine-tuned for software engi|N
moonshotai/kimi-k2|kimi-k2|MoonshotAI: Kimi K2 0711|C|0.0000,0.0000|131072,131072|JST|-|Kimi K2 Instruct is a large-scale Mixture-of-Experts (MoE) language model develo|Y
moonshotai/kimi-k2-0905|kimi-k2-0905|MoonshotAI: Kimi K2 0905|C|0.0000,0.0000|262144,262144|JST|-|Kimi K2 0905 is the September update of [Kimi K2 0711](moonshotai/kimi-k2). It i|Y
moonshotai/kimi-k2-0905:exacto|kimi-k2-0905:exacto|MoonshotAI: Kimi K2 0905 (exacto)|C|0.0000,0.0000|262144,65536|JST|-|Kimi K2 0905 is the September update of [Kimi K2 0711](moonshotai/kimi-k2). It i|Y
moonshotai/kimi-k2-thinking|kimi-k2-thinking|MoonshotAI: Kimi K2 Thinking|C|0.0000,0.0000|262144,65535|JKST|-|Kimi K2 Thinking is Moonshot AI's most advanced open reasoning model to date, ex|Y
moonshotai/kimi-k2:free|kimi-k2:free|MoonshotAI: Kimi K2 0711 (free)|C|-|32768,8192|-|-|Kimi K2 Instruct is a large-scale Mixture-of-Experts (MoE) language model develo|Y

# =============================================================================
# TNGTECH (6 models)
# =============================================================================
tngtech/deepseek-r1t-chimera|deepseek-r1t-chimera|TNG: DeepSeek R1T Chimera|C|0.0000,0.0000|163840,163840|JKS|-|DeepSeek-R1T-Chimera is created by merging DeepSeek-R1 and DeepSeek-V3 (0324), c|N
tngtech/deepseek-r1t-chimera:free|deepseek-r1t-chimera|TNG: DeepSeek R1T Chimera (free)|C|-|163840,40960|K|-|DeepSeek-R1T-Chimera is created by merging DeepSeek-R1 and DeepSeek-V3 (0324), c|N
tngtech/deepseek-r1t2-chimera|deepseek-r1t2-chimer|TNG: DeepSeek R1T2 Chimera|C|0.0000,0.0000|163840,163840|JKST|-|DeepSeek-TNG-R1T2-Chimera is the second-generation Chimera model from TNG Tech.|Y
tngtech/deepseek-r1t2-chimera:free|deepseek-r1t2-chimer|TNG: DeepSeek R1T2 Chimera (free)|C|-|163840,40960|K|-|DeepSeek-TNG-R1T2-Chimera is the second-generation Chimera model from TNG Tech.|N
tngtech/tng-r1t-chimera|tng-r1t-chimera|TNG: R1T Chimera|C|0.0000,0.0000|163840,65536|JKST|-|TNG-R1T-Chimera is an experimental LLM with a faible for creative storytelling a|Y
tngtech/tng-r1t-chimera:free|tng-r1t-chimera:free|TNG: R1T Chimera (free)|C|-|163840,163840|JKST|-|TNG-R1T-Chimera is an experimental LLM with a faible for creative storytelling a|Y

# =============================================================================
# RAG (2 models)
# =============================================================================
rag/bge-large-en|bge-large|BAAI: BGE-Large English|C|0.0000,0.0000|512,512|VSTJ|-|BGE-Large for semantic search|Y
rag/e5-large|e5-large|Alibaba: E5-Large|C|0.0000,0.0000|512,512|VSTJ|-|E5-Large for dense passage retrieval|Y

# =============================================================================
# CALIBRATION (1 models)
# =============================================================================
calibration/temperature-scaling|temp-scale-f|Temperature Scaling|C|0.0000,0.0000|512,512|VST|-|Confidence calibration|Y

# =============================================================================
# TELECOM (2 models)
# =============================================================================
telecom/customergpt|customergpt|telecom-customer|CustomerGPT: Telecom|C|0.0000,0.0000|8192,2048|VSTJ|-|Telecom customer experience optimization|Y
telecom/networkgpt|networkgpt|telecom-network|NetworkGPT: Optimization|C|0.0000,0.0000|8192,2048|VSTJ|-|Network optimization and performance management|Y

# =============================================================================
# ANTHRACITE-ORG (1 models)
# =============================================================================
anthracite-org/magnum-v4-72b|magnum-v4-72b|Magnum v4 72B|C|0.0000,0.0000|16384,2048|J|-|This is a series of models designed to replicate the prose quality of the Claude|Y

# =============================================================================
# HR (3 models)
# =============================================================================
hr/hranalytics|hranalytics|hr-analytics|HRAnalytics: Insights|C|0.0000,0.0000|8192,2048|VSTJ|-|HR analytics and employee engagement prediction|Y
hr/recruiterbot|recruiterbot|hr-recruit|RecruiterBot: Hiring|C|0.0000,0.0000|8192,2048|VSTJ|-|Resume screening and candidate ranking|Y
hr/traininggpt|traininggpt|hr-training|TrainingGPT: Learning|C|0.0000,0.0000|8192,2048|VSTJ|-|Personalized training and skill development|Y

# =============================================================================
# DISTILL (1 models)
# =============================================================================
distill/student-teacher-model|distill-f|Student-Teacher Model|C|0.0001,0.0003|512,512|VSTJ|-|Knowledge distillation pair|Y

# =============================================================================
# Z-AI (9 models)
# =============================================================================
z-ai/glm-4-32b|glm-4-32b|Z.AI: GLM 4 32B|C|0.0000,0.0000|128000,32000|T|-|GLM 4 32B is a cost-effective foundation language model.  It can efficiently per|Y
z-ai/glm-4.5|glm-4.5|Z.AI: GLM 4.5|C|0.0000,0.0000|131072,65536|JKST|-|GLM-4.5 is our latest flagship foundation model, purpose-built for agent-based a|Y
z-ai/glm-4.5-air|glm-4.5-air|Z.AI: GLM 4.5 Air|C|0.0000,0.0000|131072,98304|JKST|-|GLM-4.5-Air is the lightweight variant of our latest flagship model family, also|Y
z-ai/glm-4.5-air:free|glm-4.5-air:free|Z.AI: GLM 4.5 Air (free)|C|-|131072,131072|JKST|-|GLM-4.5-Air is the lightweight variant of our latest flagship model family, also|Y
z-ai/glm-4.5v|glm-4.5v|Z.AI: GLM 4.5V|C|0.0000,0.0000|65536,16384|JKSTV|-|GLM-4.5V is a vision-language foundation model for multimodal agent applications|Y
z-ai/glm-4.6|glm-4.6|Z.AI: GLM 4.6|C|0.0000,0.0000|202752,65536|JKST|-|Compared with GLM-4.5, this generation brings several key improvements:  Longer|Y
z-ai/glm-4.6:exacto|glm-4.6:exacto|Z.AI: GLM 4.6 (exacto)|C|0.0000,0.0000|204800,131072|JKST|-|Compared with GLM-4.5, this generation brings several key improvements:  Longer|Y
z-ai/glm-4.6v|glm-4.6v|Z.AI: GLM 4.6V|C|0.0000,0.0000|131072,24000|JKSTV|-|GLM-4.6V is a large multimodal model designed for high-fidelity visual understan|Y
z-ai/glm-4.7|glm-4.7|Z.AI: GLM 4.7|C|0.0000,0.0000|202752,65535|JKST|-|GLM-4.7 is Z.AI's latest flagship model, featuring upgrades in two key areas: en|Y

# =============================================================================
# REALESTATE (2 models)
# =============================================================================
realestate/investgpt|investgpt|reale-invest|InvestGPT: Real Estate|C|0.0000,0.0000|8192,2048|VSTJ|-|Real estate investment analysis and portfolio optimization|Y
realestate/propertygpt|propertygpt|reale-property|PropertyGPT: Valuation|C|0.0000,0.0000|8192,2048|VSTJ|-|Property valuation and market analysis|Y

# =============================================================================
# COGVLM (1 models)
# =============================================================================
cogvlm/cogvlm2-9b-instruct|cogvlm2-9b|Hugging Face: CogVLM2 9B|C|0.0000,0.0000|8192,2048|VST|-|Chinese-optimized vision language model|Y

# =============================================================================
# PRIME-INTELLECT (1 models)
# =============================================================================
prime-intellect/intellect-3|intellect-3|Prime Intellect: INTELLECT-3|C|0.0000,0.0000|131072,131072|JKST|-|INTELLECT-3 is a 106B-parameter Mixture-of-Experts model (12B active) post-train|Y

# =============================================================================
# SUMMARIZATION (3 models)
# =============================================================================
summarization/bart-large-cnn|bart-cnn|summarize-bart|Facebook: BART CNN|C|0.0000,0.0000|1024,512|T|-|BART large for abstractive summarization|Y
summarization/pegasus-cnn-dailymail|pegasus-cnn|summarize-news|Google: Pegasus CNN/DailyMail|C|0.0000,0.0000|1024,512|T|-|PEGASUS fine-tuned for news summarization|Y
summarization/pegasus-pubmed|pegasus-pubmed|summarize-medical|Google: Pegasus PubMed|C|0.0000,0.0000|1024,512|T|-|PEGASUS for medical paper summarization|Y

# =============================================================================
# OPEN (7 models)
# =============================================================================
open/codeqwen-7b|codeqwen-7b|Alibaba: CodeQwen 7B|C|0.0001,0.0001|8192,2048|VSTJK|-|Qwen specialized for code generation|Y
open/deepseek-coder-33b|deepseek-coder-33b|DeepSeek: Coder 33B|C|0.0008,0.0008|4096,2048|VSTJK|-|DeepSeek Code model 33B|Y
open/llava-1.6-34b|llava-34b|NousResearch: LLaVA 1.6 34B|C|0.0008,0.0008|4096,2048|VSTJK|-|LLaVA 1.6 34B multimodal|Y
open/lol-gpt3-175b-instruct|lol-gpt3-175b|OpenAI: LOL-GPT3 175B|C|0.0010,0.0020|2048,2048|VSTJ|-|GPT3 style model 175B|Y
open/mathstral-7b|mathstral-7b|Mistral: Mathstral 7B|C|0.0001,0.0001|8192,2048|VSTJ|-|Mistral specialized for mathematics|Y
open/qwen-vl-plus|qwen-vl-plus|Alibaba: Qwen VL Plus|C|0.0003,0.0003|32768,2048|VSTJK|-|Qwen Vision-Language Plus|Y
open/starcoder2-15b|starcoder2-15b|BigCode: StarCoder2 15B|C|0.0003,0.0003|16384,2048|VSTJK|-|Next generation code model|Y

# =============================================================================
# VASTAI (1 models)
# =============================================================================
vastai/neural-chat-7b|neural-chat-7b-v|Intel: Neural Chat 7B (Vast)|C|0.0001,0.0002|8192,2048|VSTJ|-|Intel Neural Chat via Vast AI|Y

# =============================================================================
# EMBEDDING (13 models)
# =============================================================================
embedding/all-minilm-l6-v2|minilm-l6|minilm-small|Sentence Transformers: MiniLM L6 v2|C|0.0000,0.0000|512,256|VS|-|MiniLM L6 v2 lightweight embeddings|Y
embedding/bge-base-en|bge-base|bge-small|BAAI: BGE Base EN|C|0.0000,0.0000|512,256|VS|-|BAAI BGE Base for English embeddings|Y
embedding/bge-large-en|bge-large|bge-large-en|BAAI: BGE Large EN|C|0.0000,0.0000|512,256|VS|-|BAAI BGE Large English embeddings|Y
embedding/bge-large-zh|bge-large-zh-f|BAAI: BGE Large ZH|C|0.0000,0.0000|512,1024|VSTJ|-|BGE Chinese embeddings|Y
embedding/bge-m3|bge-m3|bge-multilingual|BAAI: BGE-M3|C|0.0000,0.0000|8192,512|VS|-|BAAI BGE-M3 for 100+ languages|Y
embedding/e5-base|e5-base|e5-small|Hugging Face: E5 Base|C|0.0000,0.0000|512,256|VS|-|E5 Base for semantic search|Y
embedding/e5-large|e5-large|e5-large-v2|Hugging Face: E5 Large|C|0.0000,0.0000|512,512|VS|-|E5 Large improved embeddings|Y
embedding/jina-embeddings-v2|jina-embeddings|jina-v2|Jina AI: Embeddings v2|C|0.0000,0.0000|8192,768|VS|-|Jina Embeddings v2 for long context|Y
embedding/jina-embeddings-v3|jina-v3-f|Jina: Embeddings v3|C|0.0000,0.0000|8192,1536|VSTJ|-|Jina embeddings v3 large|Y
embedding/multilingual-e5-large|multilingual-e5-f|E5: Multilingual Large|C|0.0000,0.0000|512,1024|VSTJ|-|E5 multilingual embeddings|Y
embedding/sentence-transformers-base|st-base|sentence-transformer|Sentence Transformers: Base|C|0.0000,0.0000|512,256|VS|-|Sentence Transformers Base model|Y
embedding/text-embedding-3-large|ada-large-f|OpenAI: Embedding 3 Large|C|0.0001,0.0013|8191,3072|VSTJ|-|OpenAI text embedding large|Y
embedding/text-embedding-3-small|ada-small-f|OpenAI: Embedding 3 Small|C|0.0000,0.0002|8191,1536|VSTJ|-|OpenAI text embedding small|Y

# =============================================================================
# # BETA (1 models)
# =============================================================================
# BETA/PREVIEW MODELS|||C|-|0,0|-|-||Y

# =============================================================================
# NEX-AGI (1 models)
# =============================================================================
nex-agi/deepseek-v3.1-nex-n1:free|deepseek-v3.1-nex-n1|Nex AGI: DeepSeek V3.1 Nex N1 (free)|C|-|131072,163840|JST|-|DeepSeek V3.1 Nex-N1 is the flagship release of the Nex-N1 series - a post-train|Y

# =============================================================================
# AFRICA (2 models)
# =============================================================================
africa/afriberta-base|afriberta|afriberta-model|AfriCLIP: AfriBERTa Base|C|0.0000,0.0000|512,256|S|-|AfriBERTa for African language NLP|Y
africa/naija-bert|naija-bert|nigerian-bert|NaijaBERT: Nigerian|C|0.0000,0.0000|512,256|S|-|NaijaBERT for Nigerian Pidgin and English|Y

# =============================================================================
# # TIME SERIES  (1 models)
# =============================================================================
# TIME SERIES / FORECASTING|||C|-|0,0|-|-||Y

# =============================================================================
# LLAMA (2 models)
# =============================================================================
llama/llama-3.2-11b-vision-instruct|llama-3.2-11b-vision|Meta: Llama 3.2 11B Vision|C|0.0000,0.0000|8192,4096|VST|-|Compact Llama 3.2 vision model optimized for edge deployment|Y
llama/llama-3.2-90b-vision-instruct|llama-3.2-90b-vision|Meta: Llama 3.2 90B Vision|C|0.0000,0.0000|8192,4096|VST|-|Llama 3.2 90B with native vision understanding for images and charts|Y

# =============================================================================
# INSTRUCT (5 models)
# =============================================================================
instruct/llama-3-8b-instruct-multilingual|llama-3-8b-inst|llama-3-inst|Meta: Llama 3 8B Instruct Multilingual|C|0.0000,0.0000|8192,2048|VSTJ|-|Llama 3 8B instruction-tuned for multilingual use|Y
instruct/mistral-7b-instruct-v0.2-toolcalling|mistral-tools|mistral-tool|Mistral: 7B Instruct Tool-Calling|C|0.0000,0.0000|32768,4096|VSTJ|-|Mistral 7B instruction-tuned for tool calling|Y
instruct/neural-chat-7b-v3.3|neural-chat-v33-i|Intel: Neural Chat 7B v3.3|C|0.0001,0.0002|8192,2048|VSTJ|-|Intel Neural Chat v3.3|Y
instruct/openchat-3.5|openchat-3.5-i|OpenChat: 3.5 Instruct|C|0.0001,0.0001|8192,2048|VSTJ|-|OpenChat 3.5 instruction tuned|Y
instruct/qwen-7b-instruct-specialized|qwen-7b-inst|qwen-inst|Alibaba: Qwen 7B Instruct Specialized|C|0.0000,0.0000|4096,2048|VSTJ|-|Qwen 7B instruction-tuned for specialized domains|Y

# =============================================================================
# IBM-GRANITE (1 models)
# =============================================================================
ibm-granite/granite-4.0-h-micro|granite-4.0-h-micro|IBM: Granite 4.0 Micro|C|0.0000,0.0000|131000,32750|-|-|Granite-4.0-H-Micro is a 3B parameter from the Granite 4 family of models. These|Y

# =============================================================================
# LATINAMERICAN (2 models)
# =============================================================================
latinamerican/latamgpt-base|latamgpt|latam-base|LatamGPT: Base|C|0.0000,0.0000|2048,1024|VT|-|LatamGPT Base for efficient Latin American use|Y
latinamerican/latamgpt-large|latamgpt-large|latam-large|LatamGPT: Large|C|0.0000,0.0000|4096,2048|VSTJ|-|LatamGPT Large optimized for Spanish and Portuguese|Y

# =============================================================================
# PROVIDER (8 models)
# =============================================================================
provider/ai21-jamba|jamba-f|AI21: Jamba|C|0.0005,0.0015|8192,2048|VSTJ|-|Jamba hybrid SSM-Transformer|Y
provider/ai21-jamba-instruct|jamba-instruct-f|AI21: Jamba Instruct|C|0.0005,0.0015|8192,2048|VSTJ|-|Jamba instruction tuned|Y
provider/cohere-command-r|cohere-r-f|Cohere: Command R|C|0.0005,0.0015|4096,2048|VSTJ|-|Cohere Command R optimized model|Y
provider/cohere-command-r-plus|cohere-r-plus-f|Cohere: Command R Plus|C|0.0030,0.01|4096,2048|VSTJ|-|Cohere Command R Plus extended|Y
provider/playai-tts|playai-tts|Play.ai: TTS Model|C|0.0000,0.0000|1000,1|VT|-|Play.ai ultra-realistic text-to-speech|Y
provider/stability-stablediffusion3|stable-diffusion-3-f|Stability: Stable Diffusion 3|C|0.0001,0.0004|1024,1024|VT|-|Latent diffusion 3 generation|Y
provider/writer-palmyra|palmyra-f|Writer: Palmyra|C|0.0003,0.0009|4096,2048|VSTJ|-|Palmyra open model|Y
provider/xi-api-nova|xi-api-nova|ElevenLabs: Nova Speech|C|0.0000,0.0001|1000,1|VT|-|ElevenLabs Nova speech synthesis|Y

# =============================================================================
# SWITCHPOINT (1 models)
# =============================================================================
switchpoint/router|router|Switchpoint Router|C|0.0000,0.0000|131072,32768|K|-|Switchpoint AI's router instantly analyzes your request and directs it to the op|N

# =============================================================================
# EDGE (7 models)
# =============================================================================
edge/distilbert-base|distilbert-f|DistilBERT Base|C|0.0001,0.0001|512,512|VSTJ|-|Lightweight BERT 40% smaller|Y
edge/llm-mini|llm-mini-f|ModelSuite: LLM Mini|C|0.0000,0.0000|2048,256|VST|-|Mini LLM for edge devices|Y
edge/mobilebert|mobilebert-f|MobileBERT|C|0.0000,0.0001|512,512|VSTJ|-|BERT optimized for mobile|Y
edge/mobilelm-500m|mobilelm-500m-e|MobileLM: 500M|C|0.0000,0.0000|2048,256|VST|-|Ultra-lightweight mobile LLM|Y
edge/phi-2-small|phi-2-small-e|Microsoft: Phi-2 Small|C|0.0001,0.0001|2048,512|VST|-|Phi-2 2.7B model|Y
edge/qwen-1.5-0.5b|qwen-0.5b-f|Alibaba: Qwen1.5 0.5B|C|0.0000,0.0000|32768,128|VST|-|Qwen ultra-small 0.5B|Y
edge/tinylm-1.3b|tinylm-f|TinyLM 1.3B|C|0.0001,0.0001|512,512|VSTJ|-|TinyLM tiny language model|Y

# =============================================================================
# AUTOMOTIVE (2 models)
# =============================================================================
automotive/autonomousgpt|autonomousgpt|auto-av|AutonomousGPT: Driving|C|0.0000,0.0000|8192,2048|VSTJK|-|Autonomous vehicle decision making and planning|Y
automotive/diagnosticsgpt|diagnosticsgpt|auto-diag|DiagnosticsGPT: Vehicles|C|0.0000,0.0000|8192,2048|VSTJ|-|Vehicle diagnostics and maintenance prediction|Y

# =============================================================================
# LONGCONTEXT (3 models)
# =============================================================================
longcontext/claude-opus-4-200k|claude-opus-200k-lc|Anthropic: Claude Opus 200K Context|C|5.0,25.0,0.50|200000,32000|VSTJKC|-|Claude Opus with 200K context window|Y
longcontext/gemini-1.5-pro-1m|gemini-1m-lc|Google: Gemini 1.5 Pro 1M|C|0.0075,0.03|1000000,8192|VSTJK|-|Gemini 1.5 Pro with 1M context|Y
longcontext/gpt-4-turbo-128k|gpt-4-128k-lc|OpenAI: GPT-4 Turbo 128K|C|0.0010,0.0030,0.0005|128000,4096|VSTJK|-|GPT-4 Turbo with 128K context|Y

# =============================================================================
# TOT (1 models)
# =============================================================================
tot/tot-planner|tot-f|Tree-of-Thoughts Planner|C|0.0005,0.0015|4096,2048|VSTJK|-|Tree-of-thoughts planning|Y

# =============================================================================
# COGNITIVECOMPUTATIONS (1 models)
# =============================================================================
cognitivecomputations/dolphin-mistral-24b-venice-edition:free|dolphin-mistral-24b-|Venice: Uncensored (free)|C|-|32768,8192|JS|-|Venice Uncensored Dolphin Mistral 24B Venice Edition is a fine-tuned variant of|Y

# =============================================================================
# ALIGNED (2 models)
# =============================================================================
aligned/llama-3-70b-preference-aligned|llama-3-aligned|llama-aligned|Meta: Llama 3 70B Preference Aligned|C|0.0000,0.0000|8192,2048|VSTJK|-|Llama 3 70B aligned to human preferences with RLHF|Y
aligned/mistral-large-preference-aligned|mistral-aligned|mistral-rlhf|Mistral: Large Preference Aligned|C|1.5,4.5|128000,8192|VSTJ|-|Mistral Large fine-tuned with RLHF preference alignment|Y

# =============================================================================
# STS (1 models)
# =============================================================================
sts/sentence-transformers-msmarco|sts-msmarco-f|STS MSMARCO|C|0.0001,0.0003|512,512|VSTJ|-|MSMARCO semantic similarity|Y

# =============================================================================
# ANALYTICS (3 models)
# =============================================================================
analytics/analyticsgpt|analyticsgpt|analytics-data|AnalyticsGPT: Analytics|C|0.0000,0.0000|8192,2048|VSTJ|-|Advanced analytics and insight generation|Y
analytics/datavizgpt|datavizgpt|analytics-viz|DataVizGPT: Visualization|C|0.0000,0.0000|8192,2048|VSTJ|-|Data visualization and storytelling|Y
analytics/predictgpt|predictgpt|analytics-predict|PredictGPT: Forecasting|C|0.0000,0.0000|8192,2048|VSTJ|-|Time series prediction and trend analysis|Y

# =============================================================================
# STEPFUN-AI (1 models)
# =============================================================================
stepfun-ai/step3|step3|StepFun: Step3|C|0.0000,0.0000|65536,65536|JKSTV|-|Step3 is a cutting-edge multimodal reasoning model-built on a Mixture-of-Experts|Y

# =============================================================================
# MEDICAL (4 models)
# =============================================================================
medical/biobert-base|biobert|biobert-model|BioBERT: Base|C|0.0000,0.0000|512,256|S|-|BioBERT for biomedical text mining|Y
medical/biolinkbert-base|biolinkbert|biolink-bert|BioLinkBERT: Base|C|0.0000,0.0000|512,256|S|-|BioLinkBERT with biomedical entity linking|Y
medical/clinical-bert|clinical-bert|clinbert|ClinicalBERT: MIMIC|C|0.0000,0.0000|512,256|S|-|ClinicalBERT trained on MIMIC clinical notes|Y
medical/pubmedbert|pubmedbert|pubmed-bert|PubMedBERT: Domain|C|0.0000,0.0000|512,256|S|-|PubMedBERT trained on PubMed abstracts|Y

# =============================================================================
# ALFREDPROS (1 models)
# =============================================================================
alfredpros/codellama-7b-instruct-solidity|codellama-7b-instruc|AlfredPros: CodeLLaMa 7B Instruct Solidity|C|0.0000,0.0000|4096,4096|-|-|A finetuned 7 billion parameters Code LLaMA - Instruct model to generate Solidit|Y

# =============================================================================
# REASONING (10 models)
# =============================================================================
reasoning/internlm2-math-20b|internlm-math-f|Shanghai AI: InternLM2 Math|C|0.0005,0.0005|4096,2048|VSTJK|-|InternLM2 specialized for math|Y
reasoning/llama-3-reasoning-70b|llama-3-reasoning|Meta: Llama 3 Reasoning 70B|C|0.0010,0.0010|8192,2048|VSTJK|-|Llama 3 with enhanced reasoning|Y
reasoning/llemma-34b|llemma-34b|llemma-large|Meta: Llemma 34B|C|0.0000,0.0000|4096,4096|JT|-|Llemma 34B for advanced mathematical proofs|Y
reasoning/llemma-7b|llemma-7b|llemma|Meta: Llemma 7B|C|0.0000,0.0000|4096,2048|JT|-|Llemma 7B specialized for mathematical reasoning|Y
reasoning/mathcoder-34b|mathcoder-34b|mathcoder-large|MathCoder: 34B|C|0.0000,0.0000|4096,4096|JT|-|MathCoder 34B for complex mathematics|Y
reasoning/mathcoder-7b|mathcoder-7b|mathcoder|MathCoder: 7B|C|0.0000,0.0000|4096,2048|JT|-|MathCoder 7B for solving math problems|Y
reasoning/mixture-of-agents|moa-reasoning-f|MoA: Mixture of Agents|C|0.0015,0.0045|8192,2048|VSTJK|-|Ensemble of specialized agents|Y
reasoning/phi-3-reasoning|phi-3-reasoning|Microsoft: Phi-3 Reasoning|C|0.0002,0.0002|131072,2048|VSTJK|-|Phi-3 with reasoning capabilities|Y
reasoning/phi-3.5-mini|phi-35-mini-f|Microsoft: Phi-3.5 Mini|C|0.0001,0.0001|131072,1024|VSTJK|-|Phi-3.5 Mini with reasoning|Y
reasoning/phi-3.5-moe|phi-35-moe-f|Microsoft: Phi-3.5 MoE|C|0.0003,0.0003|131072,1024|VSTJK|-|Phi-3.5 Mixture of Experts|Y

# =============================================================================
# CHINA (3 models)
# =============================================================================
china/baichuan2-13b|baichuan2-13b-c|Baichuan: Baichuan2 13B|C|0.0004,0.0004|4096,2048|VSTJ|-|Baichuan 13B Chinese optimized|Y
china/internlm2-20b|internlm2-20b-c|Shanghai AI Lab: InternLM2 20B|C|0.0005,0.0005|4096,2048|VSTJ|-|InternLM2 20B multilingual|Y
china/qwen-72b|qwen-72b-c|Alibaba: Qwen 72B|C|0.0008,0.0008|32768,2048|VSTJ|-|Qwen 72B large language model|Y

# =============================================================================
# GRYPHE (1 models)
# =============================================================================
gryphe/mythomax-l2-13b|mythomax-l2-13b|MythoMax 13B|C|0.0000,0.0000|4096,1024|JS|-|One of the highest performing and most popular fine-tunes of Llama 2 13B, with r|Y

# =============================================================================
# SAO10K (5 models)
# =============================================================================
sao10k/l3-euryale-70b|l3-euryale-70b|Sao10k: Llama 3 Euryale 70B v2.1|C|0.0000,0.0000|8192,8192|T|-|Euryale 70B v2.1 is a model focused on creative roleplay from [Sao10k](https://k|Y
sao10k/l3-lunaris-8b|l3-lunaris-8b|Sao10K: Llama 3 8B Lunaris|C|0.0000,0.0000|8192,2048|JS|-|Lunaris 8B is a versatile generalist and roleplaying model based on Llama 3. It'|Y
sao10k/l3.1-70b-hanami-x1|l3.1-70b-hanami-x1|Sao10K: Llama 3.1 70B Hanami x1|C|0.0000,0.0000|16000,4000|-|-|This is [Sao10K](/sao10k)'s experiment over [Euryale v2.2](/sao10k/l3.1-euryale-|Y
sao10k/l3.1-euryale-70b|l3.1-euryale-70b|Sao10K: Llama 3.1 Euryale 70B v2.2|C|0.0000,0.0000|32768,8192|JST|-|Euryale L3.1 70B v2.2 is a model focused on creative roleplay from [Sao10k](http|Y
sao10k/l3.3-euryale-70b|l3.3-euryale-70b|Sao10K: Llama 3.3 Euryale 70B|C|0.0000,0.0000|131072,16384|JS|-|Euryale L3.3 70B is a model focused on creative roleplay from [Sao10k](https://k|Y

# =============================================================================
# ENERGY (2 models)
# =============================================================================
energy/renewable-optimizer|renewable-opt|energy-renewable|RenewableOptimizer LLM|C|0.0000,0.0000|8192,2048|VSTJ|-|Renewable energy production optimization|Y
energy/smartgridgpt|smartgrid|energy-grid|SmartGridGPT: Analytics|C|0.0000,0.0000|8192,2048|VSTJ|-|Smart grid optimization and forecasting|Y

# =============================================================================
# KOREAN (2 models)
# =============================================================================
korean/koalpaca-13b|koalpaca|koalpaca-korean|KoAlpaca: 13B|C|0.0000,0.0000|4096,2048|T|-|KoAlpaca 13B for Korean instruction tasks|Y
korean/solar-ko-7b|solar-ko|solar-korean|Upstage: Solar 7B Korean|C|0.0000,0.0000|4096,2048|VSTJ|-|Solar 7B optimized for Korean language|Y

# =============================================================================
# DIALOGUE (4 models)
# =============================================================================
dialogue/airoboros-13b|airoboros-13b|airoboros-large|Airoboros: 13B|C|0.0000,0.0000|4096,2048|VSTJ|-|Larger Airoboros 13B variant|Y
dialogue/airoboros-7b|airoboros-7b|airoboros|Airoboros: 7B|C|0.0000,0.0000|4096,2048|VSTJ|-|Airoboros 7B dialogue and instruction model|Y
dialogue/evolutionaryqa-7b|evolutionaryqa|evo-qa|EvolutionaryQA: 7B|C|0.0000,0.0000|4096,2048|VSTJ|-|EvolutionaryQA for question answering|Y
dialogue/neural-chat-8b|neural-chat-8b|neural-chat|Intel: Neural Chat 8B|C|0.0000,0.0000|8192,2048|VSTJ|-|Intel Neural Chat 8B for conversations|Y

# =============================================================================
# RL (3 models)
# =============================================================================
rl/gpt2-medium-policy|gpt2-policy|policy-model|OpenAI: GPT-2 Medium Policy|C|0.0000,0.0000|1024,512|T|-|GPT-2 Medium for policy learning|Y
rl/t5-base-policy|t5-policy|seq2seq-policy|Google: T5 Base Policy|C|0.0000,0.0000|512,512|T|-|T5 Base for sequence-to-sequence tasks|Y
rl/trl-ppo|trl-ppo-f|TRL PPO Model|C|0.0004,0.0012|4096,2048|VSTJ|-|PPO reinforcement learning|Y

# =============================================================================
# AUDIO (37 models)
# =============================================================================
audio/audfprint|audfprint-a|AudFPrint|C|0.0001,0.0002|16000,128|VT|-|Robust audio matching|Y
audio/audioset-classifier|audioset-cls-a|AudioSet Classifier|C|0.0001,0.0002|16000,512|VT|-|Environmental sound classification|Y
audio/clap-base|clap-base|Salesforce: CLAP Base|C|0.0000,0.0000|16000,1|VT|-|CLIP for audio understanding|Y
audio/conformer-large|conformer-large-a|Conformer Large ASR|C|0.0001,0.0002|16000,512|VT|-|Conformer for speech recognition|Y
audio/demucs|demucs-a|Meta: Demucs|C|0.0002,0.0005|16000,16000|VT|-|Source separation SOTA|Y
audio/demucs-base|demucs|audio-separation|Meta: Demucs|C|0.0000,0.0000|448,2048|V|-|Demucs for music source separation|Y
audio/ecapa-tdnn|ecapa-tdnn-a|ECAPA-TDNN Speaker|C|0.0001,0.0003|16000,256|VT|-|Speaker verification SOTA|Y
audio/fastpitch|fastpitch-a|FastPitch TTS|C|0.0001,0.0001|512,16000|VT|-|Fast pitch-based TTS|Y
audio/fsd50k-tagger|fsd50k-tagger-a|FSD50K Tagger|C|0.0001,0.0003|16000,512|VT|-|Freesound Dataset tagging|Y
audio/glow-tts|glow-tts-a|Glow-TTS|C|0.0000,0.0001|512,16000|VT|-|Efficient TTS flow-based model|Y
audio/hubert-xlarge|hubert-xlarge-a|HuBERT XLarge ASR|C|0.0001,0.0002|16000,512|VT|-|Self-supervised HuBERT XL|Y
audio/hybrid-spectrogram|hybrid-spec-a|Hybrid Spectrogram|C|0.0001,0.0004|16000,16000|VT|-|Hybrid time-frequency separation|Y
audio/jukebox|jukebox-a|OpenAI: Jukebox|C|0.0002,0.0008|8192,16000|VT|-|Music generation with lyrics|Y
audio/jukebox-5b|jukebox-5b|music-gen|OpenAI: Jukebox 5B|C|0.0000,0.0000|2048,2048|V|-|Jukebox 5B for music generation|Y
audio/multi-modal-emotion|multimodal-emo-a|Multimodal Emotion|C|0.0001,0.0005|16000,512|VT|-|Audio-visual emotion detection|Y
audio/music-vae|musicvae|music-vae|Google: MusicVAE|C|0.0000,0.0000|512,512|V|-|MusicVAE for music composition|Y
audio/musicbert|musicbert-a|MusicBERT|C|0.0001,0.0003|512,512|VT|-|Music understanding BERT|Y
audio/musicgen|musicgen-a|Meta: MusicGen|C|0.0001,0.0005|1024,16000|VT|-|Controllable music generation|Y
audio/naturalspeech3|naturalspeech3-a|Microsoft: NaturalSpeech3|C|0.0001,0.0003|512,16000|VT|-|Neural audio codec TTS|Y
audio/riffusion|riffusion-a|Riffusion Spectro|C|0.0001,0.0003|512,512|VT|-|Spectrogram diffusion for music|Y
audio/ser-wav2vec|ser-wav2vec-a|SER Wav2Vec2|C|0.0001,0.0003|16000,512|VT|-|Speech emotion recognition|Y
audio/shazam-fingerprint|shazam-fp-a|Shazam Fingerprinting|C|0.0001,0.0001|16000,128|VT|-|Audio fingerprinting MFCC|Y
audio/spleeter|spleeter-a|Deezer: Spleeter|C|0.0001,0.0003|16000,16000|VT|-|Fast source separation|Y
audio/urbansound8k|urbansound8k-a|UrbanSound8K Model|C|0.0001,0.0003|16000,512|VT|-|Urban sound recognition|Y
audio/vits|vits-a|VITS Neural Vocoder|C|0.0001,0.0002|512,16000|VT|-|Variational inference TTS|Y
audio/voicefilter|voicefilter|audio-filter|Google: VoiceFilter|C|0.0000,0.0000|448,2048|V|-|VoiceFilter for speech enhancement|Y
audio/wav2vec2-base|wav2vec2|wav2vec|Facebook: Wav2Vec2 Base|C|0.0000,0.0000|448,2048|V|-|Wav2Vec2 Base for speech representation|Y
audio/wav2vec2-base-finetuned-multilingual-asr|wav2vec-multi|wav2vec-asr-ft|Hugging Face: Wav2Vec2 Multilingual ASR FT|C|0.0000,0.0000|448,2048|V|-|Wav2Vec2 fine-tuned for multilingual speech recognition|Y
audio/wav2vec2-large|wav2vec2-large|wav2vec-large|Facebook: Wav2Vec2 Large|C|0.0000,0.0000|448,2048|V|-|Wav2Vec2 Large for speech tasks|Y
audio/wav2vec2-xlarge|wav2vec2-xlarge-a|Facebook: Wav2Vec2 XL|C|0.0001,0.0001|16000,512|VT|-|Self-supervised speech XL|Y
audio/whisper-base|whisper-base|whisper-std|OpenAI: Whisper Base|C|0,0|448,2048|V|-|Whisper Base for basic transcription|Y
audio/whisper-base-finetuned-medical-terminology|whisper-medical|whisper-med-ft|OpenAI: Whisper Medical FT|C|0.0000,0.0000|448,2048|V|-|Whisper base fine-tuned for medical terminology|Y
audio/whisper-large|whisper-large-a|OpenAI: Whisper Large|C|0.0001,0.0003|16000,512|VT|-|Robust multilingual ASR|Y
audio/whisper-medium|whisper-medium|whisper-acc|OpenAI: Whisper Medium|C|0,0|448,2048|V|-|Whisper Medium for high accuracy|Y
audio/whisper-small|whisper-small|whisper-med|OpenAI: Whisper Small|C|0,0|448,2048|V|-|Whisper Small for improved accuracy|Y
audio/whisper-tiny|whisper-tiny|whisper-small|OpenAI: Whisper Tiny|C|0,0|448,2048|V|-|Whisper Tiny for speech recognition on edge|Y
audio/xvector|xvector-a|X-Vector Speaker|C|0.0001,0.0003|16000,256|VT|-|DNN-based speaker embedding|Y

# =============================================================================
# RESEARCH (2 models)
# =============================================================================
research/allenai-olmo-7b|olmo-7b-r|AllenAI: OLMo 7B|C|0.0001,0.0003|2048,2048|VSTJ|-|Open Language Model from AllenAI|Y
research/stability-stablelm-2|stablelm-2-r|Stability AI: StableLM 2|C|0.0001,0.0001|4096,2048|VSTJ|-|StableLM 2 foundation model|Y

# =============================================================================
# RELACE (2 models)
# =============================================================================
relace/relace-apply-3|relace-apply-3|Relace: Relace Apply 3|C|0.0000,0.0000|256000,128000|-|-|Relace Apply 3 is a specialized code-patching LLM that merges AI-suggested edits|Y
relace/relace-search|relace-search|Relace: Relace Search|C|0.0000,0.0000|256000,128000|T|-|The relace-search model uses 4-12 'view_file' and 'grep' tools in parallel to ex|Y

# =============================================================================
# FOLLOW (2 models)
# =============================================================================
follow/alpaca-7b-instruction|alpaca-7b-inst|alpaca|Stanford: Alpaca 7B Instruction|C|0.0000,0.0000|4096,2048|VSTJ|-|Alpaca 7B fine-tuned for instruction following|Y
follow/vicuna-13b-instruction|vicuna-13b-inst|vicuna|LMSYS: Vicuna 13B Instruction|C|0.0000,0.0000|4096,2048|VSTJ|-|Vicuna 13B instruction-tuned for conversational AI|Y

# =============================================================================
# COT (1 models)
# =============================================================================
cot/cot-generator|cot-f|CoT Generator|C|0.0004,0.0012|4096,2048|VSTJ|-|Chain-of-thought generator|Y

# =============================================================================
# ANOMALY (2 models)
# =============================================================================
anomaly/autoencoder-vae|vae-anomaly|anomaly-vae|Variational Autoencoder|C|0.0000,0.0000|512,256|S|-|VAE for unsupervised anomaly detection|Y
anomaly/isolation-forest-lstm|isolation-lstm|anomaly-detect|IsolationForest LSTM|C|0.0000,0.0000|512,256|S|-|LSTM-based anomaly detection|Y

# =============================================================================
# AION-LABS (3 models)
# =============================================================================
aion-labs/aion-1.0|aion-1.0|AionLabs: Aion-1.0|C|0.0000,0.0000|131072,32768|K|-|Aion-1.0 is a multi-model system designed for high performance across various ta|N
aion-labs/aion-1.0-mini|aion-1.0-mini|AionLabs: Aion-1.0-Mini|C|0.0000,0.0000|131072,32768|K|-|Aion-1.0-Mini 32B parameter model is a distilled version of the DeepSeek-R1 mode|N
aion-labs/aion-rp-llama-3.1-8b|aion-rp-llama-3.1-8b|AionLabs: Aion-RP 1.0 (8B)|C|0.0000,0.0000|32768,32768|-|-|Aion-RP-Llama-3.1-8B ranks the highest in the character evaluation portion of th|Y

# =============================================================================
# ELEUTHERAI (1 models)
# =============================================================================
eleutherai/llemma_7b|llemma_7b|EleutherAI: Llemma 7b|C|0.0000,0.0000|4096,4096|-|-|Llemma 7B is a language model for mathematics. It was initialized with Code Llam|Y

# =============================================================================
# RANKING (2 models)
# =============================================================================
ranking/lambdamart|lambdamart|rank-ltr|LambdaMART|C|0.0000,0.0000|512,256|S|-|Learning-to-rank with LambdaMART|Y
ranking/listwise-ranker|listwise|rank-listwise|Listwise Ranker|C|0.0000,0.0000|512,256|S|-|Listwise learning-to-rank model|Y

# =============================================================================
# GOVERNMENT (2 models)
# =============================================================================
government/citizen-services|citizensvc|gov-citizen|CitizenServices LLM|C|0.0000,0.0000|8192,2048|VSTJ|-|Public sector citizen service improvements|Y
government/compliance-bot|compliance-bot|gov-compliance|ComplianceBot: Regulations|C|0.0000,0.0000|8192,2048|VSTJ|-|Regulatory compliance and policy analysis|Y

# =============================================================================
# # GGML (1 models)
# =============================================================================
# GGML/GGUF VARIANTS FOR LOCAL DEPLOYMENT|||C|-|0,0|-|-||Y

# =============================================================================
# MARKETING (4 models)
# =============================================================================
marketing/copygpt|copygpt|marketing-copy|CopyGPT: Marketing|C|0.0000,0.0000|8192,2048|VSTJ|-|Marketing copy generation and optimization|Y
marketing/newsgpt|newsgpt|marketing-news|NewsGPT: Journalism|C|0.0000,0.0000|8192,2048|VSTJ|-|News article generation and fact-checking|Y
marketing/seo-optimizer|seo-optimizer|marketing-seo|SEOGPT: Optimizer|C|0.0000,0.0000|4096,2048|VSTJ|-|SEO content generation and optimization|Y
marketing/socialmediagpt|socialgpt|marketing-social|SocialMediaGPT: Posts|C|0.0000,0.0000|4096,2048|VSTJ|-|Social media content creation and scheduling|Y

# =============================================================================
# COREFERENCE (1 models)
# =============================================================================
coreference/coref-roberta|coref-roberta|coreference|AllenAI: Coreference RoBERTa|C|0.0000,0.0000|512,256|S|-|RoBERTa for coreference resolution|Y

# =============================================================================
# BYTEDANCE-SEED (2 models)
# =============================================================================
bytedance-seed/seed-1.6|seed-1.6|ByteDance Seed: Seed 1.6|C|0.0000,0.0000|262144,32768|JKSTV|-|Seed 1.6 is a general-purpose model released by the ByteDance Seed team. It inco|Y
bytedance-seed/seed-1.6-flash|seed-1.6-flash|ByteDance Seed: Seed 1.6 Flash|C|0.0000,0.0000|262144,16384|JKSTV|-|Seed 1.6 Flash is an ultra-fast multimodal deep thinking model by ByteDance Seed|Y

# =============================================================================
# TENCENT (1 models)
# =============================================================================
tencent/hunyuan-a13b-instruct|hunyuan-a13b-instruc|Tencent: Hunyuan A13B Instruct|C|0.0000,0.0000|131072,131072|JKS|-|Hunyuan-A13B is a 13B active parameter Mixture-of-Experts (MoE) language model d|N

# =============================================================================
# SEMANTIC (3 models)
# =============================================================================
semantic/paraphrase-MiniLM-L6-v2|paraphrase-minilm|semantic-small|Sentence Transformers: Paraphrase MiniLM|C|0.0000,0.0000|512,256|VS|-|Paraphrase detection with MiniLM|Y
semantic/paraphrase-multilingual-MiniLM-L12-v2|paraphrase-multilingual|semantic-multilingual|Sentence Transformers: Paraphrase Multilingual|C|0.0000,0.0000|512,256|VS|-|Multilingual paraphrase detection|Y
semantic/semantic-search-qa-msmarco-distilbert-base-v4|semantic-search-qa|semantic-qa|Sentence Transformers: Semantic Search QA|C|0.0000,0.0000|512,256|VS|-|Semantic search for question answering|Y

# =============================================================================
# CODING (4 models)
# =============================================================================
coding/debuggpt|debuggpt|coding-debug|DebugGPT: Testing|C|0.0000,0.0000|8192,2048|VTJK|-|Automated debugging and test generation|Y
coding/devgpt|devgpt|coding-dev|DevGPT: Development|C|0.0000,0.0000|8192,2048|VTJK|-|AI pair programmer for full-stack development|Y
coding/docgpt|docgpt|coding-doc|DocGPT: Documentation|C|0.0000,0.0000|4096,2048|VSTJ|-|Code documentation and explanation generation|Y
coding/securegpt|securegpt|coding-security|SecureGPT: Security|C|0.0000,0.0000|8192,2048|VTJK|-|Security vulnerability detection and remediation|Y

# =============================================================================
# RETRIEVAL (2 models)
# =============================================================================
retrieval/bge-base-finetuned-uat-retrieval|bge-retrieval|bge-ret-ft|BAAI: BGE Retrieval FT|C|0.0000,0.0000|512,256|VS|-|BGE fine-tuned for UAT and legal document retrieval|Y
retrieval/e5-large-finetuned-domain-specific|e5-domain|e5-domain-ft|Hugging Face: E5 Domain-Specific FT|C|0.0000,0.0000|512,512|VS|-|E5 Large fine-tuned for domain-specific semantic search|Y

# =============================================================================
# RECOMMENDATION (2 models)
# =============================================================================
recommendation/collaborative-filtering-embedding|collab-embed|recommend-collab|Collaborative Filtering Embedding|C|0.0000,0.0000|512,256|S|-|Collaborative filtering with embeddings|Y
recommendation/neural-collaborative-filtering|ncf|recommend-neural|Neural Collaborative Filtering|C|0.0000,0.0000|512,256|S|-|Neural network for recommendations|Y

# =============================================================================
# UNDI95 (1 models)
# =============================================================================
undi95/remm-slerp-l2-13b|remm-slerp-l2-13b|ReMM SLERP 13B|C|0.0000,0.0000|6144,1536|JS|-|A recreation trial of the original MythoMax-L2-B13 but with updated models. #mer|Y

# =============================================================================
# LOCAL (4 models)
# =============================================================================
local/mistral-7b-ggml-q5|mistral-local-q5|mistral-ggml|Local: Mistral 7B GGML Q5|C|0,0|32768,4096|JT|-|Mistral 7B GGML format for local deployment|Y
local/neural-chat-7b-ggml-q4|neural-local-q4|neural-ggml|Local: Neural Chat GGML Q4|C|0,0|8192,2048|JT|-|Neural Chat GGML format for edge devices|Y
local/openhermes-2.5-ggml-q5|openhermes-local|openhermes-ggml|Local: OpenHermes GGML Q5|C|0,0|4096,2048|JT|-|OpenHermes GGML for local chat deployment|Y
local/tinyllama-1.1b-ggml-q8|tinyllama-local|tinyllama-ggml|Local: TinyLlama GGML Q8|C|0,0|2048,512|JT|-|TinyLlama GGML for ultra-lightweight local inference|Y

# =============================================================================
# DOMAIN (7 models)
# =============================================================================
domain/financial-llama|financial-llama|Meta: Llama Financial|C|0.0006,0.0006|8192,2048|VSTJ|-|Llama fine-tuned for finance|Y
domain/financial-mpnet|financial-mpnet-f|Financial MPNet|C|0.0001,0.0003|384,512|VSTJ|-|MPNet financial domain|Y
domain/legal-llama-2|legal-llama-f|Legal Llama 2|C|0.0006,0.0006|8192,2048|VSTJ|-|Llama 2 fine-tuned for legal|Y
domain/legal-palm-2|legal-palm-2|Google: Legal PaLM 2|C|0.0005,0.0015|8192,2048|VSTJ|-|PaLM 2 fine-tuned for legal|Y
domain/med-gemini-1.5|med-gemini-1.5|Google: Med-Gemini 1.5|C|0.0075,0.03|1000000,8192|VSTJK|-|Gemini 1.5 specialized for medical|Y
domain/medical-falcon-40b|medical-falcon-f|Medical Falcon 40B|C|0.0008,0.0008|2048,2048|VSTJ|-|Falcon 40B medical variant|Y
domain/scientific-bert|scientific-bert-f|SciBERT|C|0.0001,0.0003|512,512|VSTJ|-|SciBERT scientific papers|Y

# =============================================================================
# THEDRUMMER (4 models)
# =============================================================================
thedrummer/cydonia-24b-v4.1|cydonia-24b-v4.1|TheDrummer: Cydonia 24B V4.1|C|0.0000,0.0000|131072,131072|JS|-|Uncensored and creative writing model based on Mistral Small 3.2 24B with good r|Y
thedrummer/rocinante-12b|rocinante-12b|TheDrummer: Rocinante 12B|C|0.0000,0.0000|32768,8192|JST|-|Rocinante 12B is designed for engaging storytelling and rich prose.  Early teste|Y
thedrummer/skyfall-36b-v2|skyfall-36b-v2|TheDrummer: Skyfall 36B V2|C|0.0000,0.0000|32768,32768|-|-|Skyfall 36B v2 is an enhanced iteration of Mistral Small 2501, specifically fine|Y
thedrummer/unslopnemo-12b|unslopnemo-12b|TheDrummer: UnslopNemo 12B|C|0.0000,0.0000|32768,8192|JST|-|UnslopNemo v4.1 is the latest addition from the creator of Rocinante, designed f|Y

# =============================================================================
# SEARCH (3 models)
# =============================================================================
search/nomic-embed-text|nomic-embed-f|Nomic: Embed Text|C|0.0001,0.0001|2048,768|VSTJ|-|Nomic embeddings 768D|Y
search/voyage-code-2|voyage-code-f|Voyage: Code 2|C|0.0001,0.0001|16000,1024|VSTJK|-|Voyage code search|Y
search/voyage-large-2|voyage-large-f|Voyage: Large 2|C|0.0001,0.0001|16000,1024|VSTJ|-|Voyage semantic search large|Y

# =============================================================================
# RETAIL (3 models)
# =============================================================================
retail/fashionai-vision|fashionai|retail-fashion|FashionAI: Vision|C|0.0000,0.0000|8192,2048|VSTJ|-|AI for fashion and apparel analysis|Y
retail/pricegpt|pricegpt|retail-price|PriceGPT: Dynamics|C|0.0000,0.0000|8192,2048|VSTJ|-|Dynamic pricing and revenue optimization|Y
retail/shopgpt|shopgpt|retail-shop|ShopGPT: E-commerce|C|0.0000,0.0000|8192,2048|VSTJ|-|E-commerce model for product recommendations|Y

# =============================================================================
# HEALTHCARE (4 models)
# =============================================================================
healthcare/biogpt|biogpt|healthcare-bio|Microsoft: BioGPT|C|0.0000,0.0000|4096,2048|VSTJ|-|BioGPT for biomedical literature understanding|Y
healthcare/clinical-llama|clinical-llama|healthcare-clinical|Meta: Clinical Llama|C|0.0000,0.0000|8192,2048|VSTJ|-|Llama variant fine-tuned for clinical notes|Y
healthcare/med-gemini|med-gemini|healthcare-med|Google: Med-Gemini|C|0.0000,0.0000|1000000,8192|VSTJ|-|Gemini variant specialized for medical applications|Y
healthcare/scigpt|scigpt|healthcare-sci|SciGPT: Scientific|C|0.0000,0.0000|4096,2048|VSTJ|-|Scientific language model for healthcare research|Y

# =============================================================================
# PARSING (2 models)
# =============================================================================
parsing/constituency-small|constituency-parse|parse-constituency|Stanford: Constituency Small|C|0.0000,0.0000|512,256|S|-|Lightweight constituency parser|Y
parsing/uddaptan-en-ud24-en_ewt-small|ddparser-en|parse-english|UDDaptation: English Parser|C|0.0000,0.0000|512,256|S|-|Small model for dependency parsing|Y

# =============================================================================
# X-AI (8 models)
# =============================================================================
x-ai/grok-3|grok-3|xAI: Grok 3|C|0.0000,0.0000|131072,32768|JST|-|Grok 3 is the latest model from xAI. It's their flagship model that excels at en|Y
x-ai/grok-3-beta|grok-3-beta|xAI: Grok 3 Beta|C|0.0000,0.0000|131072,32768|JT|-|Grok 3 is the latest model from xAI. It's their flagship model that excels at en|Y
x-ai/grok-3-mini|grok-3-mini|xAI: Grok 3 Mini|C|0.0000,0.0000|131072,32768|JKST|-|A lightweight model that thinks before responding. Fast, smart, and great for lo|Y
x-ai/grok-3-mini-beta|grok-3-mini-beta|xAI: Grok 3 Mini Beta|C|0.0000,0.0000|131072,32768|JKT|-|Grok 3 Mini is a lightweight, smaller thinking model. Unlike traditional models|N
x-ai/grok-4|grok-4|xAI: Grok 4|C|0.0000,0.0000|256000,64000|JKSTV|-|Grok 4 is xAI's latest reasoning model with a 256k context window. It supports p|Y
x-ai/grok-4-fast|grok-4-fast|xAI: Grok 4 Fast|C|0.0000,0.0000|2000000,30000|JKSTV|-|Grok 4 Fast is xAI's latest multimodal model with SOTA cost-efficiency and a 2M|Y
x-ai/grok-4.1-fast|grok-4.1-fast|xAI: Grok 4.1 Fast|C|0.0000,0.0000|2000000,30000|JKSTV|-|Grok 4.1 Fast is xAI's best agentic tool calling model that shines in real-world|Y
x-ai/grok-code-fast-1|grok-code-fast-1|xAI: Grok Code Fast 1|C|0.0000,0.0000|256000,10000|JKST|-|Grok Code Fast 1 is a speedy and economical reasoning model that excels at agent|Y

# =============================================================================
# AIML (2 models)
# =============================================================================
aiml/dhruva|dhruva|dhruva-base|AIML: Dhruva|C|0.0000,0.0000|4096,2048|T|-|Dhruva model for Indic languages support|Y
aiml/indic-llama|indic-llama|indic-13b|AIML: Indic Llama|C|0.0000,0.0000|4096,2048|T|-|Llama variant optimized for Indian languages|Y

# =============================================================================
# MULTILINGUAL (5 models)
# =============================================================================
multilingual/bloom-1b1|bloom-1b1-f|BigScience: BLOOM 1.1B|C|0.0001,0.0003|2048,1024|VSTJ|-|BLOOM 1.1B lightweight|Y
multilingual/bloom-560m|bloom-560m-f|BigScience: BLOOM 560M|C|0.0001,0.0001|2048,1024|VSTJ|-|BLOOM 560M multilingual|Y
multilingual/mpt-7b-instruct|mpt-7b-f|MPT-7B Instruct|C|0.0001,0.0003|8192,2048|VSTJ|-|MPT instruction tuned|Y
multilingual/xlm-roberta-large|xlm-roberta-f|XLM-RoBERTa Large|C|0.0001,0.0003|512,512|VSTJ|-|100+ language RoBERTa|Y
multilingual/xlm-v|xlm-v-f|XLM-V|C|0.0001,0.0003|512,512|VSTJ|-|Vision language 100+ langs|Y
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
/// use modelsuite::models::supports_structured_output;
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
        // Test Writer models via Bedrock
        let palmyra = get_model_info("palmyra-x5").unwrap();
        assert_eq!(palmyra.provider, Provider::Bedrock);
        assert!(palmyra.capabilities.max_context >= 128000);

        // Test Chinese provider aliases
        let qwen = get_model_info("qwen-2.5-72b");
        if let Some(m) = qwen {
            assert!(m.capabilities.tools);
        }

        // Test Groq inference
        let groq = get_model_info("llama-3.3-70b");
        if let Some(m) = groq {
            assert!(m.capabilities.tools);
        }

        // Test SambaNova inference
        let sambanova = get_model_info("llama-4-maverick");
        if let Some(m) = sambanova {
            assert!(m.capabilities.vision);
        }
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
