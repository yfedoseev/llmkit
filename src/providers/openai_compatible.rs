//! Generic OpenAI-compatible API provider implementation.
//!
//! This module provides a generic provider that works with any OpenAI-compatible API.
//! Many LLM providers use OpenAI's API format, making this a single implementation
//! that covers 15+ providers.
//!
//! # Supported Providers
//!
//! | Provider | Base URL | Env Var |
//! |----------|----------|---------|
//! | Together AI | `https://api.together.xyz/v1` | `TOGETHER_API_KEY` |
//! | Fireworks AI | `https://api.fireworks.ai/inference/v1` | `FIREWORKS_API_KEY` |
//! | DeepSeek | `https://api.deepseek.com/v1` | `DEEPSEEK_API_KEY` |
//! | Perplexity | `https://api.perplexity.ai` | `PERPLEXITY_API_KEY` |
//! | Anyscale | `https://api.endpoints.anyscale.com/v1` | `ANYSCALE_API_KEY` |
//! | DeepInfra | `https://api.deepinfra.com/v1/openai` | `DEEPINFRA_API_KEY` |
//! | Lepton AI | `https://llama3-1-8b.lepton.run/api/v1` | `LEPTON_API_KEY` |
//! | Novita AI | `https://api.novita.ai/v3/openai` | `NOVITA_API_KEY` |
//! | Hyperbolic | `https://api.hyperbolic.xyz/v1` | `HYPERBOLIC_API_KEY` |
//! | Cerebras | `https://api.cerebras.ai/v1` | `CEREBRAS_API_KEY` |
//! | LM Studio | `http://localhost:1234/v1` | - (local) |
//! | vLLM | `http://localhost:8000/v1` | - (local) |
//! | TGI | `http://localhost:8080/v1` | - (local) |
//! | Llamafile | `http://localhost:8080/v1` | - (local) |
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::OpenAICompatibleProvider;
//!
//! // Use a known provider
//! let together = OpenAICompatibleProvider::together_from_env()?;
//!
//! // Or a custom OpenAI-compatible endpoint
//! let custom = OpenAICompatibleProvider::custom(
//!     "my-provider",
//!     "https://my-api.example.com/v1",
//!     Some("my-api-key".to_string()),
//! )?;
//! ```

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Message, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

/// Known OpenAI-compatible provider configurations.
#[derive(Debug, Clone)]
pub struct ProviderInfo {
    /// Provider name for logging and identification.
    pub name: &'static str,
    /// Base URL for the API (without /chat/completions).
    pub base_url: &'static str,
    /// Environment variable for API key.
    pub env_var: &'static str,
    /// Whether this provider supports tool/function calling.
    pub supports_tools: bool,
    /// Whether this provider supports vision/images.
    pub supports_vision: bool,
    /// Whether this provider supports streaming.
    pub supports_streaming: bool,
    /// Default model for this provider.
    pub default_model: Option<&'static str>,
}

/// Pre-defined provider configurations.
pub mod known_providers {
    use super::ProviderInfo;

    pub const TOGETHER: ProviderInfo = ProviderInfo {
        name: "together",
        base_url: "https://api.together.xyz/v1",
        env_var: "TOGETHER_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"),
    };

    pub const FIREWORKS: ProviderInfo = ProviderInfo {
        name: "fireworks",
        base_url: "https://api.fireworks.ai/inference/v1",
        env_var: "FIREWORKS_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("accounts/fireworks/models/llama-v3p1-70b-instruct"),
    };

    pub const DEEPSEEK: ProviderInfo = ProviderInfo {
        name: "deepseek",
        base_url: "https://api.deepseek.com/v1",
        env_var: "DEEPSEEK_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("deepseek-chat"),
    };

    pub const PERPLEXITY: ProviderInfo = ProviderInfo {
        name: "perplexity",
        base_url: "https://api.perplexity.ai",
        env_var: "PERPLEXITY_API_KEY",
        supports_tools: false, // Perplexity has different tool API
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("llama-3.1-sonar-large-128k-online"),
    };

    pub const ANYSCALE: ProviderInfo = ProviderInfo {
        name: "anyscale",
        base_url: "https://api.endpoints.anyscale.com/v1",
        env_var: "ANYSCALE_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("meta-llama/Meta-Llama-3-70B-Instruct"),
    };

    pub const DEEPINFRA: ProviderInfo = ProviderInfo {
        name: "deepinfra",
        base_url: "https://api.deepinfra.com/v1/openai",
        env_var: "DEEPINFRA_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct"),
    };

    pub const LEPTON: ProviderInfo = ProviderInfo {
        name: "lepton",
        base_url: "https://llama3-1-8b.lepton.run/api/v1",
        env_var: "LEPTON_API_KEY",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const NOVITA: ProviderInfo = ProviderInfo {
        name: "novita",
        base_url: "https://api.novita.ai/v3/openai",
        env_var: "NOVITA_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("meta-llama/llama-3.1-70b-instruct"),
    };

    pub const HYPERBOLIC: ProviderInfo = ProviderInfo {
        name: "hyperbolic",
        base_url: "https://api.hyperbolic.xyz/v1",
        env_var: "HYPERBOLIC_API_KEY",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct"),
    };

    pub const CEREBRAS: ProviderInfo = ProviderInfo {
        name: "cerebras",
        base_url: "https://api.cerebras.ai/v1",
        env_var: "CEREBRAS_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("llama3.1-70b"),
    };

    // Local providers (no API key required)

    pub const LM_STUDIO: ProviderInfo = ProviderInfo {
        name: "lm_studio",
        base_url: "http://localhost:1234/v1",
        env_var: "",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    pub const VLLM: ProviderInfo = ProviderInfo {
        name: "vllm",
        base_url: "http://localhost:8000/v1",
        env_var: "",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    pub const TGI: ProviderInfo = ProviderInfo {
        name: "tgi",
        base_url: "http://localhost:8080/v1",
        env_var: "",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const LLAMAFILE: ProviderInfo = ProviderInfo {
        name: "llamafile",
        base_url: "http://localhost:8080/v1",
        env_var: "",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    // ========== Additional Cloud Providers ==========

    pub const MODAL: ProviderInfo = ProviderInfo {
        name: "modal",
        base_url: "https://api.modal.com/v1",
        env_var: "MODAL_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const LAMBDA_LABS: ProviderInfo = ProviderInfo {
        name: "lambda",
        base_url: "https://cloud.lambdalabs.com/api/v1",
        env_var: "LAMBDA_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const FRIENDLI: ProviderInfo = ProviderInfo {
        name: "friendli",
        base_url: "https://inference.friendli.ai/v1",
        env_var: "FRIENDLI_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    pub const OCTO_AI: ProviderInfo = ProviderInfo {
        name: "octoai",
        base_url: "https://text.octoai.run/v1",
        env_var: "OCTOAI_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("meta-llama-3.1-70b-instruct"),
    };

    pub const PREDIBASE: ProviderInfo = ProviderInfo {
        name: "predibase",
        base_url: "https://serving.predibase.com/v1",
        env_var: "PREDIBASE_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const NEBIUS: ProviderInfo = ProviderInfo {
        name: "nebius",
        base_url: "https://api.studio.nebius.ai/v1",
        env_var: "NEBIUS_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("meta-llama/Meta-Llama-3.1-70B-Instruct"),
    };

    pub const SILICONFLOW: ProviderInfo = ProviderInfo {
        name: "siliconflow",
        base_url: "https://api.siliconflow.cn/v1",
        env_var: "SILICONFLOW_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("Qwen/Qwen2.5-7B-Instruct"),
    };

    pub const MOONSHOT: ProviderInfo = ProviderInfo {
        name: "moonshot",
        base_url: "https://api.moonshot.cn/v1",
        env_var: "MOONSHOT_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("moonshot-v1-8k"),
    };

    pub const ZHIPU: ProviderInfo = ProviderInfo {
        name: "zhipu",
        base_url: "https://open.bigmodel.cn/api/paas/v4",
        env_var: "ZHIPU_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("glm-4"),
    };

    pub const YI: ProviderInfo = ProviderInfo {
        name: "yi",
        base_url: "https://api.lingyiwanwu.com/v1",
        env_var: "YI_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("yi-large"),
    };

    pub const MINIMAX: ProviderInfo = ProviderInfo {
        name: "minimax",
        base_url: "https://api.minimax.chat/v1",
        env_var: "MINIMAX_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("abab6-chat"),
    };

    pub const DASHSCOPE: ProviderInfo = ProviderInfo {
        name: "dashscope",
        base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
        env_var: "DASHSCOPE_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("qwen-turbo"),
    };

    // ========== Additional Local Inference Servers ==========

    pub const XINFERENCE: ProviderInfo = ProviderInfo {
        name: "xinference",
        base_url: "http://localhost:9997/v1",
        env_var: "",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    pub const FASTCHAT: ProviderInfo = ProviderInfo {
        name: "fastchat",
        base_url: "http://localhost:21002/v1",
        env_var: "",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const APHRODITE: ProviderInfo = ProviderInfo {
        name: "aphrodite",
        base_url: "http://localhost:2242/v1",
        env_var: "",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    pub const TABBY: ProviderInfo = ProviderInfo {
        name: "tabby",
        base_url: "http://localhost:8080/v1",
        env_var: "",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const KOBOLDCPP: ProviderInfo = ProviderInfo {
        name: "koboldcpp",
        base_url: "http://localhost:5001/v1",
        env_var: "",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const TEXT_GEN_WEBUI: ProviderInfo = ProviderInfo {
        name: "text-gen-webui",
        base_url: "http://localhost:5000/v1",
        env_var: "",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    // ========== New Tier 1 Providers ==========

    pub const XAI: ProviderInfo = ProviderInfo {
        name: "xai",
        base_url: "https://api.x.ai/v1",
        env_var: "XAI_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("grok-2-latest"),
    };

    pub const NVIDIA_NIM: ProviderInfo = ProviderInfo {
        name: "nvidia",
        base_url: "https://integrate.api.nvidia.com/v1",
        env_var: "NVIDIA_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("meta/llama-3.1-70b-instruct"),
    };

    pub const GITHUB_MODELS: ProviderInfo = ProviderInfo {
        name: "github",
        base_url: "https://models.inference.ai.azure.com",
        env_var: "GITHUB_TOKEN",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("gpt-4o"),
    };

    pub const AZURE_AI: ProviderInfo = ProviderInfo {
        name: "azure_ai",
        base_url: "https://api.ai.azure.com/v1",
        env_var: "AZURE_AI_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    // ========== New Tier 2 Providers ==========

    pub const FEATHERLESS: ProviderInfo = ProviderInfo {
        name: "featherless",
        base_url: "https://api.featherless.ai/v1",
        env_var: "FEATHERLESS_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const NSCALE: ProviderInfo = ProviderInfo {
        name: "nscale",
        base_url: "https://inference.nscale.com/v1",
        env_var: "NSCALE_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const VOLCENGINE: ProviderInfo = ProviderInfo {
        name: "volcengine",
        base_url: "https://ark.cn-beijing.volces.com/api/v3",
        env_var: "VOLCENGINE_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    pub const OVHCLOUD: ProviderInfo = ProviderInfo {
        name: "ovhcloud",
        base_url:
            "https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1",
        env_var: "OVHCLOUD_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const GALADRIEL: ProviderInfo = ProviderInfo {
        name: "galadriel",
        base_url: "https://api.galadriel.com/v1",
        env_var: "GALADRIEL_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    // ========== New Local/Self-hosted Providers ==========

    pub const INFINITY: ProviderInfo = ProviderInfo {
        name: "infinity",
        base_url: "http://localhost:7997/v1",
        env_var: "",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const PETALS: ProviderInfo = ProviderInfo {
        name: "petals",
        base_url: "https://chat.petals.dev/api/v1",
        env_var: "",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const TRITON: ProviderInfo = ProviderInfo {
        name: "triton",
        base_url: "http://localhost:8000/v2/models",
        env_var: "",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    // ========== New Emerging Providers ==========

    pub const BYTEZ: ProviderInfo = ProviderInfo {
        name: "bytez",
        base_url: "https://api.bytez.com/v1",
        env_var: "BYTEZ_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const MORPH: ProviderInfo = ProviderInfo {
        name: "morph",
        base_url: "https://api.morphllm.com/v1",
        env_var: "MORPH_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    pub const KLUSTER: ProviderInfo = ProviderInfo {
        name: "kluster",
        base_url: "https://api.kluster.ai/v1",
        env_var: "KLUSTER_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    // ========== Enterprise/Commercial Providers ==========

    /// Writer AI - Enterprise AI platform
    pub const WRITER: ProviderInfo = ProviderInfo {
        name: "writer",
        base_url: "https://api.writer.com/v1",
        env_var: "WRITER_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("palmyra-x-004"),
    };

    /// Reka AI - Multimodal models
    pub const REKA: ProviderInfo = ProviderInfo {
        name: "reka",
        base_url: "https://api.reka.ai/v1",
        env_var: "REKA_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("reka-core"),
    };

    /// Upstage - Korean AI company (Solar models)
    pub const UPSTAGE: ProviderInfo = ProviderInfo {
        name: "upstage",
        base_url: "https://api.upstage.ai/v1/solar",
        env_var: "UPSTAGE_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("solar-pro"),
    };

    /// AI/ML API - Model aggregator
    pub const AIML_API: ProviderInfo = ProviderInfo {
        name: "aimlapi",
        base_url: "https://api.aimlapi.com/v1",
        env_var: "AIML_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    /// Prem AI - Self-hosted AI platform
    pub const PREM: ProviderInfo = ProviderInfo {
        name: "prem",
        base_url: "https://api.premai.io/v1",
        env_var: "PREM_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// Martian - AI router/gateway
    pub const MARTIAN: ProviderInfo = ProviderInfo {
        name: "martian",
        base_url: "https://api.withmartian.com/v1",
        env_var: "MARTIAN_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    /// Centml - GPU cloud inference
    pub const CENTML: ProviderInfo = ProviderInfo {
        name: "centml",
        base_url: "https://api.centml.com/openai/v1",
        env_var: "CENTML_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// Crusoe - Cloud GPU provider
    pub const CRUSOE: ProviderInfo = ProviderInfo {
        name: "crusoe",
        base_url: "https://inference.api.crusoecloud.com/v1",
        env_var: "CRUSOE_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// CoreWeave - Cloud GPU provider
    pub const COREWEAVE: ProviderInfo = ProviderInfo {
        name: "coreweave",
        base_url: "https://inference.coreweave.com/v1",
        env_var: "COREWEAVE_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// Lightning AI - ML platform
    pub const LIGHTNING: ProviderInfo = ProviderInfo {
        name: "lightning",
        base_url: "https://api.lightning.ai/v1",
        env_var: "LIGHTNING_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// Cerebrium - Serverless ML
    pub const CEREBRIUM: ProviderInfo = ProviderInfo {
        name: "cerebrium",
        base_url: "https://api.cortex.cerebrium.ai/v1",
        env_var: "CEREBRIUM_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// Banana - Serverless GPU
    pub const BANANA: ProviderInfo = ProviderInfo {
        name: "banana",
        base_url: "https://api.banana.dev/v1",
        env_var: "BANANA_API_KEY",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// Beam - Serverless GPU
    pub const BEAM: ProviderInfo = ProviderInfo {
        name: "beam",
        base_url: "https://api.beam.cloud/v1",
        env_var: "BEAM_API_KEY",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// Mystic - ML inference platform
    pub const MYSTIC: ProviderInfo = ProviderInfo {
        name: "mystic",
        base_url: "https://api.mystic.ai/v1",
        env_var: "MYSTIC_API_KEY",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    // ========== Regional/Specialized Providers ==========

    /// Baichuan - Chinese AI (Baichuan models)
    pub const BAICHUAN: ProviderInfo = ProviderInfo {
        name: "baichuan",
        base_url: "https://api.baichuan-ai.com/v1",
        env_var: "BAICHUAN_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("Baichuan2-Turbo"),
    };

    /// Qwen via Alibaba DashScope
    pub const QWEN: ProviderInfo = ProviderInfo {
        name: "qwen",
        base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1",
        env_var: "DASHSCOPE_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("qwen-turbo"),
    };

    /// Stepfun (Step-1/2 models)
    pub const STEPFUN: ProviderInfo = ProviderInfo {
        name: "stepfun",
        base_url: "https://api.stepfun.com/v1",
        env_var: "STEPFUN_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("step-1-8k"),
    };

    /// 360 AI (Chinese provider)
    pub const AI360: ProviderInfo = ProviderInfo {
        name: "ai360",
        base_url: "https://api.360.cn/v1",
        env_var: "AI360_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// Spark (iFlytek)
    pub const SPARK: ProviderInfo = ProviderInfo {
        name: "spark",
        base_url: "https://spark-api-open.xf-yun.com/v1",
        env_var: "SPARK_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("generalv3.5"),
    };

    /// Ernie (Baidu)
    pub const ERNIE: ProviderInfo = ProviderInfo {
        name: "ernie",
        base_url: "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop",
        env_var: "ERNIE_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("ernie-4.0-8k"),
    };

    /// Hunyuan (Tencent)
    pub const HUNYUAN: ProviderInfo = ProviderInfo {
        name: "hunyuan",
        base_url: "https://hunyuan.tencentcloudapi.com/v1",
        env_var: "HUNYUAN_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("hunyuan-pro"),
    };

    // ========== Additional Local/Self-hosted ==========

    /// LocalAI - Local OpenAI alternative
    pub const LOCAL_AI: ProviderInfo = ProviderInfo {
        name: "localai",
        base_url: "http://localhost:8080/v1",
        env_var: "",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    /// GPT4All - Local models
    pub const GPT4ALL: ProviderInfo = ProviderInfo {
        name: "gpt4all",
        base_url: "http://localhost:4891/v1",
        env_var: "",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// Jan.ai - Local AI assistant
    pub const JAN: ProviderInfo = ProviderInfo {
        name: "jan",
        base_url: "http://localhost:1337/v1",
        env_var: "",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// OpenLLM - BentoML's LLM server
    pub const OPENLLM: ProviderInfo = ProviderInfo {
        name: "openllm",
        base_url: "http://localhost:3000/v1",
        env_var: "",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// Nitro - Jan's inference engine
    pub const NITRO: ProviderInfo = ProviderInfo {
        name: "nitro",
        base_url: "http://localhost:3928/v1",
        env_var: "",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// MLC LLM - Machine Learning Compilation
    pub const MLC_LLM: ProviderInfo = ProviderInfo {
        name: "mlc",
        base_url: "http://localhost:8000/v1",
        env_var: "",
        supports_tools: false,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    // ========== Proxy/Gateway Providers ==========

    /// LiteLLM Proxy
    pub const LITELLM: ProviderInfo = ProviderInfo {
        name: "litellm",
        base_url: "http://localhost:4000/v1",
        env_var: "LITELLM_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    /// Portkey - AI gateway
    pub const PORTKEY: ProviderInfo = ProviderInfo {
        name: "portkey",
        base_url: "https://api.portkey.ai/v1",
        env_var: "PORTKEY_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    /// Helicone - Observability proxy
    pub const HELICONE: ProviderInfo = ProviderInfo {
        name: "helicone",
        base_url: "https://oai.helicone.ai/v1",
        env_var: "HELICONE_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    /// Unify - AI gateway
    pub const UNIFY: ProviderInfo = ProviderInfo {
        name: "unify",
        base_url: "https://api.unify.ai/v0",
        env_var: "UNIFY_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    /// Keywords AI - AI gateway
    pub const KEYWORDS_AI: ProviderInfo = ProviderInfo {
        name: "keywordsai",
        base_url: "https://api.keywordsai.co/api",
        env_var: "KEYWORDS_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    // ========== European Regional Providers ==========

    /// Scaleway Generative APIs (France) - EU sovereign compute
    pub const SCALEWAY: ProviderInfo = ProviderInfo {
        name: "scaleway",
        base_url: "https://api.scaleway.ai/v1",
        env_var: "SCALEWAY_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("llama-3.3-70b-instruct"),
    };

    /// LightOn Paradigm (France) - Alfred models, sovereign deployment
    pub const LIGHTON: ProviderInfo = ProviderInfo {
        name: "lighton",
        base_url: "https://paradigm.lighton.ai/api/v2",
        env_var: "LIGHTON_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("alfred-40b-1023"),
    };

    /// IONOS AI Model Hub (Germany) - GDPR compliant
    pub const IONOS: ProviderInfo = ProviderInfo {
        name: "ionos",
        base_url: "https://openai.inference.de-txl.ionos.com/v1",
        env_var: "IONOS_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: None,
    };

    // ========== Chinese Regional Providers ==========

    /// SenseTime SenseNova (China) - SenseNova V6, 200k context
    pub const SENSENOVA: ProviderInfo = ProviderInfo {
        name: "sensenova",
        base_url: "https://api.sensenova.cn/v1",
        env_var: "SENSENOVA_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("SenseChat-5"),
    };

    /// Kunlun Tiangong (China) - Skywork O1, 400B MoE
    pub const TIANGONG: ProviderInfo = ProviderInfo {
        name: "tiangong",
        base_url: "https://sky-api.singularity-ai.com/v1",
        env_var: "TIANGONG_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: Some("Skywork-o1-8k"),
    };

    /// Huawei PanGu (China) - PanGu 718B, enterprise focus
    pub const PANGU: ProviderInfo = ProviderInfo {
        name: "pangu",
        base_url: "https://pangu.huaweicloud.com/v1",
        env_var: "PANGU_API_KEY",
        supports_tools: true,
        supports_vision: false,
        supports_streaming: true,
        default_model: None,
    };

    // ========== Southeast Asian Providers ==========

    /// AI Singapore SEA-LION - 11 Southeast Asian languages
    pub const SEA_LION: ProviderInfo = ProviderInfo {
        name: "sea-lion",
        base_url: "https://api.sea-lion.ai/v1",
        env_var: "SEA_LION_API_KEY",
        supports_tools: true,
        supports_vision: true,
        supports_streaming: true,
        default_model: Some("aisingapore/Qwen-SEA-LION-v4-32B-IT"),
    };
}

/// Generic provider for any OpenAI-compatible API.
///
/// This single provider implementation works with Together AI, Fireworks AI,
/// DeepSeek, Perplexity, and many other providers that use OpenAI's API format.
pub struct OpenAICompatibleProvider {
    config: ProviderConfig,
    client: Client,
    provider_info: ProviderInfo,
}

impl OpenAICompatibleProvider {
    // ========== Factory Methods for Known Providers ==========

    /// Create a Together AI provider from environment.
    pub fn together_from_env() -> Result<Self> {
        Self::from_info(known_providers::TOGETHER)
    }

    /// Create a Together AI provider with API key.
    pub fn together(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::TOGETHER, api_key)
    }

    /// Create a Fireworks AI provider from environment.
    pub fn fireworks_from_env() -> Result<Self> {
        Self::from_info(known_providers::FIREWORKS)
    }

    /// Create a Fireworks AI provider with API key.
    pub fn fireworks(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::FIREWORKS, api_key)
    }

    /// Create a DeepSeek provider from environment.
    pub fn deepseek_from_env() -> Result<Self> {
        Self::from_info(known_providers::DEEPSEEK)
    }

    /// Create a DeepSeek provider with API key.
    pub fn deepseek(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::DEEPSEEK, api_key)
    }

    /// Create a Perplexity provider from environment.
    pub fn perplexity_from_env() -> Result<Self> {
        Self::from_info(known_providers::PERPLEXITY)
    }

    /// Create a Perplexity provider with API key.
    pub fn perplexity(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::PERPLEXITY, api_key)
    }

    /// Create an Anyscale provider from environment.
    pub fn anyscale_from_env() -> Result<Self> {
        Self::from_info(known_providers::ANYSCALE)
    }

    /// Create an Anyscale provider with API key.
    pub fn anyscale(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::ANYSCALE, api_key)
    }

    /// Create a DeepInfra provider from environment.
    pub fn deepinfra_from_env() -> Result<Self> {
        Self::from_info(known_providers::DEEPINFRA)
    }

    /// Create a DeepInfra provider with API key.
    pub fn deepinfra(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::DEEPINFRA, api_key)
    }

    /// Create a Lepton AI provider from environment.
    pub fn lepton_from_env() -> Result<Self> {
        Self::from_info(known_providers::LEPTON)
    }

    /// Create a Lepton AI provider with API key and custom base URL.
    ///
    /// Lepton requires specifying the model endpoint URL.
    pub fn lepton(api_key: impl Into<String>, base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::LEPTON;
        let base_url_string = base_url.into();
        // We need to own the string, so we leak it (this is fine for static provider info)
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::from_info_with_key(info, api_key)
    }

    /// Create a Novita AI provider from environment.
    pub fn novita_from_env() -> Result<Self> {
        Self::from_info(known_providers::NOVITA)
    }

    /// Create a Novita AI provider with API key.
    pub fn novita(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::NOVITA, api_key)
    }

    /// Create a Hyperbolic provider from environment.
    pub fn hyperbolic_from_env() -> Result<Self> {
        Self::from_info(known_providers::HYPERBOLIC)
    }

    /// Create a Hyperbolic provider with API key.
    pub fn hyperbolic(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::HYPERBOLIC, api_key)
    }

    /// Create a Cerebras provider from environment.
    pub fn cerebras_from_env() -> Result<Self> {
        Self::from_info(known_providers::CEREBRAS)
    }

    /// Create a Cerebras provider with API key.
    pub fn cerebras(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::CEREBRAS, api_key)
    }

    // ========== Local Providers (No API Key Required) ==========

    /// Create a LM Studio provider (local).
    pub fn lm_studio() -> Result<Self> {
        Self::local(known_providers::LM_STUDIO)
    }

    /// Create a LM Studio provider with custom URL.
    pub fn lm_studio_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::LM_STUDIO;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a vLLM provider (local).
    pub fn vllm() -> Result<Self> {
        Self::local(known_providers::VLLM)
    }

    /// Create a vLLM provider with custom URL.
    pub fn vllm_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::VLLM;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a TGI (Text Generation Inference) provider (local).
    pub fn tgi() -> Result<Self> {
        Self::local(known_providers::TGI)
    }

    /// Create a TGI provider with custom URL.
    pub fn tgi_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::TGI;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a Llamafile provider (local).
    pub fn llamafile() -> Result<Self> {
        Self::local(known_providers::LLAMAFILE)
    }

    /// Create a Llamafile provider with custom URL.
    pub fn llamafile_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::LLAMAFILE;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    // ========== Additional Cloud Providers ==========

    /// Create a Modal provider from environment.
    pub fn modal_from_env() -> Result<Self> {
        Self::from_info(known_providers::MODAL)
    }

    /// Create a Modal provider with API key.
    pub fn modal(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::MODAL, api_key)
    }

    /// Create a Lambda Labs provider from environment.
    pub fn lambda_from_env() -> Result<Self> {
        Self::from_info(known_providers::LAMBDA_LABS)
    }

    /// Create a Lambda Labs provider with API key.
    pub fn lambda(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::LAMBDA_LABS, api_key)
    }

    /// Create a FriendliAI provider from environment.
    pub fn friendli_from_env() -> Result<Self> {
        Self::from_info(known_providers::FRIENDLI)
    }

    /// Create a FriendliAI provider with API key.
    pub fn friendli(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::FRIENDLI, api_key)
    }

    /// Create an OctoAI provider from environment.
    pub fn octoai_from_env() -> Result<Self> {
        Self::from_info(known_providers::OCTO_AI)
    }

    /// Create an OctoAI provider with API key.
    pub fn octoai(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::OCTO_AI, api_key)
    }

    /// Create a Predibase provider from environment.
    pub fn predibase_from_env() -> Result<Self> {
        Self::from_info(known_providers::PREDIBASE)
    }

    /// Create a Predibase provider with API key.
    pub fn predibase(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::PREDIBASE, api_key)
    }

    /// Create a Nebius AI provider from environment.
    pub fn nebius_from_env() -> Result<Self> {
        Self::from_info(known_providers::NEBIUS)
    }

    /// Create a Nebius AI provider with API key.
    pub fn nebius(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::NEBIUS, api_key)
    }

    /// Create a SiliconFlow provider from environment.
    pub fn siliconflow_from_env() -> Result<Self> {
        Self::from_info(known_providers::SILICONFLOW)
    }

    /// Create a SiliconFlow provider with API key.
    pub fn siliconflow(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::SILICONFLOW, api_key)
    }

    /// Create a Moonshot AI provider from environment.
    pub fn moonshot_from_env() -> Result<Self> {
        Self::from_info(known_providers::MOONSHOT)
    }

    /// Create a Moonshot AI provider with API key.
    pub fn moonshot(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::MOONSHOT, api_key)
    }

    /// Create a Zhipu AI (GLM) provider from environment.
    pub fn zhipu_from_env() -> Result<Self> {
        Self::from_info(known_providers::ZHIPU)
    }

    /// Create a Zhipu AI (GLM) provider with API key.
    pub fn zhipu(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::ZHIPU, api_key)
    }

    /// Create a 01.AI (Yi) provider from environment.
    pub fn yi_from_env() -> Result<Self> {
        Self::from_info(known_providers::YI)
    }

    /// Create a 01.AI (Yi) provider with API key.
    pub fn yi(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::YI, api_key)
    }

    /// Create a Minimax provider from environment.
    pub fn minimax_from_env() -> Result<Self> {
        Self::from_info(known_providers::MINIMAX)
    }

    /// Create a Minimax provider with API key.
    pub fn minimax(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::MINIMAX, api_key)
    }

    /// Create an Alibaba DashScope (Qwen) provider from environment.
    pub fn dashscope_from_env() -> Result<Self> {
        Self::from_info(known_providers::DASHSCOPE)
    }

    /// Create an Alibaba DashScope (Qwen) provider with API key.
    pub fn dashscope(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::DASHSCOPE, api_key)
    }

    // ========== Additional Local Inference Servers ==========

    /// Create a Xinference provider (local).
    pub fn xinference() -> Result<Self> {
        Self::local(known_providers::XINFERENCE)
    }

    /// Create a Xinference provider with custom URL.
    pub fn xinference_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::XINFERENCE;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a FastChat provider (local).
    pub fn fastchat() -> Result<Self> {
        Self::local(known_providers::FASTCHAT)
    }

    /// Create a FastChat provider with custom URL.
    pub fn fastchat_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::FASTCHAT;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create an Aphrodite Engine provider (local).
    pub fn aphrodite() -> Result<Self> {
        Self::local(known_providers::APHRODITE)
    }

    /// Create an Aphrodite Engine provider with custom URL.
    pub fn aphrodite_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::APHRODITE;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a Tabby provider (local).
    pub fn tabby() -> Result<Self> {
        Self::local(known_providers::TABBY)
    }

    /// Create a Tabby provider with custom URL.
    pub fn tabby_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::TABBY;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a KoboldCpp provider (local).
    pub fn koboldcpp() -> Result<Self> {
        Self::local(known_providers::KOBOLDCPP)
    }

    /// Create a KoboldCpp provider with custom URL.
    pub fn koboldcpp_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::KOBOLDCPP;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a Text Generation WebUI provider (local).
    pub fn text_gen_webui() -> Result<Self> {
        Self::local(known_providers::TEXT_GEN_WEBUI)
    }

    /// Create a Text Generation WebUI provider with custom URL.
    pub fn text_gen_webui_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::TEXT_GEN_WEBUI;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    // ========== New Tier 1 Cloud Providers ==========

    /// Create an xAI (Grok) provider from environment.
    pub fn xai_from_env() -> Result<Self> {
        Self::from_info(known_providers::XAI)
    }

    /// Create an xAI (Grok) provider with API key.
    pub fn xai(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::XAI, api_key)
    }

    /// Create an Nvidia NIM provider from environment.
    pub fn nvidia_nim_from_env() -> Result<Self> {
        Self::from_info(known_providers::NVIDIA_NIM)
    }

    /// Create an Nvidia NIM provider with API key.
    pub fn nvidia_nim(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::NVIDIA_NIM, api_key)
    }

    /// Create a GitHub Models provider from environment.
    pub fn github_models_from_env() -> Result<Self> {
        Self::from_info(known_providers::GITHUB_MODELS)
    }

    /// Create a GitHub Models provider with API key (GitHub token).
    pub fn github_models(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::GITHUB_MODELS, api_key)
    }

    /// Create an Azure AI provider from environment.
    pub fn azure_ai_from_env() -> Result<Self> {
        Self::from_info(known_providers::AZURE_AI)
    }

    /// Create an Azure AI provider with API key.
    pub fn azure_ai(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::AZURE_AI, api_key)
    }

    // ========== New Tier 2 Cloud Providers ==========

    /// Create a Featherless AI provider from environment.
    pub fn featherless_from_env() -> Result<Self> {
        Self::from_info(known_providers::FEATHERLESS)
    }

    /// Create a Featherless AI provider with API key.
    pub fn featherless(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::FEATHERLESS, api_key)
    }

    /// Create an Nscale provider from environment.
    pub fn nscale_from_env() -> Result<Self> {
        Self::from_info(known_providers::NSCALE)
    }

    /// Create an Nscale provider with API key.
    pub fn nscale(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::NSCALE, api_key)
    }

    /// Create a Volcengine (ByteDance) provider from environment.
    pub fn volcengine_from_env() -> Result<Self> {
        Self::from_info(known_providers::VOLCENGINE)
    }

    /// Create a Volcengine (ByteDance) provider with API key.
    pub fn volcengine(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::VOLCENGINE, api_key)
    }

    /// Create an OVHcloud AI provider from environment.
    pub fn ovhcloud_from_env() -> Result<Self> {
        Self::from_info(known_providers::OVHCLOUD)
    }

    /// Create an OVHcloud AI provider with API key.
    pub fn ovhcloud(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::OVHCLOUD, api_key)
    }

    /// Create a Galadriel provider from environment.
    pub fn galadriel_from_env() -> Result<Self> {
        Self::from_info(known_providers::GALADRIEL)
    }

    /// Create a Galadriel provider with API key.
    pub fn galadriel(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::GALADRIEL, api_key)
    }

    // ========== New Local/Self-hosted Providers ==========

    /// Create an Infinity provider (local embeddings server).
    pub fn infinity() -> Result<Self> {
        Self::local(known_providers::INFINITY)
    }

    /// Create an Infinity provider with custom URL.
    pub fn infinity_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::INFINITY;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a Petals provider (distributed inference).
    pub fn petals() -> Result<Self> {
        Self::local(known_providers::PETALS)
    }

    /// Create a Petals provider with custom URL.
    pub fn petals_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::PETALS;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a Triton Inference Server provider (local).
    pub fn triton() -> Result<Self> {
        Self::local(known_providers::TRITON)
    }

    /// Create a Triton Inference Server provider with custom URL.
    pub fn triton_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::TRITON;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    // ========== New Emerging Providers ==========

    /// Create a Bytez provider from environment.
    pub fn bytez_from_env() -> Result<Self> {
        Self::from_info(known_providers::BYTEZ)
    }

    /// Create a Bytez provider with API key.
    pub fn bytez(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::BYTEZ, api_key)
    }

    /// Create a Morph provider from environment.
    pub fn morph_from_env() -> Result<Self> {
        Self::from_info(known_providers::MORPH)
    }

    /// Create a Morph provider with API key.
    pub fn morph(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::MORPH, api_key)
    }

    /// Create a Kluster provider from environment.
    pub fn kluster_from_env() -> Result<Self> {
        Self::from_info(known_providers::KLUSTER)
    }

    /// Create a Kluster provider with API key.
    pub fn kluster(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::KLUSTER, api_key)
    }

    // ========== Enterprise/Commercial Providers ==========

    /// Create a Writer AI provider from environment.
    pub fn writer_from_env() -> Result<Self> {
        Self::from_info(known_providers::WRITER)
    }

    /// Create a Writer AI provider with API key.
    pub fn writer(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::WRITER, api_key)
    }

    /// Create a Reka AI provider from environment.
    pub fn reka_from_env() -> Result<Self> {
        Self::from_info(known_providers::REKA)
    }

    /// Create a Reka AI provider with API key.
    pub fn reka(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::REKA, api_key)
    }

    /// Create an Upstage (Solar) provider from environment.
    pub fn upstage_from_env() -> Result<Self> {
        Self::from_info(known_providers::UPSTAGE)
    }

    /// Create an Upstage (Solar) provider with API key.
    pub fn upstage(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::UPSTAGE, api_key)
    }

    /// Create an AI/ML API provider from environment.
    pub fn aimlapi_from_env() -> Result<Self> {
        Self::from_info(known_providers::AIML_API)
    }

    /// Create an AI/ML API provider with API key.
    pub fn aimlapi(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::AIML_API, api_key)
    }

    /// Create a Prem AI provider from environment.
    pub fn prem_from_env() -> Result<Self> {
        Self::from_info(known_providers::PREM)
    }

    /// Create a Prem AI provider with API key.
    pub fn prem(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::PREM, api_key)
    }

    /// Create a Martian provider from environment.
    pub fn martian_from_env() -> Result<Self> {
        Self::from_info(known_providers::MARTIAN)
    }

    /// Create a Martian provider with API key.
    pub fn martian(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::MARTIAN, api_key)
    }

    /// Create a Centml provider from environment.
    pub fn centml_from_env() -> Result<Self> {
        Self::from_info(known_providers::CENTML)
    }

    /// Create a Centml provider with API key.
    pub fn centml(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::CENTML, api_key)
    }

    /// Create a Crusoe Cloud provider from environment.
    pub fn crusoe_from_env() -> Result<Self> {
        Self::from_info(known_providers::CRUSOE)
    }

    /// Create a Crusoe Cloud provider with API key.
    pub fn crusoe(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::CRUSOE, api_key)
    }

    /// Create a CoreWeave provider from environment.
    pub fn coreweave_from_env() -> Result<Self> {
        Self::from_info(known_providers::COREWEAVE)
    }

    /// Create a CoreWeave provider with API key.
    pub fn coreweave(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::COREWEAVE, api_key)
    }

    /// Create a Lightning AI provider from environment.
    pub fn lightning_from_env() -> Result<Self> {
        Self::from_info(known_providers::LIGHTNING)
    }

    /// Create a Lightning AI provider with API key.
    pub fn lightning(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::LIGHTNING, api_key)
    }

    /// Create a Cerebrium provider from environment.
    pub fn cerebrium_from_env() -> Result<Self> {
        Self::from_info(known_providers::CEREBRIUM)
    }

    /// Create a Cerebrium provider with API key.
    pub fn cerebrium(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::CEREBRIUM, api_key)
    }

    /// Create a Banana provider from environment.
    pub fn banana_from_env() -> Result<Self> {
        Self::from_info(known_providers::BANANA)
    }

    /// Create a Banana provider with API key.
    pub fn banana(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::BANANA, api_key)
    }

    /// Create a Beam provider from environment.
    pub fn beam_from_env() -> Result<Self> {
        Self::from_info(known_providers::BEAM)
    }

    /// Create a Beam provider with API key.
    pub fn beam(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::BEAM, api_key)
    }

    /// Create a Mystic provider from environment.
    pub fn mystic_from_env() -> Result<Self> {
        Self::from_info(known_providers::MYSTIC)
    }

    /// Create a Mystic provider with API key.
    pub fn mystic(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::MYSTIC, api_key)
    }

    // ========== Regional/Specialized Providers ==========

    /// Create a Baichuan provider from environment.
    pub fn baichuan_from_env() -> Result<Self> {
        Self::from_info(known_providers::BAICHUAN)
    }

    /// Create a Baichuan provider with API key.
    pub fn baichuan(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::BAICHUAN, api_key)
    }

    /// Create a Qwen (DashScope) provider from environment.
    pub fn qwen_from_env() -> Result<Self> {
        Self::from_info(known_providers::QWEN)
    }

    /// Create a Qwen (DashScope) provider with API key.
    pub fn qwen(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::QWEN, api_key)
    }

    /// Create a Stepfun provider from environment.
    pub fn stepfun_from_env() -> Result<Self> {
        Self::from_info(known_providers::STEPFUN)
    }

    /// Create a Stepfun provider with API key.
    pub fn stepfun(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::STEPFUN, api_key)
    }

    /// Create a 360 AI provider from environment.
    pub fn ai360_from_env() -> Result<Self> {
        Self::from_info(known_providers::AI360)
    }

    /// Create a 360 AI provider with API key.
    pub fn ai360(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::AI360, api_key)
    }

    /// Create a Spark (iFlytek) provider from environment.
    pub fn spark_from_env() -> Result<Self> {
        Self::from_info(known_providers::SPARK)
    }

    /// Create a Spark (iFlytek) provider with API key.
    pub fn spark(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::SPARK, api_key)
    }

    /// Create an Ernie (Baidu) provider from environment.
    pub fn ernie_from_env() -> Result<Self> {
        Self::from_info(known_providers::ERNIE)
    }

    /// Create an Ernie (Baidu) provider with API key.
    pub fn ernie(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::ERNIE, api_key)
    }

    /// Create a Hunyuan (Tencent) provider from environment.
    pub fn hunyuan_from_env() -> Result<Self> {
        Self::from_info(known_providers::HUNYUAN)
    }

    /// Create a Hunyuan (Tencent) provider with API key.
    pub fn hunyuan(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::HUNYUAN, api_key)
    }

    // ========== Additional Local/Self-hosted ==========

    /// Create a LocalAI provider (local).
    pub fn localai() -> Result<Self> {
        Self::local(known_providers::LOCAL_AI)
    }

    /// Create a LocalAI provider with custom URL.
    pub fn localai_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::LOCAL_AI;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a GPT4All provider (local).
    pub fn gpt4all() -> Result<Self> {
        Self::local(known_providers::GPT4ALL)
    }

    /// Create a GPT4All provider with custom URL.
    pub fn gpt4all_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::GPT4ALL;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a Jan provider (local).
    pub fn jan() -> Result<Self> {
        Self::local(known_providers::JAN)
    }

    /// Create a Jan provider with custom URL.
    pub fn jan_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::JAN;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create an OpenLLM provider (local).
    pub fn openllm() -> Result<Self> {
        Self::local(known_providers::OPENLLM)
    }

    /// Create an OpenLLM provider with custom URL.
    pub fn openllm_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::OPENLLM;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a Nitro provider (local).
    pub fn nitro() -> Result<Self> {
        Self::local(known_providers::NITRO)
    }

    /// Create a Nitro provider with custom URL.
    pub fn nitro_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::NITRO;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create an MLC LLM provider (local).
    pub fn mlc() -> Result<Self> {
        Self::local(known_providers::MLC_LLM)
    }

    /// Create an MLC LLM provider with custom URL.
    pub fn mlc_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::MLC_LLM;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    // ========== Proxy/Gateway Providers ==========

    /// Create a LiteLLM proxy provider (local).
    pub fn litellm() -> Result<Self> {
        Self::local(known_providers::LITELLM)
    }

    /// Create a LiteLLM proxy provider with custom URL.
    pub fn litellm_url(base_url: impl Into<String>) -> Result<Self> {
        let mut info = known_providers::LITELLM;
        let base_url_string = base_url.into();
        let leaked: &'static str = Box::leak(base_url_string.into_boxed_str());
        info.base_url = leaked;
        Self::local(info)
    }

    /// Create a LiteLLM proxy provider with API key.
    pub fn litellm_with_key(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::LITELLM, api_key)
    }

    /// Create a Portkey gateway provider from environment.
    pub fn portkey_from_env() -> Result<Self> {
        Self::from_info(known_providers::PORTKEY)
    }

    /// Create a Portkey gateway provider with API key.
    pub fn portkey(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::PORTKEY, api_key)
    }

    /// Create a Helicone proxy provider from environment.
    pub fn helicone_from_env() -> Result<Self> {
        Self::from_info(known_providers::HELICONE)
    }

    /// Create a Helicone proxy provider with API key.
    pub fn helicone(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::HELICONE, api_key)
    }

    /// Create a Unify gateway provider from environment.
    pub fn unify_from_env() -> Result<Self> {
        Self::from_info(known_providers::UNIFY)
    }

    /// Create a Unify gateway provider with API key.
    pub fn unify(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::UNIFY, api_key)
    }

    /// Create a Keywords AI gateway provider from environment.
    pub fn keywordsai_from_env() -> Result<Self> {
        Self::from_info(known_providers::KEYWORDS_AI)
    }

    /// Create a Keywords AI gateway provider with API key.
    pub fn keywordsai(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::KEYWORDS_AI, api_key)
    }

    // ========== European Regional Providers ==========

    /// Create a Scaleway provider from environment.
    pub fn scaleway_from_env() -> Result<Self> {
        Self::from_info(known_providers::SCALEWAY)
    }

    /// Create a Scaleway provider with API key.
    pub fn scaleway(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::SCALEWAY, api_key)
    }

    /// Create a LightOn Paradigm provider from environment.
    pub fn lighton_from_env() -> Result<Self> {
        Self::from_info(known_providers::LIGHTON)
    }

    /// Create a LightOn Paradigm provider with API key.
    pub fn lighton(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::LIGHTON, api_key)
    }

    /// Create an IONOS AI provider from environment.
    pub fn ionos_from_env() -> Result<Self> {
        Self::from_info(known_providers::IONOS)
    }

    /// Create an IONOS AI provider with API key.
    pub fn ionos(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::IONOS, api_key)
    }

    // ========== Chinese Regional Providers ==========

    /// Create a SenseTime SenseNova provider from environment.
    pub fn sensenova_from_env() -> Result<Self> {
        Self::from_info(known_providers::SENSENOVA)
    }

    /// Create a SenseTime SenseNova provider with API key.
    pub fn sensenova(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::SENSENOVA, api_key)
    }

    /// Create a Kunlun Tiangong provider from environment.
    pub fn tiangong_from_env() -> Result<Self> {
        Self::from_info(known_providers::TIANGONG)
    }

    /// Create a Kunlun Tiangong provider with API key.
    pub fn tiangong(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::TIANGONG, api_key)
    }

    /// Create a Huawei PanGu provider from environment.
    pub fn pangu_from_env() -> Result<Self> {
        Self::from_info(known_providers::PANGU)
    }

    /// Create a Huawei PanGu provider with API key.
    pub fn pangu(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::PANGU, api_key)
    }

    // ========== Southeast Asian Providers ==========

    /// Create an AI Singapore SEA-LION provider from environment.
    pub fn sea_lion_from_env() -> Result<Self> {
        Self::from_info(known_providers::SEA_LION)
    }

    /// Create an AI Singapore SEA-LION provider with API key.
    pub fn sea_lion(api_key: impl Into<String>) -> Result<Self> {
        Self::from_info_with_key(known_providers::SEA_LION, api_key)
    }

    // ========== Custom Provider ==========

    /// Create a custom OpenAI-compatible provider.
    ///
    /// Use this for any provider that uses OpenAI's API format but isn't
    /// in the pre-defined list.
    ///
    /// # Arguments
    ///
    /// * `name` - Provider name for logging
    /// * `base_url` - Base URL (e.g., "https://api.example.com/v1")
    /// * `api_key` - Optional API key
    ///
    /// # Example
    ///
    /// ```ignore
    /// let provider = OpenAICompatibleProvider::custom(
    ///     "my-provider",
    ///     "https://my-api.example.com/v1",
    ///     Some("my-api-key".to_string()),
    /// )?;
    /// ```
    pub fn custom(
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Result<Self> {
        let name_string = name.into();
        let base_url_string = base_url.into();

        // Leak strings to get static references
        let name_static: &'static str = Box::leak(name_string.into_boxed_str());
        let base_url_static: &'static str = Box::leak(base_url_string.into_boxed_str());

        let info = ProviderInfo {
            name: name_static,
            base_url: base_url_static,
            env_var: "",
            supports_tools: true,
            supports_vision: true,
            supports_streaming: true,
            default_model: None,
        };

        let config = ProviderConfig {
            api_key,
            base_url: Some(base_url_static.to_string()),
            ..Default::default()
        };

        Self::new(config, info)
    }

    // ========== Internal Construction ==========

    fn from_info(info: ProviderInfo) -> Result<Self> {
        let config = ProviderConfig::from_env(info.env_var);
        let config = config.with_base_url(info.base_url);
        Self::new(config, info)
    }

    fn from_info_with_key(info: ProviderInfo, api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key).with_base_url(info.base_url);
        Self::new(config, info)
    }

    fn local(info: ProviderInfo) -> Result<Self> {
        let config = ProviderConfig {
            api_key: None,
            base_url: Some(info.base_url.to_string()),
            ..Default::default()
        };
        Self::new(config, info)
    }

    /// Create a new provider with custom configuration.
    pub fn new(config: ProviderConfig, provider_info: ProviderInfo) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", key)
                    .parse()
                    .map_err(|_| Error::config("Invalid API key format"))?,
            );
        }

        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        // Add custom headers
        for (key, value) in &config.custom_headers {
            headers.insert(
                reqwest::header::HeaderName::try_from(key.as_str())
                    .map_err(|_| Error::config(format!("Invalid header name: {}", key)))?,
                value
                    .parse()
                    .map_err(|_| Error::config(format!("Invalid header value for {}", key)))?,
            );
        }

        let client = Client::builder()
            .timeout(config.timeout)
            .default_headers(headers)
            .build()?;

        Ok(Self {
            config,
            client,
            provider_info,
        })
    }

    fn api_url(&self) -> String {
        let base = self
            .config
            .base_url
            .as_deref()
            .unwrap_or(self.provider_info.base_url);

        // Ensure we have /chat/completions endpoint
        if base.ends_with("/chat/completions") {
            base.to_string()
        } else if base.ends_with('/') {
            format!("{}chat/completions", base)
        } else {
            format!("{}/chat/completions", base)
        }
    }

    /// Convert our unified request to OpenAI's format.
    fn convert_request(&self, request: &CompletionRequest) -> OpenAIRequest {
        let mut messages: Vec<OpenAIMessage> = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(OpenAIContent::Text(system.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Convert messages
        for msg in &request.messages {
            messages.extend(self.convert_message(msg));
        }

        // Convert tools (only if provider supports them)
        let tools = if self.provider_info.supports_tools {
            request.tools.as_ref().map(|tools| {
                tools
                    .iter()
                    .map(|t| OpenAITool {
                        tool_type: "function".to_string(),
                        function: OpenAIFunction {
                            name: t.name.clone(),
                            description: Some(t.description.clone()),
                            parameters: t.input_schema.clone(),
                        },
                    })
                    .collect()
            })
        } else {
            None
        };

        OpenAIRequest {
            model: request.model.clone(),
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop_sequences.clone(),
            stream: request.stream,
            tools,
            stream_options: if request.stream {
                Some(StreamOptions {
                    include_usage: true,
                })
            } else {
                None
            },
        }
    }

    fn convert_message(&self, message: &Message) -> Vec<OpenAIMessage> {
        let mut result = Vec::new();

        match message.role {
            Role::System => {
                let text = message.text_content();
                if !text.is_empty() {
                    result.push(OpenAIMessage {
                        role: "system".to_string(),
                        content: Some(OpenAIContent::Text(text)),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
            Role::User => {
                // Check if we have tool results
                let tool_results: Vec<_> = message
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => Some((tool_use_id.clone(), content.clone())),
                        _ => None,
                    })
                    .collect();

                if !tool_results.is_empty() {
                    // Tool results become separate "tool" role messages
                    for (tool_call_id, content) in tool_results {
                        result.push(OpenAIMessage {
                            role: "tool".to_string(),
                            content: Some(OpenAIContent::Text(content)),
                            tool_calls: None,
                            tool_call_id: Some(tool_call_id),
                        });
                    }
                } else {
                    // Regular user message
                    let content_parts: Vec<OpenAIContentPart> = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text { text } => {
                                Some(OpenAIContentPart::Text { text: text.clone() })
                            }
                            ContentBlock::Image { media_type, data }
                                if self.provider_info.supports_vision =>
                            {
                                Some(OpenAIContentPart::ImageUrl {
                                    image_url: ImageUrl {
                                        url: format!("data:{};base64,{}", media_type, data),
                                        detail: None,
                                    },
                                })
                            }
                            ContentBlock::ImageUrl { url }
                                if self.provider_info.supports_vision =>
                            {
                                Some(OpenAIContentPart::ImageUrl {
                                    image_url: ImageUrl {
                                        url: url.clone(),
                                        detail: None,
                                    },
                                })
                            }
                            _ => None,
                        })
                        .collect();

                    if content_parts.len() == 1 {
                        if let OpenAIContentPart::Text { text } = &content_parts[0] {
                            result.push(OpenAIMessage {
                                role: "user".to_string(),
                                content: Some(OpenAIContent::Text(text.clone())),
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        } else {
                            result.push(OpenAIMessage {
                                role: "user".to_string(),
                                content: Some(OpenAIContent::Parts(content_parts)),
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        }
                    } else if !content_parts.is_empty() {
                        result.push(OpenAIMessage {
                            role: "user".to_string(),
                            content: Some(OpenAIContent::Parts(content_parts)),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                }
            }
            Role::Assistant => {
                // Check for tool calls
                let tool_calls: Vec<OpenAIToolCall> = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolUse { id, name, input } => Some(OpenAIToolCall {
                            id: id.clone(),
                            call_type: "function".to_string(),
                            function: OpenAIFunctionCall {
                                name: name.clone(),
                                arguments: input.to_string(),
                            },
                        }),
                        _ => None,
                    })
                    .collect();

                let text_content = message.text_content();

                result.push(OpenAIMessage {
                    role: "assistant".to_string(),
                    content: if text_content.is_empty() {
                        None
                    } else {
                        Some(OpenAIContent::Text(text_content))
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    tool_call_id: None,
                });
            }
        }

        result
    }

    fn convert_response(&self, response: OpenAIResponse) -> CompletionResponse {
        let choice = response.choices.into_iter().next().unwrap_or_default();
        let mut content = Vec::new();

        // Add text content
        if let Some(text) = choice.message.content {
            content.push(ContentBlock::Text { text });
        }

        // Add tool calls
        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in tool_calls {
                let input = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                content.push(ContentBlock::ToolUse {
                    id: tc.id,
                    name: tc.function.name,
                    input,
                });
            }
        }

        let stop_reason = match choice.finish_reason.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            Some("tool_calls") => StopReason::ToolUse,
            Some("content_filter") => StopReason::ContentFilter,
            _ => StopReason::EndTurn,
        };

        let (input_tokens, output_tokens) = match response.usage {
            Some(u) => (u.prompt_tokens, u.completion_tokens),
            None => (0, 0),
        };

        CompletionResponse {
            id: response.id,
            model: response.model,
            content,
            stop_reason,
            usage: Usage {
                input_tokens,
                output_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        }
    }

    async fn handle_error_response(&self, response: reqwest::Response) -> Error {
        let status = response.status().as_u16();

        match response.json::<OpenAIErrorResponse>().await {
            Ok(err) => {
                let error_type = err.error.error_type.as_deref().unwrap_or("unknown");
                let message = &err.error.message;

                match error_type {
                    "invalid_api_key" | "authentication_error" => Error::auth(message),
                    "rate_limit_exceeded" => Error::rate_limited(message, None),
                    "invalid_request_error" => Error::invalid_request(message),
                    "model_not_found" => Error::ModelNotFound(message.clone()),
                    "context_length_exceeded" => Error::ContextLengthExceeded(message.clone()),
                    "server_error" => Error::server(500, message),
                    _ => Error::server(status, message),
                }
            }
            Err(_) => Error::server(status, "Unknown error"),
        }
    }
}

#[async_trait]
impl Provider for OpenAICompatibleProvider {
    fn name(&self) -> &str {
        self.provider_info.name
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        // Only require API key for remote providers
        if !self.provider_info.env_var.is_empty() {
            self.config.require_api_key()?;
        }

        let mut api_request = self.convert_request(&request);
        api_request.stream = false;

        let response = self
            .client
            .post(self.api_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let openai_response: OpenAIResponse = response.json().await?;
        Ok(self.convert_response(openai_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        if !self.provider_info.supports_streaming {
            return Err(Error::invalid_request(format!(
                "Provider {} does not support streaming",
                self.provider_info.name
            )));
        }

        // Only require API key for remote providers
        if !self.provider_info.env_var.is_empty() {
            self.config.require_api_key()?;
        }

        let mut api_request = self.convert_request(&request);
        api_request.stream = true;

        let response = self
            .client
            .post(self.api_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let stream = parse_openai_stream(response);
        Ok(Box::pin(stream))
    }

    fn supports_tools(&self) -> bool {
        self.provider_info.supports_tools
    }

    fn supports_vision(&self) -> bool {
        self.provider_info.supports_vision
    }

    fn supports_streaming(&self) -> bool {
        self.provider_info.supports_streaming
    }

    fn default_model(&self) -> Option<&str> {
        self.provider_info.default_model
    }
}

/// Parse OpenAI SSE stream into our unified StreamChunk format.
fn parse_openai_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
    use async_stream::try_stream;
    use futures::StreamExt;

    try_stream! {
        let mut event_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut tool_call_builders: std::collections::HashMap<usize, (String, String, String)> = std::collections::HashMap::new();
        let mut sent_start = false;

        while let Some(chunk) = event_stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete SSE lines
            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..]; // Skip "data: "

                if data == "[DONE]" {
                    yield StreamChunk {
                        event_type: StreamEventType::MessageStop,
                        index: None,
                        delta: None,
                        stop_reason: None,
                        usage: None,
                    };
                    continue;
                }

                if let Ok(parsed) = serde_json::from_str::<OpenAIStreamResponse>(data) {
                    if !sent_start {
                        yield StreamChunk {
                            event_type: StreamEventType::MessageStart,
                            index: None,
                            delta: None,
                            stop_reason: None,
                            usage: None,
                        };
                        sent_start = true;
                    }

                    for choice in &parsed.choices {
                        // Handle text content
                        if let Some(ref content) = choice.delta.content {
                            yield StreamChunk {
                                event_type: StreamEventType::ContentBlockDelta,
                                index: Some(0),
                                delta: Some(ContentDelta::TextDelta { text: content.clone() }),
                                stop_reason: None,
                                usage: None,
                            };
                        }

                        // Handle tool calls
                        if let Some(ref tool_calls) = choice.delta.tool_calls {
                            for tc in tool_calls {
                                let idx = tc.index.unwrap_or(0);
                                let entry = tool_call_builders.entry(idx).or_insert_with(|| {
                                    (String::new(), String::new(), String::new())
                                });

                                if let Some(ref id) = tc.id {
                                    entry.0 = id.clone();
                                }
                                if let Some(ref func) = tc.function {
                                    if let Some(ref name) = func.name {
                                        entry.1 = name.clone();
                                    }
                                    if let Some(ref args) = func.arguments {
                                        entry.2.push_str(args);
                                    }
                                }

                                yield StreamChunk {
                                    event_type: StreamEventType::ContentBlockDelta,
                                    index: Some(idx + 1), // Offset by 1 for text block
                                    delta: Some(ContentDelta::ToolUseDelta {
                                        id: tc.id.clone(),
                                        name: tc.function.as_ref().and_then(|f| f.name.clone()),
                                        input_json_delta: tc.function.as_ref().and_then(|f| f.arguments.clone()),
                                    }),
                                    stop_reason: None,
                                    usage: None,
                                };
                            }
                        }

                        // Handle finish reason
                        if let Some(ref reason) = choice.finish_reason {
                            let stop_reason = match reason.as_str() {
                                "stop" => StopReason::EndTurn,
                                "length" => StopReason::MaxTokens,
                                "tool_calls" => StopReason::ToolUse,
                                "content_filter" => StopReason::ContentFilter,
                                _ => StopReason::EndTurn,
                            };

                            yield StreamChunk {
                                event_type: StreamEventType::MessageDelta,
                                index: None,
                                delta: None,
                                stop_reason: Some(stop_reason),
                                usage: None,
                            };
                        }
                    }

                    // Handle usage
                    if let Some(ref usage) = parsed.usage {
                        yield StreamChunk {
                            event_type: StreamEventType::MessageDelta,
                            index: None,
                            delta: None,
                            stop_reason: None,
                            usage: Some(Usage {
                                input_tokens: usage.prompt_tokens,
                                output_tokens: usage.completion_tokens,
                                cache_creation_input_tokens: 0,
                                cache_read_input_tokens: 0,
                            }),
                        };
                    }
                }
            }
        }
    }
}

// ========== OpenAI API Types ==========

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAIContent {
    Text(String),
    Parts(Vec<OpenAIContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAIContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Serialize)]
struct ImageUrl {
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    id: String,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamResponse {
    choices: Vec<OpenAIStreamChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamToolCall {
    index: Option<usize>,
    id: Option<String>,
    function: Option<OpenAIStreamFunction>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorResponse {
    error: OpenAIError,
}

#[derive(Debug, Deserialize)]
struct OpenAIError {
    #[serde(rename = "type")]
    error_type: Option<String>,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_info() {
        assert_eq!(known_providers::TOGETHER.name, "together");
        const { assert!(known_providers::TOGETHER.supports_tools) };
        const { assert!(known_providers::DEEPSEEK.supports_streaming) };
        const { assert!(!known_providers::PERPLEXITY.supports_tools) };
    }

    #[test]
    fn test_custom_provider_creation() {
        let provider = OpenAICompatibleProvider::custom(
            "test-provider",
            "https://api.test.com/v1",
            Some("test-key".to_string()),
        )
        .unwrap();

        assert_eq!(provider.name(), "test-provider");
        assert!(provider.supports_tools());
        assert!(provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_local_provider() {
        let provider = OpenAICompatibleProvider::lm_studio().unwrap();

        assert_eq!(provider.name(), "lm_studio");
        assert!(provider.api_url().contains("localhost:1234"));
    }

    #[test]
    fn test_api_url_construction() {
        // Test with trailing slash
        let provider =
            OpenAICompatibleProvider::custom("test", "https://api.test.com/v1/", None).unwrap();
        assert_eq!(
            provider.api_url(),
            "https://api.test.com/v1/chat/completions"
        );

        // Test without trailing slash
        let provider =
            OpenAICompatibleProvider::custom("test", "https://api.test.com/v1", None).unwrap();
        assert_eq!(
            provider.api_url(),
            "https://api.test.com/v1/chat/completions"
        );
    }

    #[test]
    fn test_request_conversion() {
        let provider =
            OpenAICompatibleProvider::custom("test", "https://api.test.com/v1", None).unwrap();

        let request = CompletionRequest::new("test-model", vec![Message::user("Hello")])
            .with_system("You are helpful")
            .with_max_tokens(1024);

        let openai_req = provider.convert_request(&request);

        assert_eq!(openai_req.model, "test-model");
        assert_eq!(openai_req.max_tokens, Some(1024));
        assert_eq!(openai_req.messages.len(), 2); // system + user
    }
}
