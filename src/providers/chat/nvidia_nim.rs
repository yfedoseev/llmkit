//! Nvidia NIM provider for AI inference microservices.
//!
//! Nvidia NIM (NVIDIA Inference Microservice) provides optimized
//! inference for AI models with an OpenAI-compatible API.
//!
//! # Features
//! - OpenAI-compatible API
//! - Nvidia-optimized models (Llama, Mistral, Nemotron, etc.)
//! - Streaming responses
//! - Function calling support
//! - Vision model support

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Nvidia NIM provider for optimized AI inference
pub struct NvidiaNIMProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl NvidiaNIMProvider {
    /// Create a new Nvidia NIM provider with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://integrate.api.nvidia.com/v1".to_string(),
        }
    }

    /// Create with custom base URL (for self-hosted NIM)
    pub fn with_base_url(api_key: &str, base_url: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        }
    }

    /// Create from environment variable `NVIDIA_NIM_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("NVIDIA_NIM_API_KEY").map_err(|_| {
            Error::Configuration("NVIDIA_NIM_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            // Nvidia Nemotron models
            "nvidia/llama-3.1-nemotron-70b-instruct".to_string(),
            "nvidia/nemotron-4-340b-instruct".to_string(),
            "nvidia/nemotron-mini-4b-instruct".to_string(),
            // Llama models (Nvidia optimized)
            "meta/llama-3.3-70b-instruct".to_string(),
            "meta/llama-3.2-3b-instruct".to_string(),
            "meta/llama-3.2-1b-instruct".to_string(),
            "meta/llama-3.1-405b-instruct".to_string(),
            "meta/llama-3.1-70b-instruct".to_string(),
            "meta/llama-3.1-8b-instruct".to_string(),
            // Vision models
            "meta/llama-3.2-90b-vision-instruct".to_string(),
            "meta/llama-3.2-11b-vision-instruct".to_string(),
            // Mistral models
            "mistralai/mistral-large-2-instruct".to_string(),
            "mistralai/mixtral-8x22b-instruct-v0.1".to_string(),
            "mistralai/mixtral-8x7b-instruct-v0.1".to_string(),
            "mistralai/mistral-7b-instruct-v0.3".to_string(),
            // Google Gemma
            "google/gemma-2-27b-it".to_string(),
            "google/gemma-2-9b-it".to_string(),
            "google/gemma-2-2b-it".to_string(),
            // Microsoft Phi
            "microsoft/phi-3.5-mini-instruct".to_string(),
            "microsoft/phi-3-medium-128k-instruct".to_string(),
            // Qwen
            "qwen/qwen2.5-72b-instruct".to_string(),
            "qwen/qwen2.5-7b-instruct".to_string(),
            // DeepSeek
            "deepseek-ai/deepseek-r1".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<NvidiaNIMModelInfo> {
        match model {
            "nvidia/llama-3.1-nemotron-70b-instruct" => Some(NvidiaNIMModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            "nvidia/nemotron-4-340b-instruct" => Some(NvidiaNIMModelInfo {
                name: model.to_string(),
                context_window: 4096,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 4096,
            }),
            "meta/llama-3.3-70b-instruct" => Some(NvidiaNIMModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            "meta/llama-3.1-405b-instruct" => Some(NvidiaNIMModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            "meta/llama-3.2-90b-vision-instruct" => Some(NvidiaNIMModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_vision: true,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            "meta/llama-3.2-11b-vision-instruct" => Some(NvidiaNIMModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_vision: true,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            "mistralai/mistral-large-2-instruct" => Some(NvidiaNIMModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            "deepseek-ai/deepseek-r1" => Some(NvidiaNIMModelInfo {
                name: model.to_string(),
                context_window: 65536,
                supports_vision: false,
                supports_tools: false,
                max_output_tokens: 32768,
            }),
            _ => None,
        }
    }
}

/// Nvidia NIM model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NvidiaNIMModelInfo {
    /// Model name
    pub name: String,
    /// Context window size
    pub context_window: u32,
    /// Whether this model supports vision
    pub supports_vision: bool,
    /// Whether this model supports function calling
    pub supports_tools: bool,
    /// Maximum output tokens
    pub max_output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nvidia_nim_provider_creation() {
        let provider = NvidiaNIMProvider::new("test-key");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.base_url, "https://integrate.api.nvidia.com/v1");
    }

    #[test]
    fn test_nvidia_nim_custom_base_url() {
        let provider = NvidiaNIMProvider::with_base_url("test-key", "https://custom.api/v1");
        assert_eq!(provider.base_url, "https://custom.api/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = NvidiaNIMProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.contains("nemotron")));
    }

    #[test]
    fn test_get_model_info() {
        let info =
            NvidiaNIMProvider::get_model_info("nvidia/llama-3.1-nemotron-70b-instruct").unwrap();
        assert!(info.supports_tools);
        assert!(!info.supports_vision);
        assert_eq!(info.context_window, 131072);
    }

    #[test]
    fn test_get_model_info_vision() {
        let info = NvidiaNIMProvider::get_model_info("meta/llama-3.2-90b-vision-instruct").unwrap();
        assert!(info.supports_vision);
        assert!(info.supports_tools);
    }

    #[test]
    fn test_model_info_invalid() {
        let info = NvidiaNIMProvider::get_model_info("invalid-model");
        assert!(info.is_none());
    }
}
