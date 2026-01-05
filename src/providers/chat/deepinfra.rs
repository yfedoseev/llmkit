//! DeepInfra provider for serverless GPU inference.
//!
//! DeepInfra provides fast, scalable inference for open-source models
//! with an OpenAI-compatible API.
//!
//! # Features
//! - OpenAI-compatible API
//! - Wide range of open-source models (Llama, Mistral, Qwen, etc.)
//! - Streaming responses
//! - Function calling support
//! - Embeddings support

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// DeepInfra provider for serverless inference
pub struct DeepInfraProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl DeepInfraProvider {
    /// Create a new DeepInfra provider with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.deepinfra.com/v1/openai".to_string(),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(api_key: &str, base_url: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        }
    }

    /// Create from environment variable `DEEPINFRA_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("DEEPINFRA_API_KEY").map_err(|_| {
            Error::Configuration("DEEPINFRA_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            // Llama models
            "meta-llama/Llama-3.3-70B-Instruct".to_string(),
            "meta-llama/Llama-3.3-70B-Instruct-Turbo".to_string(),
            "meta-llama/Meta-Llama-3.1-405B-Instruct".to_string(),
            "meta-llama/Meta-Llama-3.1-70B-Instruct".to_string(),
            "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string(),
            // Qwen models
            "Qwen/Qwen2.5-72B-Instruct".to_string(),
            "Qwen/Qwen2.5-Coder-32B-Instruct".to_string(),
            "Qwen/QwQ-32B-Preview".to_string(),
            // Mistral models
            "mistralai/Mistral-7B-Instruct-v0.3".to_string(),
            "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(),
            "mistralai/Mixtral-8x22B-Instruct-v0.1".to_string(),
            // DeepSeek models
            "deepseek-ai/DeepSeek-R1".to_string(),
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B".to_string(),
            "deepseek-ai/DeepSeek-V3".to_string(),
            // Google Gemma
            "google/gemma-2-27b-it".to_string(),
            "google/gemma-2-9b-it".to_string(),
            // Microsoft
            "microsoft/WizardLM-2-8x22B".to_string(),
            // Nvidia
            "nvidia/Llama-3.1-Nemotron-70B-Instruct".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<DeepInfraModelInfo> {
        match model {
            "meta-llama/Llama-3.3-70B-Instruct" | "meta-llama/Llama-3.3-70B-Instruct-Turbo" => {
                Some(DeepInfraModelInfo {
                    name: model.to_string(),
                    context_window: 131072,
                    supports_vision: false,
                    supports_tools: true,
                    max_output_tokens: 16384,
                })
            }
            "meta-llama/Meta-Llama-3.1-405B-Instruct" => Some(DeepInfraModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            "meta-llama/Meta-Llama-3.1-70B-Instruct" => Some(DeepInfraModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            "meta-llama/Meta-Llama-3.1-8B-Instruct" => Some(DeepInfraModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            "Qwen/Qwen2.5-72B-Instruct" => Some(DeepInfraModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            "Qwen/Qwen2.5-Coder-32B-Instruct" => Some(DeepInfraModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            "Qwen/QwQ-32B-Preview" => Some(DeepInfraModelInfo {
                name: model.to_string(),
                context_window: 32768,
                supports_vision: false,
                supports_tools: false,
                max_output_tokens: 16384,
            }),
            "deepseek-ai/DeepSeek-R1" => Some(DeepInfraModelInfo {
                name: model.to_string(),
                context_window: 65536,
                supports_vision: false,
                supports_tools: false,
                max_output_tokens: 32768,
            }),
            "deepseek-ai/DeepSeek-V3" => Some(DeepInfraModelInfo {
                name: model.to_string(),
                context_window: 65536,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            _ => None,
        }
    }
}

/// DeepInfra model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepInfraModelInfo {
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
    fn test_deepinfra_provider_creation() {
        let provider = DeepInfraProvider::new("test-key");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.base_url, "https://api.deepinfra.com/v1/openai");
    }

    #[test]
    fn test_deepinfra_custom_base_url() {
        let provider = DeepInfraProvider::with_base_url("test-key", "https://custom.api/v1");
        assert_eq!(provider.base_url, "https://custom.api/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = DeepInfraProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.contains("Llama-3.3-70B-Instruct")));
    }

    #[test]
    fn test_get_model_info() {
        let info = DeepInfraProvider::get_model_info("meta-llama/Llama-3.3-70B-Instruct").unwrap();
        assert!(info.supports_tools);
        assert!(!info.supports_vision);
        assert_eq!(info.context_window, 131072);
    }

    #[test]
    fn test_get_model_info_deepseek() {
        let info = DeepInfraProvider::get_model_info("deepseek-ai/DeepSeek-R1").unwrap();
        assert!(!info.supports_tools); // R1 reasoning model
        assert_eq!(info.context_window, 65536);
    }

    #[test]
    fn test_model_info_invalid() {
        let info = DeepInfraProvider::get_model_info("invalid-model");
        assert!(info.is_none());
    }
}
