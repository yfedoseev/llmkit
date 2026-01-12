//! OctoAI provider for fast LLM inference.
//!
//! OctoAI provides optimized inference with fast cold start times.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// OctoAI provider
pub struct OctoAIProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl OctoAIProvider {
    /// Create a new OctoAI provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://text.octoai.run/v1".to_string(),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(api_key: &str, base_url: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        }
    }

    /// Create from environment variable
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OCTOAI_API_KEY")
            .map_err(|_| Error::Configuration("OCTOAI_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "llama-2-13b-chat".to_string(),
            "llama-2-70b-chat".to_string(),
            "codellama-34b-instruct".to_string(),
            "mistral-7b-instruct".to_string(),
            "mixtral-8x7b-instruct".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<OctoAIModelInfo> {
        match model {
            m if m.contains("llama-2-13b") => Some(OctoAIModelInfo {
                name: model.to_string(),
                context_window: 4096,
                supports_tools: false,
                max_output_tokens: 4096,
            }),
            m if m.contains("llama-2-70b") => Some(OctoAIModelInfo {
                name: model.to_string(),
                context_window: 4096,
                supports_tools: false,
                max_output_tokens: 4096,
            }),
            m if m.contains("codellama") => Some(OctoAIModelInfo {
                name: model.to_string(),
                context_window: 16384,
                supports_tools: false,
                max_output_tokens: 4096,
            }),
            m if m.contains("mistral-7b") => Some(OctoAIModelInfo {
                name: model.to_string(),
                context_window: 32768,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            m if m.contains("mixtral") => Some(OctoAIModelInfo {
                name: model.to_string(),
                context_window: 32768,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            _ => None,
        }
    }
}

/// OctoAI model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OctoAIModelInfo {
    pub name: String,
    pub context_window: u32,
    pub supports_tools: bool,
    pub max_output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = OctoAIProvider::new("test-key");
        assert_eq!(provider.base_url, "https://text.octoai.run/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = OctoAIProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 5);
    }

    #[test]
    fn test_model_info() {
        let info = OctoAIProvider::get_model_info("mixtral-8x7b-instruct").unwrap();
        assert!(info.supports_tools);
    }
}
