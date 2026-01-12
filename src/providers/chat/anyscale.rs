//! Anyscale provider for scalable LLM inference.
//!
//! Anyscale provides fast, scalable inference for open-source models
//! with an OpenAI-compatible API.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Anyscale provider for scalable inference
pub struct AnyscaleProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl AnyscaleProvider {
    /// Create a new Anyscale provider with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.anyscale.com/v1".to_string(),
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
        let api_key = std::env::var("ANYSCALE_API_KEY").map_err(|_| {
            Error::Configuration("ANYSCALE_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "meta-llama/Llama-3.3-70B-Instruct".to_string(),
            "meta-llama/Meta-Llama-3.1-405B-Instruct".to_string(),
            "meta-llama/Meta-Llama-3.1-70B-Instruct".to_string(),
            "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string(),
            "mistralai/Mixtral-8x22B-Instruct-v0.1".to_string(),
            "mistralai/Mixtral-8x7B-Instruct-v0.1".to_string(),
            "codellama/CodeLlama-70b-Instruct-hf".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<AnyscaleModelInfo> {
        match model {
            m if m.contains("405B") => Some(AnyscaleModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            m if m.contains("70B") => Some(AnyscaleModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            m if m.contains("8B") => Some(AnyscaleModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            _ => None,
        }
    }
}

/// Anyscale model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnyscaleModelInfo {
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
        let provider = AnyscaleProvider::new("test-key");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.base_url, "https://api.anyscale.com/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = AnyscaleProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_get_model_info() {
        let info = AnyscaleProvider::get_model_info("meta-llama/Meta-Llama-3.1-70B-Instruct");
        assert!(info.is_some());
    }
}
