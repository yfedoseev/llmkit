//! Nebius AI Studio provider for EU-based inference.
//!
//! Nebius provides EU-sovereign AI inference with an OpenAI-compatible API.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Nebius AI provider
pub struct NebiusProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl NebiusProvider {
    /// Create a new Nebius provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.studio.nebius.ai/v1".to_string(),
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
        let api_key = std::env::var("NEBIUS_API_KEY").map_err(|_| {
            Error::Configuration("NEBIUS_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "meta-llama/Meta-Llama-3.1-405B-Instruct".to_string(),
            "meta-llama/Meta-Llama-3.1-70B-Instruct".to_string(),
            "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string(),
            "Qwen/Qwen2.5-72B-Instruct".to_string(),
            "mistralai/Mixtral-8x22B-Instruct-v0.1".to_string(),
            "deepseek-ai/DeepSeek-V3".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<NebiusModelInfo> {
        match model {
            m if m.contains("405B") => Some(NebiusModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            m if m.contains("70B") || m.contains("72B") => Some(NebiusModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            _ => None,
        }
    }
}

/// Nebius model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NebiusModelInfo {
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
        let provider = NebiusProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.studio.nebius.ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = NebiusProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }
}
