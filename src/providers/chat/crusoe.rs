//! Crusoe provider for GPU cloud inference.
//!
//! Crusoe provides sustainable GPU cloud with AI inference capabilities.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Crusoe provider
pub struct CrusoeProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl CrusoeProvider {
    /// Create a new Crusoe provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://inference.crusoecloud.com/v1".to_string(),
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
        let api_key = std::env::var("CRUSOE_API_KEY")
            .map_err(|_| Error::Configuration("CRUSOE_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "llama-3-1-8b-instruct".to_string(),
            "llama-3-1-70b-instruct".to_string(),
            "llama-3-1-405b-instruct".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<CrusoeModelInfo> {
        match model {
            m if m.contains("405b") => Some(CrusoeModelInfo {
                name: model.to_string(),
                context_window: 128000,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            m if m.contains("70b") => Some(CrusoeModelInfo {
                name: model.to_string(),
                context_window: 128000,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            m if m.contains("8b") => Some(CrusoeModelInfo {
                name: model.to_string(),
                context_window: 128000,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            _ => None,
        }
    }
}

/// Crusoe model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrusoeModelInfo {
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
        let provider = CrusoeProvider::new("test-key");
        assert_eq!(provider.base_url, "https://inference.crusoecloud.com/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = CrusoeProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 3);
    }

    #[test]
    fn test_model_info() {
        let info = CrusoeProvider::get_model_info("llama-3-1-405b-instruct").unwrap();
        assert!(info.supports_tools);
        assert_eq!(info.context_window, 128000);
    }
}
