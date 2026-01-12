//! Lightning AI provider for PyTorch-based inference.
//!
//! Lightning AI provides infrastructure for training and serving models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Lightning AI provider
pub struct LightningProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl LightningProvider {
    /// Create a new Lightning provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.lightning.ai/v1".to_string(),
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
        let api_key = std::env::var("LIGHTNING_API_KEY")
            .map_err(|_| Error::Configuration("LIGHTNING_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "lit-llama-2-7b".to_string(),
            "lit-mistral-7b".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<LightningModelInfo> {
        match model {
            m if m.contains("llama-2-7b") => Some(LightningModelInfo {
                name: model.to_string(),
                context_window: 4096,
                framework: "pytorch".to_string(),
                max_output_tokens: 4096,
            }),
            m if m.contains("mistral-7b") => Some(LightningModelInfo {
                name: model.to_string(),
                context_window: 32768,
                framework: "pytorch".to_string(),
                max_output_tokens: 8192,
            }),
            _ => None,
        }
    }
}

/// Lightning model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LightningModelInfo {
    pub name: String,
    pub context_window: u32,
    pub framework: String,
    pub max_output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = LightningProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.lightning.ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = LightningProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let info = LightningProvider::get_model_info("lit-mistral-7b").unwrap();
        assert_eq!(info.framework, "pytorch");
    }
}
