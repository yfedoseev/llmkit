//! Cerebrium provider for serverless AI inference.
//!
//! Cerebrium provides serverless ML infrastructure with fast cold starts.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Cerebrium provider
pub struct CerebriumProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl CerebriumProvider {
    /// Create a new Cerebrium provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.cortex.cerebrium.ai/v4".to_string(),
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
        let api_key = std::env::var("CEREBRIUM_API_KEY")
            .map_err(|_| Error::Configuration("CEREBRIUM_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "llama-2-7b".to_string(),
            "llama-2-13b".to_string(),
            "mistral-7b".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<CerebriumModelInfo> {
        match model {
            m if m.contains("llama-2-7b") => Some(CerebriumModelInfo {
                name: model.to_string(),
                context_window: 4096,
                serverless: true,
                max_output_tokens: 4096,
            }),
            m if m.contains("llama-2-13b") => Some(CerebriumModelInfo {
                name: model.to_string(),
                context_window: 4096,
                serverless: true,
                max_output_tokens: 4096,
            }),
            m if m.contains("mistral-7b") => Some(CerebriumModelInfo {
                name: model.to_string(),
                context_window: 32768,
                serverless: true,
                max_output_tokens: 8192,
            }),
            _ => None,
        }
    }
}

/// Cerebrium model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CerebriumModelInfo {
    pub name: String,
    pub context_window: u32,
    pub serverless: bool,
    pub max_output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = CerebriumProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.cortex.cerebrium.ai/v4");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = CerebriumProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let info = CerebriumProvider::get_model_info("mistral-7b").unwrap();
        assert!(info.serverless);
    }
}
