//! Scaleway provider for EU cloud AI inference.
//!
//! Scaleway provides EU-based cloud infrastructure with AI services.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Scaleway provider
pub struct ScalewayProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl ScalewayProvider {
    /// Create a new Scaleway provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.scaleway.ai/v1".to_string(),
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
        let api_key = std::env::var("SCALEWAY_API_KEY")
            .map_err(|_| Error::Configuration("SCALEWAY_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "llama-3-1-8b-instruct".to_string(),
            "llama-3-1-70b-instruct".to_string(),
            "mistral-7b-instruct".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<ScalewayModelInfo> {
        match model {
            m if m.contains("llama-3-1-8b") => Some(ScalewayModelInfo {
                name: model.to_string(),
                context_window: 128000,
                region: "EU".to_string(),
                max_output_tokens: 8192,
            }),
            m if m.contains("llama-3-1-70b") => Some(ScalewayModelInfo {
                name: model.to_string(),
                context_window: 128000,
                region: "EU".to_string(),
                max_output_tokens: 8192,
            }),
            m if m.contains("mistral-7b") => Some(ScalewayModelInfo {
                name: model.to_string(),
                context_window: 32768,
                region: "EU".to_string(),
                max_output_tokens: 8192,
            }),
            _ => None,
        }
    }
}

/// Scaleway model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalewayModelInfo {
    pub name: String,
    pub context_window: u32,
    pub region: String,
    pub max_output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = ScalewayProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.scaleway.ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = ScalewayProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 3);
    }

    #[test]
    fn test_model_info() {
        let info = ScalewayProvider::get_model_info("llama-3-1-70b-instruct").unwrap();
        assert_eq!(info.region, "EU");
    }
}
