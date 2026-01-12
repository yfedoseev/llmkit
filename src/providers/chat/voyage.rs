//! Voyage AI provider for embeddings.
//!
//! Voyage AI provides high-quality text embedding models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Voyage AI provider
pub struct VoyageProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl VoyageProvider {
    /// Create a new Voyage provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.voyageai.com/v1".to_string(),
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
        let api_key = std::env::var("VOYAGE_API_KEY").map_err(|_| {
            Error::Configuration("VOYAGE_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "voyage-3".to_string(),
            "voyage-3-lite".to_string(),
            "voyage-code-3".to_string(),
            "voyage-finance-2".to_string(),
            "voyage-law-2".to_string(),
            "voyage-multilingual-2".to_string(),
            "voyage-large-2-instruct".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<VoyageModelInfo> {
        match model {
            "voyage-3" => Some(VoyageModelInfo {
                name: model.to_string(),
                dimensions: 1024,
                max_tokens: 32000,
                model_type: VoyageModelType::GeneralPurpose,
            }),
            "voyage-3-lite" => Some(VoyageModelInfo {
                name: model.to_string(),
                dimensions: 512,
                max_tokens: 32000,
                model_type: VoyageModelType::GeneralPurpose,
            }),
            "voyage-code-3" => Some(VoyageModelInfo {
                name: model.to_string(),
                dimensions: 1024,
                max_tokens: 32000,
                model_type: VoyageModelType::Code,
            }),
            "voyage-finance-2" => Some(VoyageModelInfo {
                name: model.to_string(),
                dimensions: 1024,
                max_tokens: 16000,
                model_type: VoyageModelType::Finance,
            }),
            "voyage-law-2" => Some(VoyageModelInfo {
                name: model.to_string(),
                dimensions: 1024,
                max_tokens: 16000,
                model_type: VoyageModelType::Legal,
            }),
            _ => None,
        }
    }
}

/// Voyage model type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VoyageModelType {
    GeneralPurpose,
    Code,
    Finance,
    Legal,
    Multilingual,
}

/// Voyage model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoyageModelInfo {
    pub name: String,
    pub dimensions: u32,
    pub max_tokens: u32,
    pub model_type: VoyageModelType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = VoyageProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.voyageai.com/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = VoyageProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.contains(&"voyage-3".to_string()));
    }

    #[test]
    fn test_get_model_info() {
        let info = VoyageProvider::get_model_info("voyage-code-3").unwrap();
        assert_eq!(info.dimensions, 1024);
    }
}
