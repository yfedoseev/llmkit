//! Preferred Networks PLaMo provider.
//!
//! PLaMo is a Japanese-focused large language model by Preferred Networks.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// PLaMo provider
pub struct PLaMoProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl PLaMoProvider {
    /// Create a new PLaMo provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.preferrednetworks.ai/v1".to_string(),
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
        let api_key = std::env::var("PLAMO_API_KEY")
            .map_err(|_| Error::Configuration("PLAMO_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["plamo-13b".to_string(), "plamo-100b".to_string()])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<PLaMoModelInfo> {
        match model {
            "plamo-13b" => Some(PLaMoModelInfo {
                name: model.to_string(),
                parameters: "13B".to_string(),
                context_length: 8192,
                language_focus: "Japanese, English".to_string(),
            }),
            "plamo-100b" => Some(PLaMoModelInfo {
                name: model.to_string(),
                parameters: "100B".to_string(),
                context_length: 16384,
                language_focus: "Japanese, English".to_string(),
            }),
            _ => None,
        }
    }
}

/// PLaMo model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PLaMoModelInfo {
    pub name: String,
    pub parameters: String,
    pub context_length: u32,
    pub language_focus: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = PLaMoProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.preferrednetworks.ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = PLaMoProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let info = PLaMoProvider::get_model_info("plamo-100b").unwrap();
        assert_eq!(info.parameters, "100B");
    }
}
