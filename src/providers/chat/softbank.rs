//! SoftBank AI provider.
//!
//! SoftBank's AI platform for enterprise applications.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// SoftBank AI provider
pub struct SoftBankProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl SoftBankProvider {
    /// Create a new SoftBank provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.softbank.ai/v1".to_string(),
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
        let api_key = std::env::var("SOFTBANK_API_KEY")
            .map_err(|_| Error::Configuration("SOFTBANK_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["softbank-llm-jp".to_string()])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<SoftBankModelInfo> {
        match model {
            "softbank-llm-jp" => Some(SoftBankModelInfo {
                name: model.to_string(),
                context_length: 8192,
                language_focus: "Japanese, English".to_string(),
                deployment: "Enterprise".to_string(),
            }),
            _ => None,
        }
    }
}

/// SoftBank model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoftBankModelInfo {
    pub name: String,
    pub context_length: u32,
    pub language_focus: String,
    pub deployment: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = SoftBankProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.softbank.ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = SoftBankProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let info = SoftBankProvider::get_model_info("softbank-llm-jp").unwrap();
        assert!(info.language_focus.contains("Japanese"));
    }
}
