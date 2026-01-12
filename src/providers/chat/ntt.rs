//! NTT provider for tsuzumi models.
//!
//! NTT's tsuzumi is a Japanese enterprise language model.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// NTT provider
pub struct NTTProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl NTTProvider {
    /// Create a new NTT provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.ntt.com/ai/v1".to_string(),
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
        let api_key = std::env::var("NTT_API_KEY")
            .map_err(|_| Error::Configuration("NTT_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "tsuzumi-7b".to_string(),
            "tsuzumi-7b-instruct".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<NTTModelInfo> {
        match model {
            "tsuzumi-7b" => Some(NTTModelInfo {
                name: model.to_string(),
                parameters: "7B".to_string(),
                context_length: 8192,
                language_focus: "Japanese, English".to_string(),
                use_case: "General purpose".to_string(),
            }),
            "tsuzumi-7b-instruct" => Some(NTTModelInfo {
                name: model.to_string(),
                parameters: "7B".to_string(),
                context_length: 8192,
                language_focus: "Japanese, English".to_string(),
                use_case: "Instruction following".to_string(),
            }),
            _ => None,
        }
    }
}

/// NTT model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NTTModelInfo {
    pub name: String,
    pub parameters: String,
    pub context_length: u32,
    pub language_focus: String,
    pub use_case: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = NTTProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.ntt.com/ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = NTTProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let info = NTTProvider::get_model_info("tsuzumi-7b").unwrap();
        assert_eq!(info.parameters, "7B");
    }
}
