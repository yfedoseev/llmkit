//! Silo AI provider for Viking models.
//!
//! Silo AI's Viking is a Nordic-focused large language model.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Silo AI provider
pub struct SiloAIProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl SiloAIProvider {
    /// Create a new Silo AI provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.silo.ai/v1".to_string(),
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
        let api_key = std::env::var("SILOAI_API_KEY")
            .map_err(|_| Error::Configuration("SILOAI_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "Viking-33B".to_string(),
            "Viking-13B".to_string(),
            "Viking-7B".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<SiloAIModelInfo> {
        match model {
            "Viking-33B" => Some(SiloAIModelInfo {
                name: model.to_string(),
                parameters: "33B".to_string(),
                context_length: 32768,
                supported_languages: vec![
                    "Finnish".to_string(),
                    "Swedish".to_string(),
                    "Norwegian".to_string(),
                    "Danish".to_string(),
                    "Icelandic".to_string(),
                    "English".to_string(),
                ],
                gdpr_compliant: true,
            }),
            "Viking-13B" => Some(SiloAIModelInfo {
                name: model.to_string(),
                parameters: "13B".to_string(),
                context_length: 16384,
                supported_languages: vec![
                    "Finnish".to_string(),
                    "Swedish".to_string(),
                    "Norwegian".to_string(),
                    "Danish".to_string(),
                    "English".to_string(),
                ],
                gdpr_compliant: true,
            }),
            "Viking-7B" => Some(SiloAIModelInfo {
                name: model.to_string(),
                parameters: "7B".to_string(),
                context_length: 8192,
                supported_languages: vec![
                    "Finnish".to_string(),
                    "Swedish".to_string(),
                    "English".to_string(),
                ],
                gdpr_compliant: true,
            }),
            _ => None,
        }
    }
}

/// Silo AI model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiloAIModelInfo {
    pub name: String,
    pub parameters: String,
    pub context_length: u32,
    pub supported_languages: Vec<String>,
    pub gdpr_compliant: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = SiloAIProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.silo.ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = SiloAIProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 2);
    }

    #[test]
    fn test_model_info() {
        let info = SiloAIProvider::get_model_info("Viking-33B").unwrap();
        assert!(info.supported_languages.contains(&"Finnish".to_string()));
        assert!(info.gdpr_compliant);
    }
}
