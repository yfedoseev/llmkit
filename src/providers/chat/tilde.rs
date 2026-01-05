//! Tilde AI provider.
//!
//! Tilde specializes in Baltic and European language AI models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Tilde AI provider
pub struct TildeProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl TildeProvider {
    /// Create a new Tilde provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.tilde.ai/v1".to_string(),
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
        let api_key = std::env::var("TILDE_API_KEY")
            .map_err(|_| Error::Configuration("TILDE_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "tilde-llm-lv".to_string(),
            "tilde-llm-lt".to_string(),
            "tilde-llm-et".to_string(),
            "tilde-translate".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<TildeModelInfo> {
        match model {
            "tilde-llm-lv" => Some(TildeModelInfo {
                name: model.to_string(),
                context_length: 8192,
                primary_language: "Latvian".to_string(),
                gdpr_compliant: true,
                capabilities: vec!["chat".to_string(), "completion".to_string()],
            }),
            "tilde-llm-lt" => Some(TildeModelInfo {
                name: model.to_string(),
                context_length: 8192,
                primary_language: "Lithuanian".to_string(),
                gdpr_compliant: true,
                capabilities: vec!["chat".to_string(), "completion".to_string()],
            }),
            "tilde-llm-et" => Some(TildeModelInfo {
                name: model.to_string(),
                context_length: 8192,
                primary_language: "Estonian".to_string(),
                gdpr_compliant: true,
                capabilities: vec!["chat".to_string(), "completion".to_string()],
            }),
            "tilde-translate" => Some(TildeModelInfo {
                name: model.to_string(),
                context_length: 4096,
                primary_language: "Multi-Baltic".to_string(),
                gdpr_compliant: true,
                capabilities: vec!["translation".to_string()],
            }),
            _ => None,
        }
    }
}

/// Tilde model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TildeModelInfo {
    pub name: String,
    pub context_length: u32,
    pub primary_language: String,
    pub gdpr_compliant: bool,
    pub capabilities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = TildeProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.tilde.ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = TildeProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 3);
    }

    #[test]
    fn test_model_info() {
        let info = TildeProvider::get_model_info("tilde-llm-lv").unwrap();
        assert_eq!(info.primary_language, "Latvian");
        assert!(info.gdpr_compliant);
    }
}
