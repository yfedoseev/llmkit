//! IONOS AI provider.
//!
//! IONOS offers European sovereign cloud AI services with GDPR compliance.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// IONOS AI provider
pub struct IONOSProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl IONOSProvider {
    /// Create a new IONOS provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.ionos.com/ai/v1".to_string(),
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
        let api_key = std::env::var("IONOS_API_KEY")
            .map_err(|_| Error::Configuration("IONOS_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "ionos-llama-3.1-70b".to_string(),
            "ionos-llama-3.1-8b".to_string(),
            "ionos-mistral-7b".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<IONOSModelInfo> {
        match model {
            "ionos-llama-3.1-70b" => Some(IONOSModelInfo {
                name: model.to_string(),
                context_length: 128000,
                data_residency: "EU".to_string(),
                gdpr_compliant: true,
                capabilities: vec!["chat".to_string(), "completion".to_string()],
            }),
            "ionos-llama-3.1-8b" => Some(IONOSModelInfo {
                name: model.to_string(),
                context_length: 128000,
                data_residency: "EU".to_string(),
                gdpr_compliant: true,
                capabilities: vec!["chat".to_string(), "fast-inference".to_string()],
            }),
            "ionos-mistral-7b" => Some(IONOSModelInfo {
                name: model.to_string(),
                context_length: 32768,
                data_residency: "EU".to_string(),
                gdpr_compliant: true,
                capabilities: vec!["chat".to_string()],
            }),
            _ => None,
        }
    }
}

/// IONOS model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IONOSModelInfo {
    pub name: String,
    pub context_length: u32,
    pub data_residency: String,
    pub gdpr_compliant: bool,
    pub capabilities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = IONOSProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.ionos.com/ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = IONOSProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 2);
    }

    #[test]
    fn test_model_info() {
        let info = IONOSProvider::get_model_info("ionos-llama-3.1-70b").unwrap();
        assert!(info.gdpr_compliant);
        assert_eq!(info.data_residency, "EU");
    }
}
