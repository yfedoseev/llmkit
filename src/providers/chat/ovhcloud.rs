//! OVHCloud provider for EU-sovereign AI inference.
//!
//! OVHCloud provides EU-based AI inference with GDPR compliance.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// OVHCloud provider
pub struct OVHCloudProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl OVHCloudProvider {
    /// Create a new OVHCloud provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://llama-3-1-70b-instruct.endpoints.kepler.ai.cloud.ovh.net/api/openai_compat/v1".to_string(),
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
        let api_key = std::env::var("OVH_AI_ENDPOINTS_ACCESS_TOKEN").map_err(|_| {
            Error::Configuration("OVH_AI_ENDPOINTS_ACCESS_TOKEN not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "llama-3-1-70b-instruct".to_string(),
            "mistral-7b-instruct".to_string(),
            "mixtral-8x7b-instruct".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<OVHCloudModelInfo> {
        match model {
            m if m.contains("llama-3-1-70b") => Some(OVHCloudModelInfo {
                name: model.to_string(),
                context_window: 128000,
                region: "EU".to_string(),
                gdpr_compliant: true,
            }),
            m if m.contains("mistral-7b") => Some(OVHCloudModelInfo {
                name: model.to_string(),
                context_window: 32768,
                region: "EU".to_string(),
                gdpr_compliant: true,
            }),
            m if m.contains("mixtral") => Some(OVHCloudModelInfo {
                name: model.to_string(),
                context_window: 32768,
                region: "EU".to_string(),
                gdpr_compliant: true,
            }),
            _ => None,
        }
    }
}

/// OVHCloud model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OVHCloudModelInfo {
    pub name: String,
    pub context_window: u32,
    pub region: String,
    pub gdpr_compliant: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = OVHCloudProvider::new("test-key");
        assert!(provider.base_url.contains("ovh.net"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = OVHCloudProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 3);
    }

    #[test]
    fn test_model_info() {
        let info = OVHCloudProvider::get_model_info("llama-3-1-70b-instruct").unwrap();
        assert!(info.gdpr_compliant);
        assert_eq!(info.region, "EU");
    }
}
