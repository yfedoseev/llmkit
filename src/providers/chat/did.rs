//! D-ID provider for digital human and talking head generation.
//!
//! D-ID creates AI-generated videos of talking digital humans.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// D-ID provider
pub struct DIDProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl DIDProvider {
    /// Create a new D-ID provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.d-id.com/v1".to_string(),
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
        let api_key = std::env::var("DID_API_KEY")
            .map_err(|_| Error::Configuration("DID_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available features
    pub async fn list_features(&self) -> Result<Vec<String>> {
        Ok(vec![
            "talks".to_string(),
            "clips".to_string(),
            "streams".to_string(),
            "agents".to_string(),
        ])
    }

    /// Get feature details
    pub fn get_feature_info(feature: &str) -> Option<DIDFeatureInfo> {
        match feature {
            "talks" => Some(DIDFeatureInfo {
                name: feature.to_string(),
                description: "Generate talking head videos".to_string(),
                inputs: vec![
                    "source_url".to_string(),
                    "script".to_string(),
                    "voice".to_string(),
                ],
                output: "video".to_string(),
                max_duration_seconds: 300,
            }),
            "clips" => Some(DIDFeatureInfo {
                name: feature.to_string(),
                description: "Create video clips with presenters".to_string(),
                inputs: vec!["presenter_id".to_string(), "script".to_string()],
                output: "video".to_string(),
                max_duration_seconds: 120,
            }),
            "streams" => Some(DIDFeatureInfo {
                name: feature.to_string(),
                description: "Real-time streaming digital humans".to_string(),
                inputs: vec!["presenter_id".to_string()],
                output: "stream".to_string(),
                max_duration_seconds: 0, // Real-time
            }),
            "agents" => Some(DIDFeatureInfo {
                name: feature.to_string(),
                description: "Interactive AI agents with digital humans".to_string(),
                inputs: vec!["agent_id".to_string(), "knowledge_base".to_string()],
                output: "interactive".to_string(),
                max_duration_seconds: 0, // Interactive
            }),
            _ => None,
        }
    }
}

/// D-ID feature information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DIDFeatureInfo {
    pub name: String,
    pub description: String,
    pub inputs: Vec<String>,
    pub output: String,
    pub max_duration_seconds: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = DIDProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.d-id.com/v1");
    }

    #[tokio::test]
    async fn test_list_features() {
        let provider = DIDProvider::new("test-key");
        let features = provider.list_features().await.unwrap();
        assert!(features.len() >= 3);
    }

    #[test]
    fn test_feature_info() {
        let info = DIDProvider::get_feature_info("talks").unwrap();
        assert_eq!(info.output, "video");
    }
}
