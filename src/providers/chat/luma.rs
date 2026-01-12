//! Luma Labs provider for Dream Machine video generation.
//!
//! Luma's Dream Machine provides high-quality AI video generation.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Luma Labs provider
pub struct LumaProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl LumaProvider {
    /// Create a new Luma provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.lumalabs.ai/dream-machine/v1".to_string(),
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
        let api_key = std::env::var("LUMA_API_KEY")
            .map_err(|_| Error::Configuration("LUMA_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "dream-machine-1.5".to_string(),
            "dream-machine-1.0".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<LumaModelInfo> {
        match model {
            "dream-machine-1.5" => Some(LumaModelInfo {
                name: model.to_string(),
                modality: "video".to_string(),
                max_duration_seconds: 10,
                supported_inputs: vec!["text".to_string(), "image".to_string()],
                resolution: "1080p".to_string(),
                features: vec![
                    "text-to-video".to_string(),
                    "image-to-video".to_string(),
                    "extend".to_string(),
                    "camera-motion".to_string(),
                ],
            }),
            "dream-machine-1.0" => Some(LumaModelInfo {
                name: model.to_string(),
                modality: "video".to_string(),
                max_duration_seconds: 5,
                supported_inputs: vec!["text".to_string(), "image".to_string()],
                resolution: "720p".to_string(),
                features: vec!["text-to-video".to_string(), "image-to-video".to_string()],
            }),
            _ => None,
        }
    }
}

/// Luma model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LumaModelInfo {
    pub name: String,
    pub modality: String,
    pub max_duration_seconds: u32,
    pub supported_inputs: Vec<String>,
    pub resolution: String,
    pub features: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = LumaProvider::new("test-key");
        assert!(provider.base_url.contains("lumalabs.ai"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = LumaProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let info = LumaProvider::get_model_info("dream-machine-1.5").unwrap();
        assert_eq!(info.modality, "video");
    }
}
