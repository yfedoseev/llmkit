//! Kling AI provider (Kuaishou) for video generation.
//!
//! Kling provides high-quality AI video generation from Kuaishou.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Kling AI provider
pub struct KlingProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl KlingProvider {
    /// Create a new Kling provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.kling.ai/v1".to_string(),
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
        let api_key = std::env::var("KLING_API_KEY")
            .map_err(|_| Error::Configuration("KLING_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "kling-1.5".to_string(),
            "kling-1.0".to_string(),
            "kling-pro".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<KlingModelInfo> {
        match model {
            "kling-1.5" => Some(KlingModelInfo {
                name: model.to_string(),
                modality: "video".to_string(),
                max_duration_seconds: 10,
                supported_inputs: vec!["text".to_string(), "image".to_string()],
                resolution: "1080p".to_string(),
                features: vec![
                    "text-to-video".to_string(),
                    "image-to-video".to_string(),
                    "motion-brush".to_string(),
                ],
            }),
            "kling-1.0" => Some(KlingModelInfo {
                name: model.to_string(),
                modality: "video".to_string(),
                max_duration_seconds: 5,
                supported_inputs: vec!["text".to_string(), "image".to_string()],
                resolution: "720p".to_string(),
                features: vec!["text-to-video".to_string(), "image-to-video".to_string()],
            }),
            "kling-pro" => Some(KlingModelInfo {
                name: model.to_string(),
                modality: "video".to_string(),
                max_duration_seconds: 60,
                supported_inputs: vec!["text".to_string(), "image".to_string()],
                resolution: "4K".to_string(),
                features: vec![
                    "text-to-video".to_string(),
                    "image-to-video".to_string(),
                    "long-form".to_string(),
                ],
            }),
            _ => None,
        }
    }
}

/// Kling model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KlingModelInfo {
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
        let provider = KlingProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.kling.ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = KlingProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 2);
    }

    #[test]
    fn test_model_info() {
        let info = KlingProvider::get_model_info("kling-pro").unwrap();
        assert_eq!(info.resolution, "4K");
    }
}
