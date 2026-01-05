//! Pika Labs provider for video generation.
//!
//! Pika provides AI-powered video generation from text and images.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Pika Labs provider
pub struct PikaProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl PikaProvider {
    /// Create a new Pika provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.pika.art/v1".to_string(),
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
        let api_key = std::env::var("PIKA_API_KEY")
            .map_err(|_| Error::Configuration("PIKA_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["pika-1.5".to_string(), "pika-1.0".to_string()])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<PikaModelInfo> {
        match model {
            "pika-1.5" => Some(PikaModelInfo {
                name: model.to_string(),
                modality: "video".to_string(),
                max_duration_seconds: 10,
                supported_inputs: vec!["text".to_string(), "image".to_string()],
                resolutions: vec!["1080p".to_string(), "720p".to_string()],
                features: vec![
                    "text-to-video".to_string(),
                    "image-to-video".to_string(),
                    "video-extend".to_string(),
                    "lip-sync".to_string(),
                ],
            }),
            "pika-1.0" => Some(PikaModelInfo {
                name: model.to_string(),
                modality: "video".to_string(),
                max_duration_seconds: 4,
                supported_inputs: vec!["text".to_string(), "image".to_string()],
                resolutions: vec!["720p".to_string()],
                features: vec!["text-to-video".to_string(), "image-to-video".to_string()],
            }),
            _ => None,
        }
    }
}

/// Pika model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PikaModelInfo {
    pub name: String,
    pub modality: String,
    pub max_duration_seconds: u32,
    pub supported_inputs: Vec<String>,
    pub resolutions: Vec<String>,
    pub features: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = PikaProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.pika.art/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = PikaProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let info = PikaProvider::get_model_info("pika-1.5").unwrap();
        assert_eq!(info.modality, "video");
    }
}
