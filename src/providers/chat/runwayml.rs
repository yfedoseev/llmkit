//! RunwayML provider for video and image generation.
//!
//! RunwayML provides AI-powered creative tools for video and image generation.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// RunwayML provider
pub struct RunwayMLProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl RunwayMLProvider {
    /// Create a new RunwayML provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.runwayml.com/v1".to_string(),
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
        let api_key = std::env::var("RUNWAYML_API_KEY")
            .map_err(|_| Error::Configuration("RUNWAYML_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "gen-3-alpha".to_string(),
            "gen-2".to_string(),
            "gen-1".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<RunwayMLModelInfo> {
        match model {
            "gen-3-alpha" => Some(RunwayMLModelInfo {
                name: model.to_string(),
                modality: "video".to_string(),
                max_duration_seconds: 10,
                supports_image_to_video: true,
            }),
            "gen-2" => Some(RunwayMLModelInfo {
                name: model.to_string(),
                modality: "video".to_string(),
                max_duration_seconds: 4,
                supports_image_to_video: true,
            }),
            "gen-1" => Some(RunwayMLModelInfo {
                name: model.to_string(),
                modality: "video".to_string(),
                max_duration_seconds: 4,
                supports_image_to_video: false,
            }),
            _ => None,
        }
    }
}

/// RunwayML model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunwayMLModelInfo {
    pub name: String,
    pub modality: String,
    pub max_duration_seconds: u32,
    pub supports_image_to_video: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = RunwayMLProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.runwayml.com/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = RunwayMLProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 3);
    }

    #[test]
    fn test_model_info() {
        let info = RunwayMLProvider::get_model_info("gen-3-alpha").unwrap();
        assert_eq!(info.modality, "video");
        assert!(info.supports_image_to_video);
    }
}
