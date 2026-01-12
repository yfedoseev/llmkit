//! Stability AI provider for image generation.
//!
//! Stability AI provides state-of-the-art image generation models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Stability AI provider
pub struct StabilityProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl StabilityProvider {
    /// Create a new Stability AI provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.stability.ai/v2beta".to_string(),
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
        let api_key = std::env::var("STABILITY_API_KEY").map_err(|_| {
            Error::Configuration("STABILITY_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "stable-diffusion-3.5-large".to_string(),
            "stable-diffusion-3.5-large-turbo".to_string(),
            "stable-diffusion-3.5-medium".to_string(),
            "stable-diffusion-3-medium".to_string(),
            "stable-image-ultra".to_string(),
            "stable-image-core".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<StabilityModelInfo> {
        match model {
            "stable-diffusion-3.5-large" => Some(StabilityModelInfo {
                name: model.to_string(),
                model_type: StabilityModelType::ImageGeneration,
                max_resolution: 1536,
                supports_inpainting: true,
            }),
            "stable-diffusion-3.5-large-turbo" => Some(StabilityModelInfo {
                name: model.to_string(),
                model_type: StabilityModelType::ImageGeneration,
                max_resolution: 1536,
                supports_inpainting: true,
            }),
            "stable-image-ultra" => Some(StabilityModelInfo {
                name: model.to_string(),
                model_type: StabilityModelType::ImageGeneration,
                max_resolution: 2048,
                supports_inpainting: false,
            }),
            _ => None,
        }
    }
}

/// Stability model type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StabilityModelType {
    ImageGeneration,
    ImageEditing,
    Upscaling,
}

/// Stability model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityModelInfo {
    pub name: String,
    pub model_type: StabilityModelType,
    pub max_resolution: u32,
    pub supports_inpainting: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = StabilityProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.stability.ai/v2beta");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = StabilityProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m.contains("stable-diffusion")));
    }

    #[test]
    fn test_get_model_info() {
        let info = StabilityProvider::get_model_info("stable-diffusion-3.5-large").unwrap();
        assert!(info.supports_inpainting);
    }
}
