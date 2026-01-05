//! Runware provider for fast image generation.
//!
//! Runware provides ultra-fast image generation API.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Runware provider
pub struct RunwareProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl RunwareProvider {
    /// Create a new Runware provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.runware.ai/v1".to_string(),
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
        let api_key = std::env::var("RUNWARE_API_KEY")
            .map_err(|_| Error::Configuration("RUNWARE_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "runware-flux".to_string(),
            "runware-sdxl".to_string(),
            "runware-sd3".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<RunwareModelInfo> {
        match model {
            "runware-flux" => Some(RunwareModelInfo {
                name: model.to_string(),
                base_model: "FLUX".to_string(),
                inference_time_ms: 500,
                features: vec![
                    "text-to-image".to_string(),
                    "image-to-image".to_string(),
                    "inpainting".to_string(),
                ],
                price_per_image: 0.001,
            }),
            "runware-sdxl" => Some(RunwareModelInfo {
                name: model.to_string(),
                base_model: "SDXL".to_string(),
                inference_time_ms: 800,
                features: vec![
                    "text-to-image".to_string(),
                    "image-to-image".to_string(),
                    "controlnet".to_string(),
                ],
                price_per_image: 0.0008,
            }),
            "runware-sd3" => Some(RunwareModelInfo {
                name: model.to_string(),
                base_model: "SD3".to_string(),
                inference_time_ms: 1200,
                features: vec!["text-to-image".to_string(), "image-to-image".to_string()],
                price_per_image: 0.0012,
            }),
            _ => None,
        }
    }
}

/// Runware model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunwareModelInfo {
    pub name: String,
    pub base_model: String,
    pub inference_time_ms: u32,
    pub features: Vec<String>,
    pub price_per_image: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = RunwareProvider::new("test-key");
        assert!(provider.base_url.contains("runware"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = RunwareProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 2);
    }

    #[test]
    fn test_model_info() {
        let info = RunwareProvider::get_model_info("runware-flux").unwrap();
        assert!(info.inference_time_ms < 1000);
    }
}
