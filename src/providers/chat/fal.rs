//! Fal.ai provider for fast image generation.
//!
//! Fal.ai provides serverless AI infrastructure with optimized model inference.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Fal.ai provider
pub struct FalProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl FalProvider {
    /// Create a new Fal provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://fal.run".to_string(),
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
        let api_key = std::env::var("FAL_API_KEY")
            .map_err(|_| Error::Configuration("FAL_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "fal-ai/flux-pro/v1.1-ultra".to_string(),
            "fal-ai/flux-pro".to_string(),
            "fal-ai/flux/dev".to_string(),
            "fal-ai/flux/schnell".to_string(),
            "fal-ai/flux-lora".to_string(),
            "fal-ai/stable-diffusion-v3".to_string(),
            "fal-ai/aura-flow".to_string(),
            "fal-ai/fast-sdxl".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<FalModelInfo> {
        match model {
            "fal-ai/flux-pro/v1.1-ultra" => Some(FalModelInfo {
                name: model.to_string(),
                description: "FLUX Pro 1.1 Ultra - Highest quality".to_string(),
                category: "image-generation".to_string(),
                inference_time_seconds: 10.0,
                features: vec![
                    "text-to-image".to_string(),
                    "raw-mode".to_string(),
                    "aspect-ratio".to_string(),
                ],
            }),
            "fal-ai/flux/schnell" => Some(FalModelInfo {
                name: model.to_string(),
                description: "FLUX Schnell - Fast generation".to_string(),
                category: "image-generation".to_string(),
                inference_time_seconds: 1.0,
                features: vec!["text-to-image".to_string(), "fast".to_string()],
            }),
            "fal-ai/flux-lora" => Some(FalModelInfo {
                name: model.to_string(),
                description: "FLUX with LoRA support".to_string(),
                category: "image-generation".to_string(),
                inference_time_seconds: 3.0,
                features: vec![
                    "text-to-image".to_string(),
                    "lora".to_string(),
                    "custom-styles".to_string(),
                ],
            }),
            "fal-ai/stable-diffusion-v3" => Some(FalModelInfo {
                name: model.to_string(),
                description: "Stable Diffusion 3".to_string(),
                category: "image-generation".to_string(),
                inference_time_seconds: 4.0,
                features: vec!["text-to-image".to_string(), "image-to-image".to_string()],
            }),
            "fal-ai/aura-flow" => Some(FalModelInfo {
                name: model.to_string(),
                description: "Aura Flow model".to_string(),
                category: "image-generation".to_string(),
                inference_time_seconds: 5.0,
                features: vec!["text-to-image".to_string()],
            }),
            "fal-ai/fast-sdxl" => Some(FalModelInfo {
                name: model.to_string(),
                description: "Fast SDXL".to_string(),
                category: "image-generation".to_string(),
                inference_time_seconds: 2.0,
                features: vec![
                    "text-to-image".to_string(),
                    "fast".to_string(),
                    "lcm".to_string(),
                ],
            }),
            _ => None,
        }
    }
}

/// Fal model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalModelInfo {
    pub name: String,
    pub description: String,
    pub category: String,
    pub inference_time_seconds: f64,
    pub features: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = FalProvider::new("test-key");
        assert_eq!(provider.base_url, "https://fal.run");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = FalProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 5);
    }

    #[test]
    fn test_model_info() {
        let info = FalProvider::get_model_info("fal-ai/flux/schnell").unwrap();
        assert!(info.features.contains(&"fast".to_string()));
    }
}
