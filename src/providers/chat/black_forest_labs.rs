//! Black Forest Labs provider for FLUX image generation.
//!
//! Black Forest Labs is the creator of FLUX, a state-of-the-art image generation model.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Black Forest Labs provider
pub struct BlackForestLabsProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl BlackForestLabsProvider {
    /// Create a new Black Forest Labs provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.bfl.ml/v1".to_string(),
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
        let api_key = std::env::var("BFL_API_KEY")
            .map_err(|_| Error::Configuration("BFL_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "flux-pro-1.1-ultra".to_string(),
            "flux-pro-1.1".to_string(),
            "flux-pro".to_string(),
            "flux-dev".to_string(),
            "flux-schnell".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<BlackForestLabsModelInfo> {
        match model {
            "flux-pro-1.1-ultra" => Some(BlackForestLabsModelInfo {
                name: model.to_string(),
                description: "Highest quality FLUX model".to_string(),
                max_resolution: "2048x2048".to_string(),
                features: vec![
                    "text-to-image".to_string(),
                    "ultra-quality".to_string(),
                    "raw-mode".to_string(),
                ],
                speed: "slow".to_string(),
                license: "commercial".to_string(),
            }),
            "flux-pro-1.1" => Some(BlackForestLabsModelInfo {
                name: model.to_string(),
                description: "Professional FLUX model".to_string(),
                max_resolution: "1536x1536".to_string(),
                features: vec![
                    "text-to-image".to_string(),
                    "image-to-image".to_string(),
                    "inpainting".to_string(),
                ],
                speed: "medium".to_string(),
                license: "commercial".to_string(),
            }),
            "flux-pro" => Some(BlackForestLabsModelInfo {
                name: model.to_string(),
                description: "Original professional model".to_string(),
                max_resolution: "1024x1024".to_string(),
                features: vec!["text-to-image".to_string()],
                speed: "medium".to_string(),
                license: "commercial".to_string(),
            }),
            "flux-dev" => Some(BlackForestLabsModelInfo {
                name: model.to_string(),
                description: "Development/research model".to_string(),
                max_resolution: "1024x1024".to_string(),
                features: vec!["text-to-image".to_string()],
                speed: "medium".to_string(),
                license: "non-commercial".to_string(),
            }),
            "flux-schnell" => Some(BlackForestLabsModelInfo {
                name: model.to_string(),
                description: "Fast generation model".to_string(),
                max_resolution: "1024x1024".to_string(),
                features: vec!["text-to-image".to_string(), "fast".to_string()],
                speed: "fast".to_string(),
                license: "apache-2.0".to_string(),
            }),
            _ => None,
        }
    }
}

/// Black Forest Labs model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackForestLabsModelInfo {
    pub name: String,
    pub description: String,
    pub max_resolution: String,
    pub features: Vec<String>,
    pub speed: String,
    pub license: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = BlackForestLabsProvider::new("test-key");
        assert!(provider.base_url.contains("bfl.ml"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = BlackForestLabsProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 4);
    }

    #[test]
    fn test_model_info() {
        let info = BlackForestLabsProvider::get_model_info("flux-schnell").unwrap();
        assert_eq!(info.speed, "fast");
    }
}
