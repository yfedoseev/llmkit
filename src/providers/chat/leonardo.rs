//! Leonardo.AI provider for image generation.
//!
//! Leonardo.AI provides high-quality AI image generation for games and creative work.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Leonardo.AI provider
pub struct LeonardoProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl LeonardoProvider {
    /// Create a new Leonardo provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://cloud.leonardo.ai/api/rest/v1".to_string(),
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
        let api_key = std::env::var("LEONARDO_API_KEY")
            .map_err(|_| Error::Configuration("LEONARDO_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "Leonardo Phoenix".to_string(),
            "Leonardo Diffusion XL".to_string(),
            "Leonardo Kino XL".to_string(),
            "Leonardo Vision XL".to_string(),
            "PhotoReal".to_string(),
            "Alchemy".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<LeonardoModelInfo> {
        match model {
            "Leonardo Phoenix" => Some(LeonardoModelInfo {
                name: model.to_string(),
                description: "Latest flagship model".to_string(),
                max_resolution: "2048x2048".to_string(),
                features: vec![
                    "text-to-image".to_string(),
                    "image-to-image".to_string(),
                    "inpainting".to_string(),
                    "prompt-enhance".to_string(),
                ],
                style: "versatile".to_string(),
            }),
            "Leonardo Diffusion XL" => Some(LeonardoModelInfo {
                name: model.to_string(),
                description: "High-quality general purpose".to_string(),
                max_resolution: "1536x1536".to_string(),
                features: vec!["text-to-image".to_string(), "controlnet".to_string()],
                style: "photorealistic".to_string(),
            }),
            "Leonardo Kino XL" => Some(LeonardoModelInfo {
                name: model.to_string(),
                description: "Cinematic style images".to_string(),
                max_resolution: "1536x1536".to_string(),
                features: vec!["text-to-image".to_string(), "cinematic-style".to_string()],
                style: "cinematic".to_string(),
            }),
            "PhotoReal" => Some(LeonardoModelInfo {
                name: model.to_string(),
                description: "Photorealistic image generation".to_string(),
                max_resolution: "1024x1024".to_string(),
                features: vec!["text-to-image".to_string(), "photorealistic".to_string()],
                style: "photorealistic".to_string(),
            }),
            _ => None,
        }
    }
}

/// Leonardo model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeonardoModelInfo {
    pub name: String,
    pub description: String,
    pub max_resolution: String,
    pub features: Vec<String>,
    pub style: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = LeonardoProvider::new("test-key");
        assert!(provider.base_url.contains("leonardo.ai"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = LeonardoProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 4);
    }

    #[test]
    fn test_model_info() {
        let info = LeonardoProvider::get_model_info("Leonardo Phoenix").unwrap();
        assert_eq!(info.style, "versatile");
    }
}
