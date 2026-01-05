//! Clarifai provider for visual AI.
//!
//! Clarifai provides computer vision and visual recognition AI.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Clarifai provider
pub struct ClarifaiProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl ClarifaiProvider {
    /// Create a new Clarifai provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.clarifai.com/v2".to_string(),
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
        let api_key = std::env::var("CLARIFAI_API_KEY")
            .map_err(|_| Error::Configuration("CLARIFAI_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available capabilities
    pub async fn list_capabilities(&self) -> Result<Vec<String>> {
        Ok(vec![
            "image-recognition".to_string(),
            "object-detection".to_string(),
            "face-detection".to_string(),
            "visual-search".to_string(),
            "image-generation".to_string(),
            "moderation".to_string(),
        ])
    }

    /// Get capability details
    pub fn get_capability_info(capability: &str) -> Option<ClarifaiCapabilityInfo> {
        match capability {
            "image-recognition" => Some(ClarifaiCapabilityInfo {
                name: capability.to_string(),
                description: "Classify images with thousands of concepts".to_string(),
                models: vec![
                    "general-image-recognition".to_string(),
                    "food-item-recognition".to_string(),
                    "apparel-recognition".to_string(),
                ],
                use_cases: vec!["tagging".to_string(), "organization".to_string()],
            }),
            "object-detection" => Some(ClarifaiCapabilityInfo {
                name: capability.to_string(),
                description: "Detect and locate objects in images".to_string(),
                models: vec![
                    "general-detection".to_string(),
                    "face-detection".to_string(),
                    "logo-detection".to_string(),
                ],
                use_cases: vec!["counting".to_string(), "localization".to_string()],
            }),
            "face-detection" => Some(ClarifaiCapabilityInfo {
                name: capability.to_string(),
                description: "Detect and analyze faces".to_string(),
                models: vec![
                    "face-detection".to_string(),
                    "demographics".to_string(),
                    "face-embedding".to_string(),
                ],
                use_cases: vec!["identity".to_string(), "analytics".to_string()],
            }),
            "visual-search" => Some(ClarifaiCapabilityInfo {
                name: capability.to_string(),
                description: "Search by image similarity".to_string(),
                models: vec!["visual-embeddings".to_string()],
                use_cases: vec!["e-commerce".to_string(), "cataloging".to_string()],
            }),
            "image-generation" => Some(ClarifaiCapabilityInfo {
                name: capability.to_string(),
                description: "Generate images from text".to_string(),
                models: vec!["dall-e".to_string(), "stable-diffusion".to_string()],
                use_cases: vec!["content-creation".to_string(), "design".to_string()],
            }),
            _ => None,
        }
    }
}

/// Clarifai capability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClarifaiCapabilityInfo {
    pub name: String,
    pub description: String,
    pub models: Vec<String>,
    pub use_cases: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = ClarifaiProvider::new("test-key");
        assert!(provider.base_url.contains("clarifai.com"));
    }

    #[tokio::test]
    async fn test_list_capabilities() {
        let provider = ClarifaiProvider::new("test-key");
        let capabilities = provider.list_capabilities().await.unwrap();
        assert!(capabilities.len() >= 4);
    }

    #[test]
    fn test_capability_info() {
        let info = ClarifaiProvider::get_capability_info("image-recognition").unwrap();
        assert!(!info.models.is_empty());
    }
}
