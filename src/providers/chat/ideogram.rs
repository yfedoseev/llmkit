//! Ideogram provider for AI image generation with text.
//!
//! Ideogram specializes in AI image generation with accurate text rendering.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Ideogram provider
pub struct IdeogramProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl IdeogramProvider {
    /// Create a new Ideogram provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.ideogram.ai".to_string(),
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
        let api_key = std::env::var("IDEOGRAM_API_KEY")
            .map_err(|_| Error::Configuration("IDEOGRAM_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "V_2".to_string(),
            "V_2_TURBO".to_string(),
            "V_1".to_string(),
            "V_1_TURBO".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<IdeogramModelInfo> {
        match model {
            "V_2" => Some(IdeogramModelInfo {
                name: model.to_string(),
                description: "Latest model with best text rendering".to_string(),
                resolutions: vec![
                    "RESOLUTION_1024_1024".to_string(),
                    "RESOLUTION_1280_720".to_string(),
                    "RESOLUTION_720_1280".to_string(),
                ],
                features: vec![
                    "text-in-image".to_string(),
                    "style-reference".to_string(),
                    "color-palette".to_string(),
                    "magic-prompt".to_string(),
                ],
                styles: vec![
                    "AUTO".to_string(),
                    "GENERAL".to_string(),
                    "REALISTIC".to_string(),
                    "DESIGN".to_string(),
                    "RENDER_3D".to_string(),
                    "ANIME".to_string(),
                ],
            }),
            "V_2_TURBO" => Some(IdeogramModelInfo {
                name: model.to_string(),
                description: "Fast generation with good quality".to_string(),
                resolutions: vec![
                    "RESOLUTION_1024_1024".to_string(),
                    "RESOLUTION_1280_720".to_string(),
                ],
                features: vec!["text-in-image".to_string(), "fast-generation".to_string()],
                styles: vec![
                    "AUTO".to_string(),
                    "GENERAL".to_string(),
                    "REALISTIC".to_string(),
                ],
            }),
            "V_1" => Some(IdeogramModelInfo {
                name: model.to_string(),
                description: "Original model".to_string(),
                resolutions: vec!["RESOLUTION_1024_1024".to_string()],
                features: vec!["text-in-image".to_string()],
                styles: vec!["AUTO".to_string(), "GENERAL".to_string()],
            }),
            _ => None,
        }
    }
}

/// Ideogram model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IdeogramModelInfo {
    pub name: String,
    pub description: String,
    pub resolutions: Vec<String>,
    pub features: Vec<String>,
    pub styles: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = IdeogramProvider::new("test-key");
        assert!(provider.base_url.contains("ideogram.ai"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = IdeogramProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 2);
    }

    #[test]
    fn test_model_info() {
        let info = IdeogramProvider::get_model_info("V_2").unwrap();
        assert!(info.features.contains(&"text-in-image".to_string()));
    }
}
