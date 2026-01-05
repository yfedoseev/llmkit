//! Sarvam AI provider for Indic language models.
//!
//! Sarvam AI specializes in Indian language AI models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Sarvam AI provider
pub struct SarvamProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl SarvamProvider {
    /// Create a new Sarvam provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.sarvam.ai/v1".to_string(),
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
        let api_key = std::env::var("SARVAM_API_KEY")
            .map_err(|_| Error::Configuration("SARVAM_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "sarvam-2b".to_string(),
            "sarvam-7b".to_string(),
            "sarvam-translate".to_string(),
            "sarvam-tts".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<SarvamModelInfo> {
        match model {
            "sarvam-2b" => Some(SarvamModelInfo {
                name: model.to_string(),
                parameters: "2B".to_string(),
                context_length: 4096,
                supported_languages: vec![
                    "Hindi".to_string(),
                    "Tamil".to_string(),
                    "Telugu".to_string(),
                    "Bengali".to_string(),
                    "Marathi".to_string(),
                    "English".to_string(),
                ],
                modality: "text".to_string(),
            }),
            "sarvam-7b" => Some(SarvamModelInfo {
                name: model.to_string(),
                parameters: "7B".to_string(),
                context_length: 8192,
                supported_languages: vec![
                    "Hindi".to_string(),
                    "Tamil".to_string(),
                    "Telugu".to_string(),
                    "Bengali".to_string(),
                    "Marathi".to_string(),
                    "Gujarati".to_string(),
                    "Kannada".to_string(),
                    "Malayalam".to_string(),
                    "Punjabi".to_string(),
                    "English".to_string(),
                ],
                modality: "text".to_string(),
            }),
            "sarvam-translate" => Some(SarvamModelInfo {
                name: model.to_string(),
                parameters: "2B".to_string(),
                context_length: 2048,
                supported_languages: vec!["10+ Indian languages".to_string()],
                modality: "translation".to_string(),
            }),
            "sarvam-tts" => Some(SarvamModelInfo {
                name: model.to_string(),
                parameters: "N/A".to_string(),
                context_length: 1000,
                supported_languages: vec!["Hindi".to_string(), "English".to_string()],
                modality: "text-to-speech".to_string(),
            }),
            _ => None,
        }
    }
}

/// Sarvam model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SarvamModelInfo {
    pub name: String,
    pub parameters: String,
    pub context_length: u32,
    pub supported_languages: Vec<String>,
    pub modality: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = SarvamProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.sarvam.ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = SarvamProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 3);
    }

    #[test]
    fn test_model_info() {
        let info = SarvamProvider::get_model_info("sarvam-7b").unwrap();
        assert!(info.supported_languages.len() >= 5);
    }
}
