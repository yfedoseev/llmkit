//! Krutrim AI provider (Ola AI).
//!
//! Krutrim is Ola's AI platform for Indian language models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Krutrim AI provider
pub struct KrutrimProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl KrutrimProvider {
    /// Create a new Krutrim provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.olakrutrim.com/v1".to_string(),
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
        let api_key = std::env::var("KRUTRIM_API_KEY")
            .map_err(|_| Error::Configuration("KRUTRIM_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "Krutrim-spectre-v2".to_string(),
            "Krutrim-2".to_string(),
            "Krutrim-Pro".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<KrutrimModelInfo> {
        match model {
            "Krutrim-spectre-v2" => Some(KrutrimModelInfo {
                name: model.to_string(),
                context_length: 32768,
                supported_languages: vec![
                    "Hindi".to_string(),
                    "Tamil".to_string(),
                    "Telugu".to_string(),
                    "Kannada".to_string(),
                    "Malayalam".to_string(),
                    "Bengali".to_string(),
                    "Marathi".to_string(),
                    "Gujarati".to_string(),
                    "Odia".to_string(),
                    "Punjabi".to_string(),
                    "English".to_string(),
                ],
                capabilities: vec!["chat".to_string(), "completion".to_string()],
            }),
            "Krutrim-2" => Some(KrutrimModelInfo {
                name: model.to_string(),
                context_length: 16384,
                supported_languages: vec!["22 Indian languages".to_string(), "English".to_string()],
                capabilities: vec!["chat".to_string()],
            }),
            "Krutrim-Pro" => Some(KrutrimModelInfo {
                name: model.to_string(),
                context_length: 32768,
                supported_languages: vec!["22 Indian languages".to_string(), "English".to_string()],
                capabilities: vec![
                    "chat".to_string(),
                    "code".to_string(),
                    "reasoning".to_string(),
                ],
            }),
            _ => None,
        }
    }
}

/// Krutrim model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KrutrimModelInfo {
    pub name: String,
    pub context_length: u32,
    pub supported_languages: Vec<String>,
    pub capabilities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = KrutrimProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.olakrutrim.com/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = KrutrimProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 2);
    }

    #[test]
    fn test_model_info() {
        let info = KrutrimProvider::get_model_info("Krutrim-spectre-v2").unwrap();
        assert_eq!(info.context_length, 32768);
    }
}
