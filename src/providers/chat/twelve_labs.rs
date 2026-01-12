//! Twelve Labs provider for video understanding AI.
//!
//! Twelve Labs provides video understanding, search, and analysis.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Twelve Labs provider
pub struct TwelveLabsProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl TwelveLabsProvider {
    /// Create a new Twelve Labs provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.twelvelabs.io/v1.2".to_string(),
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
        let api_key = std::env::var("TWELVE_LABS_API_KEY")
            .map_err(|_| Error::Configuration("TWELVE_LABS_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "pegasus-1.1".to_string(),
            "marengo-2.6".to_string(),
            "pegasus-1".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<TwelveLabsModelInfo> {
        match model {
            "pegasus-1.1" => Some(TwelveLabsModelInfo {
                name: model.to_string(),
                modality: "video-understanding".to_string(),
                capabilities: vec![
                    "video-to-text".to_string(),
                    "summarization".to_string(),
                    "chapter-generation".to_string(),
                    "highlight-detection".to_string(),
                    "question-answering".to_string(),
                ],
                max_video_duration_minutes: 120,
                supported_formats: vec!["mp4".to_string(), "mov".to_string(), "avi".to_string()],
            }),
            "marengo-2.6" => Some(TwelveLabsModelInfo {
                name: model.to_string(),
                modality: "video-search".to_string(),
                capabilities: vec![
                    "semantic-search".to_string(),
                    "visual-search".to_string(),
                    "text-in-video".to_string(),
                    "object-detection".to_string(),
                ],
                max_video_duration_minutes: 60,
                supported_formats: vec!["mp4".to_string(), "mov".to_string()],
            }),
            "pegasus-1" => Some(TwelveLabsModelInfo {
                name: model.to_string(),
                modality: "video-understanding".to_string(),
                capabilities: vec![
                    "video-to-text".to_string(),
                    "summarization".to_string(),
                    "question-answering".to_string(),
                ],
                max_video_duration_minutes: 60,
                supported_formats: vec!["mp4".to_string(), "mov".to_string()],
            }),
            _ => None,
        }
    }
}

/// Twelve Labs model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwelveLabsModelInfo {
    pub name: String,
    pub modality: String,
    pub capabilities: Vec<String>,
    pub max_video_duration_minutes: u32,
    pub supported_formats: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = TwelveLabsProvider::new("test-key");
        assert!(provider.base_url.contains("twelvelabs.io"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = TwelveLabsProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 2);
    }

    #[test]
    fn test_model_info() {
        let info = TwelveLabsProvider::get_model_info("pegasus-1.1").unwrap();
        assert!(info.capabilities.contains(&"summarization".to_string()));
    }
}
