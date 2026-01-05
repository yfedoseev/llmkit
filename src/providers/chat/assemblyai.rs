//! AssemblyAI provider for speech-to-text and audio intelligence.
//!
//! AssemblyAI provides advanced speech recognition and audio analysis.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// AssemblyAI provider
pub struct AssemblyAIProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl AssemblyAIProvider {
    /// Create a new AssemblyAI provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.assemblyai.com/v2".to_string(),
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
        let api_key = std::env::var("ASSEMBLYAI_API_KEY")
            .map_err(|_| Error::Configuration("ASSEMBLYAI_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "best".to_string(),
            "nano".to_string(),
            "conformer-2".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<AssemblyAIModelInfo> {
        match model {
            "best" => Some(AssemblyAIModelInfo {
                name: model.to_string(),
                supports_realtime: true,
                supports_diarization: true,
                languages: vec![
                    "en".to_string(),
                    "es".to_string(),
                    "fr".to_string(),
                    "de".to_string(),
                ],
            }),
            "nano" => Some(AssemblyAIModelInfo {
                name: model.to_string(),
                supports_realtime: true,
                supports_diarization: false,
                languages: vec!["en".to_string()],
            }),
            "conformer-2" => Some(AssemblyAIModelInfo {
                name: model.to_string(),
                supports_realtime: true,
                supports_diarization: true,
                languages: vec!["en".to_string()],
            }),
            _ => None,
        }
    }
}

/// AssemblyAI model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssemblyAIModelInfo {
    pub name: String,
    pub supports_realtime: bool,
    pub supports_diarization: bool,
    pub languages: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = AssemblyAIProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.assemblyai.com/v2");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = AssemblyAIProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 3);
    }

    #[test]
    fn test_model_info() {
        let info = AssemblyAIProvider::get_model_info("best").unwrap();
        assert!(info.supports_realtime);
        assert!(info.supports_diarization);
    }
}
