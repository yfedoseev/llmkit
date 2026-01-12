//! Play.ht provider for text-to-speech.
//!
//! Play.ht provides AI voice generation and text-to-speech services.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Play.ht provider
pub struct PlayHTProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    user_id: String,
    #[allow(dead_code)]
    base_url: String,
}

impl PlayHTProvider {
    /// Create a new PlayHT provider
    pub fn new(api_key: &str, user_id: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            user_id: user_id.to_string(),
            base_url: "https://api.play.ht/api/v2".to_string(),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(api_key: &str, user_id: &str, base_url: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            user_id: user_id.to_string(),
            base_url: base_url.to_string(),
        }
    }

    /// Create from environment variables
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("PLAYHT_API_KEY")
            .map_err(|_| Error::Configuration("PLAYHT_API_KEY not set".to_string()))?;
        let user_id = std::env::var("PLAYHT_USER_ID")
            .map_err(|_| Error::Configuration("PLAYHT_USER_ID not set".to_string()))?;
        Ok(Self::new(&api_key, &user_id))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "PlayHT2.0".to_string(),
            "PlayHT2.0-turbo".to_string(),
            "PlayHT1.0".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<PlayHTModelInfo> {
        match model {
            "PlayHT2.0" => Some(PlayHTModelInfo {
                name: model.to_string(),
                description: "High-quality voice synthesis".to_string(),
                voices: 800,
                languages: 142,
                features: vec![
                    "ultra-realistic".to_string(),
                    "emotion-control".to_string(),
                    "voice-cloning".to_string(),
                    "ssml".to_string(),
                ],
                latency_ms: 500,
            }),
            "PlayHT2.0-turbo" => Some(PlayHTModelInfo {
                name: model.to_string(),
                description: "Low-latency voice synthesis".to_string(),
                voices: 800,
                languages: 142,
                features: vec![
                    "low-latency".to_string(),
                    "streaming".to_string(),
                    "real-time".to_string(),
                ],
                latency_ms: 100,
            }),
            "PlayHT1.0" => Some(PlayHTModelInfo {
                name: model.to_string(),
                description: "Standard voice synthesis".to_string(),
                voices: 600,
                languages: 100,
                features: vec!["standard".to_string(), "ssml".to_string()],
                latency_ms: 1000,
            }),
            _ => None,
        }
    }
}

/// Play.ht model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlayHTModelInfo {
    pub name: String,
    pub description: String,
    pub voices: u32,
    pub languages: u32,
    pub features: Vec<String>,
    pub latency_ms: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = PlayHTProvider::new("test-key", "test-user");
        assert!(provider.base_url.contains("play.ht"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = PlayHTProvider::new("test-key", "test-user");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 2);
    }

    #[test]
    fn test_model_info() {
        let info = PlayHTProvider::get_model_info("PlayHT2.0").unwrap();
        assert_eq!(info.voices, 800);
    }
}
