//! ElevenLabs provider for text-to-speech and voice AI.
//!
//! ElevenLabs provides high-quality voice synthesis and cloning.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// ElevenLabs provider
pub struct ElevenLabsProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl ElevenLabsProvider {
    /// Create a new ElevenLabs provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.elevenlabs.io/v1".to_string(),
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
        let api_key = std::env::var("ELEVENLABS_API_KEY").map_err(|_| {
            Error::Configuration("ELEVENLABS_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "eleven_turbo_v2_5".to_string(),
            "eleven_turbo_v2".to_string(),
            "eleven_multilingual_v2".to_string(),
            "eleven_monolingual_v1".to_string(),
            "eleven_english_sts_v2".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<ElevenLabsModelInfo> {
        match model {
            "eleven_turbo_v2_5" => Some(ElevenLabsModelInfo {
                name: model.to_string(),
                model_type: ElevenLabsModelType::TextToSpeech,
                languages: 32,
                latency: ElevenLabsLatency::Low,
            }),
            "eleven_turbo_v2" => Some(ElevenLabsModelInfo {
                name: model.to_string(),
                model_type: ElevenLabsModelType::TextToSpeech,
                languages: 1,
                latency: ElevenLabsLatency::Low,
            }),
            "eleven_multilingual_v2" => Some(ElevenLabsModelInfo {
                name: model.to_string(),
                model_type: ElevenLabsModelType::TextToSpeech,
                languages: 29,
                latency: ElevenLabsLatency::Medium,
            }),
            "eleven_english_sts_v2" => Some(ElevenLabsModelInfo {
                name: model.to_string(),
                model_type: ElevenLabsModelType::SpeechToSpeech,
                languages: 1,
                latency: ElevenLabsLatency::Medium,
            }),
            _ => None,
        }
    }
}

/// ElevenLabs model type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElevenLabsModelType {
    TextToSpeech,
    SpeechToSpeech,
    VoiceCloning,
}

/// ElevenLabs latency category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElevenLabsLatency {
    Low,
    Medium,
    High,
}

/// ElevenLabs model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElevenLabsModelInfo {
    pub name: String,
    pub model_type: ElevenLabsModelType,
    pub languages: u32,
    pub latency: ElevenLabsLatency,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = ElevenLabsProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.elevenlabs.io/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = ElevenLabsProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_get_model_info() {
        let info = ElevenLabsProvider::get_model_info("eleven_turbo_v2_5").unwrap();
        assert_eq!(info.languages, 32);
    }
}
