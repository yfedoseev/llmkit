//! Deepgram provider for speech-to-text and audio intelligence.
//!
//! Deepgram provides speech recognition and audio transcription.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Deepgram provider
pub struct DeepgramProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl DeepgramProvider {
    /// Create a new Deepgram provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.deepgram.com/v1".to_string(),
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
        let api_key = std::env::var("DEEPGRAM_API_KEY").map_err(|_| {
            Error::Configuration("DEEPGRAM_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "nova-2".to_string(),
            "nova-2-general".to_string(),
            "nova-2-meeting".to_string(),
            "nova-2-phonecall".to_string(),
            "nova-2-voicemail".to_string(),
            "nova-2-finance".to_string(),
            "nova-2-medical".to_string(),
            "enhanced".to_string(),
            "base".to_string(),
            "whisper-large".to_string(),
            "whisper-medium".to_string(),
            "whisper-small".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<DeepgramModelInfo> {
        match model {
            m if m.starts_with("nova-2") => Some(DeepgramModelInfo {
                name: model.to_string(),
                model_type: DeepgramModelType::SpeechToText,
                languages: vec!["en".to_string(), "es".to_string(), "fr".to_string()],
                supports_diarization: true,
            }),
            m if m.starts_with("whisper") => Some(DeepgramModelInfo {
                name: model.to_string(),
                model_type: DeepgramModelType::SpeechToText,
                languages: vec!["multilingual".to_string()],
                supports_diarization: false,
            }),
            "enhanced" | "base" => Some(DeepgramModelInfo {
                name: model.to_string(),
                model_type: DeepgramModelType::SpeechToText,
                languages: vec!["en".to_string()],
                supports_diarization: true,
            }),
            _ => None,
        }
    }
}

/// Deepgram model type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeepgramModelType {
    SpeechToText,
    TextToSpeech,
    AudioIntelligence,
}

/// Deepgram model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeepgramModelInfo {
    pub name: String,
    pub model_type: DeepgramModelType,
    pub languages: Vec<String>,
    pub supports_diarization: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = DeepgramProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.deepgram.com/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = DeepgramProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.contains(&"nova-2".to_string()));
    }

    #[test]
    fn test_get_model_info() {
        let info = DeepgramProvider::get_model_info("nova-2").unwrap();
        assert!(info.supports_diarization);
    }
}
