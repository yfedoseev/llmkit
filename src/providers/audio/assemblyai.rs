//! AssemblyAI provider for advanced audio transcription services.
//!
//! AssemblyAI provides state-of-the-art speech-to-text transcription with
//! advanced features like speaker diarization, entity recognition, and
//! content moderation.
//!
//! # Features
//! - High-accuracy speech-to-text transcription
//! - Speaker diarization (multi-speaker detection)
//! - Entity recognition in transcripts
//! - Sentiment analysis
//! - Content moderation
//! - Support for multiple audio formats

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// AssemblyAI provider for audio transcription
pub struct AssemblyAIProvider {
    #[allow(dead_code)]
    api_key: String,
}

impl AssemblyAIProvider {
    /// Create a new AssemblyAI provider with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
        }
    }

    /// Create from environment variable `ASSEMBLYAI_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("ASSEMBLYAI_API_KEY").map_err(|_| {
            Error::Configuration("ASSEMBLYAI_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of supported audio formats
    pub async fn supported_formats(&self) -> Result<Vec<String>> {
        // Mock implementation
        Ok(vec![
            "wav".to_string(),
            "mp3".to_string(),
            "aac".to_string(),
            "flac".to_string(),
            "ogg".to_string(),
            "m4a".to_string(),
        ])
    }

    /// Get transcription configuration info
    pub fn get_config_info(config_type: &str) -> Option<TranscriptionConfig> {
        match config_type {
            "basic" => Some(TranscriptionConfig {
                config_name: "basic".to_string(),
                supports_diarization: false,
                supports_entity_detection: false,
                supports_sentiment_analysis: false,
                max_audio_duration_minutes: 180,
                accuracy_tier: "default".to_string(),
            }),
            "advanced" => Some(TranscriptionConfig {
                config_name: "advanced".to_string(),
                supports_diarization: true,
                supports_entity_detection: true,
                supports_sentiment_analysis: false,
                max_audio_duration_minutes: 360,
                accuracy_tier: "best".to_string(),
            }),
            "professional" => Some(TranscriptionConfig {
                config_name: "professional".to_string(),
                supports_diarization: true,
                supports_entity_detection: true,
                supports_sentiment_analysis: true,
                max_audio_duration_minutes: 720,
                accuracy_tier: "best".to_string(),
            }),
            _ => None,
        }
    }
}

/// Transcription configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptionConfig {
    /// Configuration name
    pub config_name: String,
    /// Whether speaker diarization is supported
    pub supports_diarization: bool,
    /// Whether entity detection is supported
    pub supports_entity_detection: bool,
    /// Whether sentiment analysis is supported
    pub supports_sentiment_analysis: bool,
    /// Maximum audio duration in minutes
    pub max_audio_duration_minutes: u32,
    /// Accuracy tier (default or best)
    pub accuracy_tier: String,
}

/// Audio language for transcription
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AudioLanguage {
    /// English
    #[default]
    English,
    /// Spanish
    Spanish,
    /// French
    French,
    /// German
    German,
    /// Chinese (Simplified)
    ChineseSimplified,
    /// Chinese (Traditional)
    ChineseTraditional,
    /// Japanese
    Japanese,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assemblyai_provider_creation() {
        let provider = AssemblyAIProvider::new("test-key");
        assert_eq!(provider.api_key, "test-key");
    }

    #[tokio::test]
    async fn test_supported_formats() {
        let provider = AssemblyAIProvider::new("test-key");
        let formats = provider.supported_formats().await.unwrap();
        assert!(!formats.is_empty());
        assert!(formats.contains(&"mp3".to_string()));
        assert!(formats.contains(&"wav".to_string()));
    }

    #[test]
    fn test_get_config_info() {
        let config = AssemblyAIProvider::get_config_info("basic").unwrap();
        assert_eq!(config.config_name, "basic");
        assert!(!config.supports_diarization);
        assert_eq!(config.max_audio_duration_minutes, 180);
    }

    #[test]
    fn test_advanced_config() {
        let config = AssemblyAIProvider::get_config_info("advanced").unwrap();
        assert_eq!(config.config_name, "advanced");
        assert!(config.supports_diarization);
        assert!(config.supports_entity_detection);
        assert!(!config.supports_sentiment_analysis);
    }

    #[test]
    fn test_professional_config() {
        let config = AssemblyAIProvider::get_config_info("professional").unwrap();
        assert_eq!(config.config_name, "professional");
        assert!(config.supports_diarization);
        assert!(config.supports_entity_detection);
        assert!(config.supports_sentiment_analysis);
        assert_eq!(config.accuracy_tier, "best");
    }

    #[test]
    fn test_config_info_invalid() {
        let config = AssemblyAIProvider::get_config_info("invalid-config");
        assert!(config.is_none());
    }

    #[test]
    fn test_audio_language_default() {
        assert_eq!(AudioLanguage::default(), AudioLanguage::English);
    }

    #[test]
    fn test_config_duration_tiers() {
        let basic = AssemblyAIProvider::get_config_info("basic").unwrap();
        let advanced = AssemblyAIProvider::get_config_info("advanced").unwrap();
        let professional = AssemblyAIProvider::get_config_info("professional").unwrap();

        assert!(basic.max_audio_duration_minutes < advanced.max_audio_duration_minutes);
        assert!(advanced.max_audio_duration_minutes < professional.max_audio_duration_minutes);
    }
}
