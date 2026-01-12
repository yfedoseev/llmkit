//! Rev AI provider for speech-to-text and transcription.
//!
//! Rev provides professional speech-to-text and transcription services.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Rev AI provider
pub struct RevProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl RevProvider {
    /// Create a new Rev provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.rev.ai/speechtotext/v1".to_string(),
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
        let api_key = std::env::var("REV_API_KEY")
            .map_err(|_| Error::Configuration("REV_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available services
    pub async fn list_services(&self) -> Result<Vec<String>> {
        Ok(vec![
            "async-transcription".to_string(),
            "streaming-transcription".to_string(),
            "human-transcription".to_string(),
            "captions".to_string(),
        ])
    }

    /// Get service details
    pub fn get_service_info(service: &str) -> Option<RevServiceInfo> {
        match service {
            "async-transcription" => Some(RevServiceInfo {
                name: service.to_string(),
                description: "Asynchronous speech-to-text".to_string(),
                accuracy: "High".to_string(),
                languages: 36,
                features: vec![
                    "speaker-diarization".to_string(),
                    "punctuation".to_string(),
                    "custom-vocabulary".to_string(),
                    "timestamps".to_string(),
                ],
                price_per_minute: 0.02,
            }),
            "streaming-transcription" => Some(RevServiceInfo {
                name: service.to_string(),
                description: "Real-time streaming transcription".to_string(),
                accuracy: "High".to_string(),
                languages: 10,
                features: vec![
                    "real-time".to_string(),
                    "low-latency".to_string(),
                    "interim-results".to_string(),
                ],
                price_per_minute: 0.035,
            }),
            "human-transcription" => Some(RevServiceInfo {
                name: service.to_string(),
                description: "Human-powered transcription".to_string(),
                accuracy: "99%+".to_string(),
                languages: 15,
                features: vec![
                    "human-reviewed".to_string(),
                    "high-accuracy".to_string(),
                    "timestamps".to_string(),
                ],
                price_per_minute: 1.50,
            }),
            "captions" => Some(RevServiceInfo {
                name: service.to_string(),
                description: "Closed captions and subtitles".to_string(),
                accuracy: "High".to_string(),
                languages: 15,
                features: vec!["srt".to_string(), "vtt".to_string(), "styling".to_string()],
                price_per_minute: 1.25,
            }),
            _ => None,
        }
    }
}

/// Rev service information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RevServiceInfo {
    pub name: String,
    pub description: String,
    pub accuracy: String,
    pub languages: u32,
    pub features: Vec<String>,
    pub price_per_minute: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = RevProvider::new("test-key");
        assert!(provider.base_url.contains("rev.ai"));
    }

    #[tokio::test]
    async fn test_list_services() {
        let provider = RevProvider::new("test-key");
        let services = provider.list_services().await.unwrap();
        assert!(services.len() >= 3);
    }

    #[test]
    fn test_service_info() {
        let info = RevProvider::get_service_info("async-transcription").unwrap();
        assert_eq!(info.languages, 36);
    }
}
