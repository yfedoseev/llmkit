//! Speechmatics provider for speech recognition.
//!
//! Speechmatics provides enterprise-grade speech recognition with global language support.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Speechmatics provider
pub struct SpeechmaticsProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl SpeechmaticsProvider {
    /// Create a new Speechmatics provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://asr.api.speechmatics.com/v2".to_string(),
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
        let api_key = std::env::var("SPEECHMATICS_API_KEY")
            .map_err(|_| Error::Configuration("SPEECHMATICS_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available services
    pub async fn list_services(&self) -> Result<Vec<String>> {
        Ok(vec![
            "batch".to_string(),
            "real-time".to_string(),
            "translation".to_string(),
        ])
    }

    /// Get service details
    pub fn get_service_info(service: &str) -> Option<SpeechmaticsServiceInfo> {
        match service {
            "batch" => Some(SpeechmaticsServiceInfo {
                name: service.to_string(),
                description: "Batch transcription".to_string(),
                languages: 50,
                features: vec![
                    "speaker-diarization".to_string(),
                    "entity-detection".to_string(),
                    "sentiment-analysis".to_string(),
                    "topic-detection".to_string(),
                    "summarization".to_string(),
                ],
                deployment_options: vec!["cloud".to_string(), "on-premise".to_string()],
            }),
            "real-time" => Some(SpeechmaticsServiceInfo {
                name: service.to_string(),
                description: "Real-time streaming transcription".to_string(),
                languages: 40,
                features: vec![
                    "low-latency".to_string(),
                    "speaker-diarization".to_string(),
                    "custom-dictionary".to_string(),
                ],
                deployment_options: vec!["cloud".to_string(), "on-premise".to_string()],
            }),
            "translation" => Some(SpeechmaticsServiceInfo {
                name: service.to_string(),
                description: "Speech translation".to_string(),
                languages: 30,
                features: vec!["speech-to-text".to_string(), "translation".to_string()],
                deployment_options: vec!["cloud".to_string()],
            }),
            _ => None,
        }
    }
}

/// Speechmatics service information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechmaticsServiceInfo {
    pub name: String,
    pub description: String,
    pub languages: u32,
    pub features: Vec<String>,
    pub deployment_options: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = SpeechmaticsProvider::new("test-key");
        assert!(provider.base_url.contains("speechmatics.com"));
    }

    #[tokio::test]
    async fn test_list_services() {
        let provider = SpeechmaticsProvider::new("test-key");
        let services = provider.list_services().await.unwrap();
        assert!(services.len() >= 2);
    }

    #[test]
    fn test_service_info() {
        let info = SpeechmaticsProvider::get_service_info("batch").unwrap();
        assert_eq!(info.languages, 50);
    }
}
