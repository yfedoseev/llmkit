//! ElevenLabs API provider implementation.
//!
//! This module provides access to ElevenLabs' text-to-speech and voice cloning APIs.
//! ElevenLabs offers high-quality, natural-sounding voice synthesis.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::ElevenLabsProvider;
//!
//! // From environment variable
//! let provider = ElevenLabsProvider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = ElevenLabsProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Features
//!
//! - Text-to-speech synthesis
//! - Voice cloning
//! - Multiple voice options
//! - Streaming audio output
//!
//! # Environment Variables
//!
//! - `ELEVENLABS_API_KEY` - Your ElevenLabs API key

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

const ELEVENLABS_API_URL: &str = "https://api.elevenlabs.io/v1";

/// ElevenLabs API provider.
///
/// Provides access to ElevenLabs' text-to-speech capabilities.
/// This provider wraps TTS in a chat-like interface.
pub struct ElevenLabsProvider {
    config: ProviderConfig,
    client: Client,
}

impl ElevenLabsProvider {
    /// Create a new ElevenLabs provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.api_key {
            headers.insert(
                "xi-api-key",
                key.parse()
                    .map_err(|_| Error::config("Invalid API key format"))?,
            );
        }

        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        let client = Client::builder()
            .timeout(config.timeout)
            .default_headers(headers)
            .build()?;

        Ok(Self { config, client })
    }

    /// Create a new ElevenLabs provider from environment variable.
    ///
    /// Reads the API key from `ELEVENLABS_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("ELEVENLABS_API_KEY");
        Self::new(config)
    }

    /// Create a new ElevenLabs provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn tts_url(&self, voice_id: &str) -> String {
        format!(
            "{}/text-to-speech/{}",
            self.config
                .base_url
                .as_deref()
                .unwrap_or(ELEVENLABS_API_URL),
            voice_id
        )
    }

    fn voices_url(&self) -> String {
        format!(
            "{}/voices",
            self.config
                .base_url
                .as_deref()
                .unwrap_or(ELEVENLABS_API_URL)
        )
    }

    /// List available voices.
    pub async fn list_voices(&self) -> Result<Vec<Voice>> {
        let response = self.client.get(self.voices_url()).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("ElevenLabs API error {}: {}", status, error_text),
            ));
        }

        let api_response: VoicesResponse = response.json().await?;
        Ok(api_response.voices)
    }

    /// Synthesize speech from text.
    ///
    /// Returns the audio data as bytes (MP3 format by default).
    pub async fn synthesize(
        &self,
        text: &str,
        voice_id: &str,
        options: SynthesizeOptions,
    ) -> Result<Vec<u8>> {
        let request = ElevenLabsRequest {
            text: text.to_string(),
            model_id: options
                .model_id
                .unwrap_or_else(|| "eleven_monolingual_v1".to_string()),
            voice_settings: options.voice_settings,
        };

        let response = self
            .client
            .post(self.tts_url(voice_id))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("ElevenLabs API error {}: {}", status, error_text),
            ));
        }

        Ok(response.bytes().await?.to_vec())
    }
}

/// Options for speech synthesis.
#[derive(Debug, Default)]
pub struct SynthesizeOptions {
    /// Model to use (e.g., "eleven_monolingual_v1", "eleven_multilingual_v2")
    pub model_id: Option<String>,
    /// Voice settings
    pub voice_settings: Option<VoiceSettings>,
}

/// Voice settings for synthesis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceSettings {
    /// Stability (0.0 to 1.0)
    pub stability: f32,
    /// Similarity boost (0.0 to 1.0)
    pub similarity_boost: f32,
    /// Style (0.0 to 1.0, optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub style: Option<f32>,
    /// Use speaker boost
    #[serde(skip_serializing_if = "Option::is_none")]
    pub use_speaker_boost: Option<bool>,
}

/// Voice information.
#[derive(Debug, Clone, Deserialize)]
pub struct Voice {
    /// Voice ID
    pub voice_id: String,
    /// Voice name
    pub name: String,
    /// Voice category
    pub category: Option<String>,
    /// Voice description
    pub description: Option<String>,
    /// Voice labels
    pub labels: Option<std::collections::HashMap<String, String>>,
}

#[async_trait]
impl Provider for ElevenLabsProvider {
    fn name(&self) -> &str {
        "elevenlabs"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        // Extract text from the last user message
        let text = request
            .messages
            .iter()
            .rfind(|m| matches!(m.role, Role::User))
            .and_then(|m| {
                m.content.iter().find_map(|block| {
                    if let ContentBlock::Text { text } = block {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
            })
            .ok_or_else(|| Error::invalid_request("No text provided for synthesis"))?;

        // Use model field as voice_id, default to Rachel
        let voice_id = if request.model.is_empty() || request.model == "default" {
            "21m00Tcm4TlvDq8ikWAM" // Rachel
        } else {
            &request.model
        };

        let options = SynthesizeOptions {
            model_id: Some("eleven_monolingual_v1".to_string()),
            voice_settings: Some(VoiceSettings {
                stability: 0.5,
                similarity_boost: 0.75,
                style: None,
                use_speaker_boost: None,
            }),
        };

        let audio_data = self.synthesize(&text, voice_id, options).await?;

        // Return audio data as base64
        let base64_audio =
            base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &audio_data);

        Ok(CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: voice_id.to_string(),
            content: vec![ContentBlock::Text {
                text: format!("data:audio/mpeg;base64,{}", base64_audio),
            }],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: text.len() as u32,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        })
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // Fall back to regular completion
        let response = self.complete(request).await?;

        let stream = async_stream::try_stream! {
            yield StreamChunk {
                event_type: StreamEventType::ContentBlockStart,
                index: Some(0),
                delta: None,
                stop_reason: None,
                usage: None,
            };

            for block in response.content {
                if let ContentBlock::Text { text } = block {
                    yield StreamChunk {
                        event_type: StreamEventType::ContentBlockDelta,
                        index: Some(0),
                        delta: Some(ContentDelta::TextDelta { text }),
                        stop_reason: None,
                        usage: None,
                    };
                }
            }

            yield StreamChunk {
                event_type: StreamEventType::MessageStop,
                index: None,
                delta: None,
                stop_reason: Some(StopReason::EndTurn),
                usage: None,
            };
        };

        Ok(Box::pin(stream))
    }
}

// ElevenLabs API types

#[derive(Debug, Serialize)]
struct ElevenLabsRequest {
    text: String,
    model_id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    voice_settings: Option<VoiceSettings>,
}

#[derive(Debug, Deserialize)]
struct VoicesResponse {
    voices: Vec<Voice>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = ElevenLabsProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.name(), "elevenlabs");
    }

    #[test]
    fn test_provider_with_api_key() {
        let provider = ElevenLabsProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "elevenlabs");
    }

    #[test]
    fn test_tts_url() {
        let provider = ElevenLabsProvider::new(ProviderConfig::new("test-key")).unwrap();
        let url = provider.tts_url("voice-123");
        assert_eq!(url, "https://api.elevenlabs.io/v1/text-to-speech/voice-123");
    }

    #[test]
    fn test_tts_url_custom_base() {
        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.elevenlabs.io".to_string());
        let provider = ElevenLabsProvider::new(config).unwrap();
        let url = provider.tts_url("voice-123");
        assert_eq!(url, "https://custom.elevenlabs.io/text-to-speech/voice-123");
    }

    #[test]
    fn test_voices_url() {
        let provider = ElevenLabsProvider::new(ProviderConfig::new("test-key")).unwrap();
        let url = provider.voices_url();
        assert_eq!(url, "https://api.elevenlabs.io/v1/voices");
    }

    #[test]
    fn test_synthesize_options_default() {
        let options = SynthesizeOptions::default();
        assert!(options.model_id.is_none());
        assert!(options.voice_settings.is_none());
    }

    #[test]
    fn test_voice_settings_serialization() {
        let settings = VoiceSettings {
            stability: 0.5,
            similarity_boost: 0.75,
            style: Some(0.3),
            use_speaker_boost: Some(true),
        };

        let json = serde_json::to_string(&settings).unwrap();
        assert!(json.contains("0.5"));
        assert!(json.contains("0.75"));
        assert!(json.contains("0.3"));
        assert!(json.contains("true"));
    }

    #[test]
    fn test_voice_deserialization() {
        let json = r#"{
            "voice_id": "voice-123",
            "name": "Rachel",
            "category": "premade",
            "description": "A calm voice"
        }"#;

        let voice: Voice = serde_json::from_str(json).unwrap();
        assert_eq!(voice.voice_id, "voice-123");
        assert_eq!(voice.name, "Rachel");
        assert_eq!(voice.category, Some("premade".to_string()));
        assert_eq!(voice.description, Some("A calm voice".to_string()));
    }

    #[test]
    fn test_request_serialization() {
        let request = ElevenLabsRequest {
            text: "Hello world".to_string(),
            model_id: "eleven_monolingual_v1".to_string(),
            voice_settings: Some(VoiceSettings {
                stability: 0.5,
                similarity_boost: 0.75,
                style: None,
                use_speaker_boost: None,
            }),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("Hello world"));
        assert!(json.contains("eleven_monolingual_v1"));
        assert!(json.contains("0.5"));
    }

    #[test]
    fn test_voices_response_deserialization() {
        let json = r#"{
            "voices": [{
                "voice_id": "v1",
                "name": "Voice One"
            }, {
                "voice_id": "v2",
                "name": "Voice Two"
            }]
        }"#;

        let response: VoicesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.voices.len(), 2);
        assert_eq!(response.voices[0].voice_id, "v1");
        assert_eq!(response.voices[1].name, "Voice Two");
    }
}
