//! Deepgram API provider implementation.
//!
//! This module provides access to Deepgram's speech-to-text and audio intelligence APIs.
//! Deepgram offers fast and accurate transcription with features like speaker diarization.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::DeepgramProvider;
//!
//! // From environment variable
//! let provider = DeepgramProvider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = DeepgramProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Features
//!
//! - Speech-to-text transcription
//! - Real-time streaming transcription
//! - Speaker diarization
//! - Language detection
//! - Sentiment analysis
//!
//! # Environment Variables
//!
//! - `DEEPGRAM_API_KEY` - Your Deepgram API key

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

const DEEPGRAM_API_URL_V1: &str = "https://api.deepgram.com/v1";
const DEEPGRAM_API_URL_V3: &str = "https://api.deepgram.com/v3";

/// Deepgram API version for selecting model features and endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeepgramVersion {
    /// API v1 (2023-12-01) - legacy support
    V1,
    /// API v3 (2025-01-01) - latest with Nova-3 models
    V3,
}

impl Default for DeepgramVersion {
    fn default() -> Self {
        Self::V1
    }
}

impl DeepgramVersion {
    /// Get the base API URL for this version.
    pub fn api_url(&self) -> &'static str {
        match self {
            Self::V1 => DEEPGRAM_API_URL_V1,
            Self::V3 => DEEPGRAM_API_URL_V3,
        }
    }

    /// Get the API version string for headers.
    pub fn version_header(&self) -> &'static str {
        match self {
            Self::V1 => "2023-12-01",
            Self::V3 => "2025-01-01",
        }
    }
}

/// Deepgram provider configuration.
#[derive(Debug, Clone)]
pub struct DeepgramConfig {
    /// Base provider configuration (API key, timeout, etc.)
    pub provider_config: ProviderConfig,
    /// API version to use
    pub version: DeepgramVersion,
}

impl DeepgramConfig {
    /// Create a new Deepgram config with the given API key and version.
    pub fn new(api_key: impl Into<String>, version: DeepgramVersion) -> Self {
        Self {
            provider_config: ProviderConfig::new(api_key),
            version,
        }
    }

    /// Create a Deepgram config from environment, using the specified version.
    pub fn from_env(version: DeepgramVersion) -> Self {
        Self {
            provider_config: ProviderConfig::from_env("DEEPGRAM_API_KEY"),
            version,
        }
    }
}

impl Default for DeepgramConfig {
    fn default() -> Self {
        Self {
            provider_config: ProviderConfig::default(),
            version: DeepgramVersion::V1,
        }
    }
}

/// Deepgram API provider.
///
/// Provides access to Deepgram's speech-to-text capabilities.
/// This provider wraps audio transcription in a chat-like interface.
pub struct DeepgramProvider {
    config: DeepgramConfig,
    client: Client,
}

impl DeepgramProvider {
    /// Create a new Deepgram provider with the given API version (V1 legacy).
    pub fn new(version: DeepgramVersion) -> Result<Self> {
        let config = DeepgramConfig::from_env(version);
        Self::with_config(config)
    }

    /// Create a new Deepgram provider from environment variable with default version (V1).
    ///
    /// Reads the API key from `DEEPGRAM_API_KEY`.
    pub fn from_env() -> Result<Self> {
        Self::new(DeepgramVersion::V1)
    }

    /// Create a new Deepgram provider with an API key and default version (V1).
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = DeepgramConfig::new(api_key, DeepgramVersion::V1);
        Self::with_config(config)
    }

    /// Create a new Deepgram provider with specific API key and version.
    pub fn with_api_key_and_version(
        api_key: impl Into<String>,
        version: DeepgramVersion,
    ) -> Result<Self> {
        let config = DeepgramConfig::new(api_key, version);
        Self::with_config(config)
    }

    /// Create a new Deepgram provider with custom configuration.
    pub fn with_config(config: DeepgramConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.provider_config.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Token {}", key)
                    .parse()
                    .map_err(|_| Error::config("Invalid API key format"))?,
            );
        }

        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        let client = Client::builder()
            .timeout(config.provider_config.timeout)
            .default_headers(headers)
            .build()?;

        Ok(Self { config, client })
    }

    /// Get the listen API endpoint URL for the configured version.
    fn listen_url(&self) -> String {
        format!(
            "{}/listen",
            self.config
                .provider_config
                .base_url
                .as_deref()
                .unwrap_or_else(|| { self.config.version.api_url() })
        )
    }

    /// Transcribe audio from a URL.
    pub async fn transcribe_url(
        &self,
        audio_url: &str,
        options: TranscribeOptions,
    ) -> Result<TranscribeResponse> {
        let request = DeepgramRequest {
            url: audio_url.to_string(),
        };

        let mut url = self.listen_url();
        let mut params = vec![];

        if let Some(model) = options.model {
            params.push(format!("model={}", model));
        }
        if options.smart_format {
            params.push("smart_format=true".to_string());
        }
        if options.diarize {
            params.push("diarize=true".to_string());
        }
        if let Some(language) = options.language {
            params.push(format!("language={}", language));
        }
        if options.punctuate {
            params.push("punctuate=true".to_string());
        }

        if !params.is_empty() {
            url = format!("{}?{}", url, params.join("&"));
        }

        let response = self.client.post(&url).json(&request).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Deepgram API error {}: {}", status, error_text),
            ));
        }

        let api_response: DeepgramResponse = response.json().await?;

        let transcript = api_response
            .results
            .channels
            .first()
            .and_then(|c| c.alternatives.first())
            .map(|a| a.transcript.clone())
            .unwrap_or_default();

        Ok(TranscribeResponse {
            transcript,
            confidence: api_response
                .results
                .channels
                .first()
                .and_then(|c| c.alternatives.first())
                .map(|a| a.confidence),
            words: api_response
                .results
                .channels
                .first()
                .and_then(|c| c.alternatives.first())
                .map(|a| a.words.clone())
                .unwrap_or_default(),
        })
    }
}

/// Options for transcription.
#[derive(Debug, Default)]
pub struct TranscribeOptions {
    /// Model to use (e.g., "nova-2", "whisper")
    pub model: Option<String>,
    /// Enable smart formatting
    pub smart_format: bool,
    /// Enable speaker diarization
    pub diarize: bool,
    /// Language code (e.g., "en", "es")
    pub language: Option<String>,
    /// Enable punctuation
    pub punctuate: bool,
}

/// Response from transcription.
#[derive(Debug)]
pub struct TranscribeResponse {
    /// Full transcript text
    pub transcript: String,
    /// Confidence score
    pub confidence: Option<f64>,
    /// Word-level details
    pub words: Vec<Word>,
}

/// Word in transcript with timing.
#[derive(Debug, Clone, Deserialize)]
pub struct Word {
    /// The word text
    pub word: String,
    /// Start time in seconds
    pub start: f64,
    /// End time in seconds
    pub end: f64,
    /// Confidence score
    pub confidence: f64,
}

#[async_trait]
impl Provider for DeepgramProvider {
    fn name(&self) -> &str {
        "deepgram"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        // Extract audio URL from the last user message
        let audio_url = request
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
            .ok_or_else(|| Error::invalid_request("No audio URL provided"))?;

        let options = TranscribeOptions {
            model: Some(request.model.clone()),
            smart_format: true,
            punctuate: true,
            ..Default::default()
        };

        let result = self.transcribe_url(&audio_url, options).await?;

        Ok(CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: request.model,
            content: vec![ContentBlock::Text {
                text: result.transcript,
            }],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 0,
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
        // Fall back to regular completion for now
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
                        delta: Some(ContentDelta::Text { text }),
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

// Deepgram API types

#[derive(Debug, Serialize)]
struct DeepgramRequest {
    url: String,
}

#[derive(Debug, Deserialize)]
struct DeepgramResponse {
    results: DeepgramResults,
}

#[derive(Debug, Deserialize)]
struct DeepgramResults {
    channels: Vec<DeepgramChannel>,
}

#[derive(Debug, Deserialize)]
struct DeepgramChannel {
    alternatives: Vec<DeepgramAlternative>,
}

#[derive(Debug, Deserialize)]
struct DeepgramAlternative {
    transcript: String,
    confidence: f64,
    #[serde(default)]
    words: Vec<Word>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = DeepgramProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "deepgram");
    }

    #[test]
    fn test_provider_with_api_key() {
        let provider = DeepgramProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "deepgram");
    }

    #[test]
    fn test_listen_url_v1() {
        let provider =
            DeepgramProvider::with_api_key_and_version("test-key", DeepgramVersion::V1).unwrap();
        assert_eq!(provider.listen_url(), "https://api.deepgram.com/v1/listen");
    }

    #[test]
    fn test_listen_url_v3() {
        let provider =
            DeepgramProvider::with_api_key_and_version("test-key", DeepgramVersion::V3).unwrap();
        assert_eq!(provider.listen_url(), "https://api.deepgram.com/v3/listen");
    }

    #[test]
    fn test_version_enum() {
        assert_eq!(DeepgramVersion::V1.api_url(), "https://api.deepgram.com/v1");
        assert_eq!(DeepgramVersion::V3.api_url(), "https://api.deepgram.com/v3");
        assert_eq!(DeepgramVersion::V1.version_header(), "2023-12-01");
        assert_eq!(DeepgramVersion::V3.version_header(), "2025-01-01");
        assert_eq!(DeepgramVersion::default(), DeepgramVersion::V1);
    }

    #[test]
    fn test_transcribe_options_default() {
        let options = TranscribeOptions::default();
        assert!(options.model.is_none());
        assert!(!options.smart_format);
        assert!(!options.diarize);
        assert!(options.language.is_none());
        assert!(!options.punctuate);
    }

    #[test]
    fn test_transcribe_options_with_values() {
        let options = TranscribeOptions {
            model: Some("nova-2".to_string()),
            smart_format: true,
            diarize: true,
            language: Some("en".to_string()),
            punctuate: true,
        };

        assert_eq!(options.model, Some("nova-2".to_string()));
        assert!(options.smart_format);
        assert!(options.diarize);
        assert_eq!(options.language, Some("en".to_string()));
        assert!(options.punctuate);
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "results": {
                "channels": [{
                    "alternatives": [{
                        "transcript": "Hello world",
                        "confidence": 0.95,
                        "words": [{
                            "word": "Hello",
                            "start": 0.0,
                            "end": 0.5,
                            "confidence": 0.98
                        }]
                    }]
                }]
            }
        }"#;

        let response: DeepgramResponse = serde_json::from_str(json).unwrap();
        let channel = &response.results.channels[0];
        let alternative = &channel.alternatives[0];

        assert_eq!(alternative.transcript, "Hello world");
        assert_eq!(alternative.confidence, 0.95);
        assert_eq!(alternative.words.len(), 1);
        assert_eq!(alternative.words[0].word, "Hello");
    }

    #[test]
    fn test_word_deserialization() {
        let json = r#"{
            "word": "test",
            "start": 1.5,
            "end": 2.0,
            "confidence": 0.99
        }"#;

        let word: Word = serde_json::from_str(json).unwrap();
        assert_eq!(word.word, "test");
        assert_eq!(word.start, 1.5);
        assert_eq!(word.end, 2.0);
        assert_eq!(word.confidence, 0.99);
    }

    #[test]
    fn test_request_serialization() {
        let request = DeepgramRequest {
            url: "https://example.com/audio.mp3".to_string(),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("https://example.com/audio.mp3"));
    }
}
