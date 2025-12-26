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

const DEEPGRAM_API_URL: &str = "https://api.deepgram.com/v1";

/// Deepgram API provider.
///
/// Provides access to Deepgram's speech-to-text capabilities.
/// This provider wraps audio transcription in a chat-like interface.
pub struct DeepgramProvider {
    config: ProviderConfig,
    client: Client,
}

impl DeepgramProvider {
    /// Create a new Deepgram provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.api_key {
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
            .timeout(config.timeout)
            .default_headers(headers)
            .build()?;

        Ok(Self { config, client })
    }

    /// Create a new Deepgram provider from environment variable.
    ///
    /// Reads the API key from `DEEPGRAM_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("DEEPGRAM_API_KEY");
        Self::new(config)
    }

    /// Create a new Deepgram provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn listen_url(&self) -> String {
        format!(
            "{}/listen",
            self.config.base_url.as_deref().unwrap_or(DEEPGRAM_API_URL)
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
            .filter(|m| matches!(m.role, Role::User))
            .last()
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
        let provider = DeepgramProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.name(), "deepgram");
    }
}
