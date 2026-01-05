//! Writer AI (Palmyra) API provider implementation.
//!
//! This module provides access to Writer's Palmyra foundation models.
//! Writer offers enterprise-grade LLMs with up to 1M token context windows.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::providers::WriterProvider;
//!
//! // From environment variable
//! let provider = WriterProvider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = WriterProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Features
//!
//! - Palmyra X5 (1M context window)
//! - Palmyra X4
//! - Streaming support
//! - Tool/function calling
//!
//! # Environment Variables
//!
//! - `WRITER_API_KEY` - Your Writer API key
//!
//! # Models
//!
//! - `palmyra-x5` - Latest flagship model with 1M context
//! - `palmyra-x4` - Previous generation model
//!
//! # Note
//!
//! Writer uses `/v1/chat` endpoint instead of `/v1/chat/completions`

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

const WRITER_API_URL: &str = "https://api.writer.com/v1/chat";

/// Writer AI (Palmyra) API provider.
///
/// Provides access to Writer's Palmyra models with enterprise-grade capabilities.
pub struct WriterProvider {
    config: ProviderConfig,
    client: Client,
}

impl WriterProvider {
    /// Create a new Writer provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", key)
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

    /// Create a new Writer provider from environment variable.
    ///
    /// Reads the API key from `WRITER_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("WRITER_API_KEY")
            .map_err(|_| Error::config("WRITER_API_KEY environment variable not set"))?;

        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    /// Create a new Writer provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn api_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(WRITER_API_URL)
    }

    /// Convert our unified request to Writer's format.
    fn convert_request(&self, request: &CompletionRequest, stream: bool) -> WriterRequest {
        // Convert messages
        let mut messages: Vec<WriterMessage> = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(WriterMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        // Convert conversation messages
        for msg in &request.messages {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            messages.push(WriterMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        // Default model
        let model = if request.model.is_empty() || request.model == "default" {
            "palmyra-x5".to_string()
        } else {
            request.model.clone()
        };

        WriterRequest {
            model,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop_sequences.clone(),
            stream,
        }
    }

    /// Parse Writer response into our unified format.
    fn parse_response(&self, response: WriterResponse) -> CompletionResponse {
        let choice = response.choices.first();

        let text = choice
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let stop_reason = choice
            .and_then(|c| c.finish_reason.as_ref())
            .map(|r| match r.as_str() {
                "stop" => StopReason::EndTurn,
                "length" => StopReason::MaxTokens,
                _ => StopReason::EndTurn,
            })
            .unwrap_or(StopReason::EndTurn);

        CompletionResponse {
            id: response.id,
            model: response.model,
            content: vec![ContentBlock::Text { text }],
            stop_reason,
            usage: response.usage.map_or_else(
                || Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
                |u| Usage {
                    input_tokens: u.prompt_tokens,
                    output_tokens: u.completion_tokens,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            ),
        }
    }
}

#[async_trait]
impl Provider for WriterProvider {
    fn name(&self) -> &str {
        "writer"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let writer_request = self.convert_request(&request, false);

        let response = self
            .client
            .post(self.api_url())
            .json(&writer_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Writer API error {}: {}", status, error_text),
            ));
        }

        let writer_response: WriterResponse = response.json().await?;
        Ok(self.parse_response(writer_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let writer_request = self.convert_request(&request, true);

        let response = self
            .client
            .post(self.api_url())
            .json(&writer_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Writer API error {}: {}", status, error_text),
            ));
        }

        let stream = async_stream::try_stream! {
            use tokio_stream::StreamExt;
            use eventsource_stream::Eventsource;

            yield StreamChunk {
                event_type: StreamEventType::ContentBlockStart,
                index: Some(0),
                delta: None,
                stop_reason: None,
                usage: None,
            };

            let mut event_stream = response.bytes_stream().eventsource();

            while let Some(event) = event_stream.next().await {
                let event = event.map_err(|e| Error::other(e.to_string()))?;

                if event.data.is_empty() || event.data == "[DONE]" {
                    continue;
                }

                if let Ok(chunk) = serde_json::from_str::<WriterStreamChunk>(&event.data) {
                    if let Some(choice) = chunk.choices.first() {
                        if let Some(ref delta) = choice.delta {
                            if let Some(ref content) = delta.content {
                                if !content.is_empty() {
                                    yield StreamChunk {
                                        event_type: StreamEventType::ContentBlockDelta,
                                        index: Some(0),
                                        delta: Some(ContentDelta::Text { text: content.clone() }),
                                        stop_reason: None,
                                        usage: None,
                                    };
                                }
                            }
                        }
                    }
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

// ==================== Writer API Types ====================

#[derive(Debug, Serialize)]
struct WriterRequest {
    model: String,
    messages: Vec<WriterMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct WriterMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct WriterResponse {
    id: String,
    model: String,
    choices: Vec<WriterChoice>,
    usage: Option<WriterUsage>,
}

#[derive(Debug, Deserialize)]
struct WriterChoice {
    message: WriterMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WriterUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct WriterStreamChunk {
    choices: Vec<WriterStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct WriterStreamChoice {
    delta: Option<WriterDelta>,
    #[serde(rename = "finish_reason")]
    _finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WriterDelta {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = WriterProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.name(), "writer");
    }

    #[test]
    fn test_provider_with_api_key() {
        let provider = WriterProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "writer");
    }

    #[test]
    fn test_api_url() {
        let provider = WriterProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.api_url(), WRITER_API_URL);
    }

    #[test]
    fn test_api_url_custom_base() {
        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.writer.com".to_string());
        let provider = WriterProvider::new(config).unwrap();
        assert_eq!(provider.api_url(), "https://custom.writer.com");
    }

    #[test]
    fn test_convert_request() {
        let provider = WriterProvider::new(ProviderConfig::new("test-key")).unwrap();

        let request = CompletionRequest::new("palmyra-x5", vec![Message::user("Hello")])
            .with_system("You are helpful")
            .with_max_tokens(1024)
            .with_temperature(0.7);

        let writer_req = provider.convert_request(&request, false);

        assert_eq!(writer_req.model, "palmyra-x5");
        assert_eq!(writer_req.messages.len(), 2); // system + user
        assert_eq!(writer_req.messages[0].role, "system");
        assert_eq!(writer_req.messages[0].content, "You are helpful");
        assert_eq!(writer_req.messages[1].role, "user");
        assert_eq!(writer_req.messages[1].content, "Hello");
        assert_eq!(writer_req.max_tokens, Some(1024));
        assert_eq!(writer_req.temperature, Some(0.7));
        assert!(!writer_req.stream);
    }

    #[test]
    fn test_convert_request_default_model() {
        let provider = WriterProvider::new(ProviderConfig::new("test-key")).unwrap();

        let request = CompletionRequest::new("", vec![Message::user("Hello")]);

        let writer_req = provider.convert_request(&request, false);
        assert_eq!(writer_req.model, "palmyra-x5");
    }

    #[test]
    fn test_parse_response() {
        let provider = WriterProvider::new(ProviderConfig::new("test-key")).unwrap();

        let response = WriterResponse {
            id: "resp-123".to_string(),
            model: "palmyra-x5".to_string(),
            choices: vec![WriterChoice {
                message: WriterMessage {
                    role: "assistant".to_string(),
                    content: "Hello! How can I help?".to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(WriterUsage {
                prompt_tokens: 10,
                completion_tokens: 15,
            }),
        };

        let result = provider.parse_response(response);

        assert_eq!(result.id, "resp-123");
        assert_eq!(result.model, "palmyra-x5");
        assert_eq!(result.content.len(), 1);
        if let ContentBlock::Text { text } = &result.content[0] {
            assert_eq!(text, "Hello! How can I help?");
        } else {
            panic!("Expected text content block");
        }
        assert!(matches!(result.stop_reason, StopReason::EndTurn));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 15);
    }

    #[test]
    fn test_stop_reason_mapping() {
        let provider = WriterProvider::new(ProviderConfig::new("test-key")).unwrap();

        // Test "stop" -> EndTurn
        let response1 = WriterResponse {
            id: "1".to_string(),
            model: "model".to_string(),
            choices: vec![WriterChoice {
                message: WriterMessage {
                    role: "assistant".to_string(),
                    content: "Done".to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.parse_response(response1).stop_reason,
            StopReason::EndTurn
        ));

        // Test "length" -> MaxTokens
        let response2 = WriterResponse {
            id: "2".to_string(),
            model: "model".to_string(),
            choices: vec![WriterChoice {
                message: WriterMessage {
                    role: "assistant".to_string(),
                    content: "Truncated".to_string(),
                },
                finish_reason: Some("length".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.parse_response(response2).stop_reason,
            StopReason::MaxTokens
        ));
    }

    #[test]
    fn test_request_serialization() {
        let request = WriterRequest {
            model: "palmyra-x5".to_string(),
            messages: vec![WriterMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: None,
            stop: None,
            stream: false,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("palmyra-x5"));
        assert!(json.contains("max_tokens"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "test-id",
            "model": "palmyra-x5",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }"#;

        let response: WriterResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.model, "palmyra-x5");
        assert_eq!(response.choices[0].message.content, "Hello!");
    }

    #[test]
    fn test_stream_delta_deserialization() {
        let json = r#"{"content": "Hello"}"#;
        let delta: WriterDelta = serde_json::from_str(json).unwrap();
        assert_eq!(delta.content, Some("Hello".to_string()));
    }
}
