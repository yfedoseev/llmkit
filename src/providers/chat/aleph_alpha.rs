//! Aleph Alpha API provider implementation.
//!
//! This module provides access to Aleph Alpha's Luminous models for chat completions.
//! Aleph Alpha is a European AI company offering models with strong multilingual capabilities.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::providers::AlephAlphaProvider;
//!
//! // From environment variable
//! let provider = AlephAlphaProvider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = AlephAlphaProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! - `luminous-supreme` - Most capable Luminous model
//! - `luminous-extended` - Extended context model
//! - `luminous-base` - Base model
//! - `llama-3.1-70b-instruct` - Llama 3.1 70B via Aleph Alpha
//!
//! # Environment Variables
//!
//! - `ALEPH_ALPHA_API_KEY` - Your Aleph Alpha API key

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Message, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

const ALEPH_ALPHA_API_URL: &str = "https://api.aleph-alpha.com/chat/completions";

/// Aleph Alpha API provider.
///
/// Provides access to Aleph Alpha's Luminous family of models.
pub struct AlephAlphaProvider {
    config: ProviderConfig,
    client: Client,
}

impl AlephAlphaProvider {
    /// Create a new Aleph Alpha provider with the given configuration.
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

    /// Create a new Aleph Alpha provider from environment variable.
    ///
    /// Reads the API key from `ALEPH_ALPHA_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("ALEPH_ALPHA_API_KEY");
        Self::new(config)
    }

    /// Create a new Aleph Alpha provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn api_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or(ALEPH_ALPHA_API_URL)
    }

    /// Convert our unified request to Aleph Alpha's format.
    fn convert_request(&self, request: &CompletionRequest) -> AlephAlphaRequest {
        let mut messages: Vec<AlephAlphaMessage> = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(AlephAlphaMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        // Convert messages
        for msg in &request.messages {
            messages.push(self.convert_message(msg));
        }

        AlephAlphaRequest {
            model: request.model.clone(),
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: Some(false),
            stop: request.stop_sequences.clone(),
        }
    }

    fn convert_message(&self, msg: &Message) -> AlephAlphaMessage {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        };

        // Extract text content
        let content = msg
            .content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::Text { text } = block {
                    Some(text.clone())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        AlephAlphaMessage {
            role: role.to_string(),
            content,
        }
    }

    fn convert_response(&self, response: AlephAlphaResponse) -> Result<CompletionResponse> {
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| Error::invalid_request("No choices in response"))?;

        let content = vec![ContentBlock::Text {
            text: choice.message.content,
        }];

        let stop_reason = match choice.finish_reason.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        let usage = response
            .usage
            .map(|u| Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            })
            .unwrap_or(Usage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            });

        Ok(CompletionResponse {
            id: response.id,
            model: response.model,
            content,
            stop_reason,
            usage,
        })
    }
}

#[async_trait]
impl Provider for AlephAlphaProvider {
    fn name(&self) -> &str {
        "aleph-alpha"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let api_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Aleph Alpha API error {}: {}", status, error_text),
            ));
        }

        let api_response: AlephAlphaResponse = response.json().await?;
        self.convert_response(api_response)
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        use async_stream::try_stream;
        use futures::StreamExt;

        let mut api_request = self.convert_request(&request);
        api_request.stream = Some(true);

        let response = self
            .client
            .post(self.api_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Aleph Alpha API error {}: {}", status, error_text),
            ));
        }

        let stream = try_stream! {
            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();

            while let Some(chunk_result) = byte_stream.next().await {
                let chunk = chunk_result?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(line_end) = buffer.find('\n') {
                    let line = buffer[..line_end].trim().to_string();
                    buffer = buffer[line_end + 1..].to_string();

                    if line.is_empty() || !line.starts_with("data: ") {
                        continue;
                    }

                    let data = &line[6..];
                    if data == "[DONE]" {
                        yield StreamChunk {
                            event_type: StreamEventType::MessageStop,
                            index: None,
                            delta: None,
                            stop_reason: Some(StopReason::EndTurn),
                            usage: None,
                        };
                        return;
                    }

                    if let Ok(event) = serde_json::from_str::<AlephAlphaStreamEvent>(data) {
                        if let Some(choice) = event.choices.first() {
                            if let Some(ref delta) = choice.delta {
                                if let Some(ref content) = delta.content {
                                    yield StreamChunk {
                                        event_type: StreamEventType::ContentBlockDelta,
                                        index: Some(0),
                                        delta: Some(ContentDelta::Text {
                                            text: content.clone(),
                                        }),
                                        stop_reason: None,
                                        usage: None,
                                    };
                                }
                            }
                        }

                        if let Some(usage) = event.usage {
                            yield StreamChunk {
                                event_type: StreamEventType::MessageDelta,
                                index: None,
                                delta: None,
                                stop_reason: None,
                                usage: Some(Usage {
                                    input_tokens: usage.prompt_tokens,
                                    output_tokens: usage.completion_tokens,
                                    cache_creation_input_tokens: 0,
                                    cache_read_input_tokens: 0,
                                }),
                            };
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }
}

// Aleph Alpha API types

#[derive(Debug, Serialize)]
struct AlephAlphaRequest {
    model: String,
    messages: Vec<AlephAlphaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct AlephAlphaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AlephAlphaResponse {
    id: String,
    model: String,
    choices: Vec<AlephAlphaChoice>,
    usage: Option<AlephAlphaUsage>,
}

#[derive(Debug, Deserialize)]
struct AlephAlphaChoice {
    message: AlephAlphaMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AlephAlphaUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AlephAlphaStreamEvent {
    choices: Vec<AlephAlphaStreamChoice>,
    usage: Option<AlephAlphaUsage>,
}

#[derive(Debug, Deserialize)]
struct AlephAlphaStreamChoice {
    delta: Option<AlephAlphaDelta>,
}

#[derive(Debug, Deserialize)]
struct AlephAlphaDelta {
    content: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = AlephAlphaProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.name(), "aleph-alpha");
    }

    #[test]
    fn test_provider_with_api_key() {
        let provider = AlephAlphaProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "aleph-alpha");
    }

    #[test]
    fn test_api_url() {
        let provider = AlephAlphaProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.api_url(), ALEPH_ALPHA_API_URL);
    }

    #[test]
    fn test_api_url_custom_base() {
        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.aleph-alpha.com".to_string());
        let provider = AlephAlphaProvider::new(config).unwrap();
        assert_eq!(provider.api_url(), "https://custom.aleph-alpha.com");
    }

    #[test]
    fn test_convert_request() {
        let provider = AlephAlphaProvider::new(ProviderConfig::new("test-key")).unwrap();

        let mut request = CompletionRequest::new(
            "luminous-supreme",
            vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "Hello".to_string(),
                }],
            }],
        );
        request.system = Some("You are helpful".to_string());
        request.max_tokens = Some(100);
        request.temperature = Some(0.7);

        let api_request = provider.convert_request(&request);
        assert_eq!(api_request.model, "luminous-supreme");
        assert_eq!(api_request.messages.len(), 2);
        assert_eq!(api_request.messages[0].role, "system");
        assert_eq!(api_request.messages[0].content, "You are helpful");
        assert_eq!(api_request.messages[1].role, "user");
        assert_eq!(api_request.messages[1].content, "Hello");
        assert_eq!(api_request.max_tokens, Some(100));
        assert_eq!(api_request.temperature, Some(0.7));
    }

    #[test]
    fn test_convert_message() {
        let provider = AlephAlphaProvider::new(ProviderConfig::new("test-key")).unwrap();

        let user_msg = Message::user("Hello");
        let result = provider.convert_message(&user_msg);
        assert_eq!(result.role, "user");
        assert_eq!(result.content, "Hello");

        let assistant_msg = Message::assistant("Hi there!");
        let result = provider.convert_message(&assistant_msg);
        assert_eq!(result.role, "assistant");
        assert_eq!(result.content, "Hi there!");
    }

    #[test]
    fn test_convert_response() {
        let provider = AlephAlphaProvider::new(ProviderConfig::new("test-key")).unwrap();

        let response = AlephAlphaResponse {
            id: "resp-123".to_string(),
            model: "luminous-supreme".to_string(),
            choices: vec![AlephAlphaChoice {
                message: AlephAlphaMessage {
                    role: "assistant".to_string(),
                    content: "Hello! How can I help?".to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(AlephAlphaUsage {
                prompt_tokens: 10,
                completion_tokens: 15,
            }),
        };

        let result = provider.convert_response(response).unwrap();

        assert_eq!(result.id, "resp-123");
        assert_eq!(result.model, "luminous-supreme");
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
        let provider = AlephAlphaProvider::new(ProviderConfig::new("test-key")).unwrap();

        // Test "stop" -> EndTurn
        let response1 = AlephAlphaResponse {
            id: "1".to_string(),
            model: "model".to_string(),
            choices: vec![AlephAlphaChoice {
                message: AlephAlphaMessage {
                    role: "assistant".to_string(),
                    content: "Done".to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.convert_response(response1).unwrap().stop_reason,
            StopReason::EndTurn
        ));

        // Test "length" -> MaxTokens
        let response2 = AlephAlphaResponse {
            id: "2".to_string(),
            model: "model".to_string(),
            choices: vec![AlephAlphaChoice {
                message: AlephAlphaMessage {
                    role: "assistant".to_string(),
                    content: "Truncated".to_string(),
                },
                finish_reason: Some("length".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.convert_response(response2).unwrap().stop_reason,
            StopReason::MaxTokens
        ));
    }

    #[test]
    fn test_request_serialization() {
        let request = AlephAlphaRequest {
            model: "luminous-supreme".to_string(),
            messages: vec![AlephAlphaMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(100),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stream: Some(false),
            stop: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("luminous-supreme"));
        assert!(json.contains("Hello"));
        assert!(json.contains("100"));
        assert!(json.contains("0.7"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "resp-123",
            "model": "luminous-supreme",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 10
            }
        }"#;

        let response: AlephAlphaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "resp-123");
        assert_eq!(response.model, "luminous-supreme");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.content, "Test response");
    }
}
