//! SambaNova Cloud provider implementation.
//!
//! This module provides access to SambaNova's high-performance inference platform.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::providers::SambaNovaProvider;
//!
//! // From environment variable
//! let provider = SambaNovaProvider::from_env()?;
//!
//! // With explicit API key
//! let provider = SambaNovaProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! - `Meta-Llama-3.1-8B-Instruct`
//! - `Meta-Llama-3.1-70B-Instruct`
//! - `Meta-Llama-3.1-405B-Instruct`
//!
//! # Environment Variables
//!
//! - `SAMBANOVA_API_KEY` - Your SambaNova API key

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

const SAMBANOVA_API_URL: &str = "https://api.sambanova.ai/v1/chat/completions";

/// SambaNova Cloud provider.
///
/// Provides access to high-performance inference on SambaNova hardware.
pub struct SambaNovaProvider {
    config: ProviderConfig,
    client: Client,
}

impl SambaNovaProvider {
    /// Create provider from environment variable.
    ///
    /// Reads: `SAMBANOVA_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("SAMBANOVA_API_KEY").ok();

        let config = ProviderConfig {
            api_key,
            ..Default::default()
        };

        Self::new(config)
    }

    /// Create provider with explicit API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    /// Create provider with custom config.
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

    fn api_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(SAMBANOVA_API_URL)
    }

    /// Build messages for the request.
    fn build_messages(&self, request: &CompletionRequest) -> Vec<SNMessage> {
        let mut messages = Vec::new();

        if let Some(ref system) = request.system {
            messages.push(SNMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        for msg in &request.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };

            messages.push(SNMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        messages
    }

    /// Convert unified request to SambaNova format.
    fn convert_request(&self, request: &CompletionRequest) -> SNRequest {
        let messages = self.build_messages(request);

        // Convert response format for structured output
        let response_format = request.response_format.as_ref().map(|rf| {
            use crate::types::StructuredOutputType;
            match rf.format_type {
                StructuredOutputType::JsonObject => SNResponseFormat::JsonObject,
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        SNResponseFormat::JsonSchema {
                            json_schema: SNJsonSchema {
                                name: schema_def.name.clone(),
                                description: schema_def.description.clone(),
                                schema: schema_def.schema.clone(),
                                strict: Some(schema_def.strict),
                            },
                        }
                    } else {
                        SNResponseFormat::JsonObject
                    }
                }
                StructuredOutputType::Text => SNResponseFormat::Text,
            }
        });

        SNRequest {
            model: request.model.clone(),
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: Some(request.stream),
            stop: request.stop_sequences.clone(),
            response_format,
        }
    }

    /// Convert SambaNova response to unified format.
    fn convert_response(&self, response: SNResponse) -> CompletionResponse {
        let mut content = Vec::new();
        let mut stop_reason = StopReason::EndTurn;

        if let Some(choice) = response.choices.into_iter().next() {
            if let Some(text) = choice.message.content {
                if !text.is_empty() {
                    content.push(ContentBlock::Text { text });
                }
            }

            stop_reason = match choice.finish_reason.as_deref() {
                Some("stop") => StopReason::EndTurn,
                Some("length") => StopReason::MaxTokens,
                _ => StopReason::EndTurn,
            };
        }

        let usage = response
            .usage
            .map(|u| Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            })
            .unwrap_or_default();

        CompletionResponse {
            id: response.id,
            model: response.model,
            content,
            stop_reason,
            usage,
        }
    }

    /// Handle error responses.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        if let Ok(error_resp) = serde_json::from_str::<SNErrorResponse>(body) {
            let message = error_resp
                .error
                .map(|e| e.message)
                .unwrap_or_else(|| "Unknown error".to_string());
            match status.as_u16() {
                401 => Error::auth(message),
                403 => Error::auth(message),
                404 => Error::ModelNotFound(message),
                429 => Error::rate_limited(message, None),
                500..=599 => Error::server(status.as_u16(), message),
                _ => Error::other(message),
            }
        } else {
            Error::server(status.as_u16(), format!("HTTP {}: {}", status, body))
        }
    }
}

#[async_trait]
impl Provider for SambaNovaProvider {
    fn name(&self) -> &str {
        "sambanova"
    }

    fn default_model(&self) -> Option<&str> {
        Some("Meta-Llama-3.1-70B-Instruct")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let sn_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .json(&sn_request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let sn_response: SNResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

        Ok(self.convert_response(sn_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let mut sn_request = self.convert_request(&request);
        sn_request.stream = Some(true);

        let response = self
            .client
            .post(self.api_url())
            .json(&sn_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await?;
            return Err(self.handle_error_response(status, &body));
        }

        let stream = async_stream::try_stream! {
            use futures::StreamExt;

            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut chunk_index = 0usize;

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| Error::stream(format!("Stream error: {}", e)))?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    let data = if let Some(stripped) = line.strip_prefix("data: ") {
                        stripped
                    } else {
                        continue;
                    };

                    if data == "[DONE]" {
                        yield StreamChunk {
                            event_type: StreamEventType::MessageStop,
                            index: None,
                            delta: None,
                            stop_reason: None,
                            usage: None,
                        };
                        break;
                    }

                    if let Ok(chunk_resp) = serde_json::from_str::<SNStreamChunk>(data) {
                        if let Some(choice) = chunk_resp.choices.into_iter().next() {
                            if let Some(delta) = choice.delta {
                                if let Some(content) = delta.content {
                                    if !content.is_empty() {
                                        yield StreamChunk {
                                            event_type: StreamEventType::ContentBlockDelta,
                                            index: Some(chunk_index),
                                            delta: Some(ContentDelta::Text { text: content }),
                                            stop_reason: None,
                                            usage: None,
                                        };
                                        chunk_index += 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        false
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

// ============ Request/Response Types ============

#[derive(Debug, Serialize)]
struct SNRequest {
    model: String,
    messages: Vec<SNMessage>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<SNResponseFormat>,
}

/// Response format for structured outputs.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SNResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: SNJsonSchema },
}

/// JSON schema for structured output.
#[derive(Debug, Serialize)]
struct SNJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct SNMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct SNResponse {
    id: String,
    model: String,
    choices: Vec<SNChoice>,
    usage: Option<SNUsage>,
}

#[derive(Debug, Deserialize)]
struct SNChoice {
    message: SNResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SNResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SNUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct SNStreamChunk {
    choices: Vec<SNStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct SNStreamChoice {
    delta: Option<SNDelta>,
}

#[derive(Debug, Deserialize)]
struct SNDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SNErrorResponse {
    error: Option<SNError>,
}

#[derive(Debug, Deserialize)]
struct SNError {
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = SambaNovaProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "sambanova");
        assert!(provider.supports_tools());
        assert!(provider.supports_streaming());
        assert!(!provider.supports_vision());
    }

    #[test]
    fn test_default_model() {
        let provider = SambaNovaProvider::with_api_key("test-key").unwrap();
        assert_eq!(
            provider.default_model(),
            Some("Meta-Llama-3.1-70B-Instruct")
        );
    }

    #[test]
    fn test_api_url() {
        let provider = SambaNovaProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.api_url(), SAMBANOVA_API_URL);
    }

    #[test]
    fn test_api_url_custom_base() {
        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.sambanova.ai".to_string());
        let provider = SambaNovaProvider::new(config).unwrap();
        assert_eq!(provider.api_url(), "https://custom.sambanova.ai");
    }

    #[test]
    fn test_message_building() {
        let provider = SambaNovaProvider::with_api_key("test-key").unwrap();

        let request =
            CompletionRequest::new("Meta-Llama-3.1-70B-Instruct", vec![Message::user("Hello")])
                .with_system("You are helpful");

        let sn_req = provider.convert_request(&request);

        assert_eq!(sn_req.messages.len(), 2);
        assert_eq!(sn_req.messages[0].role, "system");
        assert_eq!(sn_req.messages[0].content, "You are helpful");
        assert_eq!(sn_req.messages[1].role, "user");
        assert_eq!(sn_req.messages[1].content, "Hello");
    }

    #[test]
    fn test_convert_response() {
        let provider = SambaNovaProvider::with_api_key("test-key").unwrap();

        let response = SNResponse {
            id: "resp-123".to_string(),
            model: "Meta-Llama-3.1-70B-Instruct".to_string(),
            choices: vec![SNChoice {
                message: SNResponseMessage {
                    content: Some("Hello! How can I help?".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(SNUsage {
                prompt_tokens: 10,
                completion_tokens: 15,
            }),
        };

        let result = provider.convert_response(response);

        assert_eq!(result.id, "resp-123");
        assert_eq!(result.model, "Meta-Llama-3.1-70B-Instruct");
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
        let provider = SambaNovaProvider::with_api_key("test-key").unwrap();

        // Test "stop" -> EndTurn
        let response1 = SNResponse {
            id: "1".to_string(),
            model: "model".to_string(),
            choices: vec![SNChoice {
                message: SNResponseMessage {
                    content: Some("Done".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.convert_response(response1).stop_reason,
            StopReason::EndTurn
        ));

        // Test "length" -> MaxTokens
        let response2 = SNResponse {
            id: "2".to_string(),
            model: "model".to_string(),
            choices: vec![SNChoice {
                message: SNResponseMessage {
                    content: Some("Truncated".to_string()),
                },
                finish_reason: Some("length".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.convert_response(response2).stop_reason,
            StopReason::MaxTokens
        ));
    }

    #[test]
    fn test_request_serialization() {
        let request = SNRequest {
            model: "Meta-Llama-3.1-70B-Instruct".to_string(),
            messages: vec![SNMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(1024),
            temperature: Some(0.7),
            top_p: None,
            stream: Some(false),
            stop: None,
            response_format: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("Meta-Llama-3.1-70B-Instruct"));
        assert!(json.contains("Hello"));
        assert!(json.contains("1024"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "resp-123",
            "model": "Meta-Llama-3.1-70B-Instruct",
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }"#;

        let response: SNResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "resp-123");
        assert_eq!(
            response.choices[0].message.content,
            Some("Hello!".to_string())
        );
    }
}
