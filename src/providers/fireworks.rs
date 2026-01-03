//! Fireworks AI provider implementation.
//!
//! This module provides access to Fireworks AI's fast inference platform.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::FireworksProvider;
//!
//! // From environment variable
//! let provider = FireworksProvider::from_env()?;
//!
//! // With explicit API key
//! let provider = FireworksProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! - `accounts/fireworks/models/llama-v3p1-70b-instruct`
//! - `accounts/fireworks/models/llama-v3p1-405b-instruct`
//! - `accounts/fireworks/models/mixtral-8x22b-instruct`
//! - `accounts/fireworks/models/qwen2p5-72b-instruct`
//!
//! # Environment Variables
//!
//! - `FIREWORKS_API_KEY` - Your Fireworks AI API key

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

const FIREWORKS_API_URL: &str = "https://api.fireworks.ai/inference/v1/chat/completions";

/// Fireworks AI provider.
///
/// Provides access to fast inference on Fireworks AI infrastructure.
pub struct FireworksProvider {
    config: ProviderConfig,
    client: Client,
}

impl FireworksProvider {
    /// Create provider from environment variable.
    ///
    /// Reads: `FIREWORKS_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("FIREWORKS_API_KEY").ok();

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
        self.config.base_url.as_deref().unwrap_or(FIREWORKS_API_URL)
    }

    /// Build messages for the request.
    fn build_messages(&self, request: &CompletionRequest) -> Vec<FWMessage> {
        let mut messages = Vec::new();

        if let Some(ref system) = request.system {
            messages.push(FWMessage {
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

            messages.push(FWMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        messages
    }

    /// Convert unified request to Fireworks format.
    fn convert_request(&self, request: &CompletionRequest) -> FWRequest {
        let messages = self.build_messages(request);

        // Convert response format for structured output
        let response_format = request.response_format.as_ref().map(|rf| {
            use crate::types::StructuredOutputType;
            match rf.format_type {
                StructuredOutputType::JsonObject => FWResponseFormat::JsonObject,
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        FWResponseFormat::JsonSchema {
                            json_schema: FWJsonSchema {
                                name: schema_def.name.clone(),
                                description: schema_def.description.clone(),
                                schema: schema_def.schema.clone(),
                                strict: Some(schema_def.strict),
                            },
                        }
                    } else {
                        FWResponseFormat::JsonObject
                    }
                }
                StructuredOutputType::Text => FWResponseFormat::Text,
            }
        });

        FWRequest {
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

    /// Convert Fireworks response to unified format.
    fn convert_response(&self, response: FWResponse) -> CompletionResponse {
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
                Some("tool_calls") => StopReason::ToolUse,
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
        if let Ok(error_resp) = serde_json::from_str::<FWErrorResponse>(body) {
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
impl Provider for FireworksProvider {
    fn name(&self) -> &str {
        "fireworks"
    }

    fn default_model(&self) -> Option<&str> {
        Some("accounts/fireworks/models/llama-v3p1-70b-instruct")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let fw_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .json(&fw_request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let fw_response: FWResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

        Ok(self.convert_response(fw_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let mut fw_request = self.convert_request(&request);
        fw_request.stream = Some(true);

        let response = self
            .client
            .post(self.api_url())
            .json(&fw_request)
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

                    if let Ok(chunk_resp) = serde_json::from_str::<FWStreamChunk>(data) {
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
        true // Fireworks supports vision models
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

// ============ Request/Response Types ============

#[derive(Debug, Serialize)]
struct FWRequest {
    model: String,
    messages: Vec<FWMessage>,
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
    response_format: Option<FWResponseFormat>,
}

/// Response format for structured outputs.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum FWResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: FWJsonSchema },
}

/// JSON schema for structured output.
#[derive(Debug, Serialize)]
struct FWJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct FWMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct FWResponse {
    id: String,
    model: String,
    choices: Vec<FWChoice>,
    usage: Option<FWUsage>,
}

#[derive(Debug, Deserialize)]
struct FWChoice {
    message: FWResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FWResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FWUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct FWStreamChunk {
    choices: Vec<FWStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct FWStreamChoice {
    delta: Option<FWDelta>,
}

#[derive(Debug, Deserialize)]
struct FWDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct FWErrorResponse {
    error: Option<FWError>,
}

#[derive(Debug, Deserialize)]
struct FWError {
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = FireworksProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "fireworks");
        assert!(provider.supports_tools());
        assert!(provider.supports_streaming());
        assert!(provider.supports_vision());
    }

    #[test]
    fn test_default_model() {
        let provider = FireworksProvider::with_api_key("test-key").unwrap();
        assert_eq!(
            provider.default_model(),
            Some("accounts/fireworks/models/llama-v3p1-70b-instruct")
        );
    }

    #[test]
    fn test_api_url() {
        let provider = FireworksProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.api_url(), FIREWORKS_API_URL);
    }

    #[test]
    fn test_api_url_custom_base() {
        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.fireworks.ai".to_string());
        let provider = FireworksProvider::new(config).unwrap();
        assert_eq!(provider.api_url(), "https://custom.fireworks.ai");
    }

    #[test]
    fn test_message_building() {
        let provider = FireworksProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new(
            "accounts/fireworks/models/llama-v3p1-70b-instruct",
            vec![Message::user("Hello")],
        )
        .with_system("You are helpful");

        let fw_req = provider.convert_request(&request);

        assert_eq!(fw_req.messages.len(), 2);
        assert_eq!(fw_req.messages[0].role, "system");
        assert_eq!(fw_req.messages[0].content, "You are helpful");
        assert_eq!(fw_req.messages[1].role, "user");
        assert_eq!(fw_req.messages[1].content, "Hello");
    }

    #[test]
    fn test_convert_response() {
        let provider = FireworksProvider::with_api_key("test-key").unwrap();

        let response = FWResponse {
            id: "resp-123".to_string(),
            model: "accounts/fireworks/models/llama-v3p1-70b-instruct".to_string(),
            choices: vec![FWChoice {
                message: FWResponseMessage {
                    content: Some("Hello! How can I help?".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(FWUsage {
                prompt_tokens: 10,
                completion_tokens: 15,
            }),
        };

        let result = provider.convert_response(response);

        assert_eq!(result.id, "resp-123");
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
        let provider = FireworksProvider::with_api_key("test-key").unwrap();

        // Test "stop" -> EndTurn
        let response1 = FWResponse {
            id: "1".to_string(),
            model: "model".to_string(),
            choices: vec![FWChoice {
                message: FWResponseMessage {
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
        let response2 = FWResponse {
            id: "2".to_string(),
            model: "model".to_string(),
            choices: vec![FWChoice {
                message: FWResponseMessage {
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

        // Test "tool_calls" -> ToolUse
        let response3 = FWResponse {
            id: "3".to_string(),
            model: "model".to_string(),
            choices: vec![FWChoice {
                message: FWResponseMessage { content: None },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.convert_response(response3).stop_reason,
            StopReason::ToolUse
        ));
    }

    #[test]
    fn test_request_serialization() {
        let request = FWRequest {
            model: "accounts/fireworks/models/llama-v3p1-70b-instruct".to_string(),
            messages: vec![FWMessage {
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
        assert!(json.contains("llama-v3p1-70b-instruct"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "resp-123",
            "model": "accounts/fireworks/models/llama-v3p1-70b-instruct",
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }"#;

        let response: FWResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "resp-123");
        assert_eq!(
            response.choices[0].message.content,
            Some("Hello!".to_string())
        );
    }
}
