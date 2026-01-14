//! Cerebras Inference provider implementation.
//!
//! This module provides access to Cerebras's ultra-fast inference platform.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::CerebrasProvider;
//!
//! // From environment variable
//! let provider = CerebrasProvider::from_env()?;
//!
//! // With explicit API key
//! let provider = CerebrasProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! - `llama3.1-8b` - Llama 3.1 8B
//! - `llama3.1-70b` - Llama 3.1 70B
//! - `llama-3.3-70b` - Llama 3.3 70B
//!
//! # Environment Variables
//!
//! - `CEREBRAS_API_KEY` - Your Cerebras API key

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

const CEREBRAS_API_URL: &str = "https://api.cerebras.ai/v1/chat/completions";

/// Cerebras Inference provider.
///
/// Provides access to ultra-fast inference on Cerebras hardware.
pub struct CerebrasProvider {
    config: ProviderConfig,
    client: Client,
}

impl CerebrasProvider {
    /// Create provider from environment variable.
    ///
    /// Reads: `CEREBRAS_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("CEREBRAS_API_KEY").ok();

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
        self.config.base_url.as_deref().unwrap_or(CEREBRAS_API_URL)
    }

    /// Build messages for the request.
    fn build_messages(&self, request: &CompletionRequest) -> Vec<CerebrasMessage> {
        let mut messages = Vec::new();

        if let Some(ref system) = request.system {
            messages.push(CerebrasMessage {
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

            messages.push(CerebrasMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        messages
    }

    /// Convert unified request to Cerebras format.
    fn convert_request(&self, request: &CompletionRequest) -> CerebrasRequest {
        let messages = self.build_messages(request);

        // Convert response format for structured output
        let response_format = request.response_format.as_ref().map(|rf| {
            use crate::types::StructuredOutputType;
            match rf.format_type {
                StructuredOutputType::JsonObject => CerebrasResponseFormat::JsonObject,
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        CerebrasResponseFormat::JsonSchema {
                            json_schema: CerebrasJsonSchema {
                                name: schema_def.name.clone(),
                                description: schema_def.description.clone(),
                                schema: schema_def.schema.clone(),
                                strict: Some(schema_def.strict),
                            },
                        }
                    } else {
                        CerebrasResponseFormat::JsonObject
                    }
                }
                StructuredOutputType::Text => CerebrasResponseFormat::Text,
            }
        });

        CerebrasRequest {
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

    /// Convert Cerebras response to unified format.
    fn convert_response(&self, response: CerebrasResponse) -> CompletionResponse {
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
        if let Ok(error_resp) = serde_json::from_str::<CerebrasErrorResponse>(body) {
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
impl Provider for CerebrasProvider {
    fn name(&self) -> &str {
        "cerebras"
    }

    fn default_model(&self) -> Option<&str> {
        Some("llama3.1-70b")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let cerebras_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .json(&cerebras_request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let cerebras_response: CerebrasResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

        Ok(self.convert_response(cerebras_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let mut cerebras_request = self.convert_request(&request);
        cerebras_request.stream = Some(true);

        let response = self
            .client
            .post(self.api_url())
            .json(&cerebras_request)
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

                    if let Ok(chunk_resp) = serde_json::from_str::<CerebrasStreamChunk>(data) {
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
        false
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
struct CerebrasRequest {
    model: String,
    messages: Vec<CerebrasMessage>,
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
    response_format: Option<CerebrasResponseFormat>,
}

/// Response format for structured outputs.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CerebrasResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: CerebrasJsonSchema },
}

/// JSON schema for structured output.
#[derive(Debug, Serialize)]
struct CerebrasJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct CerebrasMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct CerebrasResponse {
    id: String,
    model: String,
    choices: Vec<CerebrasChoice>,
    usage: Option<CerebrasUsage>,
}

#[derive(Debug, Deserialize)]
struct CerebrasChoice {
    message: CerebrasResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CerebrasResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CerebrasUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct CerebrasStreamChunk {
    choices: Vec<CerebrasStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct CerebrasStreamChoice {
    delta: Option<CerebrasDelta>,
}

#[derive(Debug, Deserialize)]
struct CerebrasDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CerebrasErrorResponse {
    error: Option<CerebrasError>,
}

#[derive(Debug, Deserialize)]
struct CerebrasError {
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = CerebrasProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "cerebras");
        assert!(!provider.supports_tools());
        assert!(!provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_default_model() {
        let provider = CerebrasProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.default_model(), Some("llama3.1-70b"));
    }

    #[test]
    fn test_api_url() {
        let provider = CerebrasProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.api_url(), CEREBRAS_API_URL);

        // Test custom base URL
        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.cerebras.ai/v1".to_string());
        let provider = CerebrasProvider::new(config).unwrap();
        assert_eq!(provider.api_url(), "https://custom.cerebras.ai/v1");
    }

    #[test]
    fn test_message_building() {
        let provider = CerebrasProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("llama3.1-70b", vec![Message::user("Hello")])
            .with_system("You are helpful");

        let cerebras_req = provider.convert_request(&request);

        assert_eq!(cerebras_req.messages.len(), 2);
        assert_eq!(cerebras_req.messages[0].role, "system");
        assert_eq!(cerebras_req.messages[0].content, "You are helpful");
        assert_eq!(cerebras_req.messages[1].role, "user");
        assert_eq!(cerebras_req.messages[1].content, "Hello");
    }

    #[test]
    fn test_request_parameters() {
        let provider = CerebrasProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("llama3.1-70b", vec![Message::user("Hello")])
            .with_max_tokens(1024)
            .with_temperature(0.7)
            .with_stop_sequences(vec!["STOP".to_string()]);

        let cerebras_req = provider.convert_request(&request);

        assert_eq!(cerebras_req.model, "llama3.1-70b");
        assert_eq!(cerebras_req.max_tokens, Some(1024));
        assert_eq!(cerebras_req.temperature, Some(0.7));
        assert_eq!(cerebras_req.stop, Some(vec!["STOP".to_string()]));
    }

    #[test]
    fn test_response_parsing() {
        let provider = CerebrasProvider::with_api_key("test-key").unwrap();

        let response = CerebrasResponse {
            id: "resp-123".to_string(),
            model: "llama3.1-70b".to_string(),
            choices: vec![CerebrasChoice {
                message: CerebrasResponseMessage {
                    content: Some("Hello there!".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(CerebrasUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
            }),
        };

        let result = provider.convert_response(response);

        assert_eq!(result.id, "resp-123");
        assert_eq!(result.model, "llama3.1-70b");
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlock::Text { text } => {
                assert_eq!(text, "Hello there!");
            }
            other => {
                panic!("Expected text content, got {:?}", other);
            }
        }
        assert!(matches!(result.stop_reason, StopReason::EndTurn));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 20);
    }

    #[test]
    fn test_stop_reason_mapping() {
        let provider = CerebrasProvider::with_api_key("test-key").unwrap();

        // Test "stop" -> EndTurn
        let response1 = CerebrasResponse {
            id: "resp-1".to_string(),
            model: "llama3.1-70b".to_string(),
            choices: vec![CerebrasChoice {
                message: CerebrasResponseMessage {
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
        let response2 = CerebrasResponse {
            id: "resp-2".to_string(),
            model: "llama3.1-70b".to_string(),
            choices: vec![CerebrasChoice {
                message: CerebrasResponseMessage {
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
        let request = CerebrasRequest {
            model: "llama3.1-70b".to_string(),
            messages: vec![CerebrasMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stream: Some(false),
            stop: None,
            response_format: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("llama3.1-70b"));
        assert!(json.contains("\"max_tokens\":1000"));
        assert!(json.contains("\"temperature\":0.7"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "chatcmpl-abc123",
            "model": "llama3.1-70b",
            "choices": [{
                "message": {"content": "Hi!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }"#;

        let response: CerebrasResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "chatcmpl-abc123");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.content, Some("Hi!".to_string()));
        assert_eq!(response.usage.as_ref().unwrap().prompt_tokens, 5);
    }

    #[test]
    fn test_multi_turn_conversation() {
        let provider = CerebrasProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new(
            "llama3.1-70b",
            vec![
                Message::user("Hello"),
                Message::assistant("Hi there!"),
                Message::user("How are you?"),
            ],
        )
        .with_system("Be friendly");

        let cerebras_req = provider.convert_request(&request);

        assert_eq!(cerebras_req.messages.len(), 4);
        assert_eq!(cerebras_req.messages[0].role, "system");
        assert_eq!(cerebras_req.messages[1].role, "user");
        assert_eq!(cerebras_req.messages[2].role, "assistant");
        assert_eq!(cerebras_req.messages[3].role, "user");
    }

    #[test]
    fn test_error_handling() {
        let provider = CerebrasProvider::with_api_key("test-key").unwrap();

        // Test 401 -> auth error
        let error = provider.handle_error_response(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error": {"message": "Invalid API key"}}"#,
        );
        assert!(matches!(error, Error::Authentication(_)));

        // Test 404 -> model not found
        let error = provider.handle_error_response(
            reqwest::StatusCode::NOT_FOUND,
            r#"{"error": {"message": "Model not found"}}"#,
        );
        assert!(matches!(error, Error::ModelNotFound(_)));

        // Test 429 -> rate limited
        let error = provider.handle_error_response(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error": {"message": "Rate limit exceeded"}}"#,
        );
        assert!(matches!(error, Error::RateLimited { .. }));
    }
}
