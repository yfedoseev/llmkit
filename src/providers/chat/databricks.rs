#![allow(dead_code)]
//! Databricks Model Serving provider implementation.
//!
//! This module provides access to Databricks Foundation Model APIs and
//! custom model endpoints.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::providers::DatabricksProvider;
//!
//! // From environment variables
//! let provider = DatabricksProvider::from_env()?;
//!
//! // With explicit credentials
//! let provider = DatabricksProvider::new(
//!     "https://your-workspace.cloud.databricks.com",
//!     "your-token"
//! )?;
//! ```
//!
//! # Supported Models
//!
//! Foundation Model APIs:
//! - `databricks-meta-llama-3-1-70b-instruct`
//! - `databricks-meta-llama-3-1-405b-instruct`
//! - `databricks-dbrx-instruct`
//! - `databricks-mixtral-8x7b-instruct`
//!
//! # Environment Variables
//!
//! - `DATABRICKS_TOKEN` - Your Databricks personal access token
//! - `DATABRICKS_HOST` - Your Databricks workspace URL

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

/// Databricks Model Serving provider.
///
/// Provides access to Databricks Foundation Model APIs and custom endpoints.
pub struct DatabricksProvider {
    config: ProviderConfig,
    client: Client,
    host: String,
}

impl DatabricksProvider {
    /// Create provider from environment variables.
    ///
    /// Reads: `DATABRICKS_TOKEN` and `DATABRICKS_HOST`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("DATABRICKS_TOKEN").ok();
        let host = std::env::var("DATABRICKS_HOST")
            .map_err(|_| Error::config("DATABRICKS_HOST environment variable not set"))?;

        let config = ProviderConfig {
            api_key,
            ..Default::default()
        };

        Self::with_config(host, config)
    }

    /// Create provider with host URL and token.
    pub fn new(host: impl Into<String>, token: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(token);
        Self::with_config(host, config)
    }

    /// Create provider with custom config.
    fn with_config(host: impl Into<String>, config: ProviderConfig) -> Result<Self> {
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

        let mut host_str = host.into();
        // Remove trailing slash if present
        if host_str.ends_with('/') {
            host_str.pop();
        }

        Ok(Self {
            config,
            client,
            host: host_str,
        })
    }

    /// Get the API URL for a given model.
    fn get_api_url(&self, model: &str) -> String {
        // Databricks uses /serving-endpoints/{model}/invocations for Foundation Model APIs
        format!("{}/serving-endpoints/{}/invocations", self.host, model)
    }

    /// Build messages array for the request.
    fn build_messages(&self, request: &CompletionRequest) -> Vec<DBMessage> {
        let mut messages = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(DBMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        // Add conversation messages
        for msg in &request.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };

            messages.push(DBMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        messages
    }

    /// Convert unified request to Databricks format.
    fn convert_request(&self, request: &CompletionRequest) -> DBRequest {
        let messages = self.build_messages(request);

        // Convert response format for structured output
        let response_format = request.response_format.as_ref().map(|rf| {
            use crate::types::StructuredOutputType;
            match rf.format_type {
                StructuredOutputType::JsonObject => DBResponseFormat::JsonObject,
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        DBResponseFormat::JsonSchema {
                            json_schema: DBJsonSchema {
                                name: schema_def.name.clone(),
                                description: schema_def.description.clone(),
                                schema: schema_def.schema.clone(),
                                strict: Some(schema_def.strict),
                            },
                        }
                    } else {
                        DBResponseFormat::JsonObject
                    }
                }
                StructuredOutputType::Text => DBResponseFormat::Text,
            }
        });

        DBRequest {
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: Some(request.stream),
            stop: request.stop_sequences.clone(),
            response_format,
        }
    }

    /// Convert Databricks response to unified format.
    fn convert_response(&self, response: DBResponse, model: &str) -> CompletionResponse {
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
            id: response
                .id
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            model: model.to_string(),
            content,
            stop_reason,
            usage,
        }
    }

    /// Handle error responses from Databricks API.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        if let Ok(error_resp) = serde_json::from_str::<DBErrorResponse>(body) {
            let message = error_resp
                .message
                .unwrap_or_else(|| error_resp.error.unwrap_or_default());
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
impl Provider for DatabricksProvider {
    fn name(&self) -> &str {
        "databricks"
    }

    fn default_model(&self) -> Option<&str> {
        Some("databricks-meta-llama-3-1-70b-instruct")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let url = self.get_api_url(&request.model);
        let db_request = self.convert_request(&request);

        let response = self.client.post(&url).json(&db_request).send().await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let db_response: DBResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

        Ok(self.convert_response(db_response, &request.model))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let url = self.get_api_url(&request.model);
        let mut db_request = self.convert_request(&request);
        db_request.stream = Some(true);

        let response = self.client.post(&url).json(&db_request).send().await?;

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

                // Process SSE lines
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    // Handle SSE data format
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

                    if let Ok(chunk_resp) = serde_json::from_str::<DBStreamChunk>(data) {
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
        true // Databricks Foundation Model APIs support tool calling
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
struct DBRequest {
    messages: Vec<DBMessage>,
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
    response_format: Option<DBResponseFormat>,
}

/// Response format for structured outputs.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum DBResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: DBJsonSchema },
}

/// JSON schema for structured output.
#[derive(Debug, Serialize)]
struct DBJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct DBMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct DBResponse {
    id: Option<String>,
    choices: Vec<DBChoice>,
    usage: Option<DBUsage>,
}

#[derive(Debug, Deserialize)]
struct DBChoice {
    message: DBResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DBResponseMessage {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DBUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct DBStreamChunk {
    choices: Vec<DBStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct DBStreamChoice {
    delta: Option<DBDelta>,
}

#[derive(Debug, Deserialize)]
struct DBDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DBErrorResponse {
    error: Option<String>,
    message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider =
            DatabricksProvider::new("https://workspace.cloud.databricks.com", "test-token")
                .unwrap();
        assert_eq!(provider.name(), "databricks");
        assert!(provider.supports_tools());
        assert!(!provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_default_model() {
        let provider =
            DatabricksProvider::new("https://workspace.cloud.databricks.com", "test-token")
                .unwrap();
        assert_eq!(
            provider.default_model(),
            Some("databricks-meta-llama-3-1-70b-instruct")
        );
    }

    #[test]
    fn test_api_url() {
        let provider =
            DatabricksProvider::new("https://workspace.cloud.databricks.com", "test-token")
                .unwrap();
        let url = provider.get_api_url("databricks-dbrx-instruct");
        assert_eq!(
            url,
            "https://workspace.cloud.databricks.com/serving-endpoints/databricks-dbrx-instruct/invocations"
        );
    }

    #[test]
    fn test_trailing_slash_removed() {
        let provider =
            DatabricksProvider::new("https://workspace.cloud.databricks.com/", "test-token")
                .unwrap();
        assert!(!provider.host.ends_with('/'));
    }

    #[test]
    fn test_message_building() {
        let provider =
            DatabricksProvider::new("https://workspace.cloud.databricks.com", "test-token")
                .unwrap();

        let request =
            CompletionRequest::new("databricks-dbrx-instruct", vec![Message::user("Hello")])
                .with_system("You are helpful");

        let db_req = provider.convert_request(&request);

        assert_eq!(db_req.messages.len(), 2);
        assert_eq!(db_req.messages[0].role, "system");
        assert_eq!(db_req.messages[0].content, "You are helpful");
        assert_eq!(db_req.messages[1].role, "user");
        assert_eq!(db_req.messages[1].content, "Hello");
    }

    #[test]
    fn test_request_parameters() {
        let provider =
            DatabricksProvider::new("https://workspace.cloud.databricks.com", "test-token")
                .unwrap();

        let request =
            CompletionRequest::new("databricks-dbrx-instruct", vec![Message::user("Hello")])
                .with_max_tokens(500)
                .with_temperature(0.8)
                .with_top_p(0.9);

        let db_req = provider.convert_request(&request);

        assert_eq!(db_req.max_tokens, Some(500));
        assert_eq!(db_req.temperature, Some(0.8));
        assert_eq!(db_req.top_p, Some(0.9));
    }

    #[test]
    fn test_response_parsing() {
        let provider =
            DatabricksProvider::new("https://workspace.cloud.databricks.com", "test-token")
                .unwrap();

        let db_response = DBResponse {
            id: Some("resp-123".to_string()),
            choices: vec![DBChoice {
                message: DBResponseMessage {
                    content: Some("Hello! How can I help?".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(DBUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
            }),
        };

        let response = provider.convert_response(db_response, "databricks-dbrx-instruct");

        assert_eq!(response.id, "resp-123");
        assert_eq!(response.model, "databricks-dbrx-instruct");
        assert_eq!(response.content.len(), 1);
        if let ContentBlock::Text { text } = &response.content[0] {
            assert_eq!(text, "Hello! How can I help?");
        } else {
            panic!("Expected Text content block");
        }
        assert!(matches!(response.stop_reason, StopReason::EndTurn));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 20);
    }

    #[test]
    fn test_stop_reason_mapping() {
        let provider =
            DatabricksProvider::new("https://workspace.cloud.databricks.com", "test-token")
                .unwrap();

        // Test "stop" -> EndTurn
        let response1 = DBResponse {
            id: None,
            choices: vec![DBChoice {
                message: DBResponseMessage {
                    content: Some("Done".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.convert_response(response1, "model").stop_reason,
            StopReason::EndTurn
        ));

        // Test "length" -> MaxTokens
        let response2 = DBResponse {
            id: None,
            choices: vec![DBChoice {
                message: DBResponseMessage {
                    content: Some("Truncated...".to_string()),
                },
                finish_reason: Some("length".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.convert_response(response2, "model").stop_reason,
            StopReason::MaxTokens
        ));
    }

    #[test]
    fn test_error_handling() {
        let provider =
            DatabricksProvider::new("https://workspace.cloud.databricks.com", "test-token")
                .unwrap();

        // Test 401 - auth error
        let error = provider.handle_error_response(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error": "Invalid token"}"#,
        );
        assert!(matches!(error, Error::Authentication(_)));

        // Test 404 - model not found
        let error = provider.handle_error_response(
            reqwest::StatusCode::NOT_FOUND,
            r#"{"message": "Model not found"}"#,
        );
        assert!(matches!(error, Error::ModelNotFound(_)));

        // Test 429 - rate limited
        let error = provider.handle_error_response(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error": "Rate limit exceeded"}"#,
        );
        assert!(matches!(error, Error::RateLimited { .. }));
    }

    #[test]
    fn test_multi_turn_conversation() {
        let provider =
            DatabricksProvider::new("https://workspace.cloud.databricks.com", "test-token")
                .unwrap();

        let request = CompletionRequest::new(
            "databricks-dbrx-instruct",
            vec![
                Message::user("What is 2+2?"),
                Message::assistant("4"),
                Message::user("And 3+3?"),
            ],
        )
        .with_system("You are a math tutor");

        let db_req = provider.convert_request(&request);

        // system + 3 user/assistant messages
        assert_eq!(db_req.messages.len(), 4);
        assert_eq!(db_req.messages[0].role, "system");
        assert_eq!(db_req.messages[1].role, "user");
        assert_eq!(db_req.messages[2].role, "assistant");
        assert_eq!(db_req.messages[3].role, "user");
    }

    #[test]
    fn test_request_serialization() {
        let request = DBRequest {
            messages: vec![DBMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: None,
            stream: Some(false),
            stop: None,
            response_format: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("Hello"));
        assert!(json.contains("1000"));
        assert!(json.contains("0.7"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "resp-123",
            "choices": [{
                "message": {
                    "content": "Hello there!"
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 25
            }
        }"#;

        let response: DBResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, Some("resp-123".to_string()));
        assert_eq!(response.choices.len(), 1);
        assert_eq!(
            response.choices[0].message.content,
            Some("Hello there!".to_string())
        );
        assert_eq!(response.usage.as_ref().unwrap().prompt_tokens, 15);
    }
}
