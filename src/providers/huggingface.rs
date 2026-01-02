#![allow(dead_code)]
//! HuggingFace Inference API provider implementation.
//!
//! This module provides access to HuggingFace's Inference API for chat completions.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::HuggingFaceProvider;
//!
//! // Serverless Inference API
//! let provider = HuggingFaceProvider::from_env()?;
//!
//! // Dedicated Inference Endpoint
//! let provider = HuggingFaceProvider::endpoint(
//!     "https://your-endpoint.huggingface.cloud",
//!     "your-api-key"
//! )?;
//! ```
//!
//! # Supported Models
//!
//! Works with any text-generation model on HuggingFace Hub that supports
//! the Messages API format, including:
//! - Meta Llama models
//! - Mistral models
//! - Microsoft Phi models
//! - And many more
//!
//! # Environment Variables
//!
//! - `HUGGINGFACE_API_KEY` or `HF_TOKEN` - Your HuggingFace API token

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

const HF_INFERENCE_API_URL: &str = "https://api-inference.huggingface.co/models";

/// HuggingFace Inference API provider.
///
/// Supports both serverless inference and dedicated endpoints.
pub struct HuggingFaceProvider {
    config: ProviderConfig,
    client: Client,
    /// Optional custom endpoint URL (for dedicated endpoints)
    endpoint_url: Option<String>,
}

impl HuggingFaceProvider {
    /// Create provider from environment variable.
    ///
    /// Reads: `HUGGINGFACE_API_KEY` or `HF_TOKEN`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("HUGGINGFACE_API_KEY")
            .or_else(|_| std::env::var("HF_TOKEN"))
            .ok();

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

        Ok(Self {
            config,
            client,
            endpoint_url: None,
        })
    }

    /// Create provider for a dedicated inference endpoint.
    ///
    /// # Arguments
    ///
    /// * `endpoint_url` - The URL of your dedicated endpoint
    /// * `api_key` - Your HuggingFace API token
    pub fn endpoint(endpoint_url: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);

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

        Ok(Self {
            config,
            client,
            endpoint_url: Some(endpoint_url.into()),
        })
    }

    /// Get the API URL for a given model.
    fn get_api_url(&self, model: &str) -> String {
        if let Some(ref endpoint) = self.endpoint_url {
            format!("{}/v1/chat/completions", endpoint.trim_end_matches('/'))
        } else {
            // Serverless inference API
            format!("{}/{}/v1/chat/completions", HF_INFERENCE_API_URL, model)
        }
    }

    /// Convert unified request to HuggingFace format.
    fn convert_request(&self, request: &CompletionRequest) -> HFRequest {
        let mut messages = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(HFMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        // Convert messages
        for msg in &request.messages {
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

            messages.push(HFMessage {
                role: role.to_string(),
                content,
            });
        }

        // Convert tools if present
        let tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|tool| HFTool {
                    r#type: "function".to_string(),
                    function: HFFunction {
                        name: tool.name.clone(),
                        description: Some(tool.description.clone()),
                        parameters: tool.input_schema.clone(),
                    },
                })
                .collect()
        });

        HFRequest {
            model: request.model.clone(),
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: Some(request.stream),
            tools,
            stop: request.stop_sequences.clone(),
        }
    }

    /// Convert HuggingFace response to unified format.
    fn convert_response(&self, response: HFResponse) -> CompletionResponse {
        let mut content = Vec::new();
        let mut stop_reason = StopReason::EndTurn;

        if let Some(choice) = response.choices.into_iter().next() {
            // Handle finish reason
            if let Some(ref reason) = choice.finish_reason {
                stop_reason = match reason.as_str() {
                    "stop" => StopReason::EndTurn,
                    "length" => StopReason::MaxTokens,
                    "tool_calls" => StopReason::ToolUse,
                    _ => StopReason::EndTurn,
                };
            }

            // Extract text content
            if let Some(text) = choice.message.content {
                if !text.is_empty() {
                    content.push(ContentBlock::Text { text });
                }
            }

            // Handle tool calls
            if let Some(tool_calls) = choice.message.tool_calls {
                for tc in tool_calls {
                    content.push(ContentBlock::ToolUse {
                        id: tc.id,
                        name: tc.function.name,
                        input: serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new())),
                    });
                }
            }
        }

        let usage = response.usage.map(|u| Usage {
            input_tokens: u.prompt_tokens,
            output_tokens: u.completion_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        });

        CompletionResponse {
            id: response
                .id
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            model: response.model.unwrap_or_default(),
            content,
            stop_reason,
            usage: usage.unwrap_or_default(),
        }
    }

    /// Handle error responses from HuggingFace API.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        if let Ok(error_resp) = serde_json::from_str::<HFErrorResponse>(body) {
            let message = error_resp.error;
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
impl Provider for HuggingFaceProvider {
    fn name(&self) -> &str {
        "huggingface"
    }

    fn default_model(&self) -> Option<&str> {
        Some("meta-llama/Llama-3.2-3B-Instruct")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let url = self.get_api_url(&request.model);
        let hf_request = self.convert_request(&request);

        let response = self.client.post(&url).json(&hf_request).send().await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let hf_response: HFResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

        Ok(self.convert_response(hf_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let url = self.get_api_url(&request.model);
        let mut hf_request = self.convert_request(&request);
        hf_request.stream = Some(true);

        let response = self.client.post(&url).json(&hf_request).send().await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await?;
            return Err(self.handle_error_response(status, &body));
        }

        let stream = async_stream::try_stream! {
            use futures::StreamExt;
            use eventsource_stream::Eventsource;

            let mut event_stream = response.bytes_stream().eventsource();
            let mut chunk_index = 0usize;

            while let Some(event) = event_stream.next().await {
                let event = event.map_err(|e| Error::stream(format!("Stream error: {}", e)))?;

                if event.data == "[DONE]" {
                    yield StreamChunk {
                        event_type: StreamEventType::MessageStop,
                        index: None,
                        delta: None,
                        stop_reason: None,
                        usage: None,
                    };
                    break;
                }

                if let Ok(chunk) = serde_json::from_str::<HFStreamChunk>(&event.data) {
                    if let Some(choice) = chunk.choices.into_iter().next() {
                        if let Some(delta) = choice.delta {
                            if let Some(content) = delta.content {
                                if !content.is_empty() {
                                    yield StreamChunk {
                                        event_type: StreamEventType::ContentBlockDelta,
                                        index: Some(chunk_index),
                                        delta: Some(ContentDelta::TextDelta { text: content }),
                                        stop_reason: None,
                                        usage: None,
                                    };
                                    chunk_index += 1;
                                }
                            }

                            // Handle tool calls in streaming
                            if let Some(tool_calls) = delta.tool_calls {
                                for tc in tool_calls {
                                    if let Some(function) = tc.function {
                                        // Use ToolUseDelta with name for start, or just arguments for continuation
                                        yield StreamChunk {
                                            event_type: StreamEventType::ContentBlockDelta,
                                            index: Some(chunk_index),
                                            delta: Some(ContentDelta::ToolUseDelta {
                                                id: tc.id,
                                                name: function.name,
                                                input_json_delta: function.arguments,
                                            }),
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
        false // Depends on the model
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

// ============ Request/Response Types ============

#[derive(Debug, Serialize)]
struct HFRequest {
    model: String,
    messages: Vec<HFMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<HFTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct HFMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct HFTool {
    r#type: String,
    function: HFFunction,
}

#[derive(Debug, Serialize)]
struct HFFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct HFResponse {
    id: Option<String>,
    model: Option<String>,
    choices: Vec<HFChoice>,
    usage: Option<HFUsage>,
}

#[derive(Debug, Deserialize)]
struct HFChoice {
    message: HFResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HFResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<HFToolCall>>,
}

#[derive(Debug, Deserialize)]
struct HFToolCall {
    id: String,
    function: HFToolCallFunction,
}

#[derive(Debug, Deserialize)]
struct HFToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct HFUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct HFStreamChunk {
    choices: Vec<HFStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct HFStreamChoice {
    delta: Option<HFStreamDelta>,
}

#[derive(Debug, Deserialize)]
struct HFStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<HFStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct HFStreamToolCall {
    id: Option<String>,
    function: Option<HFStreamFunction>,
}

#[derive(Debug, Deserialize)]
struct HFStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct HFErrorResponse {
    error: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = HuggingFaceProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "huggingface");
        assert!(provider.supports_tools());
        assert!(!provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_default_model() {
        let provider = HuggingFaceProvider::with_api_key("test-key").unwrap();
        assert_eq!(
            provider.default_model(),
            Some("meta-llama/Llama-3.2-3B-Instruct")
        );
    }

    #[test]
    fn test_serverless_api_url() {
        let provider = HuggingFaceProvider::with_api_key("test-key").unwrap();
        let url = provider.get_api_url("meta-llama/Llama-3.2-3B-Instruct");
        assert_eq!(
            url,
            "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct/v1/chat/completions"
        );
    }

    #[test]
    fn test_endpoint_api_url() {
        let provider =
            HuggingFaceProvider::endpoint("https://my-endpoint.huggingface.cloud", "test-key")
                .unwrap();
        let url = provider.get_api_url("any-model");
        assert_eq!(
            url,
            "https://my-endpoint.huggingface.cloud/v1/chat/completions"
        );
    }

    #[test]
    fn test_request_conversion() {
        let provider = HuggingFaceProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new(
            "meta-llama/Llama-3.2-3B-Instruct",
            vec![Message::user("Hello")],
        )
        .with_system("You are helpful")
        .with_max_tokens(1024);

        let hf_req = provider.convert_request(&request);

        assert_eq!(hf_req.model, "meta-llama/Llama-3.2-3B-Instruct");
        assert_eq!(hf_req.messages.len(), 2);
        assert_eq!(hf_req.messages[0].role, "system");
        assert_eq!(hf_req.messages[0].content, "You are helpful");
        assert_eq!(hf_req.messages[1].role, "user");
        assert_eq!(hf_req.messages[1].content, "Hello");
        assert_eq!(hf_req.max_tokens, Some(1024));
    }

    #[test]
    fn test_request_parameters() {
        let provider = HuggingFaceProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new(
            "meta-llama/Llama-3.2-3B-Instruct",
            vec![Message::user("Hello")],
        )
        .with_max_tokens(500)
        .with_temperature(0.8)
        .with_top_p(0.9);

        let hf_req = provider.convert_request(&request);

        assert_eq!(hf_req.max_tokens, Some(500));
        assert_eq!(hf_req.temperature, Some(0.8));
        assert_eq!(hf_req.top_p, Some(0.9));
    }

    #[test]
    fn test_response_parsing() {
        let provider = HuggingFaceProvider::with_api_key("test-key").unwrap();

        let hf_response = HFResponse {
            id: Some("resp-123".to_string()),
            model: Some("meta-llama/Llama-3.2-3B-Instruct".to_string()),
            choices: vec![HFChoice {
                message: HFResponseMessage {
                    content: Some("Hello! How can I help?".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(HFUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
            }),
        };

        let response = provider.convert_response(hf_response);

        assert_eq!(response.id, "resp-123");
        assert_eq!(response.model, "meta-llama/Llama-3.2-3B-Instruct");
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
        let provider = HuggingFaceProvider::with_api_key("test-key").unwrap();

        // Test "stop" -> EndTurn
        let response1 = HFResponse {
            id: None,
            model: None,
            choices: vec![HFChoice {
                message: HFResponseMessage {
                    content: Some("Done".to_string()),
                    tool_calls: None,
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
        let response2 = HFResponse {
            id: None,
            model: None,
            choices: vec![HFChoice {
                message: HFResponseMessage {
                    content: Some("Truncated...".to_string()),
                    tool_calls: None,
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
        let response3 = HFResponse {
            id: None,
            model: None,
            choices: vec![HFChoice {
                message: HFResponseMessage {
                    content: None,
                    tool_calls: None,
                },
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
    fn test_tool_call_response() {
        let provider = HuggingFaceProvider::with_api_key("test-key").unwrap();

        let hf_response = HFResponse {
            id: Some("tool-resp-123".to_string()),
            model: Some("model".to_string()),
            choices: vec![HFChoice {
                message: HFResponseMessage {
                    content: None,
                    tool_calls: Some(vec![HFToolCall {
                        id: "call_abc123".to_string(),
                        function: HFToolCallFunction {
                            name: "get_weather".to_string(),
                            arguments: r#"{"location": "Paris"}"#.to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: None,
        };

        let response = provider.convert_response(hf_response);

        assert_eq!(response.content.len(), 1);
        assert!(matches!(response.stop_reason, StopReason::ToolUse));

        if let ContentBlock::ToolUse { id, name, input } = &response.content[0] {
            assert_eq!(id, "call_abc123");
            assert_eq!(name, "get_weather");
            assert_eq!(input.get("location").unwrap().as_str().unwrap(), "Paris");
        } else {
            panic!("Expected ToolUse content block");
        }
    }

    #[test]
    fn test_error_handling() {
        let provider = HuggingFaceProvider::with_api_key("test-key").unwrap();

        // Test 401 - auth error
        let error = provider.handle_error_response(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"error": "Invalid token"}"#,
        );
        assert!(matches!(error, Error::Authentication(_)));

        // Test 404 - model not found
        let error = provider.handle_error_response(
            reqwest::StatusCode::NOT_FOUND,
            r#"{"error": "Model not found"}"#,
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
        let provider = HuggingFaceProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new(
            "meta-llama/Llama-3.2-3B-Instruct",
            vec![
                Message::user("What is 2+2?"),
                Message::assistant("4"),
                Message::user("And 3+3?"),
            ],
        )
        .with_system("You are a math tutor");

        let hf_req = provider.convert_request(&request);

        // system + 3 user/assistant messages
        assert_eq!(hf_req.messages.len(), 4);
        assert_eq!(hf_req.messages[0].role, "system");
        assert_eq!(hf_req.messages[1].role, "user");
        assert_eq!(hf_req.messages[2].role, "assistant");
        assert_eq!(hf_req.messages[3].role, "user");
    }
}
