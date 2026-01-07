//! DeepSeek provider implementation.
//!
//! This module provides access to DeepSeek's AI models.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::providers::DeepSeekProvider;
//!
//! // From environment variable
//! let provider = DeepSeekProvider::from_env()?;
//!
//! // With explicit API key
//! let provider = DeepSeekProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! - `deepseek-chat` - DeepSeek V3 chat model (default)
//! - `deepseek-reasoner` - DeepSeek R1 reasoning model
//! - `deepseek-v3.2` - DeepSeek V3.2 (Dec 2025, improved reasoning)
//! - `deepseek-v3.2-speciale` - DeepSeek V3.2 Speciale (high-compute reasoning, GPT-5 level)
//!
//! # Environment Variables
//!
//! - `DEEPSEEK_API_KEY` - Your DeepSeek API key

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Role, StopReason,
    StreamChunk, StreamEventType, ThinkingType, Usage,
};

const DEEPSEEK_API_URL: &str = "https://api.deepseek.com/chat/completions";

/// DeepSeek provider.
///
/// Provides access to DeepSeek's AI models including DeepSeek V3 and R1.
pub struct DeepSeekProvider {
    config: ProviderConfig,
    client: Client,
}

impl DeepSeekProvider {
    /// Create provider from environment variable.
    ///
    /// Reads: `DEEPSEEK_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("DEEPSEEK_API_KEY").ok();

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
        self.config.base_url.as_deref().unwrap_or(DEEPSEEK_API_URL)
    }

    /// Build messages for the request.
    fn build_messages(&self, request: &CompletionRequest) -> Vec<DSMessage> {
        let mut messages = Vec::new();

        if let Some(ref system) = request.system {
            messages.push(DSMessage {
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

            messages.push(DSMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        messages
    }

    /// Select the appropriate model based on request configuration.
    ///
    /// For DeepSeek-R1, when thinking is enabled, uses "deepseek-reasoner" model.
    /// Otherwise, uses the standard "deepseek-chat" model or explicitly specified model.
    fn select_model(&self, request: &CompletionRequest) -> String {
        // If thinking is enabled, use the reasoning model
        if request
            .thinking
            .as_ref()
            .map(|t| matches!(t.thinking_type, crate::types::ThinkingType::Enabled))
            .unwrap_or(false)
        {
            "deepseek-reasoner".to_string()
        } else {
            // Use the provided model or fallback to default
            if request.model != "deepseek-chat" && request.model != "deepseek-reasoner" {
                // If a non-standard model was specified, keep it
                request.model.clone()
            } else {
                // Use deepseek-chat (standard model)
                "deepseek-chat".to_string()
            }
        }
    }

    /// Convert unified request to DeepSeek format.
    fn convert_request(&self, request: &CompletionRequest) -> DSRequest {
        let messages = self.build_messages(request);

        // Select model based on thinking configuration
        let model = self.select_model(request);

        // Convert response format for structured output
        let response_format = request.response_format.as_ref().map(|rf| {
            use crate::types::StructuredOutputType;
            match rf.format_type {
                StructuredOutputType::JsonObject => DSResponseFormat::JsonObject,
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        DSResponseFormat::JsonSchema {
                            json_schema: DSJsonSchema {
                                name: schema_def.name.clone(),
                                description: schema_def.description.clone(),
                                schema: schema_def.schema.clone(),
                                strict: Some(schema_def.strict),
                            },
                        }
                    } else {
                        DSResponseFormat::JsonObject
                    }
                }
                StructuredOutputType::Text => DSResponseFormat::Text,
            }
        });

        // Map ThinkingConfig to enable_thinking parameter
        // Only set to false if explicitly disabled, otherwise let the model use its default
        let enable_thinking =
            request
                .thinking
                .as_ref()
                .map(|thinking| match thinking.thinking_type {
                    ThinkingType::Disabled => false,
                    ThinkingType::Enabled => true,
                });

        DSRequest {
            model,
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: Some(request.stream),
            stop: request.stop_sequences.clone(),
            response_format,
            enable_thinking,
        }
    }

    /// Convert DeepSeek response to unified format.
    fn convert_response(&self, response: DSResponse) -> CompletionResponse {
        let mut content = Vec::new();
        let mut stop_reason = StopReason::EndTurn;

        if let Some(choice) = response.choices.into_iter().next() {
            // Handle reasoning content (for deepseek-reasoner)
            if let Some(reasoning) = choice.message.reasoning_content {
                if !reasoning.is_empty() {
                    content.push(ContentBlock::Thinking {
                        thinking: reasoning,
                    });
                }
            }

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
                cache_creation_input_tokens: u.prompt_cache_miss_tokens.unwrap_or(0),
                cache_read_input_tokens: u.prompt_cache_hit_tokens.unwrap_or(0),
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
        if let Ok(error_resp) = serde_json::from_str::<DSErrorResponse>(body) {
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
impl Provider for DeepSeekProvider {
    fn name(&self) -> &str {
        "deepseek"
    }

    fn default_model(&self) -> Option<&str> {
        Some("deepseek-chat")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let ds_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .json(&ds_request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let ds_response: DSResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

        Ok(self.convert_response(ds_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let mut ds_request = self.convert_request(&request);
        ds_request.stream = Some(true);

        let response = self
            .client
            .post(self.api_url())
            .json(&ds_request)
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

                    if let Ok(chunk_resp) = serde_json::from_str::<DSStreamChunk>(data) {
                        if let Some(choice) = chunk_resp.choices.into_iter().next() {
                            if let Some(delta) = choice.delta {
                                // Handle reasoning content (for deepseek-reasoner)
                                if let Some(reasoning) = delta.reasoning_content {
                                    if !reasoning.is_empty() {
                                        yield StreamChunk {
                                            event_type: StreamEventType::ContentBlockDelta,
                                            index: Some(chunk_index),
                                            delta: Some(ContentDelta::Thinking {
                                                thinking: reasoning,
                                            }),
                                            stop_reason: None,
                                            usage: None,
                                        };
                                        chunk_index += 1;
                                    }
                                }

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
struct DSRequest {
    model: String,
    messages: Vec<DSMessage>,
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
    response_format: Option<DSResponseFormat>,
    /// Enable or disable thinking/reasoning for DeepSeek R1 models.
    /// When false, the model skips the reasoning phase for faster responses.
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_thinking: Option<bool>,
}

/// Response format for structured outputs.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum DSResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: DSJsonSchema },
}

/// JSON schema for structured output.
#[derive(Debug, Serialize)]
struct DSJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct DSMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct DSResponse {
    id: String,
    model: String,
    choices: Vec<DSChoice>,
    usage: Option<DSUsage>,
}

#[derive(Debug, Deserialize)]
struct DSChoice {
    message: DSResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DSResponseMessage {
    content: Option<String>,
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DSUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    prompt_cache_hit_tokens: Option<u32>,
    prompt_cache_miss_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct DSStreamChunk {
    choices: Vec<DSStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct DSStreamChoice {
    delta: Option<DSDelta>,
}

#[derive(Debug, Deserialize)]
struct DSDelta {
    content: Option<String>,
    reasoning_content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DSErrorResponse {
    error: Option<DSError>,
}

#[derive(Debug, Deserialize)]
struct DSError {
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = DeepSeekProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "deepseek");
        assert!(provider.supports_tools());
        assert!(provider.supports_streaming());
        assert!(!provider.supports_vision());
    }

    #[test]
    fn test_default_model() {
        let provider = DeepSeekProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.default_model(), Some("deepseek-chat"));
    }

    #[test]
    fn test_api_url() {
        let provider = DeepSeekProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.api_url(), DEEPSEEK_API_URL);
    }

    #[test]
    fn test_api_url_custom_base() {
        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.deepseek.com".to_string());
        let provider = DeepSeekProvider::new(config).unwrap();
        assert_eq!(provider.api_url(), "https://custom.deepseek.com");
    }

    #[test]
    fn test_message_building() {
        let provider = DeepSeekProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("deepseek-chat", vec![Message::user("Hello")])
            .with_system("You are helpful");

        let ds_req = provider.convert_request(&request);

        assert_eq!(ds_req.messages.len(), 2);
        assert_eq!(ds_req.messages[0].role, "system");
        assert_eq!(ds_req.messages[0].content, "You are helpful");
        assert_eq!(ds_req.messages[1].role, "user");
        assert_eq!(ds_req.messages[1].content, "Hello");
    }

    #[test]
    fn test_convert_response() {
        let provider = DeepSeekProvider::with_api_key("test-key").unwrap();

        let response = DSResponse {
            id: "resp-123".to_string(),
            model: "deepseek-chat".to_string(),
            choices: vec![DSChoice {
                message: DSResponseMessage {
                    content: Some("Hello! How can I help?".to_string()),
                    reasoning_content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(DSUsage {
                prompt_tokens: 10,
                completion_tokens: 15,
                prompt_cache_hit_tokens: Some(5),
                prompt_cache_miss_tokens: Some(5),
            }),
        };

        let result = provider.convert_response(response);

        assert_eq!(result.id, "resp-123");
        assert_eq!(result.model, "deepseek-chat");
        assert_eq!(result.content.len(), 1);
        if let ContentBlock::Text { text } = &result.content[0] {
            assert_eq!(text, "Hello! How can I help?");
        } else {
            panic!("Expected text content block");
        }
        assert!(matches!(result.stop_reason, StopReason::EndTurn));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 15);
        assert_eq!(result.usage.cache_read_input_tokens, 5);
        assert_eq!(result.usage.cache_creation_input_tokens, 5);
    }

    #[test]
    fn test_response_with_reasoning() {
        let provider = DeepSeekProvider::with_api_key("test-key").unwrap();

        let response = DSResponse {
            id: "resp-456".to_string(),
            model: "deepseek-reasoner".to_string(),
            choices: vec![DSChoice {
                message: DSResponseMessage {
                    content: Some("The answer is 42.".to_string()),
                    reasoning_content: Some("Let me think step by step...".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };

        let result = provider.convert_response(response);

        assert_eq!(result.content.len(), 2);
        // First should be the reasoning/thinking
        if let ContentBlock::Thinking { thinking } = &result.content[0] {
            assert_eq!(thinking, "Let me think step by step...");
        } else {
            panic!("Expected thinking content block");
        }
        // Second should be the text
        if let ContentBlock::Text { text } = &result.content[1] {
            assert_eq!(text, "The answer is 42.");
        } else {
            panic!("Expected text content block");
        }
    }

    #[test]
    fn test_stop_reason_mapping() {
        let provider = DeepSeekProvider::with_api_key("test-key").unwrap();

        // Test "stop" -> EndTurn
        let response1 = DSResponse {
            id: "1".to_string(),
            model: "model".to_string(),
            choices: vec![DSChoice {
                message: DSResponseMessage {
                    content: Some("Done".to_string()),
                    reasoning_content: None,
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
        let response2 = DSResponse {
            id: "2".to_string(),
            model: "model".to_string(),
            choices: vec![DSChoice {
                message: DSResponseMessage {
                    content: Some("Truncated".to_string()),
                    reasoning_content: None,
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
        let response3 = DSResponse {
            id: "3".to_string(),
            model: "model".to_string(),
            choices: vec![DSChoice {
                message: DSResponseMessage {
                    content: None,
                    reasoning_content: None,
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
    fn test_request_serialization() {
        let request = DSRequest {
            model: "deepseek-chat".to_string(),
            messages: vec![DSMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(1024),
            temperature: Some(0.7),
            top_p: None,
            stream: Some(false),
            stop: None,
            response_format: None,
            enable_thinking: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("deepseek-chat"));
        assert!(json.contains("Hello"));
        assert!(json.contains("1024"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "resp-123",
            "model": "deepseek-chat",
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }"#;

        let response: DSResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "resp-123");
        assert_eq!(
            response.choices[0].message.content,
            Some("Hello!".to_string())
        );
    }

    #[test]
    fn test_model_selection_thinking_enabled() {
        let provider = DeepSeekProvider::with_api_key("test-key").unwrap();

        let request =
            CompletionRequest::new("deepseek-chat", vec![Message::user("Solve this problem")])
                .with_thinking(5000);

        let ds_req = provider.convert_request(&request);

        // When thinking is enabled, should use deepseek-reasoner
        assert_eq!(ds_req.model, "deepseek-reasoner");
    }

    #[test]
    fn test_model_selection_thinking_disabled() {
        use crate::types::ThinkingConfig;

        let provider = DeepSeekProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("deepseek-chat", vec![Message::user("Hello")])
            .with_thinking_config(ThinkingConfig::disabled());

        let ds_req = provider.convert_request(&request);

        // When thinking is disabled, should use deepseek-chat
        assert_eq!(ds_req.model, "deepseek-chat");
    }

    #[test]
    fn test_model_selection_no_thinking() {
        let provider = DeepSeekProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("deepseek-chat", vec![Message::user("Hello")]);

        let ds_req = provider.convert_request(&request);

        // When no thinking config is provided, should use the default model
        assert_eq!(ds_req.model, "deepseek-chat");
    }

    #[test]
    fn test_model_selection_with_reasoning_content_in_response() {
        let provider = DeepSeekProvider::with_api_key("test-key").unwrap();

        let response = DSResponse {
            id: "resp-789".to_string(),
            model: "deepseek-reasoner".to_string(),
            choices: vec![DSChoice {
                message: DSResponseMessage {
                    content: Some("The answer is 8.".to_string()),
                    reasoning_content: Some("Thinking step by step: 4+4=8".to_string()),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(DSUsage {
                prompt_tokens: 20,
                completion_tokens: 30,
                prompt_cache_hit_tokens: None,
                prompt_cache_miss_tokens: None,
            }),
        };

        let result = provider.convert_response(response);

        // Should contain both thinking and text content
        assert_eq!(result.content.len(), 2);
        if let ContentBlock::Thinking { thinking } = &result.content[0] {
            assert_eq!(thinking, "Thinking step by step: 4+4=8");
        } else {
            panic!("Expected thinking content block");
        }
        if let ContentBlock::Text { text } = &result.content[1] {
            assert_eq!(text, "The answer is 8.");
        } else {
            panic!("Expected text content block");
        }
    }
}
