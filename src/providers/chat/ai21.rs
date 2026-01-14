//! AI21 Labs API provider implementation.
//!
//! This module provides access to AI21's Jamba models for chat completions.
//! AI21 uses an OpenAI-compatible API format.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::AI21Provider;
//!
//! // From environment variable
//! let provider = AI21Provider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = AI21Provider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! - `jamba-1.5-large` - Most capable Jamba model
//! - `jamba-1.5-mini` - Smaller, faster Jamba model
//!
//! # Environment Variables
//!
//! - `AI21_API_KEY` - Your AI21 API key

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Message, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

const AI21_API_URL: &str = "https://api.ai21.com/studio/v1/chat/completions";

/// AI21 Labs API provider.
///
/// Provides access to AI21's Jamba family of models using an OpenAI-compatible API.
pub struct AI21Provider {
    config: ProviderConfig,
    client: Client,
}

impl AI21Provider {
    /// Create a new AI21 provider with the given configuration.
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

    /// Create a new AI21 provider from environment variable.
    ///
    /// Reads the API key from `AI21_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("AI21_API_KEY");
        Self::new(config)
    }

    /// Create a new AI21 provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn api_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(AI21_API_URL)
    }

    /// Convert our unified request to AI21's OpenAI-compatible format.
    fn convert_request(&self, request: &CompletionRequest) -> AI21Request {
        let mut messages: Vec<AI21Message> = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(AI21Message {
                role: "system".to_string(),
                content: AI21Content::Text(system.clone()),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Convert messages
        for msg in &request.messages {
            messages.extend(self.convert_message(msg));
        }

        // Convert tools
        let tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| AI21Tool {
                    tool_type: "function".to_string(),
                    function: AI21Function {
                        name: t.name.clone(),
                        description: Some(t.description.clone()),
                        parameters: t.input_schema.clone(),
                    },
                })
                .collect()
        });

        // Convert response format for structured output
        let response_format = request.response_format.as_ref().map(|rf| {
            use crate::types::StructuredOutputType;
            match rf.format_type {
                StructuredOutputType::JsonObject => AI21ResponseFormat::JsonObject,
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        AI21ResponseFormat::JsonSchema {
                            json_schema: AI21JsonSchema {
                                name: schema_def.name.clone(),
                                description: schema_def.description.clone(),
                                schema: schema_def.schema.clone(),
                                strict: Some(schema_def.strict),
                            },
                        }
                    } else {
                        AI21ResponseFormat::JsonObject
                    }
                }
                StructuredOutputType::Text => AI21ResponseFormat::Text,
            }
        });

        AI21Request {
            model: request.model.clone(),
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop_sequences.clone(),
            stream: false,
            tools,
            response_format,
        }
    }

    fn convert_message(&self, message: &Message) -> Vec<AI21Message> {
        let mut result = Vec::new();

        match message.role {
            Role::System => {
                let text = message.text_content();
                if !text.is_empty() {
                    result.push(AI21Message {
                        role: "system".to_string(),
                        content: AI21Content::Text(text),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
            Role::User => {
                // Check if we have tool results
                let tool_results: Vec<_> = message
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolResult {
                            tool_use_id,
                            content,
                            ..
                        } => Some((tool_use_id.clone(), content.clone())),
                        _ => None,
                    })
                    .collect();

                if !tool_results.is_empty() {
                    for (tool_call_id, content) in tool_results {
                        result.push(AI21Message {
                            role: "tool".to_string(),
                            content: AI21Content::Text(content),
                            tool_calls: None,
                            tool_call_id: Some(tool_call_id),
                        });
                    }
                } else {
                    let text = message.text_content();
                    if !text.is_empty() {
                        result.push(AI21Message {
                            role: "user".to_string(),
                            content: AI21Content::Text(text),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                }
            }
            Role::Assistant => {
                let tool_calls: Vec<AI21ToolCall> = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolUse { id, name, input } => Some(AI21ToolCall {
                            id: id.clone(),
                            call_type: "function".to_string(),
                            function: AI21FunctionCall {
                                name: name.clone(),
                                arguments: input.to_string(),
                            },
                        }),
                        _ => None,
                    })
                    .collect();

                let text_content = message.text_content();

                result.push(AI21Message {
                    role: "assistant".to_string(),
                    content: if text_content.is_empty() {
                        AI21Content::Text(String::new())
                    } else {
                        AI21Content::Text(text_content)
                    },
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                    tool_call_id: None,
                });
            }
        }

        result
    }

    fn convert_response(&self, response: AI21Response) -> CompletionResponse {
        let choice = response.choices.into_iter().next().unwrap_or_default();
        let mut content = Vec::new();

        // Add text content
        if let Some(text) = choice.message.content {
            if !text.is_empty() {
                content.push(ContentBlock::Text { text });
            }
        }

        // Add tool calls
        if let Some(tool_calls) = choice.message.tool_calls {
            for tc in tool_calls {
                let input = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or_else(|_| Value::Object(serde_json::Map::new()));
                content.push(ContentBlock::ToolUse {
                    id: tc.id,
                    name: tc.function.name,
                    input,
                });
            }
        }

        let stop_reason = match choice.finish_reason.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            Some("tool_calls") => StopReason::ToolUse,
            Some("content_filter") => StopReason::ContentFilter,
            _ => StopReason::EndTurn,
        };

        let (input_tokens, output_tokens) = match response.usage {
            Some(u) => (u.prompt_tokens, u.completion_tokens),
            None => (0, 0),
        };

        CompletionResponse {
            id: response.id,
            model: response.model,
            content,
            stop_reason,
            usage: Usage {
                input_tokens,
                output_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        }
    }

    async fn handle_error_response(&self, response: reqwest::Response) -> Error {
        let status = response.status().as_u16();

        match response.json::<AI21ErrorResponse>().await {
            Ok(err) => {
                let message = &err.detail;

                match status {
                    401 => Error::auth(message),
                    429 => Error::rate_limited(message, None),
                    400 => Error::invalid_request(message),
                    404 => Error::ModelNotFound(message.clone()),
                    _ => Error::server(status, message),
                }
            }
            Err(_) => Error::server(status, "Unknown error"),
        }
    }
}

#[async_trait]
impl Provider for AI21Provider {
    fn name(&self) -> &str {
        "ai21"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.config.require_api_key()?;

        let mut api_request = self.convert_request(&request);
        api_request.stream = false;

        let response = self
            .client
            .post(self.api_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let ai21_response: AI21Response = response.json().await?;
        Ok(self.convert_response(ai21_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        self.config.require_api_key()?;

        let mut api_request = self.convert_request(&request);
        api_request.stream = true;

        let response = self
            .client
            .post(self.api_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let stream = parse_ai21_stream(response);
        Ok(Box::pin(stream))
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        false // AI21 Jamba doesn't support vision
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supported_models(&self) -> Option<&[&str]> {
        Some(&["jamba-1.5-large", "jamba-1.5-mini"])
    }

    fn default_model(&self) -> Option<&str> {
        Some("jamba-1.5-mini")
    }
}

/// Parse AI21 streaming response (OpenAI-compatible SSE format).
fn parse_ai21_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
    use async_stream::try_stream;
    use futures::StreamExt;

    try_stream! {
        let mut event_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut sent_start = false;

        while let Some(chunk) = event_stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete SSE lines
            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..]; // Skip "data: "

                if data == "[DONE]" {
                    yield StreamChunk {
                        event_type: StreamEventType::MessageStop,
                        index: None,
                        delta: None,
                        stop_reason: None,
                        usage: None,
                    };
                    continue;
                }

                if let Ok(parsed) = serde_json::from_str::<AI21StreamResponse>(data) {
                    if !sent_start {
                        yield StreamChunk {
                            event_type: StreamEventType::MessageStart,
                            index: None,
                            delta: None,
                            stop_reason: None,
                            usage: None,
                        };
                        sent_start = true;
                    }

                    for choice in &parsed.choices {
                        // Handle text content
                        if let Some(ref content) = choice.delta.content {
                            yield StreamChunk {
                                event_type: StreamEventType::ContentBlockDelta,
                                index: Some(0),
                                delta: Some(ContentDelta::Text { text: content.clone() }),
                                stop_reason: None,
                                usage: None,
                            };
                        }

                        // Handle tool calls
                        if let Some(ref tool_calls) = choice.delta.tool_calls {
                            for tc in tool_calls {
                                let idx = tc.index.unwrap_or(0);
                                yield StreamChunk {
                                    event_type: StreamEventType::ContentBlockDelta,
                                    index: Some(idx + 1),
                                    delta: Some(ContentDelta::ToolUse {
                                        id: tc.id.clone(),
                                        name: tc.function.as_ref().and_then(|f| f.name.clone()),
                                        input_json_delta: tc.function.as_ref().and_then(|f| f.arguments.clone()),
                                    }),
                                    stop_reason: None,
                                    usage: None,
                                };
                            }
                        }

                        // Handle finish reason
                        if let Some(ref reason) = choice.finish_reason {
                            let stop_reason = match reason.as_str() {
                                "stop" => StopReason::EndTurn,
                                "length" => StopReason::MaxTokens,
                                "tool_calls" => StopReason::ToolUse,
                                _ => StopReason::EndTurn,
                            };

                            yield StreamChunk {
                                event_type: StreamEventType::MessageDelta,
                                index: None,
                                delta: None,
                                stop_reason: Some(stop_reason),
                                usage: None,
                            };
                        }
                    }

                    // Handle usage
                    if let Some(ref usage) = parsed.usage {
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
    }
}

// ========== AI21 API Types (OpenAI-compatible) ==========

#[derive(Debug, Serialize)]
struct AI21Request {
    model: String,
    messages: Vec<AI21Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AI21Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<AI21ResponseFormat>,
}

/// Response format for structured outputs.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AI21ResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: AI21JsonSchema },
}

/// JSON schema for structured output.
#[derive(Debug, Serialize)]
struct AI21JsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct AI21Message {
    role: String,
    content: AI21Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<AI21ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AI21Content {
    Text(String),
}

#[derive(Debug, Serialize, Deserialize)]
struct AI21Tool {
    #[serde(rename = "type")]
    tool_type: String,
    function: AI21Function,
}

#[derive(Debug, Serialize, Deserialize)]
struct AI21Function {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct AI21ToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: AI21FunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct AI21FunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct AI21Response {
    id: String,
    model: String,
    choices: Vec<AI21Choice>,
    usage: Option<AI21Usage>,
}

#[derive(Debug, Default, Deserialize)]
struct AI21Choice {
    message: AI21ResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct AI21ResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<AI21ToolCall>>,
}

#[derive(Debug, Deserialize)]
struct AI21StreamResponse {
    choices: Vec<AI21StreamChoice>,
    usage: Option<AI21Usage>,
}

#[derive(Debug, Deserialize)]
struct AI21StreamChoice {
    delta: AI21StreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct AI21StreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<AI21StreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct AI21StreamToolCall {
    index: Option<usize>,
    id: Option<String>,
    function: Option<AI21StreamFunction>,
}

#[derive(Debug, Deserialize)]
struct AI21StreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AI21Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AI21ErrorResponse {
    detail: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = AI21Provider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "ai21");
        assert!(provider.supports_tools());
        assert!(!provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_default_model() {
        let provider = AI21Provider::with_api_key("test-key").unwrap();
        assert_eq!(provider.default_model(), Some("jamba-1.5-mini"));
    }

    #[test]
    fn test_supported_models() {
        let provider = AI21Provider::with_api_key("test-key").unwrap();
        let models = provider.supported_models().unwrap();
        assert!(models.contains(&"jamba-1.5-large"));
        assert!(models.contains(&"jamba-1.5-mini"));
    }

    #[test]
    fn test_api_url() {
        let provider = AI21Provider::with_api_key("test-key").unwrap();
        assert_eq!(provider.api_url(), AI21_API_URL);

        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.ai21.com/v1".to_string());
        let provider = AI21Provider::new(config).unwrap();
        assert_eq!(provider.api_url(), "https://custom.ai21.com/v1");
    }

    #[test]
    fn test_request_conversion() {
        let provider = AI21Provider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("jamba-1.5-mini", vec![Message::user("Hello")])
            .with_system("You are helpful")
            .with_max_tokens(1024);

        let ai21_req = provider.convert_request(&request);

        assert_eq!(ai21_req.model, "jamba-1.5-mini");
        assert_eq!(ai21_req.max_tokens, Some(1024));
        assert_eq!(ai21_req.messages.len(), 2); // system + user
    }

    #[test]
    fn test_request_parameters() {
        let provider = AI21Provider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("jamba-1.5-mini", vec![Message::user("Hello")])
            .with_max_tokens(2048)
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_stop_sequences(vec!["STOP".to_string()]);

        let ai21_req = provider.convert_request(&request);

        assert_eq!(ai21_req.max_tokens, Some(2048));
        assert_eq!(ai21_req.temperature, Some(0.7));
        assert_eq!(ai21_req.top_p, Some(0.9));
        assert_eq!(ai21_req.stop, Some(vec!["STOP".to_string()]));
    }

    #[test]
    fn test_message_roles() {
        let provider = AI21Provider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new(
            "jamba-1.5-mini",
            vec![
                Message::user("Hi"),
                Message::assistant("Hello!"),
                Message::user("How are you?"),
            ],
        )
        .with_system("Be helpful");

        let ai21_req = provider.convert_request(&request);

        assert_eq!(ai21_req.messages.len(), 4); // system + 3 messages
        assert_eq!(ai21_req.messages[0].role, "system");
        assert_eq!(ai21_req.messages[1].role, "user");
        assert_eq!(ai21_req.messages[2].role, "assistant");
        assert_eq!(ai21_req.messages[3].role, "user");
    }

    #[test]
    fn test_response_parsing() {
        let provider = AI21Provider::with_api_key("test-key").unwrap();

        let response = AI21Response {
            id: "resp-123".to_string(),
            model: "jamba-1.5-mini".to_string(),
            choices: vec![AI21Choice {
                message: AI21ResponseMessage {
                    content: Some("Hello there!".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(AI21Usage {
                prompt_tokens: 10,
                completion_tokens: 20,
            }),
        };

        let result = provider.convert_response(response);

        assert_eq!(result.id, "resp-123");
        assert_eq!(result.model, "jamba-1.5-mini");
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
        let provider = AI21Provider::with_api_key("test-key").unwrap();

        // Test "stop" -> EndTurn
        let response1 = AI21Response {
            id: "resp-1".to_string(),
            model: "jamba-1.5-mini".to_string(),
            choices: vec![AI21Choice {
                message: AI21ResponseMessage {
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
        let response2 = AI21Response {
            id: "resp-2".to_string(),
            model: "jamba-1.5-mini".to_string(),
            choices: vec![AI21Choice {
                message: AI21ResponseMessage {
                    content: Some("Truncated".to_string()),
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

        // Test "content_filter" -> ContentFilter
        let response3 = AI21Response {
            id: "resp-3".to_string(),
            model: "jamba-1.5-mini".to_string(),
            choices: vec![AI21Choice {
                message: AI21ResponseMessage {
                    content: Some("Filtered".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("content_filter".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.convert_response(response3).stop_reason,
            StopReason::ContentFilter
        ));
    }

    #[test]
    fn test_tool_call_response() {
        let provider = AI21Provider::with_api_key("test-key").unwrap();

        let response = AI21Response {
            id: "resp-tool".to_string(),
            model: "jamba-1.5-mini".to_string(),
            choices: vec![AI21Choice {
                message: AI21ResponseMessage {
                    content: None,
                    tool_calls: Some(vec![AI21ToolCall {
                        id: "call-abc".to_string(),
                        call_type: "function".to_string(),
                        function: AI21FunctionCall {
                            name: "get_weather".to_string(),
                            arguments: r#"{"city": "London"}"#.to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: None,
        };

        let result = provider.convert_response(response);

        assert!(matches!(result.stop_reason, StopReason::ToolUse));
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call-abc");
                assert_eq!(name, "get_weather");
                assert_eq!(input["city"], "London");
            }
            other => {
                panic!("Expected tool use content, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_request_serialization() {
        let request = AI21Request {
            model: "jamba-1.5-mini".to_string(),
            messages: vec![AI21Message {
                role: "user".to_string(),
                content: AI21Content::Text("Hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: None,
            stop: None,
            stream: false,
            tools: None,
            response_format: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("jamba-1.5-mini"));
        assert!(json.contains("\"max_tokens\":1000"));
        assert!(json.contains("\"temperature\":0.7"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "resp-abc123",
            "model": "jamba-1.5-mini",
            "choices": [{
                "message": {"content": "Hi!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }"#;

        let response: AI21Response = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "resp-abc123");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.content, Some("Hi!".to_string()));
        assert_eq!(response.usage.as_ref().unwrap().prompt_tokens, 5);
    }
}
