//! Groq API provider implementation.
//!
//! Groq provides ultra-fast inference for open-source models like Llama and Mixtral.
//! Uses an OpenAI-compatible API.

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

const GROQ_API_URL: &str = "https://api.groq.com/openai/v1/chat/completions";

/// Groq API provider.
///
/// Groq provides ultra-fast inference for models like Llama 3, Mixtral, and Gemma.
///
/// # Example
///
/// ```ignore
/// use modelsuite::providers::groq::GroqProvider;
///
/// let provider = GroqProvider::from_env()?;
/// // or
/// let provider = GroqProvider::with_api_key("gsk_...")?;
/// ```
pub struct GroqProvider {
    config: ProviderConfig,
    client: Client,
}

impl GroqProvider {
    /// Create a new Groq provider with the given configuration.
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

        // Add custom headers
        for (key, value) in &config.custom_headers {
            headers.insert(
                reqwest::header::HeaderName::try_from(key.as_str())
                    .map_err(|_| Error::config(format!("Invalid header name: {}", key)))?,
                value
                    .parse()
                    .map_err(|_| Error::config(format!("Invalid header value for {}", key)))?,
            );
        }

        let client = Client::builder()
            .timeout(config.timeout)
            .default_headers(headers)
            .build()?;

        Ok(Self { config, client })
    }

    /// Create a new Groq provider from environment variable.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("GROQ_API_KEY");
        Self::new(config)
    }

    /// Create a new Groq provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn api_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(GROQ_API_URL)
    }

    /// Convert our unified request to Groq's format (OpenAI-compatible).
    fn convert_request(&self, request: &CompletionRequest) -> GroqRequest {
        let mut messages: Vec<GroqMessage> = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(GroqMessage {
                role: "system".to_string(),
                content: Some(GroqContent::Text(system.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Convert messages
        for msg in &request.messages {
            let groq_msg = self.convert_message(msg);
            messages.push(groq_msg);
        }

        // Convert tools if present
        let tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| GroqTool {
                    r#type: "function".to_string(),
                    function: GroqFunction {
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
                StructuredOutputType::JsonObject => GroqResponseFormat::JsonObject,
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        GroqResponseFormat::JsonSchema {
                            json_schema: GroqJsonSchema {
                                name: schema_def.name.clone(),
                                description: schema_def.description.clone(),
                                schema: schema_def.schema.clone(),
                                strict: Some(schema_def.strict),
                            },
                        }
                    } else {
                        GroqResponseFormat::JsonObject
                    }
                }
                StructuredOutputType::Text => GroqResponseFormat::Text,
            }
        });

        GroqRequest {
            model: request.model.clone(),
            messages,
            tools,
            tool_choice: if request.tools.is_some() {
                Some("auto".to_string())
            } else {
                None
            },
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop_sequences.clone(),
            stream: Some(request.stream),
            response_format,
        }
    }

    /// Convert a single message to Groq format.
    fn convert_message(&self, msg: &Message) -> GroqMessage {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        };

        // Check for tool results first
        for block in &msg.content {
            if let ContentBlock::ToolResult {
                tool_use_id,
                content,
                ..
            } = block
            {
                return GroqMessage {
                    role: "tool".to_string(),
                    content: Some(GroqContent::Text(content.clone())),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                };
            }
        }

        // Check for tool use (assistant messages with tool calls)
        let tool_calls: Vec<GroqToolCall> = msg
            .content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::ToolUse { id, name, input } = block {
                    Some(GroqToolCall {
                        id: id.clone(),
                        r#type: "function".to_string(),
                        function: GroqFunctionCall {
                            name: name.clone(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        },
                    })
                } else {
                    None
                }
            })
            .collect();

        // Extract text content (Groq doesn't support vision)
        let text_content: String = msg
            .content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::Text { text } = block {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n");

        GroqMessage {
            role: role.to_string(),
            content: if text_content.is_empty() {
                None
            } else {
                Some(GroqContent::Text(text_content))
            },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
        }
    }

    /// Convert Groq response to our unified format.
    fn convert_response(&self, response: GroqResponse) -> CompletionResponse {
        let choice = response.choices.into_iter().next();

        let mut content = Vec::new();
        let mut stop_reason = StopReason::EndTurn;

        if let Some(choice) = choice {
            // Handle text content
            if let Some(text) = choice.message.content {
                content.push(ContentBlock::Text { text });
            }

            // Handle tool calls
            if let Some(tool_calls) = choice.message.tool_calls {
                for tc in tool_calls {
                    let input: Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or(Value::Null);
                    content.push(ContentBlock::ToolUse {
                        id: tc.id,
                        name: tc.function.name,
                        input,
                    });
                }
                stop_reason = StopReason::ToolUse;
            }

            // Determine stop reason
            if let Some(finish_reason) = choice.finish_reason {
                stop_reason = match finish_reason.as_str() {
                    "stop" => StopReason::EndTurn,
                    "length" => StopReason::MaxTokens,
                    "tool_calls" => StopReason::ToolUse,
                    _ => StopReason::EndTurn,
                };
            }
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
}

#[async_trait]
impl Provider for GroqProvider {
    fn name(&self) -> &str {
        "groq"
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        false // Groq doesn't support vision yet
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let groq_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .json(&groq_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(status.as_u16(), error_text));
        }

        let groq_response: GroqResponse = response.json().await?;
        Ok(self.convert_response(groq_response))
    }

    async fn complete_stream(
        &self,
        mut request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        request.stream = true;
        let groq_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .json(&groq_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(status.as_u16(), error_text));
        }

        let stream = parse_groq_stream(response);
        Ok(Box::pin(stream))
    }
}

/// Parse Groq SSE stream into our unified StreamChunk format.
fn parse_groq_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
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

                if let Ok(parsed) = serde_json::from_str::<GroqStreamEvent>(data) {
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

                    for choice in parsed.choices {
                        // Handle text content
                        if let Some(ref delta) = choice.delta {
                            if let Some(ref content) = delta.content {
                                yield StreamChunk {
                                    event_type: StreamEventType::ContentBlockDelta,
                                    index: Some(0),
                                    delta: Some(ContentDelta::Text { text: content.clone() }),
                                    stop_reason: None,
                                    usage: None,
                                };
                            }

                            // Handle tool calls
                            if let Some(ref tool_calls) = delta.tool_calls {
                                for tc in tool_calls {
                                    yield StreamChunk {
                                        event_type: StreamEventType::ContentBlockDelta,
                                        index: Some(1),
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
                        }

                        // Handle finish reason
                        if let Some(ref finish_reason) = choice.finish_reason {
                            let stop_reason = match finish_reason.as_str() {
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

// ============================================================================
// Groq API Types (OpenAI-compatible)
// ============================================================================

#[derive(Debug, Serialize)]
struct GroqRequest {
    model: String,
    messages: Vec<GroqMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GroqTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<GroqResponseFormat>,
}

/// Response format for structured outputs.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum GroqResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: GroqJsonSchema },
}

/// JSON schema for structured output.
#[derive(Debug, Serialize)]
struct GroqJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct GroqMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<GroqContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<GroqToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
enum GroqContent {
    Text(String),
}

#[derive(Debug, Serialize)]
struct GroqTool {
    r#type: String,
    function: GroqFunction,
}

#[derive(Debug, Serialize)]
struct GroqFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct GroqToolCall {
    id: String,
    r#type: String,
    function: GroqFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct GroqFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct GroqResponse {
    id: String,
    model: String,
    choices: Vec<GroqChoice>,
    usage: Option<GroqUsage>,
}

#[derive(Debug, Deserialize)]
struct GroqChoice {
    message: GroqResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GroqResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<GroqToolCall>>,
}

#[derive(Debug, Deserialize)]
struct GroqUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// Streaming types
#[derive(Debug, Deserialize)]
struct GroqStreamEvent {
    choices: Vec<GroqStreamChoice>,
    usage: Option<GroqUsage>,
}

#[derive(Debug, Deserialize)]
struct GroqStreamChoice {
    delta: Option<GroqStreamDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GroqStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<GroqStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct GroqStreamToolCall {
    id: Option<String>,
    function: Option<GroqStreamFunction>,
}

#[derive(Debug, Deserialize)]
struct GroqStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{CompletionRequest, Message};

    #[test]
    fn test_provider_creation() {
        let provider = GroqProvider::with_api_key("test-key");
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.name(), "groq");
        assert!(provider.supports_tools());
        assert!(!provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_request_conversion() {
        let provider = GroqProvider::with_api_key("test-key").unwrap();

        let request =
            CompletionRequest::new("llama-3.1-70b-versatile", vec![Message::user("Hello!")])
                .with_system("You are a helpful assistant.")
                .with_max_tokens(1024);

        let groq_request = provider.convert_request(&request);

        assert_eq!(groq_request.model, "llama-3.1-70b-versatile");
        assert_eq!(groq_request.messages.len(), 2); // system + user
        assert_eq!(groq_request.messages[0].role, "system");
        assert_eq!(groq_request.messages[1].role, "user");
        assert_eq!(groq_request.max_tokens, Some(1024));
    }

    #[test]
    fn test_response_conversion() {
        let provider = GroqProvider::with_api_key("test-key").unwrap();

        let groq_response = GroqResponse {
            id: "chatcmpl-123".to_string(),
            model: "llama-3.1-70b-versatile".to_string(),
            choices: vec![GroqChoice {
                message: GroqResponseMessage {
                    content: Some("Hello! How can I help you?".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(GroqUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
            }),
        };

        let response = provider.convert_response(groq_response);

        assert_eq!(response.id, "chatcmpl-123");
        assert_eq!(response.model, "llama-3.1-70b-versatile");
        assert_eq!(response.content.len(), 1);
        assert!(matches!(response.stop_reason, StopReason::EndTurn));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 20);
    }

    #[test]
    fn test_api_url() {
        let provider = GroqProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.api_url(), GROQ_API_URL);

        let config = ProviderConfig::new("test-key").with_base_url("https://custom.groq.com/v1");
        let provider = GroqProvider::new(config).unwrap();
        assert_eq!(provider.api_url(), "https://custom.groq.com/v1");
    }

    #[test]
    fn test_request_parameters() {
        let provider = GroqProvider::with_api_key("test-key").unwrap();

        let request =
            CompletionRequest::new("llama-3.1-70b-versatile", vec![Message::user("Hello")])
                .with_max_tokens(2048)
                .with_temperature(0.7)
                .with_top_p(0.9)
                .with_stop_sequences(vec!["STOP".to_string()]);

        let groq_req = provider.convert_request(&request);

        assert_eq!(groq_req.max_tokens, Some(2048));
        assert_eq!(groq_req.temperature, Some(0.7));
        assert_eq!(groq_req.top_p, Some(0.9));
        assert_eq!(groq_req.stop, Some(vec!["STOP".to_string()]));
    }

    #[test]
    fn test_stop_reason_mapping() {
        let provider = GroqProvider::with_api_key("test-key").unwrap();

        // Test "stop" -> EndTurn
        let response1 = GroqResponse {
            id: "resp-1".to_string(),
            model: "llama-3.1-70b-versatile".to_string(),
            choices: vec![GroqChoice {
                message: GroqResponseMessage {
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
        let response2 = GroqResponse {
            id: "resp-2".to_string(),
            model: "llama-3.1-70b-versatile".to_string(),
            choices: vec![GroqChoice {
                message: GroqResponseMessage {
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

        // Test "tool_calls" -> ToolUse
        let response3 = GroqResponse {
            id: "resp-3".to_string(),
            model: "llama-3.1-70b-versatile".to_string(),
            choices: vec![GroqChoice {
                message: GroqResponseMessage {
                    content: None,
                    tool_calls: Some(vec![GroqToolCall {
                        id: "call-123".to_string(),
                        r#type: "function".to_string(),
                        function: GroqFunctionCall {
                            name: "get_weather".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
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
        let provider = GroqProvider::with_api_key("test-key").unwrap();

        let response = GroqResponse {
            id: "resp-tool".to_string(),
            model: "llama-3.1-70b-versatile".to_string(),
            choices: vec![GroqChoice {
                message: GroqResponseMessage {
                    content: None,
                    tool_calls: Some(vec![GroqToolCall {
                        id: "call-abc".to_string(),
                        r#type: "function".to_string(),
                        function: GroqFunctionCall {
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
        if let ContentBlock::ToolUse { id, name, input } = &result.content[0] {
            assert_eq!(id, "call-abc");
            assert_eq!(name, "get_weather");
            assert_eq!(input["city"], "London");
        } else {
            panic!("Expected tool use content");
        }
    }

    #[test]
    fn test_request_serialization() {
        let request = GroqRequest {
            model: "llama-3.1-70b-versatile".to_string(),
            messages: vec![GroqMessage {
                role: "user".to_string(),
                content: Some(GroqContent::Text("Hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: None,
            tool_choice: None,
            max_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: None,
            stop: None,
            stream: Some(false),
            response_format: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("llama-3.1-70b-versatile"));
        assert!(json.contains("\"max_tokens\":1000"));
        assert!(json.contains("\"temperature\":0.7"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "chatcmpl-abc123",
            "model": "llama-3.1-70b-versatile",
            "choices": [{
                "message": {"content": "Hi!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10}
        }"#;

        let response: GroqResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "chatcmpl-abc123");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(response.choices[0].message.content, Some("Hi!".to_string()));
        assert_eq!(response.usage.as_ref().unwrap().prompt_tokens, 5);
    }

    #[test]
    fn test_multi_turn_conversation() {
        let provider = GroqProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new(
            "llama-3.1-70b-versatile",
            vec![
                Message::user("Hello"),
                Message::assistant("Hi there!"),
                Message::user("How are you?"),
            ],
        )
        .with_system("Be friendly");

        let groq_req = provider.convert_request(&request);

        assert_eq!(groq_req.messages.len(), 4);
        assert_eq!(groq_req.messages[0].role, "system");
        assert_eq!(groq_req.messages[1].role, "user");
        assert_eq!(groq_req.messages[2].role, "assistant");
        assert_eq!(groq_req.messages[3].role, "user");
    }
}
