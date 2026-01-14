//! Ollama provider implementation for local models.
//!
//! Ollama allows running LLMs locally. This provider connects to a local
//! or remote Ollama server.

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

const DEFAULT_OLLAMA_URL: &str = "http://localhost:11434";

/// Ollama provider for local model inference.
///
/// Supports models like:
/// - llama3.2
/// - mistral
/// - codellama
/// - phi3
/// - qwen2.5
/// - And many more from the Ollama library
pub struct OllamaProvider {
    _config: ProviderConfig,
    client: Client,
    base_url: String,
}

impl OllamaProvider {
    /// Create a new Ollama provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let base_url = config
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_OLLAMA_URL.to_string());

        let mut headers = reqwest::header::HeaderMap::new();
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

        Ok(Self {
            _config: config,
            client,
            base_url,
        })
    }

    /// Create a new Ollama provider with default settings (localhost:11434).
    pub fn default_local() -> Result<Self> {
        Self::new(ProviderConfig::default())
    }

    /// Create a new Ollama provider connecting to a specific URL.
    pub fn with_url(url: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::default().with_base_url(url);
        Self::new(config)
    }

    fn chat_url(&self) -> String {
        format!("{}/api/chat", self.base_url)
    }

    /// Convert our unified request to Ollama's format.
    fn convert_request(&self, request: &CompletionRequest) -> OllamaRequest {
        let mut messages: Vec<OllamaMessage> = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(OllamaMessage {
                role: "system".to_string(),
                content: system.clone(),
                images: None,
                tool_calls: None,
            });
        }

        // Convert messages
        for msg in &request.messages {
            messages.extend(self.convert_message(msg));
        }

        // Convert tools if present
        let tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| OllamaTool {
                    tool_type: "function".to_string(),
                    function: OllamaFunction {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.input_schema.clone(),
                    },
                })
                .collect()
        });

        // Build options
        let options = OllamaOptions {
            temperature: request.temperature,
            top_p: request.top_p,
            num_predict: request.max_tokens.map(|t| t as i32),
            stop: request.stop_sequences.clone(),
        };

        // Handle structured output - Ollama uses format: "json" for JSON mode
        let format = request.response_format.as_ref().and_then(|rf| {
            use crate::types::StructuredOutputType;
            match rf.format_type {
                StructuredOutputType::JsonObject | StructuredOutputType::JsonSchema => {
                    Some("json".to_string())
                }
                StructuredOutputType::Text => None,
            }
        });

        OllamaRequest {
            model: request.model.clone(),
            messages,
            stream: request.stream,
            tools,
            options: Some(options),
            format,
            keep_alive: None,
        }
    }

    fn convert_message(&self, message: &Message) -> Vec<OllamaMessage> {
        let mut result = Vec::new();

        match message.role {
            Role::System => {
                let text = message.text_content();
                if !text.is_empty() {
                    result.push(OllamaMessage {
                        role: "system".to_string(),
                        content: text,
                        images: None,
                        tool_calls: None,
                    });
                }
            }
            Role::User => {
                // Check for tool results first
                let tool_results: Vec<_> = message
                    .content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolResult {
                            tool_use_id: _,
                            content,
                            ..
                        } => Some(content.clone()),
                        _ => None,
                    })
                    .collect();

                if !tool_results.is_empty() {
                    // Ollama uses "tool" role for tool responses
                    for content in tool_results {
                        result.push(OllamaMessage {
                            role: "tool".to_string(),
                            content,
                            images: None,
                            tool_calls: None,
                        });
                    }
                } else {
                    // Extract images (base64)
                    let images: Vec<String> = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Image { data, .. } => Some(data.clone()),
                            _ => None,
                        })
                        .collect();

                    let text = message.text_content();

                    result.push(OllamaMessage {
                        role: "user".to_string(),
                        content: text,
                        images: if images.is_empty() {
                            None
                        } else {
                            Some(images)
                        },
                        tool_calls: None,
                    });
                }
            }
            Role::Assistant => {
                // Check for tool calls
                let tool_calls: Vec<OllamaToolCall> = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolUse { id, name, input } => Some(OllamaToolCall {
                            id: Some(id.clone()),
                            function: OllamaFunctionCall {
                                name: name.clone(),
                                arguments: input.clone(),
                            },
                        }),
                        _ => None,
                    })
                    .collect();

                let text = message.text_content();

                result.push(OllamaMessage {
                    role: "assistant".to_string(),
                    content: text,
                    images: None,
                    tool_calls: if tool_calls.is_empty() {
                        None
                    } else {
                        Some(tool_calls)
                    },
                });
            }
        }

        result
    }

    fn convert_response(&self, response: OllamaResponse) -> CompletionResponse {
        let mut content = Vec::new();

        // Add text content
        if !response.message.content.is_empty() {
            content.push(ContentBlock::Text {
                text: response.message.content,
            });
        }

        // Add tool calls
        if let Some(tool_calls) = response.message.tool_calls {
            for tc in tool_calls {
                content.push(ContentBlock::ToolUse {
                    id: tc.id.unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
                    name: tc.function.name,
                    input: tc.function.arguments,
                });
            }
        }

        // Determine stop reason
        let stop_reason = if content
            .iter()
            .any(|c| matches!(c, ContentBlock::ToolUse { .. }))
        {
            StopReason::ToolUse
        } else if response.done_reason.as_deref() == Some("length") {
            StopReason::MaxTokens
        } else {
            StopReason::EndTurn
        };

        CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: response.model,
            content,
            stop_reason,
            usage: Usage {
                input_tokens: response.prompt_eval_count.unwrap_or(0),
                output_tokens: response.eval_count.unwrap_or(0),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        }
    }

    async fn handle_error_response(&self, response: reqwest::Response) -> Error {
        let status = response.status().as_u16();

        match response.text().await {
            Ok(text) => {
                // Try to parse as JSON error
                if let Ok(err) = serde_json::from_str::<OllamaError>(&text) {
                    if err.error.contains("model") && err.error.contains("not found") {
                        return Error::ModelNotFound(err.error);
                    }
                    return Error::other(err.error);
                }
                Error::server(status, text)
            }
            Err(_) => Error::server(status, "Unknown error"),
        }
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    fn name(&self) -> &str {
        "ollama"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let mut api_request = self.convert_request(&request);
        api_request.stream = false;

        let response = self
            .client
            .post(self.chat_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let ollama_response: OllamaResponse = response.json().await?;
        Ok(self.convert_response(ollama_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let mut api_request = self.convert_request(&request);
        api_request.stream = true;

        let response = self
            .client
            .post(self.chat_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let stream = parse_ollama_stream(response);
        Ok(Box::pin(stream))
    }

    fn supports_tools(&self) -> bool {
        true // Many Ollama models support tools
    }

    fn supports_vision(&self) -> bool {
        true // Models like llava, bakllava support vision
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn default_model(&self) -> Option<&str> {
        Some("llama3.2")
    }
}

/// Parse Ollama's NDJSON stream.
fn parse_ollama_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
    use async_stream::try_stream;
    use futures::StreamExt;

    try_stream! {
        let mut event_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut sent_start = false;
        let mut total_input_tokens = 0u32;
        let mut total_output_tokens = 0u32;

        while let Some(chunk) = event_stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete JSON lines
            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                if let Ok(parsed) = serde_json::from_str::<OllamaStreamResponse>(&line) {
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

                    // Text content
                    if !parsed.message.content.is_empty() {
                        yield StreamChunk {
                            event_type: StreamEventType::ContentBlockDelta,
                            index: Some(0),
                            delta: Some(ContentDelta::Text {
                                text: parsed.message.content,
                            }),
                            stop_reason: None,
                            usage: None,
                        };
                    }

                    // Tool calls
                    if let Some(tool_calls) = parsed.message.tool_calls {
                        for (idx, tc) in tool_calls.into_iter().enumerate() {
                            yield StreamChunk {
                                event_type: StreamEventType::ContentBlockDelta,
                                index: Some(idx + 1),
                                delta: Some(ContentDelta::ToolUse {
                                    id: tc.id,
                                    name: Some(tc.function.name),
                                    input_json_delta: Some(tc.function.arguments.to_string()),
                                }),
                                stop_reason: None,
                                usage: None,
                            };
                        }
                    }

                    // Track usage
                    if let Some(prompt_eval_count) = parsed.prompt_eval_count {
                        total_input_tokens = prompt_eval_count;
                    }
                    if let Some(eval_count) = parsed.eval_count {
                        total_output_tokens = eval_count;
                    }

                    // Done
                    if parsed.done {
                        let stop_reason = match parsed.done_reason.as_deref() {
                            Some("length") => StopReason::MaxTokens,
                            _ => StopReason::EndTurn,
                        };

                        yield StreamChunk {
                            event_type: StreamEventType::MessageDelta,
                            index: None,
                            delta: None,
                            stop_reason: Some(stop_reason),
                            usage: Some(Usage {
                                input_tokens: total_input_tokens,
                                output_tokens: total_output_tokens,
                                cache_creation_input_tokens: 0,
                                cache_read_input_tokens: 0,
                            }),
                        };

                        yield StreamChunk {
                            event_type: StreamEventType::MessageStop,
                            index: None,
                            delta: None,
                            stop_reason: None,
                            usage: None,
                        };
                    }
                }
            }
        }
    }
}

// Ollama API types

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OllamaTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    keep_alive: Option<String>,
}

#[derive(Debug, Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OllamaFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaFunction {
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaToolCall {
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    function: OllamaFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OllamaFunctionCall {
    name: String,
    arguments: Value,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    model: String,
    message: OllamaResponseMessage,
    #[serde(rename = "done")]
    _done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamResponse {
    message: OllamaStreamMessage,
    done: bool,
    #[serde(default)]
    done_reason: Option<String>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
    #[serde(default)]
    eval_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Option<Vec<OllamaToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OllamaError {
    error: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = OllamaProvider::default_local().unwrap();
        assert_eq!(provider.name(), "ollama");
        assert!(provider.supports_tools());
        assert!(provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_default_model() {
        let provider = OllamaProvider::default_local().unwrap();
        assert_eq!(provider.default_model(), Some("llama3.2"));
    }

    #[test]
    fn test_custom_url() {
        let provider = OllamaProvider::with_url("http://192.168.1.100:11434").unwrap();
        assert_eq!(provider.base_url, "http://192.168.1.100:11434");
    }

    #[test]
    fn test_chat_url() {
        let provider = OllamaProvider::default_local().unwrap();
        assert_eq!(provider.chat_url(), "http://localhost:11434/api/chat");

        let provider = OllamaProvider::with_url("http://remote:11434").unwrap();
        assert_eq!(provider.chat_url(), "http://remote:11434/api/chat");
    }

    #[test]
    fn test_request_conversion() {
        let provider = OllamaProvider::default_local().unwrap();
        let request = CompletionRequest::new("llama3.2", vec![Message::user("Hello")])
            .with_system("You are helpful")
            .with_max_tokens(1024);

        let ollama_req = provider.convert_request(&request);

        assert_eq!(ollama_req.model, "llama3.2");
        assert_eq!(ollama_req.options.as_ref().unwrap().num_predict, Some(1024));
        assert_eq!(ollama_req.messages.len(), 2); // system + user
    }

    #[test]
    fn test_request_parameters() {
        let provider = OllamaProvider::default_local().unwrap();
        let request = CompletionRequest::new("llama3.2", vec![Message::user("Hello")])
            .with_temperature(0.7)
            .with_top_p(0.9)
            .with_stop_sequences(vec!["STOP".to_string()]);

        let ollama_req = provider.convert_request(&request);

        let options = ollama_req.options.unwrap();
        assert_eq!(options.temperature, Some(0.7));
        assert_eq!(options.top_p, Some(0.9));
        assert_eq!(options.stop, Some(vec!["STOP".to_string()]));
    }

    #[test]
    fn test_response_parsing() {
        let provider = OllamaProvider::default_local().unwrap();

        let response = OllamaResponse {
            model: "llama3.2".to_string(),
            message: OllamaResponseMessage {
                content: "Hello there!".to_string(),
                tool_calls: None,
            },
            _done: true,
            done_reason: Some("stop".to_string()),
            prompt_eval_count: Some(10),
            eval_count: Some(20),
        };

        let result = provider.convert_response(response);

        assert_eq!(result.model, "llama3.2");
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
        let provider = OllamaProvider::default_local().unwrap();

        // Test "stop" -> EndTurn (default)
        let response1 = OllamaResponse {
            model: "llama3.2".to_string(),
            message: OllamaResponseMessage {
                content: "Done".to_string(),
                tool_calls: None,
            },
            _done: true,
            done_reason: Some("stop".to_string()),
            prompt_eval_count: None,
            eval_count: None,
        };
        assert!(matches!(
            provider.convert_response(response1).stop_reason,
            StopReason::EndTurn
        ));

        // Test "length" -> MaxTokens
        let response2 = OllamaResponse {
            model: "llama3.2".to_string(),
            message: OllamaResponseMessage {
                content: "Truncated".to_string(),
                tool_calls: None,
            },
            _done: true,
            done_reason: Some("length".to_string()),
            prompt_eval_count: None,
            eval_count: None,
        };
        assert!(matches!(
            provider.convert_response(response2).stop_reason,
            StopReason::MaxTokens
        ));
    }

    #[test]
    fn test_tool_call_response() {
        let provider = OllamaProvider::default_local().unwrap();

        let response = OllamaResponse {
            model: "llama3.2".to_string(),
            message: OllamaResponseMessage {
                content: String::new(),
                tool_calls: Some(vec![OllamaToolCall {
                    id: Some("call-123".to_string()),
                    function: OllamaFunctionCall {
                        name: "get_weather".to_string(),
                        arguments: serde_json::json!({"city": "London"}),
                    },
                }]),
            },
            _done: true,
            done_reason: None,
            prompt_eval_count: None,
            eval_count: None,
        };

        let result = provider.convert_response(response);

        assert!(matches!(result.stop_reason, StopReason::ToolUse));
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlock::ToolUse { id, name, input } => {
                assert_eq!(id, "call-123");
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
        let request = OllamaRequest {
            model: "llama3.2".to_string(),
            messages: vec![OllamaMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
                images: None,
                tool_calls: None,
            }],
            stream: false,
            tools: None,
            options: Some(OllamaOptions {
                temperature: Some(0.7),
                top_p: None,
                num_predict: Some(1000),
                stop: None,
            }),
            format: None,
            keep_alive: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("llama3.2"));
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"num_predict\":1000"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "model": "llama3.2",
            "message": {"content": "Hi!"},
            "done": true,
            "done_reason": "stop",
            "prompt_eval_count": 5,
            "eval_count": 10
        }"#;

        let response: OllamaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.model, "llama3.2");
        assert_eq!(response.message.content, "Hi!");
        assert!(response._done);
        assert_eq!(response.prompt_eval_count, Some(5));
    }

    #[test]
    fn test_multi_turn_conversation() {
        let provider = OllamaProvider::default_local().unwrap();

        let request = CompletionRequest::new(
            "llama3.2",
            vec![
                Message::user("Hello"),
                Message::assistant("Hi there!"),
                Message::user("How are you?"),
            ],
        )
        .with_system("Be friendly");

        let ollama_req = provider.convert_request(&request);

        assert_eq!(ollama_req.messages.len(), 4);
        assert_eq!(ollama_req.messages[0].role, "system");
        assert_eq!(ollama_req.messages[1].role, "user");
        assert_eq!(ollama_req.messages[2].role, "assistant");
        assert_eq!(ollama_req.messages[3].role, "user");
    }

    #[test]
    fn test_json_format_mode() {
        use crate::types::{StructuredOutput, StructuredOutputType};

        let provider = OllamaProvider::default_local().unwrap();
        let request = CompletionRequest::new("llama3.2", vec![Message::user("Return JSON")])
            .with_response_format(StructuredOutput {
                format_type: StructuredOutputType::JsonObject,
                json_schema: None,
            });

        let ollama_req = provider.convert_request(&request);
        assert_eq!(ollama_req.format, Some("json".to_string()));
    }
}
