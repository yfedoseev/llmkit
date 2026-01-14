//! OpenRouter API provider implementation.
//!
//! OpenRouter provides access to 100+ models through a single API endpoint.
//! It uses an OpenAI-compatible format with some extensions.

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
    StreamChunk, StreamEventType, ThinkingEffort, ThinkingType, Usage,
};

const OPENROUTER_API_URL: &str = "https://openrouter.ai/api/v1/chat/completions";

/// OpenRouter API provider.
///
/// OpenRouter provides access to models from multiple providers including:
/// - Anthropic (Claude)
/// - OpenAI (GPT)
/// - Google (Gemini)
/// - Meta (Llama)
/// - Mistral
/// - And many more
pub struct OpenRouterProvider {
    config: ProviderConfig,
    client: Client,
    /// Optional app name for OpenRouter analytics
    app_name: Option<String>,
    /// Optional site URL for OpenRouter analytics
    site_url: Option<String>,
}

impl OpenRouterProvider {
    /// Create a new OpenRouter provider with the given configuration.
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

        Ok(Self {
            config,
            client,
            app_name: None,
            site_url: None,
        })
    }

    /// Create a new OpenRouter provider from environment variable.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("OPENROUTER_API_KEY");
        Self::new(config)
    }

    /// Create a new OpenRouter provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    /// Set the app name for OpenRouter analytics.
    pub fn with_app_name(mut self, name: impl Into<String>) -> Self {
        self.app_name = Some(name.into());
        self
    }

    /// Set the site URL for OpenRouter analytics.
    pub fn with_site_url(mut self, url: impl Into<String>) -> Self {
        self.site_url = Some(url.into());
        self
    }

    fn api_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or(OPENROUTER_API_URL)
    }

    /// Convert our unified request to OpenRouter's format.
    fn convert_request(&self, request: &CompletionRequest) -> OpenRouterRequest {
        let mut messages: Vec<OpenRouterMessage> = Vec::new();

        // Check if we need to inject /no_think for Qwen3 models
        // OpenRouter's reasoning.effort: "none" doesn't work for Qwen3, so we need the prompt directive
        let is_qwen3 = request.model.to_lowercase().contains("qwen3");
        let thinking_disabled = request
            .thinking
            .as_ref()
            .map(|t| !t.is_enabled())
            .unwrap_or(false);
        let inject_no_think = is_qwen3 && thinking_disabled;

        // Add system message if present (with /no_think prefix for Qwen3 if needed)
        if let Some(ref system) = request.system {
            let system_content = if inject_no_think {
                format!("/no_think\n\n{}", system)
            } else {
                system.clone()
            };
            messages.push(OpenRouterMessage {
                role: "system".to_string(),
                content: Some(OpenRouterContent::Text(system_content)),
                tool_calls: None,
                tool_call_id: None,
            });
        } else if inject_no_think {
            // No system message but we need /no_think - create a minimal system message
            messages.push(OpenRouterMessage {
                role: "system".to_string(),
                content: Some(OpenRouterContent::Text("/no_think".to_string())),
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
                .map(|t| OpenRouterTool {
                    tool_type: "function".to_string(),
                    function: OpenRouterFunction {
                        name: t.name.clone(),
                        description: Some(t.description.clone()),
                        parameters: t.input_schema.clone(),
                    },
                })
                .collect()
        });

        // Convert response format for structured output (passthrough to underlying provider)
        let response_format = request.response_format.as_ref().map(|rf| {
            use crate::types::StructuredOutputType;
            match rf.format_type {
                StructuredOutputType::JsonObject => ORResponseFormat::JsonObject,
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        ORResponseFormat::JsonSchema {
                            json_schema: ORJsonSchema {
                                name: schema_def.name.clone(),
                                description: schema_def.description.clone(),
                                schema: schema_def.schema.clone(),
                                strict: Some(schema_def.strict),
                            },
                        }
                    } else {
                        ORResponseFormat::JsonObject
                    }
                }
                StructuredOutputType::Text => ORResponseFormat::Text,
            }
        });

        // Convert ThinkingConfig to OpenRouter reasoning parameter
        let reasoning = request.thinking.as_ref().map(|thinking| {
            let effort = match thinking.thinking_type {
                ThinkingType::Disabled => Some("none".to_string()),
                ThinkingType::Enabled => thinking.effort.as_ref().map(|e| match e {
                    ThinkingEffort::Low => "low".to_string(),
                    ThinkingEffort::Medium => "medium".to_string(),
                    ThinkingEffort::High => "high".to_string(),
                    ThinkingEffort::Max => "max".to_string(),
                }),
            };

            OpenRouterReasoning {
                effort,
                max_tokens: thinking.budget_tokens,
                exclude: if thinking.exclude_from_response {
                    Some(true)
                } else {
                    None
                },
            }
        });

        OpenRouterRequest {
            model: request.model.clone(),
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop_sequences.clone(),
            stream: request.stream,
            tools,
            stream_options: if request.stream {
                Some(StreamOptions {
                    include_usage: true,
                })
            } else {
                None
            },
            response_format,
            // OpenRouter specific
            transforms: None,
            route: None,
            reasoning,
        }
    }

    fn convert_message(&self, message: &Message) -> Vec<OpenRouterMessage> {
        let mut result = Vec::new();

        match message.role {
            Role::System => {
                let text = message.text_content();
                if !text.is_empty() {
                    result.push(OpenRouterMessage {
                        role: "system".to_string(),
                        content: Some(OpenRouterContent::Text(text)),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
            }
            Role::User => {
                // Check for tool results
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
                        result.push(OpenRouterMessage {
                            role: "tool".to_string(),
                            content: Some(OpenRouterContent::Text(content)),
                            tool_calls: None,
                            tool_call_id: Some(tool_call_id),
                        });
                    }
                } else {
                    let content_parts: Vec<OpenRouterContentPart> = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text { text } => {
                                Some(OpenRouterContentPart::Text { text: text.clone() })
                            }
                            ContentBlock::Image { media_type, data } => {
                                Some(OpenRouterContentPart::ImageUrl {
                                    image_url: ImageUrl {
                                        url: format!("data:{};base64,{}", media_type, data),
                                    },
                                })
                            }
                            ContentBlock::ImageUrl { url } => {
                                Some(OpenRouterContentPart::ImageUrl {
                                    image_url: ImageUrl { url: url.clone() },
                                })
                            }
                            _ => None,
                        })
                        .collect();

                    if content_parts.len() == 1 {
                        if let OpenRouterContentPart::Text { text } = &content_parts[0] {
                            result.push(OpenRouterMessage {
                                role: "user".to_string(),
                                content: Some(OpenRouterContent::Text(text.clone())),
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        } else {
                            result.push(OpenRouterMessage {
                                role: "user".to_string(),
                                content: Some(OpenRouterContent::Parts(content_parts)),
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        }
                    } else if !content_parts.is_empty() {
                        result.push(OpenRouterMessage {
                            role: "user".to_string(),
                            content: Some(OpenRouterContent::Parts(content_parts)),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                }
            }
            Role::Assistant => {
                let tool_calls: Vec<OpenRouterToolCall> = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolUse { id, name, input } => Some(OpenRouterToolCall {
                            id: id.clone(),
                            call_type: "function".to_string(),
                            function: OpenRouterFunctionCall {
                                name: name.clone(),
                                arguments: input.to_string(),
                            },
                        }),
                        _ => None,
                    })
                    .collect();

                let text_content = message.text_content();

                result.push(OpenRouterMessage {
                    role: "assistant".to_string(),
                    content: if text_content.is_empty() {
                        None
                    } else {
                        Some(OpenRouterContent::Text(text_content))
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

    fn convert_response(&self, response: OpenRouterResponse) -> CompletionResponse {
        let choice = response.choices.into_iter().next().unwrap_or_default();
        let mut content = Vec::new();

        if let Some(text) = choice.message.content {
            content.push(ContentBlock::Text { text });
        }

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
            model: response.model.unwrap_or_default(),
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

        match response.json::<OpenRouterErrorResponse>().await {
            Ok(err) => {
                let code = err.error.code.unwrap_or(status as i32);
                let message = &err.error.message;

                match code {
                    401 => Error::auth(message),
                    429 => Error::rate_limited(message, None),
                    400 => Error::invalid_request(message),
                    404 => Error::ModelNotFound(message.clone()),
                    _ if status >= 500 => Error::server(status, message),
                    _ => Error::other(message),
                }
            }
            Err(_) => Error::server(status, "Unknown error"),
        }
    }
}

#[async_trait]
impl Provider for OpenRouterProvider {
    fn name(&self) -> &str {
        "openrouter"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.config.require_api_key()?;

        let mut api_request = self.convert_request(&request);
        api_request.stream = false;

        let mut req_builder = self.client.post(self.api_url()).json(&api_request);

        // Add OpenRouter specific headers
        if let Some(ref app_name) = self.app_name {
            req_builder = req_builder.header("X-Title", app_name);
        }
        if let Some(ref site_url) = self.site_url {
            req_builder = req_builder.header("HTTP-Referer", site_url);
        }

        let response = req_builder.send().await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let openrouter_response: OpenRouterResponse = response.json().await?;
        Ok(self.convert_response(openrouter_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        self.config.require_api_key()?;

        let mut api_request = self.convert_request(&request);
        api_request.stream = true;

        let mut req_builder = self.client.post(self.api_url()).json(&api_request);

        if let Some(ref app_name) = self.app_name {
            req_builder = req_builder.header("X-Title", app_name);
        }
        if let Some(ref site_url) = self.site_url {
            req_builder = req_builder.header("HTTP-Referer", site_url);
        }

        let response = req_builder.send().await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let stream = parse_openrouter_stream(response);
        Ok(Box::pin(stream))
    }

    fn supports_tools(&self) -> bool {
        true // Most models on OpenRouter support tools
    }

    fn supports_vision(&self) -> bool {
        true // Many models support vision
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn default_model(&self) -> Option<&str> {
        Some("anthropic/claude-3.5-sonnet")
    }
}

/// Parse OpenRouter SSE stream (OpenAI-compatible format).
fn parse_openrouter_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
    use async_stream::try_stream;
    use futures::StreamExt;

    try_stream! {
        let mut event_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut sent_start = false;

        while let Some(chunk) = event_stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() || !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..];

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

                if let Ok(parsed) = serde_json::from_str::<OpenRouterStreamResponse>(data) {
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
                        if let Some(ref content) = choice.delta.content {
                            yield StreamChunk {
                                event_type: StreamEventType::ContentBlockDelta,
                                index: Some(0),
                                delta: Some(ContentDelta::Text { text: content.clone() }),
                                stop_reason: None,
                                usage: None,
                            };
                        }

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

                        if let Some(ref reason) = choice.finish_reason {
                            let stop_reason = match reason.as_str() {
                                "stop" => StopReason::EndTurn,
                                "length" => StopReason::MaxTokens,
                                "tool_calls" => StopReason::ToolUse,
                                "content_filter" => StopReason::ContentFilter,
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

// OpenRouter API types (mostly OpenAI-compatible)

#[derive(Debug, Serialize)]
struct OpenRouterRequest {
    model: String,
    messages: Vec<OpenRouterMessage>,
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
    tools: Option<Vec<OpenRouterTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<ORResponseFormat>,
    // OpenRouter specific
    #[serde(skip_serializing_if = "Option::is_none")]
    transforms: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    route: Option<String>,
    /// Reasoning/thinking control for models that support it
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<OpenRouterReasoning>,
}

/// OpenRouter reasoning configuration.
/// See: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
#[derive(Debug, Serialize)]
struct OpenRouterReasoning {
    /// Effort level for reasoning: "none", "low", "medium", "high", "max"
    #[serde(skip_serializing_if = "Option::is_none")]
    effort: Option<String>,
    /// Maximum tokens for reasoning
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    /// If true, reasoning is performed but excluded from the response
    #[serde(skip_serializing_if = "Option::is_none")]
    exclude: Option<bool>,
}

/// Response format for structured outputs (OpenAI-compatible passthrough).
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ORResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: ORJsonSchema },
}

/// JSON schema for structured output.
#[derive(Debug, Serialize)]
struct ORJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

#[derive(Debug, Serialize)]
struct OpenRouterMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenRouterContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenRouterToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenRouterContent {
    Text(String),
    Parts(Vec<OpenRouterContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenRouterContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Serialize)]
struct ImageUrl {
    url: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenRouterTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenRouterFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenRouterFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenRouterToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenRouterFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenRouterFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenRouterResponse {
    id: String,
    model: Option<String>,
    choices: Vec<OpenRouterChoice>,
    usage: Option<OpenRouterUsage>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenRouterChoice {
    message: OpenRouterResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenRouterResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenRouterToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterStreamResponse {
    choices: Vec<OpenRouterStreamChoice>,
    usage: Option<OpenRouterUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterStreamChoice {
    delta: OpenRouterStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenRouterStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenRouterStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterStreamToolCall {
    index: Option<usize>,
    id: Option<String>,
    function: Option<OpenRouterStreamFunction>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenRouterErrorResponse {
    error: OpenRouterError,
}

#[derive(Debug, Deserialize)]
struct OpenRouterError {
    code: Option<i32>,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = OpenRouterProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "openrouter");
        assert!(provider.supports_tools());
        assert!(provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_provider_with_app_name() {
        let provider = OpenRouterProvider::with_api_key("test-key")
            .unwrap()
            .with_app_name("MyApp");
        assert_eq!(provider.app_name, Some("MyApp".to_string()));
    }

    #[test]
    fn test_provider_with_site_url() {
        let provider = OpenRouterProvider::with_api_key("test-key")
            .unwrap()
            .with_site_url("https://myapp.com");
        assert_eq!(provider.site_url, Some("https://myapp.com".to_string()));
    }

    #[test]
    fn test_api_url() {
        let provider = OpenRouterProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.api_url(), OPENROUTER_API_URL);
    }

    #[test]
    fn test_api_url_custom_base() {
        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.openrouter.ai".to_string());
        let provider = OpenRouterProvider::new(config).unwrap();
        assert_eq!(provider.api_url(), "https://custom.openrouter.ai");
    }

    #[test]
    fn test_default_model() {
        let provider = OpenRouterProvider::with_api_key("test-key").unwrap();
        assert_eq!(
            provider.default_model(),
            Some("anthropic/claude-3.5-sonnet")
        );
    }

    #[test]
    fn test_request_conversion() {
        let provider = OpenRouterProvider::with_api_key("test-key").unwrap();
        let request =
            CompletionRequest::new("anthropic/claude-3.5-sonnet", vec![Message::user("Hello")])
                .with_system("You are helpful")
                .with_max_tokens(1024)
                .with_temperature(0.7);

        let openrouter_req = provider.convert_request(&request);

        assert_eq!(openrouter_req.model, "anthropic/claude-3.5-sonnet");
        assert_eq!(openrouter_req.max_tokens, Some(1024));
        assert_eq!(openrouter_req.temperature, Some(0.7));
        assert_eq!(openrouter_req.messages.len(), 2); // system + user
    }

    #[test]
    fn test_response_parsing() {
        let provider = OpenRouterProvider::with_api_key("test-key").unwrap();

        let response = OpenRouterResponse {
            id: "resp-123".to_string(),
            model: Some("anthropic/claude-3.5-sonnet".to_string()),
            choices: vec![OpenRouterChoice {
                message: OpenRouterResponseMessage {
                    content: Some("Hello! How can I help?".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(OpenRouterUsage {
                prompt_tokens: 10,
                completion_tokens: 15,
            }),
        };

        let result = provider.convert_response(response);

        assert_eq!(result.id, "resp-123");
        assert_eq!(result.model, "anthropic/claude-3.5-sonnet");
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlock::Text { text } => {
                assert_eq!(text, "Hello! How can I help?");
            }
            other => {
                panic!("Expected text content block, got {:?}", other);
            }
        }
        assert!(matches!(result.stop_reason, StopReason::EndTurn));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 15);
    }

    #[test]
    fn test_stop_reason_mapping() {
        let provider = OpenRouterProvider::with_api_key("test-key").unwrap();

        // Test "stop" -> EndTurn
        let response1 = OpenRouterResponse {
            id: "1".to_string(),
            model: Some("model".to_string()),
            choices: vec![OpenRouterChoice {
                message: OpenRouterResponseMessage {
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
        let response2 = OpenRouterResponse {
            id: "2".to_string(),
            model: Some("model".to_string()),
            choices: vec![OpenRouterChoice {
                message: OpenRouterResponseMessage {
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
        let response3 = OpenRouterResponse {
            id: "3".to_string(),
            model: Some("model".to_string()),
            choices: vec![OpenRouterChoice {
                message: OpenRouterResponseMessage {
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

        // Test "content_filter" -> ContentFilter
        let response4 = OpenRouterResponse {
            id: "4".to_string(),
            model: Some("model".to_string()),
            choices: vec![OpenRouterChoice {
                message: OpenRouterResponseMessage {
                    content: Some("Filtered".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("content_filter".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.convert_response(response4).stop_reason,
            StopReason::ContentFilter
        ));
    }

    #[test]
    fn test_request_serialization() {
        let request = OpenRouterRequest {
            model: "anthropic/claude-3.5-sonnet".to_string(),
            messages: vec![OpenRouterMessage {
                role: "user".to_string(),
                content: Some(OpenRouterContent::Text("Hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
            }],
            max_tokens: Some(1024),
            temperature: Some(0.7),
            top_p: None,
            stop: None,
            stream: false,
            tools: None,
            stream_options: None,
            response_format: None,
            transforms: None,
            route: None,
            reasoning: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("anthropic/claude-3.5-sonnet"));
        assert!(json.contains("Hello"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "test-id",
            "model": "anthropic/claude-3.5-sonnet",
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }"#;

        let response: OpenRouterResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "test-id");
        assert_eq!(
            response.model,
            Some("anthropic/claude-3.5-sonnet".to_string())
        );
        assert_eq!(
            response.choices[0].message.content,
            Some("Hello!".to_string())
        );
    }

    #[test]
    fn test_error_response_deserialization() {
        let json = r#"{"error": {"code": 401, "message": "Unauthorized"}}"#;
        let error: OpenRouterErrorResponse = serde_json::from_str(json).unwrap();
        assert_eq!(error.error.code, Some(401));
        assert_eq!(error.error.message, "Unauthorized");
    }
}
