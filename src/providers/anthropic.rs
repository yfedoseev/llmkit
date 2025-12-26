//! Anthropic Claude API provider implementation.

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
    StreamChunk, StreamEventType, TokenCountRequest, TokenCountResult, Usage,
};

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_TOKEN_COUNT_URL: &str = "https://api.anthropic.com/v1/messages/count_tokens";
const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Anthropic Claude API provider.
pub struct AnthropicProvider {
    config: ProviderConfig,
    client: Client,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.api_key {
            headers.insert(
                "x-api-key",
                key.parse()
                    .map_err(|_| Error::config("Invalid API key format"))?,
            );
        }

        headers.insert("anthropic-version", ANTHROPIC_VERSION.parse().unwrap());

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

    /// Create a new Anthropic provider from environment variable.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("ANTHROPIC_API_KEY");
        Self::new(config)
    }

    /// Create a new Anthropic provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn api_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(ANTHROPIC_API_URL)
    }

    /// Convert our unified request to Anthropic's format.
    fn convert_request(&self, request: &CompletionRequest) -> AnthropicRequest {
        use crate::types::{CacheControl, ThinkingType};

        // Convert messages
        let messages: Vec<AnthropicMessage> = request
            .messages
            .iter()
            .filter(|m| m.role != Role::System) // System messages handled separately
            .map(|m| self.convert_message(m))
            .collect();

        // Get system prompt with optional cache control
        let system_text = request.system.clone().or_else(|| {
            request
                .messages
                .iter()
                .find(|m| m.role == Role::System)
                .map(|m| m.text_content())
        });

        // Build system content with cache control if specified
        let system = system_text.map(|text| {
            if let Some(ref cache_control) = request.system_cache_control {
                AnthropicSystemContent::Structured(vec![AnthropicSystemBlock {
                    block_type: "text".to_string(),
                    text,
                    cache_control: Some(AnthropicCacheControl {
                        cache_type: match cache_control.cache_control {
                            CacheControl::Ephemeral => "ephemeral".to_string(),
                            CacheControl::Extended => "ephemeral".to_string(),
                        },
                    }),
                }])
            } else {
                AnthropicSystemContent::Simple(text)
            }
        });

        // Convert tools
        let tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| AnthropicTool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.input_schema.clone(),
                })
                .collect()
        });

        // Convert thinking configuration
        let thinking = request.thinking.as_ref().map(|t| AnthropicThinking {
            thinking_type: match t.thinking_type {
                ThinkingType::Enabled => "enabled".to_string(),
                ThinkingType::Disabled => "disabled".to_string(),
            },
            budget_tokens: t.budget_tokens,
        });

        // Convert structured output format
        let output_format = request.response_format.as_ref().and_then(|rf| {
            use crate::types::StructuredOutputType;
            match rf.format_type {
                StructuredOutputType::JsonSchema => {
                    rf.json_schema
                        .as_ref()
                        .map(|schema_def| AnthropicOutputFormat {
                            format_type: "json_schema".to_string(),
                            json_schema: Some(AnthropicJsonSchema {
                                name: schema_def.name.clone(),
                                description: schema_def.description.clone(),
                                schema: schema_def.schema.clone(),
                            }),
                        })
                }
                StructuredOutputType::JsonObject => {
                    // Anthropic doesn't have a simple JSON object mode like OpenAI,
                    // so we skip this (model will try to return JSON based on prompt)
                    None
                }
                StructuredOutputType::Text => None,
            }
        });

        AnthropicRequest {
            model: request.model.clone(),
            messages,
            system,
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            top_p: request.top_p,
            stop_sequences: request.stop_sequences.clone(),
            stream: request.stream,
            tools,
            thinking,
            output_format,
        }
    }

    fn convert_message(&self, message: &Message) -> AnthropicMessage {
        let content: Vec<AnthropicContent> = message
            .content
            .iter()
            .map(|block| self.convert_content_block(block))
            .collect();

        AnthropicMessage {
            role: match message.role {
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
                Role::System => "user".to_string(), // Should be filtered out
            },
            content,
        }
    }

    fn convert_content_block(&self, block: &ContentBlock) -> AnthropicContent {
        use crate::types::{CacheControl, DocumentSource};

        match block {
            ContentBlock::Text { text } => AnthropicContent::Text {
                text: text.clone(),
                cache_control: None,
            },
            ContentBlock::TextWithCache {
                text,
                cache_control,
            } => AnthropicContent::Text {
                text: text.clone(),
                cache_control: Some(AnthropicCacheControl {
                    cache_type: match cache_control.cache_control {
                        CacheControl::Ephemeral => "ephemeral".to_string(),
                        CacheControl::Extended => "ephemeral".to_string(), // Extended uses same type, different header
                    },
                }),
            },
            ContentBlock::Image { media_type, data } => AnthropicContent::Image {
                source: ImageSource {
                    source_type: "base64".to_string(),
                    media_type: media_type.clone(),
                    data: data.clone(),
                },
                cache_control: None,
            },
            ContentBlock::ImageUrl { url } => AnthropicContent::Image {
                source: ImageSource {
                    source_type: "url".to_string(),
                    media_type: String::new(),
                    data: url.clone(),
                },
                cache_control: None,
            },
            ContentBlock::Document {
                source,
                cache_control,
            } => {
                let doc_source = match source {
                    DocumentSource::Base64 { media_type, data } => {
                        DocumentSourceAnthropic::Base64 {
                            media_type: media_type.clone(),
                            data: data.clone(),
                        }
                    }
                    DocumentSource::Url { url } => {
                        DocumentSourceAnthropic::Url { url: url.clone() }
                    }
                    DocumentSource::File { file_id: _ } => {
                        // File API not directly supported in this path, convert to placeholder
                        DocumentSourceAnthropic::Base64 {
                            media_type: "application/pdf".to_string(),
                            data: String::new(),
                        }
                    }
                };
                AnthropicContent::Document {
                    source: doc_source,
                    cache_control: cache_control.as_ref().map(|cc| AnthropicCacheControl {
                        cache_type: match cc.cache_control {
                            CacheControl::Ephemeral => "ephemeral".to_string(),
                            CacheControl::Extended => "ephemeral".to_string(),
                        },
                    }),
                }
            }
            ContentBlock::ToolUse { id, name, input } => AnthropicContent::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            },
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => AnthropicContent::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: content.clone(),
                is_error: *is_error,
            },
            ContentBlock::Thinking { thinking } => AnthropicContent::Thinking {
                thinking: thinking.clone(),
            },
        }
    }

    fn convert_response(&self, response: AnthropicResponse) -> CompletionResponse {
        let content: Vec<ContentBlock> = response
            .content
            .into_iter()
            .map(|c| self.convert_anthropic_content(c))
            .collect();

        CompletionResponse {
            id: response.id,
            model: response.model,
            content,
            stop_reason: match response.stop_reason.as_str() {
                "end_turn" => StopReason::EndTurn,
                "max_tokens" => StopReason::MaxTokens,
                "tool_use" => StopReason::ToolUse,
                "stop_sequence" => StopReason::StopSequence,
                _ => StopReason::EndTurn,
            },
            usage: Usage {
                input_tokens: response.usage.input_tokens,
                output_tokens: response.usage.output_tokens,
                cache_creation_input_tokens: response
                    .usage
                    .cache_creation_input_tokens
                    .unwrap_or(0),
                cache_read_input_tokens: response.usage.cache_read_input_tokens.unwrap_or(0),
            },
        }
    }

    fn convert_anthropic_content(&self, content: AnthropicContent) -> ContentBlock {
        use crate::types::DocumentSource;

        match content {
            AnthropicContent::Text {
                text,
                cache_control: _,
            } => ContentBlock::Text { text },
            AnthropicContent::Image {
                source,
                cache_control: _,
            } => {
                if source.source_type == "url" {
                    ContentBlock::ImageUrl { url: source.data }
                } else {
                    ContentBlock::Image {
                        media_type: source.media_type,
                        data: source.data,
                    }
                }
            }
            AnthropicContent::Document {
                source,
                cache_control: _,
            } => {
                let doc_source = match source {
                    DocumentSourceAnthropic::Base64 { media_type, data } => {
                        DocumentSource::Base64 { media_type, data }
                    }
                    DocumentSourceAnthropic::Url { url } => DocumentSource::Url { url },
                };
                ContentBlock::Document {
                    source: doc_source,
                    cache_control: None,
                }
            }
            AnthropicContent::ToolUse { id, name, input } => {
                ContentBlock::ToolUse { id, name, input }
            }
            AnthropicContent::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            },
            AnthropicContent::Thinking { thinking } => ContentBlock::Thinking { thinking },
        }
    }

    async fn handle_error_response(&self, response: reqwest::Response) -> Error {
        let status = response.status().as_u16();

        match response.json::<AnthropicErrorResponse>().await {
            Ok(err) => {
                let error_type = err.error.error_type.as_deref().unwrap_or("unknown");
                let message = &err.error.message;

                match error_type {
                    "authentication_error" => Error::auth(message),
                    "rate_limit_error" => Error::rate_limited(message, None),
                    "invalid_request_error" => Error::invalid_request(message),
                    "not_found_error" => Error::ModelNotFound(message.clone()),
                    "overloaded_error" => Error::server(503, message),
                    _ => Error::server(status, message),
                }
            }
            Err(_) => Error::server(status, "Unknown error"),
        }
    }

    /// Convert a token count request to Anthropic's format.
    fn convert_token_count_request(
        &self,
        request: &TokenCountRequest,
    ) -> AnthropicTokenCountRequest {
        let messages: Vec<AnthropicMessage> = request
            .messages
            .iter()
            .map(|m| self.convert_message(m))
            .collect();

        let tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| AnthropicTool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.input_schema.clone(),
                })
                .collect()
        });

        let system = request
            .system
            .as_ref()
            .map(|s| AnthropicSystemContent::Simple(s.clone()));

        AnthropicTokenCountRequest {
            model: request.model.clone(),
            messages,
            system,
            tools,
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.config.require_api_key()?;

        // Check if structured output is requested (for beta header)
        let needs_structured_output_beta = request
            .response_format
            .as_ref()
            .is_some_and(|rf| rf.format_type == crate::types::StructuredOutputType::JsonSchema);

        let mut api_request = self.convert_request(&request);
        api_request.stream = false;

        let mut req_builder = self.client.post(self.api_url()).json(&api_request);

        // Add beta header for structured outputs
        if needs_structured_output_beta {
            req_builder = req_builder.header("anthropic-beta", "structured-outputs-2025-11-13");
        }

        let response = req_builder.send().await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let anthropic_response: AnthropicResponse = response.json().await?;
        Ok(self.convert_response(anthropic_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        self.config.require_api_key()?;

        // Check if structured output is requested (for beta header)
        let needs_structured_output_beta = request
            .response_format
            .as_ref()
            .is_some_and(|rf| rf.format_type == crate::types::StructuredOutputType::JsonSchema);

        let mut api_request = self.convert_request(&request);
        api_request.stream = true;

        let mut req_builder = self.client.post(self.api_url()).json(&api_request);

        // Add beta header for structured outputs
        if needs_structured_output_beta {
            req_builder = req_builder.header("anthropic-beta", "structured-outputs-2025-11-13");
        }

        let response = req_builder.send().await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let stream = parse_anthropic_stream(response);
        Ok(Box::pin(stream))
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supported_models(&self) -> Option<&[&str]> {
        Some(&[
            // Claude 4.5 (latest - Dec 2025)
            "claude-opus-4-5-20251101",
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251015",
            // Claude 4 (May 2025)
            "claude-opus-4-20250514",
            "claude-sonnet-4-20250514",
            // Claude 3.5
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            // Claude 3 (legacy)
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ])
    }

    fn default_model(&self) -> Option<&str> {
        Some("claude-sonnet-4-5-20250929")
    }

    async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCountResult> {
        self.config.require_api_key()?;

        let api_request = self.convert_token_count_request(&request);

        let response = self
            .client
            .post(ANTHROPIC_TOKEN_COUNT_URL)
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let token_response: AnthropicTokenCountResponse = response.json().await?;
        Ok(TokenCountResult::new(token_response.input_tokens))
    }

    fn supports_token_counting(&self) -> bool {
        true
    }
}

/// Parse Anthropic SSE stream into our unified StreamChunk format.
fn parse_anthropic_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
    use async_stream::try_stream;
    use futures::StreamExt;

    try_stream! {
        let mut event_stream = response.bytes_stream();
        let mut buffer = String::new();

        while let Some(chunk) = event_stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete SSE events
            while let Some(pos) = buffer.find("\n\n") {
                let event_str = buffer[..pos].to_string();
                buffer = buffer[pos + 2..].to_string();

                // Parse SSE event
                let mut event_type = String::new();
                let mut data = String::new();

                for line in event_str.lines() {
                    if let Some(value) = line.strip_prefix("event: ") {
                        event_type = value.to_string();
                    } else if let Some(value) = line.strip_prefix("data: ") {
                        data = value.to_string();
                    }
                }

                if data.is_empty() || data == "[DONE]" {
                    continue;
                }

                // Parse the JSON data based on event type
                if let Ok(parsed) = serde_json::from_str::<Value>(&data) {
                    if let Some(chunk) = parse_anthropic_event(&event_type, &parsed) {
                        yield chunk;
                    }
                }
            }
        }
    }
}

fn parse_anthropic_event(event_type: &str, data: &Value) -> Option<StreamChunk> {
    match event_type {
        "message_start" => Some(StreamChunk {
            event_type: StreamEventType::MessageStart,
            index: None,
            delta: None,
            stop_reason: None,
            usage: data
                .get("message")
                .and_then(|m| m.get("usage"))
                .and_then(parse_usage),
        }),
        "content_block_start" => {
            let index = data.get("index")?.as_u64()? as usize;
            let content_block = data.get("content_block")?;
            let block_type = content_block.get("type")?.as_str()?;

            let delta = match block_type {
                "text" => Some(ContentDelta::TextDelta {
                    text: content_block
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                }),
                "tool_use" => Some(ContentDelta::ToolUseDelta {
                    id: content_block
                        .get("id")
                        .and_then(|i| i.as_str())
                        .map(String::from),
                    name: content_block
                        .get("name")
                        .and_then(|n| n.as_str())
                        .map(String::from),
                    input_json_delta: None,
                }),
                "thinking" => Some(ContentDelta::ThinkingDelta {
                    thinking: String::new(),
                }),
                _ => None,
            };

            Some(StreamChunk {
                event_type: StreamEventType::ContentBlockStart,
                index: Some(index),
                delta,
                stop_reason: None,
                usage: None,
            })
        }
        "content_block_delta" => {
            let index = data.get("index")?.as_u64()? as usize;
            let delta_obj = data.get("delta")?;
            let delta_type = delta_obj.get("type")?.as_str()?;

            let delta = match delta_type {
                "text_delta" => Some(ContentDelta::TextDelta {
                    text: delta_obj
                        .get("text")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                }),
                "input_json_delta" => Some(ContentDelta::ToolUseDelta {
                    id: None,
                    name: None,
                    input_json_delta: delta_obj
                        .get("partial_json")
                        .and_then(|j| j.as_str())
                        .map(String::from),
                }),
                "thinking_delta" => Some(ContentDelta::ThinkingDelta {
                    thinking: delta_obj
                        .get("thinking")
                        .and_then(|t| t.as_str())
                        .unwrap_or("")
                        .to_string(),
                }),
                _ => None,
            };

            Some(StreamChunk {
                event_type: StreamEventType::ContentBlockDelta,
                index: Some(index),
                delta,
                stop_reason: None,
                usage: None,
            })
        }
        "content_block_stop" => {
            let index = data.get("index")?.as_u64()? as usize;
            Some(StreamChunk {
                event_type: StreamEventType::ContentBlockStop,
                index: Some(index),
                delta: None,
                stop_reason: None,
                usage: None,
            })
        }
        "message_delta" => {
            let delta = data.get("delta")?;
            let stop_reason = delta
                .get("stop_reason")
                .and_then(|s| s.as_str())
                .map(|s| match s {
                    "end_turn" => StopReason::EndTurn,
                    "max_tokens" => StopReason::MaxTokens,
                    "tool_use" => StopReason::ToolUse,
                    "stop_sequence" => StopReason::StopSequence,
                    _ => StopReason::EndTurn,
                });

            Some(StreamChunk {
                event_type: StreamEventType::MessageDelta,
                index: None,
                delta: None,
                stop_reason,
                usage: data.get("usage").and_then(parse_usage),
            })
        }
        "message_stop" => Some(StreamChunk {
            event_type: StreamEventType::MessageStop,
            index: None,
            delta: None,
            stop_reason: None,
            usage: None,
        }),
        "ping" => Some(StreamChunk {
            event_type: StreamEventType::Ping,
            index: None,
            delta: None,
            stop_reason: None,
            usage: None,
        }),
        "error" => Some(StreamChunk {
            event_type: StreamEventType::Error,
            index: None,
            delta: None,
            stop_reason: None,
            usage: None,
        }),
        _ => None,
    }
}

fn parse_usage(value: &Value) -> Option<Usage> {
    Some(Usage {
        input_tokens: value.get("input_tokens")?.as_u64()? as u32,
        output_tokens: value
            .get("output_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        cache_creation_input_tokens: value
            .get("cache_creation_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
        cache_read_input_tokens: value
            .get("cache_read_input_tokens")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32,
    })
}

// Anthropic API types

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<AnthropicSystemContent>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_format: Option<AnthropicOutputFormat>,
}

/// Anthropic structured output format.
#[derive(Debug, Serialize)]
struct AnthropicOutputFormat {
    #[serde(rename = "type")]
    format_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<AnthropicJsonSchema>,
}

/// Anthropic JSON schema definition for structured output.
#[derive(Debug, Serialize)]
struct AnthropicJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: Value,
}

/// Request for token counting
#[derive(Debug, Serialize)]
struct AnthropicTokenCountRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<AnthropicSystemContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
}

/// Response from token counting
#[derive(Debug, Deserialize)]
struct AnthropicTokenCountResponse {
    input_tokens: u32,
}

/// System content can be a simple string or structured with cache control
#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AnthropicSystemContent {
    Simple(String),
    Structured(Vec<AnthropicSystemBlock>),
}

#[derive(Debug, Serialize)]
struct AnthropicSystemBlock {
    #[serde(rename = "type")]
    block_type: String,
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicCacheControl>,
}

#[derive(Debug, Serialize)]
struct AnthropicThinking {
    #[serde(rename = "type")]
    thinking_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContent {
    Text {
        text: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Image {
        source: ImageSource,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    Document {
        source: DocumentSourceAnthropic,
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<AnthropicCacheControl>,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(default)]
        is_error: bool,
    },
    Thinking {
        thinking: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicCacheControl {
    #[serde(rename = "type")]
    cache_type: String,
}

impl AnthropicCacheControl {
    #[allow(dead_code)]
    fn ephemeral() -> Self {
        Self {
            cache_type: "ephemeral".to_string(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum DocumentSourceAnthropic {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Debug, Serialize, Deserialize)]
struct ImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<AnthropicContent>,
    stop_reason: String,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
    #[serde(default)]
    cache_creation_input_tokens: Option<u32>,
    #[serde(default)]
    cache_read_input_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct AnthropicErrorResponse {
    error: AnthropicError,
}

#[derive(Debug, Deserialize)]
struct AnthropicError {
    #[serde(rename = "type")]
    error_type: Option<String>,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = AnthropicProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "anthropic");
        assert!(provider.supports_tools());
        assert!(provider.supports_vision());
    }

    #[test]
    fn test_request_conversion() {
        let provider = AnthropicProvider::with_api_key("test-key").unwrap();
        let request =
            CompletionRequest::new("claude-sonnet-4-5-20250929", vec![Message::user("Hello")])
                .with_system("You are helpful")
                .with_max_tokens(1024);

        let anthropic_req = provider.convert_request(&request);

        assert_eq!(anthropic_req.model, "claude-sonnet-4-5-20250929");
        // System is now AnthropicSystemContent, check it's Some
        assert!(anthropic_req.system.is_some());
        match anthropic_req.system {
            Some(AnthropicSystemContent::Simple(text)) => {
                assert_eq!(text, "You are helpful");
            }
            _ => panic!("Expected simple system content"),
        }
        assert_eq!(anthropic_req.max_tokens, 1024);
        assert_eq!(anthropic_req.messages.len(), 1);
    }

    #[test]
    fn test_request_with_caching() {
        let provider = AnthropicProvider::with_api_key("test-key").unwrap();
        let request =
            CompletionRequest::new("claude-sonnet-4-5-20250929", vec![Message::user("Hello")])
                .with_system("You are helpful")
                .with_system_caching();

        let anthropic_req = provider.convert_request(&request);

        // System should be structured with cache control
        match anthropic_req.system {
            Some(AnthropicSystemContent::Structured(blocks)) => {
                assert_eq!(blocks.len(), 1);
                assert_eq!(blocks[0].text, "You are helpful");
                assert!(blocks[0].cache_control.is_some());
            }
            _ => panic!("Expected structured system content with cache control"),
        }
    }

    #[test]
    fn test_request_with_thinking() {
        let provider = AnthropicProvider::with_api_key("test-key").unwrap();
        let request =
            CompletionRequest::new("claude-sonnet-4-5-20250929", vec![Message::user("Hello")])
                .with_thinking(5000);

        let anthropic_req = provider.convert_request(&request);

        assert!(anthropic_req.thinking.is_some());
        let thinking = anthropic_req.thinking.unwrap();
        assert_eq!(thinking.thinking_type, "enabled");
        assert_eq!(thinking.budget_tokens, Some(5000));
    }

    #[test]
    fn test_structured_output_json_schema() {
        let provider = AnthropicProvider::with_api_key("test-key").unwrap();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let request =
            CompletionRequest::new("claude-sonnet-4-5-20250929", vec![Message::user("Hello")])
                .with_json_schema("person", schema.clone());

        let anthropic_req = provider.convert_request(&request);

        assert!(anthropic_req.output_format.is_some());
        let output_format = anthropic_req.output_format.unwrap();
        assert_eq!(output_format.format_type, "json_schema");
        assert!(output_format.json_schema.is_some());
        let json_schema = output_format.json_schema.unwrap();
        assert_eq!(json_schema.name, "person");
        assert_eq!(json_schema.schema, schema);
    }

    #[test]
    fn test_token_counting_support() {
        let provider = AnthropicProvider::with_api_key("test-key").unwrap();
        assert!(provider.supports_token_counting());
    }

    #[test]
    fn test_token_count_request_conversion() {
        let provider = AnthropicProvider::with_api_key("test-key").unwrap();
        let request = TokenCountRequest::new(
            "claude-sonnet-4-5-20250929",
            vec![Message::user("Hello, world!")],
        )
        .with_system("You are a helpful assistant");

        let anthropic_req = provider.convert_token_count_request(&request);

        assert_eq!(anthropic_req.model, "claude-sonnet-4-5-20250929");
        assert_eq!(anthropic_req.messages.len(), 1);
        assert!(anthropic_req.system.is_some());
        assert!(anthropic_req.tools.is_none());
    }
}
