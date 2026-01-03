//! OpenAI API provider implementation.

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

const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// OpenAI API provider.
pub struct OpenAIProvider {
    config: ProviderConfig,
    client: Client,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider with the given configuration.
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

        if let Some(ref org_id) = config.organization_id {
            headers.insert(
                "OpenAI-Organization",
                org_id
                    .parse()
                    .map_err(|_| Error::config("Invalid organization ID"))?,
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

    /// Create a new OpenAI provider from environment variable.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("OPENAI_API_KEY");
        Self::new(config)
    }

    /// Create a new OpenAI provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn api_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(OPENAI_API_URL)
    }

    /// Convert our unified request to OpenAI's format.
    fn convert_request(&self, request: &CompletionRequest) -> OpenAIRequest {
        let mut messages: Vec<OpenAIMessage> = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(OpenAIMessage {
                role: "system".to_string(),
                content: Some(OpenAIContent::Text(system.clone())),
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
                .map(|t| OpenAITool {
                    tool_type: "function".to_string(),
                    function: OpenAIFunction {
                        name: t.name.clone(),
                        description: Some(t.description.clone()),
                        parameters: t.input_schema.clone(),
                    },
                })
                .collect()
        });

        // Convert response format (structured outputs)
        let response_format = request.response_format.as_ref().map(|rf| {
            use crate::types::StructuredOutputType;
            match rf.format_type {
                StructuredOutputType::JsonObject => OpenAIResponseFormat::JsonObject,
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        OpenAIResponseFormat::JsonSchema {
                            json_schema: OpenAIJsonSchema {
                                name: schema_def.name.clone(),
                                description: schema_def.description.clone(),
                                schema: schema_def.schema.clone(),
                                strict: Some(schema_def.strict),
                            },
                        }
                    } else {
                        // Fallback to simple JSON mode if no schema provided
                        OpenAIResponseFormat::JsonObject
                    }
                }
                StructuredOutputType::Text => OpenAIResponseFormat::Text,
            }
        });

        // Convert prediction (for speculative decoding)
        let prediction = request.prediction.as_ref().map(|p| OpenAIPrediction {
            prediction_type: "content".to_string(),
            content: p.content.clone(),
        });

        OpenAIRequest {
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
            prediction,
        }
    }

    fn convert_message(&self, message: &Message) -> Vec<OpenAIMessage> {
        let mut result = Vec::new();

        match message.role {
            Role::System => {
                let text = message.text_content();
                if !text.is_empty() {
                    result.push(OpenAIMessage {
                        role: "system".to_string(),
                        content: Some(OpenAIContent::Text(text)),
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
                    // Tool results become separate "tool" role messages
                    for (tool_call_id, content) in tool_results {
                        result.push(OpenAIMessage {
                            role: "tool".to_string(),
                            content: Some(OpenAIContent::Text(content)),
                            tool_calls: None,
                            tool_call_id: Some(tool_call_id),
                        });
                    }
                } else {
                    // Regular user message
                    let content_parts: Vec<OpenAIContentPart> = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text { text } => {
                                Some(OpenAIContentPart::Text { text: text.clone() })
                            }
                            ContentBlock::Image { media_type, data } => {
                                Some(OpenAIContentPart::ImageUrl {
                                    image_url: ImageUrl {
                                        url: format!("data:{};base64,{}", media_type, data),
                                        detail: None,
                                    },
                                })
                            }
                            ContentBlock::ImageUrl { url } => Some(OpenAIContentPart::ImageUrl {
                                image_url: ImageUrl {
                                    url: url.clone(),
                                    detail: None,
                                },
                            }),
                            _ => None,
                        })
                        .collect();

                    if content_parts.len() == 1 {
                        if let OpenAIContentPart::Text { text } = &content_parts[0] {
                            result.push(OpenAIMessage {
                                role: "user".to_string(),
                                content: Some(OpenAIContent::Text(text.clone())),
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        } else {
                            result.push(OpenAIMessage {
                                role: "user".to_string(),
                                content: Some(OpenAIContent::Parts(content_parts)),
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        }
                    } else if !content_parts.is_empty() {
                        result.push(OpenAIMessage {
                            role: "user".to_string(),
                            content: Some(OpenAIContent::Parts(content_parts)),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                }
            }
            Role::Assistant => {
                // Check for tool calls
                let tool_calls: Vec<OpenAIToolCall> = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolUse { id, name, input } => Some(OpenAIToolCall {
                            id: id.clone(),
                            call_type: "function".to_string(),
                            function: OpenAIFunctionCall {
                                name: name.clone(),
                                arguments: input.to_string(),
                            },
                        }),
                        _ => None,
                    })
                    .collect();

                let text_content = message.text_content();

                result.push(OpenAIMessage {
                    role: "assistant".to_string(),
                    content: if text_content.is_empty() {
                        None
                    } else {
                        Some(OpenAIContent::Text(text_content))
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

    fn convert_response(&self, response: OpenAIResponse) -> CompletionResponse {
        let choice = response.choices.into_iter().next().unwrap_or_default();
        let mut content = Vec::new();

        // Add text content
        if let Some(text) = choice.message.content {
            content.push(ContentBlock::Text { text });
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

        match response.json::<OpenAIErrorResponse>().await {
            Ok(err) => {
                let error_type = err.error.error_type.as_deref().unwrap_or("unknown");
                let message = &err.error.message;

                match error_type {
                    "invalid_api_key" | "authentication_error" => Error::auth(message),
                    "rate_limit_exceeded" => Error::rate_limited(message, None),
                    "invalid_request_error" => Error::invalid_request(message),
                    "model_not_found" => Error::ModelNotFound(message.clone()),
                    "context_length_exceeded" => Error::ContextLengthExceeded(message.clone()),
                    "server_error" => Error::server(500, message),
                    _ => Error::server(status, message),
                }
            }
            Err(_) => Error::server(status, "Unknown error"),
        }
    }
}

#[async_trait]
impl Provider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
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

        let openai_response: OpenAIResponse = response.json().await?;
        Ok(self.convert_response(openai_response))
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

        let stream = parse_openai_stream(response);
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
            // GPT-5 (latest)
            "gpt-5",
            // GPT-4.1 series (April 2025 - 1M context)
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            // o-series reasoning models
            "o4-mini", // Latest, best for math/coding/visual
            "o3",      // Most powerful reasoning
            "o3-mini", // Cost-efficient reasoning
            "o3-pro",  // Extended thinking
            "o1",      // Previous generation
            "o1-mini",
            "o1-preview",
            // GPT-4o series (multimodal)
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-audio-preview",
            "gpt-4o-realtime-preview",
            // GPT-4 (legacy)
            "gpt-4-turbo",
            "gpt-4",
            // GPT-3.5 (legacy)
            "gpt-3.5-turbo",
        ])
    }

    fn default_model(&self) -> Option<&str> {
        Some("gpt-4o")
    }
}

/// Parse OpenAI SSE stream into our unified StreamChunk format.
fn parse_openai_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
    use async_stream::try_stream;
    use futures::StreamExt;

    try_stream! {
        let mut event_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut tool_call_builders: std::collections::HashMap<usize, (String, String, String)> = std::collections::HashMap::new();
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

                if let Ok(parsed) = serde_json::from_str::<OpenAIStreamResponse>(data) {
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
                                let entry = tool_call_builders.entry(idx).or_insert_with(|| {
                                    (String::new(), String::new(), String::new())
                                });

                                if let Some(ref id) = tc.id {
                                    entry.0 = id.clone();
                                }
                                if let Some(ref func) = tc.function {
                                    if let Some(ref name) = func.name {
                                        entry.1 = name.clone();
                                    }
                                    if let Some(ref args) = func.arguments {
                                        entry.2.push_str(args);
                                    }
                                }

                                yield StreamChunk {
                                    event_type: StreamEventType::ContentBlockDelta,
                                    index: Some(idx + 1), // Offset by 1 for text block
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

// OpenAI API types

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
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
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    prediction: Option<OpenAIPrediction>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// OpenAI response format for structured outputs
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAIResponseFormat {
    /// Simple JSON mode (model must be instructed to output JSON via prompt)
    JsonObject,
    /// Structured output with JSON Schema enforcement
    JsonSchema { json_schema: OpenAIJsonSchema },
    /// Plain text (default)
    Text,
}

#[derive(Debug, Serialize)]
struct OpenAIJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

/// OpenAI predicted output for speculative decoding
#[derive(Debug, Serialize)]
struct OpenAIPrediction {
    #[serde(rename = "type")]
    prediction_type: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<OpenAIContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAIContent {
    Text(String),
    Parts(Vec<OpenAIContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAIContentPart {
    Text { text: String },
    ImageUrl { image_url: ImageUrl },
}

#[derive(Debug, Serialize)]
struct ImageUrl {
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    detail: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    id: String,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIChoice {
    message: OpenAIResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamResponse {
    choices: Vec<OpenAIStreamChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: OpenAIStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct OpenAIStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<OpenAIStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamToolCall {
    index: Option<usize>,
    id: Option<String>,
    function: Option<OpenAIStreamFunction>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAIErrorResponse {
    error: OpenAIError,
}

#[derive(Debug, Deserialize)]
struct OpenAIError {
    #[serde(rename = "type")]
    error_type: Option<String>,
    message: String,
}

// ============================================================================
// Embedding API
// ============================================================================

use crate::embedding::{
    Embedding, EmbeddingInput, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse,
    EmbeddingUsage, EncodingFormat,
};

const OPENAI_EMBEDDINGS_URL: &str = "https://api.openai.com/v1/embeddings";

impl OpenAIProvider {
    fn embeddings_url(&self) -> String {
        self.config
            .base_url
            .as_ref()
            .map(|url| url.replace("/chat/completions", "/embeddings"))
            .unwrap_or_else(|| OPENAI_EMBEDDINGS_URL.to_string())
    }
}

#[async_trait]
impl EmbeddingProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.config.require_api_key()?;

        let input = match &request.input {
            EmbeddingInput::Single(text) => OpenAIEmbeddingInput::Single(text.clone()),
            EmbeddingInput::Batch(texts) => OpenAIEmbeddingInput::Batch(texts.clone()),
        };

        let api_request = OpenAIEmbeddingRequest {
            model: request.model.clone(),
            input,
            dimensions: request.dimensions,
            encoding_format: request.encoding_format.map(|f| match f {
                EncodingFormat::Float => "float".to_string(),
                EncodingFormat::Base64 => "base64".to_string(),
            }),
        };

        let response = self
            .client
            .post(self.embeddings_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let api_response: OpenAIEmbeddingResponse = response.json().await?;

        let embeddings = api_response
            .data
            .into_iter()
            .map(|e| Embedding::new(e.index, e.embedding))
            .collect();

        Ok(EmbeddingResponse {
            model: api_response.model,
            embeddings,
            usage: EmbeddingUsage::new(
                api_response.usage.prompt_tokens,
                api_response.usage.total_tokens,
            ),
        })
    }

    fn embedding_dimensions(&self, model: &str) -> Option<usize> {
        match model {
            "text-embedding-3-small" => Some(1536),
            "text-embedding-3-large" => Some(3072),
            "text-embedding-ada-002" => Some(1536),
            _ => None,
        }
    }

    fn default_embedding_model(&self) -> Option<&str> {
        Some("text-embedding-3-small")
    }

    fn max_batch_size(&self) -> usize {
        2048
    }

    fn supports_dimensions(&self, model: &str) -> bool {
        matches!(model, "text-embedding-3-small" | "text-embedding-3-large")
    }

    fn supported_embedding_models(&self) -> Option<&[&str]> {
        Some(&[
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ])
    }
}

// OpenAI Embedding API types

#[derive(Debug, Serialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: OpenAIEmbeddingInput,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum OpenAIEmbeddingInput {
    Single(String),
    Batch(Vec<String>),
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingResponse {
    model: String,
    data: Vec<OpenAIEmbeddingData>,
    usage: OpenAIEmbeddingUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingData {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct OpenAIEmbeddingUsage {
    prompt_tokens: u32,
    total_tokens: u32,
}

// ============================================================================
// Image Generation API
// ============================================================================

use crate::image::{
    GeneratedImage, ImageFormat, ImageGenerationRequest, ImageGenerationResponse, ImageProvider,
    ImageQuality, ImageSize, ImageStyle,
};

const OPENAI_IMAGES_URL: &str = "https://api.openai.com/v1/images/generations";

impl OpenAIProvider {
    fn images_url(&self) -> String {
        self.config
            .base_url
            .as_ref()
            .map(|url| url.replace("/chat/completions", "/images/generations"))
            .unwrap_or_else(|| OPENAI_IMAGES_URL.to_string())
    }
}

#[async_trait]
impl ImageProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn generate_image(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse> {
        self.config.require_api_key()?;

        let size = request
            .size
            .unwrap_or(ImageSize::Square1024)
            .to_openai_string();

        let quality = request.quality.map(|q| match q {
            ImageQuality::Standard => "standard",
            ImageQuality::Hd => "hd",
        });

        let style = request.style.map(|s| match s {
            ImageStyle::Natural => "natural",
            ImageStyle::Vivid => "vivid",
        });

        let response_format = request.response_format.map(|f| match f {
            ImageFormat::Url => "url",
            ImageFormat::B64Json => "b64_json",
        });

        let api_request = OpenAIImageRequest {
            model: request.model,
            prompt: request.prompt,
            n: request.n,
            size: Some(size),
            quality,
            style,
            response_format,
        };

        let response = self
            .client
            .post(self.images_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let api_response: OpenAIImageResponse = response.json().await?;

        let images = api_response
            .data
            .into_iter()
            .map(|img| {
                let mut generated = if let Some(url) = img.url {
                    GeneratedImage::from_url(url)
                } else if let Some(b64) = img.b64_json {
                    GeneratedImage::from_b64(b64)
                } else {
                    GeneratedImage::from_url("")
                };
                if let Some(revised) = img.revised_prompt {
                    generated = generated.with_revised_prompt(revised);
                }
                generated
            })
            .collect();

        Ok(ImageGenerationResponse {
            created: api_response.created,
            images,
        })
    }

    fn supported_sizes(&self) -> &[ImageSize] {
        &[
            ImageSize::Square256,
            ImageSize::Square512,
            ImageSize::Square1024,
            ImageSize::Portrait1024x1792,
            ImageSize::Landscape1792x1024,
        ]
    }

    fn max_images_per_request(&self) -> u8 {
        10 // DALL-E 2 limit; DALL-E 3 is 1
    }

    fn default_image_model(&self) -> Option<&str> {
        Some("dall-e-3")
    }

    fn supported_image_models(&self) -> Option<&[&str]> {
        Some(&["dall-e-2", "dall-e-3"])
    }
}

// OpenAI Image API types

#[derive(Debug, Serialize)]
struct OpenAIImageRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    n: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    quality: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    style: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'static str>,
}

#[derive(Debug, Deserialize)]
struct OpenAIImageResponse {
    created: u64,
    data: Vec<OpenAIImageData>,
}

#[derive(Debug, Deserialize)]
struct OpenAIImageData {
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    b64_json: Option<String>,
    #[serde(default)]
    revised_prompt: Option<String>,
}

// ============================================================================
// Audio APIs (TTS and STT)
// ============================================================================

use crate::audio::{
    AudioFormat, AudioInput, SpeechProvider, SpeechRequest, SpeechResponse, TranscriptFormat,
    TranscriptSegment, TranscriptWord, TranscriptionProvider, TranscriptionRequest,
    TranscriptionResponse, VoiceInfo,
};

const OPENAI_SPEECH_URL: &str = "https://api.openai.com/v1/audio/speech";
const OPENAI_TRANSCRIPTION_URL: &str = "https://api.openai.com/v1/audio/transcriptions";
const OPENAI_TRANSLATION_URL: &str = "https://api.openai.com/v1/audio/translations";

impl OpenAIProvider {
    fn speech_url(&self) -> String {
        self.config
            .base_url
            .as_ref()
            .map(|url| url.replace("/chat/completions", "/audio/speech"))
            .unwrap_or_else(|| OPENAI_SPEECH_URL.to_string())
    }

    fn transcription_url(&self) -> String {
        self.config
            .base_url
            .as_ref()
            .map(|url| url.replace("/chat/completions", "/audio/transcriptions"))
            .unwrap_or_else(|| OPENAI_TRANSCRIPTION_URL.to_string())
    }

    fn translation_url(&self) -> String {
        self.config
            .base_url
            .as_ref()
            .map(|url| url.replace("/chat/completions", "/audio/translations"))
            .unwrap_or_else(|| OPENAI_TRANSLATION_URL.to_string())
    }
}

#[async_trait]
impl SpeechProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn speech(&self, request: SpeechRequest) -> Result<SpeechResponse> {
        self.config.require_api_key()?;

        let response_format = request.response_format.map(|f| match f {
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Opus => "opus",
            AudioFormat::Aac => "aac",
            AudioFormat::Flac => "flac",
            AudioFormat::Wav => "wav",
            AudioFormat::Pcm => "pcm",
        });

        let api_request = OpenAISpeechRequest {
            model: request.model,
            input: request.input,
            voice: request.voice,
            response_format,
            speed: request.speed,
        };

        let response = self
            .client
            .post(self.speech_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let audio = response.bytes().await?.to_vec();

        Ok(SpeechResponse::new(
            audio,
            request.response_format.unwrap_or(AudioFormat::Mp3),
        ))
    }

    fn available_voices(&self) -> &[VoiceInfo] {
        // OpenAI TTS voices: alloy, echo, fable, onyx, nova, shimmer
        // For now, return empty slice - voices can be discovered via API
        &[]
    }

    fn supported_formats(&self) -> &[AudioFormat] {
        &[
            AudioFormat::Mp3,
            AudioFormat::Opus,
            AudioFormat::Aac,
            AudioFormat::Flac,
            AudioFormat::Wav,
            AudioFormat::Pcm,
        ]
    }

    fn default_speech_model(&self) -> Option<&str> {
        Some("tts-1")
    }
}

#[async_trait]
impl TranscriptionProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn transcribe(&self, request: TranscriptionRequest) -> Result<TranscriptionResponse> {
        self.config.require_api_key()?;

        // Build multipart form
        let mut form = reqwest::multipart::Form::new().text("model", request.model.clone());

        // Add audio file
        match &request.audio {
            AudioInput::File(path) => {
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("audio.mp3")
                    .to_string();
                let data = tokio::fs::read(path).await.map_err(|e| {
                    Error::invalid_request(format!("Failed to read audio file: {}", e))
                })?;
                let part = reqwest::multipart::Part::bytes(data).file_name(filename);
                form = form.part("file", part);
            }
            AudioInput::Bytes {
                data,
                filename,
                media_type: _,
            } => {
                let part =
                    reqwest::multipart::Part::bytes(data.clone()).file_name(filename.clone());
                form = form.part("file", part);
            }
            AudioInput::Url(_) => {
                return Err(Error::invalid_request(
                    "OpenAI transcription does not support URLs directly",
                ));
            }
        }

        // Add optional parameters
        if let Some(language) = &request.language {
            form = form.text("language", language.clone());
        }

        if let Some(prompt) = &request.prompt {
            form = form.text("prompt", prompt.clone());
        }

        let response_format = request
            .response_format
            .unwrap_or(TranscriptFormat::VerboseJson);
        let format_str = match response_format {
            TranscriptFormat::Text => "text",
            TranscriptFormat::Json => "json",
            TranscriptFormat::VerboseJson => "verbose_json",
            TranscriptFormat::Srt => "srt",
            TranscriptFormat::Vtt => "vtt",
        };
        form = form.text("response_format", format_str);

        if let Some(granularities) = &request.timestamp_granularities {
            for g in granularities {
                let g_str = match g {
                    crate::audio::TimestampGranularity::Word => "word",
                    crate::audio::TimestampGranularity::Segment => "segment",
                };
                form = form.text("timestamp_granularities[]", g_str);
            }
        }

        let response = self
            .client
            .post(self.transcription_url())
            .multipart(form)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        // Parse response based on format
        match response_format {
            TranscriptFormat::Text | TranscriptFormat::Srt | TranscriptFormat::Vtt => {
                let text = response.text().await?;
                Ok(TranscriptionResponse::new(text))
            }
            TranscriptFormat::Json => {
                let json_response: OpenAITranscriptionSimple = response.json().await?;
                Ok(TranscriptionResponse::new(json_response.text))
            }
            TranscriptFormat::VerboseJson => {
                let json_response: OpenAITranscriptionVerbose = response.json().await?;

                let segments = json_response.segments.map(|segs| {
                    segs.into_iter()
                        .map(|s| TranscriptSegment {
                            id: s.id,
                            start: s.start,
                            end: s.end,
                            text: s.text,
                        })
                        .collect()
                });

                let words = json_response.words.map(|ws| {
                    ws.into_iter()
                        .map(|w| TranscriptWord {
                            word: w.word,
                            start: w.start,
                            end: w.end,
                        })
                        .collect()
                });

                Ok(TranscriptionResponse::new(json_response.text)
                    .with_language(json_response.language)
                    .with_duration(json_response.duration)
                    .with_segments(segments.unwrap_or_default())
                    .with_words(words.unwrap_or_default()))
            }
        }
    }

    async fn translate(&self, request: TranscriptionRequest) -> Result<TranscriptionResponse> {
        self.config.require_api_key()?;

        // Build multipart form
        let mut form = reqwest::multipart::Form::new().text("model", request.model.clone());

        // Add audio file
        match &request.audio {
            AudioInput::File(path) => {
                let filename = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("audio.mp3")
                    .to_string();
                let data = tokio::fs::read(path).await.map_err(|e| {
                    Error::invalid_request(format!("Failed to read audio file: {}", e))
                })?;
                let part = reqwest::multipart::Part::bytes(data).file_name(filename);
                form = form.part("file", part);
            }
            AudioInput::Bytes {
                data,
                filename,
                media_type: _,
            } => {
                let part =
                    reqwest::multipart::Part::bytes(data.clone()).file_name(filename.clone());
                form = form.part("file", part);
            }
            AudioInput::Url(_) => {
                return Err(Error::invalid_request(
                    "OpenAI translation does not support URLs directly",
                ));
            }
        }

        if let Some(prompt) = &request.prompt {
            form = form.text("prompt", prompt.clone());
        }

        form = form.text("response_format", "json");

        let response = self
            .client
            .post(self.translation_url())
            .multipart(form)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let json_response: OpenAITranscriptionSimple = response.json().await?;
        Ok(TranscriptionResponse::new(json_response.text))
    }

    fn default_transcription_model(&self) -> Option<&str> {
        Some("whisper-1")
    }
}

// OpenAI Audio API types

#[derive(Debug, Serialize)]
struct OpenAISpeechRequest {
    model: String,
    input: String,
    voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct OpenAITranscriptionSimple {
    text: String,
}

#[derive(Debug, Deserialize)]
struct OpenAITranscriptionVerbose {
    text: String,
    #[serde(default)]
    language: String,
    #[serde(default)]
    duration: f32,
    #[serde(default)]
    segments: Option<Vec<OpenAITranscriptSegment>>,
    #[serde(default)]
    words: Option<Vec<OpenAITranscriptWord>>,
}

#[derive(Debug, Deserialize)]
struct OpenAITranscriptSegment {
    id: usize,
    start: f32,
    end: f32,
    text: String,
}

#[derive(Debug, Deserialize)]
struct OpenAITranscriptWord {
    word: String,
    start: f32,
    end: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Provider;
    use crate::types::StructuredOutput;

    #[test]
    fn test_provider_creation() {
        let provider = OpenAIProvider::with_api_key("test-key").unwrap();
        assert_eq!(Provider::name(&provider), "openai");
        assert!(provider.supports_tools());
        assert!(provider.supports_vision());
    }

    #[test]
    fn test_request_conversion() {
        let provider = OpenAIProvider::with_api_key("test-key").unwrap();
        let request = CompletionRequest::new("gpt-4o", vec![Message::user("Hello")])
            .with_system("You are helpful")
            .with_max_tokens(1024);

        let openai_req = provider.convert_request(&request);

        assert_eq!(openai_req.model, "gpt-4o");
        assert_eq!(openai_req.max_tokens, Some(1024));
        assert_eq!(openai_req.messages.len(), 2); // system + user
    }

    #[test]
    fn test_structured_output_json_object() {
        let provider = OpenAIProvider::with_api_key("test-key").unwrap();
        let request = CompletionRequest::new("gpt-4o", vec![Message::user("Return JSON")])
            .with_response_format(StructuredOutput::json_object());

        let openai_req = provider.convert_request(&request);

        assert!(openai_req.response_format.is_some());
        match openai_req.response_format.unwrap() {
            OpenAIResponseFormat::JsonObject => {} // Expected
            _ => panic!("Expected JsonObject format"),
        }
    }

    #[test]
    fn test_structured_output_json_schema() {
        let provider = OpenAIProvider::with_api_key("test-key").unwrap();

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name", "age"]
        });

        // Use StructuredOutput::json_schema_with_description for test
        let request = CompletionRequest::new("gpt-4o", vec![Message::user("Get person info")])
            .with_response_format(StructuredOutput::json_schema_with_description(
                "Person",
                "A person object",
                schema.clone(),
            ));

        let openai_req = provider.convert_request(&request);

        assert!(openai_req.response_format.is_some());
        match openai_req.response_format.unwrap() {
            OpenAIResponseFormat::JsonSchema { json_schema } => {
                assert_eq!(json_schema.name, "Person");
                assert_eq!(json_schema.description, Some("A person object".to_string()));
                assert_eq!(json_schema.schema, schema);
                assert_eq!(json_schema.strict, Some(true));
            }
            _ => panic!("Expected JsonSchema format"),
        }
    }

    #[test]
    fn test_structured_output_json_schema_simple() {
        let provider = OpenAIProvider::with_api_key("test-key").unwrap();

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "result": { "type": "string" }
            }
        });

        // Use with_json_schema builder method
        let request = CompletionRequest::new("gpt-4o", vec![Message::user("Get result")])
            .with_json_schema("Result", schema.clone());

        let openai_req = provider.convert_request(&request);

        assert!(openai_req.response_format.is_some());
        match openai_req.response_format.unwrap() {
            OpenAIResponseFormat::JsonSchema { json_schema } => {
                assert_eq!(json_schema.name, "Result");
                assert_eq!(json_schema.description, None);
                assert_eq!(json_schema.schema, schema);
                assert_eq!(json_schema.strict, Some(true));
            }
            _ => panic!("Expected JsonSchema format"),
        }
    }

    #[test]
    fn test_predicted_output() {
        let provider = OpenAIProvider::with_api_key("test-key").unwrap();

        let predicted_content = "fn main() {\n    println!(\"Hello, world!\");\n}";
        let request = CompletionRequest::new("gpt-4o", vec![Message::user("Update the code")])
            .with_prediction(predicted_content);

        let openai_req = provider.convert_request(&request);

        assert!(openai_req.prediction.is_some());
        let prediction = openai_req.prediction.unwrap();
        assert_eq!(prediction.prediction_type, "content");
        assert_eq!(prediction.content, predicted_content);
    }

    #[test]
    fn test_combined_features() {
        let provider = OpenAIProvider::with_api_key("test-key").unwrap();

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "code": { "type": "string" }
            }
        });

        let request = CompletionRequest::new("gpt-4o", vec![Message::user("Generate code")])
            .with_system("You are a code generator")
            .with_max_tokens(4096)
            .with_json_schema("CodeOutput", schema)
            .with_prediction("fn example() {}");

        let openai_req = provider.convert_request(&request);

        assert_eq!(openai_req.model, "gpt-4o");
        assert_eq!(openai_req.max_tokens, Some(4096));
        assert!(openai_req.response_format.is_some());
        assert!(openai_req.prediction.is_some());
    }
}
