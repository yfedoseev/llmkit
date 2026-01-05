//! Azure OpenAI API provider implementation.
//!
//! This provider supports Azure-hosted OpenAI models with all standard features.
//!
//! # Configuration
//!
//! Azure OpenAI requires:
//! - `resource_name`: Your Azure OpenAI resource name
//! - `deployment_id`: The deployment name for your model
//! - `api_key`: Azure API key
//! - `api_version`: API version (defaults to "2024-08-01-preview")
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::providers::azure::{AzureOpenAIProvider, AzureConfig};
//!
//! let config = AzureConfig::new("my-resource", "gpt-4o-deployment", "api-key");
//! let provider = AzureOpenAIProvider::new(config)?;
//! ```

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Message, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

/// Default Azure OpenAI API version.
pub const DEFAULT_API_VERSION: &str = "2025-01-01-preview";

/// Configuration for Azure OpenAI provider.
#[derive(Debug, Clone)]
pub struct AzureConfig {
    /// Azure OpenAI resource name (e.g., "my-resource" in my-resource.openai.azure.com)
    pub resource_name: String,

    /// Deployment ID (the name you gave your model deployment)
    pub deployment_id: String,

    /// Azure API key
    pub api_key: String,

    /// API version (defaults to "2024-08-01-preview")
    pub api_version: String,

    /// Request timeout
    pub timeout: std::time::Duration,

    /// Custom base URL (overrides resource_name-based URL)
    pub base_url: Option<String>,
}

impl AzureConfig {
    /// Create a new Azure config with required fields.
    pub fn new(
        resource_name: impl Into<String>,
        deployment_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Self {
        Self {
            resource_name: resource_name.into(),
            deployment_id: deployment_id.into(),
            api_key: api_key.into(),
            api_version: DEFAULT_API_VERSION.to_string(),
            timeout: std::time::Duration::from_secs(120),
            base_url: None,
        }
    }

    /// Create from environment variables.
    ///
    /// Reads:
    /// - `AZURE_OPENAI_RESOURCE_NAME` or `AZURE_OPENAI_ENDPOINT`
    /// - `AZURE_OPENAI_DEPLOYMENT_ID` or `AZURE_OPENAI_DEPLOYMENT`
    /// - `AZURE_OPENAI_API_KEY`
    /// - `AZURE_OPENAI_API_VERSION` (optional)
    pub fn from_env() -> Result<Self> {
        let resource_name = std::env::var("AZURE_OPENAI_RESOURCE_NAME")
            .or_else(|_| {
                // Try to extract from endpoint URL
                std::env::var("AZURE_OPENAI_ENDPOINT").map(|url| {
                    url.trim_start_matches("https://")
                        .split('.')
                        .next()
                        .unwrap_or(&url)
                        .to_string()
                })
            })
            .map_err(|_| {
                Error::config("AZURE_OPENAI_RESOURCE_NAME or AZURE_OPENAI_ENDPOINT required")
            })?;

        let deployment_id = std::env::var("AZURE_OPENAI_DEPLOYMENT_ID")
            .or_else(|_| std::env::var("AZURE_OPENAI_DEPLOYMENT"))
            .map_err(|_| Error::config("AZURE_OPENAI_DEPLOYMENT_ID required"))?;

        let api_key = std::env::var("AZURE_OPENAI_API_KEY")
            .map_err(|_| Error::config("AZURE_OPENAI_API_KEY required"))?;

        let api_version = std::env::var("AZURE_OPENAI_API_VERSION")
            .unwrap_or_else(|_| DEFAULT_API_VERSION.to_string());

        Ok(Self {
            resource_name,
            deployment_id,
            api_key,
            api_version,
            timeout: std::time::Duration::from_secs(120),
            base_url: std::env::var("AZURE_OPENAI_ENDPOINT").ok(),
        })
    }

    /// Builder: Set the API version.
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Builder: Set timeout.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Builder: Set custom base URL (overrides resource name).
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Get the full API endpoint URL.
    fn api_url(&self) -> String {
        if let Some(ref base) = self.base_url {
            format!(
                "{}/openai/deployments/{}/chat/completions?api-version={}",
                base.trim_end_matches('/'),
                self.deployment_id,
                self.api_version
            )
        } else {
            format!(
                "https://{}.openai.azure.com/openai/deployments/{}/chat/completions?api-version={}",
                self.resource_name, self.deployment_id, self.api_version
            )
        }
    }
}

/// Azure OpenAI API provider.
///
/// Provides access to OpenAI models hosted on Azure with full feature support
/// including streaming, tools, vision, and structured outputs.
pub struct AzureOpenAIProvider {
    config: AzureConfig,
    client: Client,
}

impl AzureOpenAIProvider {
    /// Create a new Azure OpenAI provider with the given configuration.
    pub fn new(config: AzureConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        // Azure uses api-key header instead of Bearer token
        headers.insert(
            "api-key",
            config
                .api_key
                .parse()
                .map_err(|_| Error::config("Invalid API key format"))?,
        );

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

    /// Create a new Azure OpenAI provider from environment variables.
    pub fn from_env() -> Result<Self> {
        let config = AzureConfig::from_env()?;
        Self::new(config)
    }

    /// Convert our unified request to Azure OpenAI's format.
    fn convert_request(&self, request: &CompletionRequest) -> AzureOpenAIRequest {
        let mut messages: Vec<AzureMessage> = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(AzureMessage {
                role: "system".to_string(),
                content: Some(AzureContent::Text(system.clone())),
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
                .map(|t| AzureTool {
                    tool_type: "function".to_string(),
                    function: AzureFunction {
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
                StructuredOutputType::JsonObject => AzureResponseFormat::JsonObject,
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        AzureResponseFormat::JsonSchema {
                            json_schema: AzureJsonSchema {
                                name: schema_def.name.clone(),
                                description: schema_def.description.clone(),
                                schema: schema_def.schema.clone(),
                                strict: Some(schema_def.strict),
                            },
                        }
                    } else {
                        AzureResponseFormat::JsonObject
                    }
                }
                StructuredOutputType::Text => AzureResponseFormat::Text,
            }
        });

        AzureOpenAIRequest {
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
        }
    }

    fn convert_message(&self, message: &Message) -> Vec<AzureMessage> {
        let mut result = Vec::new();

        match message.role {
            Role::System => {
                let text = message.text_content();
                if !text.is_empty() {
                    result.push(AzureMessage {
                        role: "system".to_string(),
                        content: Some(AzureContent::Text(text)),
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
                        result.push(AzureMessage {
                            role: "tool".to_string(),
                            content: Some(AzureContent::Text(content)),
                            tool_calls: None,
                            tool_call_id: Some(tool_call_id),
                        });
                    }
                } else {
                    // Regular user message
                    let content_parts: Vec<AzureContentPart> = message
                        .content
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text { text } => {
                                Some(AzureContentPart::Text { text: text.clone() })
                            }
                            ContentBlock::Image { media_type, data } => {
                                Some(AzureContentPart::ImageUrl {
                                    image_url: ImageUrl {
                                        url: format!("data:{};base64,{}", media_type, data),
                                        detail: None,
                                    },
                                })
                            }
                            ContentBlock::ImageUrl { url } => Some(AzureContentPart::ImageUrl {
                                image_url: ImageUrl {
                                    url: url.clone(),
                                    detail: None,
                                },
                            }),
                            _ => None,
                        })
                        .collect();

                    if content_parts.len() == 1 {
                        if let AzureContentPart::Text { text } = &content_parts[0] {
                            result.push(AzureMessage {
                                role: "user".to_string(),
                                content: Some(AzureContent::Text(text.clone())),
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        } else {
                            result.push(AzureMessage {
                                role: "user".to_string(),
                                content: Some(AzureContent::Parts(content_parts)),
                                tool_calls: None,
                                tool_call_id: None,
                            });
                        }
                    } else if !content_parts.is_empty() {
                        result.push(AzureMessage {
                            role: "user".to_string(),
                            content: Some(AzureContent::Parts(content_parts)),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                }
            }
            Role::Assistant => {
                // Check for tool calls
                let tool_calls: Vec<AzureToolCall> = message
                    .content
                    .iter()
                    .filter_map(|block| match block {
                        ContentBlock::ToolUse { id, name, input } => Some(AzureToolCall {
                            id: id.clone(),
                            call_type: "function".to_string(),
                            function: AzureFunctionCall {
                                name: name.clone(),
                                arguments: input.to_string(),
                            },
                        }),
                        _ => None,
                    })
                    .collect();

                let text_content = message.text_content();

                result.push(AzureMessage {
                    role: "assistant".to_string(),
                    content: if text_content.is_empty() {
                        None
                    } else {
                        Some(AzureContent::Text(text_content))
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

    fn convert_response(&self, response: AzureResponse) -> CompletionResponse {
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

        match response.json::<AzureErrorResponse>().await {
            Ok(err) => {
                let error_code = err.error.code.as_deref().unwrap_or("unknown");
                let message = &err.error.message;

                match error_code {
                    "Unauthorized" | "401" => Error::auth(message),
                    "RateLimitExceeded" | "429" => Error::rate_limited(message, None),
                    "InvalidRequest" | "400" => Error::invalid_request(message),
                    "DeploymentNotFound" | "404" => Error::ModelNotFound(message.clone()),
                    "ContextLengthExceeded" => Error::ContextLengthExceeded(message.clone()),
                    "ContentFilter" => {
                        Error::invalid_request(format!("Content filtered: {}", message))
                    }
                    _ => Error::server(status, message),
                }
            }
            Err(_) => Error::server(status, "Unknown error"),
        }
    }
}

#[async_trait]
impl Provider for AzureOpenAIProvider {
    fn name(&self) -> &str {
        "azure"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let mut api_request = self.convert_request(&request);
        api_request.stream = false;

        let response = self
            .client
            .post(self.config.api_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let azure_response: AzureResponse = response.json().await?;
        Ok(self.convert_response(azure_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let mut api_request = self.convert_request(&request);
        api_request.stream = true;

        let response = self
            .client
            .post(self.config.api_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let stream = parse_azure_stream(response);
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
        // Azure deployments can use any of these models
        Some(&[
            // GPT-5 (latest)
            "gpt-5",
            // GPT-4.1 series (April 2025)
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            // o-series reasoning models
            "o4-mini",
            "o3",
            "o3-mini",
            "o3-pro",
            "o1",
            "o1-mini",
            "o1-preview",
            // GPT-4o series
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4o-realtime-preview",
            "gpt-4o-transcribe",
            // GPT-4 (legacy)
            "gpt-4-turbo",
            "gpt-4",
            // GPT-3.5 (legacy)
            "gpt-35-turbo",
            "gpt-35-turbo-16k",
        ])
    }

    fn default_model(&self) -> Option<&str> {
        Some("gpt-4o")
    }
}

/// Parse Azure OpenAI SSE stream into our unified StreamChunk format.
fn parse_azure_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
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

                if let Ok(parsed) = serde_json::from_str::<AzureStreamResponse>(data) {
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

// Azure OpenAI API types (nearly identical to OpenAI)

#[derive(Debug, Serialize)]
struct AzureOpenAIRequest {
    messages: Vec<AzureMessage>,
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
    tools: Option<Vec<AzureTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream_options: Option<StreamOptions>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<AzureResponseFormat>,
}

#[derive(Debug, Serialize)]
struct StreamOptions {
    include_usage: bool,
}

/// Azure response format for structured outputs
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AzureResponseFormat {
    JsonObject,
    JsonSchema { json_schema: AzureJsonSchema },
    Text,
}

#[derive(Debug, Serialize)]
struct AzureJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct AzureMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<AzureContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<AzureToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum AzureContent {
    Text(String),
    Parts(Vec<AzureContentPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AzureContentPart {
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
struct AzureTool {
    #[serde(rename = "type")]
    tool_type: String,
    function: AzureFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct AzureFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct AzureToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: AzureFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct AzureFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct AzureResponse {
    id: String,
    model: String,
    choices: Vec<AzureChoice>,
    usage: Option<AzureUsage>,
}

#[derive(Debug, Default, Deserialize)]
struct AzureChoice {
    message: AzureResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct AzureResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<AzureToolCall>>,
}

#[derive(Debug, Deserialize)]
struct AzureStreamResponse {
    choices: Vec<AzureStreamChoice>,
    usage: Option<AzureUsage>,
}

#[derive(Debug, Deserialize)]
struct AzureStreamChoice {
    delta: AzureStreamDelta,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct AzureStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<AzureStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct AzureStreamToolCall {
    index: Option<usize>,
    id: Option<String>,
    function: Option<AzureStreamFunction>,
}

#[derive(Debug, Deserialize)]
struct AzureStreamFunction {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AzureUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct AzureErrorResponse {
    error: AzureError,
}

#[derive(Debug, Deserialize)]
struct AzureError {
    code: Option<String>,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::StructuredOutput;

    #[test]
    fn test_azure_config_creation() {
        let config = AzureConfig::new("my-resource", "my-deployment", "test-key");
        assert_eq!(config.resource_name, "my-resource");
        assert_eq!(config.deployment_id, "my-deployment");
        assert_eq!(config.api_version, DEFAULT_API_VERSION);
    }

    #[test]
    fn test_azure_api_url() {
        let config = AzureConfig::new("my-resource", "my-deployment", "test-key");
        let url = config.api_url();
        assert!(url.contains("my-resource.openai.azure.com"));
        assert!(url.contains("my-deployment"));
        assert!(url.contains("api-version="));
    }

    #[test]
    fn test_azure_api_url_with_custom_base() {
        let config = AzureConfig::new("my-resource", "my-deployment", "test-key")
            .with_base_url("https://custom.endpoint.com");
        let url = config.api_url();
        assert!(url.starts_with("https://custom.endpoint.com"));
        assert!(url.contains("my-deployment"));
    }

    #[test]
    fn test_provider_creation() {
        let config = AzureConfig::new("my-resource", "my-deployment", "test-key");
        let provider = AzureOpenAIProvider::new(config).unwrap();
        assert_eq!(provider.name(), "azure");
        assert!(provider.supports_tools());
        assert!(provider.supports_vision());
    }

    #[test]
    fn test_request_conversion() {
        let config = AzureConfig::new("my-resource", "my-deployment", "test-key");
        let provider = AzureOpenAIProvider::new(config).unwrap();
        let request = CompletionRequest::new("gpt-4o", vec![Message::user("Hello")])
            .with_system("You are helpful")
            .with_max_tokens(1024);

        let azure_req = provider.convert_request(&request);

        assert_eq!(azure_req.max_tokens, Some(1024));
        assert_eq!(azure_req.messages.len(), 2); // system + user
    }

    #[test]
    fn test_structured_output_json_schema() {
        let config = AzureConfig::new("my-resource", "my-deployment", "test-key");
        let provider = AzureOpenAIProvider::new(config).unwrap();

        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name", "age"]
        });

        let request = CompletionRequest::new("gpt-4o", vec![Message::user("Get person info")])
            .with_response_format(StructuredOutput::json_schema_with_description(
                "Person",
                "A person object",
                schema.clone(),
            ));

        let azure_req = provider.convert_request(&request);

        assert!(azure_req.response_format.is_some());
        match azure_req.response_format.unwrap() {
            AzureResponseFormat::JsonSchema { json_schema } => {
                assert_eq!(json_schema.name, "Person");
                assert_eq!(json_schema.description, Some("A person object".to_string()));
            }
            _ => panic!("Expected JsonSchema format"),
        }
    }
}
