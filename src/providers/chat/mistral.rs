//! Mistral AI provider implementation.
//!
//! Mistral provides state-of-the-art language models including Mistral 7B,
//! Mixtral, and their large/small variants.
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

const MISTRAL_API_URL: &str = "https://api.mistral.ai/v1/chat/completions";
const MISTRAL_EU_API_URL: &str = "https://api.eu.mistral.ai/v1/chat/completions";

/// Mistral region for API endpoint selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MistralRegion {
    /// Global API endpoint (default)
    #[default]
    Global,
    /// European Union API endpoint (GDPR-compliant)
    EU,
}

impl MistralRegion {
    /// Get the API endpoint URL for this region.
    pub fn api_url(&self) -> &'static str {
        match self {
            Self::Global => MISTRAL_API_URL,
            Self::EU => MISTRAL_EU_API_URL,
        }
    }
}

/// Mistral AI provider configuration.
#[derive(Debug, Clone)]
pub struct MistralConfig {
    /// Base provider configuration (API key, timeout, etc.)
    pub provider_config: ProviderConfig,
    /// Region for API endpoint
    pub region: MistralRegion,
}

impl MistralConfig {
    /// Create a new Mistral config with the given API key and region.
    pub fn new(api_key: impl Into<String>, region: MistralRegion) -> Self {
        Self {
            provider_config: ProviderConfig::new(api_key),
            region,
        }
    }

    /// Create a Mistral config from environment, using the specified region.
    pub fn from_env(region: MistralRegion) -> Self {
        Self {
            provider_config: ProviderConfig::from_env("MISTRAL_API_KEY"),
            region,
        }
    }
}

impl Default for MistralConfig {
    fn default() -> Self {
        Self {
            provider_config: ProviderConfig::default(),
            region: MistralRegion::Global,
        }
    }
}

/// Mistral AI provider.
///
/// Mistral provides models like Mistral 7B, Mixtral 8x7B, and their large variants.
///
/// # Example
///
/// ```ignore
/// use llmkit::providers::mistral::{MistralProvider, MistralRegion};
///
/// let provider = MistralProvider::from_env()?;
/// // or with specific region
/// let provider = MistralProvider::new(MistralRegion::EU)?;
/// ```
pub struct MistralProvider {
    config: MistralConfig,
    client: Client,
}

impl MistralProvider {
    /// Create a new Mistral provider with the given region.
    pub fn new(region: MistralRegion) -> Result<Self> {
        let config = MistralConfig::from_env(region);
        Self::with_config(config)
    }

    /// Create a new Mistral provider with global region from environment variable.
    pub fn from_env() -> Result<Self> {
        Self::new(MistralRegion::Global)
    }

    /// Create a new Mistral provider with an API key and global region.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = MistralConfig::new(api_key, MistralRegion::Global);
        Self::with_config(config)
    }

    /// Create a new Mistral provider with specific API key and region.
    pub fn with_api_key_and_region(
        api_key: impl Into<String>,
        region: MistralRegion,
    ) -> Result<Self> {
        let config = MistralConfig::new(api_key, region);
        Self::with_config(config)
    }

    /// Create a new Mistral provider with custom configuration.
    pub fn with_config(config: MistralConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.provider_config.api_key {
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
        for (key, value) in &config.provider_config.custom_headers {
            headers.insert(
                reqwest::header::HeaderName::try_from(key.as_str())
                    .map_err(|_| Error::config(format!("Invalid header name: {}", key)))?,
                value
                    .parse()
                    .map_err(|_| Error::config(format!("Invalid header value for {}", key)))?,
            );
        }

        let client = Client::builder()
            .timeout(config.provider_config.timeout)
            .default_headers(headers)
            .build()?;

        Ok(Self { config, client })
    }

    /// Get the API URL for the current region.
    fn api_url(&self) -> &str {
        self.config
            .provider_config
            .base_url
            .as_deref()
            .unwrap_or_else(|| self.config.region.api_url())
    }

    /// Convert our unified request to Mistral's format (OpenAI-compatible).
    fn convert_request(&self, request: &CompletionRequest) -> MistralRequest {
        let mut messages: Vec<MistralMessage> = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(MistralMessage {
                role: "system".to_string(),
                content: Some(MistralContent::Text(system.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        // Convert messages
        for msg in &request.messages {
            let mistral_msg = self.convert_message(msg);
            messages.push(mistral_msg);
        }

        // Convert tools if present
        let tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| MistralTool {
                    r#type: "function".to_string(),
                    function: MistralFunction {
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
                StructuredOutputType::JsonObject => MistralResponseFormat::JsonObject,
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        MistralResponseFormat::JsonSchema {
                            json_schema: MistralJsonSchema {
                                name: schema_def.name.clone(),
                                description: schema_def.description.clone(),
                                schema: schema_def.schema.clone(),
                                strict: Some(schema_def.strict),
                            },
                        }
                    } else {
                        MistralResponseFormat::JsonObject
                    }
                }
                StructuredOutputType::Text => MistralResponseFormat::Text,
            }
        });

        MistralRequest {
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
            stream: Some(request.stream),
            safe_prompt: None,
            random_seed: None,
            response_format,
        }
    }

    /// Convert a single message to Mistral format.
    fn convert_message(&self, msg: &Message) -> MistralMessage {
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
                return MistralMessage {
                    role: "tool".to_string(),
                    content: Some(MistralContent::Text(content.clone())),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                };
            }
        }

        // Check for tool use (assistant messages with tool calls)
        let tool_calls: Vec<MistralToolCall> = msg
            .content
            .iter()
            .filter_map(|block| {
                if let ContentBlock::ToolUse { id, name, input } = block {
                    Some(MistralToolCall {
                        id: id.clone(),
                        r#type: "function".to_string(),
                        function: MistralFunctionCall {
                            name: name.clone(),
                            arguments: serde_json::to_string(input).unwrap_or_default(),
                        },
                    })
                } else {
                    None
                }
            })
            .collect();

        // Extract text content (Mistral doesn't support vision in chat completions)
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

        MistralMessage {
            role: role.to_string(),
            content: if text_content.is_empty() {
                None
            } else {
                Some(MistralContent::Text(text_content))
            },
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls)
            },
            tool_call_id: None,
        }
    }

    /// Convert Mistral response to our unified format.
    fn convert_response(&self, response: MistralResponse) -> CompletionResponse {
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
impl Provider for MistralProvider {
    fn name(&self) -> &str {
        "mistral"
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        false // Mistral chat API doesn't support vision yet
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supported_models(&self) -> Option<&[&str]> {
        Some(&[
            "mistral-large-latest",
            "mistral-large-2411",
            "mistral-small-latest",
            "mistral-small-2409",
            "codestral-latest",
            "codestral-2405",
            "ministral-8b-latest",
            "ministral-3b-latest",
            "open-mistral-7b",
            "open-mixtral-8x7b",
            "open-mixtral-8x22b",
        ])
    }

    fn default_model(&self) -> Option<&str> {
        Some("mistral-large-latest")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let mistral_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .json(&mistral_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(status.as_u16(), error_text));
        }

        let mistral_response: MistralResponse = response.json().await?;
        Ok(self.convert_response(mistral_response))
    }

    async fn complete_stream(
        &self,
        mut request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        request.stream = true;
        let mistral_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .json(&mistral_request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(status.as_u16(), error_text));
        }

        let stream = parse_mistral_stream(response);
        Ok(Box::pin(stream))
    }
}

/// Parse Mistral SSE stream into our unified StreamChunk format.
fn parse_mistral_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
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

                if let Ok(parsed) = serde_json::from_str::<MistralStreamEvent>(data) {
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
// Mistral API Types (OpenAI-compatible with some Mistral extensions)
// ============================================================================

#[derive(Debug, Serialize)]
struct MistralRequest {
    model: String,
    messages: Vec<MistralMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<MistralTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    safe_prompt: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    random_seed: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<MistralResponseFormat>,
}

/// Response format for structured outputs.
#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum MistralResponseFormat {
    Text,
    JsonObject,
    JsonSchema { json_schema: MistralJsonSchema },
}

/// JSON schema for structured output.
#[derive(Debug, Serialize)]
struct MistralJsonSchema {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    schema: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    strict: Option<bool>,
}

#[derive(Debug, Serialize)]
struct MistralMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<MistralContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<MistralToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Clone)]
#[serde(untagged)]
enum MistralContent {
    Text(String),
}

#[derive(Debug, Serialize)]
struct MistralTool {
    r#type: String,
    function: MistralFunction,
}

#[derive(Debug, Serialize)]
struct MistralFunction {
    name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    parameters: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct MistralToolCall {
    id: String,
    r#type: String,
    function: MistralFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct MistralFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct MistralResponse {
    id: String,
    model: String,
    choices: Vec<MistralChoice>,
    usage: Option<MistralUsage>,
}

#[derive(Debug, Deserialize)]
struct MistralChoice {
    message: MistralResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MistralResponseMessage {
    content: Option<String>,
    tool_calls: Option<Vec<MistralToolCall>>,
}

#[derive(Debug, Deserialize)]
struct MistralUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

// Streaming types
#[derive(Debug, Deserialize)]
struct MistralStreamEvent {
    choices: Vec<MistralStreamChoice>,
    usage: Option<MistralUsage>,
}

#[derive(Debug, Deserialize)]
struct MistralStreamChoice {
    delta: Option<MistralStreamDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MistralStreamDelta {
    content: Option<String>,
    tool_calls: Option<Vec<MistralStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
struct MistralStreamToolCall {
    id: Option<String>,
    function: Option<MistralStreamFunction>,
}

#[derive(Debug, Deserialize)]
struct MistralStreamFunction {
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
        let provider = MistralProvider::with_api_key("test-key");
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.name(), "mistral");
        assert!(provider.supports_tools());
        assert!(!provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_default_model() {
        let provider = MistralProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.default_model(), Some("mistral-large-latest"));
    }

    #[test]
    fn test_supported_models() {
        let provider = MistralProvider::with_api_key("test-key").unwrap();
        let models = provider.supported_models().unwrap();
        assert!(models.contains(&"mistral-large-latest"));
        assert!(models.contains(&"codestral-latest"));
        assert!(models.contains(&"open-mixtral-8x7b"));
    }

    #[test]
    fn test_request_conversion() {
        let provider = MistralProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("mistral-large-latest", vec![Message::user("Hello!")])
            .with_system("You are a helpful assistant.")
            .with_max_tokens(1024);

        let mistral_request = provider.convert_request(&request);

        assert_eq!(mistral_request.model, "mistral-large-latest");
        assert_eq!(mistral_request.messages.len(), 2); // system + user
        assert_eq!(mistral_request.messages[0].role, "system");
        assert_eq!(mistral_request.messages[1].role, "user");
    }

    #[test]
    fn test_request_parameters() {
        let provider = MistralProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("mistral-large-latest", vec![Message::user("Hello")])
            .with_max_tokens(500)
            .with_temperature(0.8)
            .with_top_p(0.9);

        let mistral_request = provider.convert_request(&request);

        assert_eq!(mistral_request.max_tokens, Some(500));
        assert_eq!(mistral_request.temperature, Some(0.8));
        assert_eq!(mistral_request.top_p, Some(0.9));
    }

    #[test]
    fn test_response_conversion() {
        let provider = MistralProvider::with_api_key("test-key").unwrap();

        let mistral_response = MistralResponse {
            id: "cmpl-123".to_string(),
            model: "mistral-large-latest".to_string(),
            choices: vec![MistralChoice {
                message: MistralResponseMessage {
                    content: Some("Hello! How can I help you?".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(MistralUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
            }),
        };

        let response = provider.convert_response(mistral_response);

        assert_eq!(response.id, "cmpl-123");
        assert_eq!(response.model, "mistral-large-latest");
        assert_eq!(response.content.len(), 1);
        assert!(matches!(response.stop_reason, StopReason::EndTurn));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 20);
    }

    #[test]
    fn test_stop_reason_mapping() {
        let provider = MistralProvider::with_api_key("test-key").unwrap();

        // Test "stop" -> EndTurn
        let response1 = MistralResponse {
            id: "1".to_string(),
            model: "mistral-large-latest".to_string(),
            choices: vec![MistralChoice {
                message: MistralResponseMessage {
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
        let response2 = MistralResponse {
            id: "2".to_string(),
            model: "mistral-large-latest".to_string(),
            choices: vec![MistralChoice {
                message: MistralResponseMessage {
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
        let response3 = MistralResponse {
            id: "3".to_string(),
            model: "mistral-large-latest".to_string(),
            choices: vec![MistralChoice {
                message: MistralResponseMessage {
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
        let provider = MistralProvider::with_api_key("test-key").unwrap();

        let mistral_response = MistralResponse {
            id: "tool-resp-123".to_string(),
            model: "mistral-large-latest".to_string(),
            choices: vec![MistralChoice {
                message: MistralResponseMessage {
                    content: None,
                    tool_calls: Some(vec![MistralToolCall {
                        id: "call_abc123".to_string(),
                        r#type: "function".to_string(),
                        function: MistralFunctionCall {
                            name: "get_weather".to_string(),
                            arguments: r#"{"location": "Paris"}"#.to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: Some(MistralUsage {
                prompt_tokens: 50,
                completion_tokens: 30,
            }),
        };

        let response = provider.convert_response(mistral_response);

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
    fn test_api_url_global() {
        let provider = MistralProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.api_url(), MISTRAL_API_URL);
    }

    #[test]
    fn test_api_url_eu() {
        let provider =
            MistralProvider::with_api_key_and_region("test-key", MistralRegion::EU).unwrap();
        assert_eq!(provider.api_url(), MISTRAL_EU_API_URL);
        assert_eq!(
            provider.api_url(),
            "https://api.eu.mistral.ai/v1/chat/completions"
        );
    }

    #[test]
    fn test_region_enum() {
        assert_eq!(MistralRegion::Global.api_url(), MISTRAL_API_URL);
        assert_eq!(MistralRegion::EU.api_url(), MISTRAL_EU_API_URL);
        assert_eq!(MistralRegion::default(), MistralRegion::Global);
    }

    #[test]
    fn test_request_serialization() {
        let request = MistralRequest {
            model: "mistral-large-latest".to_string(),
            messages: vec![MistralMessage {
                role: "user".to_string(),
                content: Some(MistralContent::Text("Hello".to_string())),
                tool_calls: None,
                tool_call_id: None,
            }],
            tools: None,
            tool_choice: None,
            max_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: None,
            stream: Some(false),
            safe_prompt: None,
            random_seed: None,
            response_format: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("mistral-large-latest"));
        assert!(json.contains("Hello"));
        assert!(json.contains("1000"));
        assert!(json.contains("0.7"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "resp-123",
            "model": "mistral-large-latest",
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

        let response: MistralResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, "resp-123");
        assert_eq!(response.model, "mistral-large-latest");
        assert_eq!(response.choices.len(), 1);
        assert_eq!(
            response.choices[0].message.content,
            Some("Hello there!".to_string())
        );
        assert_eq!(response.choices[0].finish_reason, Some("stop".to_string()));
        assert_eq!(response.usage.as_ref().unwrap().prompt_tokens, 15);
        assert_eq!(response.usage.as_ref().unwrap().completion_tokens, 25);
    }

    #[test]
    fn test_multi_turn_conversation() {
        let provider = MistralProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new(
            "mistral-large-latest",
            vec![
                Message::user("What is 2+2?"),
                Message::assistant("4"),
                Message::user("And 3+3?"),
            ],
        )
        .with_system("You are a math tutor");

        let mistral_request = provider.convert_request(&request);

        // system + 3 user/assistant messages
        assert_eq!(mistral_request.messages.len(), 4);
        assert_eq!(mistral_request.messages[0].role, "system");
        assert_eq!(mistral_request.messages[1].role, "user");
        assert_eq!(mistral_request.messages[2].role, "assistant");
        assert_eq!(mistral_request.messages[3].role, "user");
    }

    #[test]
    fn test_tool_result_message() {
        let provider = MistralProvider::with_api_key("test-key").unwrap();

        let tool_result_msg = Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "call_abc123".to_string(),
                content: "The weather in Paris is sunny, 22°C".to_string(),
                is_error: false,
            }],
        };

        let mistral_msg = provider.convert_message(&tool_result_msg);

        assert_eq!(mistral_msg.role, "tool");
        assert_eq!(mistral_msg.tool_call_id, Some("call_abc123".to_string()));
        if let Some(MistralContent::Text(content)) = mistral_msg.content {
            assert!(content.contains("sunny"));
        } else {
            panic!("Expected text content");
        }
    }
}
