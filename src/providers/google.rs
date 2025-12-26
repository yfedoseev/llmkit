//! Google AI (Gemini) provider implementation.
//!
//! This module provides access to Google's Gemini models via the Google AI API.
//! For enterprise/production use with GCP integration, see the Vertex AI provider.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::GoogleProvider;
//!
//! // From environment variable
//! let provider = GoogleProvider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = GoogleProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! - `gemini-2.0-flash-exp` - Latest experimental model
//! - `gemini-1.5-pro` - Best for complex tasks
//! - `gemini-1.5-flash` - Fast and efficient
//! - `gemini-1.5-flash-8b` - Smallest, fastest model
//! - `gemini-pro` - Legacy model
//!
//! # Environment Variables
//!
//! - `GOOGLE_API_KEY` - Your Google AI API key

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

const GOOGLE_AI_BASE_URL: &str = "https://generativelanguage.googleapis.com/v1beta";

/// Google AI (Gemini) provider.
///
/// Provides access to Google's Gemini models through the Google AI API.
pub struct GoogleProvider {
    config: ProviderConfig,
    client: Client,
}

impl GoogleProvider {
    /// Create a new Google provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let client = Client::builder().timeout(config.timeout).build()?;

        Ok(Self { config, client })
    }

    /// Create a new Google provider from environment variable.
    ///
    /// Reads the API key from `GOOGLE_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("GOOGLE_API_KEY");
        Self::new(config)
    }

    /// Create a new Google provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn base_url(&self) -> &str {
        self.config
            .base_url
            .as_deref()
            .unwrap_or(GOOGLE_AI_BASE_URL)
    }

    fn api_url(&self, model: &str, streaming: bool) -> String {
        let base = self.base_url();
        let method = if streaming {
            "streamGenerateContent"
        } else {
            "generateContent"
        };
        format!(
            "{}/models/{}:{}?key={}",
            base,
            model,
            method,
            self.config.api_key.as_deref().unwrap_or("")
        )
    }

    /// Convert our unified request to Google's format.
    fn convert_request(&self, request: &CompletionRequest) -> GeminiRequest {
        use crate::types::StructuredOutputType;

        let mut contents = Vec::new();

        // Convert messages
        for msg in &request.messages {
            contents.extend(self.convert_message(msg));
        }

        // Handle structured output
        let (response_mime_type, response_schema) = if let Some(ref rf) = request.response_format {
            match rf.format_type {
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        (
                            Some("application/json".to_string()),
                            Some(schema_def.schema.clone()),
                        )
                    } else {
                        (Some("application/json".to_string()), None)
                    }
                }
                StructuredOutputType::JsonObject => (Some("application/json".to_string()), None),
                StructuredOutputType::Text => (None, None),
            }
        } else {
            (None, None)
        };

        // Build generation config
        let generation_config = Some(GeminiGenerationConfig {
            temperature: request.temperature,
            top_p: request.top_p,
            max_output_tokens: request.max_tokens,
            stop_sequences: request.stop_sequences.clone(),
            response_mime_type,
            response_schema,
        });

        // Build system instruction if present
        let system_instruction = request.system.as_ref().map(|s| GeminiContent {
            role: None, // System instruction doesn't have a role
            parts: vec![GeminiPart::Text { text: s.clone() }],
        });

        // Convert tools
        let tools = request.tools.as_ref().map(|tools| {
            vec![GeminiTool {
                function_declarations: tools
                    .iter()
                    .map(|t| GeminiFunctionDeclaration {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: Some(t.input_schema.clone()),
                    })
                    .collect(),
            }]
        });

        GeminiRequest {
            contents,
            generation_config,
            system_instruction,
            tools,
        }
    }

    fn convert_message(&self, message: &Message) -> Vec<GeminiContent> {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "model",
            Role::System => "user", // System messages handled separately
        };

        let mut parts = Vec::new();

        for block in &message.content {
            match block {
                ContentBlock::Text { text } => {
                    parts.push(GeminiPart::Text { text: text.clone() });
                }
                ContentBlock::Image { media_type, data } => {
                    parts.push(GeminiPart::InlineData {
                        inline_data: GeminiBlob {
                            mime_type: media_type.clone(),
                            data: data.clone(),
                        },
                    });
                }
                ContentBlock::ImageUrl { url } => {
                    // Gemini supports file URIs for Google Cloud Storage
                    parts.push(GeminiPart::FileData {
                        file_data: GeminiFileData {
                            mime_type: None,
                            file_uri: url.clone(),
                        },
                    });
                }
                ContentBlock::ToolUse { id, name, input } => {
                    parts.push(GeminiPart::FunctionCall {
                        function_call: GeminiFunctionCall {
                            name: name.clone(),
                            args: input.clone(),
                        },
                    });
                    // Store the ID for later reference (Gemini doesn't use IDs)
                    let _ = id;
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    parts.push(GeminiPart::FunctionResponse {
                        function_response: GeminiFunctionResponse {
                            name: tool_use_id.clone(), // Use tool_use_id as name fallback
                            response: serde_json::json!({
                                "result": content,
                                "is_error": is_error
                            }),
                        },
                    });
                }
                ContentBlock::Thinking { .. } => {
                    // Skip thinking blocks
                }
                _ => {
                    // Skip other blocks (Document, TextWithCache, etc.)
                }
            }
        }

        if parts.is_empty() {
            return vec![];
        }

        vec![GeminiContent {
            role: Some(role.to_string()),
            parts,
        }]
    }

    fn convert_response(&self, response: GeminiResponse) -> CompletionResponse {
        let mut content = Vec::new();
        let mut stop_reason = StopReason::EndTurn;

        // Get the first candidate
        if let Some(candidate) = response.candidates.into_iter().next() {
            // Extract stop reason first
            if let Some(ref reason) = candidate.finish_reason {
                stop_reason = match reason.as_str() {
                    "STOP" => StopReason::EndTurn,
                    "MAX_TOKENS" => StopReason::MaxTokens,
                    "SAFETY" => StopReason::ContentFilter,
                    "RECITATION" => StopReason::ContentFilter,
                    _ => StopReason::EndTurn,
                };
            }

            if let Some(gemini_content) = candidate.content {
                for part in gemini_content.parts {
                    match part {
                        GeminiPart::Text { text } => {
                            content.push(ContentBlock::Text { text });
                        }
                        GeminiPart::FunctionCall { function_call } => {
                            content.push(ContentBlock::ToolUse {
                                id: uuid::Uuid::new_v4().to_string(),
                                name: function_call.name,
                                input: function_call.args,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }

        let (input_tokens, output_tokens) = response
            .usage_metadata
            .map(|u| (u.prompt_token_count, u.candidates_token_count))
            .unwrap_or((0, 0));

        CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: "gemini".to_string(),
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

        match response.json::<GeminiErrorResponse>().await {
            Ok(err) => {
                let message = &err.error.message;
                let code = err.error.code;

                match code {
                    401 | 403 => Error::auth(message),
                    429 => Error::rate_limited(message, None),
                    400 => {
                        if message.contains("not found") {
                            Error::ModelNotFound(message.clone())
                        } else {
                            Error::invalid_request(message)
                        }
                    }
                    _ => Error::server(status, message),
                }
            }
            Err(_) => Error::server(status, "Unknown error"),
        }
    }
}

#[async_trait]
impl Provider for GoogleProvider {
    fn name(&self) -> &str {
        "google"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.config.require_api_key()?;

        let model = &request.model;
        let api_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url(model, false))
            .header("Content-Type", "application/json")
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let gemini_response: GeminiResponse = response.json().await?;
        Ok(self.convert_response(gemini_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        self.config.require_api_key()?;

        let model = &request.model;
        let api_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url(model, true))
            .header("Content-Type", "application/json")
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let stream = parse_gemini_stream(response);
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
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-pro-latest",
            "gemini-1.5-flash",
            "gemini-1.5-flash-latest",
            "gemini-1.5-flash-8b",
            "gemini-pro",
            "gemini-pro-vision",
        ])
    }

    fn default_model(&self) -> Option<&str> {
        Some("gemini-1.5-flash")
    }
}

/// Parse Gemini streaming response.
fn parse_gemini_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
    use async_stream::try_stream;
    use futures::StreamExt;

    try_stream! {
        let mut event_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut sent_start = false;
        let mut in_array = false;

        while let Some(chunk) = event_stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Gemini streams as JSON array elements
            // Format: [{"candidates":...},{"candidates":...},...]
            loop {
                // Skip whitespace and array brackets
                // We need to own the trimmed string to avoid borrow issues
                let trimmed = buffer.trim_start().to_string();
                if trimmed.is_empty() {
                    break;
                }

                if let Some(rest) = trimmed.strip_prefix('[') {
                    in_array = true;
                    buffer = rest.to_string();
                    continue;
                }

                if let Some(rest) = trimmed.strip_prefix(',') {
                    buffer = rest.to_string();
                    continue;
                }

                if let Some(rest) = trimmed.strip_prefix(']') {
                    buffer = rest.to_string();
                    yield StreamChunk {
                        event_type: StreamEventType::MessageStop,
                        index: None,
                        delta: None,
                        stop_reason: None,
                        usage: None,
                    };
                    break;
                }

                if !in_array && !trimmed.starts_with('{') {
                    break;
                }

                // Try to parse a complete JSON object
                let mut depth = 0;
                let mut end_pos = None;
                let mut in_string = false;
                let mut escape_next = false;

                for (i, c) in trimmed.char_indices() {
                    if escape_next {
                        escape_next = false;
                        continue;
                    }

                    match c {
                        '\\' if in_string => escape_next = true,
                        '"' => in_string = !in_string,
                        '{' if !in_string => depth += 1,
                        '}' if !in_string => {
                            depth -= 1;
                            if depth == 0 {
                                end_pos = Some(i + 1);
                                break;
                            }
                        }
                        _ => {}
                    }
                }

                if let Some(end) = end_pos {
                    let json_str = &trimmed[..end];
                    buffer = trimmed[end..].to_string();

                    if let Ok(parsed) = serde_json::from_str::<GeminiStreamResponse>(json_str) {
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

                        // Process candidates
                        for candidate in &parsed.candidates {
                            if let Some(ref content) = candidate.content {
                                for part in &content.parts {
                                    if let GeminiPart::Text { text } = part {
                                        yield StreamChunk {
                                            event_type: StreamEventType::ContentBlockDelta,
                                            index: Some(0),
                                            delta: Some(ContentDelta::TextDelta { text: text.clone() }),
                                            stop_reason: None,
                                            usage: None,
                                        };
                                    }
                                }
                            }

                            // Check for finish reason
                            if let Some(ref reason) = candidate.finish_reason {
                                let stop_reason = match reason.as_str() {
                                    "STOP" => StopReason::EndTurn,
                                    "MAX_TOKENS" => StopReason::MaxTokens,
                                    "SAFETY" => StopReason::ContentFilter,
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

                        // Handle usage metadata
                        if let Some(ref usage) = parsed.usage_metadata {
                            yield StreamChunk {
                                event_type: StreamEventType::MessageDelta,
                                index: None,
                                delta: None,
                                stop_reason: None,
                                usage: Some(Usage {
                                    input_tokens: usage.prompt_token_count,
                                    output_tokens: usage.candidates_token_count,
                                    cache_creation_input_tokens: 0,
                                    cache_read_input_tokens: 0,
                                }),
                            };
                        }
                    }
                } else {
                    // Incomplete JSON, wait for more data
                    break;
                }
            }
        }
    }
}

// ========== Gemini API Types ==========

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<GeminiGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<GeminiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<GeminiTool>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<GeminiPart>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum GeminiPart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: GeminiBlob,
    },
    FileData {
        #[serde(rename = "fileData")]
        file_data: GeminiFileData,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: GeminiFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: GeminiFunctionResponse,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiBlob {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiFileData {
    #[serde(skip_serializing_if = "Option::is_none")]
    mime_type: Option<String>,
    file_uri: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct GeminiFunctionResponse {
    name: String,
    response: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    /// MIME type for structured output (e.g., "application/json")
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
    /// JSON Schema for structured output (OpenAPI 3.0 format)
    #[serde(skip_serializing_if = "Option::is_none")]
    response_schema: Option<Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct GeminiTool {
    function_declarations: Vec<GeminiFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct GeminiFunctionDeclaration {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiStreamResponse {
    candidates: Vec<GeminiCandidate>,
    #[serde(default)]
    usage_metadata: Option<GeminiUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiCandidate {
    content: Option<GeminiContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct GeminiUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: u32,
}

#[derive(Debug, Deserialize)]
struct GeminiErrorResponse {
    error: GeminiError,
}

#[derive(Debug, Deserialize)]
struct GeminiError {
    code: u16,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = GoogleProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "google");
        assert!(provider.supports_tools());
        assert!(provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_api_url() {
        let provider = GoogleProvider::with_api_key("test-key").unwrap();

        let url = provider.api_url("gemini-1.5-flash", false);
        assert!(url.contains("gemini-1.5-flash"));
        assert!(url.contains("generateContent"));
        assert!(url.contains("key=test-key"));

        let stream_url = provider.api_url("gemini-1.5-flash", true);
        assert!(stream_url.contains("streamGenerateContent"));
    }

    #[test]
    fn test_request_conversion() {
        let provider = GoogleProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("gemini-1.5-flash", vec![Message::user("Hello")])
            .with_system("You are helpful")
            .with_max_tokens(1024);

        let gemini_req = provider.convert_request(&request);

        assert_eq!(gemini_req.contents.len(), 1);
        assert!(gemini_req.system_instruction.is_some());
        assert!(gemini_req.generation_config.is_some());

        let config = gemini_req.generation_config.unwrap();
        assert_eq!(config.max_output_tokens, Some(1024));
    }

    #[test]
    fn test_message_conversion() {
        let provider = GoogleProvider::with_api_key("test-key").unwrap();

        let user_msg = Message::user("Hello");
        let contents = provider.convert_message(&user_msg);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, Some("user".to_string()));

        let assistant_msg = Message::assistant("Hi there!");
        let contents = provider.convert_message(&assistant_msg);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role, Some("model".to_string()));
    }

    #[test]
    fn test_default_model() {
        let provider = GoogleProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.default_model(), Some("gemini-1.5-flash"));
    }

    #[test]
    fn test_structured_output_json_schema() {
        let provider = GoogleProvider::with_api_key("test-key").unwrap();
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "age"]
        });

        let request = CompletionRequest::new("gemini-1.5-flash", vec![Message::user("Hello")])
            .with_json_schema("person", schema.clone());

        let gemini_req = provider.convert_request(&request);

        let config = gemini_req.generation_config.unwrap();
        assert_eq!(
            config.response_mime_type,
            Some("application/json".to_string())
        );
        assert!(config.response_schema.is_some());
        assert_eq!(config.response_schema.unwrap(), schema);
    }

    #[test]
    fn test_structured_output_json_object() {
        let provider = GoogleProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("gemini-1.5-flash", vec![Message::user("Hello")])
            .with_json_output();

        let gemini_req = provider.convert_request(&request);

        let config = gemini_req.generation_config.unwrap();
        assert_eq!(
            config.response_mime_type,
            Some("application/json".to_string())
        );
        assert!(config.response_schema.is_none()); // No schema for simple JSON mode
    }
}
