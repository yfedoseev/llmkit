//! Google Vertex AI provider implementation.
//!
//! This module provides access to Google's Gemini models via Vertex AI,
//! Google Cloud's enterprise ML platform.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::VertexProvider;
//!
//! // From environment variables
//! let provider = VertexProvider::from_env()?;
//!
//! // With explicit configuration
//! let provider = VertexProvider::new(
//!     "my-project-id",
//!     "us-central1",
//!     "your-access-token",
//! )?;
//! ```
//!
//! # Authentication
//!
//! Vertex AI uses Google Cloud authentication. You can provide an access token
//! directly or use the gcloud CLI to obtain one:
//!
//! ```bash
//! gcloud auth print-access-token
//! ```
//!
//! # Environment Variables
//!
//! - `GOOGLE_CLOUD_PROJECT` or `VERTEX_PROJECT` - Your GCP project ID
//! - `GOOGLE_CLOUD_LOCATION` or `VERTEX_LOCATION` - Region (default: us-central1)
//! - `VERTEX_ACCESS_TOKEN` - OAuth2 access token
//!
//! # Supported Models
//!
//! - `gemini-2.0-flash-exp` - Latest experimental
//! - `gemini-1.5-pro` - Best for complex tasks
//! - `gemini-1.5-flash` - Fast and efficient

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

/// Vertex AI provider configuration.
#[derive(Debug, Clone)]
pub struct VertexConfig {
    /// GCP project ID.
    pub project_id: String,
    /// GCP location/region (e.g., "us-central1").
    pub location: String,
    /// OAuth2 access token.
    pub access_token: String,
    /// Request timeout.
    pub timeout: std::time::Duration,
}

impl VertexConfig {
    /// Create a new Vertex AI configuration.
    pub fn new(
        project_id: impl Into<String>,
        location: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Self {
        Self {
            project_id: project_id.into(),
            location: location.into(),
            access_token: access_token.into(),
            timeout: std::time::Duration::from_secs(300),
        }
    }

    /// Create configuration from environment variables.
    ///
    /// Reads:
    /// - `GOOGLE_CLOUD_PROJECT` or `VERTEX_PROJECT`
    /// - `GOOGLE_CLOUD_LOCATION` or `VERTEX_LOCATION` (default: us-central1)
    /// - `VERTEX_ACCESS_TOKEN`
    pub fn from_env() -> Result<Self> {
        let project_id = std::env::var("GOOGLE_CLOUD_PROJECT")
            .or_else(|_| std::env::var("VERTEX_PROJECT"))
            .map_err(|_| {
                Error::config("GOOGLE_CLOUD_PROJECT or VERTEX_PROJECT environment variable not set")
            })?;

        let location = std::env::var("GOOGLE_CLOUD_LOCATION")
            .or_else(|_| std::env::var("VERTEX_LOCATION"))
            .unwrap_or_else(|_| "us-central1".to_string());

        let access_token = std::env::var("VERTEX_ACCESS_TOKEN")
            .map_err(|_| Error::config("VERTEX_ACCESS_TOKEN environment variable not set"))?;

        Ok(Self {
            project_id,
            location,
            access_token,
            timeout: std::time::Duration::from_secs(300),
        })
    }

    /// Set the request timeout.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Google Vertex AI provider.
///
/// Provides access to Gemini models through Google Cloud's Vertex AI platform.
pub struct VertexProvider {
    config: VertexConfig,
    client: Client,
}

impl VertexProvider {
    /// Create a new Vertex AI provider.
    pub fn new(
        project_id: impl Into<String>,
        location: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Result<Self> {
        let config = VertexConfig::new(project_id, location, access_token);
        Self::with_config(config)
    }

    /// Create a new Vertex AI provider from environment variables.
    pub fn from_env() -> Result<Self> {
        let config = VertexConfig::from_env()?;
        Self::with_config(config)
    }

    /// Create a new Vertex AI provider with custom configuration.
    pub fn with_config(config: VertexConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", config.access_token)
                .parse()
                .map_err(|_| Error::config("Invalid access token format"))?,
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

    fn api_url(&self, model: &str, streaming: bool) -> String {
        let method = if streaming {
            "streamGenerateContent"
        } else {
            "generateContent"
        };

        format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/google/models/{}:{}",
            self.config.location,
            self.config.project_id,
            self.config.location,
            model,
            method
        )
    }

    /// Convert our unified request to Vertex AI's format.
    fn convert_request(&self, request: &CompletionRequest) -> VertexRequest {
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
        let generation_config = Some(VertexGenerationConfig {
            temperature: request.temperature,
            top_p: request.top_p,
            max_output_tokens: request.max_tokens,
            stop_sequences: request.stop_sequences.clone(),
            response_mime_type,
            response_schema,
        });

        // Build system instruction if present
        let system_instruction = request.system.as_ref().map(|s| VertexContent {
            role: None,
            parts: vec![VertexPart::Text { text: s.clone() }],
        });

        // Convert tools
        let tools = request.tools.as_ref().map(|tools| {
            vec![VertexTool {
                function_declarations: tools
                    .iter()
                    .map(|t| VertexFunctionDeclaration {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: Some(t.input_schema.clone()),
                    })
                    .collect(),
            }]
        });

        VertexRequest {
            contents,
            generation_config,
            system_instruction,
            tools,
        }
    }

    fn convert_message(&self, message: &Message) -> Vec<VertexContent> {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "model",
            Role::System => "user",
        };

        let mut parts = Vec::new();

        for block in &message.content {
            match block {
                ContentBlock::Text { text } => {
                    parts.push(VertexPart::Text { text: text.clone() });
                }
                ContentBlock::Image { media_type, data } => {
                    parts.push(VertexPart::InlineData {
                        inline_data: VertexBlob {
                            mime_type: media_type.clone(),
                            data: data.clone(),
                        },
                    });
                }
                ContentBlock::ImageUrl { url } => {
                    parts.push(VertexPart::FileData {
                        file_data: VertexFileData {
                            mime_type: None,
                            file_uri: url.clone(),
                        },
                    });
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    parts.push(VertexPart::FunctionCall {
                        function_call: VertexFunctionCall {
                            name: name.clone(),
                            args: input.clone(),
                        },
                    });
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    parts.push(VertexPart::FunctionResponse {
                        function_response: VertexFunctionResponse {
                            name: tool_use_id.clone(),
                            response: serde_json::json!({
                                "result": content,
                                "is_error": is_error
                            }),
                        },
                    });
                }
                ContentBlock::Thinking { .. } => {}
                _ => {
                    // Skip other blocks (Document, TextWithCache, etc.)
                }
            }
        }

        if parts.is_empty() {
            return vec![];
        }

        vec![VertexContent {
            role: Some(role.to_string()),
            parts,
        }]
    }

    fn convert_response(&self, response: VertexResponse) -> CompletionResponse {
        let mut content = Vec::new();
        let mut stop_reason = StopReason::EndTurn;

        if let Some(candidate) = response.candidates.into_iter().next() {
            // Extract stop reason first
            if let Some(ref reason) = candidate.finish_reason {
                stop_reason = match reason.as_str() {
                    "STOP" => StopReason::EndTurn,
                    "MAX_TOKENS" => StopReason::MaxTokens,
                    "SAFETY" => StopReason::ContentFilter,
                    _ => StopReason::EndTurn,
                };
            }

            if let Some(vertex_content) = candidate.content {
                for part in vertex_content.parts {
                    match part {
                        VertexPart::Text { text } => {
                            content.push(ContentBlock::Text { text });
                        }
                        VertexPart::FunctionCall { function_call } => {
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
            model: "vertex".to_string(),
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

        match response.json::<VertexErrorResponse>().await {
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
impl Provider for VertexProvider {
    fn name(&self) -> &str {
        "vertex"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = &request.model;
        let api_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url(model, false))
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let vertex_response: VertexResponse = response.json().await?;
        Ok(self.convert_response(vertex_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let model = &request.model;
        let api_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url(model, true))
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let stream = parse_vertex_stream(response);
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
            "gemini-1.5-flash",
            "gemini-1.5-flash-002",
            "gemini-1.5-pro-002",
        ])
    }

    fn default_model(&self) -> Option<&str> {
        Some("gemini-1.5-flash")
    }
}

/// Parse Vertex AI streaming response.
fn parse_vertex_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
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

            loop {
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

                // Parse JSON object
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

                    if let Ok(parsed) = serde_json::from_str::<VertexStreamResponse>(json_str) {
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

                        for candidate in &parsed.candidates {
                            if let Some(ref content) = candidate.content {
                                for part in &content.parts {
                                    if let VertexPart::Text { text } = part {
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
                    break;
                }
            }
        }
    }
}

// ========== Vertex AI API Types ==========

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexRequest {
    contents: Vec<VertexContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<VertexGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<VertexContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<VertexTool>>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<VertexPart>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum VertexPart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: VertexBlob,
    },
    FileData {
        #[serde(rename = "fileData")]
        file_data: VertexFileData,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: VertexFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: VertexFunctionResponse,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexBlob {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexFileData {
    #[serde(skip_serializing_if = "Option::is_none")]
    mime_type: Option<String>,
    file_uri: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexFunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexFunctionResponse {
    name: String,
    response: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexGenerationConfig {
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
struct VertexTool {
    function_declarations: Vec<VertexFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct VertexFunctionDeclaration {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexResponse {
    candidates: Vec<VertexCandidate>,
    #[serde(default)]
    usage_metadata: Option<VertexUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexStreamResponse {
    candidates: Vec<VertexCandidate>,
    #[serde(default)]
    usage_metadata: Option<VertexUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexCandidate {
    content: Option<VertexContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: u32,
}

#[derive(Debug, Deserialize)]
struct VertexErrorResponse {
    error: VertexError,
}

#[derive(Debug, Deserialize)]
struct VertexError {
    code: u16,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertex_config_creation() {
        let config = VertexConfig::new("my-project", "us-central1", "test-token");
        assert_eq!(config.project_id, "my-project");
        assert_eq!(config.location, "us-central1");
        assert_eq!(config.access_token, "test-token");
    }

    #[test]
    fn test_provider_creation() {
        let provider = VertexProvider::new("my-project", "us-central1", "test-token").unwrap();
        assert_eq!(provider.name(), "vertex");
        assert!(provider.supports_tools());
        assert!(provider.supports_vision());
    }

    #[test]
    fn test_api_url() {
        let provider = VertexProvider::new("my-project", "us-central1", "test-token").unwrap();

        let url = provider.api_url("gemini-1.5-flash", false);
        assert!(url.contains("my-project"));
        assert!(url.contains("us-central1"));
        assert!(url.contains("gemini-1.5-flash"));
        assert!(url.contains("generateContent"));

        let stream_url = provider.api_url("gemini-1.5-flash", true);
        assert!(stream_url.contains("streamGenerateContent"));
    }

    #[test]
    fn test_request_conversion() {
        let provider = VertexProvider::new("my-project", "us-central1", "test-token").unwrap();

        let request = CompletionRequest::new("gemini-1.5-flash", vec![Message::user("Hello")])
            .with_system("You are helpful")
            .with_max_tokens(1024);

        let vertex_req = provider.convert_request(&request);

        assert_eq!(vertex_req.contents.len(), 1);
        assert!(vertex_req.system_instruction.is_some());
        assert!(vertex_req.generation_config.is_some());
    }

    #[test]
    fn test_default_model() {
        let provider = VertexProvider::new("my-project", "us-central1", "test-token").unwrap();
        assert_eq!(provider.default_model(), Some("gemini-1.5-flash"));
    }
}
