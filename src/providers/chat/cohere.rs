//! Cohere API provider implementation.
//!
//! This module provides access to Cohere's Command models for chat completions.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::CohereProvider;
//!
//! // From environment variable
//! let provider = CohereProvider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = CohereProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! - `command-r-plus` - Most capable, best for complex tasks
//! - `command-r` - Balanced performance and speed
//! - `command` - Fast, efficient model
//! - `command-light` - Fastest, most economical
//!
//! # Environment Variables
//!
//! - `COHERE_API_KEY` or `CO_API_KEY` - Your Cohere API key

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

const COHERE_API_URL: &str = "https://api.cohere.ai/v1/chat";

/// Cohere API provider.
///
/// Provides access to Cohere's Command family of models.
pub struct CohereProvider {
    config: ProviderConfig,
    client: Client,
}

impl CohereProvider {
    /// Create a new Cohere provider with the given configuration.
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

    /// Create a new Cohere provider from environment variable.
    ///
    /// Reads the API key from `COHERE_API_KEY` or `CO_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("COHERE_API_KEY")
            .or_else(|_| std::env::var("CO_API_KEY"))
            .ok();

        let config = ProviderConfig {
            api_key,
            ..Default::default()
        };

        Self::new(config)
    }

    /// Create a new Cohere provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn api_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(COHERE_API_URL)
    }

    /// Convert our unified request to Cohere's format.
    fn convert_request(&self, request: &CompletionRequest) -> CohereRequest {
        // Extract the last user message as the main message
        let message = request
            .messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .map(|m| m.text_content())
            .unwrap_or_default();

        // Build chat history from all messages except the last user message
        let mut chat_history = Vec::new();
        let mut skip_last_user = true;

        for msg in request.messages.iter().rev() {
            if skip_last_user && msg.role == Role::User {
                skip_last_user = false;
                continue;
            }

            let role = match msg.role {
                Role::User => "USER",
                Role::Assistant => "CHATBOT",
                Role::System => "SYSTEM",
            };

            chat_history.push(CohereChatMessage {
                role: role.to_string(),
                message: msg.text_content(),
            });
        }

        chat_history.reverse();

        // Convert tools
        let tools = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| CohereTool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameter_definitions: Some(convert_json_schema_to_cohere_params(
                        &t.input_schema,
                    )),
                })
                .collect()
        });

        CohereRequest {
            model: request.model.clone(),
            message,
            preamble: request.system.clone(),
            chat_history: if chat_history.is_empty() {
                None
            } else {
                Some(chat_history)
            },
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stop_sequences: request.stop_sequences.clone(),
            stream: Some(false),
            tools,
        }
    }

    fn convert_response(&self, response: CohereResponse) -> CompletionResponse {
        let mut content = Vec::new();

        // Add text content
        if !response.text.is_empty() {
            content.push(ContentBlock::Text {
                text: response.text,
            });
        }

        // Add tool calls
        if let Some(tool_calls) = response.tool_calls {
            for tc in tool_calls {
                content.push(ContentBlock::ToolUse {
                    id: uuid::Uuid::new_v4().to_string(),
                    name: tc.name,
                    input: tc.parameters,
                });
            }
        }

        let stop_reason = match response.finish_reason.as_deref() {
            Some("COMPLETE") => StopReason::EndTurn,
            Some("MAX_TOKENS") => StopReason::MaxTokens,
            Some("TOOL_CALL") => StopReason::ToolUse,
            Some("ERROR") | Some("ERROR_TOXIC") | Some("ERROR_LIMIT") => StopReason::ContentFilter,
            _ => StopReason::EndTurn,
        };

        let (input_tokens, output_tokens) = response
            .meta
            .and_then(|m| m.tokens)
            .map(|t| (t.input_tokens, t.output_tokens))
            .unwrap_or((0, 0));

        CompletionResponse {
            id: response.generation_id.unwrap_or_default(),
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

        match response.json::<CohereErrorResponse>().await {
            Ok(err) => {
                let message = &err.message;

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

/// Convert JSON Schema to Cohere parameter definitions.
fn convert_json_schema_to_cohere_params(
    schema: &Value,
) -> std::collections::HashMap<String, CohereParameterDefinition> {
    let mut params = std::collections::HashMap::new();

    if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
        let required: Vec<String> = schema
            .get("required")
            .and_then(|r| r.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        for (name, prop) in properties {
            let param_type = prop
                .get("type")
                .and_then(|t| t.as_str())
                .unwrap_or("string")
                .to_string();

            let description = prop
                .get("description")
                .and_then(|d| d.as_str())
                .map(String::from);

            params.insert(
                name.clone(),
                CohereParameterDefinition {
                    param_type,
                    description,
                    required: required.contains(name),
                },
            );
        }
    }

    params
}

#[async_trait]
impl Provider for CohereProvider {
    fn name(&self) -> &str {
        "cohere"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.config.require_api_key()?;

        let api_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let cohere_response: CohereResponse = response.json().await?;
        Ok(self.convert_response(cohere_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        self.config.require_api_key()?;

        let mut api_request = self.convert_request(&request);
        api_request.stream = Some(true);

        let response = self
            .client
            .post(self.api_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let stream = parse_cohere_stream(response);
        Ok(Box::pin(stream))
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        false // Cohere doesn't support vision in chat API
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supported_models(&self) -> Option<&[&str]> {
        Some(&[
            "command-r-plus",
            "command-r",
            "command",
            "command-light",
            "command-r-plus-08-2024",
            "command-r-08-2024",
        ])
    }

    fn default_model(&self) -> Option<&str> {
        Some("command-r")
    }
}

/// Parse Cohere streaming response.
fn parse_cohere_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
    use async_stream::try_stream;
    use futures::StreamExt;

    try_stream! {
        let mut event_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut sent_start = false;

        while let Some(chunk) = event_stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // Process complete lines (Cohere uses newline-delimited JSON)
            while let Some(pos) = buffer.find('\n') {
                let line = buffer[..pos].trim().to_string();
                buffer = buffer[pos + 1..].to_string();

                if line.is_empty() {
                    continue;
                }

                if let Ok(event) = serde_json::from_str::<CohereStreamEvent>(&line) {
                    match event.event_type.as_str() {
                        "stream-start" => {
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
                        }
                        "text-generation" => {
                            if let Some(text) = event.text {
                                yield StreamChunk {
                                    event_type: StreamEventType::ContentBlockDelta,
                                    index: Some(0),
                                    delta: Some(ContentDelta::Text { text }),
                                    stop_reason: None,
                                    usage: None,
                                };
                            }
                        }
                        "stream-end" => {
                            let stop_reason = event.finish_reason.map(|r| {
                                match r.as_str() {
                                    "COMPLETE" => StopReason::EndTurn,
                                    "MAX_TOKENS" => StopReason::MaxTokens,
                                    "TOOL_CALL" => StopReason::ToolUse,
                                    _ => StopReason::EndTurn,
                                }
                            });

                            // Extract usage from response
                            let usage = event.response.and_then(|r| {
                                r.meta.and_then(|m| m.tokens).map(|t| Usage {
                                    input_tokens: t.input_tokens,
                                    output_tokens: t.output_tokens,
                                    cache_creation_input_tokens: 0,
                                    cache_read_input_tokens: 0,
                                })
                            });

                            yield StreamChunk {
                                event_type: StreamEventType::MessageDelta,
                                index: None,
                                delta: None,
                                stop_reason,
                                usage,
                            };

                            yield StreamChunk {
                                event_type: StreamEventType::MessageStop,
                                index: None,
                                delta: None,
                                stop_reason: None,
                                usage: None,
                            };
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}

// ========== Cohere API Types ==========

#[derive(Debug, Serialize)]
struct CohereRequest {
    model: String,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    preamble: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_history: Option<Vec<CohereChatMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<CohereTool>>,
}

#[derive(Debug, Serialize)]
struct CohereChatMessage {
    role: String,
    message: String,
}

#[derive(Debug, Serialize)]
struct CohereTool {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameter_definitions: Option<std::collections::HashMap<String, CohereParameterDefinition>>,
}

#[derive(Debug, Serialize)]
struct CohereParameterDefinition {
    #[serde(rename = "type")]
    param_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    required: bool,
}

#[derive(Debug, Deserialize)]
struct CohereResponse {
    text: String,
    #[serde(default)]
    generation_id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<CohereToolCall>>,
    #[serde(default)]
    meta: Option<CohereMeta>,
}

#[derive(Debug, Deserialize)]
struct CohereToolCall {
    name: String,
    parameters: Value,
}

#[derive(Debug, Deserialize)]
struct CohereMeta {
    tokens: Option<CohereTokens>,
}

#[derive(Debug, Deserialize)]
struct CohereTokens {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct CohereStreamEvent {
    event_type: String,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    response: Option<CohereResponse>,
}

#[derive(Debug, Deserialize)]
struct CohereErrorResponse {
    message: String,
}

// ============================================================================
// EmbeddingProvider Implementation
// ============================================================================

use crate::embedding::{
    Embedding, EmbeddingInput, EmbeddingInputType, EmbeddingProvider, EmbeddingRequest,
    EmbeddingResponse, EmbeddingUsage,
};

const COHERE_EMBED_URL: &str = "https://api.cohere.ai/v1/embed";

impl CohereProvider {
    fn embed_url(&self) -> String {
        self.config
            .base_url
            .as_ref()
            .map(|url| url.replace("/chat", "/embed"))
            .unwrap_or_else(|| COHERE_EMBED_URL.to_string())
    }
}

#[async_trait]
impl EmbeddingProvider for CohereProvider {
    fn name(&self) -> &str {
        "cohere"
    }

    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        self.config.require_api_key()?;

        let texts = match &request.input {
            EmbeddingInput::Single(text) => vec![text.clone()],
            EmbeddingInput::Batch(texts) => texts.clone(),
        };

        let input_type = request.input_type.map(|t| match t {
            EmbeddingInputType::Query => "search_query".to_string(),
            EmbeddingInputType::Document => "search_document".to_string(),
        });

        let api_request = CohereEmbedRequest {
            model: request.model.clone(),
            texts,
            input_type: input_type.unwrap_or_else(|| "search_document".to_string()),
            embedding_types: Some(vec!["float".to_string()]),
            truncate: Some("END".to_string()),
        };

        let response = self
            .client
            .post(self.embed_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let api_response: CohereEmbedResponse = response.json().await?;

        // Cohere returns embeddings in embedding_types.float array
        let float_embeddings = api_response
            .embeddings
            .and_then(|e| e.float)
            .unwrap_or_default();

        let embeddings = float_embeddings
            .into_iter()
            .enumerate()
            .map(|(i, values)| Embedding::new(i, values))
            .collect();

        let usage = api_response.meta.and_then(|m| m.billed_units).map_or_else(
            || EmbeddingUsage::new(0, 0),
            |u| EmbeddingUsage::new(u.input_tokens.unwrap_or(0), u.input_tokens.unwrap_or(0)),
        );

        Ok(EmbeddingResponse {
            model: request.model,
            embeddings,
            usage,
        })
    }

    fn embedding_dimensions(&self, model: &str) -> Option<usize> {
        match model {
            "embed-english-v3.0" | "embed-multilingual-v3.0" => Some(1024),
            "embed-english-light-v3.0" | "embed-multilingual-light-v3.0" => Some(384),
            "embed-english-v2.0" => Some(4096),
            _ => None,
        }
    }

    fn default_embedding_model(&self) -> Option<&str> {
        Some("embed-english-v3.0")
    }

    fn max_batch_size(&self) -> usize {
        96 // Cohere's limit
    }

    fn supported_embedding_models(&self) -> Option<&[&str]> {
        Some(&[
            "embed-english-v3.0",
            "embed-multilingual-v3.0",
            "embed-english-light-v3.0",
            "embed-multilingual-light-v3.0",
        ])
    }
}

#[derive(Debug, Serialize)]
struct CohereEmbedRequest {
    model: String,
    texts: Vec<String>,
    input_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_types: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncate: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CohereEmbedResponse {
    #[serde(default)]
    embeddings: Option<CohereEmbeddings>,
    #[serde(default)]
    meta: Option<CohereEmbedMeta>,
}

#[derive(Debug, Deserialize)]
struct CohereEmbeddings {
    #[serde(default)]
    float: Option<Vec<Vec<f32>>>,
}

#[derive(Debug, Deserialize)]
struct CohereEmbedMeta {
    #[serde(default)]
    billed_units: Option<CohereBilledUnits>,
}

#[derive(Debug, Deserialize)]
struct CohereBilledUnits {
    #[serde(default)]
    input_tokens: Option<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Provider;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = CohereProvider::with_api_key("test-key").unwrap();
        assert_eq!(Provider::name(&provider), "cohere");
        assert!(provider.supports_tools());
        assert!(!provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_default_model() {
        let provider = CohereProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.default_model(), Some("command-r"));
    }

    #[test]
    fn test_supported_models() {
        let provider = CohereProvider::with_api_key("test-key").unwrap();
        let models = provider.supported_models().unwrap();
        assert!(models.contains(&"command-r-plus"));
        assert!(models.contains(&"command-r"));
        assert!(models.contains(&"command"));
    }

    #[test]
    fn test_request_conversion() {
        let provider = CohereProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("command-r", vec![Message::user("Hello")])
            .with_system("You are helpful")
            .with_max_tokens(1024);

        let cohere_req = provider.convert_request(&request);

        assert_eq!(cohere_req.model, "command-r");
        assert_eq!(cohere_req.message, "Hello");
        assert_eq!(cohere_req.preamble, Some("You are helpful".to_string()));
        assert_eq!(cohere_req.max_tokens, Some(1024));
    }

    #[test]
    fn test_request_parameters() {
        let provider = CohereProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("command-r", vec![Message::user("Hello")])
            .with_max_tokens(500)
            .with_temperature(0.8);

        let cohere_req = provider.convert_request(&request);

        assert_eq!(cohere_req.max_tokens, Some(500));
        assert_eq!(cohere_req.temperature, Some(0.8));
    }

    #[test]
    fn test_chat_history_conversion() {
        let provider = CohereProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new(
            "command-r",
            vec![
                Message::user("Hi"),
                Message::assistant("Hello!"),
                Message::user("How are you?"),
            ],
        );

        let cohere_req = provider.convert_request(&request);

        // Last user message becomes the main message
        assert_eq!(cohere_req.message, "How are you?");

        // Previous messages become chat history
        let history = cohere_req.chat_history.unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, "USER");
        assert_eq!(history[0].message, "Hi");
        assert_eq!(history[1].role, "CHATBOT");
        assert_eq!(history[1].message, "Hello!");
    }

    #[test]
    fn test_response_parsing() {
        let provider = CohereProvider::with_api_key("test-key").unwrap();

        let cohere_response = CohereResponse {
            text: "Hello! How can I help?".to_string(),
            generation_id: Some("gen-123".to_string()),
            model: Some("command-r".to_string()),
            finish_reason: Some("COMPLETE".to_string()),
            tool_calls: None,
            meta: Some(CohereMeta {
                tokens: Some(CohereTokens {
                    input_tokens: 10,
                    output_tokens: 20,
                }),
            }),
        };

        let response = provider.convert_response(cohere_response);

        assert_eq!(response.id, "gen-123");
        assert_eq!(response.model, "command-r");
        assert_eq!(response.content.len(), 1);
        if let ContentBlock::Text { text } = &response.content[0] {
            assert_eq!(text, "Hello! How can I help?");
        } else {
            panic!("Expected Text content block");
        }
        assert!(matches!(response.stop_reason, StopReason::EndTurn));
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 20);
    }

    #[test]
    fn test_stop_reason_mapping() {
        let provider = CohereProvider::with_api_key("test-key").unwrap();

        // Test "COMPLETE" -> EndTurn
        let response1 = CohereResponse {
            text: "Done".to_string(),
            generation_id: None,
            model: None,
            finish_reason: Some("COMPLETE".to_string()),
            tool_calls: None,
            meta: None,
        };
        assert!(matches!(
            provider.convert_response(response1).stop_reason,
            StopReason::EndTurn
        ));

        // Test "MAX_TOKENS" -> MaxTokens
        let response2 = CohereResponse {
            text: "Truncated...".to_string(),
            generation_id: None,
            model: None,
            finish_reason: Some("MAX_TOKENS".to_string()),
            tool_calls: None,
            meta: None,
        };
        assert!(matches!(
            provider.convert_response(response2).stop_reason,
            StopReason::MaxTokens
        ));

        // Test "TOOL_CALL" -> ToolUse
        let response3 = CohereResponse {
            text: "".to_string(),
            generation_id: None,
            model: None,
            finish_reason: Some("TOOL_CALL".to_string()),
            tool_calls: None,
            meta: None,
        };
        assert!(matches!(
            provider.convert_response(response3).stop_reason,
            StopReason::ToolUse
        ));

        // Test "ERROR_TOXIC" -> ContentFilter
        let response4 = CohereResponse {
            text: "".to_string(),
            generation_id: None,
            model: None,
            finish_reason: Some("ERROR_TOXIC".to_string()),
            tool_calls: None,
            meta: None,
        };
        assert!(matches!(
            provider.convert_response(response4).stop_reason,
            StopReason::ContentFilter
        ));
    }

    #[test]
    fn test_tool_call_response() {
        let provider = CohereProvider::with_api_key("test-key").unwrap();

        let cohere_response = CohereResponse {
            text: "".to_string(),
            generation_id: Some("gen-123".to_string()),
            model: Some("command-r".to_string()),
            finish_reason: Some("TOOL_CALL".to_string()),
            tool_calls: Some(vec![CohereToolCall {
                name: "get_weather".to_string(),
                parameters: serde_json::json!({"location": "Paris"}),
            }]),
            meta: None,
        };

        let response = provider.convert_response(cohere_response);

        assert_eq!(response.content.len(), 1);
        assert!(matches!(response.stop_reason, StopReason::ToolUse));

        if let ContentBlock::ToolUse { name, input, .. } = &response.content[0] {
            assert_eq!(name, "get_weather");
            assert_eq!(input.get("location").unwrap().as_str().unwrap(), "Paris");
        } else {
            panic!("Expected ToolUse content block");
        }
    }

    #[test]
    fn test_json_schema_to_cohere_params() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city name"
                },
                "unit": {
                    "type": "string",
                    "description": "Temperature unit"
                }
            },
            "required": ["location"]
        });

        let params = convert_json_schema_to_cohere_params(&schema);

        assert_eq!(params.len(), 2);
        assert!(params.get("location").unwrap().required);
        assert!(!params.get("unit").unwrap().required);
        assert_eq!(params.get("location").unwrap().param_type, "string");
        assert_eq!(
            params.get("location").unwrap().description,
            Some("The city name".to_string())
        );
    }

    #[test]
    fn test_api_url() {
        let provider = CohereProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.api_url(), COHERE_API_URL);

        let config = ProviderConfig::new("test-key").with_base_url("https://custom.cohere.ai/v1");
        let provider = CohereProvider::new(config).unwrap();
        assert_eq!(provider.api_url(), "https://custom.cohere.ai/v1");
    }

    #[test]
    fn test_embedding_provider() {
        use crate::embedding::EmbeddingProvider;

        let provider = CohereProvider::with_api_key("test-key").unwrap();

        assert_eq!(EmbeddingProvider::name(&provider), "cohere");
        assert_eq!(
            provider.default_embedding_model(),
            Some("embed-english-v3.0")
        );
        assert_eq!(provider.max_batch_size(), 96);
        assert_eq!(
            provider.embedding_dimensions("embed-english-v3.0"),
            Some(1024)
        );
        assert_eq!(
            provider.embedding_dimensions("embed-english-light-v3.0"),
            Some(384)
        );
    }
}
