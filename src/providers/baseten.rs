#![allow(dead_code)]
//! Baseten API provider implementation.
//!
//! This module provides access to Baseten's model hosting platform.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::BasetenProvider;
//!
//! // From environment variable
//! let provider = BasetenProvider::from_env()?;
//!
//! // With explicit API key and model ID
//! let provider = BasetenProvider::with_model("model-id", "your-api-key")?;
//! ```
//!
//! # Environment Variables
//!
//! - `BASETEN_API_KEY` - Your Baseten API key
//! - `BASETEN_MODEL_ID` - Optional default model ID

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

const BASETEN_API_URL: &str = "https://model-{model_id}.api.baseten.co/production/predict";

/// Baseten API provider.
///
/// Provides access to models deployed on Baseten.
pub struct BasetenProvider {
    config: ProviderConfig,
    client: Client,
    /// Default model ID for predictions
    default_model_id: Option<String>,
}

impl BasetenProvider {
    /// Create provider from environment variables.
    ///
    /// Reads: `BASETEN_API_KEY` and optionally `BASETEN_MODEL_ID`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("BASETEN_API_KEY").ok();
        let model_id = std::env::var("BASETEN_MODEL_ID").ok();

        let config = ProviderConfig {
            api_key,
            ..Default::default()
        };

        let mut provider = Self::new(config)?;
        provider.default_model_id = model_id;
        Ok(provider)
    }

    /// Create provider with explicit API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    /// Create provider with model ID and API key.
    pub fn with_model(model_id: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);

        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Api-Key {}", key)
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

        Ok(Self {
            config,
            client,
            default_model_id: Some(model_id.into()),
        })
    }

    /// Create provider with custom config.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Api-Key {}", key)
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

        Ok(Self {
            config,
            client,
            default_model_id: None,
        })
    }

    /// Get the API URL for a given model.
    fn get_api_url(&self, model: &str) -> String {
        // Model can be a full URL, model ID, or we use the default
        if model.starts_with("http") {
            model.to_string()
        } else {
            let model_id = if model.is_empty() {
                self.default_model_id.as_deref().unwrap_or("")
            } else {
                model
            };
            BASETEN_API_URL.replace("{model_id}", model_id)
        }
    }

    /// Build the prompt from messages.
    fn build_prompt(&self, request: &CompletionRequest) -> String {
        let mut prompt = String::new();

        // Add system prompt if present
        if let Some(ref system) = request.system {
            prompt.push_str(&format!("[INST] <<SYS>>\n{}\n<</SYS>>\n\n", system));
        } else {
            prompt.push_str("[INST] ");
        }

        // Convert messages
        for (i, msg) in request.messages.iter().enumerate() {
            let content = msg
                .content
                .iter()
                .filter_map(|block| {
                    if let ContentBlock::Text { text } = block {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");

            match msg.role {
                Role::User => {
                    if i > 0 && request.system.is_some() {
                        prompt.push_str("[INST] ");
                    }
                    prompt.push_str(&content);
                    prompt.push_str(" [/INST]");
                }
                Role::Assistant => {
                    prompt.push(' ');
                    prompt.push_str(&content);
                    prompt.push_str(" </s><s>");
                }
                Role::System => {
                    prompt.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", content));
                }
            }
        }

        prompt
    }

    /// Convert unified request to Baseten format.
    fn convert_request(&self, request: &CompletionRequest) -> BasetenRequest {
        let prompt = self.build_prompt(request);

        BasetenRequest {
            prompt,
            max_new_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: Some(request.stream),
        }
    }

    /// Convert Baseten response to unified format.
    fn convert_response(&self, response: BasetenResponse, model: &str) -> CompletionResponse {
        let mut content = Vec::new();

        // Handle different response formats
        let text = match response.output {
            Some(BasetenOutput::String(s)) => s,
            Some(BasetenOutput::Object { generated_text, .. }) => {
                generated_text.unwrap_or_default()
            }
            Some(BasetenOutput::Array(arr)) => arr.join(""),
            None => response.data.unwrap_or_default(),
        };

        if !text.is_empty() {
            content.push(ContentBlock::Text { text });
        }

        CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: model.to_string(),
            content,
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }
    }

    /// Handle error responses from Baseten API.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        if let Ok(error_resp) = serde_json::from_str::<BasetenErrorResponse>(body) {
            let message = error_resp
                .error
                .unwrap_or_else(|| error_resp.message.unwrap_or_default());
            match status.as_u16() {
                401 => Error::auth(message),
                403 => Error::auth(message),
                404 => Error::ModelNotFound(message),
                422 => Error::invalid_request(message),
                429 => Error::rate_limited(message, None),
                500..=599 => Error::server(status.as_u16(), message),
                _ => Error::other(message),
            }
        } else {
            Error::server(status.as_u16(), format!("HTTP {}: {}", status, body))
        }
    }
}

#[async_trait]
impl Provider for BasetenProvider {
    fn name(&self) -> &str {
        "baseten"
    }

    fn default_model(&self) -> Option<&str> {
        self.default_model_id.as_deref()
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let url = self.get_api_url(&request.model);
        let baseten_request = self.convert_request(&request);

        let response = self.client.post(&url).json(&baseten_request).send().await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let baseten_response: BasetenResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

        Ok(self.convert_response(baseten_response, &request.model))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let url = self.get_api_url(&request.model);
        let mut baseten_request = self.convert_request(&request);
        baseten_request.stream = Some(true);

        let response = self.client.post(&url).json(&baseten_request).send().await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await?;
            return Err(self.handle_error_response(status, &body));
        }

        let stream = async_stream::try_stream! {
            use futures::StreamExt;

            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut chunk_index = 0usize;

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| Error::stream(format!("Stream error: {}", e)))?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete lines
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    // Handle SSE data format
                    let data = if let Some(stripped) = line.strip_prefix("data: ") {
                        stripped
                    } else {
                        &line
                    };

                    if data == "[DONE]" {
                        yield StreamChunk {
                            event_type: StreamEventType::MessageStop,
                            index: None,
                            delta: None,
                            stop_reason: None,
                            usage: None,
                        };
                        break;
                    }

                    // Try to parse as JSON
                    if let Ok(chunk_resp) = serde_json::from_str::<BasetenStreamChunk>(data) {
                        if let Some(token) = chunk_resp.token {
                            if !token.is_empty() {
                                yield StreamChunk {
                                    event_type: StreamEventType::ContentBlockDelta,
                                    index: Some(chunk_index),
                                    delta: Some(ContentDelta::TextDelta { text: token }),
                                    stop_reason: None,
                                    usage: None,
                                };
                                chunk_index += 1;
                            }
                        }
                    } else if !data.is_empty() && !data.starts_with('{') {
                        // Plain text token
                        yield StreamChunk {
                            event_type: StreamEventType::ContentBlockDelta,
                            index: Some(chunk_index),
                            delta: Some(ContentDelta::TextDelta { text: data.to_string() }),
                            stop_reason: None,
                            usage: None,
                        };
                        chunk_index += 1;
                    }
                }
            }
        };

        Ok(Box::pin(stream))
    }

    fn supports_tools(&self) -> bool {
        false
    }

    fn supports_vision(&self) -> bool {
        false
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

// ============ Request/Response Types ============

#[derive(Debug, Serialize)]
struct BasetenRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct BasetenResponse {
    output: Option<BasetenOutput>,
    data: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum BasetenOutput {
    String(String),
    Object { generated_text: Option<String> },
    Array(Vec<String>),
}

#[derive(Debug, Deserialize)]
struct BasetenStreamChunk {
    token: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BasetenErrorResponse {
    error: Option<String>,
    message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = BasetenProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "baseten");
        assert!(!provider.supports_tools());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_with_model() {
        let provider = BasetenProvider::with_model("model-123", "test-key").unwrap();
        assert_eq!(provider.default_model(), Some("model-123"));
    }

    #[test]
    fn test_api_url() {
        let provider = BasetenProvider::with_model("abc123", "test-key").unwrap();

        let url = provider.get_api_url("xyz789");
        assert_eq!(
            url,
            "https://model-xyz789.api.baseten.co/production/predict"
        );

        let url = provider.get_api_url("");
        assert_eq!(
            url,
            "https://model-abc123.api.baseten.co/production/predict"
        );

        let url = provider.get_api_url("https://custom.endpoint.com/predict");
        assert_eq!(url, "https://custom.endpoint.com/predict");
    }

    #[test]
    fn test_prompt_building() {
        let provider = BasetenProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("model-123", vec![Message::user("Hello")])
            .with_system("You are helpful");

        let baseten_req = provider.convert_request(&request);

        assert!(baseten_req.prompt.contains("You are helpful"));
        assert!(baseten_req.prompt.contains("Hello"));
        assert!(baseten_req.prompt.contains("[INST]"));
    }
}
