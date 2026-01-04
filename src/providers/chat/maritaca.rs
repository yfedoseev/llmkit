//! Maritaca AI (Sabiá) API provider implementation.
//!
//! This module provides access to Maritaca AI's Sabiá foundation models.
//! Sabiá models are optimized for Portuguese language understanding and generation.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::MaritacaProvider;
//!
//! // From environment variable
//! let provider = MaritacaProvider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = MaritacaProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Features
//!
//! - Sabiá 3 (latest, most capable)
//! - Sabiá 2 Small (faster, lightweight)
//! - Streaming support
//! - Portuguese language optimization
//!
//! # Environment Variables
//!
//! - `MARITALK_API_KEY` - Your Maritaca API key
//!
//! # Models
//!
//! - `sabia-3` - Latest and most capable model
//! - `sabia-2-small` - Smaller, faster variant

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

const MARITACA_API_URL: &str = "https://chat.maritaca.ai/api/chat/inference";

/// Maritaca AI (Sabiá) API provider.
///
/// Provides access to Maritaca's Sabiá models optimized for Portuguese.
pub struct MaritacaProvider {
    config: ProviderConfig,
    client: Client,
}

impl MaritacaProvider {
    /// Create a new Maritaca provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        // Maritaca uses custom "Key" auth format, not Bearer
        if let Some(ref key) = config.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Key {}", key)
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

    /// Create a new Maritaca provider from environment variable.
    ///
    /// Reads the API key from `MARITALK_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("MARITALK_API_KEY")
            .map_err(|_| Error::config("MARITALK_API_KEY environment variable not set"))?;

        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    /// Create a new Maritaca provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn api_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(MARITACA_API_URL)
    }

    /// Convert our unified request to Maritaca's format.
    fn convert_request(&self, request: &CompletionRequest, stream: bool) -> MaritacaRequest {
        // Convert messages
        let mut messages: Vec<MaritacaMessage> = Vec::new();

        // Add system message if present (as a user message with system prefix)
        if let Some(ref system) = request.system {
            messages.push(MaritacaMessage {
                role: "user".to_string(),
                content: format!("[System]: {}", system),
            });
            messages.push(MaritacaMessage {
                role: "assistant".to_string(),
                content: "Understood. I will follow these instructions.".to_string(),
            });
        }

        // Convert conversation messages
        for msg in &request.messages {
            let role = match msg.role {
                Role::System => {
                    // Handle inline system messages
                    messages.push(MaritacaMessage {
                        role: "user".to_string(),
                        content: format!("[System]: {}", msg.text_content()),
                    });
                    messages.push(MaritacaMessage {
                        role: "assistant".to_string(),
                        content: "Understood.".to_string(),
                    });
                    continue;
                }
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            messages.push(MaritacaMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        // Default model
        let model = if request.model.is_empty() || request.model == "default" {
            "sabia-3".to_string()
        } else {
            request.model.clone()
        };

        MaritacaRequest {
            model,
            messages,
            do_sample: Some(true),
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stopping_tokens: request.stop_sequences.clone(),
            stream,
        }
    }

    /// Parse Maritaca response into our unified format.
    fn parse_response(&self, response: MaritacaResponse, model: String) -> CompletionResponse {
        CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model,
            content: vec![ContentBlock::Text {
                text: response.answer,
            }],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 0, // Maritaca doesn't return token counts
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        }
    }
}

#[async_trait]
impl Provider for MaritacaProvider {
    fn name(&self) -> &str {
        "maritaca"
    }

    fn default_model(&self) -> Option<&str> {
        Some("sabia-3")
    }

    fn supported_models(&self) -> Result<Vec<&'static str>> {
        Ok(vec![
            "sabia-3",       // Latest and most capable model
            "sabia-2-small", // Smaller, faster variant
        ])
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = request.model.clone();
        let maritaca_request = self.convert_request(&request, false);

        let response = self
            .client
            .post(self.api_url())
            .json(&maritaca_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Maritaca API error {}: {}", status, error_text),
            ));
        }

        let maritaca_response: MaritacaResponse = response.json().await?;

        let model_used = if model.is_empty() || model == "default" {
            "sabia-3".to_string()
        } else {
            model
        };

        Ok(self.parse_response(maritaca_response, model_used))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let _model = request.model.clone();
        let maritaca_request = self.convert_request(&request, true);

        let response = self
            .client
            .post(self.api_url())
            .json(&maritaca_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Maritaca API error {}: {}", status, error_text),
            ));
        }

        let stream = async_stream::try_stream! {
            use tokio_stream::StreamExt;

            yield StreamChunk {
                event_type: StreamEventType::ContentBlockStart,
                index: Some(0),
                delta: None,
                stop_reason: None,
                usage: None,
            };

            let mut reader = response.bytes_stream();

            while let Some(chunk_result) = reader.next().await {
                let chunk = chunk_result.map_err(|e| Error::other(e.to_string()))?;
                let text = String::from_utf8_lossy(&chunk);

                // Maritaca streams with "data: " prefix like SSE
                for line in text.lines() {
                    let line = line.trim();
                    if line.is_empty() || line == "data: [DONE]" {
                        continue;
                    }

                    let data = line.strip_prefix("data: ").unwrap_or(line);

                    if let Ok(stream_chunk) = serde_json::from_str::<MaritacaStreamChunk>(data) {
                        if let Some(text) = stream_chunk.text {
                            if !text.is_empty() {
                                yield StreamChunk {
                                    event_type: StreamEventType::ContentBlockDelta,
                                    index: Some(0),
                                    delta: Some(ContentDelta::Text { text }),
                                    stop_reason: None,
                                    usage: None,
                                };
                            }
                        }
                    }
                }
            }

            yield StreamChunk {
                event_type: StreamEventType::MessageStop,
                index: None,
                delta: None,
                stop_reason: Some(StopReason::EndTurn),
                usage: None,
            };
        };

        Ok(Box::pin(stream))
    }
}

// ==================== Maritaca API Types ====================

#[derive(Debug, Serialize)]
struct MaritacaRequest {
    model: String,
    messages: Vec<MaritacaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    do_sample: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stopping_tokens: Option<Vec<String>>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct MaritacaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct MaritacaResponse {
    answer: String,
}

#[derive(Debug, Deserialize)]
struct MaritacaStreamChunk {
    text: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = MaritacaProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.name(), "maritaca");
    }

    #[test]
    fn test_provider_with_api_key() {
        let provider = MaritacaProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "maritaca");
    }

    #[test]
    fn test_api_url() {
        let provider = MaritacaProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.api_url(), MARITACA_API_URL);
    }

    #[test]
    fn test_api_url_custom_base() {
        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.maritaca.ai".to_string());
        let provider = MaritacaProvider::new(config).unwrap();
        assert_eq!(provider.api_url(), "https://custom.maritaca.ai");
    }

    #[test]
    fn test_convert_request() {
        let provider = MaritacaProvider::new(ProviderConfig::new("test-key")).unwrap();

        let request = CompletionRequest::new("sabia-3", vec![Message::user("Olá")])
            .with_max_tokens(1024)
            .with_temperature(0.7);

        let maritaca_req = provider.convert_request(&request, false);

        assert_eq!(maritaca_req.model, "sabia-3");
        assert_eq!(maritaca_req.messages.len(), 1);
        assert_eq!(maritaca_req.messages[0].role, "user");
        assert_eq!(maritaca_req.messages[0].content, "Olá");
        assert_eq!(maritaca_req.max_tokens, Some(1024));
        assert_eq!(maritaca_req.temperature, Some(0.7));
        assert!(!maritaca_req.stream);
    }

    #[test]
    fn test_convert_request_with_system_prefix() {
        let provider = MaritacaProvider::new(ProviderConfig::new("test-key")).unwrap();

        let request = CompletionRequest::new("sabia-3", vec![Message::user("Olá")])
            .with_system("Você é prestativo");

        let maritaca_req = provider.convert_request(&request, false);

        // System message should be converted to user message with [System]: prefix
        assert!(maritaca_req.messages.len() >= 2);
        assert_eq!(maritaca_req.messages[0].role, "user");
        assert!(maritaca_req.messages[0].content.contains("[System]:"));
        assert!(maritaca_req.messages[0]
            .content
            .contains("Você é prestativo"));
        // Followed by assistant acknowledgment
        assert_eq!(maritaca_req.messages[1].role, "assistant");
    }

    #[test]
    fn test_convert_request_default_model() {
        let provider = MaritacaProvider::new(ProviderConfig::new("test-key")).unwrap();

        let request = CompletionRequest::new("", vec![Message::user("Olá")]);

        let maritaca_req = provider.convert_request(&request, false);
        assert_eq!(maritaca_req.model, "sabia-3");
    }

    #[test]
    fn test_parse_response() {
        let provider = MaritacaProvider::new(ProviderConfig::new("test-key")).unwrap();

        let response = MaritacaResponse {
            answer: "Olá! Estou bem, obrigado.".to_string(),
        };

        let result = provider.parse_response(response, "sabia-3".to_string());

        assert_eq!(result.model, "sabia-3");
        assert_eq!(result.content.len(), 1);
        if let ContentBlock::Text { text } = &result.content[0] {
            assert_eq!(text, "Olá! Estou bem, obrigado.");
        } else {
            panic!("Expected text content block");
        }
        assert!(matches!(result.stop_reason, StopReason::EndTurn));
    }

    #[test]
    fn test_request_serialization() {
        let request = MaritacaRequest {
            model: "sabia-3".to_string(),
            messages: vec![MaritacaMessage {
                role: "user".to_string(),
                content: "Olá, como vai?".to_string(),
            }],
            do_sample: Some(true),
            max_tokens: Some(200),
            temperature: Some(0.7),
            top_p: None,
            stopping_tokens: None,
            stream: false,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("sabia-3"));
        assert!(json.contains("do_sample"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{"answer": "Olá! Estou bem, obrigado."}"#;
        let response: MaritacaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.answer, "Olá! Estou bem, obrigado.");
    }

    #[test]
    fn test_stream_chunk_deserialization() {
        let json = r#"{"text": "Hello"}"#;
        let chunk: MaritacaStreamChunk = serde_json::from_str(json).unwrap();
        assert_eq!(chunk.text, Some("Hello".to_string()));
    }
}
