//! Naver CLOVA Studio (HyperCLOVA X) API provider implementation.
//!
//! This module provides access to Naver's HyperCLOVA X foundation models via CLOVA Studio.
//! HyperCLOVA X offers high-quality Korean language models with multimodal and reasoning capabilities.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::ClovaProvider;
//!
//! // From environment variable
//! let provider = ClovaProvider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = ClovaProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Features
//!
//! - HCX-005 (multimodal, vision)
//! - HCX-007 (reasoning, deep thinking)
//! - HCX-DASH-002 (lightweight, fast)
//! - Streaming support (SSE)
//! - Tool/function calling
//! - AI content filtering
//!
//! # Environment Variables
//!
//! - `CLOVASTUDIO_API_KEY` - Your CLOVA Studio API key
//! - `NCP_CLOVASTUDIO_API_KEY` - Alternative env var name
//!
//! # Models
//!
//! - `HCX-005` - Multimodal model with vision capabilities
//! - `HCX-007` - Reasoning model for complex problems
//! - `HCX-DASH-002` - Lightweight model for fast responses

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

const CLOVA_API_URL: &str = "https://clovastudio.stream.ntruss.com/v3/chat-completions";

/// Naver CLOVA Studio (HyperCLOVA X) API provider.
///
/// Provides access to Naver's HyperCLOVA X models including HCX-005, HCX-007, and HCX-DASH-002.
pub struct ClovaProvider {
    config: ProviderConfig,
    client: Client,
}

impl ClovaProvider {
    /// Create a new CLOVA provider with the given configuration.
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

        headers.insert(reqwest::header::ACCEPT, "application/json".parse().unwrap());

        let client = Client::builder()
            .timeout(config.timeout)
            .default_headers(headers)
            .build()?;

        Ok(Self { config, client })
    }

    /// Create a new CLOVA provider from environment variable.
    ///
    /// Reads the API key from `CLOVASTUDIO_API_KEY` or `NCP_CLOVASTUDIO_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("CLOVASTUDIO_API_KEY")
            .or_else(|_| std::env::var("NCP_CLOVASTUDIO_API_KEY"))
            .map_err(|_| {
                Error::config(
                    "CLOVASTUDIO_API_KEY or NCP_CLOVASTUDIO_API_KEY environment variable not set",
                )
            })?;

        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    /// Create a new CLOVA provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    /// Build the API URL for a specific model.
    fn api_url(&self, model: &str) -> String {
        let base = self.config.base_url.as_deref().unwrap_or(CLOVA_API_URL);
        format!("{}/{}", base, model)
    }

    /// Convert our unified request to CLOVA's format.
    fn convert_request(&self, request: &CompletionRequest) -> ClovaRequest {
        // Convert messages
        let mut messages: Vec<ClovaMessage> = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(ClovaMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        // Convert conversation messages
        for msg in &request.messages {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            messages.push(ClovaMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        ClovaRequest {
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            top_p: request.top_p,
            top_k: None,
            repeat_penalty: None,
            stop_before: request.stop_sequences.clone(),
            include_ai_filters: Some(true),
        }
    }

    /// Parse CLOVA response into our unified format.
    fn parse_response(&self, response: ClovaResponse, model: String) -> CompletionResponse {
        let text = response
            .message
            .as_ref()
            .map(|m| m.content.clone())
            .unwrap_or_default();

        let stop_reason = response
            .stop_reason
            .as_ref()
            .map(|r| match r.as_str() {
                "stop_before" | "end_token" => StopReason::EndTurn,
                "length" => StopReason::MaxTokens,
                _ => StopReason::EndTurn,
            })
            .unwrap_or(StopReason::EndTurn);

        CompletionResponse {
            id: response
                .id
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            model,
            content: vec![ContentBlock::Text { text }],
            stop_reason,
            usage: response.usage.map_or_else(
                || Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
                |u| Usage {
                    input_tokens: u.prompt_tokens.unwrap_or(0),
                    output_tokens: u.completion_tokens.unwrap_or(0),
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            ),
        }
    }
}

#[async_trait]
impl Provider for ClovaProvider {
    fn name(&self) -> &str {
        "clova"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        // Default model is HCX-DASH-002 (lightweight)
        let model = if request.model.is_empty() || request.model == "default" {
            "HCX-DASH-002".to_string()
        } else {
            request.model.clone()
        };

        let clova_request = self.convert_request(&request);

        // Generate unique request ID
        let request_id = uuid::Uuid::new_v4().to_string();

        let response = self
            .client
            .post(self.api_url(&model))
            .header("X-NCP-CLOVASTUDIO-REQUEST-ID", &request_id)
            .json(&clova_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("CLOVA API error {}: {}", status, error_text),
            ));
        }

        let clova_response: ClovaResponse = response.json().await?;
        Ok(self.parse_response(clova_response, model))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // Default model
        let model = if request.model.is_empty() || request.model == "default" {
            "HCX-DASH-002".to_string()
        } else {
            request.model.clone()
        };

        let clova_request = self.convert_request(&request);

        // Generate unique request ID
        let request_id = uuid::Uuid::new_v4().to_string();

        let response = self
            .client
            .post(self.api_url(&model))
            .header("X-NCP-CLOVASTUDIO-REQUEST-ID", &request_id)
            .header("Accept", "text/event-stream")
            .json(&clova_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("CLOVA API error {}: {}", status, error_text),
            ));
        }

        let stream = async_stream::try_stream! {
            use tokio_stream::StreamExt;
            use eventsource_stream::Eventsource;

            yield StreamChunk {
                event_type: StreamEventType::ContentBlockStart,
                index: Some(0),
                delta: None,
                stop_reason: None,
                usage: None,
            };

            let mut event_stream = response.bytes_stream().eventsource();

            while let Some(event) = event_stream.next().await {
                let event = event.map_err(|e| Error::other(e.to_string()))?;

                if event.data.is_empty() || event.data == "[DONE]" {
                    continue;
                }

                if let Ok(chunk) = serde_json::from_str::<ClovaStreamChunk>(&event.data) {
                    if let Some(ref message) = chunk.message {
                        if !message.content.is_empty() {
                            yield StreamChunk {
                                event_type: StreamEventType::ContentBlockDelta,
                                index: Some(0),
                                delta: Some(ContentDelta::TextDelta { text: message.content.clone() }),
                                stop_reason: None,
                                usage: None,
                            };
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

// ==================== CLOVA API Types ====================

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ClovaRequest {
    messages: Vec<ClovaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repeat_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_before: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include_ai_filters: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ClovaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClovaResponse {
    #[serde(default)]
    id: Option<String>,
    message: Option<ClovaMessage>,
    #[serde(default)]
    stop_reason: Option<String>,
    usage: Option<ClovaUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ClovaUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ClovaStreamChunk {
    message: Option<ClovaMessage>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = ClovaProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.name(), "clova");
    }

    #[test]
    fn test_provider_with_api_key() {
        let provider = ClovaProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "clova");
    }

    #[test]
    fn test_api_url_construction() {
        let provider = ClovaProvider::new(ProviderConfig::new("test-key")).unwrap();
        let url = provider.api_url("HCX-005");
        assert_eq!(
            url,
            "https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-005"
        );
    }

    #[test]
    fn test_api_url_custom_base() {
        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.clova.ai".to_string());
        let provider = ClovaProvider::new(config).unwrap();
        let url = provider.api_url("HCX-005");
        assert_eq!(url, "https://custom.clova.ai/HCX-005");
    }

    #[test]
    fn test_convert_request() {
        let provider = ClovaProvider::new(ProviderConfig::new("test-key")).unwrap();

        let request = CompletionRequest::new("HCX-005", vec![Message::user("안녕하세요")])
            .with_system("당신은 도움이 되는 조수입니다")
            .with_max_tokens(1024)
            .with_temperature(0.7);

        let clova_req = provider.convert_request(&request);

        assert_eq!(clova_req.messages.len(), 2); // system + user
        assert_eq!(clova_req.messages[0].role, "system");
        assert_eq!(
            clova_req.messages[0].content,
            "당신은 도움이 되는 조수입니다"
        );
        assert_eq!(clova_req.messages[1].role, "user");
        assert_eq!(clova_req.messages[1].content, "안녕하세요");
        assert_eq!(clova_req.max_tokens, Some(1024));
        assert_eq!(clova_req.temperature, Some(0.7));
        assert_eq!(clova_req.include_ai_filters, Some(true));
    }

    #[test]
    fn test_parse_response() {
        let provider = ClovaProvider::new(ProviderConfig::new("test-key")).unwrap();

        let response = ClovaResponse {
            id: Some("resp-123".to_string()),
            message: Some(ClovaMessage {
                role: "assistant".to_string(),
                content: "안녕하세요! 도움이 필요하시면 말씀해주세요.".to_string(),
            }),
            stop_reason: Some("stop_before".to_string()),
            usage: Some(ClovaUsage {
                prompt_tokens: Some(10),
                completion_tokens: Some(15),
            }),
        };

        let result = provider.parse_response(response, "HCX-005".to_string());

        assert_eq!(result.id, "resp-123");
        assert_eq!(result.model, "HCX-005");
        assert_eq!(result.content.len(), 1);
        if let ContentBlock::Text { text } = &result.content[0] {
            assert_eq!(text, "안녕하세요! 도움이 필요하시면 말씀해주세요.");
        } else {
            panic!("Expected text content block");
        }
        assert!(matches!(result.stop_reason, StopReason::EndTurn));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 15);
    }

    #[test]
    fn test_stop_reason_mapping() {
        let provider = ClovaProvider::new(ProviderConfig::new("test-key")).unwrap();

        // Test "stop_before" -> EndTurn
        let response1 = ClovaResponse {
            id: None,
            message: Some(ClovaMessage {
                role: "assistant".to_string(),
                content: "Done".to_string(),
            }),
            stop_reason: Some("stop_before".to_string()),
            usage: None,
        };
        assert!(matches!(
            provider
                .parse_response(response1, "model".to_string())
                .stop_reason,
            StopReason::EndTurn
        ));

        // Test "end_token" -> EndTurn
        let response2 = ClovaResponse {
            id: None,
            message: Some(ClovaMessage {
                role: "assistant".to_string(),
                content: "Done".to_string(),
            }),
            stop_reason: Some("end_token".to_string()),
            usage: None,
        };
        assert!(matches!(
            provider
                .parse_response(response2, "model".to_string())
                .stop_reason,
            StopReason::EndTurn
        ));

        // Test "length" -> MaxTokens
        let response3 = ClovaResponse {
            id: None,
            message: Some(ClovaMessage {
                role: "assistant".to_string(),
                content: "Truncated".to_string(),
            }),
            stop_reason: Some("length".to_string()),
            usage: None,
        };
        assert!(matches!(
            provider
                .parse_response(response3, "model".to_string())
                .stop_reason,
            StopReason::MaxTokens
        ));
    }

    #[test]
    fn test_request_serialization() {
        let request = ClovaRequest {
            messages: vec![ClovaMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            temperature: Some(0.7),
            max_tokens: Some(256),
            top_p: None,
            top_k: None,
            repeat_penalty: None,
            stop_before: None,
            include_ai_filters: Some(true),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("maxTokens")); // camelCase
        assert!(json.contains("includeAiFilters"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "id": "test-id",
            "message": {"role": "assistant", "content": "Hello!"},
            "stopReason": "stop_before",
            "usage": {"promptTokens": 10, "completionTokens": 5}
        }"#;

        let response: ClovaResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.id, Some("test-id".to_string()));
        assert!(response.message.is_some());
        assert_eq!(response.message.as_ref().unwrap().content, "Hello!");
    }

    #[test]
    fn test_stream_chunk_deserialization() {
        let json = r#"{"message": {"role": "assistant", "content": "Hello"}}"#;
        let chunk: ClovaStreamChunk = serde_json::from_str(json).unwrap();
        assert!(chunk.message.is_some());
        assert_eq!(chunk.message.as_ref().unwrap().content, "Hello");
    }
}
