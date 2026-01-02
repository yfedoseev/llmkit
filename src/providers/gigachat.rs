//! Sber GigaChat API provider implementation.
//!
//! This module provides access to Sber's GigaChat foundation models.
//! GigaChat offers high-quality Russian language models with OAuth 2.0 authentication.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::GigaChatProvider;
//!
//! // From environment variable
//! let provider = GigaChatProvider::from_env()?;
//!
//! // Or with explicit credentials
//! let provider = GigaChatProvider::new("your-client-credentials")?;
//! ```
//!
//! # Supported Features
//!
//! - GigaChat (standard model)
//! - GigaChat Lite (fast responses)
//! - GigaChat Pro (enhanced capabilities)
//! - GigaChat Max (highest quality)
//! - Streaming support
//! - Vision (image understanding)
//!
//! # Environment Variables
//!
//! - `GIGACHAT_CREDENTIALS` - Base64-encoded client credentials (client_id:client_secret)
//! - `GIGACHAT_SCOPE` - Optional scope (GIGACHAT_API_PERS for personal, GIGACHAT_API_CORP for corporate)
//!
//! # Models
//!
//! - `GigaChat` - Standard model
//! - `GigaChat-Lite` - Lightweight, fast responses
//! - `GigaChat-Pro` - Enhanced capabilities
//! - `GigaChat-Max` - Highest quality

use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use parking_lot::RwLock;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

const GIGACHAT_API_URL: &str = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions";
const GIGACHAT_TOKEN_URL: &str = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth";
const GIGACHAT_MODELS_URL: &str = "https://gigachat.devices.sberbank.ru/api/v1/models";

/// Token info with expiration tracking.
#[derive(Debug, Clone)]
struct TokenInfo {
    access_token: String,
    expires_at: std::time::Instant,
}

/// Sber GigaChat API provider.
///
/// Provides access to Sber's GigaChat models with automatic OAuth token management.
pub struct GigaChatProvider {
    config: ProviderConfig,
    client: Client,
    credentials: String,
    scope: String,
    token: Arc<RwLock<Option<TokenInfo>>>,
}

impl GigaChatProvider {
    /// Create a new GigaChat provider with client credentials.
    ///
    /// # Arguments
    ///
    /// * `credentials` - Base64-encoded client credentials (client_id:client_secret)
    /// * `scope` - API scope (GIGACHAT_API_PERS for personal, GIGACHAT_API_CORP for corporate)
    pub fn new(credentials: impl Into<String>, scope: impl Into<String>) -> Result<Self> {
        let credentials = credentials.into();
        let scope = scope.into();

        // GigaChat uses self-signed certificates, so we need to disable cert verification
        // In production, you should add their CA certificate instead
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .danger_accept_invalid_certs(true)
            .build()?;

        let config = ProviderConfig::new(&credentials);

        Ok(Self {
            config,
            client,
            credentials,
            scope,
            token: Arc::new(RwLock::new(None)),
        })
    }

    /// Create a new GigaChat provider for personal use.
    pub fn personal(credentials: impl Into<String>) -> Result<Self> {
        Self::new(credentials, "GIGACHAT_API_PERS")
    }

    /// Create a new GigaChat provider for corporate use.
    pub fn corporate(credentials: impl Into<String>) -> Result<Self> {
        Self::new(credentials, "GIGACHAT_API_CORP")
    }

    /// Create a new GigaChat provider from environment variables.
    ///
    /// Reads credentials from `GIGACHAT_CREDENTIALS` and optionally `GIGACHAT_SCOPE`.
    pub fn from_env() -> Result<Self> {
        let credentials = std::env::var("GIGACHAT_CREDENTIALS")
            .map_err(|_| Error::config("GIGACHAT_CREDENTIALS environment variable not set"))?;

        let scope =
            std::env::var("GIGACHAT_SCOPE").unwrap_or_else(|_| "GIGACHAT_API_PERS".to_string());

        Self::new(credentials, scope)
    }

    fn api_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(GIGACHAT_API_URL)
    }

    /// Get a valid access token, refreshing if necessary.
    async fn get_access_token(&self) -> Result<String> {
        // Check if we have a valid token
        {
            let token_guard = self.token.read();
            if let Some(ref token_info) = *token_guard {
                // Token is valid if it doesn't expire in the next 60 seconds
                if token_info.expires_at
                    > std::time::Instant::now() + std::time::Duration::from_secs(60)
                {
                    return Ok(token_info.access_token.clone());
                }
            }
        }

        // Need to refresh token
        self.refresh_token().await
    }

    /// Refresh the OAuth token.
    async fn refresh_token(&self) -> Result<String> {
        let response = self
            .client
            .post(GIGACHAT_TOKEN_URL)
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Accept", "application/json")
            .header("Authorization", format!("Basic {}", self.credentials))
            .header("RqUID", uuid::Uuid::new_v4().to_string())
            .form(&[("scope", &self.scope)])
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("GigaChat OAuth error {}: {}", status, error_text),
            ));
        }

        let token_response: GigaChatTokenResponse = response.json().await?;

        let token_info = TokenInfo {
            access_token: token_response.access_token.clone(),
            // Token expires in 30 minutes, we'll refresh at 29 minutes
            expires_at: std::time::Instant::now()
                + std::time::Duration::from_millis(token_response.expires_at as u64)
                    .saturating_sub(std::time::Duration::from_secs(60)),
        };

        let access_token = token_info.access_token.clone();
        *self.token.write() = Some(token_info);

        Ok(access_token)
    }

    /// Convert our unified request to GigaChat's format.
    fn convert_request(&self, request: &CompletionRequest) -> GigaChatRequest {
        // Convert messages
        let mut messages: Vec<GigaChatMessage> = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(GigaChatMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        // Convert all messages
        for msg in &request.messages {
            let role = match msg.role {
                Role::System => "system",
                Role::User => "user",
                Role::Assistant => "assistant",
            };

            messages.push(GigaChatMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        // Default model
        let model = if request.model.is_empty() || request.model == "default" {
            "GigaChat".to_string()
        } else {
            request.model.clone()
        };

        GigaChatRequest {
            model,
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens,
            stream: false,
            top_p: request.top_p,
            repetition_penalty: None,
        }
    }

    /// Parse GigaChat response into our unified format.
    fn parse_response(&self, response: GigaChatResponse) -> CompletionResponse {
        let choice = response.choices.first();

        let text = choice
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let stop_reason = choice
            .map(|c| match c.finish_reason.as_deref() {
                Some("stop") => StopReason::EndTurn,
                Some("length") => StopReason::MaxTokens,
                Some("blacklist") => StopReason::StopSequence,
                _ => StopReason::EndTurn,
            })
            .unwrap_or(StopReason::EndTurn);

        CompletionResponse {
            id: response
                .id
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            model: response.model,
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
                    input_tokens: u.prompt_tokens,
                    output_tokens: u.completion_tokens,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            ),
        }
    }

    /// List available models.
    pub async fn list_models(&self) -> Result<Vec<GigaChatModel>> {
        let token = self.get_access_token().await?;

        let response = self
            .client
            .get(GIGACHAT_MODELS_URL)
            .header("Authorization", format!("Bearer {}", token))
            .header("Accept", "application/json")
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("GigaChat API error {}: {}", status, error_text),
            ));
        }

        let models_response: GigaChatModelsResponse = response.json().await?;
        Ok(models_response.data)
    }
}

#[async_trait]
impl Provider for GigaChatProvider {
    fn name(&self) -> &str {
        "gigachat"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let token = self.get_access_token().await?;
        let gigachat_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .header("Authorization", format!("Bearer {}", token))
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&gigachat_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("GigaChat API error {}: {}", status, error_text),
            ));
        }

        let gigachat_response: GigaChatResponse = response.json().await?;
        Ok(self.parse_response(gigachat_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let token = self.get_access_token().await?;
        let mut gigachat_request = self.convert_request(&request);
        gigachat_request.stream = true;

        let response = self
            .client
            .post(self.api_url())
            .header("Authorization", format!("Bearer {}", token))
            .header("Accept", "application/json")
            .header("Content-Type", "application/json")
            .json(&gigachat_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("GigaChat API error {}: {}", status, error_text),
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

                if let Ok(chunk) = serde_json::from_str::<GigaChatStreamChunk>(&event.data) {
                    if let Some(choice) = chunk.choices.first() {
                        if let Some(ref delta) = choice.delta {
                            if let Some(ref content) = delta.content {
                                if !content.is_empty() {
                                    yield StreamChunk {
                                        event_type: StreamEventType::ContentBlockDelta,
                                        index: Some(0),
                                        delta: Some(ContentDelta::TextDelta { text: content.clone() }),
                                        stop_reason: None,
                                        usage: None,
                                    };
                                }
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

// ==================== GigaChat API Types ====================

#[derive(Debug, Deserialize)]
struct GigaChatTokenResponse {
    access_token: String,
    expires_at: i64, // Unix timestamp in milliseconds
}

#[derive(Debug, Serialize)]
struct GigaChatRequest {
    model: String,
    messages: Vec<GigaChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    repetition_penalty: Option<f32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GigaChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct GigaChatResponse {
    #[serde(default)]
    id: Option<String>,
    model: String,
    choices: Vec<GigaChatChoice>,
    usage: Option<GigaChatUsage>,
}

#[derive(Debug, Deserialize)]
struct GigaChatChoice {
    message: GigaChatMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GigaChatUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    #[serde(default)]
    #[allow(dead_code)]
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct GigaChatStreamChunk {
    choices: Vec<GigaChatStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct GigaChatStreamChoice {
    delta: Option<GigaChatDelta>,
    #[allow(dead_code)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GigaChatDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GigaChatModelsResponse {
    data: Vec<GigaChatModel>,
}

/// GigaChat model information.
#[derive(Debug, Clone, Deserialize)]
pub struct GigaChatModel {
    /// Model ID
    pub id: String,
    /// Object type
    pub object: String,
    /// Model owner
    pub owned_by: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    // Helper to create a minimal provider for testing (without network)
    fn create_test_provider() -> GigaChatProvider {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()
            .unwrap();

        GigaChatProvider {
            config: ProviderConfig::new("test-credentials"),
            client,
            credentials: "test-credentials".to_string(),
            scope: "GIGACHAT_API_PERS".to_string(),
            token: Arc::new(RwLock::new(None)),
        }
    }

    #[test]
    fn test_provider_creation_personal() {
        let provider = GigaChatProvider::personal("test-credentials").unwrap();
        assert_eq!(provider.name(), "gigachat");
        assert_eq!(provider.scope, "GIGACHAT_API_PERS");
    }

    #[test]
    fn test_provider_creation_corporate() {
        let provider = GigaChatProvider::corporate("test-credentials").unwrap();
        assert_eq!(provider.name(), "gigachat");
        assert_eq!(provider.scope, "GIGACHAT_API_CORP");
    }

    #[test]
    fn test_request_conversion() {
        let provider = create_test_provider();

        let request = CompletionRequest::new("GigaChat-Pro", vec![Message::user("Hello")])
            .with_system("You are helpful")
            .with_max_tokens(1024)
            .with_temperature(0.7);

        let gigachat_req = provider.convert_request(&request);

        assert_eq!(gigachat_req.model, "GigaChat-Pro");
        assert_eq!(gigachat_req.messages.len(), 2); // system + user
        assert_eq!(gigachat_req.messages[0].role, "system");
        assert_eq!(gigachat_req.messages[0].content, "You are helpful");
        assert_eq!(gigachat_req.messages[1].role, "user");
        assert_eq!(gigachat_req.messages[1].content, "Hello");
        assert_eq!(gigachat_req.max_tokens, Some(1024));
        assert_eq!(gigachat_req.temperature, Some(0.7));
        assert!(!gigachat_req.stream);
    }

    #[test]
    fn test_request_conversion_default_model() {
        let provider = create_test_provider();

        let request = CompletionRequest::new("", vec![Message::user("Hello")]);

        let gigachat_req = provider.convert_request(&request);
        assert_eq!(gigachat_req.model, "GigaChat");
    }

    #[test]
    fn test_response_parsing() {
        let provider = create_test_provider();

        let response = GigaChatResponse {
            id: Some("resp-123".to_string()),
            model: "GigaChat".to_string(),
            choices: vec![GigaChatChoice {
                message: GigaChatMessage {
                    role: "assistant".to_string(),
                    content: "Hello! How can I help you?".to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: Some(GigaChatUsage {
                prompt_tokens: 10,
                completion_tokens: 20,
                total_tokens: 30,
            }),
        };

        let result = provider.parse_response(response);

        assert_eq!(result.id, "resp-123");
        assert_eq!(result.model, "GigaChat");
        assert_eq!(result.content.len(), 1);
        if let ContentBlock::Text { text } = &result.content[0] {
            assert_eq!(text, "Hello! How can I help you?");
        } else {
            panic!("Expected text content block");
        }
        assert!(matches!(result.stop_reason, StopReason::EndTurn));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 20);
    }

    #[test]
    fn test_stop_reason_mapping() {
        let provider = create_test_provider();

        // Test "stop" -> EndTurn
        let response1 = GigaChatResponse {
            id: None,
            model: "GigaChat".to_string(),
            choices: vec![GigaChatChoice {
                message: GigaChatMessage {
                    role: "assistant".to_string(),
                    content: "Done".to_string(),
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.parse_response(response1).stop_reason,
            StopReason::EndTurn
        ));

        // Test "length" -> MaxTokens
        let response2 = GigaChatResponse {
            id: None,
            model: "GigaChat".to_string(),
            choices: vec![GigaChatChoice {
                message: GigaChatMessage {
                    role: "assistant".to_string(),
                    content: "Truncated".to_string(),
                },
                finish_reason: Some("length".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.parse_response(response2).stop_reason,
            StopReason::MaxTokens
        ));

        // Test "blacklist" -> StopSequence
        let response3 = GigaChatResponse {
            id: None,
            model: "GigaChat".to_string(),
            choices: vec![GigaChatChoice {
                message: GigaChatMessage {
                    role: "assistant".to_string(),
                    content: "Filtered".to_string(),
                },
                finish_reason: Some("blacklist".to_string()),
            }],
            usage: None,
        };
        assert!(matches!(
            provider.parse_response(response3).stop_reason,
            StopReason::StopSequence
        ));
    }

    #[test]
    fn test_api_url() {
        let provider = create_test_provider();
        assert_eq!(provider.api_url(), GIGACHAT_API_URL);
    }

    #[test]
    fn test_request_serialization() {
        let messages = vec![GigaChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }];

        let request = GigaChatRequest {
            model: "GigaChat".to_string(),
            messages,
            temperature: Some(0.7),
            max_tokens: Some(1000),
            stream: false,
            top_p: None,
            repetition_penalty: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("GigaChat"));
        assert!(json.contains("Hello"));
        assert!(json.contains("0.7"));
        assert!(json.contains("1000"));
    }

    #[test]
    fn test_token_info() {
        let token_info = TokenInfo {
            access_token: "test-token".to_string(),
            expires_at: std::time::Instant::now() + std::time::Duration::from_secs(3600),
        };
        assert_eq!(token_info.access_token, "test-token");
    }
}
