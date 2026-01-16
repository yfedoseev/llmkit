#![allow(dead_code)]
//! Cloudflare Workers AI provider implementation.
//!
//! This module provides access to Cloudflare's Workers AI platform.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::CloudflareProvider;
//!
//! // From environment variables
//! let provider = CloudflareProvider::from_env()?;
//!
//! // With explicit credentials
//! let provider = CloudflareProvider::new("account-id", "api-token")?;
//! ```
//!
//! # Supported Models
//!
//! - `@cf/meta/llama-3-8b-instruct`
//! - `@cf/mistral/mistral-7b-instruct-v0.1`
//! - `@cf/google/gemma-7b-it`
//! - And many more at <https://developers.cloudflare.com/workers-ai/models/>
//!
//! # Environment Variables
//!
//! - `CLOUDFLARE_API_TOKEN` - Your Cloudflare API token
//! - `CLOUDFLARE_ACCOUNT_ID` - Your Cloudflare account ID

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

const CLOUDFLARE_API_URL: &str = "https://api.cloudflare.com/client/v4/accounts";

/// Cloudflare Workers AI provider.
///
/// Provides access to models hosted on Cloudflare's edge network.
pub struct CloudflareProvider {
    config: ProviderConfig,
    client: Client,
    account_id: String,
}

impl CloudflareProvider {
    /// Create provider from environment variables.
    ///
    /// Reads: `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("CLOUDFLARE_API_TOKEN").ok();
        let account_id = std::env::var("CLOUDFLARE_ACCOUNT_ID")
            .map_err(|_| Error::config("CLOUDFLARE_ACCOUNT_ID environment variable not set"))?;

        let config = ProviderConfig {
            api_key,
            ..Default::default()
        };

        Self::with_config(account_id, config)
    }

    /// Create provider with account ID and API token.
    pub fn new(account_id: impl Into<String>, api_token: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_token);
        Self::with_config(account_id, config)
    }

    /// Create provider with custom config.
    fn with_config(account_id: impl Into<String>, config: ProviderConfig) -> Result<Self> {
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

        Ok(Self {
            config,
            client,
            account_id: account_id.into(),
        })
    }

    /// Get the API URL for a given model.
    fn get_api_url(&self, model: &str) -> String {
        format!(
            "{}/{}/ai/run/{}",
            CLOUDFLARE_API_URL, self.account_id, model
        )
    }

    /// Build messages array for the request.
    fn build_messages(&self, request: &CompletionRequest) -> Vec<CFMessage> {
        let mut messages = Vec::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            messages.push(CFMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        // Add conversation messages
        for msg in &request.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };

            messages.push(CFMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        messages
    }

    /// Convert unified request to Cloudflare format.
    fn convert_request(&self, request: &CompletionRequest) -> CFRequest {
        let messages = self.build_messages(request);

        CFRequest {
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            stream: Some(request.stream),
        }
    }

    /// Convert Cloudflare response to unified format.
    fn convert_response(&self, response: CFResponse, model: &str) -> CompletionResponse {
        let mut content = Vec::new();

        if let Some(result) = response.result {
            if let Some(text) = result.response {
                if !text.is_empty() {
                    content.push(ContentBlock::Text { text });
                }
            }
        }

        CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: model.to_string(),
            content,
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }
    }

    /// Handle error responses from Cloudflare API.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        if let Ok(error_resp) = serde_json::from_str::<CFErrorResponse>(body) {
            let message = error_resp
                .errors
                .first()
                .map(|e| e.message.clone())
                .unwrap_or_else(|| "Unknown error".to_string());
            match status.as_u16() {
                401 => Error::auth(message),
                403 => Error::auth(message),
                404 => Error::ModelNotFound(message),
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
impl Provider for CloudflareProvider {
    fn name(&self) -> &str {
        "cloudflare"
    }

    fn default_model(&self) -> Option<&str> {
        Some("@cf/meta/llama-3-8b-instruct")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let url = self.get_api_url(&request.model);
        let cf_request = self.convert_request(&request);

        let response = self.client.post(&url).json(&cf_request).send().await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let cf_response: CFResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

        Ok(self.convert_response(cf_response, &request.model))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let url = self.get_api_url(&request.model);
        let mut cf_request = self.convert_request(&request);
        cf_request.stream = Some(true);

        let response = self.client.post(&url).json(&cf_request).send().await?;

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

                // Process SSE lines
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
                        continue;
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

                    if let Ok(chunk_resp) = serde_json::from_str::<CFStreamChunk>(data) {
                        if let Some(text) = chunk_resp.response {
                            if !text.is_empty() {
                                yield StreamChunk {
                                    event_type: StreamEventType::ContentBlockDelta,
                                    index: Some(chunk_index),
                                    delta: Some(ContentDelta::Text { text }),
                                    stop_reason: None,
                                    usage: None,
                                };
                                chunk_index += 1;
                            }
                        }
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
struct CFRequest {
    messages: Vec<CFMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct CFMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct CFResponse {
    result: Option<CFResult>,
    success: bool,
}

#[derive(Debug, Deserialize)]
struct CFResult {
    response: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CFStreamChunk {
    response: Option<String>,
}

#[derive(Debug, Deserialize)]
struct CFErrorResponse {
    errors: Vec<CFError>,
}

#[derive(Debug, Deserialize)]
struct CFError {
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = CloudflareProvider::new("account-123", "test-token").unwrap();
        assert_eq!(provider.name(), "cloudflare");
        assert!(!provider.supports_tools());
        assert!(!provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_default_model() {
        let provider = CloudflareProvider::new("account-123", "test-token").unwrap();
        assert_eq!(
            provider.default_model(),
            Some("@cf/meta/llama-3-8b-instruct")
        );
    }

    #[test]
    fn test_account_id_stored() {
        let provider = CloudflareProvider::new("my-account-id", "test-token").unwrap();
        assert_eq!(provider.account_id, "my-account-id");
    }

    #[test]
    fn test_api_url() {
        let provider = CloudflareProvider::new("acc123", "test-token").unwrap();
        let url = provider.get_api_url("@cf/meta/llama-3-8b-instruct");
        assert_eq!(
            url,
            "https://api.cloudflare.com/client/v4/accounts/acc123/ai/run/@cf/meta/llama-3-8b-instruct"
        );
    }

    #[test]
    fn test_api_url_different_model() {
        let provider = CloudflareProvider::new("acc456", "test-token").unwrap();
        let url = provider.get_api_url("@cf/mistral/mistral-7b-instruct-v0.1");
        assert_eq!(
            url,
            "https://api.cloudflare.com/client/v4/accounts/acc456/ai/run/@cf/mistral/mistral-7b-instruct-v0.1"
        );
    }

    #[test]
    fn test_message_building() {
        let provider = CloudflareProvider::new("account-123", "test-token").unwrap();

        let request =
            CompletionRequest::new("@cf/meta/llama-3-8b-instruct", vec![Message::user("Hello")])
                .with_system("You are helpful");

        let cf_req = provider.convert_request(&request);

        assert_eq!(cf_req.messages.len(), 2);
        assert_eq!(cf_req.messages[0].role, "system");
        assert_eq!(cf_req.messages[0].content, "You are helpful");
        assert_eq!(cf_req.messages[1].role, "user");
        assert_eq!(cf_req.messages[1].content, "Hello");
    }

    #[test]
    fn test_request_parameters() {
        let provider = CloudflareProvider::new("account-123", "test-token").unwrap();

        let request =
            CompletionRequest::new("@cf/meta/llama-3-8b-instruct", vec![Message::user("Hello")])
                .with_max_tokens(1024)
                .with_temperature(0.7)
                .with_top_p(0.9);

        let cf_req = provider.convert_request(&request);

        assert_eq!(cf_req.max_tokens, Some(1024));
        assert_eq!(cf_req.temperature, Some(0.7));
        assert_eq!(cf_req.top_p, Some(0.9));
    }

    #[test]
    fn test_response_parsing() {
        let provider = CloudflareProvider::new("account-123", "test-token").unwrap();

        let response = CFResponse {
            result: Some(CFResult {
                response: Some("Hello there!".to_string()),
            }),
            success: true,
        };

        let result = provider.convert_response(response, "@cf/meta/llama-3-8b-instruct");

        assert_eq!(result.model, "@cf/meta/llama-3-8b-instruct");
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlock::Text { text } => {
                assert_eq!(text, "Hello there!");
            }
            other => {
                panic!("Expected text content, got {:?}", other);
            }
        }
        assert!(matches!(result.stop_reason, StopReason::EndTurn));
    }

    #[test]
    fn test_response_empty_result() {
        let provider = CloudflareProvider::new("account-123", "test-token").unwrap();

        let response = CFResponse {
            result: None,
            success: true,
        };

        let result = provider.convert_response(response, "@cf/meta/llama-3-8b-instruct");
        assert!(result.content.is_empty());
    }

    #[test]
    fn test_request_serialization() {
        let request = CFRequest {
            messages: vec![CFMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: Some(0.9),
            stream: Some(false),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"max_tokens\":1000"));
        assert!(json.contains("\"temperature\":0.7"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "result": {"response": "Hi!"},
            "success": true
        }"#;

        let response: CFResponse = serde_json::from_str(json).unwrap();
        assert!(response.success);
        assert_eq!(response.result.unwrap().response, Some("Hi!".to_string()));
    }

    #[test]
    fn test_multi_turn_conversation() {
        let provider = CloudflareProvider::new("account-123", "test-token").unwrap();

        let request = CompletionRequest::new(
            "@cf/meta/llama-3-8b-instruct",
            vec![
                Message::user("Hello"),
                Message::assistant("Hi there!"),
                Message::user("How are you?"),
            ],
        )
        .with_system("Be friendly");

        let cf_req = provider.convert_request(&request);

        assert_eq!(cf_req.messages.len(), 4);
        assert_eq!(cf_req.messages[0].role, "system");
        assert_eq!(cf_req.messages[1].role, "user");
        assert_eq!(cf_req.messages[2].role, "assistant");
        assert_eq!(cf_req.messages[3].role, "user");
    }

    #[test]
    fn test_error_handling() {
        let provider = CloudflareProvider::new("account-123", "test-token").unwrap();

        // Test 401 -> auth error
        let error = provider.handle_error_response(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"errors": [{"message": "Invalid API token"}]}"#,
        );
        assert!(matches!(error, Error::Authentication(_)));

        // Test 404 -> model not found
        let error = provider.handle_error_response(
            reqwest::StatusCode::NOT_FOUND,
            r#"{"errors": [{"message": "Model not found"}]}"#,
        );
        assert!(matches!(error, Error::ModelNotFound(_)));

        // Test 429 -> rate limited
        let error = provider.handle_error_response(
            reqwest::StatusCode::TOO_MANY_REQUESTS,
            r#"{"errors": [{"message": "Rate limit exceeded"}]}"#,
        );
        assert!(matches!(error, Error::RateLimited { .. }));
    }
}
