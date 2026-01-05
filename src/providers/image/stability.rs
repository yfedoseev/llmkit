//! Stability AI provider implementation.
//!
//! This module provides access to Stability AI's image generation API.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::providers::StabilityProvider;
//!
//! // From environment variable
//! let provider = StabilityProvider::from_env()?;
//!
//! // With explicit API key
//! let provider = StabilityProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! - `stable-diffusion-3.5-large` - Latest Stable Diffusion 3.5 Large
//! - `stable-diffusion-3-large` - Stable Diffusion 3 Large
//! - `stable-diffusion-3-medium` - Stable Diffusion 3 Medium
//!
//! # Environment Variables
//!
//! - `STABILITY_API_KEY` - Your Stability AI API key
//! - `STABILITY_API_URL` - Optional custom Stability endpoint

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, StopReason, StreamChunk,
    StreamEventType, Usage,
};

const STABILITY_API_URL: &str = "https://api.stability.ai/v2beta/stable-image/generate/core";

/// Stability AI provider for image generation.
///
/// Provides access to Stability AI's image generation models.
pub struct StabilityProvider {
    config: ProviderConfig,
    client: Client,
}

#[derive(Debug, Serialize)]
struct StabilityRequest {
    prompt: String,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_format: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StabilityResponse {
    #[serde(default)]
    image: Option<String>, // Base64 encoded image
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct StabilityErrorResponse {
    name: Option<String>,
    message: Option<String>,
}

impl StabilityProvider {
    /// Create provider from environment variable.
    ///
    /// Reads: `STABILITY_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("STABILITY_API_KEY").ok();

        let config = ProviderConfig {
            api_key,
            ..Default::default()
        };

        Self::new(config)
    }

    /// Create provider with explicit API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    /// Create provider with custom config.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.api_key {
            headers.insert(
                "Authorization",
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

    fn api_url(&self) -> String {
        self.config
            .base_url
            .clone()
            .unwrap_or_else(|| STABILITY_API_URL.to_string())
    }

    /// Extract text from request as image prompt.
    fn extract_prompt(&self, request: &CompletionRequest) -> String {
        // Combine all text content from messages as the image prompt
        let mut prompts = Vec::new();

        for message in &request.messages {
            for content_block in &message.content {
                if let ContentBlock::Text { text } = content_block {
                    prompts.push(text.clone());
                }
            }
        }

        prompts.join(" ")
    }

    /// Convert unified request to Stability format.
    fn convert_request(&self, request: &CompletionRequest) -> StabilityRequest {
        let prompt = self.extract_prompt(request);

        StabilityRequest {
            prompt,
            model: request.model.clone(),
            output_format: Some("png".to_string()),
        }
    }

    /// Convert Stability response to unified format.
    fn convert_response(
        &self,
        response: StabilityResponse,
        request_model: String,
    ) -> CompletionResponse {
        let mut content = Vec::new();

        if let Some(image_data) = response.image {
            // Store image as base64 encoded PNG
            content.push(ContentBlock::Image {
                media_type: "image/png".to_string(),
                data: image_data,
            });
        }

        let stop_reason = match response.finish_reason.as_deref() {
            Some("success") => StopReason::EndTurn,
            _ => StopReason::EndTurn,
        };

        CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: request_model,
            content,
            stop_reason,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        }
    }

    /// Handle error responses.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        if let Ok(error_resp) = serde_json::from_str::<StabilityErrorResponse>(body) {
            let message = format!(
                "{}: {}",
                error_resp.name.as_deref().unwrap_or("Error"),
                error_resp.message.as_deref().unwrap_or("Unknown error")
            );
            match status.as_u16() {
                401 => Error::auth(message),
                403 => Error::auth(message),
                404 => Error::other(message),
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
impl Provider for StabilityProvider {
    fn name(&self) -> &str {
        "stability"
    }

    fn default_model(&self) -> Option<&str> {
        Some("stable-diffusion-3.5-large")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let stability_request = self.convert_request(&request);
        let request_model = request.model.clone();

        let response = self
            .client
            .post(self.api_url())
            .json(&stability_request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let stability_response: StabilityResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {}", e)))?;

        Ok(self.convert_response(stability_response, request_model))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // Stability AI doesn't support streaming, fall back to complete and convert
        let response = self.complete(request).await?;

        let chunks = vec![
            Ok(StreamChunk {
                event_type: StreamEventType::MessageStart,
                index: None,
                delta: None,
                stop_reason: None,
                usage: None,
            }),
            Ok(StreamChunk {
                event_type: StreamEventType::ContentBlockDelta,
                index: Some(0),
                delta: Some(ContentDelta::Text {
                    text: "[Image generated]".to_string(),
                }),
                stop_reason: None,
                usage: None,
            }),
            Ok(StreamChunk {
                event_type: StreamEventType::MessageStop,
                index: None,
                delta: None,
                stop_reason: Some(response.stop_reason),
                usage: Some(response.usage),
            }),
        ];

        let stream = futures::stream::iter(chunks);
        Ok(Box::pin(stream))
    }

    async fn count_tokens(
        &self,
        request: crate::types::TokenCountRequest,
    ) -> Result<crate::types::TokenCountResult> {
        // For image generation, estimate based on prompt text
        let total_chars: usize = request
            .messages
            .iter()
            .map(|m| m.text_content().len())
            .sum();
        let token_count = (total_chars / 4) as u32;
        Ok(crate::types::TokenCountResult {
            input_tokens: token_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stability_provider_name() {
        let config = ProviderConfig::new("test-key");
        let provider = StabilityProvider::new(config).unwrap();
        assert_eq!(provider.name(), "stability");
    }

    #[test]
    fn test_stability_default_model() {
        let config = ProviderConfig::new("test-key");
        let provider = StabilityProvider::new(config).unwrap();
        assert_eq!(provider.default_model(), Some("stable-diffusion-3.5-large"));
    }

    #[test]
    fn test_stability_extract_prompt() {
        use crate::types::{Message, Role};

        let config = ProviderConfig::new("test-key");
        let provider = StabilityProvider::new(config).unwrap();

        let message = Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "a beautiful landscape".to_string(),
            }],
        };

        let request = CompletionRequest::new("stability", vec![message]);

        let prompt = provider.extract_prompt(&request);
        assert_eq!(prompt, "a beautiful landscape");
    }
}
