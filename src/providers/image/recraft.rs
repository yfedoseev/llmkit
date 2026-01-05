//! Recraft provider implementation.
//!
//! This module provides access to Recraft's image generation API.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::providers::RecraftProvider;
//!
//! // From environment variable
//! let provider = RecraftProvider::from_env()?;
//!
//! // With explicit API key
//! let provider = RecraftProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! - `recraft-v3` - Latest Recraft v3 model (ranked #1 on image benchmarks)
//!
//! # Environment Variables
//!
//! - `RECRAFT_API_TOKEN` - Your Recraft API token

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, StopReason, StreamChunk,
    StreamEventType, Usage,
};
use futures::Stream;
use std::pin::Pin;

const RECRAFT_API_URL: &str = "https://external.api.recraft.ai/v1";

/// Recraft provider for image generation.
///
/// Provides access to Recraft's image generation models.
pub struct RecraftProvider {
    #[allow(dead_code)]
    config: ProviderConfig,
    client: Client,
}

#[derive(Debug, Serialize)]
struct RecraftImageRequest {
    prompt: String,
    model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    style: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    substyle: Option<String>,
    n: u32,
}

#[derive(Debug, Deserialize)]
struct RecraftImageResponse {
    data: Vec<RecraftImage>,
}

#[derive(Debug, Deserialize)]
struct RecraftImage {
    url: String,
    #[serde(default)]
    #[allow(dead_code)]
    b64_json: Option<String>,
}

impl RecraftProvider {
    /// Create provider from environment variable.
    ///
    /// Reads: `RECRAFT_API_TOKEN`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("RECRAFT_API_TOKEN").ok();

        if api_key.is_none() {
            return Err(Error::config(
                "RECRAFT_API_TOKEN environment variable not set",
            ));
        }

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
            let bearer = format!("Bearer {}", key);
            headers.insert(
                "Authorization",
                bearer
                    .parse()
                    .map_err(|_| Error::config("Invalid API key format"))?,
            );
        }

        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json"
                .parse()
                .map_err(|_| Error::config("Failed to set content type"))?,
        );

        let client = Client::builder()
            .timeout(config.timeout)
            .default_headers(headers)
            .build()?;

        Ok(Self { config, client })
    }

    fn extract_prompt(&self, request: &CompletionRequest) -> String {
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

    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        match status.as_u16() {
            401 => Error::auth(format!("Recraft authentication failed: {}", body)),
            429 => Error::rate_limited("Recraft rate limited", None),
            500..=599 => Error::server(status.as_u16(), body.to_string()),
            _ => Error::other(format!("Recraft error ({}): {}", status, body)),
        }
    }
}

#[async_trait]
impl Provider for RecraftProvider {
    fn name(&self) -> &str {
        "recraft"
    }

    fn default_model(&self) -> Option<&str> {
        Some("recraft-v3")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let prompt = self.extract_prompt(&request);
        let model = if request.model.is_empty() {
            "recraft-v3".to_string()
        } else {
            request.model.clone()
        };

        let image_request = RecraftImageRequest {
            prompt,
            model: model.clone(),
            size: Some("1024x1024".to_string()),
            style: None,
            substyle: None,
            n: 1,
        };

        let response = self
            .client
            .post(format!("{}/images/generations", RECRAFT_API_URL))
            .json(&image_request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let image_response: RecraftImageResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {}", e)))?;

        let image_url = image_response
            .data
            .first()
            .map(|img| img.url.clone())
            .ok_or_else(|| Error::other("No image URL in response"))?;

        Ok(CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model,
            content: vec![ContentBlock::Text {
                text: format!("Image generated: {}", image_url),
            }],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        })
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // Recraft doesn't support streaming, fall back to complete and convert
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

        Ok(Box::pin(futures::stream::iter(chunks)))
    }

    fn supports_vision(&self) -> bool {
        false
    }

    fn supports_tools(&self) -> bool {
        false
    }

    fn supports_streaming(&self) -> bool {
        false
    }
}
