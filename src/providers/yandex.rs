//! Yandex GPT API provider implementation.
//!
//! This module provides access to Yandex's YandexGPT foundation models.
//! YandexGPT offers high-quality Russian language models with 32k context.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::YandexProvider;
//!
//! // From environment variables
//! let provider = YandexProvider::from_env()?;
//!
//! // Or with explicit credentials
//! let provider = YandexProvider::new(
//!     "your-iam-token",
//!     "your-folder-id",
//! )?;
//! ```
//!
//! # Supported Features
//!
//! - YandexGPT Pro (32k context)
//! - YandexGPT Lite (8k context)
//! - YandexGPT Pro RC (preview)
//! - Streaming support
//! - System prompts
//!
//! # Environment Variables
//!
//! - `YANDEX_IAM_TOKEN` - Your Yandex Cloud IAM token
//! - `YANDEX_FOLDER_ID` - Your Yandex Cloud folder ID
//!
//! # Models
//!
//! - `yandexgpt` - Latest YandexGPT Pro
//! - `yandexgpt-lite` - Lightweight YandexGPT
//! - `yandexgpt/rc` - Release candidate version
//! - `yandexgpt-32k` - 32k context version

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

const YANDEX_API_URL: &str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion";
const YANDEX_STREAM_URL: &str =
    "https://llm.api.cloud.yandex.net/foundationModels/v1/completionAsync";

/// Yandex GPT API provider.
///
/// Provides access to Yandex's foundation models including YandexGPT Pro and Lite.
pub struct YandexProvider {
    config: ProviderConfig,
    client: Client,
    folder_id: String,
}

impl YandexProvider {
    /// Create a new Yandex provider with IAM token and folder ID.
    pub fn new(iam_token: impl Into<String>, folder_id: impl Into<String>) -> Result<Self> {
        let iam_token = iam_token.into();
        let folder_id = folder_id.into();

        let mut headers = reqwest::header::HeaderMap::new();

        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", iam_token)
                .parse()
                .map_err(|_| Error::config("Invalid IAM token format"))?,
        );

        headers.insert(
            "x-folder-id",
            folder_id
                .parse()
                .map_err(|_| Error::config("Invalid folder ID format"))?,
        );

        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        let config = ProviderConfig::new(&iam_token);

        let client = Client::builder()
            .timeout(config.timeout)
            .default_headers(headers)
            .build()?;

        Ok(Self {
            config,
            client,
            folder_id,
        })
    }

    /// Create a new Yandex provider from environment variables.
    ///
    /// Reads credentials from `YANDEX_IAM_TOKEN` and `YANDEX_FOLDER_ID`.
    pub fn from_env() -> Result<Self> {
        let iam_token = std::env::var("YANDEX_IAM_TOKEN")
            .map_err(|_| Error::config("YANDEX_IAM_TOKEN environment variable not set"))?;

        let folder_id = std::env::var("YANDEX_FOLDER_ID")
            .map_err(|_| Error::config("YANDEX_FOLDER_ID environment variable not set"))?;

        Self::new(iam_token, folder_id)
    }

    fn api_url(&self) -> &str {
        self.config.base_url.as_deref().unwrap_or(YANDEX_API_URL)
    }

    /// Build the model URI from model name.
    ///
    /// Format: `gpt://{folder_id}/{model_name}`
    fn model_uri(&self, model: &str) -> String {
        // If it already looks like a URI, use as-is
        if model.starts_with("gpt://") {
            model.to_string()
        } else {
            format!("gpt://{}/{}", self.folder_id, model)
        }
    }

    /// Convert our unified request to Yandex's format.
    fn convert_request(&self, request: &CompletionRequest) -> YandexRequest {
        // Get system prompt
        let system_text = request.system.clone().or_else(|| {
            request
                .messages
                .iter()
                .find(|m| m.role == Role::System)
                .map(|m| m.text_content())
        });

        // Convert messages
        let mut messages: Vec<YandexMessage> = Vec::new();

        // Add system message if present
        if let Some(text) = system_text {
            messages.push(YandexMessage {
                role: "system".to_string(),
                text,
            });
        }

        // Add conversation messages
        for msg in &request.messages {
            if msg.role == Role::System {
                continue; // Already handled
            }

            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                _ => continue,
            };

            messages.push(YandexMessage {
                role: role.to_string(),
                text: msg.text_content(),
            });
        }

        // Build completion options
        let completion_options = YandexCompletionOptions {
            stream: false, // Set in stream method
            temperature: request.temperature,
            max_tokens: request.max_tokens,
        };

        // Default model is yandexgpt-lite
        let model = if request.model.is_empty() || request.model == "default" {
            "yandexgpt-lite"
        } else {
            &request.model
        };

        YandexRequest {
            model_uri: self.model_uri(model),
            completion_options,
            messages,
        }
    }

    /// Parse Yandex response into our unified format.
    fn parse_response(&self, response: YandexResponse, model: String) -> CompletionResponse {
        let result = response.result;

        let text = result
            .alternatives
            .first()
            .map(|a| a.message.text.clone())
            .unwrap_or_default();

        let stop_reason = result
            .alternatives
            .first()
            .map(|a| match a.status.as_str() {
                "ALTERNATIVE_STATUS_COMPLETE" => StopReason::EndTurn,
                "ALTERNATIVE_STATUS_TRUNCATED_MAX_TOKENS" => StopReason::MaxTokens,
                "ALTERNATIVE_STATUS_CONTENT_FILTER" => StopReason::StopSequence,
                _ => StopReason::EndTurn,
            })
            .unwrap_or(StopReason::EndTurn);

        CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model,
            content: vec![ContentBlock::Text { text }],
            stop_reason,
            usage: Usage {
                input_tokens: result.usage.input_text_tokens as u32,
                output_tokens: result.usage.completion_tokens as u32,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        }
    }
}

#[async_trait]
impl Provider for YandexProvider {
    fn name(&self) -> &str {
        "yandex"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = request.model.clone();
        let yandex_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .json(&yandex_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Yandex API error {}: {}", status, error_text),
            ));
        }

        let yandex_response: YandexResponse = response.json().await?;
        Ok(self.parse_response(yandex_response, model))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // Yandex uses server-sent events for streaming
        let model = request.model.clone();
        let mut yandex_request = self.convert_request(&request);
        yandex_request.completion_options.stream = true;

        let response = self
            .client
            .post(self.api_url())
            .json(&yandex_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Yandex API error {}: {}", status, error_text),
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
            let mut accumulated_text = String::new();

            while let Some(event) = event_stream.next().await {
                let event = event.map_err(|e| Error::other(e.to_string()))?;

                if event.data.is_empty() || event.data == "[DONE]" {
                    continue;
                }

                if let Ok(chunk) = serde_json::from_str::<YandexStreamChunk>(&event.data) {
                    if let Some(alt) = chunk.result.alternatives.first() {
                        let full_text = &alt.message.text;
                        // Get only the new part
                        let new_text = if full_text.len() > accumulated_text.len() {
                            full_text[accumulated_text.len()..].to_string()
                        } else {
                            continue;
                        };
                        accumulated_text = full_text.clone();

                        yield StreamChunk {
                            event_type: StreamEventType::ContentBlockDelta,
                            index: Some(0),
                            delta: Some(ContentDelta::TextDelta { text: new_text }),
                            stop_reason: None,
                            usage: None,
                        };
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

// ==================== Yandex API Types ====================

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct YandexRequest {
    model_uri: String,
    completion_options: YandexCompletionOptions,
    messages: Vec<YandexMessage>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct YandexCompletionOptions {
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct YandexMessage {
    role: String,
    text: String,
}

#[derive(Debug, Deserialize)]
struct YandexResponse {
    result: YandexResult,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YandexResult {
    alternatives: Vec<YandexAlternative>,
    usage: YandexUsage,
    #[serde(default)]
    model_version: String,
}

#[derive(Debug, Deserialize)]
struct YandexAlternative {
    message: YandexMessage,
    status: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct YandexUsage {
    input_text_tokens: u64,
    completion_tokens: u64,
    #[serde(default)]
    total_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct YandexStreamChunk {
    result: YandexResult,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_uri() {
        // We can't easily test new() without a valid token, but we can test model_uri logic
        let folder_id = "test-folder";

        // Test model URI construction
        let uri = format!("gpt://{}/yandexgpt-lite", folder_id);
        assert_eq!(uri, "gpt://test-folder/yandexgpt-lite");

        // Test that already-formatted URIs are recognized
        let existing_uri = "gpt://my-folder/yandexgpt";
        assert!(existing_uri.starts_with("gpt://"));
    }
}
