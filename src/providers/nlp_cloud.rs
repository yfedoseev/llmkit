//! NLP Cloud API provider implementation.
//!
//! This module provides access to NLP Cloud's various hosted models for chat completions.
//! NLP Cloud offers access to open-source and proprietary models with fine-tuning support.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::NlpCloudProvider;
//!
//! // From environment variable
//! let provider = NlpCloudProvider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = NlpCloudProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! - `chatdolphin` - Fast chat model
//! - `dolphin` - General purpose
//! - `llama-3-70b-instruct` - Llama 3 70B
//! - `mixtral-8x7b-instruct` - Mixtral MoE
//!
//! # Environment Variables
//!
//! - `NLP_CLOUD_API_KEY` - Your NLP Cloud API key

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

const NLP_CLOUD_API_BASE: &str = "https://api.nlpcloud.io/v1/gpu";

/// NLP Cloud API provider.
///
/// Provides access to various models hosted on NLP Cloud.
pub struct NlpCloudProvider {
    config: ProviderConfig,
    client: Client,
}

impl NlpCloudProvider {
    /// Create a new NLP Cloud provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Token {}", key)
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

    /// Create a new NLP Cloud provider from environment variable.
    ///
    /// Reads the API key from `NLP_CLOUD_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("NLP_CLOUD_API_KEY");
        Self::new(config)
    }

    /// Create a new NLP Cloud provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn api_url(&self, model: &str) -> String {
        format!(
            "{}/{}/chatbot",
            self.config
                .base_url
                .as_deref()
                .unwrap_or(NLP_CLOUD_API_BASE),
            model
        )
    }

    /// Convert our unified request to NLP Cloud's format.
    fn convert_request(&self, request: &CompletionRequest) -> NlpCloudRequest {
        // Build conversation history
        let mut history = Vec::new();
        let mut current_input = String::new();

        for msg in &request.messages {
            let text = msg
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
                Role::User => current_input = text,
                Role::Assistant => {
                    if !current_input.is_empty() {
                        history.push(NlpCloudHistoryItem {
                            input: current_input.clone(),
                            response: text,
                        });
                        current_input.clear();
                    }
                }
                Role::System => {
                    // NLP Cloud doesn't have a system role, prepend to first user message
                    current_input = format!("{}\n\n{}", text, current_input);
                }
            }
        }

        // Add system prompt if present and not already included
        if let Some(ref system) = request.system {
            if history.is_empty() && !current_input.contains(system) {
                current_input = format!("{}\n\n{}", system, current_input);
            }
        }

        NlpCloudRequest {
            input: current_input,
            history: if history.is_empty() {
                None
            } else {
                Some(history)
            },
            max_length: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
        }
    }

    fn convert_response(&self, model: &str, response: NlpCloudResponse) -> CompletionResponse {
        CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: model.to_string(),
            content: vec![ContentBlock::Text {
                text: response.response,
            }],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            }, // NLP Cloud doesn't return token counts in this format
        }
    }
}

#[async_trait]
impl Provider for NlpCloudProvider {
    fn name(&self) -> &str {
        "nlp-cloud"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = request.model.clone();
        let api_request = self.convert_request(&request);

        let response = self
            .client
            .post(&self.api_url(&model))
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("NLP Cloud API error {}: {}", status, error_text),
            ));
        }

        let api_response: NlpCloudResponse = response.json().await?;
        Ok(self.convert_response(&model, api_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // NLP Cloud doesn't support streaming, fall back to regular completion
        let response = self.complete(request).await?;

        let stream = async_stream::try_stream! {
            // Emit start
            yield StreamChunk {
                event_type: StreamEventType::ContentBlockStart,
                index: Some(0),
                delta: None,
                stop_reason: None,
                usage: None,
            };

            // Emit the full content as a single delta
            for block in response.content {
                if let ContentBlock::Text { text } = block {
                    yield StreamChunk {
                        event_type: StreamEventType::ContentBlockDelta,
                        index: Some(0),
                        delta: Some(ContentDelta::TextDelta { text }),
                        stop_reason: None,
                        usage: None,
                    };
                }
            }

            // Emit stop
            yield StreamChunk {
                event_type: StreamEventType::MessageStop,
                index: None,
                delta: None,
                stop_reason: Some(StopReason::EndTurn),
                usage: Some(response.usage),
            };
        };

        Ok(Box::pin(stream))
    }
}

// NLP Cloud API types

#[derive(Debug, Serialize)]
struct NlpCloudRequest {
    input: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    history: Option<Vec<NlpCloudHistoryItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_length: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Debug, Serialize)]
struct NlpCloudHistoryItem {
    input: String,
    response: String,
}

#[derive(Debug, Deserialize)]
struct NlpCloudResponse {
    response: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_request() {
        use crate::types::Message;

        let provider = NlpCloudProvider::new(ProviderConfig::new("test-key")).unwrap();

        let mut request = CompletionRequest::new(
            "chatdolphin",
            vec![Message {
                role: Role::User,
                content: vec![ContentBlock::Text {
                    text: "Hello".to_string(),
                }],
            }],
        );
        request.system = Some("You are helpful".to_string());
        request.max_tokens = Some(100);

        let api_request = provider.convert_request(&request);
        assert!(api_request.input.contains("You are helpful"));
        assert!(api_request.input.contains("Hello"));
    }
}
