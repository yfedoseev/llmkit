//! DataRobot ML Ops provider implementation.
//!
//! This module provides access to DataRobot's AI platform for model inference.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::DataRobotProvider;
//!
//! // From environment variable
//! let provider = DataRobotProvider::from_env()?;
//!
//! // With explicit API key
//! let provider = DataRobotProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Environment Variables
//!
//! - `DATAROBOT_API_KEY` - Your DataRobot API key
//! - `DATAROBOT_API_URL` - Optional custom DataRobot endpoint

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

const DATAROBOT_API_URL: &str = "https://api.datarobot.com/v2/inference";

/// DataRobot ML Ops provider.
///
/// Provides access to inference on DataRobot deployed models.
pub struct DataRobotProvider {
    config: ProviderConfig,
    client: Client,
}

#[derive(Debug, Serialize, Deserialize)]
struct DataRobotMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct DataRobotRequest {
    model: String,
    messages: Vec<DataRobotMessage>,
    temperature: Option<f32>,
    max_tokens: Option<i32>,
}

#[derive(Debug, Deserialize)]
struct DataRobotResponse {
    id: String,
    model: String,
    choices: Vec<DataRobotChoice>,
    usage: Option<DataRobotUsage>,
}

#[derive(Debug, Deserialize)]
struct DataRobotChoice {
    message: DataRobotMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DataRobotUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct DataRobotErrorResponse {
    error: Option<String>,
    message: Option<String>,
}

impl DataRobotProvider {
    /// Create provider from environment variable.
    ///
    /// Reads: `DATAROBOT_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("DATAROBOT_API_KEY").ok();

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

    fn api_url(&self) -> String {
        self.config
            .base_url
            .clone()
            .unwrap_or_else(|| DATAROBOT_API_URL.to_string())
    }

    /// Build messages for the request.
    fn build_messages(&self, request: &CompletionRequest) -> Vec<DataRobotMessage> {
        let mut messages = Vec::new();

        if let Some(ref system) = request.system {
            messages.push(DataRobotMessage {
                role: "system".to_string(),
                content: system.clone(),
            });
        }

        for msg in &request.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };

            messages.push(DataRobotMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        messages
    }

    /// Convert unified request to DataRobot format.
    fn convert_request(&self, request: &CompletionRequest) -> DataRobotRequest {
        let messages = self.build_messages(request);

        DataRobotRequest {
            model: request.model.clone(),
            messages,
            temperature: request.temperature,
            max_tokens: request.max_tokens.map(|t| t as i32),
        }
    }

    /// Convert DataRobot response to unified format.
    fn convert_response(&self, response: DataRobotResponse) -> CompletionResponse {
        let mut content = Vec::new();
        let mut stop_reason = StopReason::EndTurn;

        if let Some(choice) = response.choices.into_iter().next() {
            let text = choice.message.content.clone();
            if !text.is_empty() {
                content.push(ContentBlock::Text { text });
            }

            stop_reason = match choice.finish_reason.as_deref() {
                Some("stop") => StopReason::EndTurn,
                Some("length") => StopReason::MaxTokens,
                _ => StopReason::EndTurn,
            };
        }

        let usage = response
            .usage
            .map(|u| Usage {
                input_tokens: u.prompt_tokens,
                output_tokens: u.completion_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            })
            .unwrap_or_default();

        CompletionResponse {
            id: response.id,
            model: response.model,
            content,
            stop_reason,
            usage,
        }
    }

    /// Handle error responses.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        if let Ok(error_resp) = serde_json::from_str::<DataRobotErrorResponse>(body) {
            let message = error_resp
                .error
                .or(error_resp.message)
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
impl Provider for DataRobotProvider {
    fn name(&self) -> &str {
        "datarobot"
    }

    fn default_model(&self) -> Option<&str> {
        None // Varies by deployment
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let datarobot_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .json(&datarobot_request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let datarobot_response: DataRobotResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {}", e)))?;

        Ok(self.convert_response(datarobot_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // DataRobot doesn't support streaming, fall back to complete and convert to stream
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
                delta: response.content.first().and_then(|cb| {
                    if let ContentBlock::Text { text } = cb {
                        Some(ContentDelta::Text { text: text.clone() })
                    } else {
                        None
                    }
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
    ) -> crate::error::Result<crate::types::TokenCountResult> {
        // Rough estimation: 1 token ≈ 4 characters
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
    fn test_datarobot_provider_name() {
        let config = ProviderConfig::new("test-key");
        let provider = DataRobotProvider::new(config).unwrap();
        assert_eq!(provider.name(), "datarobot");
    }

    #[test]
    fn test_datarobot_message_building() {
        use crate::types::Message;

        let config = ProviderConfig::new("test-key");
        let provider = DataRobotProvider::new(config).unwrap();

        let message = Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Test".to_string(),
            }],
        };

        let request = CompletionRequest::new("datarobot", vec![message]);

        let messages = provider.build_messages(&request);
        assert!(!messages.is_empty());
        assert_eq!(messages[0].role, "user");
    }
}
