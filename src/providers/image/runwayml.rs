//! RunwayML provider implementation.
//!
//! This module provides access to RunwayML's video generation API.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::providers::RunwayMLProvider;
//!
//! // From environment variable
//! let provider = RunwayMLProvider::from_env()?;
//!
//! // With explicit API key
//! let provider = RunwayMLProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! - `gen4_turbo` - Latest Generation 4 Turbo model
//! - `gen3a_turbo` - Generation 3A Turbo model
//! - `veo3.1` - VEO 3.1 model
//!
//! # Environment Variables
//!
//! - `RUNWAYML_API_SECRET` - Your RunwayML API secret key

use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, StopReason, StreamChunk,
    StreamEventType, Usage,
};
use futures::Stream;
use std::pin::Pin;

const RUNWAYML_API_URL: &str = "https://api.runwayml.com/v1";

/// RunwayML provider for video generation.
///
/// Provides access to RunwayML's video generation models.
pub struct RunwayMLProvider {
    #[allow(dead_code)]
    config: ProviderConfig,
    client: Client,
}

#[derive(Debug, Serialize)]
struct RunwayMLTaskRequest {
    model: String,
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    duration: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    aspect_ratio: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RunwayMLTaskResponse {
    id: String,
    status: String,
    #[serde(default)]
    result: Option<RunwayMLTaskResult>,
    #[serde(default)]
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RunwayMLTaskResult {
    #[serde(default)]
    output: Option<Vec<String>>, // Video URLs
}

impl RunwayMLProvider {
    /// Create provider from environment variable.
    ///
    /// Reads: `RUNWAYML_API_SECRET`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("RUNWAYML_API_SECRET").ok();

        if api_key.is_none() {
            return Err(Error::config(
                "RUNWAYML_API_SECRET environment variable not set",
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

    /// Create a video generation task.
    async fn create_task(&self, request: &RunwayMLTaskRequest) -> Result<String> {
        let response = self
            .client
            .post(format!("{}/tasks", RUNWAYML_API_URL))
            .json(request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let task_response: RunwayMLTaskResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {}", e)))?;

        Ok(task_response.id)
    }

    /// Poll task status until completion.
    async fn poll_task(&self, task_id: &str, timeout_secs: u64) -> Result<RunwayMLTaskResponse> {
        let start = std::time::Instant::now();
        let timeout = Duration::from_secs(timeout_secs);
        let mut delay = Duration::from_millis(500);

        loop {
            if start.elapsed() > timeout {
                return Err(Error::other("RunwayML task polling timeout"));
            }

            let response = self
                .client
                .get(format!("{}/tasks/{}", RUNWAYML_API_URL, task_id))
                .send()
                .await?;

            let status = response.status();
            let body = response.text().await?;

            if !status.is_success() {
                return Err(self.handle_error_response(status, &body));
            }

            let task_response: RunwayMLTaskResponse = serde_json::from_str(&body)
                .map_err(|e| Error::other(format!("Failed to parse response: {}", e)))?;

            match task_response.status.as_str() {
                "SUCCEEDED" => return Ok(task_response),
                "FAILED" => {
                    return Err(Error::other(format!(
                        "RunwayML task failed: {}",
                        task_response
                            .error
                            .unwrap_or_else(|| "Unknown error".to_string())
                    )))
                }
                "PENDING" | "RUNNING" => {
                    tokio::time::sleep(delay).await;
                    delay = std::cmp::min(delay.mul_f32(1.5), Duration::from_secs(10));
                }
                _ => {
                    return Err(Error::other(format!(
                        "Unknown task status: {}",
                        task_response.status
                    )))
                }
            }
        }
    }

    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        match status.as_u16() {
            401 => Error::auth(format!("RunwayML authentication failed: {}", body)),
            429 => Error::rate_limited("RunwayML rate limited", None),
            500..=599 => Error::server(status.as_u16(), body.to_string()),
            _ => Error::other(format!("RunwayML error ({}): {}", status, body)),
        }
    }
}

#[async_trait]
impl Provider for RunwayMLProvider {
    fn name(&self) -> &str {
        "runwayml"
    }

    fn default_model(&self) -> Option<&str> {
        Some("gen4_turbo")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let prompt = self.extract_prompt(&request);
        let model = if request.model.is_empty() {
            "gen4_turbo".to_string()
        } else {
            request.model.clone()
        };

        let task_request = RunwayMLTaskRequest {
            model: model.clone(),
            prompt,
            duration: None,
            aspect_ratio: None,
        };

        let task_id = self.create_task(&task_request).await?;
        let result = self.poll_task(&task_id, 300).await?; // 5 min timeout

        let video_url = result
            .result
            .and_then(|r| r.output)
            .and_then(|mut output| output.pop())
            .ok_or_else(|| Error::other("No video URL in response"))?;

        Ok(CompletionResponse {
            id: task_id,
            model,
            content: vec![ContentBlock::Text {
                text: format!("Video generated: {}", video_url),
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
        // RunwayML doesn't support streaming, fall back to complete and convert
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
                    text: "[Video generated]".to_string(),
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
