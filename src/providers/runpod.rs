#![allow(dead_code)]
//! RunPod Serverless API provider implementation.
//!
//! This module provides access to RunPod's serverless GPU inference platform.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::RunPodProvider;
//!
//! // From environment variable with endpoint ID
//! let provider = RunPodProvider::from_env()?;
//!
//! // With explicit endpoint and API key
//! let provider = RunPodProvider::new("endpoint-id", "your-api-key")?;
//! ```
//!
//! # Environment Variables
//!
//! - `RUNPOD_API_KEY` - Your RunPod API key
//! - `RUNPOD_ENDPOINT_ID` - Your serverless endpoint ID

use std::pin::Pin;
use std::time::Duration;

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

const RUNPOD_API_URL: &str = "https://api.runpod.ai/v2";

/// RunPod Serverless API provider.
///
/// Provides access to models deployed on RunPod's serverless infrastructure.
pub struct RunPodProvider {
    config: ProviderConfig,
    client: Client,
    /// The endpoint ID for the deployed model
    endpoint_id: String,
    /// Polling interval for async jobs
    poll_interval: Duration,
    /// Maximum wait time for jobs
    max_wait: Duration,
}

impl RunPodProvider {
    /// Create provider from environment variables.
    ///
    /// Reads: `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("RUNPOD_API_KEY").ok();
        let endpoint_id = std::env::var("RUNPOD_ENDPOINT_ID")
            .map_err(|_| Error::config("RUNPOD_ENDPOINT_ID environment variable not set"))?;

        let config = ProviderConfig {
            api_key,
            ..Default::default()
        };

        Self::with_config(endpoint_id, config)
    }

    /// Create provider with endpoint ID and API key.
    pub fn new(endpoint_id: impl Into<String>, api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::with_config(endpoint_id, config)
    }

    /// Create provider with endpoint ID and custom config.
    fn with_config(endpoint_id: impl Into<String>, config: ProviderConfig) -> Result<Self> {
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
            endpoint_id: endpoint_id.into(),
            poll_interval: Duration::from_millis(500),
            max_wait: Duration::from_secs(300),
        })
    }

    /// Set the polling interval for async jobs.
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Set the maximum wait time for jobs.
    pub fn with_max_wait(mut self, max_wait: Duration) -> Self {
        self.max_wait = max_wait;
        self
    }

    /// Get the run URL for synchronous execution.
    fn get_run_url(&self) -> String {
        format!("{}/{}/runsync", RUNPOD_API_URL, self.endpoint_id)
    }

    /// Get the run URL for asynchronous execution.
    fn get_run_async_url(&self) -> String {
        format!("{}/{}/run", RUNPOD_API_URL, self.endpoint_id)
    }

    /// Get the status URL for a job.
    fn get_status_url(&self, job_id: &str) -> String {
        format!("{}/{}/status/{}", RUNPOD_API_URL, self.endpoint_id, job_id)
    }

    /// Get the stream URL for a job.
    fn get_stream_url(&self, job_id: &str) -> String {
        format!("{}/{}/stream/{}", RUNPOD_API_URL, self.endpoint_id, job_id)
    }

    /// Build prompt from messages (for models that expect raw prompts).
    fn build_prompt(&self, request: &CompletionRequest) -> String {
        let mut prompt = String::new();

        // Add system message if present
        if let Some(ref system) = request.system {
            prompt.push_str(&format!("<|system|>\n{}\n", system));
        }

        // Convert messages
        for msg in &request.messages {
            let role_tag = match msg.role {
                Role::User => "<|user|>",
                Role::Assistant => "<|assistant|>",
                Role::System => "<|system|>",
            };

            let content = msg
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

            prompt.push_str(&format!("{}\n{}\n", role_tag, content));
        }

        prompt.push_str("<|assistant|>\n");
        prompt
    }

    /// Convert unified request to RunPod input format.
    fn convert_request(&self, request: &CompletionRequest) -> RunPodInput {
        let prompt = self.build_prompt(request);

        RunPodInput {
            prompt,
            max_new_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            do_sample: request.temperature.map(|t| t > 0.0),
        }
    }

    /// Execute a synchronous job.
    async fn run_sync(&self, input: &RunPodInput) -> Result<RunPodJobResponse> {
        let request = RunPodRequest {
            input: input.clone(),
        };

        let response = self
            .client
            .post(self.get_run_url())
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let job_response: RunPodJobResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

        // Check if the job completed immediately
        if job_response.status == "COMPLETED" {
            return Ok(job_response);
        }

        // Poll for completion
        self.wait_for_job(&job_response.id).await
    }

    /// Wait for a job to complete.
    async fn wait_for_job(&self, job_id: &str) -> Result<RunPodJobResponse> {
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > self.max_wait {
                return Err(Error::Timeout);
            }

            let response = self.client.get(self.get_status_url(job_id)).send().await?;

            let status = response.status();
            let body = response.text().await?;

            if !status.is_success() {
                return Err(self.handle_error_response(status, &body));
            }

            let job_response: RunPodJobResponse = serde_json::from_str(&body)
                .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

            match job_response.status.as_str() {
                "COMPLETED" => return Ok(job_response),
                "FAILED" | "CANCELLED" | "TIMED_OUT" => {
                    let error = job_response
                        .error
                        .unwrap_or_else(|| "Job failed".to_string());
                    return Err(Error::other(error));
                }
                _ => {
                    tokio::time::sleep(self.poll_interval).await;
                }
            }
        }
    }

    /// Convert RunPod job response to unified format.
    fn convert_response(&self, job: RunPodJobResponse) -> CompletionResponse {
        let mut content = Vec::new();

        if let Some(output) = job.output {
            // Handle different output formats
            let text = match output {
                RunPodOutput::String(s) => s,
                RunPodOutput::Object {
                    text,
                    generated_text,
                    output,
                } => text.or(generated_text).or(output).unwrap_or_default(),
                RunPodOutput::Array(arr) => arr.join(""),
            };

            if !text.is_empty() {
                content.push(ContentBlock::Text { text });
            }
        }

        CompletionResponse {
            id: job.id,
            model: self.endpoint_id.clone(),
            content,
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
        }
    }

    /// Handle error responses from RunPod API.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        if let Ok(error_resp) = serde_json::from_str::<RunPodErrorResponse>(body) {
            let message = error_resp
                .error
                .unwrap_or_else(|| error_resp.message.unwrap_or_default());
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
impl Provider for RunPodProvider {
    fn name(&self) -> &str {
        "runpod"
    }

    fn default_model(&self) -> Option<&str> {
        Some(self.endpoint_id.as_str())
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let input = self.convert_request(&request);
        let job = self.run_sync(&input).await?;
        Ok(self.convert_response(job))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let input = self.convert_request(&request);
        let runpod_request = RunPodRequest { input };

        // Start an async job
        let response = self
            .client
            .post(self.get_run_async_url())
            .json(&runpod_request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let job_response: RunPodJobResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

        let job_id = job_response.id;
        let stream_url = self.get_stream_url(&job_id);
        let client = self.client.clone();

        let stream = async_stream::try_stream! {
            use futures::StreamExt;

            // Poll the stream endpoint
            let response = client
                .get(&stream_url)
                .send()
                .await?;

            let status_code = response.status().as_u16();
            if status_code >= 400 {
                Err(Error::server(status_code, "Stream request failed".to_string()))?;
            }

            let mut byte_stream = response.bytes_stream();
            let mut buffer = String::new();
            let mut chunk_index = 0usize;

            while let Some(chunk) = byte_stream.next().await {
                let chunk = chunk.map_err(|e| Error::stream(format!("Stream error: {}", e)))?;
                buffer.push_str(&String::from_utf8_lossy(&chunk));

                // Process complete JSON objects
                while let Some(pos) = buffer.find('\n') {
                    let line = buffer[..pos].trim().to_string();
                    buffer = buffer[pos + 1..].to_string();

                    if line.is_empty() {
                        continue;
                    }

                    if let Ok(stream_resp) = serde_json::from_str::<RunPodStreamResponse>(&line) {
                        if let Some(output) = stream_resp.output {
                            if !output.is_empty() {
                                yield StreamChunk {
                                    event_type: StreamEventType::ContentBlockDelta,
                                    index: Some(chunk_index),
                                    delta: Some(ContentDelta::TextDelta { text: output }),
                                    stop_reason: None,
                                    usage: None,
                                };
                                chunk_index += 1;
                            }
                        }

                        if stream_resp.status == Some("COMPLETED".to_string()) {
                            yield StreamChunk {
                                event_type: StreamEventType::MessageStop,
                                index: None,
                                delta: None,
                                stop_reason: None,
                                usage: None,
                            };
                            break;
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
struct RunPodRequest {
    input: RunPodInput,
}

#[derive(Debug, Clone, Serialize)]
struct RunPodInput {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    do_sample: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct RunPodJobResponse {
    id: String,
    status: String,
    output: Option<RunPodOutput>,
    error: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum RunPodOutput {
    String(String),
    Object {
        text: Option<String>,
        generated_text: Option<String>,
        output: Option<String>,
    },
    Array(Vec<String>),
}

#[derive(Debug, Deserialize)]
struct RunPodStreamResponse {
    output: Option<String>,
    status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct RunPodErrorResponse {
    error: Option<String>,
    message: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = RunPodProvider::new("endpoint-123", "test-key").unwrap();
        assert_eq!(provider.name(), "runpod");
        assert!(!provider.supports_tools());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_default_model() {
        let provider = RunPodProvider::new("my-endpoint", "test-key").unwrap();
        assert_eq!(provider.default_model(), Some("my-endpoint"));
    }

    #[test]
    fn test_api_urls() {
        let provider = RunPodProvider::new("ep123", "test-key").unwrap();

        assert_eq!(
            provider.get_run_url(),
            "https://api.runpod.ai/v2/ep123/runsync"
        );
        assert_eq!(
            provider.get_run_async_url(),
            "https://api.runpod.ai/v2/ep123/run"
        );
        assert_eq!(
            provider.get_status_url("job456"),
            "https://api.runpod.ai/v2/ep123/status/job456"
        );
        assert_eq!(
            provider.get_stream_url("job456"),
            "https://api.runpod.ai/v2/ep123/stream/job456"
        );
    }

    #[test]
    fn test_prompt_building() {
        let provider = RunPodProvider::new("endpoint-123", "test-key").unwrap();

        let request = CompletionRequest::new("model", vec![Message::user("Hello")])
            .with_system("You are helpful");

        let input = provider.convert_request(&request);

        assert!(input.prompt.contains("<|system|>"));
        assert!(input.prompt.contains("You are helpful"));
        assert!(input.prompt.contains("<|user|>"));
        assert!(input.prompt.contains("Hello"));
        assert!(input.prompt.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_poll_interval_config() {
        let provider = RunPodProvider::new("endpoint-123", "test-key")
            .unwrap()
            .with_poll_interval(Duration::from_secs(1))
            .with_max_wait(Duration::from_secs(60));

        assert_eq!(provider.poll_interval, Duration::from_secs(1));
        assert_eq!(provider.max_wait, Duration::from_secs(60));
    }
}
