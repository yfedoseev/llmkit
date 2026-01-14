#![allow(dead_code)]
//! Replicate API provider implementation.
//!
//! This module provides access to Replicate's model hosting platform.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::ReplicateProvider;
//!
//! // From environment variable
//! let provider = ReplicateProvider::from_env()?;
//!
//! // With explicit API token
//! let provider = ReplicateProvider::with_api_key("your-api-token")?;
//! ```
//!
//! # Supported Models
//!
//! Replicate hosts many open-source models. Common ones include:
//! - `meta/llama-2-70b-chat`
//! - `meta/meta-llama-3-70b-instruct`
//! - `mistralai/mixtral-8x7b-instruct-v0.1`
//!
//! # Environment Variables
//!
//! - `REPLICATE_API_TOKEN` - Your Replicate API token

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

const REPLICATE_API_URL: &str = "https://api.replicate.com/v1";

/// Replicate API provider.
///
/// Provides access to models hosted on Replicate.
pub struct ReplicateProvider {
    config: ProviderConfig,
    client: Client,
    /// Polling interval for async predictions
    poll_interval: Duration,
    /// Maximum wait time for predictions
    max_wait: Duration,
}

impl ReplicateProvider {
    /// Create provider from environment variable.
    ///
    /// Reads: `REPLICATE_API_TOKEN`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("REPLICATE_API_TOKEN").ok();

        let config = ProviderConfig {
            api_key,
            ..Default::default()
        };

        Self::new(config)
    }

    /// Create provider with explicit API token.
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

        Ok(Self {
            config,
            client,
            poll_interval: Duration::from_millis(500),
            max_wait: Duration::from_secs(300),
        })
    }

    /// Set the polling interval for async predictions.
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }

    /// Set the maximum wait time for predictions.
    pub fn with_max_wait(mut self, max_wait: Duration) -> Self {
        self.max_wait = max_wait;
        self
    }

    /// Build the prompt from messages (Replicate often uses raw prompts).
    fn build_prompt(&self, request: &CompletionRequest) -> String {
        let mut prompt = String::new();

        // Add system prompt if present
        if let Some(ref system) = request.system {
            prompt.push_str(&format!("System: {}\n\n", system));
        }

        // Convert messages to prompt format
        for msg in &request.messages {
            let role_prefix = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
                Role::System => "System",
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

            prompt.push_str(&format!("{}: {}\n\n", role_prefix, content));
        }

        prompt.push_str("Assistant: ");
        prompt
    }

    /// Convert unified request to Replicate prediction input.
    fn convert_request(&self, request: &CompletionRequest) -> ReplicatePredictionInput {
        let prompt = self.build_prompt(request);

        ReplicatePredictionInput {
            prompt,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            top_p: request.top_p,
            system_prompt: request.system.clone(),
        }
    }

    /// Extract model owner and name from model string.
    fn parse_model(&self, model: &str) -> (String, String) {
        if let Some((owner, rest)) = model.split_once('/') {
            // Handle version specifier if present (owner/model:version)
            let name = rest.split(':').next().unwrap_or(rest);
            (owner.to_string(), name.to_string())
        } else {
            // Default to meta if no owner specified
            ("meta".to_string(), model.to_string())
        }
    }

    /// Create a prediction and wait for completion.
    async fn create_and_wait_prediction(
        &self,
        model: &str,
        input: &ReplicatePredictionInput,
        stream: bool,
    ) -> Result<ReplicatePrediction> {
        let (owner, name) = self.parse_model(model);

        // Create the prediction
        let create_request = ReplicateCreatePrediction {
            model: Some(format!("{}/{}", owner, name)),
            version: None,
            input: serde_json::to_value(input)?,
            stream,
        };

        let response = self
            .client
            .post(format!("{}/predictions", REPLICATE_API_URL))
            .json(&create_request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let prediction: ReplicatePrediction = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

        // If streaming, return immediately for stream handling
        if stream {
            return Ok(prediction);
        }

        // Poll until completion
        self.wait_for_prediction(&prediction.id).await
    }

    /// Wait for a prediction to complete.
    async fn wait_for_prediction(&self, prediction_id: &str) -> Result<ReplicatePrediction> {
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > self.max_wait {
                return Err(Error::Timeout);
            }

            let response = self
                .client
                .get(format!(
                    "{}/predictions/{}",
                    REPLICATE_API_URL, prediction_id
                ))
                .send()
                .await?;

            let status = response.status();
            let body = response.text().await?;

            if !status.is_success() {
                return Err(self.handle_error_response(status, &body));
            }

            let prediction: ReplicatePrediction = serde_json::from_str(&body)
                .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

            match prediction.status.as_str() {
                "succeeded" => return Ok(prediction),
                "failed" | "canceled" => {
                    let error = prediction
                        .error
                        .unwrap_or_else(|| "Unknown error".to_string());
                    return Err(Error::other(format!("Prediction failed: {}", error)));
                }
                _ => {
                    tokio::time::sleep(self.poll_interval).await;
                }
            }
        }
    }

    /// Convert Replicate prediction to unified response.
    fn convert_response(&self, prediction: ReplicatePrediction) -> CompletionResponse {
        let mut content = Vec::new();

        // Replicate output can be a string or array of strings
        if let Some(output) = prediction.output {
            let text = match output {
                serde_json::Value::String(s) => s,
                serde_json::Value::Array(arr) => arr
                    .into_iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect::<Vec<_>>()
                    .join(""),
                _ => String::new(),
            };

            if !text.is_empty() {
                content.push(ContentBlock::Text { text });
            }
        }

        // Extract usage metrics if available
        let usage = prediction.metrics.map(|m| Usage {
            input_tokens: m.input_token_count.unwrap_or(0),
            output_tokens: m.output_token_count.unwrap_or(0),
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        });

        CompletionResponse {
            id: prediction.id,
            model: prediction.model.unwrap_or_default(),
            content,
            stop_reason: StopReason::EndTurn,
            usage: usage.unwrap_or_default(),
        }
    }

    /// Handle error responses from Replicate API.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        if let Ok(error_resp) = serde_json::from_str::<ReplicateErrorResponse>(body) {
            let message = error_resp.detail;
            match status.as_u16() {
                401 => Error::auth(message),
                403 => Error::auth(message),
                404 => Error::ModelNotFound(message),
                422 => Error::invalid_request(message),
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
impl Provider for ReplicateProvider {
    fn name(&self) -> &str {
        "replicate"
    }

    fn default_model(&self) -> Option<&str> {
        Some("meta/meta-llama-3-70b-instruct")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let input = self.convert_request(&request);
        let prediction = self
            .create_and_wait_prediction(&request.model, &input, false)
            .await?;

        Ok(self.convert_response(prediction))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let input = self.convert_request(&request);
        let prediction = self
            .create_and_wait_prediction(&request.model, &input, true)
            .await?;

        // Get the stream URL
        let stream_url = prediction
            .urls
            .and_then(|u| u.stream)
            .ok_or_else(|| Error::not_supported("Streaming not available for this prediction"))?;

        let client = self.client.clone();

        let stream = async_stream::try_stream! {
            use futures::StreamExt;
            use eventsource_stream::Eventsource;

            let response = client
                .get(&stream_url)
                .header("Accept", "text/event-stream")
                .send()
                .await?;

            let status_code = response.status().as_u16();
            if status_code >= 400 {
                Err(Error::server(status_code, "Stream request failed".to_string()))?;
            }

            let mut event_stream = response.bytes_stream().eventsource();
            let mut chunk_index = 0usize;

            while let Some(event) = event_stream.next().await {
                let event = event.map_err(|e| Error::stream(format!("Stream error: {}", e)))?;

                match event.event.as_str() {
                    "output" => {
                        let text = event.data;
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
                    "done" => {
                        yield StreamChunk {
                            event_type: StreamEventType::MessageStop,
                            index: None,
                            delta: None,
                            stop_reason: None,
                            usage: None,
                        };
                        break;
                    }
                    "error" => {
                        Err(Error::stream(format!("Stream error: {}", event.data)))?;
                    }
                    _ => {}
                }
            }
        };

        Ok(Box::pin(stream))
    }

    fn supports_tools(&self) -> bool {
        false // Most Replicate models don't support structured tool calling
    }

    fn supports_vision(&self) -> bool {
        false // Depends on the model
    }

    fn supports_streaming(&self) -> bool {
        true
    }
}

// ============ Request/Response Types ============

#[derive(Debug, Serialize)]
struct ReplicatePredictionInput {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_prompt: Option<String>,
}

#[derive(Debug, Serialize)]
struct ReplicateCreatePrediction {
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
    input: serde_json::Value,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct ReplicatePrediction {
    id: String,
    model: Option<String>,
    status: String,
    output: Option<serde_json::Value>,
    error: Option<String>,
    metrics: Option<ReplicateMetrics>,
    urls: Option<ReplicateUrls>,
}

#[derive(Debug, Deserialize)]
struct ReplicateMetrics {
    input_token_count: Option<u32>,
    output_token_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct ReplicateUrls {
    stream: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ReplicateErrorResponse {
    detail: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = ReplicateProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "replicate");
        assert!(!provider.supports_tools());
        assert!(!provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_default_model() {
        let provider = ReplicateProvider::with_api_key("test-key").unwrap();
        assert_eq!(
            provider.default_model(),
            Some("meta/meta-llama-3-70b-instruct")
        );
    }

    #[test]
    fn test_poll_settings() {
        let provider = ReplicateProvider::with_api_key("test-key")
            .unwrap()
            .with_poll_interval(Duration::from_secs(1))
            .with_max_wait(Duration::from_secs(600));

        assert_eq!(provider.poll_interval, Duration::from_secs(1));
        assert_eq!(provider.max_wait, Duration::from_secs(600));
    }

    #[test]
    fn test_model_parsing() {
        let provider = ReplicateProvider::with_api_key("test-key").unwrap();

        let (owner, name) = provider.parse_model("meta/llama-2-70b");
        assert_eq!(owner, "meta");
        assert_eq!(name, "llama-2-70b");

        let (owner, name) = provider.parse_model("meta/llama-2-70b:abc123");
        assert_eq!(owner, "meta");
        assert_eq!(name, "llama-2-70b");

        let (owner, name) = provider.parse_model("some-model");
        assert_eq!(owner, "meta");
        assert_eq!(name, "some-model");

        let (owner, name) = provider.parse_model("mistralai/mixtral-8x7b");
        assert_eq!(owner, "mistralai");
        assert_eq!(name, "mixtral-8x7b");
    }

    #[test]
    fn test_prompt_building() {
        let provider = ReplicateProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("meta/llama-2-70b", vec![Message::user("Hello")])
            .with_system("You are helpful");

        let input = provider.convert_request(&request);

        assert!(input.prompt.contains("System: You are helpful"));
        assert!(input.prompt.contains("User: Hello"));
        assert!(input.prompt.ends_with("Assistant: "));
    }

    #[test]
    fn test_request_parameters() {
        let provider = ReplicateProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new("meta/llama-2-70b", vec![Message::user("Hello")])
            .with_system("Be helpful")
            .with_max_tokens(1024)
            .with_temperature(0.7)
            .with_top_p(0.9);

        let input = provider.convert_request(&request);

        assert_eq!(input.max_tokens, Some(1024));
        assert_eq!(input.temperature, Some(0.7));
        assert_eq!(input.top_p, Some(0.9));
        assert_eq!(input.system_prompt, Some("Be helpful".to_string()));
    }

    #[test]
    fn test_response_parsing_string() {
        let provider = ReplicateProvider::with_api_key("test-key").unwrap();

        let prediction = ReplicatePrediction {
            id: "pred-123".to_string(),
            model: Some("meta/llama-2-70b".to_string()),
            status: "succeeded".to_string(),
            output: Some(serde_json::Value::String("Hello there!".to_string())),
            error: None,
            metrics: Some(ReplicateMetrics {
                input_token_count: Some(10),
                output_token_count: Some(20),
            }),
            urls: None,
        };

        let result = provider.convert_response(prediction);

        assert_eq!(result.id, "pred-123");
        assert_eq!(result.model, "meta/llama-2-70b");
        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlock::Text { text } => {
                assert_eq!(text, "Hello there!");
            }
            other => {
                panic!("Expected text content, got {:?}", other);
            }
        }
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 20);
    }

    #[test]
    fn test_response_parsing_array() {
        let provider = ReplicateProvider::with_api_key("test-key").unwrap();

        let prediction = ReplicatePrediction {
            id: "pred-456".to_string(),
            model: Some("meta/llama-2-70b".to_string()),
            status: "succeeded".to_string(),
            output: Some(serde_json::json!(["Hello", " ", "world", "!"])),
            error: None,
            metrics: None,
            urls: None,
        };

        let result = provider.convert_response(prediction);

        assert_eq!(result.content.len(), 1);
        match &result.content[0] {
            ContentBlock::Text { text } => {
                assert_eq!(text, "Hello world!");
            }
            other => {
                panic!("Expected text content, got {:?}", other);
            }
        }
    }

    #[test]
    fn test_multi_turn_conversation() {
        let provider = ReplicateProvider::with_api_key("test-key").unwrap();

        let request = CompletionRequest::new(
            "meta/llama-2-70b",
            vec![
                Message::user("Hello"),
                Message::assistant("Hi there!"),
                Message::user("How are you?"),
            ],
        );

        let input = provider.convert_request(&request);

        assert!(input.prompt.contains("User: Hello"));
        assert!(input.prompt.contains("Assistant: Hi there!"));
        assert!(input.prompt.contains("User: How are you?"));
    }

    #[test]
    fn test_input_serialization() {
        let input = ReplicatePredictionInput {
            prompt: "Hello".to_string(),
            max_tokens: Some(1000),
            temperature: Some(0.7),
            top_p: Some(0.9),
            system_prompt: Some("Be helpful".to_string()),
        };

        let json = serde_json::to_string(&input).unwrap();
        assert!(json.contains("\"prompt\":\"Hello\""));
        assert!(json.contains("\"max_tokens\":1000"));
        assert!(json.contains("\"temperature\":0.7"));
    }

    #[test]
    fn test_prediction_deserialization() {
        let json = r#"{
            "id": "abc123",
            "model": "meta/llama-2-70b",
            "status": "succeeded",
            "output": "Hello!",
            "metrics": {
                "input_token_count": 5,
                "output_token_count": 10
            }
        }"#;

        let prediction: ReplicatePrediction = serde_json::from_str(json).unwrap();
        assert_eq!(prediction.id, "abc123");
        assert_eq!(prediction.status, "succeeded");
        assert_eq!(
            prediction.metrics.as_ref().unwrap().input_token_count,
            Some(5)
        );
    }

    #[test]
    fn test_error_handling() {
        let provider = ReplicateProvider::with_api_key("test-key").unwrap();

        // Test 401 -> auth error
        let error = provider.handle_error_response(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"detail": "Invalid API token"}"#,
        );
        assert!(matches!(error, Error::Authentication(_)));

        // Test 404 -> model not found
        let error = provider.handle_error_response(
            reqwest::StatusCode::NOT_FOUND,
            r#"{"detail": "Model not found"}"#,
        );
        assert!(matches!(error, Error::ModelNotFound(_)));

        // Test 422 -> invalid request
        let error = provider.handle_error_response(
            reqwest::StatusCode::UNPROCESSABLE_ENTITY,
            r#"{"detail": "Invalid input"}"#,
        );
        assert!(matches!(error, Error::InvalidRequest(_)));
    }
}
