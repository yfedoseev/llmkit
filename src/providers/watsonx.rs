#![allow(dead_code)]
//! IBM watsonx.ai provider implementation.
//!
//! This module provides access to IBM's watsonx.ai foundation models.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::WatsonxProvider;
//!
//! // From environment variables
//! let provider = WatsonxProvider::from_env()?;
//!
//! // With explicit credentials
//! let provider = WatsonxProvider::new("api-key", "project-id")?;
//! ```
//!
//! # Supported Models
//!
//! - `ibm/granite-13b-chat-v2`
//! - `ibm/granite-20b-multilingual`
//! - `meta-llama/llama-3-70b-instruct`
//! - `mistralai/mixtral-8x7b-instruct-v01`
//!
//! # Environment Variables
//!
//! - `WATSONX_API_KEY` - Your IBM Cloud API key
//! - `WATSONX_PROJECT_ID` - Your watsonx.ai project ID
//! - `WATSONX_URL` - Optional: API URL (defaults to us-south)

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

const WATSONX_API_URL: &str = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation";
const WATSONX_STREAM_URL: &str = "https://us-south.ml.cloud.ibm.com/ml/v1/text/generation_stream";
const IAM_TOKEN_URL: &str = "https://iam.cloud.ibm.com/identity/token";

/// IBM watsonx.ai provider.
///
/// Provides access to IBM's foundation models through watsonx.ai.
pub struct WatsonxProvider {
    config: ProviderConfig,
    client: Client,
    project_id: String,
    api_url: String,
    stream_url: String,
}

impl WatsonxProvider {
    /// Create provider from environment variables.
    ///
    /// Reads: `WATSONX_API_KEY`, `WATSONX_PROJECT_ID`, and optionally `WATSONX_URL`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("WATSONX_API_KEY").ok();
        let project_id = std::env::var("WATSONX_PROJECT_ID")
            .map_err(|_| Error::config("WATSONX_PROJECT_ID environment variable not set"))?;

        let base_url = std::env::var("WATSONX_URL").ok();

        let config = ProviderConfig {
            api_key,
            base_url,
            ..Default::default()
        };

        Self::with_config(project_id, config)
    }

    /// Create provider with API key and project ID.
    pub fn new(api_key: impl Into<String>, project_id: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::with_config(project_id, config)
    }

    /// Create provider with custom URL.
    pub fn with_url(
        api_key: impl Into<String>,
        project_id: impl Into<String>,
        url: impl Into<String>,
    ) -> Result<Self> {
        let mut config = ProviderConfig::new(api_key);
        config.base_url = Some(url.into());
        Self::with_config(project_id, config)
    }

    /// Create provider with custom config.
    fn with_config(project_id: impl Into<String>, config: ProviderConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        // watsonx uses Bearer token authentication
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

        let (api_url, stream_url) = if let Some(ref base) = config.base_url {
            (
                format!("{}/ml/v1/text/generation", base),
                format!("{}/ml/v1/text/generation_stream", base),
            )
        } else {
            (WATSONX_API_URL.to_string(), WATSONX_STREAM_URL.to_string())
        };

        Ok(Self {
            config,
            client,
            project_id: project_id.into(),
            api_url,
            stream_url,
        })
    }

    /// Build prompt from messages.
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

            prompt.push_str(&format!("{}\n{}\n", role_tag, msg.text_content()));
        }

        prompt.push_str("<|assistant|>\n");
        prompt
    }

    /// Convert unified request to watsonx format.
    fn convert_request(&self, request: &CompletionRequest) -> WatsonxRequest {
        let input = self.build_prompt(request);

        WatsonxRequest {
            model_id: request.model.clone(),
            input,
            project_id: self.project_id.clone(),
            parameters: WatsonxParameters {
                max_new_tokens: request.max_tokens,
                temperature: request.temperature,
                top_p: request.top_p,
                decoding_method: request.temperature.map(|t| {
                    if t > 0.0 {
                        "sample".to_string()
                    } else {
                        "greedy".to_string()
                    }
                }),
                stop_sequences: request.stop_sequences.clone(),
            },
        }
    }

    /// Convert watsonx response to unified format.
    fn convert_response(&self, response: WatsonxResponse, model: &str) -> CompletionResponse {
        let mut content = Vec::new();
        let mut usage = Usage::default();
        let mut stop_reason = StopReason::EndTurn;

        if let Some(result) = response.results.into_iter().next() {
            if !result.generated_text.is_empty() {
                content.push(ContentBlock::Text {
                    text: result.generated_text,
                });
            }

            usage = Usage {
                input_tokens: result.input_token_count.unwrap_or(0),
                output_tokens: result.generated_token_count.unwrap_or(0),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            };

            stop_reason = match result.stop_reason.as_deref() {
                Some("max_tokens") => StopReason::MaxTokens,
                Some("eos_token") | Some("stop_sequence") => StopReason::EndTurn,
                _ => StopReason::EndTurn,
            };
        }

        CompletionResponse {
            id: response.model_id.clone(),
            model: model.to_string(),
            content,
            stop_reason,
            usage,
        }
    }

    /// Handle error responses from watsonx API.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        if let Ok(error_resp) = serde_json::from_str::<WatsonxErrorResponse>(body) {
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
impl Provider for WatsonxProvider {
    fn name(&self) -> &str {
        "watsonx"
    }

    fn default_model(&self) -> Option<&str> {
        Some("ibm/granite-13b-chat-v2")
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let wx_request = self.convert_request(&request);

        let response = self
            .client
            .post(&self.api_url)
            .query(&[("version", "2024-05-31")])
            .json(&wx_request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let wx_response: WatsonxResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {} - {}", e, body)))?;

        Ok(self.convert_response(wx_response, &request.model))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let wx_request = self.convert_request(&request);

        let response = self
            .client
            .post(&self.stream_url)
            .query(&[("version", "2024-05-31")])
            .json(&wx_request)
            .send()
            .await?;

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

                // Process lines
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

                    if let Ok(chunk_resp) = serde_json::from_str::<WatsonxStreamChunk>(data) {
                        if let Some(result) = chunk_resp.results.into_iter().next() {
                            if !result.generated_text.is_empty() {
                                yield StreamChunk {
                                    event_type: StreamEventType::ContentBlockDelta,
                                    index: Some(chunk_index),
                                    delta: Some(ContentDelta::TextDelta {
                                        text: result.generated_text,
                                    }),
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
struct WatsonxRequest {
    model_id: String,
    input: String,
    project_id: String,
    parameters: WatsonxParameters,
}

#[derive(Debug, Serialize)]
struct WatsonxParameters {
    #[serde(skip_serializing_if = "Option::is_none")]
    max_new_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    decoding_method: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct WatsonxResponse {
    model_id: String,
    results: Vec<WatsonxResult>,
}

#[derive(Debug, Deserialize)]
struct WatsonxResult {
    generated_text: String,
    generated_token_count: Option<u32>,
    input_token_count: Option<u32>,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct WatsonxStreamChunk {
    results: Vec<WatsonxStreamResult>,
}

#[derive(Debug, Deserialize)]
struct WatsonxStreamResult {
    generated_text: String,
}

#[derive(Debug, Deserialize)]
struct WatsonxErrorResponse {
    errors: Vec<WatsonxError>,
}

#[derive(Debug, Deserialize)]
struct WatsonxError {
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_provider_creation() {
        let provider = WatsonxProvider::new("test-key", "project-123").unwrap();
        assert_eq!(provider.name(), "watsonx");
        assert!(!provider.supports_tools());
        assert!(!provider.supports_vision());
        assert!(provider.supports_streaming());
    }

    #[test]
    fn test_default_model() {
        let provider = WatsonxProvider::new("test-key", "project-123").unwrap();
        assert_eq!(provider.default_model(), Some("ibm/granite-13b-chat-v2"));
    }

    #[test]
    fn test_prompt_building() {
        let provider = WatsonxProvider::new("test-key", "project-123").unwrap();

        let request =
            CompletionRequest::new("ibm/granite-13b-chat-v2", vec![Message::user("Hello")])
                .with_system("You are helpful");

        let wx_req = provider.convert_request(&request);

        assert!(wx_req.input.contains("<|system|>"));
        assert!(wx_req.input.contains("You are helpful"));
        assert!(wx_req.input.contains("<|user|>"));
        assert!(wx_req.input.contains("Hello"));
        assert!(wx_req.input.ends_with("<|assistant|>\n"));
    }

    #[test]
    fn test_custom_url() {
        let provider =
            WatsonxProvider::with_url("test-key", "project-123", "https://eu-de.ml.cloud.ibm.com")
                .unwrap();
        assert!(provider.api_url.contains("eu-de"));
        assert!(provider.stream_url.contains("eu-de"));
    }

    #[test]
    fn test_request_parameters() {
        let provider = WatsonxProvider::new("test-key", "project-123").unwrap();

        let request =
            CompletionRequest::new("ibm/granite-13b-chat-v2", vec![Message::user("Hello")])
                .with_max_tokens(1024)
                .with_temperature(0.7)
                .with_top_p(0.9)
                .with_stop_sequences(vec!["STOP".to_string()]);

        let wx_req = provider.convert_request(&request);

        assert_eq!(wx_req.model_id, "ibm/granite-13b-chat-v2");
        assert_eq!(wx_req.project_id, "project-123");
        assert_eq!(wx_req.parameters.max_new_tokens, Some(1024));
        assert_eq!(wx_req.parameters.temperature, Some(0.7));
        assert_eq!(wx_req.parameters.top_p, Some(0.9));
        assert_eq!(
            wx_req.parameters.stop_sequences,
            Some(vec!["STOP".to_string()])
        );
        // Temperature > 0 should set decoding_method to "sample"
        assert_eq!(
            wx_req.parameters.decoding_method,
            Some("sample".to_string())
        );
    }

    #[test]
    fn test_decoding_method_greedy() {
        let provider = WatsonxProvider::new("test-key", "project-123").unwrap();

        let request =
            CompletionRequest::new("ibm/granite-13b-chat-v2", vec![Message::user("Hello")])
                .with_temperature(0.0); // Temperature 0 should use greedy

        let wx_req = provider.convert_request(&request);
        assert_eq!(
            wx_req.parameters.decoding_method,
            Some("greedy".to_string())
        );
    }

    #[test]
    fn test_response_parsing() {
        let provider = WatsonxProvider::new("test-key", "project-123").unwrap();

        let response = WatsonxResponse {
            model_id: "ibm/granite-13b-chat-v2".to_string(),
            results: vec![WatsonxResult {
                generated_text: "Hello there!".to_string(),
                generated_token_count: Some(20),
                input_token_count: Some(10),
                stop_reason: Some("eos_token".to_string()),
            }],
        };

        let result = provider.convert_response(response, "ibm/granite-13b-chat-v2");

        assert_eq!(result.model, "ibm/granite-13b-chat-v2");
        assert_eq!(result.content.len(), 1);
        if let ContentBlock::Text { text } = &result.content[0] {
            assert_eq!(text, "Hello there!");
        } else {
            panic!("Expected text content");
        }
        assert!(matches!(result.stop_reason, StopReason::EndTurn));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 20);
    }

    #[test]
    fn test_stop_reason_mapping() {
        let provider = WatsonxProvider::new("test-key", "project-123").unwrap();

        // Test "max_tokens" -> MaxTokens
        let response1 = WatsonxResponse {
            model_id: "model".to_string(),
            results: vec![WatsonxResult {
                generated_text: "Truncated".to_string(),
                generated_token_count: None,
                input_token_count: None,
                stop_reason: Some("max_tokens".to_string()),
            }],
        };
        assert!(matches!(
            provider.convert_response(response1, "model").stop_reason,
            StopReason::MaxTokens
        ));

        // Test "stop_sequence" -> EndTurn
        let response2 = WatsonxResponse {
            model_id: "model".to_string(),
            results: vec![WatsonxResult {
                generated_text: "Done".to_string(),
                generated_token_count: None,
                input_token_count: None,
                stop_reason: Some("stop_sequence".to_string()),
            }],
        };
        assert!(matches!(
            provider.convert_response(response2, "model").stop_reason,
            StopReason::EndTurn
        ));
    }

    #[test]
    fn test_request_serialization() {
        let request = WatsonxRequest {
            model_id: "ibm/granite-13b-chat-v2".to_string(),
            input: "Hello".to_string(),
            project_id: "proj-123".to_string(),
            parameters: WatsonxParameters {
                max_new_tokens: Some(1000),
                temperature: Some(0.7),
                top_p: Some(0.9),
                decoding_method: Some("sample".to_string()),
                stop_sequences: None,
            },
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("ibm/granite-13b-chat-v2"));
        assert!(json.contains("proj-123"));
        assert!(json.contains("\"max_new_tokens\":1000"));
    }

    #[test]
    fn test_response_deserialization() {
        let json = r#"{
            "model_id": "ibm/granite-13b-chat-v2",
            "results": [{
                "generated_text": "Hi!",
                "generated_token_count": 10,
                "input_token_count": 5,
                "stop_reason": "eos_token"
            }]
        }"#;

        let response: WatsonxResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.model_id, "ibm/granite-13b-chat-v2");
        assert_eq!(response.results.len(), 1);
        assert_eq!(response.results[0].generated_text, "Hi!");
    }

    #[test]
    fn test_multi_turn_conversation() {
        let provider = WatsonxProvider::new("test-key", "project-123").unwrap();

        let request = CompletionRequest::new(
            "ibm/granite-13b-chat-v2",
            vec![
                Message::user("Hello"),
                Message::assistant("Hi there!"),
                Message::user("How are you?"),
            ],
        )
        .with_system("Be friendly");

        let wx_req = provider.convert_request(&request);

        assert!(wx_req.input.contains("<|system|>"));
        assert!(wx_req.input.contains("<|user|>"));
        assert!(wx_req.input.contains("<|assistant|>"));
        assert!(wx_req.input.contains("Hello"));
        assert!(wx_req.input.contains("Hi there!"));
        assert!(wx_req.input.contains("How are you?"));
    }

    #[test]
    fn test_error_handling() {
        let provider = WatsonxProvider::new("test-key", "project-123").unwrap();

        // Test 401 -> auth error
        let error = provider.handle_error_response(
            reqwest::StatusCode::UNAUTHORIZED,
            r#"{"errors": [{"message": "Invalid API key"}]}"#,
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
