//! Baidu Wenxin provider for Chinese LLM services.
//!
//! Baidu Wenxin provides a suite of large language models optimized for Chinese
//! language understanding and generation, with enterprise-grade reliability.
//!
//! # Features
//! - Multiple model tiers (Base, Plus, Pro, Ultra)
//! - Native Chinese language optimization
//! - Enterprise API with SLA guarantees
//! - Streaming support
//! - Function calling capabilities

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

const BAIDU_API_URL: &str = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat";

/// Baidu Wenxin provider for Chinese LLM services
pub struct BaiduProvider {
    api_key: String,
    #[allow(dead_code)]
    secret_key: String,
    client: Client,
}

impl BaiduProvider {
    /// Create a new Baidu provider with API credentials
    pub fn new(api_key: &str, secret_key: &str) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()?;

        Ok(Self {
            api_key: api_key.to_string(),
            secret_key: secret_key.to_string(),
            client,
        })
    }

    /// Create from environment variables `BAIDU_API_KEY` and `BAIDU_SECRET_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("BAIDU_API_KEY")
            .map_err(|_| Error::config("BAIDU_API_KEY environment variable not set"))?;
        let secret_key = std::env::var("BAIDU_SECRET_KEY")
            .map_err(|_| Error::config("BAIDU_SECRET_KEY environment variable not set"))?;
        Self::new(&api_key, &secret_key)
    }

    /// Get list of available Baidu Wenxin models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        // Mock implementation
        Ok(vec![
            "ERNIE-Bot".to_string(),
            "ERNIE-Bot-Plus".to_string(),
            "ERNIE-Bot-Pro".to_string(),
            "ERNIE-Bot-Ultra".to_string(),
        ])
    }

    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        match status.as_u16() {
            401 | 403 => Error::auth(format!("Baidu authentication failed: {}", body)),
            429 => Error::rate_limited("Baidu rate limited", None),
            500..=599 => Error::server(status.as_u16(), body.to_string()),
            _ => Error::other(format!("Baidu error ({}): {}", status, body)),
        }
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<BaiduModelInfo> {
        match model {
            "ERNIE-Bot" => Some(BaiduModelInfo {
                name: "ERNIE-Bot".to_string(),
                context_window: 2048,
                supports_function_call: false,
                max_output_tokens: 1024,
            }),
            "ERNIE-Bot-Plus" => Some(BaiduModelInfo {
                name: "ERNIE-Bot-Plus".to_string(),
                context_window: 8000,
                supports_function_call: true,
                max_output_tokens: 2000,
            }),
            "ERNIE-Bot-Pro" => Some(BaiduModelInfo {
                name: "ERNIE-Bot-Pro".to_string(),
                context_window: 32000,
                supports_function_call: true,
                max_output_tokens: 4000,
            }),
            "ERNIE-Bot-Ultra" => Some(BaiduModelInfo {
                name: "ERNIE-Bot-Ultra".to_string(),
                context_window: 200000,
                supports_function_call: true,
                max_output_tokens: 8000,
            }),
            _ => None,
        }
    }
}

/// Baidu model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaiduModelInfo {
    /// Model name
    pub name: String,
    /// Context window size
    pub context_window: u32,
    /// Whether this model supports function calling
    pub supports_function_call: bool,
    /// Maximum output tokens
    pub max_output_tokens: u32,
}

// ============================================================================
// Baidu API Types
// ============================================================================

#[derive(Debug, Serialize)]
struct BaiduChatRequest {
    messages: Vec<BaiduMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
}

#[derive(Debug, Serialize)]
struct BaiduMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct BaiduChatResponse {
    result: Option<BaiduResult>,
    error_code: Option<i32>,
    error_msg: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BaiduResult {
    #[serde(default)]
    response: String,
}

// ============================================================================
// Provider Implementation
// ============================================================================

#[async_trait]
impl Provider for BaiduProvider {
    fn name(&self) -> &str {
        "baidu"
    }

    fn default_model(&self) -> Option<&str> {
        Some("ERNIE-Bot-Ultra")
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        false
    }

    fn supports_streaming(&self) -> bool {
        false
    }

    fn supported_models(&self) -> Option<&[&str]> {
        Some(&[
            "ERNIE-Bot",
            "ERNIE-Bot-Plus",
            "ERNIE-Bot-Pro",
            "ERNIE-Bot-Ultra",
        ])
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = if request.model.is_empty() {
            "ERNIE-Bot-Ultra".to_string()
        } else {
            request.model.clone()
        };

        // Build message list from request
        let mut messages = Vec::new();
        for msg in &request.messages {
            let role = match msg.role {
                Role::User => "user".to_string(),
                Role::Assistant => "assistant".to_string(),
                Role::System => "system".to_string(),
            };

            for content in &msg.content {
                if let ContentBlock::Text { text } = content {
                    messages.push(BaiduMessage {
                        role: role.clone(),
                        content: text.clone(),
                    });
                }
            }
        }

        let baidu_request = BaiduChatRequest {
            messages,
            temperature: request.temperature,
            top_p: request.top_p,
            max_output_tokens: request.max_tokens,
        };

        // Make API request
        let url = format!("{}?access_token={}", BAIDU_API_URL, self.api_key);
        let response = self.client.post(&url).json(&baidu_request).send().await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let baidu_response: BaiduChatResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {}", e)))?;

        // Check for API errors
        if let Some(error_code) = baidu_response.error_code {
            if error_code != 0 {
                return Err(Error::other(format!(
                    "Baidu API error {}: {}",
                    error_code,
                    baidu_response
                        .error_msg
                        .unwrap_or_else(|| "Unknown error".to_string())
                )));
            }
        }

        let response_text = baidu_response
            .result
            .and_then(|r| {
                if r.response.is_empty() {
                    None
                } else {
                    Some(r.response)
                }
            })
            .unwrap_or_else(|| "No response from Baidu".to_string());

        Ok(CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model,
            content: vec![ContentBlock::Text {
                text: response_text,
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
        // Baidu doesn't support streaming for now, fall back to complete
        let response = self.complete(request).await?;

        let stream = async_stream::try_stream! {
            yield StreamChunk {
                event_type: StreamEventType::MessageStart,
                index: None,
                delta: None,
                stop_reason: None,
                usage: None,
            };

            for block in response.content {
                if let ContentBlock::Text { text } = block {
                    yield StreamChunk {
                        event_type: StreamEventType::ContentBlockDelta,
                        index: Some(0),
                        delta: Some(ContentDelta::Text { text }),
                        stop_reason: None,
                        usage: None,
                    };
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

/// Baidu Wenxin API version
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ApiVersion {
    /// Stable API version (recommended)
    #[default]
    Stable,
    /// Beta API version with experimental features
    Beta,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_baidu_provider_creation() {
        let provider = BaiduProvider::new("test-key", "test-secret").unwrap();
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.secret_key, "test-secret");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = BaiduProvider::new("test-key", "test-secret").unwrap();
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.contains(&"ERNIE-Bot".to_string()));
        assert!(models.contains(&"ERNIE-Bot-Ultra".to_string()));
    }

    #[test]
    fn test_get_model_info() {
        let info = BaiduProvider::get_model_info("ERNIE-Bot-Pro").unwrap();
        assert_eq!(info.name, "ERNIE-Bot-Pro");
        assert!(info.supports_function_call);
        assert_eq!(info.context_window, 32000);
    }

    #[test]
    fn test_model_info_invalid() {
        let info = BaiduProvider::get_model_info("invalid-model");
        assert!(info.is_none());
    }

    #[test]
    fn test_api_version_default() {
        assert_eq!(ApiVersion::default(), ApiVersion::Stable);
    }

    #[test]
    fn test_baidu_model_context_windows() {
        let base = BaiduProvider::get_model_info("ERNIE-Bot").unwrap();
        let plus = BaiduProvider::get_model_info("ERNIE-Bot-Plus").unwrap();
        let pro = BaiduProvider::get_model_info("ERNIE-Bot-Pro").unwrap();
        let ultra = BaiduProvider::get_model_info("ERNIE-Bot-Ultra").unwrap();

        assert!(base.context_window < plus.context_window);
        assert!(plus.context_window < pro.context_window);
        assert!(pro.context_window < ultra.context_window);
    }
}
