//! xAI (Grok) provider for conversational AI.
//!
//! xAI provides the Grok family of large language models with
//! advanced reasoning and real-time knowledge capabilities.
//!
//! # Features
//! - OpenAI-compatible API
//! - Grok model family (grok-2, grok-2-mini, etc.)
//! - Streaming responses
//! - Function calling support

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// xAI provider for Grok models
pub struct XAIProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl XAIProvider {
    /// Create a new xAI provider with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.x.ai/v1".to_string(),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(api_key: &str, base_url: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        }
    }

    /// Create from environment variable `XAI_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("XAI_API_KEY").map_err(|_| {
            Error::Configuration("XAI_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "grok-2-1212".to_string(),
            "grok-2-vision-1212".to_string(),
            "grok-3".to_string(),
            "grok-3-mini".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<XAIModelInfo> {
        match model {
            "grok-2-1212" | "grok-2" => Some(XAIModelInfo {
                name: "grok-2-1212".to_string(),
                context_window: 131072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 32768,
            }),
            "grok-2-vision-1212" | "grok-2-vision" => Some(XAIModelInfo {
                name: "grok-2-vision-1212".to_string(),
                context_window: 32768,
                supports_vision: true,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            "grok-3" => Some(XAIModelInfo {
                name: "grok-3".to_string(),
                context_window: 131072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 32768,
            }),
            "grok-3-mini" => Some(XAIModelInfo {
                name: "grok-3-mini".to_string(),
                context_window: 131072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 32768,
            }),
            _ => None,
        }
    }
}

/// xAI model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XAIModelInfo {
    /// Model name
    pub name: String,
    /// Context window size
    pub context_window: u32,
    /// Whether this model supports vision
    pub supports_vision: bool,
    /// Whether this model supports function calling
    pub supports_tools: bool,
    /// Maximum output tokens
    pub max_output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xai_provider_creation() {
        let provider = XAIProvider::new("test-key");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.base_url, "https://api.x.ai/v1");
    }

    #[test]
    fn test_xai_custom_base_url() {
        let provider = XAIProvider::with_base_url("test-key", "https://custom.api/v1");
        assert_eq!(provider.base_url, "https://custom.api/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = XAIProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.contains(&"grok-2-1212".to_string()));
    }

    #[test]
    fn test_get_model_info() {
        let info = XAIProvider::get_model_info("grok-2").unwrap();
        assert_eq!(info.name, "grok-2-1212");
        assert!(info.supports_tools);
        assert!(!info.supports_vision);
    }

    #[test]
    fn test_get_model_info_vision() {
        let info = XAIProvider::get_model_info("grok-2-vision").unwrap();
        assert!(info.supports_vision);
    }

    #[test]
    fn test_model_info_invalid() {
        let info = XAIProvider::get_model_info("invalid-model");
        assert!(info.is_none());
    }
}
