//! xAI (Grok) provider for conversational AI.
//!
//! xAI provides the Grok family of large language models with
//! advanced reasoning and real-time knowledge capabilities.
//!
//! # Supported Models
//!
//! - `grok-4-1-fast-reasoning` - Latest fast model with reasoning, 2M context, $0.20/$0.50
//! - `grok-4-1-fast-non-reasoning` - Latest fast model without reasoning, 2M context, $0.20/$0.50
//! - `grok-code-fast-1` - Code-optimized model, 256K context, $0.20/$1.50
//! - `grok-4-fast-reasoning` - Fast model with reasoning, 2M context, $0.20/$0.50
//! - `grok-4-fast-non-reasoning` - Fast model without reasoning, 2M context, $0.20/$0.50
//! - `grok-4-0709` - Flagship reasoning model, 256K context, $3/$15
//! - `grok-3-mini` - Lightweight thinking model, 131K context, $0.30/$0.50
//! - `grok-3` - Previous flagship model, 131K context, $3/$15
//! - `grok-2-vision-1212` - Vision-capable model, 32K context, $2/$10
//! - `grok-2-image-1212` - Image generation, $0.07 per image
//!
//! # Features
//! - OpenAI-compatible API
//! - Grok model family with advanced reasoning
//! - Live Search for real-time knowledge
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
            "grok-4-1-fast-reasoning".to_string(),
            "grok-4-1-fast-non-reasoning".to_string(),
            "grok-code-fast-1".to_string(),
            "grok-4-fast-reasoning".to_string(),
            "grok-4-fast-non-reasoning".to_string(),
            "grok-4-0709".to_string(),
            "grok-3-mini".to_string(),
            "grok-3".to_string(),
            "grok-2-vision-1212".to_string(),
            "grok-2-image-1212".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<XAIModelInfo> {
        match model {
            "grok-4-1-fast-reasoning" | "grok-4-1-fast-non-reasoning" => Some(XAIModelInfo {
                name: model.to_string(),
                context_window: 2_000_000,
                supports_vision: true,
                supports_tools: true,
                max_output_tokens: 30000,
            }),
            "grok-code-fast-1" => Some(XAIModelInfo {
                name: "grok-code-fast-1".to_string(),
                context_window: 256_000,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 10000,
            }),
            "grok-4-fast-reasoning" | "grok-4-fast-non-reasoning" => Some(XAIModelInfo {
                name: model.to_string(),
                context_window: 2_000_000,
                supports_vision: true,
                supports_tools: true,
                max_output_tokens: 30000,
            }),
            "grok-4-0709" => Some(XAIModelInfo {
                name: "grok-4-0709".to_string(),
                context_window: 256_000,
                supports_vision: true,
                supports_tools: true,
                max_output_tokens: 64000,
            }),
            "grok-3-mini" => Some(XAIModelInfo {
                name: "grok-3-mini".to_string(),
                context_window: 131_072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 32768,
            }),
            "grok-3" => Some(XAIModelInfo {
                name: "grok-3".to_string(),
                context_window: 131_072,
                supports_vision: false,
                supports_tools: true,
                max_output_tokens: 32768,
            }),
            "grok-2-vision-1212" => Some(XAIModelInfo {
                name: "grok-2-vision-1212".to_string(),
                context_window: 32_768,
                supports_vision: true,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            "grok-2-image-1212" => Some(XAIModelInfo {
                name: "grok-2-image-1212".to_string(),
                context_window: 1024,
                supports_vision: false,
                supports_tools: false,
                max_output_tokens: 1,
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
        assert!(models.contains(&"grok-4-1-fast-reasoning".to_string()));
        assert!(models.contains(&"grok-3".to_string()));
    }

    #[test]
    fn test_get_model_info() {
        let info = XAIProvider::get_model_info("grok-4-0709").unwrap();
        assert_eq!(info.name, "grok-4-0709");
        assert!(info.supports_tools);
        assert!(info.supports_vision);
    }

    #[test]
    fn test_get_model_info_vision() {
        let info = XAIProvider::get_model_info("grok-2-vision-1212").unwrap();
        assert!(info.supports_vision);
    }

    #[test]
    fn test_model_info_invalid() {
        let info = XAIProvider::get_model_info("invalid-model");
        assert!(info.is_none());
    }
}
