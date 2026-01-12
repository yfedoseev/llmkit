//! Yi (LingYi WanWu) provider for Chinese AI models.
//!
//! Yi provides the Yi series of language models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Yi AI provider
pub struct YiProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl YiProvider {
    /// Create a new Yi provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.lingyiwanwu.com/v1".to_string(),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(api_key: &str, base_url: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: base_url.to_string(),
        }
    }

    /// Create from environment variable
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("YI_API_KEY").map_err(|_| {
            Error::Configuration("YI_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "yi-lightning".to_string(),
            "yi-large".to_string(),
            "yi-large-turbo".to_string(),
            "yi-medium".to_string(),
            "yi-medium-200k".to_string(),
            "yi-spark".to_string(),
            "yi-vision".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<YiModelInfo> {
        match model {
            "yi-lightning" | "yi-large" => Some(YiModelInfo {
                name: model.to_string(),
                context_window: 16384,
                supports_tools: true,
                supports_vision: false,
                max_output_tokens: 8192,
            }),
            "yi-medium-200k" => Some(YiModelInfo {
                name: model.to_string(),
                context_window: 204800,
                supports_tools: true,
                supports_vision: false,
                max_output_tokens: 16384,
            }),
            "yi-vision" => Some(YiModelInfo {
                name: model.to_string(),
                context_window: 16384,
                supports_tools: true,
                supports_vision: true,
                max_output_tokens: 4096,
            }),
            _ => None,
        }
    }
}

/// Yi model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct YiModelInfo {
    pub name: String,
    pub context_window: u32,
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub max_output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = YiProvider::new("test-key");
        assert!(provider.base_url.contains("lingyiwanwu.com"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = YiProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_get_model_info() {
        let info = YiProvider::get_model_info("yi-vision").unwrap();
        assert!(info.supports_vision);
    }
}
