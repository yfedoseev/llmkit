//! Moonshot AI (Kimi) provider for Chinese AI models.
//!
//! Moonshot provides the Kimi series of language models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Moonshot AI provider
pub struct MoonshotProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl MoonshotProvider {
    /// Create a new Moonshot provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.moonshot.cn/v1".to_string(),
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
        let api_key = std::env::var("MOONSHOT_API_KEY").map_err(|_| {
            Error::Configuration("MOONSHOT_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "moonshot-v1-8k".to_string(),
            "moonshot-v1-32k".to_string(),
            "moonshot-v1-128k".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<MoonshotModelInfo> {
        match model {
            "moonshot-v1-8k" => Some(MoonshotModelInfo {
                name: model.to_string(),
                context_window: 8192,
                supports_tools: true,
                max_output_tokens: 4096,
            }),
            "moonshot-v1-32k" => Some(MoonshotModelInfo {
                name: model.to_string(),
                context_window: 32768,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            "moonshot-v1-128k" => Some(MoonshotModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            _ => None,
        }
    }
}

/// Moonshot model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoonshotModelInfo {
    pub name: String,
    pub context_window: u32,
    pub supports_tools: bool,
    pub max_output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = MoonshotProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.moonshot.cn/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = MoonshotProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_get_model_info() {
        let info = MoonshotProvider::get_model_info("moonshot-v1-128k").unwrap();
        assert_eq!(info.context_window, 131072);
    }
}
