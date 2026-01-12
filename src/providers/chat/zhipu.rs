//! Zhipu AI (GLM) provider for Chinese AI models.
//!
//! Zhipu provides the GLM series of language models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Zhipu AI provider
pub struct ZhipuProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl ZhipuProvider {
    /// Create a new Zhipu provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://open.bigmodel.cn/api/paas/v4".to_string(),
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
        let api_key = std::env::var("ZHIPU_API_KEY").map_err(|_| {
            Error::Configuration("ZHIPU_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "glm-4-plus".to_string(),
            "glm-4".to_string(),
            "glm-4-air".to_string(),
            "glm-4-airx".to_string(),
            "glm-4-flash".to_string(),
            "glm-4v-plus".to_string(),
            "glm-4v".to_string(),
            "glm-4-long".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<ZhipuModelInfo> {
        match model {
            "glm-4-plus" | "glm-4" => Some(ZhipuModelInfo {
                name: model.to_string(),
                context_window: 128000,
                supports_tools: true,
                supports_vision: false,
                max_output_tokens: 8192,
            }),
            "glm-4-long" => Some(ZhipuModelInfo {
                name: model.to_string(),
                context_window: 1000000,
                supports_tools: true,
                supports_vision: false,
                max_output_tokens: 8192,
            }),
            "glm-4v-plus" | "glm-4v" => Some(ZhipuModelInfo {
                name: model.to_string(),
                context_window: 8192,
                supports_tools: true,
                supports_vision: true,
                max_output_tokens: 4096,
            }),
            "glm-4-flash" => Some(ZhipuModelInfo {
                name: model.to_string(),
                context_window: 128000,
                supports_tools: true,
                supports_vision: false,
                max_output_tokens: 4096,
            }),
            _ => None,
        }
    }
}

/// Zhipu model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZhipuModelInfo {
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
        let provider = ZhipuProvider::new("test-key");
        assert!(provider.base_url.contains("bigmodel.cn"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = ZhipuProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_get_model_info() {
        let info = ZhipuProvider::get_model_info("glm-4v-plus").unwrap();
        assert!(info.supports_vision);
    }
}
