//! Volcengine (ByteDance) provider for Chinese AI models.
//!
//! Volcengine provides AI models from ByteDance's cloud platform.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Volcengine provider
pub struct VolcengineProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl VolcengineProvider {
    /// Create a new Volcengine provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://ark.cn-beijing.volces.com/api/v3".to_string(),
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
        let api_key = std::env::var("VOLC_ACCESSKEY").map_err(|_| {
            Error::Configuration("VOLC_ACCESSKEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "doubao-pro-256k".to_string(),
            "doubao-pro-128k".to_string(),
            "doubao-pro-32k".to_string(),
            "doubao-lite-128k".to_string(),
            "doubao-lite-32k".to_string(),
            "doubao-vision-pro-32k".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<VolcengineModelInfo> {
        match model {
            "doubao-pro-256k" => Some(VolcengineModelInfo {
                name: model.to_string(),
                context_window: 262144,
                supports_tools: true,
                supports_vision: false,
                max_output_tokens: 16384,
            }),
            "doubao-pro-128k" | "doubao-lite-128k" => Some(VolcengineModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                supports_vision: false,
                max_output_tokens: 8192,
            }),
            "doubao-vision-pro-32k" => Some(VolcengineModelInfo {
                name: model.to_string(),
                context_window: 32768,
                supports_tools: true,
                supports_vision: true,
                max_output_tokens: 4096,
            }),
            _ => None,
        }
    }
}

/// Volcengine model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolcengineModelInfo {
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
        let provider = VolcengineProvider::new("test-key");
        assert!(provider.base_url.contains("volces.com"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = VolcengineProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_get_model_info() {
        let info = VolcengineProvider::get_model_info("doubao-vision-pro-32k").unwrap();
        assert!(info.supports_vision);
    }
}
