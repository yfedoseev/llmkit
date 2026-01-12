//! Baichuan AI provider for Chinese AI models.
//!
//! Baichuan provides Chinese language models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Baichuan AI provider
pub struct BaichuanProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl BaichuanProvider {
    /// Create a new Baichuan provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.baichuan-ai.com/v1".to_string(),
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
        let api_key = std::env::var("BAICHUAN_API_KEY").map_err(|_| {
            Error::Configuration("BAICHUAN_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "Baichuan4".to_string(),
            "Baichuan3-Turbo".to_string(),
            "Baichuan3-Turbo-128k".to_string(),
            "Baichuan2-Turbo".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<BaichuanModelInfo> {
        match model {
            "Baichuan4" => Some(BaichuanModelInfo {
                name: model.to_string(),
                context_window: 128000,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            "Baichuan3-Turbo-128k" => Some(BaichuanModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            "Baichuan3-Turbo" | "Baichuan2-Turbo" => Some(BaichuanModelInfo {
                name: model.to_string(),
                context_window: 32768,
                supports_tools: true,
                max_output_tokens: 4096,
            }),
            _ => None,
        }
    }
}

/// Baichuan model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaichuanModelInfo {
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
        let provider = BaichuanProvider::new("test-key");
        assert!(provider.base_url.contains("baichuan-ai.com"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = BaichuanProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_get_model_info() {
        let info = BaichuanProvider::get_model_info("Baichuan4").unwrap();
        assert_eq!(info.context_window, 128000);
    }
}
