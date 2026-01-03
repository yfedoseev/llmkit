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

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Baidu Wenxin provider for Chinese LLM services
pub struct BaiduProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    secret_key: String,
}

impl BaiduProvider {
    /// Create a new Baidu provider with API credentials
    pub fn new(api_key: &str, secret_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            secret_key: secret_key.to_string(),
        }
    }

    /// Create from environment variables `BAIDU_API_KEY` and `BAIDU_SECRET_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("BAIDU_API_KEY").map_err(|_| {
            Error::Configuration("BAIDU_API_KEY environment variable not set".to_string())
        })?;
        let secret_key = std::env::var("BAIDU_SECRET_KEY").map_err(|_| {
            Error::Configuration("BAIDU_SECRET_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key, &secret_key))
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
        let provider = BaiduProvider::new("test-key", "test-secret");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.secret_key, "test-secret");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = BaiduProvider::new("test-key", "test-secret");
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
