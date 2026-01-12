//! Spark (iFlytek) provider for Chinese AI models.
//!
//! Spark provides the Spark series of language models from iFlytek.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Spark AI provider
pub struct SparkProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl SparkProvider {
    /// Create a new Spark provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://spark-api-open.xf-yun.com/v1".to_string(),
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
        let api_key = std::env::var("SPARK_API_KEY").map_err(|_| {
            Error::Configuration("SPARK_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "spark-4.0-ultra".to_string(),
            "spark-pro-128k".to_string(),
            "spark-max-32k".to_string(),
            "spark-pro".to_string(),
            "spark-lite".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<SparkModelInfo> {
        match model {
            "spark-4.0-ultra" => Some(SparkModelInfo {
                name: model.to_string(),
                context_window: 8192,
                supports_tools: true,
                max_output_tokens: 4096,
            }),
            "spark-pro-128k" => Some(SparkModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            "spark-max-32k" => Some(SparkModelInfo {
                name: model.to_string(),
                context_window: 32768,
                supports_tools: true,
                max_output_tokens: 4096,
            }),
            "spark-pro" | "spark-lite" => Some(SparkModelInfo {
                name: model.to_string(),
                context_window: 8192,
                supports_tools: true,
                max_output_tokens: 4096,
            }),
            _ => None,
        }
    }
}

/// Spark model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparkModelInfo {
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
        let provider = SparkProvider::new("test-key");
        assert!(provider.base_url.contains("xf-yun.com"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = SparkProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_get_model_info() {
        let info = SparkProvider::get_model_info("spark-pro-128k").unwrap();
        assert_eq!(info.context_window, 131072);
    }
}
