//! Naver Cloud provider for HyperCLOVA X models.
//!
//! Naver's HyperCLOVA X is a Korean-optimized large language model.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Naver Cloud provider
pub struct NaverProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl NaverProvider {
    /// Create a new Naver provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://clovastudio.stream.ntruss.com".to_string(),
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
        let api_key = std::env::var("NAVER_API_KEY")
            .map_err(|_| Error::Configuration("NAVER_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "HCX-003".to_string(),
            "HCX-DASH-001".to_string(),
            "HCX-002".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<NaverModelInfo> {
        match model {
            "HCX-003" => Some(NaverModelInfo {
                name: model.to_string(),
                context_length: 128000,
                language_focus: "Korean, English".to_string(),
                capabilities: vec!["chat".to_string(), "completion".to_string()],
            }),
            "HCX-DASH-001" => Some(NaverModelInfo {
                name: model.to_string(),
                context_length: 32000,
                language_focus: "Korean, English".to_string(),
                capabilities: vec!["chat".to_string(), "fast-inference".to_string()],
            }),
            "HCX-002" => Some(NaverModelInfo {
                name: model.to_string(),
                context_length: 32000,
                language_focus: "Korean, English".to_string(),
                capabilities: vec!["chat".to_string()],
            }),
            _ => None,
        }
    }
}

/// Naver model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NaverModelInfo {
    pub name: String,
    pub context_length: u32,
    pub language_focus: String,
    pub capabilities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = NaverProvider::new("test-key");
        assert_eq!(provider.base_url, "https://clovastudio.stream.ntruss.com");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = NaverProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 2);
    }

    #[test]
    fn test_model_info() {
        let info = NaverProvider::get_model_info("HCX-003").unwrap();
        assert_eq!(info.context_length, 128000);
    }
}
