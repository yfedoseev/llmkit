//! Kakao provider for KoGPT models.
//!
//! Kakao's KoGPT is a Korean-focused large language model.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Kakao provider
pub struct KakaoProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl KakaoProvider {
    /// Create a new Kakao provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.kakaobrain.com/v1".to_string(),
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
        let api_key = std::env::var("KAKAO_API_KEY")
            .map_err(|_| Error::Configuration("KAKAO_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["kogpt".to_string(), "kogpt-3".to_string()])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<KakaoModelInfo> {
        match model {
            "kogpt" | "kogpt-3" => Some(KakaoModelInfo {
                name: model.to_string(),
                context_length: 8192,
                language_focus: "Korean".to_string(),
                capabilities: vec!["chat".to_string(), "completion".to_string()],
            }),
            _ => None,
        }
    }
}

/// Kakao model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KakaoModelInfo {
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
        let provider = KakaoProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.kakaobrain.com/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = KakaoProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let info = KakaoProvider::get_model_info("kogpt").unwrap();
        assert_eq!(info.language_focus, "Korean");
    }
}
