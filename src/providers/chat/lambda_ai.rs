//! Lambda Labs provider for GPU cloud inference.
//!
//! Lambda provides GPU cloud inference with an OpenAI-compatible API.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Lambda Labs provider
pub struct LambdaProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl LambdaProvider {
    /// Create a new Lambda provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.lambdalabs.com/v1".to_string(),
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
        let api_key = std::env::var("LAMBDA_API_KEY").map_err(|_| {
            Error::Configuration("LAMBDA_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "llama3.3-70b-instruct-fp8".to_string(),
            "llama3.1-405b-instruct-fp8".to_string(),
            "llama3.1-70b-instruct-fp8".to_string(),
            "llama3.1-8b-instruct-fp8".to_string(),
            "hermes-3-llama-3.1-405b-fp8".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<LambdaModelInfo> {
        match model {
            m if m.contains("405b") => Some(LambdaModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            m if m.contains("70b") => Some(LambdaModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            m if m.contains("8b") => Some(LambdaModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            _ => None,
        }
    }
}

/// Lambda model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LambdaModelInfo {
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
        let provider = LambdaProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.lambdalabs.com/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = LambdaProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }
}
