//! Stepfun AI provider for Chinese AI models.
//!
//! Stepfun provides Chinese language models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Stepfun AI provider
pub struct StepfunProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl StepfunProvider {
    /// Create a new Stepfun provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.stepfun.com/v1".to_string(),
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
        let api_key = std::env::var("STEPFUN_API_KEY").map_err(|_| {
            Error::Configuration("STEPFUN_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "step-2-16k".to_string(),
            "step-1-256k".to_string(),
            "step-1-128k".to_string(),
            "step-1-32k".to_string(),
            "step-1v-32k".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<StepfunModelInfo> {
        match model {
            "step-2-16k" => Some(StepfunModelInfo {
                name: model.to_string(),
                context_window: 16384,
                supports_tools: true,
                supports_vision: false,
                max_output_tokens: 8192,
            }),
            "step-1-256k" => Some(StepfunModelInfo {
                name: model.to_string(),
                context_window: 262144,
                supports_tools: true,
                supports_vision: false,
                max_output_tokens: 16384,
            }),
            "step-1v-32k" => Some(StepfunModelInfo {
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

/// Stepfun model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepfunModelInfo {
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
        let provider = StepfunProvider::new("test-key");
        assert!(provider.base_url.contains("stepfun.com"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = StepfunProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_get_model_info() {
        let info = StepfunProvider::get_model_info("step-1v-32k").unwrap();
        assert!(info.supports_vision);
    }
}
