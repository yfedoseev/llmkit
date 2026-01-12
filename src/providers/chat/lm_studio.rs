//! LM Studio provider for local LLM inference.
//!
//! LM Studio provides local inference with an OpenAI-compatible API.

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// LM Studio provider
pub struct LMStudioProvider {
    #[allow(dead_code)]
    base_url: String,
}

impl LMStudioProvider {
    /// Create a new LM Studio provider with default local URL
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:1234/v1".to_string(),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
        }
    }

    /// Get list of available models (dynamic, depends on loaded models)
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["local-model".to_string()])
    }

    /// Get model details
    pub fn get_model_info(_model: &str) -> Option<LMStudioModelInfo> {
        Some(LMStudioModelInfo {
            name: "local-model".to_string(),
            context_window: 8192,
            supports_tools: true,
        })
    }
}

impl Default for LMStudioProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// LM Studio model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LMStudioModelInfo {
    pub name: String,
    pub context_window: u32,
    pub supports_tools: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = LMStudioProvider::new();
        assert_eq!(provider.base_url, "http://localhost:1234/v1");
    }

    #[test]
    fn test_custom_url() {
        let provider = LMStudioProvider::with_base_url("http://custom:8080/v1");
        assert_eq!(provider.base_url, "http://custom:8080/v1");
    }
}
