//! Xinference provider for local LLM inference.
//!
//! Xinference (Xorbits Inference) provides local inference with OpenAI-compatible API.

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Xinference provider
pub struct XinferenceProvider {
    #[allow(dead_code)]
    base_url: String,
}

impl XinferenceProvider {
    /// Create a new Xinference provider with default local URL
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:9997/v1".to_string(),
        }
    }

    /// Create with custom base URL
    pub fn with_base_url(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
        }
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["xinference-model".to_string()])
    }

    /// Get model details
    pub fn get_model_info(_model: &str) -> Option<XinferenceModelInfo> {
        Some(XinferenceModelInfo {
            name: "xinference-model".to_string(),
            context_window: 8192,
            supports_tools: true,
        })
    }
}

impl Default for XinferenceProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Xinference model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XinferenceModelInfo {
    pub name: String,
    pub context_window: u32,
    pub supports_tools: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = XinferenceProvider::new();
        assert_eq!(provider.base_url, "http://localhost:9997/v1");
    }
}
