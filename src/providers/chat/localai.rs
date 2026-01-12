//! LocalAI provider for local LLM inference.
//!
//! LocalAI provides local inference with OpenAI-compatible API.

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// LocalAI provider
pub struct LocalAIProvider {
    #[allow(dead_code)]
    base_url: String,
}

impl LocalAIProvider {
    /// Create a new LocalAI provider with default local URL
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:8080/v1".to_string(),
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
        Ok(vec!["localai-model".to_string()])
    }

    /// Get model details
    pub fn get_model_info(_model: &str) -> Option<LocalAIModelInfo> {
        Some(LocalAIModelInfo {
            name: "localai-model".to_string(),
            context_window: 8192,
            supports_tools: true,
        })
    }
}

impl Default for LocalAIProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// LocalAI model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalAIModelInfo {
    pub name: String,
    pub context_window: u32,
    pub supports_tools: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = LocalAIProvider::new();
        assert_eq!(provider.base_url, "http://localhost:8080/v1");
    }
}
