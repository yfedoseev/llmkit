//! Jan provider for local LLM inference.
//!
//! Jan provides local inference with OpenAI-compatible API.

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Jan provider
pub struct JanProvider {
    #[allow(dead_code)]
    base_url: String,
}

impl JanProvider {
    /// Create a new Jan provider with default local URL
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:1337/v1".to_string(),
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
        Ok(vec!["jan-model".to_string()])
    }

    /// Get model details
    pub fn get_model_info(_model: &str) -> Option<JanModelInfo> {
        Some(JanModelInfo {
            name: "jan-model".to_string(),
            context_window: 8192,
            supports_tools: true,
        })
    }
}

impl Default for JanProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Jan model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JanModelInfo {
    pub name: String,
    pub context_window: u32,
    pub supports_tools: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = JanProvider::new();
        assert_eq!(provider.base_url, "http://localhost:1337/v1");
    }
}
