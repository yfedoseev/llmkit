//! Llamafile provider for local LLM inference.
//!
//! Llamafile provides portable local inference with OpenAI-compatible API.

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Llamafile provider
pub struct LlamafileProvider {
    #[allow(dead_code)]
    base_url: String,
}

impl LlamafileProvider {
    /// Create a new Llamafile provider with default local URL
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
        Ok(vec!["llamafile".to_string()])
    }

    /// Get model details
    pub fn get_model_info(_model: &str) -> Option<LlamafileModelInfo> {
        Some(LlamafileModelInfo {
            name: "llamafile".to_string(),
            context_window: 8192,
            supports_tools: false,
        })
    }
}

impl Default for LlamafileProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Llamafile model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlamafileModelInfo {
    pub name: String,
    pub context_window: u32,
    pub supports_tools: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = LlamafileProvider::new();
        assert_eq!(provider.base_url, "http://localhost:8080/v1");
    }
}
