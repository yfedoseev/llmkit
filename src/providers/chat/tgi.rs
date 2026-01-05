//! TGI (Text Generation Inference) provider.
//!
//! HuggingFace TGI provides optimized inference serving.

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// TGI provider
pub struct TGIProvider {
    #[allow(dead_code)]
    base_url: String,
}

impl TGIProvider {
    /// Create a new TGI provider
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
        }
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["tgi-model".to_string()])
    }

    /// Get model details
    pub fn get_model_info(_model: &str) -> Option<TGIModelInfo> {
        Some(TGIModelInfo {
            name: "tgi-model".to_string(),
            max_input_length: 4096,
            max_total_tokens: 8192,
        })
    }
}

/// TGI model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TGIModelInfo {
    pub name: String,
    pub max_input_length: u32,
    pub max_total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = TGIProvider::new("http://localhost:8080");
        assert_eq!(provider.base_url, "http://localhost:8080");
    }
}
