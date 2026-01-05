//! Triton Inference Server provider.
//!
//! NVIDIA Triton provides optimized inference serving.

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// Triton provider
pub struct TritonProvider {
    #[allow(dead_code)]
    base_url: String,
}

impl TritonProvider {
    /// Create a new Triton provider
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
        }
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec!["triton-model".to_string()])
    }

    /// Get model details
    pub fn get_model_info(_model: &str) -> Option<TritonModelInfo> {
        Some(TritonModelInfo {
            name: "triton-model".to_string(),
            backend: "tensorrt_llm".to_string(),
            max_batch_size: 128,
        })
    }
}

/// Triton model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TritonModelInfo {
    pub name: String,
    pub backend: String,
    pub max_batch_size: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = TritonProvider::new("http://localhost:8000");
        assert_eq!(provider.base_url, "http://localhost:8000");
    }
}
