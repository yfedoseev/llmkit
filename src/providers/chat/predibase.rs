//! Predibase provider for fine-tuned LLM serving.
//!
//! Predibase specializes in serving fine-tuned models with LoRA adapters.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Predibase provider
pub struct PredibaseProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl PredibaseProvider {
    /// Create a new Predibase provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.predibase.com/v1".to_string(),
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
        let api_key = std::env::var("PREDIBASE_API_KEY")
            .map_err(|_| Error::Configuration("PREDIBASE_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "mistral-7b".to_string(),
            "llama-2-7b".to_string(),
            "llama-2-13b".to_string(),
            "codellama-7b".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<PredibaseModelInfo> {
        match model {
            m if m.contains("mistral-7b") => Some(PredibaseModelInfo {
                name: model.to_string(),
                context_window: 32768,
                supports_lora: true,
                max_output_tokens: 8192,
            }),
            m if m.contains("llama-2-7b") => Some(PredibaseModelInfo {
                name: model.to_string(),
                context_window: 4096,
                supports_lora: true,
                max_output_tokens: 4096,
            }),
            m if m.contains("llama-2-13b") => Some(PredibaseModelInfo {
                name: model.to_string(),
                context_window: 4096,
                supports_lora: true,
                max_output_tokens: 4096,
            }),
            m if m.contains("codellama") => Some(PredibaseModelInfo {
                name: model.to_string(),
                context_window: 16384,
                supports_lora: true,
                max_output_tokens: 4096,
            }),
            _ => None,
        }
    }
}

/// Predibase model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredibaseModelInfo {
    pub name: String,
    pub context_window: u32,
    pub supports_lora: bool,
    pub max_output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = PredibaseProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.predibase.com/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = PredibaseProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_model_info() {
        let info = PredibaseProvider::get_model_info("mistral-7b").unwrap();
        assert!(info.supports_lora);
    }
}
