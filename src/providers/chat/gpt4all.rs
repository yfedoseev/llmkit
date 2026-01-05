//! GPT4All provider for local LLM inference.
//!
//! GPT4All provides local inference with an OpenAI-compatible API.

use crate::error::Result;
use serde::{Deserialize, Serialize};

/// GPT4All provider for local inference
pub struct GPT4AllProvider {
    #[allow(dead_code)]
    base_url: String,
}

impl GPT4AllProvider {
    /// Create a new GPT4All provider with default local URL
    pub fn new() -> Self {
        Self {
            base_url: "http://localhost:4891/v1".to_string(),
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
        Ok(vec![
            "mistral-7b-instruct-v0.1.Q4_0.gguf".to_string(),
            "gpt4all-falcon-newbpe-q4_0.gguf".to_string(),
            "orca-mini-3b-gguf2-q4_0.gguf".to_string(),
            "replit-code-v1_5-3b-newbpe-q4_0.gguf".to_string(),
            "nous-hermes-llama2-13b.Q4_0.gguf".to_string(),
            "llama-2-7b-chat.Q4_0.gguf".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<GPT4AllModelInfo> {
        match model {
            m if m.contains("mistral-7b") => Some(GPT4AllModelInfo {
                name: model.to_string(),
                parameters: "7B".to_string(),
                quantization: "Q4_0".to_string(),
                context_window: 8192,
            }),
            m if m.contains("falcon") => Some(GPT4AllModelInfo {
                name: model.to_string(),
                parameters: "7B".to_string(),
                quantization: "Q4_0".to_string(),
                context_window: 2048,
            }),
            m if m.contains("llama2-13b") || m.contains("llama-2-7b") => Some(GPT4AllModelInfo {
                name: model.to_string(),
                parameters: if m.contains("13b") { "13B" } else { "7B" }.to_string(),
                quantization: "Q4_0".to_string(),
                context_window: 4096,
            }),
            _ => None,
        }
    }
}

impl Default for GPT4AllProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// GPT4All model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPT4AllModelInfo {
    pub name: String,
    pub parameters: String,
    pub quantization: String,
    pub context_window: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = GPT4AllProvider::new();
        assert_eq!(provider.base_url, "http://localhost:4891/v1");
    }

    #[test]
    fn test_custom_url() {
        let provider = GPT4AllProvider::with_base_url("http://custom:8080/v1");
        assert_eq!(provider.base_url, "http://custom:8080/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = GPT4AllProvider::new();
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_get_model_info() {
        let info = GPT4AllProvider::get_model_info("mistral-7b-instruct-v0.1.Q4_0.gguf").unwrap();
        assert_eq!(info.parameters, "7B");
    }
}
