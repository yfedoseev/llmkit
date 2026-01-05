//! Lepton AI provider for serverless inference.
//!
//! Lepton provides serverless LLM inference with an OpenAI-compatible API.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Lepton AI provider
pub struct LeptonProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl LeptonProvider {
    /// Create a new Lepton provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://llama3-1-405b.lepton.run/api/v1".to_string(),
        }
    }

    /// Create with custom model endpoint
    pub fn with_model(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: format!("https://{}.lepton.run/api/v1", model),
        }
    }

    /// Create from environment variable
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("LEPTON_API_KEY").map_err(|_| {
            Error::Configuration("LEPTON_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "llama3-1-405b".to_string(),
            "llama3-1-70b".to_string(),
            "llama3-1-8b".to_string(),
            "mixtral-8x7b".to_string(),
            "mistral-7b".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<LeptonModelInfo> {
        match model {
            m if m.contains("405b") => Some(LeptonModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            m if m.contains("70b") => Some(LeptonModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            _ => None,
        }
    }
}

/// Lepton model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeptonModelInfo {
    pub name: String,
    pub context_window: u32,
    pub supports_tools: bool,
    pub max_output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = LeptonProvider::new("test-key");
        assert!(provider.base_url.contains("lepton.run"));
    }

    #[test]
    fn test_with_model() {
        let provider = LeptonProvider::with_model("test-key", "llama3-1-70b");
        assert!(provider.base_url.contains("llama3-1-70b"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = LeptonProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }
}
