//! MiniMax provider for Chinese AI models.
//!
//! MiniMax provides Chinese language models with OpenAI-compatible API.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// MiniMax provider
pub struct MiniMaxProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl MiniMaxProvider {
    /// Create a new MiniMax provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.minimax.chat/v1".to_string(),
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
        let api_key = std::env::var("MINIMAX_API_KEY").map_err(|_| {
            Error::Configuration("MINIMAX_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "abab6.5s-chat".to_string(),
            "abab6.5-chat".to_string(),
            "abab5.5-chat".to_string(),
            "abab5.5s-chat".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<MiniMaxModelInfo> {
        match model {
            "abab6.5s-chat" | "abab6.5-chat" => Some(MiniMaxModelInfo {
                name: model.to_string(),
                context_window: 245760,
                supports_tools: true,
                max_output_tokens: 16384,
            }),
            "abab5.5-chat" | "abab5.5s-chat" => Some(MiniMaxModelInfo {
                name: model.to_string(),
                context_window: 16384,
                supports_tools: true,
                max_output_tokens: 8192,
            }),
            _ => None,
        }
    }
}

/// MiniMax model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniMaxModelInfo {
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
        let provider = MiniMaxProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.minimax.chat/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = MiniMaxProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_get_model_info() {
        let info = MiniMaxProvider::get_model_info("abab6.5s-chat").unwrap();
        assert_eq!(info.context_window, 245760);
    }
}
