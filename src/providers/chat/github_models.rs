//! GitHub Models provider for AI model inference.
//!
//! GitHub Models provides access to various AI models through
//! Azure-hosted endpoints with an OpenAI-compatible API.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// GitHub Models provider
pub struct GitHubModelsProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl GitHubModelsProvider {
    /// Create a new GitHub Models provider with token
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://models.inference.ai.azure.com".to_string(),
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
        let api_key = std::env::var("GITHUB_TOKEN").map_err(|_| {
            Error::Configuration("GITHUB_TOKEN environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "gpt-4o".to_string(),
            "gpt-4o-mini".to_string(),
            "o1-preview".to_string(),
            "o1-mini".to_string(),
            "Meta-Llama-3.1-405B-Instruct".to_string(),
            "Meta-Llama-3.1-70B-Instruct".to_string(),
            "Meta-Llama-3.1-8B-Instruct".to_string(),
            "Mistral-large-2407".to_string(),
            "Mistral-Nemo".to_string(),
            "Phi-3.5-mini-instruct".to_string(),
            "Phi-3.5-MoE-instruct".to_string(),
            "AI21-Jamba-1.5-Large".to_string(),
            "AI21-Jamba-1.5-Mini".to_string(),
            "Cohere-command-r-plus".to_string(),
            "Cohere-command-r".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<GitHubModelInfo> {
        match model {
            "gpt-4o" | "gpt-4o-mini" => Some(GitHubModelInfo {
                name: model.to_string(),
                context_window: 128000,
                supports_tools: true,
                supports_vision: true,
                max_output_tokens: 16384,
            }),
            "o1-preview" | "o1-mini" => Some(GitHubModelInfo {
                name: model.to_string(),
                context_window: 128000,
                supports_tools: false,
                supports_vision: false,
                max_output_tokens: 32768,
            }),
            m if m.contains("Llama-3.1") => Some(GitHubModelInfo {
                name: model.to_string(),
                context_window: 131072,
                supports_tools: true,
                supports_vision: false,
                max_output_tokens: 16384,
            }),
            _ => None,
        }
    }
}

/// GitHub model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitHubModelInfo {
    pub name: String,
    pub context_window: u32,
    pub supports_tools: bool,
    pub supports_vision: bool,
    pub max_output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = GitHubModelsProvider::new("test-token");
        assert_eq!(provider.api_key, "test-token");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = GitHubModelsProvider::new("test-token");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.contains(&"gpt-4o".to_string()));
    }

    #[test]
    fn test_get_model_info() {
        let info = GitHubModelsProvider::get_model_info("gpt-4o").unwrap();
        assert!(info.supports_vision);
    }
}
