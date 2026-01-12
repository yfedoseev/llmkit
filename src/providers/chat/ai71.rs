//! AI71 provider for Falcon models.
//!
//! AI71 is an Abu Dhabi-based AI company providing Falcon language models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// AI71 provider
pub struct AI71Provider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl AI71Provider {
    /// Create a new AI71 provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.ai71.ai/v1".to_string(),
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
        let api_key = std::env::var("AI71_API_KEY")
            .map_err(|_| Error::Configuration("AI71_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "tii/falcon-180B-chat".to_string(),
            "tii/falcon-40B-instruct".to_string(),
            "tii/falcon-7B-instruct".to_string(),
            "tii/falcon-mamba-7B".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<AI71ModelInfo> {
        match model {
            "tii/falcon-180B-chat" => Some(AI71ModelInfo {
                name: model.to_string(),
                parameters: "180B".to_string(),
                context_length: 8192,
                languages: vec![
                    "English".to_string(),
                    "Arabic".to_string(),
                    "French".to_string(),
                    "German".to_string(),
                    "Spanish".to_string(),
                ],
                capabilities: vec!["chat".to_string(), "code".to_string()],
                license: "Apache-2.0".to_string(),
            }),
            "tii/falcon-40B-instruct" => Some(AI71ModelInfo {
                name: model.to_string(),
                parameters: "40B".to_string(),
                context_length: 8192,
                languages: vec!["English".to_string(), "Arabic".to_string()],
                capabilities: vec!["chat".to_string(), "completion".to_string()],
                license: "Apache-2.0".to_string(),
            }),
            "tii/falcon-7B-instruct" => Some(AI71ModelInfo {
                name: model.to_string(),
                parameters: "7B".to_string(),
                context_length: 4096,
                languages: vec!["English".to_string()],
                capabilities: vec!["chat".to_string(), "completion".to_string()],
                license: "Apache-2.0".to_string(),
            }),
            "tii/falcon-mamba-7B" => Some(AI71ModelInfo {
                name: model.to_string(),
                parameters: "7B".to_string(),
                context_length: 16384,
                languages: vec!["English".to_string()],
                capabilities: vec!["chat".to_string(), "long-context".to_string()],
                license: "Apache-2.0".to_string(),
            }),
            _ => None,
        }
    }
}

/// AI71 model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AI71ModelInfo {
    pub name: String,
    pub parameters: String,
    pub context_length: u32,
    pub languages: Vec<String>,
    pub capabilities: Vec<String>,
    pub license: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = AI71Provider::new("test-key");
        assert!(provider.base_url.contains("ai71"));
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = AI71Provider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 3);
    }

    #[test]
    fn test_model_info() {
        let info = AI71Provider::get_model_info("tii/falcon-180B-chat").unwrap();
        assert_eq!(info.parameters, "180B");
    }
}
