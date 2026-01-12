//! LG AI Research provider for EXAONE models.
//!
//! LG's EXAONE is an expert AI model for enterprise applications.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// LG EXAONE provider
pub struct LGExaoneProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl LGExaoneProvider {
    /// Create a new LG EXAONE provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.lgresearch.ai/v1".to_string(),
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
        let api_key = std::env::var("LG_EXAONE_API_KEY")
            .map_err(|_| Error::Configuration("LG_EXAONE_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "EXAONE-3.5-32B-Instruct".to_string(),
            "EXAONE-3.5-7.8B-Instruct".to_string(),
            "EXAONE-3.5-2.4B-Instruct".to_string(),
            "EXAONE-3.0-7.8B-Instruct".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<LGExaoneModelInfo> {
        match model {
            "EXAONE-3.5-32B-Instruct" => Some(LGExaoneModelInfo {
                name: model.to_string(),
                parameters: "32B".to_string(),
                context_length: 32768,
                language_focus: "Korean, English, Japanese".to_string(),
                capabilities: vec!["chat".to_string(), "code".to_string(), "math".to_string()],
            }),
            "EXAONE-3.5-7.8B-Instruct" => Some(LGExaoneModelInfo {
                name: model.to_string(),
                parameters: "7.8B".to_string(),
                context_length: 32768,
                language_focus: "Korean, English".to_string(),
                capabilities: vec!["chat".to_string(), "code".to_string()],
            }),
            "EXAONE-3.5-2.4B-Instruct" => Some(LGExaoneModelInfo {
                name: model.to_string(),
                parameters: "2.4B".to_string(),
                context_length: 32768,
                language_focus: "Korean, English".to_string(),
                capabilities: vec!["chat".to_string()],
            }),
            "EXAONE-3.0-7.8B-Instruct" => Some(LGExaoneModelInfo {
                name: model.to_string(),
                parameters: "7.8B".to_string(),
                context_length: 8192,
                language_focus: "Korean, English".to_string(),
                capabilities: vec!["chat".to_string()],
            }),
            _ => None,
        }
    }
}

/// LG EXAONE model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LGExaoneModelInfo {
    pub name: String,
    pub parameters: String,
    pub context_length: u32,
    pub language_focus: String,
    pub capabilities: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = LGExaoneProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.lgresearch.ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = LGExaoneProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 3);
    }

    #[test]
    fn test_model_info() {
        let info = LGExaoneProvider::get_model_info("EXAONE-3.5-32B-Instruct").unwrap();
        assert_eq!(info.parameters, "32B");
    }
}
