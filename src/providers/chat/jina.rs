//! Jina AI provider for embeddings and reranking.
//!
//! Jina AI provides embedding and reranking models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Jina AI provider
pub struct JinaProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl JinaProvider {
    /// Create a new Jina provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.jina.ai/v1".to_string(),
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
        let api_key = std::env::var("JINA_API_KEY").map_err(|_| {
            Error::Configuration("JINA_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "jina-embeddings-v3".to_string(),
            "jina-embeddings-v2-base-en".to_string(),
            "jina-embeddings-v2-base-de".to_string(),
            "jina-embeddings-v2-base-zh".to_string(),
            "jina-colbert-v2".to_string(),
            "jina-reranker-v2-base-multilingual".to_string(),
            "jina-clip-v1".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<JinaModelInfo> {
        match model {
            "jina-embeddings-v3" => Some(JinaModelInfo {
                name: model.to_string(),
                dimensions: 1024,
                max_tokens: 8192,
                model_type: JinaModelType::Embedding,
            }),
            m if m.contains("embeddings-v2") => Some(JinaModelInfo {
                name: model.to_string(),
                dimensions: 768,
                max_tokens: 8192,
                model_type: JinaModelType::Embedding,
            }),
            m if m.contains("reranker") => Some(JinaModelInfo {
                name: model.to_string(),
                dimensions: 0,
                max_tokens: 8192,
                model_type: JinaModelType::Reranker,
            }),
            "jina-clip-v1" => Some(JinaModelInfo {
                name: model.to_string(),
                dimensions: 768,
                max_tokens: 77,
                model_type: JinaModelType::MultiModal,
            }),
            _ => None,
        }
    }
}

/// Jina model type
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JinaModelType {
    Embedding,
    Reranker,
    MultiModal,
}

/// Jina model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JinaModelInfo {
    pub name: String,
    pub dimensions: u32,
    pub max_tokens: u32,
    pub model_type: JinaModelType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = JinaProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.jina.ai/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = JinaProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
    }

    #[test]
    fn test_get_model_info() {
        let info = JinaProvider::get_model_info("jina-embeddings-v3").unwrap();
        assert_eq!(info.dimensions, 1024);
    }
}
