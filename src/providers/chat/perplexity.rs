//! Perplexity AI provider for online RAG and search-augmented generation.
//!
//! Perplexity AI provides cutting-edge large language models with
//! real-time web search integration for retrieval-augmented generation.
//!
//! # Features
//! - Real-time web search integration
//! - Citation support with source tracking
//! - Multiple model tiers available
//! - Streaming responses
//! - Research-focused capabilities

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Perplexity AI provider for search-augmented generation
pub struct PerplexityProvider {
    #[allow(dead_code)]
    api_key: String,
}

impl PerplexityProvider {
    /// Create a new Perplexity provider with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
        }
    }

    /// Create from environment variable `PERPLEXITY_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("PERPLEXITY_API_KEY").map_err(|_| {
            Error::Configuration("PERPLEXITY_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        // Mock implementation
        Ok(vec![
            "pplx-7b-online".to_string(),
            "pplx-70b-online".to_string(),
            "pplx-8x7b-online".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<PerplexityModelInfo> {
        match model {
            "pplx-7b-online" => Some(PerplexityModelInfo {
                name: "pplx-7b-online".to_string(),
                context_window: 8000,
                supports_search: true,
                max_output_tokens: 2000,
            }),
            "pplx-70b-online" => Some(PerplexityModelInfo {
                name: "pplx-70b-online".to_string(),
                context_window: 8000,
                supports_search: true,
                max_output_tokens: 4000,
            }),
            "pplx-8x7b-online" => Some(PerplexityModelInfo {
                name: "pplx-8x7b-online".to_string(),
                context_window: 8000,
                supports_search: true,
                max_output_tokens: 4000,
            }),
            _ => None,
        }
    }
}

/// Perplexity model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerplexityModelInfo {
    /// Model name
    pub name: String,
    /// Context window size
    pub context_window: u32,
    /// Whether this model supports web search
    pub supports_search: bool,
    /// Maximum output tokens
    pub max_output_tokens: u32,
}

/// Search configuration for Perplexity
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
pub enum PerplexitySearchMode {
    /// Default search behavior
    #[default]
    Default,
    /// Focus on recent information
    Recent,
    /// Academic and research sources
    Academic,
    /// News sources
    News,
}

/// Citation for a source in Perplexity response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    /// URL of the cited source
    pub url: String,
    /// Title of the source
    pub title: String,
    /// Relevance score (0-1)
    pub relevance: f64,
}

/// Search response with citations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchAugmentedResponse {
    /// Generated answer text
    pub answer: String,
    /// Sources used in generation
    pub citations: Vec<Citation>,
    /// Search mode used
    pub search_mode: PerplexitySearchMode,
    /// Number of sources searched
    pub num_sources: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perplexity_provider_creation() {
        let provider = PerplexityProvider::new("test-key");
        assert_eq!(provider.api_key, "test-key");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = PerplexityProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.contains(&"pplx-7b-online".to_string()));
    }

    #[test]
    fn test_get_model_info() {
        let info = PerplexityProvider::get_model_info("pplx-70b-online").unwrap();
        assert_eq!(info.name, "pplx-70b-online");
        assert!(info.supports_search);
        assert_eq!(info.context_window, 8000);
    }

    #[test]
    fn test_model_info_invalid() {
        let info = PerplexityProvider::get_model_info("invalid-model");
        assert!(info.is_none());
    }

    #[test]
    fn test_search_mode_default() {
        assert_eq!(
            PerplexitySearchMode::default(),
            PerplexitySearchMode::Default
        );
    }

    #[test]
    fn test_citation_relevance() {
        let citation = Citation {
            url: "https://example.com".to_string(),
            title: "Example Article".to_string(),
            relevance: 0.95,
        };
        assert!(citation.relevance > 0.9);
    }

    #[test]
    fn test_search_augmented_response() {
        let response = SearchAugmentedResponse {
            answer: "The answer is 42".to_string(),
            citations: vec![
                Citation {
                    url: "https://example1.com".to_string(),
                    title: "Source 1".to_string(),
                    relevance: 0.9,
                },
                Citation {
                    url: "https://example2.com".to_string(),
                    title: "Source 2".to_string(),
                    relevance: 0.85,
                },
            ],
            search_mode: PerplexitySearchMode::Recent,
            num_sources: 15,
        };
        assert_eq!(response.citations.len(), 2);
        assert_eq!(response.num_sources, 15);
    }
}
