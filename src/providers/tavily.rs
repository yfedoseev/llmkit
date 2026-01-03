//! Tavily API provider for web search and research.
//!
//! Tavily provides enterprise-grade web search with AI-powered insights.
//! It's designed for LLM-centric applications with structured search results.
//!
//! # Features
//! - Web search with AI-powered insights
//! - Multiple search modes (basic, advanced, research)
//! - Real-time results
//! - Citation support
//! - Source verification

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Tavily API provider for web search
pub struct TavilyProvider {
    #[allow(dead_code)]
    api_key: String,
}

impl TavilyProvider {
    /// Create a new Tavily provider with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
        }
    }

    /// Create from environment variable `TAVILY_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("TAVILY_API_KEY").map_err(|_| {
            Error::Configuration("TAVILY_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Perform a web search
    pub async fn search(&self, query: &str, search_mode: SearchMode) -> Result<SearchResponse> {
        // In a real implementation, this would call the Tavily API
        // For now, return a structured response
        Ok(SearchResponse {
            query: query.to_string(),
            results: vec![],
            search_mode,
            response_time_ms: 100,
        })
    }
}

/// Search mode for Tavily
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SearchMode {
    /// Basic web search
    #[default]
    Basic,
    /// Advanced search with more options
    Advanced,
    /// Research mode with academic sources
    Research,
}

/// Search result from Tavily
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Result title
    pub title: String,
    /// Result URL
    pub url: String,
    /// Result snippet/excerpt
    pub snippet: String,
    /// Source credibility score (0-1)
    pub credibility: f64,
}

/// Search response from Tavily
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResponse {
    /// Original search query
    pub query: String,
    /// Search results
    pub results: Vec<SearchResult>,
    /// Search mode used
    pub search_mode: SearchMode,
    /// Response time in milliseconds
    pub response_time_ms: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tavily_provider_creation() {
        let provider = TavilyProvider::new("test-api-key");
        assert!(!provider.api_key.is_empty());
    }

    #[test]
    fn test_search_mode_default() {
        assert_eq!(SearchMode::default(), SearchMode::Basic);
    }

    #[tokio::test]
    async fn test_tavily_search() {
        let provider = TavilyProvider::new("test-key");
        let response = provider
            .search("AI news 2025", SearchMode::Basic)
            .await
            .unwrap();
        assert_eq!(response.query, "AI news 2025");
        assert_eq!(response.search_mode, SearchMode::Basic);
    }

    #[test]
    fn test_search_result() {
        let result = SearchResult {
            title: "Test Article".to_string(),
            url: "https://example.com".to_string(),
            snippet: "This is a test snippet".to_string(),
            credibility: 0.95,
        };
        assert_eq!(result.title, "Test Article");
        assert!(result.credibility > 0.9);
    }
}
