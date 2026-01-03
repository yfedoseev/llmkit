//! Exa AI semantic search provider implementation.
//!
//! This module provides semantic web search functionality optimized for LLMs and AI agents.
//! Unlike traditional keyword-based search, Exa uses neural embeddings to find conceptually
//! relevant content.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::ExaProvider;
//!
//! let provider = ExaProvider::from_env()?;
//! let results = provider.search("latest AI breakthroughs").await?;
//! for result in results {
//!     println!("{}: {}", result.title, result.url);
//! }
//! ```
//!
//! # Authentication
//!
//! Exa uses API key authentication. Provide via:
//! - Environment variable: `EXA_API_KEY`
//! - Direct initialization: `ExaProvider::new(api_key)`
//!
//! # Environment Variables
//!
//! - `EXA_API_KEY` - Exa API key (required)

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Search result from Exa API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExaSearchResult {
    /// Title of the search result
    pub title: String,
    /// URL of the search result
    pub url: String,
    /// Published date (ISO 8601)
    pub published_date: Option<String>,
    /// Author of the content
    pub author: Option<String>,
    /// Extracted text from the page
    pub text: Option<String>,
    /// Similarity score (0-1)
    pub score: Option<f32>,
}

/// Search response from Exa API
#[derive(Debug, Deserialize)]
struct ExaResponse {
    results: Vec<ExaSearchResult>,
}

/// Search request for Exa API
#[derive(Debug, Serialize)]
struct ExaSearchRequest {
    query: String,
    #[serde(rename = "numResults")]
    num_results: u32,
    #[serde(rename = "searchType")]
    search_type: String,
    #[serde(rename = "includeText")]
    include_text: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(rename = "startPublishedDate")]
    start_published_date: Option<String>,
}

/// Exa AI semantic search provider.
///
/// Provides semantic web search functionality optimized for LLM use cases.
pub struct ExaProvider {
    api_key: String,
    client: reqwest::Client,
}

impl ExaProvider {
    /// Create provider from environment variables.
    ///
    /// Reads `EXA_API_KEY` environment variable.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("EXA_API_KEY")
            .map_err(|_| Error::config("EXA_API_KEY environment variable not set"))?;
        Ok(Self::new(&api_key))
    }

    /// Create provider with explicit API key.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Perform a semantic search.
    ///
    /// Searches the web using semantic embeddings rather than keyword matching.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query (can be a question or description)
    /// * `num_results` - Number of results to return (1-100)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let results = provider.search("What are the latest developments in AI?", 10).await?;
    /// ```
    pub async fn search(&self, query: &str, num_results: u32) -> Result<Vec<ExaSearchResult>> {
        let num_results = num_results.clamp(1, 100);

        let request = ExaSearchRequest {
            query: query.to_string(),
            num_results,
            search_type: "auto".to_string(),
            include_text: true,
            start_published_date: None,
        };

        let response = self
            .client
            .post("https://api.exa.ai/search")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await?;
            return Err(self.handle_error_response(status, &body));
        }

        let exa_response: ExaResponse = response.json().await?;
        Ok(exa_response.results)
    }

    /// Perform a semantic search with date filtering.
    pub async fn search_with_date(
        &self,
        query: &str,
        num_results: u32,
        start_published_date: &str,
    ) -> Result<Vec<ExaSearchResult>> {
        let num_results = num_results.clamp(1, 100);

        let request = ExaSearchRequest {
            query: query.to_string(),
            num_results,
            search_type: "auto".to_string(),
            include_text: true,
            start_published_date: Some(start_published_date.to_string()),
        };

        let response = self
            .client
            .post("https://api.exa.ai/search")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await?;
            return Err(self.handle_error_response(status, &body));
        }

        let exa_response: ExaResponse = response.json().await?;
        Ok(exa_response.results)
    }

    /// Handle error responses from Exa.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        match status.as_u16() {
            400 => Error::other(format!("Invalid request: {}", body)),
            401 => Error::auth("Unauthorized access to Exa API".to_string()),
            403 => Error::auth("Forbidden access to Exa API".to_string()),
            429 => Error::rate_limited("Exa API rate limit exceeded".to_string(), None),
            500..=599 => Error::server(status.as_u16(), format!("Exa error: {}", body)),
            _ => Error::other(format!("HTTP {}: {}", status, body)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exa_provider_creation() {
        let provider = ExaProvider::new("test-api-key");
        assert_eq!(provider.api_key, "test-api-key");
    }

    #[test]
    fn test_exa_api_url() {
        let provider = ExaProvider::new("test-key");
        assert!(!provider.api_key.is_empty());
    }

    #[test]
    fn test_search_result_deserialization() {
        let json = r#"{
            "title": "Test Title",
            "url": "https://example.com",
            "published_date": "2024-01-01",
            "author": "Test Author",
            "text": "Test content",
            "score": 0.95
        }"#;

        let result: ExaSearchResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.title, "Test Title");
        assert_eq!(result.url, "https://example.com");
        assert_eq!(result.score, Some(0.95));
    }
}
