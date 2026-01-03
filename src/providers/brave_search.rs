//! Brave Search API provider implementation.
//!
//! This module provides privacy-focused web search functionality with optional AI summaries.
//! Brave Search is privacy-first and supports the Model Context Protocol (MCP).
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::BraveSearchProvider;
//!
//! let provider = BraveSearchProvider::from_env()?;
//! let results = provider.search("quantum computing breakthroughs", 10).await?;
//! for result in results.web_results() {
//!     println!("{}: {}", result.title, result.url);
//! }
//! ```
//!
//! # Authentication
//!
//! Brave Search uses API key authentication. Provide via:
//! - Environment variable: `BRAVE_API_KEY`
//! - Direct initialization: `BraveSearchProvider::new(api_key)`
//!
//! # Environment Variables
//!
//! - `BRAVE_API_KEY` - Brave Search API key (required)

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Web search result from Brave API
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BraveWebResult {
    /// Title of the search result
    pub title: String,
    /// URL of the search result
    pub url: String,
    /// Description/snippet of the result
    pub description: Option<String>,
    /// When the page was last indexed/updated
    #[serde(rename = "page_age")]
    pub page_age: Option<String>,
}

/// Profile information for the result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileInfo {
    /// Profile image URL
    pub img: Option<String>,
    /// Profile name
    pub name: Option<String>,
}

/// Web search results from Brave API
#[derive(Debug, Deserialize)]
pub struct BraveWebResults {
    #[serde(default)]
    pub results: Vec<BraveWebResult>,
}

/// Full search response from Brave API
#[derive(Debug, Deserialize)]
pub struct BraveSearchResponse {
    /// Web search results
    #[serde(default)]
    pub web: Option<BraveWebResults>,
    /// AI-generated summary (if requested)
    pub summary: Option<String>,
}

/// Brave Search API provider.
///
/// Provides privacy-focused web search with optional AI summaries.
pub struct BraveSearchProvider {
    api_key: String,
    client: reqwest::Client,
}

impl BraveSearchProvider {
    /// Create provider from environment variables.
    ///
    /// Reads `BRAVE_API_KEY` environment variable.
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("BRAVE_API_KEY")
            .map_err(|_| Error::config("BRAVE_API_KEY environment variable not set"))?;
        Ok(Self::new(&api_key))
    }

    /// Create provider with explicit API key.
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            client: reqwest::Client::new(),
        }
    }

    /// Perform a web search.
    ///
    /// Searches the web using Brave Search with privacy-first approach.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query
    /// * `count` - Number of results to return (1-20)
    ///
    /// # Example
    ///
    /// ```ignore
    /// let results = provider.search("AI safety concerns", 10).await?;
    /// ```
    pub async fn search(&self, query: &str, count: u32) -> Result<Vec<BraveWebResult>> {
        let count = count.clamp(1, 20);

        let response = self
            .client
            .get("https://api.search.brave.com/res/v1/web/search")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .query(&[("q", query), ("count", &count.to_string())])
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await?;
            return Err(self.handle_error_response(status, &body));
        }

        let brave_response: BraveSearchResponse = response.json().await?;

        match brave_response.web {
            Some(web_results) => Ok(web_results.results),
            None => Ok(Vec::new()),
        }
    }

    /// Perform a web search with AI summary.
    ///
    /// Searches the web and generates an AI summary of the results.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query
    /// * `count` - Number of results to return (1-20)
    ///
    /// # Returns
    ///
    /// Tuple of (search results, optional summary)
    pub async fn search_with_summary(
        &self,
        query: &str,
        count: u32,
    ) -> Result<(Vec<BraveWebResult>, Option<String>)> {
        let count = count.clamp(1, 20);

        let response = self
            .client
            .get("https://api.search.brave.com/res/v1/web/search")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .query(&[
                ("q", query.to_string()),
                ("count", count.to_string()),
                ("summary", "true".to_string()),
            ])
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await?;
            return Err(self.handle_error_response(status, &body));
        }

        let brave_response: BraveSearchResponse = response.json().await?;

        let results = match brave_response.web {
            Some(web_results) => web_results.results,
            None => Vec::new(),
        };

        Ok((results, brave_response.summary))
    }

    /// Handle error responses from Brave.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        match status.as_u16() {
            400 => Error::other(format!("Invalid request: {}", body)),
            401 => Error::auth("Unauthorized access to Brave Search API".to_string()),
            403 => Error::auth("Forbidden access to Brave Search API".to_string()),
            429 => Error::rate_limited("Brave Search rate limit exceeded".to_string(), None),
            500..=599 => Error::server(status.as_u16(), format!("Brave Search error: {}", body)),
            _ => Error::other(format!("HTTP {}: {}", status, body)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_brave_provider_creation() {
        let provider = BraveSearchProvider::new("test-api-key");
        assert_eq!(provider.api_key, "test-api-key");
    }

    #[test]
    fn test_web_result_deserialization() {
        let json = r#"{
            "title": "Test Result",
            "url": "https://example.com",
            "description": "Test description",
            "page_age": "2024-01-01"
        }"#;

        let result: BraveWebResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.title, "Test Result");
        assert_eq!(result.url, "https://example.com");
    }

    #[test]
    fn test_search_response_deserialization() {
        let json = r#"{
            "web": {
                "results": [
                    {
                        "title": "Test",
                        "url": "https://example.com",
                        "description": "Test description"
                    }
                ]
            },
            "summary": "Test summary"
        }"#;

        let response: BraveSearchResponse = serde_json::from_str(json).unwrap();
        assert!(response.web.is_some());
        assert_eq!(response.summary, Some("Test summary".to_string()));
    }
}
