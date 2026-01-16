#![allow(dead_code)]
//! LatamGPT regional provider (API launch pending).
//!
//! This module provides a skeleton for LatamGPT, a Latin American
//! LLM service with regional optimization.
//!
//! # Status
//!
//! API is expected to launch in late January/February 2026.
//! Check: <https://latamgpt.dev> for latest status.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{CompletionRequest, CompletionResponse, StreamChunk};

const LATAMGPT_API_URL: &str = "https://api.latamgpt.dev/v1";

/// LatamGPT regional provider.
///
/// LatamGPT specializes in serving Latin American markets with
/// models optimized for Spanish and Portuguese languages.
#[derive(Debug)]
pub struct LatamGPTProvider {
    config: ProviderConfig,
}

impl LatamGPTProvider {
    /// Create a new LatamGPT provider.
    ///
    /// # Note
    ///
    /// This provider's API is currently in beta.
    /// Expected launch: January-February 2026.
    pub fn new(_config: ProviderConfig) -> Result<Self> {
        Err(Error::config(
            "LatamGPT API is launching in January-February 2026. Not available yet.",
        ))
    }

    /// Create a new LatamGPT provider from environment.
    pub fn from_env() -> Result<Self> {
        Err(Error::config(
            "LatamGPT API is launching in January-February 2026. Not available yet.",
        ))
    }

    /// Create a new LatamGPT provider with an API key.
    pub fn with_api_key(_api_key: impl Into<String>) -> Result<Self> {
        Err(Error::config(
            "LatamGPT API is launching in January-February 2026. Not available yet.",
        ))
    }

    fn _api_url(&self) -> &str {
        LATAMGPT_API_URL
    }
}

#[async_trait]
impl Provider for LatamGPTProvider {
    fn name(&self) -> &str {
        "latamgpt"
    }

    async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse> {
        Err(Error::config(
            "LatamGPT API is launching in January-February 2026. Not available yet.",
        ))
    }

    async fn complete_stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        Err(Error::config(
            "LatamGPT API is launching in January-February 2026. Not available yet.",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_latamgpt_coming_soon() {
        let config = ProviderConfig::new("test-key");
        let result = LatamGPTProvider::new(config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("2026"));
    }

    #[test]
    fn test_latamgpt_api_url_constant() {
        assert_eq!(LATAMGPT_API_URL, "https://api.latamgpt.dev/v1");
    }
}
