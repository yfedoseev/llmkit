//! LightOn France AI provider (Partnership pending).
//!
//! This module provides a skeleton for LightOn's AI services.
//! LightOn is a French AI company offering cloud-based language models.
//!
//! # Status
//!
//! API access is currently pending partnership approval.
//! Expected availability: Q1-Q2 2026 (partnership dependent).

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{CompletionRequest, CompletionResponse, StreamChunk};

const LIGHTON_API_URL: &str = "https://api.lighton.ai/v1";

/// LightOn AI provider.
///
/// LightOn offers efficient, fine-tuned language models
/// optimized for European markets and GDPR compliance.
#[derive(Debug)]
pub struct LightOnProvider {
    config: ProviderConfig,
}

impl LightOnProvider {
    /// Create a new LightOn provider.
    ///
    /// # Note
    ///
    /// This provider requires API access partnership with LightOn.
    /// Contact: partnership@lighton.ai
    pub fn new(config: ProviderConfig) -> Result<Self> {
        Err(Error::config(
            "LightOn API access requires partnership approval. Contact: partnership@lighton.ai",
        ))
    }

    /// Create a new LightOn provider from environment.
    pub fn from_env() -> Result<Self> {
        Err(Error::config(
            "LightOn API access requires partnership approval. Contact: partnership@lighton.ai",
        ))
    }

    /// Create a new LightOn provider with an API key.
    pub fn with_api_key(_api_key: impl Into<String>) -> Result<Self> {
        Err(Error::config(
            "LightOn API access requires partnership approval. Contact: partnership@lighton.ai",
        ))
    }

    fn _api_url(&self) -> &str {
        LIGHTON_API_URL
    }
}

#[async_trait]
impl Provider for LightOnProvider {
    fn name(&self) -> &str {
        "lighton"
    }

    async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse> {
        Err(Error::config(
            "LightOn API access requires partnership approval. Contact: partnership@lighton.ai",
        ))
    }

    async fn complete_stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        Err(Error::config(
            "LightOn API access requires partnership approval. Contact: partnership@lighton.ai",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lighton_requires_partnership() {
        let config = ProviderConfig::new("test-key");
        let result = LightOnProvider::new(config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("partnership"));
    }

    #[test]
    fn test_lighton_api_url_constant() {
        assert_eq!(LIGHTON_API_URL, "https://api.lighton.ai/v1");
    }
}
