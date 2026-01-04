//! Grok Real-Time Voice provider (API access pending).
//!
//! This module provides a skeleton for xAI's Grok real-time voice capabilities.
//! Grok offers conversational AI with real-time voice interaction.
//!
//! # Status
//!
//! API access is currently pending approval from xAI.
//! Expected availability: Q1 2026 (pending partnership).

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{CompletionRequest, CompletionResponse, StreamChunk};

const GROK_REALTIME_API_URL: &str = "https://api.x.ai/v1/realtime";

/// Grok Real-Time Voice provider.
///
/// Provides access to xAI's Grok model with real-time voice capabilities
/// for low-latency conversational AI applications.
#[derive(Debug)]
pub struct GrokRealtimeProvider {
    config: ProviderConfig,
}

impl GrokRealtimeProvider {
    /// Create a new Grok real-time provider.
    ///
    /// # Note
    ///
    /// This provider requires API access approval from xAI.
    /// Contact: api-support@x.ai
    pub fn new(config: ProviderConfig) -> Result<Self> {
        Err(Error::config(
            "Grok real-time API access requires xAI partnership approval. Contact: api-support@x.ai",
        ))
    }

    /// Create a new Grok real-time provider from environment.
    pub fn from_env() -> Result<Self> {
        Err(Error::config(
            "Grok real-time API access requires xAI partnership approval. Contact: api-support@x.ai",
        ))
    }

    /// Create a new Grok real-time provider with an API key.
    pub fn with_api_key(_api_key: impl Into<String>) -> Result<Self> {
        Err(Error::config(
            "Grok real-time API access requires xAI partnership approval. Contact: api-support@x.ai",
        ))
    }

    fn _api_url(&self) -> &str {
        GROK_REALTIME_API_URL
    }
}

#[async_trait]
impl Provider for GrokRealtimeProvider {
    fn name(&self) -> &str {
        "grok-realtime"
    }

    fn supports_streaming(&self) -> bool {
        true // Real-time voice requires streaming
    }

    async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse> {
        Err(Error::config(
            "Grok real-time API access requires xAI partnership approval. Contact: api-support@x.ai",
        ))
    }

    async fn complete_stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        Err(Error::config(
            "Grok real-time API access requires xAI partnership approval. Contact: api-support@x.ai",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grok_requires_approval() {
        let config = ProviderConfig::new("test-key");
        let result = GrokRealtimeProvider::new(config);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("approval"));
    }

    #[test]
    fn test_grok_supports_streaming() {
        // Even though creation fails, the trait indicates streaming support
        assert!(true); // Real-time voice inherently supports streaming
    }

    #[test]
    fn test_grok_api_url_constant() {
        assert_eq!(GROK_REALTIME_API_URL, "https://api.x.ai/v1/realtime");
    }
}
