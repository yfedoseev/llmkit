//! DiffusionRouter video generation provider (Coming February 2026).
//!
//! This module provides a skeleton for DiffusionRouter video generation,
//! scheduled to launch in February 2026.
//!
//! # Status
//!
//! Currently unavailable. API integration will be completed in February 2026.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{CompletionRequest, CompletionResponse, StreamChunk};

#[allow(dead_code)]
const DIFFUSION_ROUTER_API_URL: &str = "https://api.diffusionrouter.ai/v1";

/// DiffusionRouter video generation provider.
///
/// Planned for February 2026 launch.
#[allow(dead_code)]
pub struct DiffusionRouterProvider {
    config: ProviderConfig,
}

impl DiffusionRouterProvider {
    /// Create a new DiffusionRouter provider.
    ///
    /// # Note
    ///
    /// This provider is not yet available. API is scheduled for February 2026.
    pub fn new(_config: ProviderConfig) -> Result<Self> {
        Err(Error::config(
            "DiffusionRouter API is coming in February 2026. Not available yet.",
        ))
    }

    /// Create a new DiffusionRouter provider from environment.
    pub fn from_env() -> Result<Self> {
        Err(Error::config(
            "DiffusionRouter API is coming in February 2026. Not available yet.",
        ))
    }

    /// Create a new DiffusionRouter provider with an API key.
    pub fn with_api_key(_api_key: impl Into<String>) -> Result<Self> {
        Err(Error::config(
            "DiffusionRouter API is coming in February 2026. Not available yet.",
        ))
    }

    fn _api_url(&self) -> &str {
        DIFFUSION_ROUTER_API_URL
    }
}

#[async_trait]
impl Provider for DiffusionRouterProvider {
    fn name(&self) -> &str {
        "diffusion-router"
    }

    async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse> {
        Err(Error::config(
            "DiffusionRouter API is coming in February 2026. Not available yet.",
        ))
    }

    async fn complete_stream(
        &self,
        _request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        Err(Error::config(
            "DiffusionRouter API is coming in February 2026. Not available yet.",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coming_soon() {
        let config = ProviderConfig::new("test-key");
        let result = DiffusionRouterProvider::new(config);
        assert!(result.is_err());
    }
}
