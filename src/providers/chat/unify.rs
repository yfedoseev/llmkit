//! Unify AI provider - LLM router and gateway.
//!
//! Unify provides intelligent routing across multiple LLM providers.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Unify AI provider
pub struct UnifyProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl UnifyProvider {
    /// Create a new Unify provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.unify.ai/v0".to_string(),
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
        let api_key = std::env::var("UNIFY_API_KEY")
            .map_err(|_| Error::Configuration("UNIFY_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of routing modes
    pub async fn list_routing_modes(&self) -> Result<Vec<String>> {
        Ok(vec![
            "lowest-cost".to_string(),
            "lowest-latency".to_string(),
            "highest-quality".to_string(),
            "balanced".to_string(),
        ])
    }

    /// Get supported providers
    pub fn get_supported_providers() -> Vec<String> {
        vec![
            "openai".to_string(),
            "anthropic".to_string(),
            "google".to_string(),
            "mistral".to_string(),
            "cohere".to_string(),
            "together".to_string(),
            "groq".to_string(),
            "fireworks".to_string(),
        ]
    }

    /// Get router info
    pub fn get_router_info() -> UnifyRouterInfo {
        UnifyRouterInfo {
            name: "Unify".to_string(),
            router_type: "LLM Gateway".to_string(),
            features: vec![
                "Cost optimization".to_string(),
                "Latency optimization".to_string(),
                "Quality routing".to_string(),
                "Provider fallback".to_string(),
                "Usage analytics".to_string(),
            ],
            openai_compatible: true,
        }
    }
}

/// Unify router information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifyRouterInfo {
    pub name: String,
    pub router_type: String,
    pub features: Vec<String>,
    pub openai_compatible: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = UnifyProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.unify.ai/v0");
    }

    #[tokio::test]
    async fn test_list_routing_modes() {
        let provider = UnifyProvider::new("test-key");
        let modes = provider.list_routing_modes().await.unwrap();
        assert!(modes.len() >= 3);
    }

    #[test]
    fn test_router_info() {
        let info = UnifyProvider::get_router_info();
        assert!(info.openai_compatible);
    }
}
