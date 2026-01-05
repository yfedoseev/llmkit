//! Helicone provider - LLM observability platform.
//!
//! Helicone provides observability, monitoring, and proxy for LLM requests.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Helicone provider
pub struct HeliconeProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl HeliconeProvider {
    /// Create a new Helicone provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://oai.helicone.ai/v1".to_string(),
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
        let api_key = std::env::var("HELICONE_API_KEY")
            .map_err(|_| Error::Configuration("HELICONE_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get observability features
    pub fn get_features() -> HeliconeFeatures {
        HeliconeFeatures {
            name: "Helicone".to_string(),
            platform_type: "LLM Observability".to_string(),
            features: vec![
                "Request logging".to_string(),
                "Cost tracking".to_string(),
                "Latency monitoring".to_string(),
                "User analytics".to_string(),
                "Rate limiting".to_string(),
                "Caching".to_string(),
                "Prompt management".to_string(),
                "A/B testing".to_string(),
            ],
            proxy_endpoints: vec![
                "oai.helicone.ai".to_string(),
                "anthropic.helicone.ai".to_string(),
                "gateway.helicone.ai".to_string(),
            ],
            openai_compatible: true,
        }
    }
}

/// Helicone observability features
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeliconeFeatures {
    pub name: String,
    pub platform_type: String,
    pub features: Vec<String>,
    pub proxy_endpoints: Vec<String>,
    pub openai_compatible: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = HeliconeProvider::new("test-key");
        assert_eq!(provider.base_url, "https://oai.helicone.ai/v1");
    }

    #[test]
    fn test_features() {
        let features = HeliconeProvider::get_features();
        assert!(features.openai_compatible);
        assert!(features.features.len() >= 5);
    }
}
