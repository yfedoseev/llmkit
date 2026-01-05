//! Portkey AI Gateway provider.
//!
//! Portkey provides AI gateway with observability and reliability features.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Portkey AI Gateway provider
pub struct PortkeyProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl PortkeyProvider {
    /// Create a new Portkey provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.portkey.ai/v1".to_string(),
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
        let api_key = std::env::var("PORTKEY_API_KEY")
            .map_err(|_| Error::Configuration("PORTKEY_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get gateway capabilities
    pub fn get_capabilities() -> PortkeyCapabilities {
        PortkeyCapabilities {
            name: "Portkey".to_string(),
            gateway_type: "AI Gateway".to_string(),
            features: vec![
                "Multi-provider routing".to_string(),
                "Automatic retries".to_string(),
                "Load balancing".to_string(),
                "Caching".to_string(),
                "Request logging".to_string(),
                "Cost tracking".to_string(),
                "Semantic caching".to_string(),
                "Guardrails".to_string(),
            ],
            supported_providers: vec![
                "openai".to_string(),
                "anthropic".to_string(),
                "google".to_string(),
                "azure".to_string(),
                "aws-bedrock".to_string(),
                "cohere".to_string(),
                "mistral".to_string(),
            ],
            openai_compatible: true,
        }
    }
}

/// Portkey gateway capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortkeyCapabilities {
    pub name: String,
    pub gateway_type: String,
    pub features: Vec<String>,
    pub supported_providers: Vec<String>,
    pub openai_compatible: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = PortkeyProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.portkey.ai/v1");
    }

    #[test]
    fn test_capabilities() {
        let caps = PortkeyProvider::get_capabilities();
        assert!(caps.openai_compatible);
        assert!(caps.features.len() >= 5);
    }
}
