//! Martian AI provider - Model router with fallback.
//!
//! Martian (withmartian.com) provides intelligent model routing and fallback.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Martian AI provider
pub struct MartianProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl MartianProvider {
    /// Create a new Martian provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.withmartian.com/v1".to_string(),
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
        let api_key = std::env::var("MARTIAN_API_KEY")
            .map_err(|_| Error::Configuration("MARTIAN_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get router capabilities
    pub fn get_capabilities() -> MartianCapabilities {
        MartianCapabilities {
            name: "Martian".to_string(),
            features: vec![
                "Model routing".to_string(),
                "Automatic fallback".to_string(),
                "Cost tracking".to_string(),
                "Rate limit handling".to_string(),
                "Request caching".to_string(),
            ],
            supported_providers: vec![
                "openai".to_string(),
                "anthropic".to_string(),
                "google".to_string(),
                "mistral".to_string(),
                "groq".to_string(),
            ],
            openai_compatible: true,
        }
    }
}

/// Martian router capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MartianCapabilities {
    pub name: String,
    pub features: Vec<String>,
    pub supported_providers: Vec<String>,
    pub openai_compatible: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = MartianProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.withmartian.com/v1");
    }

    #[test]
    fn test_capabilities() {
        let caps = MartianProvider::get_capabilities();
        assert!(caps.openai_compatible);
        assert!(caps.features.len() >= 3);
    }
}
