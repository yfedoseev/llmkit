//! Nscale provider for sustainable AI infrastructure.
//!
//! Nscale provides sustainable AI compute with a focus on green energy.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Nscale provider
pub struct NscaleProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl NscaleProvider {
    /// Create a new Nscale provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.nscale.ai/v1".to_string(),
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
        let api_key = std::env::var("NSCALE_API_KEY")
            .map_err(|_| Error::Configuration("NSCALE_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get available services
    pub async fn list_services(&self) -> Result<Vec<String>> {
        Ok(vec![
            "inference".to_string(),
            "training".to_string(),
            "fine-tuning".to_string(),
        ])
    }

    /// Get platform info
    pub fn get_platform_info() -> NscalePlatformInfo {
        NscalePlatformInfo {
            name: "Nscale".to_string(),
            focus: "Sustainable AI".to_string(),
            features: vec![
                "green-energy".to_string(),
                "gpu-compute".to_string(),
                "model-hosting".to_string(),
                "fine-tuning".to_string(),
            ],
            data_centers: vec!["EU".to_string()],
            sustainability: NscaleSustainability {
                renewable_energy_percent: 100,
                carbon_neutral: true,
            },
        }
    }
}

/// Nscale platform information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NscalePlatformInfo {
    pub name: String,
    pub focus: String,
    pub features: Vec<String>,
    pub data_centers: Vec<String>,
    pub sustainability: NscaleSustainability,
}

/// Nscale sustainability info
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NscaleSustainability {
    pub renewable_energy_percent: u32,
    pub carbon_neutral: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = NscaleProvider::new("test-key");
        assert!(provider.base_url.contains("nscale"));
    }

    #[tokio::test]
    async fn test_list_services() {
        let provider = NscaleProvider::new("test-key");
        let services = provider.list_services().await.unwrap();
        assert!(!services.is_empty());
    }

    #[test]
    fn test_platform_info() {
        let info = NscaleProvider::get_platform_info();
        assert!(info.sustainability.carbon_neutral);
    }
}
