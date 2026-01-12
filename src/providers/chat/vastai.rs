//! Vast.ai provider for GPU marketplace.
//!
//! Vast.ai is a marketplace for renting GPU compute from various providers.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Vast.ai provider
pub struct VastAIProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl VastAIProvider {
    /// Create a new Vast.ai provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://console.vast.ai/api/v0".to_string(),
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
        let api_key = std::env::var("VASTAI_API_KEY")
            .map_err(|_| Error::Configuration("VASTAI_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get available GPU types
    pub async fn list_gpu_types(&self) -> Result<Vec<String>> {
        Ok(vec![
            "RTX-3090".to_string(),
            "RTX-3090-Ti".to_string(),
            "RTX-4090".to_string(),
            "A100-40GB".to_string(),
            "A100-80GB".to_string(),
            "H100-80GB".to_string(),
            "A6000".to_string(),
        ])
    }

    /// Get GPU type details
    pub fn get_gpu_info(gpu_type: &str) -> Option<VastAIGPUInfo> {
        match gpu_type {
            "RTX-3090" => Some(VastAIGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 24,
                min_price_per_hour: 0.10,
                availability: "high".to_string(),
                reliability_score: 95.0,
            }),
            "RTX-4090" => Some(VastAIGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 24,
                min_price_per_hour: 0.25,
                availability: "high".to_string(),
                reliability_score: 96.0,
            }),
            "A100-40GB" => Some(VastAIGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 40,
                min_price_per_hour: 0.70,
                availability: "medium".to_string(),
                reliability_score: 97.0,
            }),
            "A100-80GB" => Some(VastAIGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 80,
                min_price_per_hour: 1.20,
                availability: "medium".to_string(),
                reliability_score: 97.0,
            }),
            "H100-80GB" => Some(VastAIGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 80,
                min_price_per_hour: 2.50,
                availability: "limited".to_string(),
                reliability_score: 98.0,
            }),
            _ => None,
        }
    }

    /// Get platform capabilities
    pub fn get_capabilities() -> VastAICapabilities {
        VastAICapabilities {
            name: "Vast.ai".to_string(),
            features: vec![
                "marketplace".to_string(),
                "spot-pricing".to_string(),
                "ssh-access".to_string(),
                "docker".to_string(),
                "jupyter".to_string(),
                "templates".to_string(),
            ],
            pricing_model: "auction-based".to_string(),
        }
    }
}

/// Vast.ai GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VastAIGPUInfo {
    pub name: String,
    pub memory_gb: u32,
    pub min_price_per_hour: f64,
    pub availability: String,
    pub reliability_score: f64,
}

/// Vast.ai platform capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VastAICapabilities {
    pub name: String,
    pub features: Vec<String>,
    pub pricing_model: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = VastAIProvider::new("test-key");
        assert!(provider.base_url.contains("vast.ai"));
    }

    #[tokio::test]
    async fn test_list_gpu_types() {
        let provider = VastAIProvider::new("test-key");
        let gpus = provider.list_gpu_types().await.unwrap();
        assert!(gpus.len() >= 5);
    }

    #[test]
    fn test_gpu_info() {
        let info = VastAIProvider::get_gpu_info("RTX-3090").unwrap();
        assert!(info.min_price_per_hour < 0.20);
    }
}
