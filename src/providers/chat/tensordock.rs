//! TensorDock provider for affordable GPU cloud.
//!
//! TensorDock provides low-cost GPU cloud computing.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// TensorDock provider
pub struct TensorDockProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl TensorDockProvider {
    /// Create a new TensorDock provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://marketplace.tensordock.com/api/v0".to_string(),
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
        let api_key = std::env::var("TENSORDOCK_API_KEY")
            .map_err(|_| Error::Configuration("TENSORDOCK_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get available GPU types
    pub async fn list_gpu_types(&self) -> Result<Vec<String>> {
        Ok(vec![
            "RTX-3090".to_string(),
            "RTX-4090".to_string(),
            "A100-40GB".to_string(),
            "A100-80GB".to_string(),
            "H100".to_string(),
        ])
    }

    /// Get GPU type details
    pub fn get_gpu_info(gpu_type: &str) -> Option<TensorDockGPUInfo> {
        match gpu_type {
            "RTX-3090" => Some(TensorDockGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 24,
                price_per_hour: 0.20,
                provider_type: "community".to_string(),
            }),
            "RTX-4090" => Some(TensorDockGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 24,
                price_per_hour: 0.35,
                provider_type: "community".to_string(),
            }),
            "A100-40GB" => Some(TensorDockGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 40,
                price_per_hour: 0.99,
                provider_type: "enterprise".to_string(),
            }),
            "A100-80GB" => Some(TensorDockGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 80,
                price_per_hour: 1.49,
                provider_type: "enterprise".to_string(),
            }),
            "H100" => Some(TensorDockGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 80,
                price_per_hour: 2.49,
                provider_type: "enterprise".to_string(),
            }),
            _ => None,
        }
    }

    /// Get platform capabilities
    pub fn get_capabilities() -> TensorDockCapabilities {
        TensorDockCapabilities {
            name: "TensorDock".to_string(),
            features: vec![
                "spot-instances".to_string(),
                "community-gpus".to_string(),
                "on-demand".to_string(),
                "ssh-access".to_string(),
            ],
            pricing_model: "pay-as-you-go".to_string(),
        }
    }
}

/// TensorDock GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDockGPUInfo {
    pub name: String,
    pub memory_gb: u32,
    pub price_per_hour: f64,
    pub provider_type: String,
}

/// TensorDock platform capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDockCapabilities {
    pub name: String,
    pub features: Vec<String>,
    pub pricing_model: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = TensorDockProvider::new("test-key");
        assert!(provider.base_url.contains("tensordock.com"));
    }

    #[tokio::test]
    async fn test_list_gpu_types() {
        let provider = TensorDockProvider::new("test-key");
        let gpus = provider.list_gpu_types().await.unwrap();
        assert!(gpus.len() >= 4);
    }

    #[test]
    fn test_gpu_info() {
        let info = TensorDockProvider::get_gpu_info("RTX-3090").unwrap();
        assert!(info.price_per_hour < 0.30);
    }
}
