//! CoreWeave provider for GPU cloud computing.
//!
//! CoreWeave provides specialized cloud infrastructure for GPU workloads.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// CoreWeave provider
pub struct CoreWeaveProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl CoreWeaveProvider {
    /// Create a new CoreWeave provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.coreweave.com/v1".to_string(),
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
        let api_key = std::env::var("COREWEAVE_API_KEY")
            .map_err(|_| Error::Configuration("COREWEAVE_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get available GPU types
    pub async fn list_gpu_types(&self) -> Result<Vec<String>> {
        Ok(vec![
            "RTX-4090".to_string(),
            "RTX-A6000".to_string(),
            "A100-40GB-PCIe".to_string(),
            "A100-80GB-SXM".to_string(),
            "H100-80GB-SXM".to_string(),
            "H100-80GB-PCIe".to_string(),
        ])
    }

    /// Get GPU type details
    pub fn get_gpu_info(gpu_type: &str) -> Option<CoreWeaveGPUInfo> {
        match gpu_type {
            "RTX-4090" => Some(CoreWeaveGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 24,
                price_per_hour: 0.74,
                availability: "high".to_string(),
            }),
            "RTX-A6000" => Some(CoreWeaveGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 48,
                price_per_hour: 1.28,
                availability: "high".to_string(),
            }),
            "A100-80GB-SXM" => Some(CoreWeaveGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 80,
                price_per_hour: 2.21,
                availability: "medium".to_string(),
            }),
            "H100-80GB-SXM" => Some(CoreWeaveGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 80,
                price_per_hour: 4.25,
                availability: "limited".to_string(),
            }),
            _ => None,
        }
    }

    /// Get platform capabilities
    pub fn get_capabilities() -> CoreWeaveCapabilities {
        CoreWeaveCapabilities {
            name: "CoreWeave".to_string(),
            features: vec![
                "kubernetes".to_string(),
                "bare-metal".to_string(),
                "object-storage".to_string(),
                "inference-endpoints".to_string(),
                "nvlink".to_string(),
                "infiniband".to_string(),
            ],
            regions: vec!["ORD1".to_string(), "LAS1".to_string(), "LGA1".to_string()],
        }
    }
}

/// CoreWeave GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreWeaveGPUInfo {
    pub name: String,
    pub memory_gb: u32,
    pub price_per_hour: f64,
    pub availability: String,
}

/// CoreWeave platform capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoreWeaveCapabilities {
    pub name: String,
    pub features: Vec<String>,
    pub regions: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = CoreWeaveProvider::new("test-key");
        assert!(provider.base_url.contains("coreweave.com"));
    }

    #[tokio::test]
    async fn test_list_gpu_types() {
        let provider = CoreWeaveProvider::new("test-key");
        let gpus = provider.list_gpu_types().await.unwrap();
        assert!(gpus.len() >= 4);
    }

    #[test]
    fn test_gpu_info() {
        let info = CoreWeaveProvider::get_gpu_info("H100-80GB-SXM").unwrap();
        assert_eq!(info.memory_gb, 80);
    }
}
