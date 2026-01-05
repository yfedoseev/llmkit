//! Beam provider for serverless GPU infrastructure.
//!
//! Beam provides serverless GPU compute for AI workloads.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Beam provider
pub struct BeamProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl BeamProvider {
    /// Create a new Beam provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.beam.cloud".to_string(),
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
        let api_key = std::env::var("BEAM_API_KEY")
            .map_err(|_| Error::Configuration("BEAM_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get available GPU types
    pub async fn list_gpu_types(&self) -> Result<Vec<String>> {
        Ok(vec![
            "T4".to_string(),
            "A10G".to_string(),
            "A100-40GB".to_string(),
            "A100-80GB".to_string(),
        ])
    }

    /// Get GPU type details
    pub fn get_gpu_info(gpu_type: &str) -> Option<BeamGPUInfo> {
        match gpu_type {
            "T4" => Some(BeamGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 16,
                price_per_second: 0.00016,
                cold_start_ms: 5000,
            }),
            "A10G" => Some(BeamGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 24,
                price_per_second: 0.00028,
                cold_start_ms: 5000,
            }),
            "A100-40GB" => Some(BeamGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 40,
                price_per_second: 0.00069,
                cold_start_ms: 8000,
            }),
            "A100-80GB" => Some(BeamGPUInfo {
                name: gpu_type.to_string(),
                memory_gb: 80,
                price_per_second: 0.00105,
                cold_start_ms: 10000,
            }),
            _ => None,
        }
    }

    /// Get platform capabilities
    pub fn get_capabilities() -> BeamCapabilities {
        BeamCapabilities {
            name: "Beam".to_string(),
            features: vec![
                "serverless".to_string(),
                "auto-scaling".to_string(),
                "rest-api".to_string(),
                "webhooks".to_string(),
                "scheduling".to_string(),
                "volumes".to_string(),
            ],
            languages: vec!["python".to_string()],
        }
    }
}

/// Beam GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamGPUInfo {
    pub name: String,
    pub memory_gb: u32,
    pub price_per_second: f64,
    pub cold_start_ms: u32,
}

/// Beam platform capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeamCapabilities {
    pub name: String,
    pub features: Vec<String>,
    pub languages: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = BeamProvider::new("test-key");
        assert!(provider.base_url.contains("beam.cloud"));
    }

    #[tokio::test]
    async fn test_list_gpu_types() {
        let provider = BeamProvider::new("test-key");
        let gpus = provider.list_gpu_types().await.unwrap();
        assert!(gpus.len() >= 3);
    }

    #[test]
    fn test_gpu_info() {
        let info = BeamProvider::get_gpu_info("A100-80GB").unwrap();
        assert_eq!(info.memory_gb, 80);
    }
}
