//! vLLM provider for fast local and remote inference.
//!
//! vLLM is a fast and easy-to-use library for LLM inference and serving,
//! with state-of-the-art serving throughput. It can be deployed locally
//! or accessed remotely.
//!
//! # Features
//! - High-throughput LLM serving
//! - PagedAttention for efficient memory usage
//! - Supports streaming
//! - Local deployment or remote access
//! - Multi-GPU support

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// vLLM provider for fast inference
pub struct VLLMProvider {
    #[allow(dead_code)]
    base_url: String,
}

impl VLLMProvider {
    /// Create a new vLLM provider with base URL
    pub fn new(base_url: &str) -> Self {
        Self {
            base_url: base_url.to_string(),
        }
    }

    /// Create local vLLM provider (localhost:8000)
    pub fn local() -> Self {
        Self::new("http://localhost:8000")
    }

    /// Create from environment variable `VLLM_BASE_URL`
    pub fn from_env() -> Result<Self> {
        let base_url = std::env::var("VLLM_BASE_URL").map_err(|_| {
            Error::Configuration("VLLM_BASE_URL environment variable not set".to_string())
        })?;
        Ok(Self::new(&base_url))
    }

    /// Check if model is available (mock implementation)
    pub async fn check_model_available(&self, _model: &str) -> Result<bool> {
        // In real implementation, would call /v1/models endpoint
        Ok(true)
    }

    /// Get server stats
    pub async fn get_server_stats(&self) -> Result<ServerStats> {
        // Mock implementation
        Ok(ServerStats {
            total_gpu_memory_mb: 81920,
            used_gpu_memory_mb: 24576,
            temperature_c: 45.0,
            num_requests_running: 2,
            num_requests_waiting: 0,
        })
    }
}

/// Server statistics from vLLM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerStats {
    /// Total GPU memory in MB
    pub total_gpu_memory_mb: u64,
    /// Used GPU memory in MB
    pub used_gpu_memory_mb: u64,
    /// GPU temperature in Celsius
    pub temperature_c: f64,
    /// Currently running requests
    pub num_requests_running: u32,
    /// Waiting requests
    pub num_requests_waiting: u32,
}

impl ServerStats {
    /// Calculate GPU memory utilization percentage
    pub fn gpu_utilization_percent(&self) -> f64 {
        (self.used_gpu_memory_mb as f64 / self.total_gpu_memory_mb as f64) * 100.0
    }

    /// Check if GPU is overheating (above 80°C)
    pub fn is_overheating(&self) -> bool {
        self.temperature_c > 80.0
    }
}

/// Scheduling policy for vLLM
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SchedulingPolicy {
    /// Fcfs (First-Come-First-Served)
    #[default]
    Fcfs,
    /// Priority queue
    Priority,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vllm_provider_creation() {
        let provider = VLLMProvider::new("http://localhost:8000");
        assert_eq!(provider.base_url, "http://localhost:8000");
    }

    #[test]
    fn test_vllm_local() {
        let provider = VLLMProvider::local();
        assert_eq!(provider.base_url, "http://localhost:8000");
    }

    #[tokio::test]
    async fn test_check_model_available() {
        let provider = VLLMProvider::new("http://localhost:8000");
        let available = provider
            .check_model_available("meta-llama/Llama-2-7b")
            .await
            .unwrap();
        assert!(available);
    }

    #[tokio::test]
    async fn test_get_server_stats() {
        let provider = VLLMProvider::new("http://localhost:8000");
        let stats = provider.get_server_stats().await.unwrap();
        assert!(stats.total_gpu_memory_mb > 0);
        assert!(stats.used_gpu_memory_mb <= stats.total_gpu_memory_mb);
    }

    #[test]
    fn test_gpu_utilization() {
        let stats = ServerStats {
            total_gpu_memory_mb: 80000,
            used_gpu_memory_mb: 40000,
            temperature_c: 60.0,
            num_requests_running: 5,
            num_requests_waiting: 2,
        };
        assert_eq!(stats.gpu_utilization_percent(), 50.0);
    }

    #[test]
    fn test_is_overheating() {
        let cool_stats = ServerStats {
            total_gpu_memory_mb: 80000,
            used_gpu_memory_mb: 40000,
            temperature_c: 60.0,
            num_requests_running: 5,
            num_requests_waiting: 2,
        };
        assert!(!cool_stats.is_overheating());

        let hot_stats = ServerStats {
            total_gpu_memory_mb: 80000,
            used_gpu_memory_mb: 40000,
            temperature_c: 85.0,
            num_requests_running: 5,
            num_requests_waiting: 2,
        };
        assert!(hot_stats.is_overheating());
    }

    #[test]
    fn test_scheduling_policy_default() {
        assert_eq!(SchedulingPolicy::default(), SchedulingPolicy::Fcfs);
    }
}
