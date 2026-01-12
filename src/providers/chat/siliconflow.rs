//! SiliconFlow provider - High-performance inference platform.
//!
//! SiliconFlow provides optimized inference for open-source models.

use crate::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// SiliconFlow provider
pub struct SiliconFlowProvider {
    #[allow(dead_code)]
    api_key: String,
    #[allow(dead_code)]
    base_url: String,
}

impl SiliconFlowProvider {
    /// Create a new SiliconFlow provider
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            base_url: "https://api.siliconflow.cn/v1".to_string(),
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
        let api_key = std::env::var("SILICONFLOW_API_KEY")
            .map_err(|_| Error::Configuration("SILICONFLOW_API_KEY not set".to_string()))?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        Ok(vec![
            "Qwen/Qwen2.5-72B-Instruct".to_string(),
            "deepseek-ai/DeepSeek-V2.5".to_string(),
            "meta-llama/Meta-Llama-3.1-70B-Instruct".to_string(),
            "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string(),
            "THUDM/glm-4-9b-chat".to_string(),
            "01-ai/Yi-1.5-34B-Chat".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<SiliconFlowModelInfo> {
        match model {
            "Qwen/Qwen2.5-72B-Instruct" => Some(SiliconFlowModelInfo {
                name: model.to_string(),
                parameters: "72B".to_string(),
                context_length: 131072,
                price_per_1m_tokens: 0.35,
                optimized: true,
            }),
            "deepseek-ai/DeepSeek-V2.5" => Some(SiliconFlowModelInfo {
                name: model.to_string(),
                parameters: "236B MoE".to_string(),
                context_length: 65536,
                price_per_1m_tokens: 0.28,
                optimized: true,
            }),
            "meta-llama/Meta-Llama-3.1-70B-Instruct" => Some(SiliconFlowModelInfo {
                name: model.to_string(),
                parameters: "70B".to_string(),
                context_length: 128000,
                price_per_1m_tokens: 0.30,
                optimized: true,
            }),
            "meta-llama/Meta-Llama-3.1-8B-Instruct" => Some(SiliconFlowModelInfo {
                name: model.to_string(),
                parameters: "8B".to_string(),
                context_length: 128000,
                price_per_1m_tokens: 0.05,
                optimized: true,
            }),
            _ => None,
        }
    }
}

/// SiliconFlow model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SiliconFlowModelInfo {
    pub name: String,
    pub parameters: String,
    pub context_length: u32,
    pub price_per_1m_tokens: f64,
    pub optimized: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = SiliconFlowProvider::new("test-key");
        assert_eq!(provider.base_url, "https://api.siliconflow.cn/v1");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = SiliconFlowProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(models.len() >= 4);
    }

    #[test]
    fn test_model_info() {
        let info = SiliconFlowProvider::get_model_info("Qwen/Qwen2.5-72B-Instruct").unwrap();
        assert_eq!(info.parameters, "72B");
        assert!(info.optimized);
    }
}
