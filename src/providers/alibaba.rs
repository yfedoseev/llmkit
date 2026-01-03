//! Alibaba Qwen provider for advanced Chinese and multilingual LLM services.
//!
//! Alibaba's Qwen models provide state-of-the-art performance for Chinese language
//! tasks while maintaining strong multilingual capabilities. Available through
//! Alibaba's DashScope platform.
//!
//! # Features
//! - Multiple model sizes and specializations
//! - Vision capabilities with Qwen-VL
//! - Strong Chinese and multilingual support
//! - Function calling and JSON output
//! - Real-time knowledge cutoff

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Alibaba Qwen provider for multilingual LLM services
pub struct AlibabaQwenProvider {
    #[allow(dead_code)]
    api_key: String,
}

impl AlibabaQwenProvider {
    /// Create a new Alibaba Qwen provider with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
        }
    }

    /// Create from environment variable `ALIBABA_QWEN_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("ALIBABA_QWEN_API_KEY").map_err(|_| {
            Error::Configuration("ALIBABA_QWEN_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Get list of available Qwen models
    pub async fn list_models(&self) -> Result<Vec<String>> {
        // Mock implementation
        Ok(vec![
            "qwen-turbo".to_string(),
            "qwen-plus".to_string(),
            "qwen-max".to_string(),
            "qwen-max-longcontext".to_string(),
            "qwen-vl-plus".to_string(),
            "qwen-vl-max".to_string(),
        ])
    }

    /// Get model details
    pub fn get_model_info(model: &str) -> Option<AlibabaModelInfo> {
        match model {
            "qwen-turbo" => Some(AlibabaModelInfo {
                name: "qwen-turbo".to_string(),
                context_window: 8000,
                supports_vision: false,
                supports_function_call: true,
                max_output_tokens: 2000,
            }),
            "qwen-plus" => Some(AlibabaModelInfo {
                name: "qwen-plus".to_string(),
                context_window: 32000,
                supports_vision: false,
                supports_function_call: true,
                max_output_tokens: 4000,
            }),
            "qwen-max" => Some(AlibabaModelInfo {
                name: "qwen-max".to_string(),
                context_window: 32000,
                supports_vision: false,
                supports_function_call: true,
                max_output_tokens: 8000,
            }),
            "qwen-max-longcontext" => Some(AlibabaModelInfo {
                name: "qwen-max-longcontext".to_string(),
                context_window: 200000,
                supports_vision: false,
                supports_function_call: true,
                max_output_tokens: 8000,
            }),
            "qwen-vl-plus" => Some(AlibabaModelInfo {
                name: "qwen-vl-plus".to_string(),
                context_window: 16000,
                supports_vision: true,
                supports_function_call: false,
                max_output_tokens: 1000,
            }),
            "qwen-vl-max" => Some(AlibabaModelInfo {
                name: "qwen-vl-max".to_string(),
                context_window: 32000,
                supports_vision: true,
                supports_function_call: false,
                max_output_tokens: 2000,
            }),
            _ => None,
        }
    }
}

/// Alibaba model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlibabaModelInfo {
    /// Model name
    pub name: String,
    /// Context window size
    pub context_window: u32,
    /// Whether this model supports vision/images
    pub supports_vision: bool,
    /// Whether this model supports function calling
    pub supports_function_call: bool,
    /// Maximum output tokens
    pub max_output_tokens: u32,
}

/// Model specialization for Alibaba Qwen
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ModelSpecialization {
    /// General purpose language model
    #[default]
    General,
    /// Vision-language model
    Vision,
    /// Code generation specialist
    Code,
    /// Mathematical reasoning specialist
    Math,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alibaba_provider_creation() {
        let provider = AlibabaQwenProvider::new("test-key");
        assert_eq!(provider.api_key, "test-key");
    }

    #[tokio::test]
    async fn test_list_models() {
        let provider = AlibabaQwenProvider::new("test-key");
        let models = provider.list_models().await.unwrap();
        assert!(!models.is_empty());
        assert!(models.contains(&"qwen-max".to_string()));
        assert!(models.contains(&"qwen-vl-max".to_string()));
    }

    #[test]
    fn test_get_model_info() {
        let info = AlibabaQwenProvider::get_model_info("qwen-max").unwrap();
        assert_eq!(info.name, "qwen-max");
        assert!(info.supports_function_call);
        assert!(!info.supports_vision);
        assert_eq!(info.context_window, 32000);
    }

    #[test]
    fn test_vision_model_info() {
        let info = AlibabaQwenProvider::get_model_info("qwen-vl-max").unwrap();
        assert!(info.supports_vision);
        assert!(!info.supports_function_call);
    }

    #[test]
    fn test_model_info_invalid() {
        let info = AlibabaQwenProvider::get_model_info("invalid-model");
        assert!(info.is_none());
    }

    #[test]
    fn test_model_specialization_default() {
        assert_eq!(ModelSpecialization::default(), ModelSpecialization::General);
    }

    #[test]
    fn test_qwen_context_windows() {
        let turbo = AlibabaQwenProvider::get_model_info("qwen-turbo").unwrap();
        let plus = AlibabaQwenProvider::get_model_info("qwen-plus").unwrap();
        let max = AlibabaQwenProvider::get_model_info("qwen-max").unwrap();
        let long = AlibabaQwenProvider::get_model_info("qwen-max-longcontext").unwrap();

        assert!(turbo.context_window < plus.context_window);
        assert_eq!(plus.context_window, max.context_window);
        assert!(max.context_window < long.context_window);
    }
}
