//! QwQ provider for advanced Chinese reasoning and language understanding.
//!
//! QwQ is Alibaba's advanced reasoning model designed for complex tasks
//! with strong performance in Chinese language understanding and logic.
//!
//! # Features
//! - Advanced reasoning capabilities
//! - Strong Chinese language support
//! - Multi-turn conversations
//! - Structured output support

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// QwQ provider for advanced reasoning
pub struct QwQProvider {
    #[allow(dead_code)]
    api_key: String,
}

impl QwQProvider {
    /// Create a new QwQ provider with API key
    pub fn new(api_key: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
        }
    }

    /// Create from environment variable `QWQ_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("QWQ_API_KEY").map_err(|_| {
            Error::Configuration("QWQ_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key))
    }

    /// Check if model is supported
    pub fn is_supported_model(model: &str) -> bool {
        model.starts_with("qwq-") || model == "qwq"
    }

    /// Get default model
    pub fn default_model() -> &'static str {
        "qwq-32b-preview"
    }
}

/// Reasoning level for QwQ
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ReasoningLevel {
    /// Basic reasoning
    Low,
    /// Standard reasoning
    #[default]
    Standard,
    /// Advanced reasoning with more computation
    Advanced,
    /// Maximum reasoning depth
    Maximum,
}

impl ReasoningLevel {
    /// Get reasoning budget tokens for this level
    pub fn reasoning_budget(&self) -> u32 {
        match self {
            ReasoningLevel::Low => 1024,
            ReasoningLevel::Standard => 4096,
            ReasoningLevel::Advanced => 8192,
            ReasoningLevel::Maximum => 16384,
        }
    }
}

/// QwQ reasoning response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningResponse {
    /// Thinking process (if available)
    pub thinking: Option<String>,
    /// Final output text
    pub output: String,
    /// Reasoning level used
    pub reasoning_level: ReasoningLevel,
    /// Tokens used in thinking
    pub thinking_tokens: u32,
    /// Tokens used in output
    pub output_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qwq_provider_creation() {
        let provider = QwQProvider::new("test-key");
        assert_eq!(provider.api_key, "test-key");
    }

    #[test]
    fn test_is_supported_model() {
        assert!(QwQProvider::is_supported_model("qwq-32b-preview"));
        assert!(QwQProvider::is_supported_model("qwq-14b"));
        assert!(QwQProvider::is_supported_model("qwq"));
        assert!(!QwQProvider::is_supported_model("gpt-4"));
    }

    #[test]
    fn test_default_model() {
        assert_eq!(QwQProvider::default_model(), "qwq-32b-preview");
    }

    #[test]
    fn test_reasoning_level_budget() {
        assert_eq!(ReasoningLevel::Low.reasoning_budget(), 1024);
        assert_eq!(ReasoningLevel::Standard.reasoning_budget(), 4096);
        assert_eq!(ReasoningLevel::Advanced.reasoning_budget(), 8192);
        assert_eq!(ReasoningLevel::Maximum.reasoning_budget(), 16384);
    }

    #[test]
    fn test_reasoning_response() {
        let response = ReasoningResponse {
            thinking: Some("Let me think...".to_string()),
            output: "The answer is 42".to_string(),
            reasoning_level: ReasoningLevel::Advanced,
            thinking_tokens: 512,
            output_tokens: 10,
        };
        assert_eq!(response.thinking_tokens + response.output_tokens, 522);
    }

    #[test]
    fn test_reasoning_level_default() {
        assert_eq!(ReasoningLevel::default(), ReasoningLevel::Standard);
    }
}
