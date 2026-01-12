//! Provider trait and related types.
//!
//! This module defines the core `Provider` trait that all LLM providers must implement.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::error::Result;
use crate::types::{
    BatchJob, BatchRequest, BatchResult, CompletionRequest, CompletionResponse, StreamChunk,
    TokenCountRequest, TokenCountResult,
};

/// Core trait for LLM providers.
///
/// All providers (Anthropic, OpenAI, etc.) implement this trait to provide
/// a unified interface for completion requests.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Returns the provider's name (e.g., "anthropic", "openai").
    fn name(&self) -> &str;

    /// Make a completion request.
    ///
    /// This is the primary method for getting a response from the LLM.
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse>;

    /// Make a streaming completion request.
    ///
    /// Returns a stream of chunks that can be processed as they arrive.
    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>>;

    /// Check if this provider supports tool/function calling.
    fn supports_tools(&self) -> bool {
        true
    }

    /// Check if this provider supports vision (image input).
    fn supports_vision(&self) -> bool {
        false
    }

    /// Check if this provider supports streaming.
    fn supports_streaming(&self) -> bool {
        true
    }

    /// Get the list of models supported by this provider.
    ///
    /// Returns None if the provider doesn't maintain a static model list
    /// (e.g., OpenRouter which proxies many models).
    fn supported_models(&self) -> Option<&[&str]> {
        None
    }

    /// Get the default model for this provider.
    fn default_model(&self) -> Option<&str> {
        None
    }

    /// Count tokens in a request.
    ///
    /// This allows estimation of token counts before making a completion request,
    /// useful for cost estimation and context window management.
    ///
    /// Not all providers support token counting. The default implementation
    /// returns an error indicating the feature is not supported.
    async fn count_tokens(&self, _request: TokenCountRequest) -> Result<TokenCountResult> {
        Err(crate::error::Error::other(format!(
            "Token counting not supported by {} provider",
            self.name()
        )))
    }

    /// Check if this provider supports token counting.
    fn supports_token_counting(&self) -> bool {
        false
    }

    // ========== Batch Processing ==========

    /// Create a batch of requests for asynchronous processing.
    ///
    /// Batch processing can be significantly cheaper (up to 50% on some providers)
    /// and is ideal for non-time-sensitive workloads.
    ///
    /// Not all providers support batch processing. The default implementation
    /// returns an error indicating the feature is not supported.
    async fn create_batch(&self, _requests: Vec<BatchRequest>) -> Result<BatchJob> {
        Err(crate::error::Error::other(format!(
            "Batch processing not supported by {} provider",
            self.name()
        )))
    }

    /// Get the status of a batch job.
    async fn get_batch(&self, _batch_id: &str) -> Result<BatchJob> {
        Err(crate::error::Error::other(format!(
            "Batch processing not supported by {} provider",
            self.name()
        )))
    }

    /// Get the results of a completed batch.
    async fn get_batch_results(&self, _batch_id: &str) -> Result<Vec<BatchResult>> {
        Err(crate::error::Error::other(format!(
            "Batch processing not supported by {} provider",
            self.name()
        )))
    }

    /// Cancel a batch job.
    async fn cancel_batch(&self, _batch_id: &str) -> Result<BatchJob> {
        Err(crate::error::Error::other(format!(
            "Batch processing not supported by {} provider",
            self.name()
        )))
    }

    /// List recent batch jobs.
    async fn list_batches(&self, _limit: Option<u32>) -> Result<Vec<BatchJob>> {
        Err(crate::error::Error::other(format!(
            "Batch processing not supported by {} provider",
            self.name()
        )))
    }

    /// Check if this provider supports batch processing.
    fn supports_batch(&self) -> bool {
        false
    }
}

/// Configuration for a provider.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    /// API key for authentication
    pub api_key: Option<String>,

    /// Base URL for the API (override default)
    pub base_url: Option<String>,

    /// Request timeout
    pub timeout: std::time::Duration,

    /// Maximum number of retries for transient errors
    pub max_retries: u32,

    /// Organization ID (for providers that support it)
    pub organization_id: Option<String>,

    /// Custom headers to include in requests
    pub custom_headers: std::collections::HashMap<String, String>,
}

impl Default for ProviderConfig {
    fn default() -> Self {
        Self {
            api_key: None,
            base_url: None,
            timeout: std::time::Duration::from_secs(120),
            max_retries: 2,
            organization_id: None,
            custom_headers: std::collections::HashMap::new(),
        }
    }
}

impl ProviderConfig {
    /// Create a new provider config with an API key.
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: Some(api_key.into()),
            ..Default::default()
        }
    }

    /// Create a provider config from an environment variable.
    pub fn from_env(env_var: &str) -> Self {
        Self {
            api_key: std::env::var(env_var).ok(),
            ..Default::default()
        }
    }

    /// Builder: Set the base URL.
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = Some(url.into());
        self
    }

    /// Builder: Set the timeout.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Builder: Set max retries.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Builder: Set organization ID.
    pub fn with_organization(mut self, org_id: impl Into<String>) -> Self {
        self.organization_id = Some(org_id.into());
        self
    }

    /// Builder: Add a custom header.
    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.custom_headers.insert(key.into(), value.into());
        self
    }

    /// Get the API key, returning an error if not set.
    pub fn require_api_key(&self) -> Result<&str> {
        self.api_key
            .as_deref()
            .ok_or_else(|| crate::error::Error::config("API key is required"))
    }
}

/// Model information for routing and selection.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model identifier
    pub id: String,

    /// Provider name
    pub provider: String,

    /// Human-readable name
    pub name: String,

    /// Maximum context length in tokens
    pub context_length: u32,

    /// Cost per 1K input tokens (in USD)
    pub input_cost_per_1k: Option<f64>,

    /// Cost per 1K output tokens (in USD)
    pub output_cost_per_1k: Option<f64>,

    /// Whether the model supports tools
    pub supports_tools: bool,

    /// Whether the model supports vision
    pub supports_vision: bool,

    /// Whether the model supports streaming
    pub supports_streaming: bool,
}

impl ModelInfo {
    /// Estimate cost for a request.
    pub fn estimate_cost(&self, input_tokens: u32, output_tokens: u32) -> Option<f64> {
        let input_cost = self.input_cost_per_1k? * (input_tokens as f64 / 1000.0);
        let output_cost = self.output_cost_per_1k? * (output_tokens as f64 / 1000.0);
        Some(input_cost + output_cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_config_builder() {
        let config = ProviderConfig::new("test-key")
            .with_base_url("https://api.example.com")
            .with_timeout(std::time::Duration::from_secs(60))
            .with_max_retries(3);

        assert_eq!(config.api_key, Some("test-key".to_string()));
        assert_eq!(config.base_url, Some("https://api.example.com".to_string()));
        assert_eq!(config.timeout, std::time::Duration::from_secs(60));
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_model_cost_estimation() {
        let model = ModelInfo {
            id: "test-model".to_string(),
            provider: "test".to_string(),
            name: "Test Model".to_string(),
            context_length: 128000,
            input_cost_per_1k: Some(0.003),
            output_cost_per_1k: Some(0.015),
            supports_tools: true,
            supports_vision: true,
            supports_streaming: true,
        };

        let cost = model.estimate_cost(1000, 500).unwrap();
        assert!((cost - 0.0105).abs() < 0.0001); // 0.003 + 0.0075
    }
}
