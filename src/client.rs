//! Parlique client for unified LLM access.
//!
//! The `ParliqueClient` provides a unified interface to interact with multiple LLM providers.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use tokio::time::sleep;

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::retry::RetryConfig;
use crate::types::{
    BatchJob, BatchRequest, BatchResult, CompletionRequest, CompletionResponse, StreamChunk,
    TokenCountRequest, TokenCountResult,
};

/// A dynamic retrying provider that wraps `Arc<dyn Provider>`.
///
/// This enables retry logic at the client level without requiring
/// static typing of the underlying provider.
struct DynamicRetryingProvider {
    inner: Arc<dyn Provider>,
    config: RetryConfig,
}

impl DynamicRetryingProvider {
    /// Execute an operation with retry logic.
    async fn execute_with_retry<T, F, Fut>(&self, operation_name: &str, mut f: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut last_error: Option<Error> = None;

        for attempt in 0..=self.config.max_retries {
            match f().await {
                Ok(result) => {
                    if attempt > 0 {
                        tracing::info!(
                            provider = %self.inner.name(),
                            operation = %operation_name,
                            attempt = attempt + 1,
                            "Operation succeeded after retry"
                        );
                    }
                    return Ok(result);
                }
                Err(e) => {
                    if !e.is_retryable() {
                        tracing::debug!(
                            provider = %self.inner.name(),
                            operation = %operation_name,
                            error = %e,
                            "Non-retryable error, failing immediately"
                        );
                        return Err(e);
                    }

                    if attempt < self.config.max_retries {
                        // Calculate delay, respecting retry-after header if present
                        let delay = e
                            .retry_after()
                            .unwrap_or_else(|| self.config.delay_for_attempt(attempt));

                        tracing::warn!(
                            provider = %self.inner.name(),
                            operation = %operation_name,
                            attempt = attempt + 1,
                            max_retries = self.config.max_retries,
                            delay_ms = delay.as_millis(),
                            error = %e,
                            "Retryable error, will retry after delay"
                        );

                        sleep(delay).await;
                    }

                    last_error = Some(e);
                }
            }
        }

        tracing::error!(
            provider = %self.inner.name(),
            operation = %operation_name,
            max_retries = self.config.max_retries,
            "All retry attempts exhausted"
        );

        Err(last_error.unwrap_or_else(|| Error::other("Unknown retry failure")))
    }
}

#[async_trait]
impl Provider for DynamicRetryingProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let request = Arc::new(request);
        self.execute_with_retry("complete", || {
            let request = (*request).clone();
            let inner = Arc::clone(&self.inner);
            async move { inner.complete(request).await }
        })
        .await
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let request = Arc::new(request);
        self.execute_with_retry("complete_stream", || {
            let request = (*request).clone();
            let inner = Arc::clone(&self.inner);
            async move { inner.complete_stream(request).await }
        })
        .await
    }

    fn supports_tools(&self) -> bool {
        self.inner.supports_tools()
    }

    fn supports_vision(&self) -> bool {
        self.inner.supports_vision()
    }

    fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }

    async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCountResult> {
        let request = Arc::new(request);
        self.execute_with_retry("count_tokens", || {
            let request = (*request).clone();
            let inner = Arc::clone(&self.inner);
            async move { inner.count_tokens(request).await }
        })
        .await
    }

    fn supports_token_counting(&self) -> bool {
        self.inner.supports_token_counting()
    }

    async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchJob> {
        self.inner.create_batch(requests).await
    }

    async fn get_batch(&self, batch_id: &str) -> Result<BatchJob> {
        self.inner.get_batch(batch_id).await
    }

    async fn get_batch_results(&self, batch_id: &str) -> Result<Vec<BatchResult>> {
        self.inner.get_batch_results(batch_id).await
    }

    async fn cancel_batch(&self, batch_id: &str) -> Result<BatchJob> {
        self.inner.cancel_batch(batch_id).await
    }

    async fn list_batches(&self, limit: Option<u32>) -> Result<Vec<BatchJob>> {
        self.inner.list_batches(limit).await
    }

    fn supports_batch(&self) -> bool {
        self.inner.supports_batch()
    }
}

/// Main client for accessing LLM providers.
///
/// # Example
///
/// ```ignore
/// use llmkit::ParliqueClient;
///
/// let client = ParliqueClient::builder()
///     .with_anthropic_from_env()
///     .with_openai_from_env()
///     .build()?;
///
/// let response = client.complete(request).await?;
/// ```
pub struct ParliqueClient {
    providers: HashMap<String, Arc<dyn Provider>>,
    default_provider: Option<String>,
}

impl ParliqueClient {
    /// Create a new client builder.
    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    /// Get a provider by name.
    pub fn provider(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.providers.get(name).cloned()
    }

    /// Get the default provider.
    pub fn default_provider(&self) -> Option<Arc<dyn Provider>> {
        self.default_provider
            .as_ref()
            .and_then(|name| self.providers.get(name).cloned())
    }

    /// List all registered providers.
    pub fn providers(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }

    /// Make a completion request.
    ///
    /// The provider is determined from the model name prefix (e.g., "claude-" -> anthropic,
    /// "gpt-" -> openai) or falls back to the default provider.
    pub async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let provider = self.resolve_provider(&request.model)?;
        provider.complete(request).await
    }

    /// Make a streaming completion request.
    pub async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let provider = self.resolve_provider(&request.model)?;
        provider.complete_stream(request).await
    }

    /// Complete a request using a specific provider.
    pub async fn complete_with_provider(
        &self,
        provider_name: &str,
        request: CompletionRequest,
    ) -> Result<CompletionResponse> {
        let provider = self
            .providers
            .get(provider_name)
            .ok_or_else(|| Error::ProviderNotFound(provider_name.to_string()))?;
        provider.complete(request).await
    }

    /// Stream a completion using a specific provider.
    pub async fn complete_stream_with_provider(
        &self,
        provider_name: &str,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let provider = self
            .providers
            .get(provider_name)
            .ok_or_else(|| Error::ProviderNotFound(provider_name.to_string()))?;
        provider.complete_stream(request).await
    }

    /// Count tokens in a request.
    ///
    /// This allows estimation of token counts before making a completion request,
    /// useful for cost estimation and context window management.
    ///
    /// Note: Not all providers support token counting. Use `supports_token_counting`
    /// on the provider to check support.
    pub async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCountResult> {
        let provider = self.resolve_provider(&request.model)?;
        provider.count_tokens(request).await
    }

    /// Count tokens using a specific provider.
    pub async fn count_tokens_with_provider(
        &self,
        provider_name: &str,
        request: TokenCountRequest,
    ) -> Result<TokenCountResult> {
        let provider = self
            .providers
            .get(provider_name)
            .ok_or_else(|| Error::ProviderNotFound(provider_name.to_string()))?;
        provider.count_tokens(request).await
    }

    // ========== Batch Processing ==========

    /// Create a batch of requests for asynchronous processing.
    ///
    /// Batch processing can be significantly cheaper (up to 50% on some providers)
    /// and is ideal for non-time-sensitive workloads.
    ///
    /// The provider is determined from the first request's model name.
    pub async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchJob> {
        if requests.is_empty() {
            return Err(Error::invalid_request(
                "Batch must contain at least one request",
            ));
        }
        let provider = self.resolve_provider(&requests[0].request.model)?;
        provider.create_batch(requests).await
    }

    /// Create a batch using a specific provider.
    pub async fn create_batch_with_provider(
        &self,
        provider_name: &str,
        requests: Vec<BatchRequest>,
    ) -> Result<BatchJob> {
        let provider = self
            .providers
            .get(provider_name)
            .ok_or_else(|| Error::ProviderNotFound(provider_name.to_string()))?;
        provider.create_batch(requests).await
    }

    /// Get the status of a batch job.
    pub async fn get_batch(&self, provider_name: &str, batch_id: &str) -> Result<BatchJob> {
        let provider = self
            .providers
            .get(provider_name)
            .ok_or_else(|| Error::ProviderNotFound(provider_name.to_string()))?;
        provider.get_batch(batch_id).await
    }

    /// Get the results of a completed batch.
    pub async fn get_batch_results(
        &self,
        provider_name: &str,
        batch_id: &str,
    ) -> Result<Vec<BatchResult>> {
        let provider = self
            .providers
            .get(provider_name)
            .ok_or_else(|| Error::ProviderNotFound(provider_name.to_string()))?;
        provider.get_batch_results(batch_id).await
    }

    /// Cancel a batch job.
    pub async fn cancel_batch(&self, provider_name: &str, batch_id: &str) -> Result<BatchJob> {
        let provider = self
            .providers
            .get(provider_name)
            .ok_or_else(|| Error::ProviderNotFound(provider_name.to_string()))?;
        provider.cancel_batch(batch_id).await
    }

    /// List recent batch jobs for a provider.
    pub async fn list_batches(
        &self,
        provider_name: &str,
        limit: Option<u32>,
    ) -> Result<Vec<BatchJob>> {
        let provider = self
            .providers
            .get(provider_name)
            .ok_or_else(|| Error::ProviderNotFound(provider_name.to_string()))?;
        provider.list_batches(limit).await
    }

    /// Resolve which provider to use for a given model.
    fn resolve_provider(&self, model: &str) -> Result<Arc<dyn Provider>> {
        // Try to match provider by model name prefix
        let provider = self.infer_provider_from_model(model);

        if let Some(p) = provider {
            return Ok(p);
        }

        // Fall back to default provider
        self.default_provider().ok_or_else(|| {
            Error::ProviderNotFound(format!("No provider found for model: {}", model))
        })
    }

    /// Infer provider from model name.
    fn infer_provider_from_model(&self, model: &str) -> Option<Arc<dyn Provider>> {
        let model_lower = model.to_lowercase();

        // Anthropic models
        if model_lower.starts_with("claude") {
            return self.providers.get("anthropic").cloned();
        }

        // OpenAI models
        if model_lower.starts_with("gpt-")
            || model_lower.starts_with("o1")
            || model_lower.starts_with("o3")
            || model_lower.starts_with("o4")
            || model_lower.starts_with("chatgpt")
        {
            return self.providers.get("openai").cloned();
        }

        // Mistral models
        if model_lower.starts_with("mistral") || model_lower.starts_with("mixtral") {
            return self.providers.get("mistral").cloned();
        }

        // Google models
        if model_lower.starts_with("gemini") || model_lower.starts_with("palm") {
            return self
                .providers
                .get("google")
                .or_else(|| self.providers.get("vertex"))
                .cloned();
        }

        // Groq models (hosted versions)
        if model_lower.contains("groq") {
            return self.providers.get("groq").cloned();
        }

        // DeepSeek models
        if model_lower.starts_with("deepseek") {
            return self.providers.get("deepseek").cloned();
        }

        // Cohere models
        if model_lower.starts_with("command") {
            return self.providers.get("cohere").cloned();
        }

        // AI21 models
        if model_lower.starts_with("jamba") || model_lower.starts_with("j2-") {
            return self.providers.get("ai21").cloned();
        }

        // Together AI models (often with meta-llama/ prefix or together/)
        if model_lower.starts_with("meta-llama/")
            || model_lower.starts_with("together/")
            || model_lower.contains("together")
        {
            return self.providers.get("together").cloned();
        }

        // Fireworks AI models
        if model_lower.starts_with("accounts/fireworks") || model_lower.contains("fireworks/") {
            return self.providers.get("fireworks").cloned();
        }

        // Perplexity models (sonar-* models)
        if model_lower.contains("sonar") || model_lower.starts_with("pplx-") {
            return self.providers.get("perplexity").cloned();
        }

        // Cerebras models
        if model_lower.contains("cerebras") {
            return self.providers.get("cerebras").cloned();
        }

        // Local models via Ollama
        if model_lower.starts_with("llama")
            || model_lower.starts_with("codellama")
            || model_lower.starts_with("phi")
            || model_lower.starts_with("qwen")
        {
            return self.providers.get("ollama").cloned();
        }

        // HuggingFace models (typically owner/model format)
        if model_lower.contains("/") && !model_lower.contains("://") {
            // Could be HuggingFace format like "meta-llama/Llama-3.2-3B-Instruct"
            // Only match if no other provider claimed it
            return self.providers.get("huggingface").cloned();
        }

        // Replicate models (owner/model format with possible version)
        if model_lower.contains("replicate") {
            return self.providers.get("replicate").cloned();
        }

        None
    }
}

/// Builder for creating a `ParliqueClient`.
pub struct ClientBuilder {
    providers: HashMap<String, Arc<dyn Provider>>,
    default_provider: Option<String>,
    retry_config: Option<RetryConfig>,
}

impl ClientBuilder {
    /// Create a new client builder.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            default_provider: None,
            retry_config: None,
        }
    }

    /// Enable automatic retry with the specified configuration.
    ///
    /// All providers will be wrapped with retry logic using exponential backoff.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let client = ParliqueClient::builder()
    ///     .with_anthropic_from_env()
    ///     .with_retry(RetryConfig::production())
    ///     .build()?;
    /// ```
    pub fn with_retry(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(config);
        self
    }

    /// Enable automatic retry with default production configuration.
    ///
    /// Uses `RetryConfig::default()` which provides:
    /// - 10 retry attempts
    /// - Exponential backoff from 1s up to 5 minutes
    /// - Jitter enabled for better distributed retry timing
    pub fn with_default_retry(mut self) -> Self {
        self.retry_config = Some(RetryConfig::default());
        self
    }

    /// Add a custom provider.
    pub fn with_provider(mut self, name: impl Into<String>, provider: Arc<dyn Provider>) -> Self {
        let name = name.into();
        if self.default_provider.is_none() {
            self.default_provider = Some(name.clone());
        }
        self.providers.insert(name, provider);
        self
    }

    /// Set the default provider by name.
    pub fn with_default(mut self, name: impl Into<String>) -> Self {
        self.default_provider = Some(name.into());
        self
    }

    /// Add Anthropic provider from environment.
    #[cfg(feature = "anthropic")]
    pub fn with_anthropic_from_env(self) -> Self {
        match crate::providers::anthropic::AnthropicProvider::from_env() {
            Ok(provider) => self.with_provider("anthropic", Arc::new(provider)),
            Err(_) => self, // Skip if no API key
        }
    }

    /// Add Anthropic provider with API key.
    #[cfg(feature = "anthropic")]
    pub fn with_anthropic(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::anthropic::AnthropicProvider::with_api_key(api_key)?;
        Ok(self.with_provider("anthropic", Arc::new(provider)))
    }

    /// Add Anthropic provider with custom config.
    #[cfg(feature = "anthropic")]
    pub fn with_anthropic_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::anthropic::AnthropicProvider::new(config)?;
        Ok(self.with_provider("anthropic", Arc::new(provider)))
    }

    /// Add OpenAI provider from environment.
    #[cfg(feature = "openai")]
    pub fn with_openai_from_env(self) -> Self {
        match crate::providers::openai::OpenAIProvider::from_env() {
            Ok(provider) => self.with_provider("openai", Arc::new(provider)),
            Err(_) => self, // Skip if no API key
        }
    }

    /// Add OpenAI provider with API key.
    #[cfg(feature = "openai")]
    pub fn with_openai(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::openai::OpenAIProvider::with_api_key(api_key)?;
        Ok(self.with_provider("openai", Arc::new(provider)))
    }

    /// Add OpenAI provider with custom config.
    #[cfg(feature = "openai")]
    pub fn with_openai_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::openai::OpenAIProvider::new(config)?;
        Ok(self.with_provider("openai", Arc::new(provider)))
    }

    /// Add Groq provider from environment.
    #[cfg(feature = "groq")]
    pub fn with_groq_from_env(self) -> Self {
        match crate::providers::groq::GroqProvider::from_env() {
            Ok(provider) => self.with_provider("groq", Arc::new(provider)),
            Err(_) => self, // Skip if no API key
        }
    }

    /// Add Groq provider with API key.
    #[cfg(feature = "groq")]
    pub fn with_groq(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::groq::GroqProvider::with_api_key(api_key)?;
        Ok(self.with_provider("groq", Arc::new(provider)))
    }

    /// Add Groq provider with custom config.
    #[cfg(feature = "groq")]
    pub fn with_groq_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::groq::GroqProvider::new(config)?;
        Ok(self.with_provider("groq", Arc::new(provider)))
    }

    /// Add Mistral provider from environment.
    #[cfg(feature = "mistral")]
    pub fn with_mistral_from_env(self) -> Self {
        match crate::providers::mistral::MistralProvider::from_env() {
            Ok(provider) => self.with_provider("mistral", Arc::new(provider)),
            Err(_) => self, // Skip if no API key
        }
    }

    /// Add Mistral provider with API key.
    #[cfg(feature = "mistral")]
    pub fn with_mistral(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::mistral::MistralProvider::with_api_key(api_key)?;
        Ok(self.with_provider("mistral", Arc::new(provider)))
    }

    /// Add Mistral provider with custom config.
    #[cfg(feature = "mistral")]
    pub fn with_mistral_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::mistral::MistralProvider::new(config)?;
        Ok(self.with_provider("mistral", Arc::new(provider)))
    }

    /// Add Azure OpenAI provider from environment.
    ///
    /// Reads:
    /// - `AZURE_OPENAI_RESOURCE_NAME` or `AZURE_OPENAI_ENDPOINT`
    /// - `AZURE_OPENAI_DEPLOYMENT_ID` or `AZURE_OPENAI_DEPLOYMENT`
    /// - `AZURE_OPENAI_API_KEY`
    /// - `AZURE_OPENAI_API_VERSION` (optional)
    #[cfg(feature = "azure")]
    pub fn with_azure_from_env(self) -> Self {
        match crate::providers::azure::AzureOpenAIProvider::from_env() {
            Ok(provider) => self.with_provider("azure", Arc::new(provider)),
            Err(_) => self, // Skip if no configuration
        }
    }

    /// Add Azure OpenAI provider with configuration.
    #[cfg(feature = "azure")]
    pub fn with_azure(self, config: crate::providers::azure::AzureConfig) -> Result<Self> {
        let provider = crate::providers::azure::AzureOpenAIProvider::new(config)?;
        Ok(self.with_provider("azure", Arc::new(provider)))
    }

    /// Add AWS Bedrock provider from environment (async).
    ///
    /// Uses default AWS credential chain and reads region from:
    /// - `AWS_REGION` or `AWS_DEFAULT_REGION` environment variable
    /// - Falls back to "us-east-1" if not set
    ///
    /// Note: This is an async method that returns a future.
    #[cfg(feature = "bedrock")]
    pub async fn with_bedrock_from_env(self) -> Self {
        match crate::providers::bedrock::BedrockProvider::from_env_region().await {
            Ok(provider) => self.with_provider("bedrock", Arc::new(provider)),
            Err(_) => self, // Skip if no credentials
        }
    }

    /// Add AWS Bedrock provider with specified region (async).
    #[cfg(feature = "bedrock")]
    pub async fn with_bedrock_region(self, region: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::bedrock::BedrockProvider::from_env(region).await?;
        Ok(self.with_provider("bedrock", Arc::new(provider)))
    }

    /// Add AWS Bedrock provider with builder (async).
    #[cfg(feature = "bedrock")]
    pub async fn with_bedrock(
        self,
        builder: crate::providers::bedrock::BedrockBuilder,
    ) -> Result<Self> {
        let provider = builder.build().await?;
        Ok(self.with_provider("bedrock", Arc::new(provider)))
    }

    // ========== OpenAI-Compatible Providers ==========

    /// Add Together AI provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_together_from_env(self) -> Self {
        match crate::providers::openai_compatible::OpenAICompatibleProvider::together_from_env() {
            Ok(provider) => self.with_provider("together", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Together AI provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_together(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::together(api_key)?;
        Ok(self.with_provider("together", Arc::new(provider)))
    }

    /// Add Fireworks AI provider from environment.
    #[cfg(all(feature = "openai-compatible", not(feature = "fireworks")))]
    pub fn with_fireworks_from_env(self) -> Self {
        match crate::providers::openai_compatible::OpenAICompatibleProvider::fireworks_from_env() {
            Ok(provider) => self.with_provider("fireworks", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Fireworks AI provider with API key.
    #[cfg(all(feature = "openai-compatible", not(feature = "fireworks")))]
    pub fn with_fireworks(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::fireworks(api_key)?;
        Ok(self.with_provider("fireworks", Arc::new(provider)))
    }

    /// Add Fireworks AI provider from environment (dedicated provider).
    #[cfg(feature = "fireworks")]
    pub fn with_fireworks_from_env(self) -> Self {
        match crate::providers::fireworks::FireworksProvider::from_env() {
            Ok(provider) => self.with_provider("fireworks", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Fireworks AI provider with API key (dedicated provider).
    #[cfg(feature = "fireworks")]
    pub fn with_fireworks(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::fireworks::FireworksProvider::with_api_key(api_key)?;
        Ok(self.with_provider("fireworks", Arc::new(provider)))
    }

    /// Add DeepSeek provider from environment.
    #[cfg(all(feature = "openai-compatible", not(feature = "deepseek")))]
    pub fn with_deepseek_from_env(self) -> Self {
        match crate::providers::openai_compatible::OpenAICompatibleProvider::deepseek_from_env() {
            Ok(provider) => self.with_provider("deepseek", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add DeepSeek provider with API key.
    #[cfg(all(feature = "openai-compatible", not(feature = "deepseek")))]
    pub fn with_deepseek(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::deepseek(api_key)?;
        Ok(self.with_provider("deepseek", Arc::new(provider)))
    }

    /// Add DeepSeek provider from environment (dedicated provider).
    #[cfg(feature = "deepseek")]
    pub fn with_deepseek_from_env(self) -> Self {
        match crate::providers::deepseek::DeepSeekProvider::from_env() {
            Ok(provider) => self.with_provider("deepseek", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add DeepSeek provider with API key (dedicated provider).
    #[cfg(feature = "deepseek")]
    pub fn with_deepseek(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::deepseek::DeepSeekProvider::with_api_key(api_key)?;
        Ok(self.with_provider("deepseek", Arc::new(provider)))
    }

    /// Add Perplexity provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_perplexity_from_env(self) -> Self {
        match crate::providers::openai_compatible::OpenAICompatibleProvider::perplexity_from_env() {
            Ok(provider) => self.with_provider("perplexity", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Perplexity provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_perplexity(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::perplexity(api_key)?;
        Ok(self.with_provider("perplexity", Arc::new(provider)))
    }

    /// Add Anyscale provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_anyscale_from_env(self) -> Self {
        match crate::providers::openai_compatible::OpenAICompatibleProvider::anyscale_from_env() {
            Ok(provider) => self.with_provider("anyscale", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Anyscale provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_anyscale(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::anyscale(api_key)?;
        Ok(self.with_provider("anyscale", Arc::new(provider)))
    }

    /// Add DeepInfra provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_deepinfra_from_env(self) -> Self {
        match crate::providers::openai_compatible::OpenAICompatibleProvider::deepinfra_from_env() {
            Ok(provider) => self.with_provider("deepinfra", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add DeepInfra provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_deepinfra(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::deepinfra(api_key)?;
        Ok(self.with_provider("deepinfra", Arc::new(provider)))
    }

    /// Add Novita AI provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_novita_from_env(self) -> Self {
        match crate::providers::openai_compatible::OpenAICompatibleProvider::novita_from_env() {
            Ok(provider) => self.with_provider("novita", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Novita AI provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_novita(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::novita(api_key)?;
        Ok(self.with_provider("novita", Arc::new(provider)))
    }

    /// Add Hyperbolic provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_hyperbolic_from_env(self) -> Self {
        match crate::providers::openai_compatible::OpenAICompatibleProvider::hyperbolic_from_env() {
            Ok(provider) => self.with_provider("hyperbolic", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Hyperbolic provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_hyperbolic(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::hyperbolic(api_key)?;
        Ok(self.with_provider("hyperbolic", Arc::new(provider)))
    }

    /// Add Cerebras provider from environment (via OpenAI-compatible).
    #[cfg(all(feature = "openai-compatible", not(feature = "cerebras")))]
    pub fn with_cerebras_from_env(self) -> Self {
        match crate::providers::openai_compatible::OpenAICompatibleProvider::cerebras_from_env() {
            Ok(provider) => self.with_provider("cerebras", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Cerebras provider with API key (via OpenAI-compatible).
    #[cfg(all(feature = "openai-compatible", not(feature = "cerebras")))]
    pub fn with_cerebras(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::cerebras(api_key)?;
        Ok(self.with_provider("cerebras", Arc::new(provider)))
    }

    /// Add Cerebras provider from environment (dedicated provider).
    #[cfg(feature = "cerebras")]
    pub fn with_cerebras_from_env(self) -> Self {
        match crate::providers::cerebras::CerebrasProvider::from_env() {
            Ok(provider) => self.with_provider("cerebras", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Cerebras provider with API key (dedicated provider).
    #[cfg(feature = "cerebras")]
    pub fn with_cerebras(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::cerebras::CerebrasProvider::with_api_key(api_key)?;
        Ok(self.with_provider("cerebras", Arc::new(provider)))
    }

    // ========== Local OpenAI-Compatible Providers ==========

    /// Add LM Studio provider (local, default port 1234).
    #[cfg(feature = "openai-compatible")]
    pub fn with_lm_studio(self) -> Result<Self> {
        let provider = crate::providers::openai_compatible::OpenAICompatibleProvider::lm_studio()?;
        Ok(self.with_provider("lm_studio", Arc::new(provider)))
    }

    /// Add LM Studio provider with custom URL.
    #[cfg(feature = "openai-compatible")]
    pub fn with_lm_studio_url(self, base_url: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::lm_studio_url(base_url)?;
        Ok(self.with_provider("lm_studio", Arc::new(provider)))
    }

    /// Add vLLM provider (local, default port 8000).
    #[cfg(feature = "openai-compatible")]
    pub fn with_vllm(self) -> Result<Self> {
        let provider = crate::providers::openai_compatible::OpenAICompatibleProvider::vllm()?;
        Ok(self.with_provider("vllm", Arc::new(provider)))
    }

    /// Add vLLM provider with custom URL.
    #[cfg(feature = "openai-compatible")]
    pub fn with_vllm_url(self, base_url: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::vllm_url(base_url)?;
        Ok(self.with_provider("vllm", Arc::new(provider)))
    }

    /// Add TGI (Text Generation Inference) provider (local, default port 8080).
    #[cfg(feature = "openai-compatible")]
    pub fn with_tgi(self) -> Result<Self> {
        let provider = crate::providers::openai_compatible::OpenAICompatibleProvider::tgi()?;
        Ok(self.with_provider("tgi", Arc::new(provider)))
    }

    /// Add TGI provider with custom URL.
    #[cfg(feature = "openai-compatible")]
    pub fn with_tgi_url(self, base_url: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::tgi_url(base_url)?;
        Ok(self.with_provider("tgi", Arc::new(provider)))
    }

    /// Add Llamafile provider (local, default port 8080).
    #[cfg(feature = "openai-compatible")]
    pub fn with_llamafile(self) -> Result<Self> {
        let provider = crate::providers::openai_compatible::OpenAICompatibleProvider::llamafile()?;
        Ok(self.with_provider("llamafile", Arc::new(provider)))
    }

    /// Add Llamafile provider with custom URL.
    #[cfg(feature = "openai-compatible")]
    pub fn with_llamafile_url(self, base_url: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::openai_compatible::OpenAICompatibleProvider::llamafile_url(base_url)?;
        Ok(self.with_provider("llamafile", Arc::new(provider)))
    }

    /// Add a custom OpenAI-compatible provider.
    ///
    /// Use this for any provider that uses OpenAI's API format.
    ///
    /// # Arguments
    ///
    /// * `name` - Provider name for identification
    /// * `base_url` - Base URL (e.g., "https://api.example.com/v1")
    /// * `api_key` - Optional API key
    #[cfg(feature = "openai-compatible")]
    pub fn with_openai_compatible(
        self,
        name: impl Into<String>,
        base_url: impl Into<String>,
        api_key: Option<String>,
    ) -> Result<Self> {
        let name_str = name.into();
        let provider = crate::providers::openai_compatible::OpenAICompatibleProvider::custom(
            name_str.clone(),
            base_url,
            api_key,
        )?;
        Ok(self.with_provider(name_str, Arc::new(provider)))
    }

    // ========== Google Providers ==========

    /// Add Google AI (Gemini) provider from environment.
    ///
    /// Reads: `GOOGLE_API_KEY`
    #[cfg(feature = "google")]
    pub fn with_google_from_env(self) -> Self {
        match crate::providers::google::GoogleProvider::from_env() {
            Ok(provider) => self.with_provider("google", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Google AI (Gemini) provider with API key.
    #[cfg(feature = "google")]
    pub fn with_google(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::google::GoogleProvider::with_api_key(api_key)?;
        Ok(self.with_provider("google", Arc::new(provider)))
    }

    /// Add Google AI (Gemini) provider with custom config.
    #[cfg(feature = "google")]
    pub fn with_google_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::google::GoogleProvider::new(config)?;
        Ok(self.with_provider("google", Arc::new(provider)))
    }

    /// Add Google Vertex AI provider from environment.
    ///
    /// Reads:
    /// - `GOOGLE_CLOUD_PROJECT` or `VERTEX_PROJECT`
    /// - `GOOGLE_CLOUD_LOCATION` or `VERTEX_LOCATION`
    /// - `VERTEX_ACCESS_TOKEN`
    #[cfg(feature = "vertex")]
    pub fn with_vertex_from_env(self) -> Self {
        match crate::providers::vertex::VertexProvider::from_env() {
            Ok(provider) => self.with_provider("vertex", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Google Vertex AI provider with explicit configuration.
    #[cfg(feature = "vertex")]
    pub fn with_vertex(
        self,
        project_id: impl Into<String>,
        location: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Result<Self> {
        let provider =
            crate::providers::vertex::VertexProvider::new(project_id, location, access_token)?;
        Ok(self.with_provider("vertex", Arc::new(provider)))
    }

    /// Add Google Vertex AI provider with custom config.
    #[cfg(feature = "vertex")]
    pub fn with_vertex_config(
        self,
        config: crate::providers::vertex::VertexConfig,
    ) -> Result<Self> {
        let provider = crate::providers::vertex::VertexProvider::with_config(config)?;
        Ok(self.with_provider("vertex", Arc::new(provider)))
    }

    // ========== Enterprise Providers ==========

    /// Add Cohere provider from environment.
    ///
    /// Reads: `COHERE_API_KEY` or `CO_API_KEY`
    #[cfg(feature = "cohere")]
    pub fn with_cohere_from_env(self) -> Self {
        match crate::providers::cohere::CohereProvider::from_env() {
            Ok(provider) => self.with_provider("cohere", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Cohere provider with API key.
    #[cfg(feature = "cohere")]
    pub fn with_cohere(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::cohere::CohereProvider::with_api_key(api_key)?;
        Ok(self.with_provider("cohere", Arc::new(provider)))
    }

    /// Add Cohere provider with custom config.
    #[cfg(feature = "cohere")]
    pub fn with_cohere_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::cohere::CohereProvider::new(config)?;
        Ok(self.with_provider("cohere", Arc::new(provider)))
    }

    /// Add AI21 provider from environment.
    ///
    /// Reads: `AI21_API_KEY`
    #[cfg(feature = "ai21")]
    pub fn with_ai21_from_env(self) -> Self {
        match crate::providers::ai21::AI21Provider::from_env() {
            Ok(provider) => self.with_provider("ai21", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add AI21 provider with API key.
    #[cfg(feature = "ai21")]
    pub fn with_ai21(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::ai21::AI21Provider::with_api_key(api_key)?;
        Ok(self.with_provider("ai21", Arc::new(provider)))
    }

    /// Add AI21 provider with custom config.
    #[cfg(feature = "ai21")]
    pub fn with_ai21_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::ai21::AI21Provider::new(config)?;
        Ok(self.with_provider("ai21", Arc::new(provider)))
    }

    // ========== Inference Platforms ==========

    /// Add HuggingFace Inference API provider from environment.
    ///
    /// Reads: `HUGGINGFACE_API_KEY` or `HF_TOKEN`
    #[cfg(feature = "huggingface")]
    pub fn with_huggingface_from_env(self) -> Self {
        match crate::providers::huggingface::HuggingFaceProvider::from_env() {
            Ok(provider) => self.with_provider("huggingface", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add HuggingFace Inference API provider with API key.
    #[cfg(feature = "huggingface")]
    pub fn with_huggingface(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::huggingface::HuggingFaceProvider::with_api_key(api_key)?;
        Ok(self.with_provider("huggingface", Arc::new(provider)))
    }

    /// Add HuggingFace dedicated endpoint provider.
    #[cfg(feature = "huggingface")]
    pub fn with_huggingface_endpoint(
        self,
        endpoint_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self> {
        let provider =
            crate::providers::huggingface::HuggingFaceProvider::endpoint(endpoint_url, api_key)?;
        Ok(self.with_provider("huggingface", Arc::new(provider)))
    }

    /// Add HuggingFace provider with custom config.
    #[cfg(feature = "huggingface")]
    pub fn with_huggingface_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::huggingface::HuggingFaceProvider::new(config)?;
        Ok(self.with_provider("huggingface", Arc::new(provider)))
    }

    /// Add Replicate provider from environment.
    ///
    /// Reads: `REPLICATE_API_TOKEN`
    #[cfg(feature = "replicate")]
    pub fn with_replicate_from_env(self) -> Self {
        match crate::providers::replicate::ReplicateProvider::from_env() {
            Ok(provider) => self.with_provider("replicate", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Replicate provider with API token.
    #[cfg(feature = "replicate")]
    pub fn with_replicate(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::replicate::ReplicateProvider::with_api_key(api_key)?;
        Ok(self.with_provider("replicate", Arc::new(provider)))
    }

    /// Add Replicate provider with custom config.
    #[cfg(feature = "replicate")]
    pub fn with_replicate_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::replicate::ReplicateProvider::new(config)?;
        Ok(self.with_provider("replicate", Arc::new(provider)))
    }

    /// Add Baseten provider from environment.
    ///
    /// Reads: `BASETEN_API_KEY` and optionally `BASETEN_MODEL_ID`
    #[cfg(feature = "baseten")]
    pub fn with_baseten_from_env(self) -> Self {
        match crate::providers::baseten::BasetenProvider::from_env() {
            Ok(provider) => self.with_provider("baseten", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Baseten provider with API key.
    #[cfg(feature = "baseten")]
    pub fn with_baseten(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::baseten::BasetenProvider::with_api_key(api_key)?;
        Ok(self.with_provider("baseten", Arc::new(provider)))
    }

    /// Add Baseten provider with model ID and API key.
    #[cfg(feature = "baseten")]
    pub fn with_baseten_model(
        self,
        model_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self> {
        let provider = crate::providers::baseten::BasetenProvider::with_model(model_id, api_key)?;
        Ok(self.with_provider("baseten", Arc::new(provider)))
    }

    /// Add Baseten provider with custom config.
    #[cfg(feature = "baseten")]
    pub fn with_baseten_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::baseten::BasetenProvider::new(config)?;
        Ok(self.with_provider("baseten", Arc::new(provider)))
    }

    /// Add RunPod provider from environment.
    ///
    /// Reads: `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID`
    #[cfg(feature = "runpod")]
    pub fn with_runpod_from_env(self) -> Self {
        match crate::providers::runpod::RunPodProvider::from_env() {
            Ok(provider) => self.with_provider("runpod", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add RunPod provider with endpoint ID and API key.
    #[cfg(feature = "runpod")]
    pub fn with_runpod(
        self,
        endpoint_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self> {
        let provider = crate::providers::runpod::RunPodProvider::new(endpoint_id, api_key)?;
        Ok(self.with_provider("runpod", Arc::new(provider)))
    }

    // ============ Cloud Providers ============

    /// Add Cloudflare Workers AI provider from environment variables.
    ///
    /// Reads: `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`
    #[cfg(feature = "cloudflare")]
    pub fn with_cloudflare_from_env(self) -> Self {
        match crate::providers::cloudflare::CloudflareProvider::from_env() {
            Ok(provider) => self.with_provider("cloudflare", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Cloudflare Workers AI provider with account ID and API token.
    #[cfg(feature = "cloudflare")]
    pub fn with_cloudflare(
        self,
        account_id: impl Into<String>,
        api_token: impl Into<String>,
    ) -> Result<Self> {
        let provider =
            crate::providers::cloudflare::CloudflareProvider::new(account_id, api_token)?;
        Ok(self.with_provider("cloudflare", Arc::new(provider)))
    }

    /// Add IBM watsonx.ai provider from environment variables.
    ///
    /// Reads: `WATSONX_API_KEY` and `WATSONX_PROJECT_ID`
    #[cfg(feature = "watsonx")]
    pub fn with_watsonx_from_env(self) -> Self {
        match crate::providers::watsonx::WatsonxProvider::from_env() {
            Ok(provider) => self.with_provider("watsonx", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add IBM watsonx.ai provider with API key and project ID.
    #[cfg(feature = "watsonx")]
    pub fn with_watsonx(
        self,
        api_key: impl Into<String>,
        project_id: impl Into<String>,
    ) -> Result<Self> {
        let provider = crate::providers::watsonx::WatsonxProvider::new(api_key, project_id)?;
        Ok(self.with_provider("watsonx", Arc::new(provider)))
    }

    /// Add Databricks provider from environment variables.
    ///
    /// Reads: `DATABRICKS_TOKEN` and `DATABRICKS_HOST`
    #[cfg(feature = "databricks")]
    pub fn with_databricks_from_env(self) -> Self {
        match crate::providers::databricks::DatabricksProvider::from_env() {
            Ok(provider) => self.with_provider("databricks", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Databricks provider with host URL and token.
    #[cfg(feature = "databricks")]
    pub fn with_databricks(
        self,
        host: impl Into<String>,
        token: impl Into<String>,
    ) -> Result<Self> {
        let provider = crate::providers::databricks::DatabricksProvider::new(host, token)?;
        Ok(self.with_provider("databricks", Arc::new(provider)))
    }

    // ============ Specialized/Fast Inference Providers ============

    /// Add SambaNova provider from environment variables.
    ///
    /// Reads: `SAMBANOVA_API_KEY`
    #[cfg(feature = "sambanova")]
    pub fn with_sambanova_from_env(self) -> Self {
        match crate::providers::sambanova::SambaNovaProvider::from_env() {
            Ok(provider) => self.with_provider("sambanova", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add SambaNova provider with API key.
    #[cfg(feature = "sambanova")]
    pub fn with_sambanova(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::sambanova::SambaNovaProvider::with_api_key(api_key)?;
        Ok(self.with_provider("sambanova", Arc::new(provider)))
    }

    /// Build the client.
    ///
    /// If retry configuration was set via `with_retry()` or `with_default_retry()`,
    /// all providers will be wrapped with automatic retry logic.
    pub fn build(self) -> Result<ParliqueClient> {
        if self.providers.is_empty() {
            return Err(Error::config("No providers configured"));
        }

        // Wrap providers with retry logic if configured
        let providers = if let Some(retry_config) = self.retry_config {
            self.providers
                .into_iter()
                .map(|(name, provider)| {
                    let retrying = DynamicRetryingProvider {
                        inner: provider,
                        config: retry_config.clone(),
                    };
                    (name, Arc::new(retrying) as Arc<dyn Provider>)
                })
                .collect()
        } else {
            self.providers
        };

        Ok(ParliqueClient {
            providers,
            default_provider: self.default_provider,
        })
    }
}

impl Default for ClientBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_inference() {
        // This test just verifies the inference logic without actual providers
        let client = ParliqueClient {
            providers: HashMap::new(),
            default_provider: None,
        };

        // Just testing internal logic patterns
        assert!(client.infer_provider_from_model("claude-3").is_none());
        assert!(client.infer_provider_from_model("gpt-4o").is_none());
    }
}
