//! LLMKit client for unified LLM access.
//!
//! The `LLMKitClient` provides a unified interface to interact with multiple LLM providers.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use tokio::time::sleep;

use crate::embedding::{EmbeddingProvider, EmbeddingRequest, EmbeddingResponse};
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

/// Parse a model identifier in the required "provider/model" format.
///
/// The format "provider/model" (e.g., "anthropic/claude-sonnet-4-20250514") is required.
/// Returns Ok((provider, model_name)) if valid, or Err if the format is invalid.
///
/// # Examples
///
/// ```ignore
/// let (provider, model) = parse_model_identifier("anthropic/claude-sonnet-4-20250514")?;
/// assert_eq!(provider, "anthropic");
/// assert_eq!(model, "claude-sonnet-4-20250514");
///
/// // This will return an error - provider is required
/// let result = parse_model_identifier("gpt-4o");
/// assert!(result.is_err());
/// ```
fn parse_model_identifier(model: &str) -> Result<(&str, &str)> {
    if let Some(idx) = model.find('/') {
        let provider = &model[..idx];
        let model_name = &model[idx + 1..];
        // Validate provider name (shouldn't contain special chars)
        if !provider.is_empty()
            && !provider.contains('-')
            && !provider.contains('.')
            && !provider.contains(':')
        {
            return Ok((provider, model_name));
        }
    }
    Err(Error::InvalidRequest(format!(
        "Model must be in 'provider/model' format (e.g., 'openai/gpt-4o'), got: {}",
        model
    )))
}

/// Main client for accessing LLM providers.
///
/// # Model Format
///
/// Models must be specified using the `"provider/model"` format (e.g., `"openai/gpt-4o"`).
/// This ensures the client routes the request to the correct provider.
///
/// # Example
///
/// ```ignore
/// use llmkit::LLMKitClient;
///
/// let client = LLMKitClient::builder()
///     .with_anthropic_from_env()
///     .with_openai_from_env()
///     .with_baidu(api_key, secret_key)?
///     .with_alibaba(api_key)?
///     .build()?;
///
/// // Use explicit provider/model format for any provider
/// let request = CompletionRequest::new("anthropic/claude-sonnet-4-20250514", messages);
/// let response = client.complete(request).await?;
///
/// // Regional providers (Phase 2.3)
/// let request = CompletionRequest::new("baidu/ERNIE-Bot-Ultra", messages);
/// let response = client.complete(request).await?;
///
/// let request = CompletionRequest::new("alibaba/qwen-max", messages);
/// let response = client.complete(request).await?;
/// ```
pub struct LLMKitClient {
    providers: HashMap<String, Arc<dyn Provider>>,
    embedding_providers: HashMap<String, Arc<dyn EmbeddingProvider>>,
    default_provider: Option<String>,
}

impl LLMKitClient {
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
    /// The provider is determined from:
    /// 1. Explicit provider in model string (e.g., "anthropic/claude-sonnet-4-20250514")
    /// 2. Model name prefix inference (e.g., "claude-" -> anthropic, "gpt-" -> openai)
    /// 3. Default provider as fallback
    pub async fn complete(&self, mut request: CompletionRequest) -> Result<CompletionResponse> {
        let (provider, model_name) = self.resolve_provider(&request.model)?;
        request.model = model_name;
        provider.complete(request).await
    }

    /// Make a streaming completion request.
    ///
    /// The provider is determined using the same logic as `complete()`.
    pub async fn complete_stream(
        &self,
        mut request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let (provider, model_name) = self.resolve_provider(&request.model)?;
        request.model = model_name;
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
    ///
    /// The provider is determined using the same logic as `complete()`.
    pub async fn count_tokens(&self, mut request: TokenCountRequest) -> Result<TokenCountResult> {
        let (provider, model_name) = self.resolve_provider(&request.model)?;
        request.model = model_name;
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
    /// The provider is determined from the first request's model name using the
    /// same logic as `complete()`.
    pub async fn create_batch(&self, mut requests: Vec<BatchRequest>) -> Result<BatchJob> {
        if requests.is_empty() {
            return Err(Error::invalid_request(
                "Batch must contain at least one request",
            ));
        }
        let (provider, model_name) = self.resolve_provider(&requests[0].request.model)?;
        // Update the model name in all requests to strip provider prefix
        for req in &mut requests {
            let (_, req_model) = parse_model_identifier(&req.request.model)?;
            req.request.model = req_model.to_string();
        }
        // Also update the first request's model
        requests[0].request.model = model_name;
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

    // ========== Embeddings ==========

    /// Generate embeddings for text.
    ///
    /// The provider is determined from:
    /// 1. Explicit provider in model string (e.g., "openai/text-embedding-3-small")
    /// 2. Model name prefix inference (e.g., "text-embedding-" -> openai, "voyage-" -> voyage)
    /// 3. First available embedding provider as fallback
    ///
    /// # Example
    ///
    /// ```ignore
    /// use llmkit::{LLMKitClient, EmbeddingRequest};
    ///
    /// let client = LLMKitClient::builder()
    ///     .with_openai_from_env()
    ///     .build()?;
    ///
    /// // Explicit provider
    /// let request = EmbeddingRequest::new("openai/text-embedding-3-small", "Hello, world!");
    /// let response = client.embed(request).await?;
    ///
    /// // Or with inference (backward compatible)
    /// let request = EmbeddingRequest::new("text-embedding-3-small", "Hello, world!");
    /// let response = client.embed(request).await?;
    /// println!("Embedding dimensions: {}", response.dimensions());
    /// ```
    pub async fn embed(&self, mut request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let (provider, model_name) = self.resolve_embedding_provider(&request.model)?;
        request.model = model_name;
        provider.embed(request).await
    }

    /// Generate embeddings using a specific provider.
    pub async fn embed_with_provider(
        &self,
        provider_name: &str,
        request: EmbeddingRequest,
    ) -> Result<EmbeddingResponse> {
        let provider = self
            .embedding_providers
            .get(provider_name)
            .ok_or_else(|| Error::ProviderNotFound(provider_name.to_string()))?;
        provider.embed(request).await
    }

    /// List all registered embedding providers.
    pub fn embedding_providers(&self) -> Vec<&str> {
        self.embedding_providers
            .keys()
            .map(|s| s.as_str())
            .collect()
    }

    /// Check if a provider supports embeddings.
    pub fn supports_embeddings(&self, provider_name: &str) -> bool {
        self.embedding_providers.contains_key(provider_name)
    }

    /// Resolve the embedding provider for a model in "provider/model" format.
    ///
    /// The model must be in "provider/model" format (e.g., "openai/text-embedding-3-small").
    /// Returns the provider and the model name (without provider prefix).
    fn resolve_embedding_provider(
        &self,
        model: &str,
    ) -> Result<(Arc<dyn EmbeddingProvider>, String)> {
        let (provider_name, model_name) = parse_model_identifier(model)?;

        self.embedding_providers
            .get(provider_name)
            .cloned()
            .map(|p| (p, model_name.to_string()))
            .ok_or_else(|| {
                Error::ProviderNotFound(format!(
                    "Embedding provider '{}' not configured. Available providers: {:?}",
                    provider_name,
                    self.embedding_providers.keys().collect::<Vec<_>>()
                ))
            })
    }

    /// Resolve the provider for a model in "provider/model" format.
    ///
    /// The model must be in "provider/model" format (e.g., "anthropic/claude-sonnet-4-20250514").
    /// Returns the provider and the model name (without provider prefix).
    fn resolve_provider(&self, model: &str) -> Result<(Arc<dyn Provider>, String)> {
        let (provider_name, model_name) = parse_model_identifier(model)?;

        self.providers
            .get(provider_name)
            .cloned()
            .map(|p| (p, model_name.to_string()))
            .ok_or_else(|| {
                Error::ProviderNotFound(format!(
                    "Provider '{}' not configured. Available providers: {:?}",
                    provider_name,
                    self.providers.keys().collect::<Vec<_>>()
                ))
            })
    }
}

/// Builder for creating a `LLMKitClient`.
pub struct ClientBuilder {
    providers: HashMap<String, Arc<dyn Provider>>,
    embedding_providers: HashMap<String, Arc<dyn EmbeddingProvider>>,
    default_provider: Option<String>,
    retry_config: Option<RetryConfig>,
}

impl ClientBuilder {
    /// Create a new client builder.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            embedding_providers: HashMap::new(),
            default_provider: None,
            retry_config: None,
        }
    }

    /// Add an embedding provider.
    pub fn with_embedding_provider(
        mut self,
        name: impl Into<String>,
        provider: Arc<dyn EmbeddingProvider>,
    ) -> Self {
        self.embedding_providers.insert(name.into(), provider);
        self
    }

    /// Enable automatic retry with the specified configuration.
    ///
    /// All providers will be wrapped with retry logic using exponential backoff.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let client = LLMKitClient::builder()
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
        match crate::providers::chat::anthropic::AnthropicProvider::from_env() {
            Ok(provider) => self.with_provider("anthropic", Arc::new(provider)),
            Err(_) => self, // Skip if no API key
        }
    }

    /// Add Anthropic provider with API key.
    #[cfg(feature = "anthropic")]
    pub fn with_anthropic(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::chat::anthropic::AnthropicProvider::with_api_key(api_key)?;
        Ok(self.with_provider("anthropic", Arc::new(provider)))
    }

    /// Add Anthropic provider with custom config.
    #[cfg(feature = "anthropic")]
    pub fn with_anthropic_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::chat::anthropic::AnthropicProvider::new(config)?;
        Ok(self.with_provider("anthropic", Arc::new(provider)))
    }

    /// Add OpenAI provider from environment.
    ///
    /// Also registers OpenAI as an embedding provider for text-embedding-* models.
    #[cfg(feature = "openai")]
    pub fn with_openai_from_env(mut self) -> Self {
        match crate::providers::chat::openai::OpenAIProvider::from_env() {
            Ok(provider) => {
                let provider = Arc::new(provider);
                self.embedding_providers.insert(
                    "openai".to_string(),
                    Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
                );
                self.with_provider("openai", provider)
            }
            Err(_) => self, // Skip if no API key
        }
    }

    /// Add OpenAI provider with API key.
    ///
    /// Also registers OpenAI as an embedding provider for text-embedding-* models.
    #[cfg(feature = "openai")]
    pub fn with_openai(mut self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            Arc::new(crate::providers::chat::openai::OpenAIProvider::with_api_key(api_key)?);
        self.embedding_providers.insert(
            "openai".to_string(),
            Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
        );
        Ok(self.with_provider("openai", provider))
    }

    /// Add OpenAI provider with custom config.
    ///
    /// Also registers OpenAI as an embedding provider for text-embedding-* models.
    #[cfg(feature = "openai")]
    pub fn with_openai_config(mut self, config: ProviderConfig) -> Result<Self> {
        let provider = Arc::new(crate::providers::chat::openai::OpenAIProvider::new(config)?);
        self.embedding_providers.insert(
            "openai".to_string(),
            Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
        );
        Ok(self.with_provider("openai", provider))
    }

    /// Add Groq provider from environment.
    #[cfg(feature = "groq")]
    pub fn with_groq_from_env(self) -> Self {
        match crate::providers::chat::groq::GroqProvider::from_env() {
            Ok(provider) => self.with_provider("groq", Arc::new(provider)),
            Err(_) => self, // Skip if no API key
        }
    }

    /// Add Groq provider with API key.
    #[cfg(feature = "groq")]
    pub fn with_groq(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::chat::groq::GroqProvider::with_api_key(api_key)?;
        Ok(self.with_provider("groq", Arc::new(provider)))
    }

    /// Add Groq provider with custom config.
    #[cfg(feature = "groq")]
    pub fn with_groq_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::chat::groq::GroqProvider::new(config)?;
        Ok(self.with_provider("groq", Arc::new(provider)))
    }

    /// Add Mistral provider from environment.
    #[cfg(feature = "mistral")]
    pub fn with_mistral_from_env(self) -> Self {
        match crate::providers::chat::mistral::MistralProvider::from_env() {
            Ok(provider) => self.with_provider("mistral", Arc::new(provider)),
            Err(_) => self, // Skip if no API key
        }
    }

    /// Add Mistral provider with API key.
    #[cfg(feature = "mistral")]
    pub fn with_mistral(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::chat::mistral::MistralProvider::with_api_key(api_key)?;
        Ok(self.with_provider("mistral", Arc::new(provider)))
    }

    /// Add Mistral provider from environment.
    #[cfg(feature = "mistral")]
    pub fn with_mistral_config(self, _config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::chat::mistral::MistralProvider::from_env()?;
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
        match crate::providers::chat::azure::AzureOpenAIProvider::from_env() {
            Ok(provider) => self.with_provider("azure", Arc::new(provider)),
            Err(_) => self, // Skip if no configuration
        }
    }

    /// Add Azure OpenAI provider with configuration.
    #[cfg(feature = "azure")]
    pub fn with_azure(self, config: crate::providers::chat::azure::AzureConfig) -> Result<Self> {
        let provider = crate::providers::chat::azure::AzureOpenAIProvider::new(config)?;
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
        match crate::providers::chat::bedrock::BedrockProvider::from_env_region().await {
            Ok(provider) => self.with_provider("bedrock", Arc::new(provider)),
            Err(_) => self, // Skip if no credentials
        }
    }

    /// Add AWS Bedrock provider with specified region (async).
    #[cfg(feature = "bedrock")]
    pub async fn with_bedrock_region(self, region: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::chat::bedrock::BedrockProvider::from_env(region).await?;
        Ok(self.with_provider("bedrock", Arc::new(provider)))
    }

    /// Add AWS Bedrock provider with builder (async).
    #[cfg(feature = "bedrock")]
    pub async fn with_bedrock(
        self,
        builder: crate::providers::chat::bedrock::BedrockBuilder,
    ) -> Result<Self> {
        let provider = builder.build().await?;
        Ok(self.with_provider("bedrock", Arc::new(provider)))
    }

    // ========== OpenAI-Compatible Providers ==========

    /// Add Together AI provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_together_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::together_from_env(
        ) {
            Ok(provider) => self.with_provider("together", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Together AI provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_together(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::together(api_key)?;
        Ok(self.with_provider("together", Arc::new(provider)))
    }

    /// Add Fireworks AI provider from environment.
    #[cfg(all(feature = "openai-compatible", not(feature = "fireworks")))]
    pub fn with_fireworks_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::fireworks_from_env() {
            Ok(provider) => self.with_provider("fireworks", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Fireworks AI provider with API key.
    #[cfg(all(feature = "openai-compatible", not(feature = "fireworks")))]
    pub fn with_fireworks(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::fireworks(
                api_key,
            )?;
        Ok(self.with_provider("fireworks", Arc::new(provider)))
    }

    /// Add Fireworks AI provider from environment (dedicated provider).
    #[cfg(feature = "fireworks")]
    pub fn with_fireworks_from_env(self) -> Self {
        match crate::providers::chat::fireworks::FireworksProvider::from_env() {
            Ok(provider) => self.with_provider("fireworks", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Fireworks AI provider with API key (dedicated provider).
    #[cfg(feature = "fireworks")]
    pub fn with_fireworks(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::chat::fireworks::FireworksProvider::with_api_key(api_key)?;
        Ok(self.with_provider("fireworks", Arc::new(provider)))
    }

    /// Add DeepSeek provider from environment.
    #[cfg(all(feature = "openai-compatible", not(feature = "deepseek")))]
    pub fn with_deepseek_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::deepseek_from_env(
        ) {
            Ok(provider) => self.with_provider("deepseek", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add DeepSeek provider with API key.
    #[cfg(all(feature = "openai-compatible", not(feature = "deepseek")))]
    pub fn with_deepseek(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::deepseek(api_key)?;
        Ok(self.with_provider("deepseek", Arc::new(provider)))
    }

    /// Add DeepSeek provider from environment (dedicated provider).
    #[cfg(feature = "deepseek")]
    pub fn with_deepseek_from_env(self) -> Self {
        match crate::providers::chat::deepseek::DeepSeekProvider::from_env() {
            Ok(provider) => self.with_provider("deepseek", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add DeepSeek provider with API key (dedicated provider).
    #[cfg(feature = "deepseek")]
    pub fn with_deepseek(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::chat::deepseek::DeepSeekProvider::with_api_key(api_key)?;
        Ok(self.with_provider("deepseek", Arc::new(provider)))
    }

    /// Add Perplexity provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_perplexity_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::perplexity_from_env() {
            Ok(provider) => self.with_provider("perplexity", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Perplexity provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_perplexity(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::perplexity(
                api_key,
            )?;
        Ok(self.with_provider("perplexity", Arc::new(provider)))
    }

    /// Add Anyscale provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_anyscale_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::anyscale_from_env(
        ) {
            Ok(provider) => self.with_provider("anyscale", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Anyscale provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_anyscale(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::anyscale(api_key)?;
        Ok(self.with_provider("anyscale", Arc::new(provider)))
    }

    /// Add DeepInfra provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_deepinfra_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::deepinfra_from_env() {
            Ok(provider) => self.with_provider("deepinfra", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add DeepInfra provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_deepinfra(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::deepinfra(
                api_key,
            )?;
        Ok(self.with_provider("deepinfra", Arc::new(provider)))
    }

    /// Add Novita AI provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_novita_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::novita_from_env()
        {
            Ok(provider) => self.with_provider("novita", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Novita AI provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_novita(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::novita(api_key)?;
        Ok(self.with_provider("novita", Arc::new(provider)))
    }

    /// Add Hyperbolic provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_hyperbolic_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::hyperbolic_from_env() {
            Ok(provider) => self.with_provider("hyperbolic", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Hyperbolic provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_hyperbolic(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::hyperbolic(
                api_key,
            )?;
        Ok(self.with_provider("hyperbolic", Arc::new(provider)))
    }

    /// Add Cerebras provider from environment (via OpenAI-compatible).
    #[cfg(all(feature = "openai-compatible", not(feature = "cerebras")))]
    pub fn with_cerebras_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::cerebras_from_env(
        ) {
            Ok(provider) => self.with_provider("cerebras", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Cerebras provider with API key (via OpenAI-compatible).
    #[cfg(all(feature = "openai-compatible", not(feature = "cerebras")))]
    pub fn with_cerebras(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::cerebras(api_key)?;
        Ok(self.with_provider("cerebras", Arc::new(provider)))
    }

    /// Add Cerebras provider from environment (dedicated provider).
    #[cfg(feature = "cerebras")]
    pub fn with_cerebras_from_env(self) -> Self {
        match crate::providers::chat::cerebras::CerebrasProvider::from_env() {
            Ok(provider) => self.with_provider("cerebras", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Cerebras provider with API key (dedicated provider).
    #[cfg(feature = "cerebras")]
    pub fn with_cerebras(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::chat::cerebras::CerebrasProvider::with_api_key(api_key)?;
        Ok(self.with_provider("cerebras", Arc::new(provider)))
    }

    // ========== Phase 2: Additional Tier 1 Providers ==========

    /// Add Reka AI provider from environment.
    #[cfg(feature = "reka")]
    pub fn with_reka_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::reka_from_env() {
            Ok(provider) => self.with_provider("reka", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Reka AI provider with API key.
    #[cfg(feature = "reka")]
    pub fn with_reka(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::reka(api_key)?;
        Ok(self.with_provider("reka", Arc::new(provider)))
    }

    /// Add Reka AI provider with custom config.
    #[cfg(feature = "reka")]
    pub fn with_reka_config(self, config: crate::provider::ProviderConfig) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::reka_config(
                config,
            )?;
        Ok(self.with_provider("reka", Arc::new(provider)))
    }

    /// Add Nvidia NIM provider from environment.
    #[cfg(feature = "nvidia-nim")]
    pub fn with_nvidia_nim_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::nvidia_nim_from_env() {
            Ok(provider) => self.with_provider("nvidia_nim", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Nvidia NIM provider with API key.
    #[cfg(feature = "nvidia-nim")]
    pub fn with_nvidia_nim(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::nvidia_nim(
                api_key,
            )?;
        Ok(self.with_provider("nvidia_nim", Arc::new(provider)))
    }

    /// Add Nvidia NIM provider with custom config.
    #[cfg(feature = "nvidia-nim")]
    pub fn with_nvidia_nim_config(self, config: crate::provider::ProviderConfig) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::nvidia_nim_config(
                config,
            )?;
        Ok(self.with_provider("nvidia_nim", Arc::new(provider)))
    }

    /// Add Xinference provider from environment.
    #[cfg(feature = "xinference")]
    pub fn with_xinference_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::xinference_from_env() {
            Ok(provider) => self.with_provider("xinference", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Xinference provider with API key.
    #[cfg(feature = "xinference")]
    pub fn with_xinference(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::xinference(
                api_key,
            )?;
        Ok(self.with_provider("xinference", Arc::new(provider)))
    }

    /// Add Xinference provider with custom config.
    #[cfg(feature = "xinference")]
    pub fn with_xinference_config(self, config: crate::provider::ProviderConfig) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::xinference_config(
                config,
            )?;
        Ok(self.with_provider("xinference", Arc::new(provider)))
    }

    /// Add PublicAI provider from environment.
    #[cfg(feature = "public-ai")]
    pub fn with_public_ai_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::public_ai_from_env() {
            Ok(provider) => self.with_provider("public_ai", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add PublicAI provider with API key.
    #[cfg(feature = "public-ai")]
    pub fn with_public_ai(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::public_ai(
                api_key,
            )?;
        Ok(self.with_provider("public_ai", Arc::new(provider)))
    }

    /// Add PublicAI provider with custom config.
    #[cfg(feature = "public-ai")]
    pub fn with_public_ai_config(self, config: crate::provider::ProviderConfig) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::public_ai_config(
                config,
            )?;
        Ok(self.with_provider("public_ai", Arc::new(provider)))
    }

    // ========== Phase 2.3 Providers ==========

    /// Add Bytez provider from environment.
    #[cfg(feature = "bytez")]
    pub fn with_bytez_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::bytez_from_env()
        {
            Ok(provider) => self.with_provider("bytez", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Bytez provider with API key.
    #[cfg(feature = "bytez")]
    pub fn with_bytez(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::bytez(api_key)?;
        Ok(self.with_provider("bytez", Arc::new(provider)))
    }

    /// Add Bytez provider with custom config.
    #[cfg(feature = "bytez")]
    pub fn with_bytez_config(self, config: crate::provider::ProviderConfig) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::bytez_config(
                config,
            )?;
        Ok(self.with_provider("bytez", Arc::new(provider)))
    }

    /// Add Chutes provider from environment.
    #[cfg(feature = "chutes")]
    pub fn with_chutes_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::chutes_from_env()
        {
            Ok(provider) => self.with_provider("chutes", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Chutes provider with API key.
    #[cfg(feature = "chutes")]
    pub fn with_chutes(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::chutes(api_key)?;
        Ok(self.with_provider("chutes", Arc::new(provider)))
    }

    /// Add Chutes provider with custom config.
    #[cfg(feature = "chutes")]
    pub fn with_chutes_config(self, config: crate::provider::ProviderConfig) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::chutes_config(
                config,
            )?;
        Ok(self.with_provider("chutes", Arc::new(provider)))
    }

    /// Add CometAPI provider from environment.
    #[cfg(feature = "comet-api")]
    pub fn with_comet_api_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::comet_api_from_env() {
            Ok(provider) => self.with_provider("comet_api", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add CometAPI provider with API key.
    #[cfg(feature = "comet-api")]
    pub fn with_comet_api(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::comet_api(
                api_key,
            )?;
        Ok(self.with_provider("comet_api", Arc::new(provider)))
    }

    /// Add CometAPI provider with custom config.
    #[cfg(feature = "comet-api")]
    pub fn with_comet_api_config(self, config: crate::provider::ProviderConfig) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::comet_api_config(
                config,
            )?;
        Ok(self.with_provider("comet_api", Arc::new(provider)))
    }

    /// Add CompactifAI provider from environment.
    #[cfg(feature = "compactifai")]
    pub fn with_compactifai_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::compactifai_from_env()
        {
            Ok(provider) => self.with_provider("compactifai", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add CompactifAI provider with API key.
    #[cfg(feature = "compactifai")]
    pub fn with_compactifai(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::compactifai(
                api_key,
            )?;
        Ok(self.with_provider("compactifai", Arc::new(provider)))
    }

    /// Add CompactifAI provider with custom config.
    #[cfg(feature = "compactifai")]
    pub fn with_compactifai_config(self, config: crate::provider::ProviderConfig) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::compactifai_config(
                config,
            )?;
        Ok(self.with_provider("compactifai", Arc::new(provider)))
    }

    /// Add Synthetic provider from environment.
    #[cfg(feature = "synthetic")]
    pub fn with_synthetic_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::synthetic_from_env() {
            Ok(provider) => self.with_provider("synthetic", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Synthetic provider with API key.
    #[cfg(feature = "synthetic")]
    pub fn with_synthetic(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::synthetic(
                api_key,
            )?;
        Ok(self.with_provider("synthetic", Arc::new(provider)))
    }

    /// Add Synthetic provider with custom config.
    #[cfg(feature = "synthetic")]
    pub fn with_synthetic_config(self, config: crate::provider::ProviderConfig) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::synthetic_config(
                config,
            )?;
        Ok(self.with_provider("synthetic", Arc::new(provider)))
    }

    /// Add Morph provider from environment.
    #[cfg(feature = "morph")]
    pub fn with_morph_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::morph_from_env()
        {
            Ok(provider) => self.with_provider("morph", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Morph provider with API key.
    #[cfg(feature = "morph")]
    pub fn with_morph(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::morph(api_key)?;
        Ok(self.with_provider("morph", Arc::new(provider)))
    }

    /// Add Morph provider with custom config.
    #[cfg(feature = "morph")]
    pub fn with_morph_config(self, config: crate::provider::ProviderConfig) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::morph_config(
                config,
            )?;
        Ok(self.with_provider("morph", Arc::new(provider)))
    }

    /// Add Heroku AI provider from environment.
    #[cfg(feature = "heroku-ai")]
    pub fn with_heroku_ai_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::heroku_ai_from_env() {
            Ok(provider) => self.with_provider("heroku_ai", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Heroku AI provider with API key.
    #[cfg(feature = "heroku-ai")]
    pub fn with_heroku_ai(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::heroku_ai(
                api_key,
            )?;
        Ok(self.with_provider("heroku_ai", Arc::new(provider)))
    }

    /// Add Heroku AI provider with custom config.
    #[cfg(feature = "heroku-ai")]
    pub fn with_heroku_ai_config(self, config: crate::provider::ProviderConfig) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::heroku_ai_config(
                config,
            )?;
        Ok(self.with_provider("heroku_ai", Arc::new(provider)))
    }

    /// Add v0 (Vercel) provider from environment.
    #[cfg(feature = "v0")]
    pub fn with_v0_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::v0_from_env() {
            Ok(provider) => self.with_provider("v0", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add v0 (Vercel) provider with API key.
    #[cfg(feature = "v0")]
    pub fn with_v0(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::v0(api_key)?;
        Ok(self.with_provider("v0", Arc::new(provider)))
    }

    /// Add v0 (Vercel) provider with custom config.
    #[cfg(feature = "v0")]
    pub fn with_v0_config(self, config: crate::provider::ProviderConfig) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::v0_config(config)?;
        Ok(self.with_provider("v0", Arc::new(provider)))
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
        let provider = crate::providers::chat::openai_compatible::OpenAICompatibleProvider::custom(
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
        match crate::providers::chat::google::GoogleProvider::from_env() {
            Ok(provider) => self.with_provider("google", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Google AI (Gemini) provider with API key.
    #[cfg(feature = "google")]
    pub fn with_google(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::chat::google::GoogleProvider::with_api_key(api_key)?;
        Ok(self.with_provider("google", Arc::new(provider)))
    }

    /// Add Google AI (Gemini) provider with custom config.
    #[cfg(feature = "google")]
    pub fn with_google_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::chat::google::GoogleProvider::new(config)?;
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
        match crate::providers::chat::vertex::VertexProvider::from_env() {
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
        let provider = crate::providers::chat::vertex::VertexProvider::new(
            project_id,
            location,
            access_token,
        )?;
        Ok(self.with_provider("vertex", Arc::new(provider)))
    }

    /// Add Google Vertex AI provider with custom config.
    #[cfg(feature = "vertex")]
    pub fn with_vertex_config(
        self,
        config: crate::providers::chat::vertex::VertexConfig,
    ) -> Result<Self> {
        let provider = crate::providers::chat::vertex::VertexProvider::with_config(config)?;
        Ok(self.with_provider("vertex", Arc::new(provider)))
    }

    // Vertex AI Partner Models (Phase 2)

    /// Add Vertex AI with Anthropic Claude models from environment.
    #[cfg(feature = "vertex")]
    pub fn with_vertex_anthropic_from_env(self) -> Self {
        match crate::providers::chat::vertex::VertexConfig::from_env() {
            Ok(mut config) => {
                config.set_publisher("anthropic");
                match crate::providers::chat::vertex::VertexProvider::with_config(config) {
                    Ok(provider) => self.with_provider("vertex-anthropic", Arc::new(provider)),
                    Err(_) => self,
                }
            }
            Err(_) => self,
        }
    }

    /// Add Vertex AI with Anthropic Claude models and explicit configuration.
    #[cfg(feature = "vertex")]
    pub fn with_vertex_anthropic(
        self,
        project_id: impl Into<String>,
        location: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Result<Self> {
        let config = crate::providers::chat::vertex::VertexConfig::with_publisher(
            project_id,
            location,
            access_token,
            "anthropic",
        );
        let provider = crate::providers::chat::vertex::VertexProvider::with_config(config)?;
        Ok(self.with_provider("vertex-anthropic", Arc::new(provider)))
    }

    /// Add Vertex AI with DeepSeek models from environment.
    #[cfg(feature = "vertex")]
    pub fn with_vertex_deepseek_from_env(self) -> Self {
        match crate::providers::chat::vertex::VertexConfig::from_env() {
            Ok(mut config) => {
                config.set_publisher("deepseek");
                match crate::providers::chat::vertex::VertexProvider::with_config(config) {
                    Ok(provider) => self.with_provider("vertex-deepseek", Arc::new(provider)),
                    Err(_) => self,
                }
            }
            Err(_) => self,
        }
    }

    /// Add Vertex AI with DeepSeek models and explicit configuration.
    #[cfg(feature = "vertex")]
    pub fn with_vertex_deepseek(
        self,
        project_id: impl Into<String>,
        location: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Result<Self> {
        let config = crate::providers::chat::vertex::VertexConfig::with_publisher(
            project_id,
            location,
            access_token,
            "deepseek",
        );
        let provider = crate::providers::chat::vertex::VertexProvider::with_config(config)?;
        Ok(self.with_provider("vertex-deepseek", Arc::new(provider)))
    }

    /// Add Vertex AI with Meta Llama models from environment.
    #[cfg(feature = "vertex")]
    pub fn with_vertex_llama_from_env(self) -> Self {
        match crate::providers::chat::vertex::VertexConfig::from_env() {
            Ok(mut config) => {
                config.set_publisher("meta");
                match crate::providers::chat::vertex::VertexProvider::with_config(config) {
                    Ok(provider) => self.with_provider("vertex-llama", Arc::new(provider)),
                    Err(_) => self,
                }
            }
            Err(_) => self,
        }
    }

    /// Add Vertex AI with Meta Llama models and explicit configuration.
    #[cfg(feature = "vertex")]
    pub fn with_vertex_llama(
        self,
        project_id: impl Into<String>,
        location: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Result<Self> {
        let config = crate::providers::chat::vertex::VertexConfig::with_publisher(
            project_id,
            location,
            access_token,
            "meta",
        );
        let provider = crate::providers::chat::vertex::VertexProvider::with_config(config)?;
        Ok(self.with_provider("vertex-llama", Arc::new(provider)))
    }

    /// Add Vertex AI with Mistral models from environment.
    #[cfg(feature = "vertex")]
    pub fn with_vertex_mistral_from_env(self) -> Self {
        match crate::providers::chat::vertex::VertexConfig::from_env() {
            Ok(mut config) => {
                config.set_publisher("mistralai");
                match crate::providers::chat::vertex::VertexProvider::with_config(config) {
                    Ok(provider) => self.with_provider("vertex-mistral", Arc::new(provider)),
                    Err(_) => self,
                }
            }
            Err(_) => self,
        }
    }

    /// Add Vertex AI with Mistral models and explicit configuration.
    #[cfg(feature = "vertex")]
    pub fn with_vertex_mistral(
        self,
        project_id: impl Into<String>,
        location: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Result<Self> {
        let config = crate::providers::chat::vertex::VertexConfig::with_publisher(
            project_id,
            location,
            access_token,
            "mistralai",
        );
        let provider = crate::providers::chat::vertex::VertexProvider::with_config(config)?;
        Ok(self.with_provider("vertex-mistral", Arc::new(provider)))
    }

    /// Add Vertex AI with AI21 models from environment.
    #[cfg(feature = "vertex")]
    pub fn with_vertex_ai21_from_env(self) -> Self {
        match crate::providers::chat::vertex::VertexConfig::from_env() {
            Ok(mut config) => {
                config.set_publisher("ai21labs");
                match crate::providers::chat::vertex::VertexProvider::with_config(config) {
                    Ok(provider) => self.with_provider("vertex-ai21", Arc::new(provider)),
                    Err(_) => self,
                }
            }
            Err(_) => self,
        }
    }

    /// Add Vertex AI with AI21 models and explicit configuration.
    #[cfg(feature = "vertex")]
    pub fn with_vertex_ai21(
        self,
        project_id: impl Into<String>,
        location: impl Into<String>,
        access_token: impl Into<String>,
    ) -> Result<Self> {
        let config = crate::providers::chat::vertex::VertexConfig::with_publisher(
            project_id,
            location,
            access_token,
            "ai21labs",
        );
        let provider = crate::providers::chat::vertex::VertexProvider::with_config(config)?;
        Ok(self.with_provider("vertex-ai21", Arc::new(provider)))
    }

    // ========== Enterprise Providers ==========

    /// Add Cohere provider from environment.
    ///
    /// Reads: `COHERE_API_KEY` or `CO_API_KEY`
    ///
    /// Also registers Cohere as an embedding provider for embed-* models.
    #[cfg(feature = "cohere")]
    pub fn with_cohere_from_env(mut self) -> Self {
        match crate::providers::chat::cohere::CohereProvider::from_env() {
            Ok(provider) => {
                let provider = Arc::new(provider);
                self.embedding_providers.insert(
                    "cohere".to_string(),
                    Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
                );
                self.with_provider("cohere", provider)
            }
            Err(_) => self,
        }
    }

    /// Add Cohere provider with API key.
    ///
    /// Also registers Cohere as an embedding provider for embed-* models.
    #[cfg(feature = "cohere")]
    pub fn with_cohere(mut self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            Arc::new(crate::providers::chat::cohere::CohereProvider::with_api_key(api_key)?);
        self.embedding_providers.insert(
            "cohere".to_string(),
            Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
        );
        Ok(self.with_provider("cohere", provider))
    }

    /// Add Cohere provider with custom config.
    ///
    /// Also registers Cohere as an embedding provider for embed-* models.
    #[cfg(feature = "cohere")]
    pub fn with_cohere_config(mut self, config: ProviderConfig) -> Result<Self> {
        let provider = Arc::new(crate::providers::chat::cohere::CohereProvider::new(config)?);
        self.embedding_providers.insert(
            "cohere".to_string(),
            Arc::clone(&provider) as Arc<dyn EmbeddingProvider>,
        );
        Ok(self.with_provider("cohere", provider))
    }

    /// Add AI21 provider from environment.
    ///
    /// Reads: `AI21_API_KEY`
    #[cfg(feature = "ai21")]
    pub fn with_ai21_from_env(self) -> Self {
        match crate::providers::chat::ai21::AI21Provider::from_env() {
            Ok(provider) => self.with_provider("ai21", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add AI21 provider with API key.
    #[cfg(feature = "ai21")]
    pub fn with_ai21(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::chat::ai21::AI21Provider::with_api_key(api_key)?;
        Ok(self.with_provider("ai21", Arc::new(provider)))
    }

    /// Add AI21 provider with custom config.
    #[cfg(feature = "ai21")]
    pub fn with_ai21_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::chat::ai21::AI21Provider::new(config)?;
        Ok(self.with_provider("ai21", Arc::new(provider)))
    }

    // ========== Inference Platforms ==========

    /// Add HuggingFace Inference API provider from environment.
    ///
    /// Reads: `HUGGINGFACE_API_KEY` or `HF_TOKEN`
    #[cfg(feature = "huggingface")]
    pub fn with_huggingface_from_env(self) -> Self {
        match crate::providers::chat::huggingface::HuggingFaceProvider::from_env() {
            Ok(provider) => self.with_provider("huggingface", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add HuggingFace Inference API provider with API key.
    #[cfg(feature = "huggingface")]
    pub fn with_huggingface(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::huggingface::HuggingFaceProvider::with_api_key(api_key)?;
        Ok(self.with_provider("huggingface", Arc::new(provider)))
    }

    /// Add HuggingFace dedicated endpoint provider.
    #[cfg(feature = "huggingface")]
    pub fn with_huggingface_endpoint(
        self,
        endpoint_url: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self> {
        let provider = crate::providers::chat::huggingface::HuggingFaceProvider::endpoint(
            endpoint_url,
            api_key,
        )?;
        Ok(self.with_provider("huggingface", Arc::new(provider)))
    }

    /// Add HuggingFace provider with custom config.
    #[cfg(feature = "huggingface")]
    pub fn with_huggingface_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::chat::huggingface::HuggingFaceProvider::new(config)?;
        Ok(self.with_provider("huggingface", Arc::new(provider)))
    }

    /// Add Replicate provider from environment.
    ///
    /// Reads: `REPLICATE_API_TOKEN`
    #[cfg(feature = "replicate")]
    pub fn with_replicate_from_env(self) -> Self {
        match crate::providers::chat::replicate::ReplicateProvider::from_env() {
            Ok(provider) => self.with_provider("replicate", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Replicate provider with API token.
    #[cfg(feature = "replicate")]
    pub fn with_replicate(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::chat::replicate::ReplicateProvider::with_api_key(api_key)?;
        Ok(self.with_provider("replicate", Arc::new(provider)))
    }

    /// Add Replicate provider with custom config.
    #[cfg(feature = "replicate")]
    pub fn with_replicate_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::chat::replicate::ReplicateProvider::new(config)?;
        Ok(self.with_provider("replicate", Arc::new(provider)))
    }

    /// Add Baseten provider from environment.
    ///
    /// Reads: `BASETEN_API_KEY` and optionally `BASETEN_MODEL_ID`
    #[cfg(feature = "baseten")]
    pub fn with_baseten_from_env(self) -> Self {
        match crate::providers::chat::baseten::BasetenProvider::from_env() {
            Ok(provider) => self.with_provider("baseten", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Baseten provider with API key.
    #[cfg(feature = "baseten")]
    pub fn with_baseten(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::chat::baseten::BasetenProvider::with_api_key(api_key)?;
        Ok(self.with_provider("baseten", Arc::new(provider)))
    }

    /// Add Baseten provider with model ID and API key.
    #[cfg(feature = "baseten")]
    pub fn with_baseten_model(
        self,
        model_id: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self> {
        let provider =
            crate::providers::chat::baseten::BasetenProvider::with_model(model_id, api_key)?;
        Ok(self.with_provider("baseten", Arc::new(provider)))
    }

    /// Add Baseten provider with custom config.
    #[cfg(feature = "baseten")]
    pub fn with_baseten_config(self, config: ProviderConfig) -> Result<Self> {
        let provider = crate::providers::chat::baseten::BasetenProvider::new(config)?;
        Ok(self.with_provider("baseten", Arc::new(provider)))
    }

    /// Add RunPod provider from environment.
    ///
    /// Reads: `RUNPOD_API_KEY` and `RUNPOD_ENDPOINT_ID`
    #[cfg(feature = "runpod")]
    pub fn with_runpod_from_env(self) -> Self {
        match crate::providers::chat::runpod::RunPodProvider::from_env() {
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
        let provider = crate::providers::chat::runpod::RunPodProvider::new(endpoint_id, api_key)?;
        Ok(self.with_provider("runpod", Arc::new(provider)))
    }

    // ============ Cloud Providers ============

    /// Add Cloudflare Workers AI provider from environment variables.
    ///
    /// Reads: `CLOUDFLARE_API_TOKEN` and `CLOUDFLARE_ACCOUNT_ID`
    #[cfg(feature = "cloudflare")]
    pub fn with_cloudflare_from_env(self) -> Self {
        match crate::providers::chat::cloudflare::CloudflareProvider::from_env() {
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
            crate::providers::chat::cloudflare::CloudflareProvider::new(account_id, api_token)?;
        Ok(self.with_provider("cloudflare", Arc::new(provider)))
    }

    /// Add IBM watsonx.ai provider from environment variables.
    ///
    /// Reads: `WATSONX_API_KEY` and `WATSONX_PROJECT_ID`
    #[cfg(feature = "watsonx")]
    pub fn with_watsonx_from_env(self) -> Self {
        match crate::providers::chat::watsonx::WatsonxProvider::from_env() {
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
        let provider = crate::providers::chat::watsonx::WatsonxProvider::new(api_key, project_id)?;
        Ok(self.with_provider("watsonx", Arc::new(provider)))
    }

    /// Add Databricks provider from environment variables.
    ///
    /// Reads: `DATABRICKS_TOKEN` and `DATABRICKS_HOST`
    #[cfg(feature = "databricks")]
    pub fn with_databricks_from_env(self) -> Self {
        match crate::providers::chat::databricks::DatabricksProvider::from_env() {
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
        let provider = crate::providers::chat::databricks::DatabricksProvider::new(host, token)?;
        Ok(self.with_provider("databricks", Arc::new(provider)))
    }

    // ============ Specialized/Fast Inference Providers ============

    /// Add SambaNova provider from environment variables.
    ///
    /// Reads: `SAMBANOVA_API_KEY`
    #[cfg(feature = "sambanova")]
    pub fn with_sambanova_from_env(self) -> Self {
        match crate::providers::chat::sambanova::SambaNovaProvider::from_env() {
            Ok(provider) => self.with_provider("sambanova", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add SambaNova provider with API key.
    #[cfg(feature = "sambanova")]
    pub fn with_sambanova(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::chat::sambanova::SambaNovaProvider::with_api_key(api_key)?;
        Ok(self.with_provider("sambanova", Arc::new(provider)))
    }

    // ========== OpenAI-Compatible Providers (Phase 1 Expansion) ==========

    /// Add xAI (Grok) provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_xai_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::xai_from_env() {
            Ok(provider) => self.with_provider("xai", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add xAI (Grok) provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_xai(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::xai(api_key)?;
        Ok(self.with_provider("xai", Arc::new(provider)))
    }

    /// Add Lambda Labs provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_lambda_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::lambda_from_env()
        {
            Ok(provider) => self.with_provider("lambda", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Lambda Labs provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_lambda(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::lambda(api_key)?;
        Ok(self.with_provider("lambda", Arc::new(provider)))
    }

    /// Add Friendli provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_friendli_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::friendli_from_env(
        ) {
            Ok(provider) => self.with_provider("friendli", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Friendli provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_friendli(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::friendli(api_key)?;
        Ok(self.with_provider("friendli", Arc::new(provider)))
    }

    /// Add Volcengine (ByteDance) provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_volcengine_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::volcengine_from_env() {
            Ok(provider) => self.with_provider("volcengine", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Volcengine (ByteDance) provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_volcengine(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::volcengine(
                api_key,
            )?;
        Ok(self.with_provider("volcengine", Arc::new(provider)))
    }

    /// Add Meta Llama API provider from environment.
    #[cfg(feature = "openai-compatible")]
    pub fn with_meta_llama_from_env(self) -> Self {
        match crate::providers::chat::openai_compatible::OpenAICompatibleProvider::meta_llama_from_env() {
            Ok(provider) => self.with_provider("meta_llama", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Meta Llama API provider with API key.
    #[cfg(feature = "openai-compatible")]
    pub fn with_meta_llama(self, api_key: impl Into<String>) -> Result<Self> {
        let provider =
            crate::providers::chat::openai_compatible::OpenAICompatibleProvider::meta_llama(
                api_key,
            )?;
        Ok(self.with_provider("meta_llama", Arc::new(provider)))
    }

    // ========== Custom Providers (Phase 1) ==========

    /// Add DataRobot provider from environment.
    #[cfg(feature = "datarobot")]
    pub fn with_datarobot_from_env(self) -> Self {
        match crate::providers::DataRobotProvider::from_env() {
            Ok(provider) => self.with_provider("datarobot", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add DataRobot provider with API key.
    #[cfg(feature = "datarobot")]
    pub fn with_datarobot(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::DataRobotProvider::with_api_key(api_key)?;
        Ok(self.with_provider("datarobot", Arc::new(provider)))
    }

    /// Add Stability AI provider from environment.
    #[cfg(feature = "stability")]
    pub fn with_stability_from_env(self) -> Self {
        match crate::providers::StabilityProvider::from_env() {
            Ok(provider) => self.with_provider("stability", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Stability AI provider with API key.
    #[cfg(feature = "stability")]
    pub fn with_stability(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::StabilityProvider::with_api_key(api_key)?;
        Ok(self.with_provider("stability", Arc::new(provider)))
    }

    // ========== Specialized APIs (Phase 2.3B) ==========

    /// Add RunwayML provider from environment.
    #[cfg(feature = "runwayml")]
    pub fn with_runwayml_from_env(self) -> Self {
        match crate::providers::RunwayMLProvider::from_env() {
            Ok(provider) => self.with_provider("runwayml", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add RunwayML provider with API key.
    #[cfg(feature = "runwayml")]
    pub fn with_runwayml(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::RunwayMLProvider::with_api_key(api_key)?;
        Ok(self.with_provider("runwayml", Arc::new(provider)))
    }

    /// Add Recraft provider from environment.
    #[cfg(feature = "recraft")]
    pub fn with_recraft_from_env(self) -> Self {
        match crate::providers::RecraftProvider::from_env() {
            Ok(provider) => self.with_provider("recraft", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Recraft provider with API key.
    #[cfg(feature = "recraft")]
    pub fn with_recraft(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::RecraftProvider::with_api_key(api_key)?;
        Ok(self.with_provider("recraft", Arc::new(provider)))
    }

    // ========== Embedding Providers ==========

    /// Add Voyage AI provider from environment.
    #[cfg(feature = "voyage")]
    pub fn with_voyage_from_env(self) -> Self {
        match crate::providers::VoyageProvider::from_env() {
            Ok(provider) => self.with_provider("voyage", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Voyage AI provider with API key.
    #[cfg(feature = "voyage")]
    pub fn with_voyage(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::VoyageProvider::with_api_key(api_key)?;
        Ok(self.with_provider("voyage", Arc::new(provider)))
    }

    /// Add Jina AI provider from environment.
    #[cfg(feature = "jina")]
    pub fn with_jina_from_env(self) -> Self {
        match crate::providers::JinaProvider::from_env() {
            Ok(provider) => self.with_provider("jina", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Jina AI provider with API key.
    #[cfg(feature = "jina")]
    pub fn with_jina(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::JinaProvider::with_api_key(api_key)?;
        Ok(self.with_provider("jina", Arc::new(provider)))
    }

    // ========== Cloud Providers (Phase 3) ==========

    /// Add AWS SageMaker provider from environment.
    #[cfg(feature = "sagemaker")]
    pub async fn with_sagemaker_from_env(self) -> Result<Self> {
        let provider = crate::providers::SageMakerProvider::from_env().await?;
        Ok(self.with_provider("sagemaker", Arc::new(provider)))
    }

    /// Add AWS SageMaker provider with explicit configuration.
    #[cfg(feature = "sagemaker")]
    pub async fn with_sagemaker(self, region: &str, endpoint_name: &str) -> Result<Self> {
        let provider = crate::providers::SageMakerProvider::new(region, endpoint_name).await?;
        Ok(self.with_provider("sagemaker", Arc::new(provider)))
    }

    /// Add Snowflake Cortex provider from environment.
    #[cfg(feature = "snowflake")]
    pub async fn with_snowflake_from_env(self) -> Result<Self> {
        let provider = crate::providers::SnowflakeProvider::from_env().await?;
        Ok(self.with_provider("snowflake", Arc::new(provider)))
    }

    /// Add Snowflake Cortex provider with explicit configuration.
    #[cfg(feature = "snowflake")]
    pub async fn with_snowflake(
        self,
        account: &str,
        user: &str,
        password: &str,
        database: &str,
        schema: &str,
        warehouse: &str,
    ) -> Result<Self> {
        let provider = crate::providers::SnowflakeProvider::new(
            account, user, password, database, schema, warehouse,
        )
        .await?;
        Ok(self.with_provider("snowflake", Arc::new(provider)))
    }

    // ========== Specialized Providers (Phase 4) ==========

    /// Get OpenAI Realtime provider from environment variable `OPENAI_API_KEY`.
    #[cfg(feature = "openai-realtime")]
    pub fn openai_realtime_from_env(&self) -> Result<crate::providers::RealtimeProvider> {
        crate::providers::RealtimeProvider::from_env()
    }

    /// Get OpenAI Realtime provider with explicit API key.
    #[cfg(feature = "openai-realtime")]
    pub fn openai_realtime(&self, api_key: &str) -> crate::providers::RealtimeProvider {
        crate::providers::RealtimeProvider::new(api_key, "gpt-4o-realtime-preview")
    }

    /// Get OpenAI Realtime provider with explicit API key and model.
    #[cfg(feature = "openai-realtime")]
    pub fn openai_realtime_with_model(
        &self,
        api_key: &str,
        model: &str,
    ) -> crate::providers::RealtimeProvider {
        crate::providers::RealtimeProvider::new(api_key, model)
    }

    // ========== Regional Providers (Chinese Market) ==========

    /// Add Baidu Wenxin provider from environment variables.
    ///
    /// Reads `BAIDU_API_KEY` and `BAIDU_SECRET_KEY` environment variables.
    #[cfg(feature = "baidu")]
    pub fn with_baidu_from_env(self) -> Self {
        match crate::providers::BaiduProvider::from_env() {
            Ok(provider) => self.with_provider("baidu", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Baidu Wenxin provider with explicit credentials.
    #[cfg(feature = "baidu")]
    pub fn with_baidu(
        self,
        api_key: impl Into<String>,
        secret_key: impl Into<String>,
    ) -> Result<Self> {
        let provider = crate::providers::BaiduProvider::new(&api_key.into(), &secret_key.into())?;
        Ok(self.with_provider("baidu", Arc::new(provider)))
    }

    /// Add Alibaba DashScope provider from environment variable.
    ///
    /// Reads `ALIBABA_API_KEY` environment variable.
    /// Supports multiple model families: Qwen, Llama, Mistral, Baichuan, and Qwen Code models.
    #[cfg(feature = "alibaba")]
    pub fn with_alibaba_from_env(self) -> Self {
        match crate::providers::AlibabaProvider::from_env() {
            Ok(provider) => self.with_provider("alibaba", Arc::new(provider)),
            Err(_) => self,
        }
    }

    /// Add Alibaba DashScope provider with explicit API key.
    /// Supports multiple model families: Qwen, Llama, Mistral, Baichuan, and Qwen Code models.
    #[cfg(feature = "alibaba")]
    pub fn with_alibaba(self, api_key: impl Into<String>) -> Result<Self> {
        let provider = crate::providers::AlibabaProvider::new(&api_key.into())?;
        Ok(self.with_provider("alibaba", Arc::new(provider)))
    }

    /// Build the client.
    ///
    /// If retry configuration was set via `with_retry()` or `with_default_retry()`,
    /// all providers will be wrapped with automatic retry logic.
    pub fn build(self) -> Result<LLMKitClient> {
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

        Ok(LLMKitClient {
            providers,
            embedding_providers: self.embedding_providers,
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
    fn test_model_parsing_valid_format() {
        // Test valid "provider/model" format
        let (provider, model) =
            parse_model_identifier("anthropic/claude-sonnet-4-20250514").unwrap();
        assert_eq!(provider, "anthropic");
        assert_eq!(model, "claude-sonnet-4-20250514");

        let (provider, model) = parse_model_identifier("openai/gpt-4o").unwrap();
        assert_eq!(provider, "openai");
        assert_eq!(model, "gpt-4o");

        let (provider, model) = parse_model_identifier("groq/llama-3.3-70b-versatile").unwrap();
        assert_eq!(provider, "groq");
        assert_eq!(model, "llama-3.3-70b-versatile");

        let (provider, model) = parse_model_identifier("vertex/gemini-pro").unwrap();
        assert_eq!(provider, "vertex");
        assert_eq!(model, "gemini-pro");

        let (provider, model) = parse_model_identifier("mistral/mistral-large-latest").unwrap();
        assert_eq!(provider, "mistral");
        assert_eq!(model, "mistral-large-latest");

        // Test new Phase 2.3 providers
        let (provider, model) = parse_model_identifier("baidu/ERNIE-Bot-Ultra").unwrap();
        assert_eq!(provider, "baidu");
        assert_eq!(model, "ERNIE-Bot-Ultra");

        let (provider, model) = parse_model_identifier("alibaba/qwen-max").unwrap();
        assert_eq!(provider, "alibaba");
        assert_eq!(model, "qwen-max");

        let (provider, model) = parse_model_identifier("runwayml/gen4_turbo").unwrap();
        assert_eq!(provider, "runwayml");
        assert_eq!(model, "gen4_turbo");

        let (provider, model) = parse_model_identifier("recraft/recraft-v3").unwrap();
        assert_eq!(provider, "recraft");
        assert_eq!(model, "recraft-v3");
    }

    #[test]
    fn test_model_parsing_requires_provider() {
        // Models without provider prefix should return an error
        assert!(parse_model_identifier("claude-sonnet-4-20250514").is_err());
        assert!(parse_model_identifier("gpt-4o").is_err());
        assert!(parse_model_identifier("mistral-large").is_err());
        assert!(parse_model_identifier("model").is_err());
        assert!(parse_model_identifier("").is_err());
    }

    #[test]
    fn test_model_parsing_invalid_provider_format() {
        // HuggingFace-style models with hyphens in org name are not valid provider format
        // "meta-llama" contains "-" so it's NOT treated as a provider prefix
        assert!(parse_model_identifier("meta-llama/Llama-3.2-3B-Instruct").is_err());

        // Model with dots in prefix (not a valid provider)
        assert!(parse_model_identifier("v1.2.3/model").is_err());

        // Model with colons in prefix (not a valid provider)
        assert!(parse_model_identifier("namespace:tag/model").is_err());
    }

    #[test]
    fn test_model_parsing_valid_provider_like_names() {
        // "mistralai" looks like a valid provider name (no special chars)
        // This parses successfully - resolve_provider will error if not registered
        let (provider, model) = parse_model_identifier("mistralai/Mistral-7B-v0.1").unwrap();
        assert_eq!(provider, "mistralai");
        assert_eq!(model, "Mistral-7B-v0.1");
    }
}
