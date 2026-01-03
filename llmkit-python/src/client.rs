//! Synchronous LLMKit client for Python

use std::sync::Arc;

use llmkit::LLMKitClient;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::embedding::{PyEmbeddingRequest, PyEmbeddingResponse};
use crate::errors::convert_error;
use crate::types::request::{
    PyBatchJob, PyBatchRequest, PyBatchResult, PyCompletionRequest, PyTokenCountRequest,
    PyTokenCountResult,
};
use crate::types::response::PyCompletionResponse;
use crate::types::stream::PyStreamIterator;

/// Helper struct for provider configuration
struct ProviderConfigDict {
    api_key: Option<String>,
    base_url: Option<String>,
    endpoint: Option<String>,
    deployment: Option<String>,
    region: Option<String>,
    project: Option<String>,
    location: Option<String>,
    account_id: Option<String>,
    api_token: Option<String>,
    access_token: Option<String>,
    host: Option<String>,
    token: Option<String>,
    endpoint_id: Option<String>,
    model_id: Option<String>,
}

impl ProviderConfigDict {
    fn from_py_dict(dict: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            api_key: dict.get_item("api_key")?.and_then(|v| v.extract().ok()),
            base_url: dict.get_item("base_url")?.and_then(|v| v.extract().ok()),
            endpoint: dict.get_item("endpoint")?.and_then(|v| v.extract().ok()),
            deployment: dict.get_item("deployment")?.and_then(|v| v.extract().ok()),
            region: dict.get_item("region")?.and_then(|v| v.extract().ok()),
            project: dict.get_item("project")?.and_then(|v| v.extract().ok()),
            location: dict.get_item("location")?.and_then(|v| v.extract().ok()),
            account_id: dict.get_item("account_id")?.and_then(|v| v.extract().ok()),
            api_token: dict.get_item("api_token")?.and_then(|v| v.extract().ok()),
            access_token: dict
                .get_item("access_token")?
                .and_then(|v| v.extract().ok()),
            host: dict.get_item("host")?.and_then(|v| v.extract().ok()),
            token: dict.get_item("token")?.and_then(|v| v.extract().ok()),
            endpoint_id: dict.get_item("endpoint_id")?.and_then(|v| v.extract().ok()),
            model_id: dict.get_item("model_id")?.and_then(|v| v.extract().ok()),
        })
    }
}

/// Synchronous LLMKit client.
///
/// This client provides blocking API calls. For async usage, use `AsyncLLMKitClient`.
///
/// Example:
/// ```python
/// from llmkit import LLMKitClient, CompletionRequest, Message
///
/// # Create client from environment variables
/// client = LLMKitClient.from_env()
///
/// # Create client with explicit provider config
/// client = LLMKitClient(providers={
///     "anthropic": {"api_key": "sk-..."},
///     "openai": {"api_key": "sk-..."},
///     "azure": {"api_key": "...", "endpoint": "https://...", "deployment": "gpt-4"},
///     "bedrock": {"region": "us-east-1"},
/// })
///
/// # Make a completion request
/// response = client.complete(CompletionRequest(
///     model="claude-sonnet-4-20250514",
///     messages=[Message.user("Hello!")],
/// ))
/// print(response.text_content())
///
/// # Streaming
/// for chunk in client.complete_stream(request.with_streaming()):
///     if chunk.text:
///         print(chunk.text, end="", flush=True)
/// ```
#[pyclass(name = "LLMKitClient")]
pub struct PyLLMKitClient {
    inner: Arc<LLMKitClient>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyLLMKitClient {
    /// Create a new LLMKit client with provider configurations.
    ///
    /// Args:
    ///     providers: Dict of provider configurations. Each provider can have:
    ///         - api_key: API key for the provider
    ///         - base_url: Custom base URL (optional)
    ///         - endpoint: Azure OpenAI endpoint
    ///         - deployment: Azure OpenAI deployment name
    ///         - region: AWS region for Bedrock, or location for Vertex
    ///         - project: Google Cloud project ID for Vertex
    ///         - account_id: Cloudflare account ID
    ///         - api_token: Cloudflare API token
    ///     default_provider: Optional default provider name
    ///
    /// Supported providers:
    ///     anthropic, openai, azure, bedrock, vertex, google, groq, mistral,
    ///     cohere, ai21, deepseek, together, fireworks, perplexity, cerebras,
    ///     sambanova, openrouter, ollama, huggingface, replicate, cloudflare,
    ///     watsonx, databricks, baseten, runpod, anyscale, deepinfra, novita,
    ///     hyperbolic, lm_studio, vllm, tgi, llamafile
    ///
    /// Returns:
    ///     LLMKitClient: A new sync client instance
    ///
    /// Example:
    ///     client = LLMKitClient(providers={
    ///         "anthropic": {"api_key": "sk-..."},
    ///         "azure": {"api_key": "...", "endpoint": "https://...", "deployment": "gpt-4"},
    ///     })
    #[new]
    #[pyo3(signature = (providers=None, default_provider=None))]
    fn new(
        _py: Python<'_>,
        providers: Option<&Bound<'_, PyDict>>,
        default_provider: Option<String>,
    ) -> PyResult<Self> {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );

        let mut builder = LLMKitClient::builder();

        // Add providers from config dict
        if let Some(providers_dict) = providers {
            for (key, value) in providers_dict.iter() {
                let provider_name: String = key.extract()?;
                let config_dict = value.downcast::<PyDict>()?;
                let config = ProviderConfigDict::from_py_dict(config_dict)?;

                builder = Self::add_provider_to_builder(builder, &provider_name, config, &runtime)?;
            }
        }

        // Set default provider
        if let Some(provider) = default_provider {
            builder = builder.with_default(provider);
        }

        // Enable default retry
        builder = builder.with_default_retry();

        let client = builder
            .build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(client),
            runtime,
        })
    }

    /// Create client from environment variables.
    ///
    /// Automatically detects and configures all available providers from environment variables.
    ///
    /// Supported environment variables:
    ///     - ANTHROPIC_API_KEY: Anthropic (Claude)
    ///     - OPENAI_API_KEY: OpenAI (GPT)
    ///     - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT: Azure OpenAI
    ///     - AWS_REGION or AWS_DEFAULT_REGION: AWS Bedrock (uses default credential chain)
    ///     - GOOGLE_API_KEY: Google AI (Gemini)
    ///     - GOOGLE_CLOUD_PROJECT, VERTEX_LOCATION, VERTEX_ACCESS_TOKEN: Google Vertex AI
    ///     - GROQ_API_KEY: Groq
    ///     - MISTRAL_API_KEY: Mistral
    ///     - COHERE_API_KEY or CO_API_KEY: Cohere
    ///     - AI21_API_KEY: AI21 Labs
    ///     - DEEPSEEK_API_KEY: DeepSeek
    ///     - TOGETHER_API_KEY: Together AI
    ///     - FIREWORKS_API_KEY: Fireworks AI
    ///     - PERPLEXITY_API_KEY: Perplexity
    ///     - CEREBRAS_API_KEY: Cerebras
    ///     - SAMBANOVA_API_KEY: SambaNova
    ///     - OPENROUTER_API_KEY: OpenRouter
    ///     - HUGGINGFACE_API_KEY or HF_TOKEN: HuggingFace
    ///     - REPLICATE_API_TOKEN: Replicate
    ///     - CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID: Cloudflare Workers AI
    ///     - WATSONX_API_KEY, WATSONX_PROJECT_ID: IBM watsonx.ai
    ///     - DATABRICKS_TOKEN, DATABRICKS_HOST: Databricks
    ///     - BASETEN_API_KEY: Baseten
    ///     - RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID: RunPod
    ///     - ANYSCALE_API_KEY: Anyscale
    ///     - DEEPINFRA_API_KEY: DeepInfra
    ///     - NOVITA_API_KEY: Novita AI
    ///     - HYPERBOLIC_API_KEY: Hyperbolic
    ///     - OLLAMA_BASE_URL: Ollama (local, defaults to http://localhost:11434)
    ///
    /// Returns:
    ///     LLMKitClient: A new sync client instance
    #[staticmethod]
    fn from_env() -> PyResult<Self> {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );

        // Build client with all providers from environment
        let builder = LLMKitClient::builder()
            // Core providers
            .with_anthropic_from_env()
            .with_openai_from_env()
            .with_azure_from_env()
            // Google providers
            .with_google_from_env()
            .with_vertex_from_env()
            // Fast inference providers
            .with_groq_from_env()
            .with_mistral_from_env()
            .with_cerebras_from_env()
            .with_sambanova_from_env()
            .with_fireworks_from_env()
            .with_deepseek_from_env()
            // Enterprise providers
            .with_cohere_from_env()
            .with_ai21_from_env()
            // OpenAI-compatible hosted providers
            .with_together_from_env()
            .with_perplexity_from_env()
            .with_anyscale_from_env()
            .with_deepinfra_from_env()
            .with_novita_from_env()
            .with_hyperbolic_from_env()
            // Inference platforms
            .with_huggingface_from_env()
            .with_replicate_from_env()
            .with_baseten_from_env()
            .with_runpod_from_env()
            // Cloud providers
            .with_cloudflare_from_env()
            .with_watsonx_from_env()
            .with_databricks_from_env()
            // Enable retry
            .with_default_retry();

        // Build async for Bedrock (needs await), then finalize
        let client = runtime
            .block_on(async { builder.with_bedrock_from_env().await.build() })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(client),
            runtime,
        })
    }

    /// Make a completion request (synchronous/blocking).
    ///
    /// Args:
    ///     request: The completion request
    ///
    /// Returns:
    ///     CompletionResponse: The completion response
    ///
    /// Raises:
    ///     LLMKitError: If the request fails
    fn complete(
        &self,
        py: Python<'_>,
        request: PyCompletionRequest,
    ) -> PyResult<PyCompletionResponse> {
        let inner = self.inner.clone();
        let req = request.inner.clone();

        py.allow_threads(|| {
            self.runtime.block_on(async move {
                inner
                    .complete(req)
                    .await
                    .map(PyCompletionResponse::from)
                    .map_err(convert_error)
            })
        })
    }

    /// Make a streaming completion request (synchronous iterator).
    ///
    /// Args:
    ///     request: The completion request (should have streaming enabled)
    ///
    /// Returns:
    ///     StreamIterator: Iterator yielding StreamChunk objects
    ///
    /// Raises:
    ///     LLMKitError: If the request fails
    ///
    /// Example:
    ///     for chunk in client.complete_stream(request):
    ///         if chunk.text:
    ///             print(chunk.text, end="", flush=True)
    fn complete_stream(
        &self,
        py: Python<'_>,
        request: PyCompletionRequest,
    ) -> PyResult<PyStreamIterator> {
        let inner = self.inner.clone();
        let mut req = request.inner.clone();

        // Ensure streaming is enabled
        if !req.stream {
            req = req.with_streaming();
        }

        let stream = py
            .allow_threads(|| {
                self.runtime
                    .block_on(async move { inner.complete_stream(req).await })
            })
            .map_err(convert_error)?;

        Ok(PyStreamIterator::new(stream, self.runtime.clone()))
    }

    /// Make a completion request with a specific provider.
    ///
    /// Args:
    ///     provider_name: Name of the provider to use
    ///     request: The completion request
    ///
    /// Returns:
    ///     CompletionResponse: The completion response
    ///
    /// Raises:
    ///     ProviderNotFoundError: If the provider is not configured
    ///     LLMKitError: If the request fails
    fn complete_with_provider(
        &self,
        py: Python<'_>,
        provider_name: String,
        request: PyCompletionRequest,
    ) -> PyResult<PyCompletionResponse> {
        let inner = self.inner.clone();
        let req = request.inner.clone();

        py.allow_threads(|| {
            self.runtime.block_on(async move {
                inner
                    .complete_with_provider(&provider_name, req)
                    .await
                    .map(PyCompletionResponse::from)
                    .map_err(convert_error)
            })
        })
    }

    /// List all registered providers.
    ///
    /// Returns:
    ///     List[str]: Names of all registered providers
    fn providers(&self) -> Vec<String> {
        self.inner
            .providers()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Get the default provider name.
    ///
    /// Returns:
    ///     Optional[str]: Name of the default provider, or None if not set
    #[getter]
    fn default_provider(&self) -> Option<String> {
        self.inner.default_provider().map(|p| p.name().to_string())
    }

    /// Count tokens for a request.
    ///
    /// This allows estimation of token counts before making a completion request,
    /// useful for cost estimation and context window management.
    ///
    /// Note: Not all providers support token counting. Currently only Anthropic
    /// provides native token counting support.
    ///
    /// Args:
    ///     request: TokenCountRequest with model, messages, optional system and tools
    ///
    /// Returns:
    ///     TokenCountResult: Contains input_tokens count
    ///
    /// Raises:
    ///     NotSupportedError: If the provider doesn't support token counting
    ///     LLMKitError: If the request fails
    ///
    /// Example:
    /// ```python
    /// from llmkit import LLMKitClient, TokenCountRequest, Message
    ///
    /// client = LLMKitClient.from_env()
    /// request = TokenCountRequest(
    ///     model="claude-sonnet-4-20250514",
    ///     messages=[Message.user("Hello, how are you?")],
    ///     system="You are a helpful assistant",
    /// )
    /// result = client.count_tokens(request)
    /// print(f"Input tokens: {result.input_tokens}")
    /// ```
    fn count_tokens(
        &self,
        py: Python<'_>,
        request: PyTokenCountRequest,
    ) -> PyResult<PyTokenCountResult> {
        let inner = self.inner.clone();
        let req = request.inner.clone();

        py.allow_threads(|| {
            self.runtime.block_on(async move {
                inner
                    .count_tokens(req)
                    .await
                    .map(PyTokenCountResult::from)
                    .map_err(convert_error)
            })
        })
    }

    // ==================== Batch Processing ====================

    /// Create a batch processing job.
    ///
    /// Batch processing allows you to submit many requests at once for asynchronous
    /// processing, typically at a discounted rate.
    ///
    /// Note: Not all providers support batch processing. Currently Anthropic and OpenAI
    /// support this feature.
    ///
    /// Args:
    ///     requests: List of BatchRequest objects
    ///
    /// Returns:
    ///     BatchJob: Information about the created batch
    ///
    /// Raises:
    ///     NotSupportedError: If the provider doesn't support batch processing
    ///     LLMKitError: If the request fails
    ///
    /// Example:
    /// ```python
    /// requests = [
    ///     BatchRequest("req-1", CompletionRequest(model="claude-sonnet-4-20250514", messages=[Message.user("Hello")])),
    ///     BatchRequest("req-2", CompletionRequest(model="claude-sonnet-4-20250514", messages=[Message.user("Hi")])),
    /// ]
    /// job = client.create_batch(requests)
    /// print(f"Batch created: {job.id}")
    /// ```
    fn create_batch(&self, py: Python<'_>, requests: Vec<PyBatchRequest>) -> PyResult<PyBatchJob> {
        let inner = self.inner.clone();
        let reqs: Vec<_> = requests.into_iter().map(|r| r.inner).collect();

        py.allow_threads(|| {
            self.runtime.block_on(async move {
                inner
                    .create_batch(reqs)
                    .await
                    .map(PyBatchJob::from)
                    .map_err(convert_error)
            })
        })
    }

    /// Get the status of a batch job.
    ///
    /// Args:
    ///     provider_name: Name of the provider (e.g., "anthropic", "openai")
    ///     batch_id: The batch ID returned from create_batch
    ///
    /// Returns:
    ///     BatchJob: Updated batch job information
    fn get_batch(
        &self,
        py: Python<'_>,
        provider_name: String,
        batch_id: String,
    ) -> PyResult<PyBatchJob> {
        let inner = self.inner.clone();

        py.allow_threads(|| {
            self.runtime.block_on(async move {
                inner
                    .get_batch(&provider_name, &batch_id)
                    .await
                    .map(PyBatchJob::from)
                    .map_err(convert_error)
            })
        })
    }

    /// Get the results of a completed batch job.
    ///
    /// Args:
    ///     provider_name: Name of the provider (e.g., "anthropic", "openai")
    ///     batch_id: The batch ID returned from create_batch
    ///
    /// Returns:
    ///     List[BatchResult]: Results for each request in the batch
    fn get_batch_results(
        &self,
        py: Python<'_>,
        provider_name: String,
        batch_id: String,
    ) -> PyResult<Vec<PyBatchResult>> {
        let inner = self.inner.clone();

        py.allow_threads(|| {
            self.runtime.block_on(async move {
                inner
                    .get_batch_results(&provider_name, &batch_id)
                    .await
                    .map(|results| results.into_iter().map(PyBatchResult::from).collect())
                    .map_err(convert_error)
            })
        })
    }

    /// Cancel a batch job.
    ///
    /// Args:
    ///     provider_name: Name of the provider
    ///     batch_id: The batch ID to cancel
    ///
    /// Returns:
    ///     BatchJob: Updated batch job information
    fn cancel_batch(
        &self,
        py: Python<'_>,
        provider_name: String,
        batch_id: String,
    ) -> PyResult<PyBatchJob> {
        let inner = self.inner.clone();

        py.allow_threads(|| {
            self.runtime.block_on(async move {
                inner
                    .cancel_batch(&provider_name, &batch_id)
                    .await
                    .map(PyBatchJob::from)
                    .map_err(convert_error)
            })
        })
    }

    /// List batch jobs for a provider.
    ///
    /// Args:
    ///     provider_name: Name of the provider
    ///     limit: Maximum number of jobs to return (optional)
    ///
    /// Returns:
    ///     List[BatchJob]: List of batch jobs
    #[pyo3(signature = (provider_name, limit=None))]
    fn list_batches(
        &self,
        py: Python<'_>,
        provider_name: String,
        limit: Option<u32>,
    ) -> PyResult<Vec<PyBatchJob>> {
        let inner = self.inner.clone();

        py.allow_threads(|| {
            self.runtime.block_on(async move {
                inner
                    .list_batches(&provider_name, limit)
                    .await
                    .map(|jobs| jobs.into_iter().map(PyBatchJob::from).collect())
                    .map_err(convert_error)
            })
        })
    }

    // ==================== Embeddings ====================

    /// Generate embeddings for text.
    ///
    /// Creates vector representations of text that can be used for semantic search,
    /// clustering, classification, and other NLP tasks.
    ///
    /// Note: Not all providers support embeddings. Currently OpenAI and Cohere
    /// support this feature.
    ///
    /// Args:
    ///     request: EmbeddingRequest with model and text(s) to embed
    ///
    /// Returns:
    ///     EmbeddingResponse: Contains the embedding vectors and usage info
    ///
    /// Raises:
    ///     NotSupportedError: If no embedding provider is configured
    ///     LLMKitError: If the request fails
    ///
    /// Example:
    /// ```python
    /// from llmkit import LLMKitClient, EmbeddingRequest
    ///
    /// client = LLMKitClient.from_env()
    ///
    /// # Single text embedding
    /// response = client.embed(EmbeddingRequest("text-embedding-3-small", "Hello, world!"))
    /// print(f"Dimensions: {response.dimensions}")
    /// print(f"Values: {response.values()[:5]}...")  # First 5 values
    ///
    /// # Batch embeddings
    /// response = client.embed(EmbeddingRequest.batch(
    ///     "text-embedding-3-small",
    ///     ["Hello", "World", "How are you?"]
    /// ))
    /// for emb in response.embeddings:
    ///     print(f"Embedding {emb.index}: {emb.dimensions} dimensions")
    /// ```
    fn embed(&self, py: Python<'_>, request: PyEmbeddingRequest) -> PyResult<PyEmbeddingResponse> {
        let inner = self.inner.clone();
        let req = request.inner.clone();

        py.allow_threads(|| {
            self.runtime.block_on(async move {
                inner
                    .embed(req)
                    .await
                    .map(PyEmbeddingResponse::from)
                    .map_err(convert_error)
            })
        })
    }

    /// Generate embeddings with a specific provider.
    ///
    /// Args:
    ///     provider_name: Name of the embedding provider (e.g., "openai", "cohere")
    ///     request: EmbeddingRequest with model and text(s) to embed
    ///
    /// Returns:
    ///     EmbeddingResponse: Contains the embedding vectors and usage info
    ///
    /// Raises:
    ///     ProviderNotFoundError: If the provider is not configured for embeddings
    ///     LLMKitError: If the request fails
    fn embed_with_provider(
        &self,
        py: Python<'_>,
        provider_name: String,
        request: PyEmbeddingRequest,
    ) -> PyResult<PyEmbeddingResponse> {
        let inner = self.inner.clone();
        let req = request.inner.clone();

        py.allow_threads(|| {
            self.runtime.block_on(async move {
                inner
                    .embed_with_provider(&provider_name, req)
                    .await
                    .map(PyEmbeddingResponse::from)
                    .map_err(convert_error)
            })
        })
    }

    /// List all registered embedding providers.
    ///
    /// Returns:
    ///     List[str]: Names of providers that support embeddings
    fn embedding_providers(&self) -> Vec<String> {
        self.inner
            .embedding_providers()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Check if a provider supports embeddings.
    ///
    /// Args:
    ///     provider_name: Name of the provider to check
    ///
    /// Returns:
    ///     bool: True if the provider supports embeddings
    fn supports_embeddings(&self, provider_name: String) -> bool {
        self.inner.supports_embeddings(&provider_name)
    }

    fn __repr__(&self) -> String {
        let providers = self.providers();
        format!("LLMKitClient(providers={:?})", providers)
    }
}

// Helper methods (non-Python)
impl PyLLMKitClient {
    /// Add a provider to the builder based on the provider name and configuration.
    fn add_provider_to_builder(
        builder: llmkit::ClientBuilder,
        provider_name: &str,
        config: ProviderConfigDict,
        runtime: &Arc<tokio::runtime::Runtime>,
    ) -> PyResult<llmkit::ClientBuilder> {
        use llmkit::providers::chat::azure::AzureConfig;

        let err = |e: llmkit::Error| pyo3::exceptions::PyValueError::new_err(e.to_string());

        match provider_name.to_lowercase().as_str() {
            // Core providers
            "anthropic" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_anthropic(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "anthropic requires 'api_key'",
                    ))
                }
            }
            "openai" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_openai(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "openai requires 'api_key'",
                    ))
                }
            }
            "azure" | "azure_openai" => {
                let api_key = config.api_key.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("azure requires 'api_key'")
                })?;
                let deployment = config.deployment.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("azure requires 'deployment'")
                })?;

                // Either endpoint (full URL) or resource_name is required
                let azure_config = if let Some(endpoint) = config.endpoint.clone() {
                    // Extract resource name from endpoint URL
                    let resource_name = endpoint
                        .trim_start_matches("https://")
                        .split('.')
                        .next()
                        .unwrap_or("azure")
                        .to_string();
                    AzureConfig::new(resource_name, deployment, api_key).with_base_url(endpoint)
                } else if let Some(base_url) = config.base_url.clone() {
                    let resource_name = base_url
                        .trim_start_matches("https://")
                        .split('.')
                        .next()
                        .unwrap_or("azure")
                        .to_string();
                    AzureConfig::new(resource_name, deployment, api_key).with_base_url(base_url)
                } else {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "azure requires 'endpoint' or 'base_url'",
                    ));
                };

                Ok(builder.with_azure(azure_config).map_err(err)?)
            }
            "bedrock" => {
                let region = config.region.unwrap_or_else(|| "us-east-1".to_string());
                runtime.block_on(async { builder.with_bedrock_region(region).await.map_err(err) })
            }
            // Google providers
            "google" | "gemini" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_google(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "google requires 'api_key'",
                    ))
                }
            }
            "vertex" | "vertex_ai" => {
                let project = config.project.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("vertex requires 'project'")
                })?;
                let location = config.location.or(config.region).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "vertex requires 'location' or 'region'",
                    )
                })?;
                let access_token = config.access_token.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("vertex requires 'access_token'")
                })?;
                Ok(builder
                    .with_vertex(project, location, access_token)
                    .map_err(err)?)
            }
            // Fast inference providers
            "groq" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_groq(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "groq requires 'api_key'",
                    ))
                }
            }
            "mistral" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_mistral(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "mistral requires 'api_key'",
                    ))
                }
            }
            "cerebras" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_cerebras(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "cerebras requires 'api_key'",
                    ))
                }
            }
            "sambanova" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_sambanova(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "sambanova requires 'api_key'",
                    ))
                }
            }
            "fireworks" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_fireworks(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "fireworks requires 'api_key'",
                    ))
                }
            }
            "deepseek" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_deepseek(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "deepseek requires 'api_key'",
                    ))
                }
            }
            // Enterprise providers
            "cohere" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_cohere(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "cohere requires 'api_key'",
                    ))
                }
            }
            "ai21" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_ai21(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "ai21 requires 'api_key'",
                    ))
                }
            }
            // OpenAI-compatible hosted providers
            "together" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_together(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "together requires 'api_key'",
                    ))
                }
            }
            "perplexity" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_perplexity(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "perplexity requires 'api_key'",
                    ))
                }
            }
            "anyscale" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_anyscale(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "anyscale requires 'api_key'",
                    ))
                }
            }
            "deepinfra" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_deepinfra(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "deepinfra requires 'api_key'",
                    ))
                }
            }
            "novita" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_novita(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "novita requires 'api_key'",
                    ))
                }
            }
            "hyperbolic" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_hyperbolic(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "hyperbolic requires 'api_key'",
                    ))
                }
            }
            // Inference platforms
            "huggingface" | "hf" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_huggingface(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "huggingface requires 'api_key'",
                    ))
                }
            }
            "replicate" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_replicate(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "replicate requires 'api_key'",
                    ))
                }
            }
            "baseten" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_baseten(key).map_err(err)?)
                } else {
                    Err(pyo3::exceptions::PyValueError::new_err(
                        "baseten requires 'api_key'",
                    ))
                }
            }
            "runpod" => {
                let endpoint_id = config.endpoint_id.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("runpod requires 'endpoint_id'")
                })?;
                let api_key = config.api_key.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("runpod requires 'api_key'")
                })?;
                Ok(builder.with_runpod(endpoint_id, api_key).map_err(err)?)
            }
            // Cloud providers
            "cloudflare" => {
                let account_id = config.account_id.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("cloudflare requires 'account_id'")
                })?;
                let api_token = config.api_token.or(config.api_key).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "cloudflare requires 'api_token' or 'api_key'",
                    )
                })?;
                Ok(builder
                    .with_cloudflare(account_id, api_token)
                    .map_err(err)?)
            }
            "watsonx" => {
                let api_key = config.api_key.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("watsonx requires 'api_key'")
                })?;
                let project_id = config.project.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("watsonx requires 'project'")
                })?;
                Ok(builder.with_watsonx(api_key, project_id).map_err(err)?)
            }
            "databricks" => {
                let host = config.host.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("databricks requires 'host'")
                })?;
                let token = config.token.or(config.api_key).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "databricks requires 'token' or 'api_key'",
                    )
                })?;
                Ok(builder.with_databricks(host, token).map_err(err)?)
            }
            // Local providers
            "ollama" => {
                if let Some(base_url) = config.base_url {
                    Ok(builder
                        .with_openai_compatible("ollama", base_url, None)
                        .map_err(err)?)
                } else {
                    // Default Ollama URL
                    Ok(builder
                        .with_openai_compatible("ollama", "http://localhost:11434/v1", None)
                        .map_err(err)?)
                }
            }
            // Local providers (lm_studio, vllm, tgi, llamafile) are not yet supported in Python bindings
            // They will be implemented in a future release
            // Generic OpenAI-compatible
            "openai_compatible" => {
                let name = config.model_id.unwrap_or_else(|| "custom".to_string());
                let base_url = config.base_url.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("openai_compatible requires 'base_url'")
                })?;
                Ok(builder
                    .with_openai_compatible(name, base_url, config.api_key)
                    .map_err(err)?)
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown provider: '{}'. Supported providers: anthropic, openai, azure, bedrock, \
                google, vertex, groq, mistral, cerebras, sambanova, fireworks, deepseek, cohere, \
                ai21, together, perplexity, anyscale, deepinfra, novita, hyperbolic, huggingface, \
                replicate, baseten, runpod, cloudflare, watsonx, databricks, ollama, lm_studio, \
                vllm, tgi, llamafile, openai_compatible",
                provider_name
            ))),
        }
    }
}
