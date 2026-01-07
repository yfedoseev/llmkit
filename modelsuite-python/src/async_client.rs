//! Asynchronous ModelSuite client for Python

use std::sync::Arc;

use modelsuite::ModelSuiteClient;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyDict};

use crate::audio::{
    PySynthesisRequest, PySynthesizeResponse, PyTranscribeResponse, PyTranscriptionRequest, PyWord,
};
use crate::embedding::{PyEmbeddingRequest, PyEmbeddingResponse};
use crate::errors::convert_error;
use crate::image::{
    PyGeneratedImage, PyImageGenerationRequest, PyImageGenerationResponse, PyImageQuality,
    PyImageSize,
};
use crate::retry::PyRetryConfig;
use crate::specialized::{
    PyClassificationRequest, PyClassificationResponse, PyModerationRequest, PyModerationResponse,
    PyRankedDocument, PyRankingRequest, PyRankingResponse, PyRerankedResult, PyRerankingRequest,
    PyRerankingResponse,
};
use crate::types::request::{
    PyBatchJob, PyBatchRequest, PyBatchResult, PyCompletionRequest, PyTokenCountRequest,
    PyTokenCountResult,
};
use crate::types::response::PyCompletionResponse;
use crate::types::stream::PyAsyncStreamIterator;
use crate::video::{PyVideoGenerationRequest, PyVideoGenerationResponse};

/// Helper struct for provider configuration
#[allow(dead_code)] // access_token kept for backward compatibility
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

/// Asynchronous ModelSuite client.
///
/// This client provides async/await API calls. For sync usage, use `ModelSuiteClient`.
///
/// Example:
/// ```python
/// import asyncio
/// from modelsuite import AsyncModelSuiteClient, CompletionRequest, Message
///
/// async def main():
///     # Create client from environment variables
///     client = AsyncModelSuiteClient.from_env()
///
///     # Create client with explicit provider config
///     client = AsyncModelSuiteClient(providers={
///         "anthropic": {"api_key": "sk-..."},
///         "openai": {"api_key": "sk-..."},
///     })
///
///     # Make a completion request
///     response = await client.complete(CompletionRequest(
///         model="claude-sonnet-4-20250514",
///         messages=[Message.user("Hello!")],
///     ))
///     print(response.text_content())
///
///     # Streaming
///     async for chunk in client.complete_stream(request.with_streaming()):
///         if chunk.text:
///             print(chunk.text, end="", flush=True)
///
/// asyncio.run(main())
/// ```
#[pyclass(name = "AsyncModelSuiteClient")]
pub struct PyAsyncModelSuiteClient {
    inner: Arc<ModelSuiteClient>,
}

#[pymethods]
impl PyAsyncModelSuiteClient {
    /// Create a new async ModelSuite client with provider configurations.
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
    ///     retry_config: Retry configuration. Can be:
    ///         - None (default): Use default retry (10 retries with exponential backoff)
    ///         - RetryConfig instance: Use custom retry configuration
    ///         - False: Disable retry entirely
    ///
    /// Supported providers:
    ///     anthropic, openai, azure, bedrock, vertex, google, groq, mistral,
    ///     cohere, ai21, deepseek, together, fireworks, perplexity, cerebras,
    ///     sambanova, openrouter, ollama, huggingface, replicate, cloudflare,
    ///     watsonx, databricks, baseten, runpod, anyscale, deepinfra, novita,
    ///     hyperbolic, lm_studio, vllm, tgi, llamafile
    ///
    /// Returns:
    ///     AsyncModelSuiteClient: A new async client instance
    ///
    /// Example:
    ///     # With custom retry
    ///     client = AsyncModelSuiteClient(retry_config=RetryConfig.conservative())
    ///
    ///     # Disable retry
    ///     client = AsyncModelSuiteClient(retry_config=False)
    #[new]
    #[pyo3(signature = (providers=None, default_provider=None, retry_config=None))]
    fn new(
        py: Python<'_>,
        providers: Option<&Bound<'_, PyDict>>,
        default_provider: Option<String>,
        retry_config: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        // Create a temporary runtime for initialization
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let mut builder = ModelSuiteClient::builder();

        // Add providers from config dict
        if let Some(providers_dict) = providers {
            for (key, value) in providers_dict.iter() {
                let provider_name: String = key.extract()?;
                let config_dict = value.cast::<PyDict>()?;
                let config = ProviderConfigDict::from_py_dict(config_dict)?;

                builder = Self::add_provider_to_builder(builder, &provider_name, config, &runtime)?;
            }
        }

        // Set default provider
        if let Some(provider) = default_provider {
            builder = builder.with_default(provider);
        }

        // Configure retry
        builder = Self::apply_retry_config(py, builder, retry_config)?;

        let client = builder
            .build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(client),
        })
    }

    /// Create client from environment variables.
    ///
    /// Automatically detects and configures all available providers from environment variables.
    ///
    /// Args:
    ///     retry_config: Retry configuration. Can be:
    ///         - None (default): Use default retry (10 retries with exponential backoff)
    ///         - RetryConfig instance: Use custom retry configuration
    ///         - False: Disable retry entirely
    ///
    /// Supported environment variables:
    ///     - ANTHROPIC_API_KEY: Anthropic (Claude)
    ///     - OPENAI_API_KEY: OpenAI (GPT)
    ///     - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT: Azure OpenAI
    ///     - OPENROUTER_API_KEY: OpenRouter
    ///     - AWS_REGION or AWS_DEFAULT_REGION: AWS Bedrock (uses default credential chain)
    ///     - GOOGLE_API_KEY: Google AI (Gemini)
    ///     - GOOGLE_CLOUD_PROJECT, VERTEX_LOCATION, VERTEX_ACCESS_TOKEN: Google Vertex AI
    ///     - GROQ_API_KEY: Groq
    ///     - MISTRAL_API_KEY: Mistral
    ///     - COHERE_API_KEY or CO_API_KEY: Cohere
    ///     - AI21_API_KEY: AI21 Labs
    ///     - DEEPSEEK_API_KEY: DeepSeek
    ///     - XAI_API_KEY: xAI (Grok)
    ///     - TOGETHER_API_KEY: Together AI
    ///     - FIREWORKS_API_KEY: Fireworks AI
    ///     - PERPLEXITY_API_KEY: Perplexity
    ///     - CEREBRAS_API_KEY: Cerebras
    ///     - SAMBANOVA_API_KEY: SambaNova
    ///     - NVIDIA_NIM_API_KEY: NVIDIA NIM
    ///     - DATAROBOT_API_KEY: DataRobot
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
    ///     - LAMBDA_API_KEY: Lambda
    ///     - FRIENDLI_API_KEY: Friendli
    ///     - BAIDU_API_KEY: Baidu (ERNIE)
    ///     - ALIBABA_API_KEY: Alibaba (Qwen)
    ///     - VOLCENGINE_API_KEY: Volcengine
    ///     - MARITACA_API_KEY: Maritaca
    ///     - LIGHTON_API_KEY: LightOn
    ///     - VOYAGE_API_KEY: Voyage AI
    ///     - JINA_API_KEY: Jina AI
    ///     - STABILITY_API_KEY: Stability AI
    ///     - OLLAMA_BASE_URL: Ollama (local, defaults to http://localhost:11434)
    ///
    /// Returns:
    ///     AsyncModelSuiteClient: A new async client instance
    ///
    /// Example:
    ///     # Default retry
    ///     client = AsyncModelSuiteClient.from_env()
    ///
    ///     # Custom retry
    ///     client = AsyncModelSuiteClient.from_env(retry_config=RetryConfig.conservative())
    ///
    ///     # Disable retry
    ///     client = AsyncModelSuiteClient.from_env(retry_config=False)
    #[staticmethod]
    #[pyo3(signature = (retry_config=None))]
    fn from_env(py: Python<'_>, retry_config: Option<Py<PyAny>>) -> PyResult<Self> {
        // Create a temporary runtime for initialization
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        // Build client with all providers from environment
        let builder = ModelSuiteClient::builder()
            // Core providers
            .with_anthropic_from_env()
            .with_openai_from_env()
            .with_azure_from_env()
            .with_openrouter_from_env()
            // Google providers
            .with_google_from_env()
            // Note: Vertex is async and handled below with Bedrock
            // Fast inference providers
            .with_groq_from_env()
            .with_mistral_from_env()
            .with_cerebras_from_env()
            .with_sambanova_from_env()
            .with_fireworks_from_env()
            .with_deepseek_from_env()
            .with_xai_from_env()
            // Enterprise providers
            .with_cohere_from_env()
            .with_ai21_from_env()
            .with_nvidia_nim_from_env()
            .with_datarobot_from_env()
            // OpenAI-compatible hosted providers
            .with_together_from_env()
            .with_perplexity_from_env()
            .with_anyscale_from_env()
            .with_deepinfra_from_env()
            .with_novita_from_env()
            .with_hyperbolic_from_env()
            .with_lambda_from_env()
            .with_friendli_from_env()
            // Inference platforms
            .with_huggingface_from_env()
            .with_replicate_from_env()
            .with_baseten_from_env()
            .with_runpod_from_env()
            // Cloud providers
            .with_cloudflare_from_env()
            .with_watsonx_from_env()
            .with_databricks_from_env()
            // Asian providers
            .with_baidu_from_env()
            .with_alibaba_from_env()
            .with_volcengine_from_env()
            // Regional providers
            .with_maritaca_from_env()
            .with_lighton_from_env()
            // Embedding/multimodal providers
            .with_voyage_from_env()
            .with_jina_from_env()
            .with_stability_from_env();

        // Apply retry config
        let builder = Self::apply_retry_config(py, builder, retry_config)?;

        // Build async for Vertex and Bedrock (both need await), then finalize
        let client = runtime
            .block_on(async {
                builder
                    .with_vertex_from_env()
                    .await
                    .with_bedrock_from_env()
                    .await
                    .build()
            })
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(client),
        })
    }

    /// Make a completion request (asynchronous).
    ///
    /// Args:
    ///     request: The completion request
    ///
    /// Returns:
    ///     CompletionResponse: The completion response
    ///
    /// Raises:
    ///     ModelSuiteError: If the request fails
    fn complete<'py>(
        &self,
        py: Python<'py>,
        request: PyCompletionRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let req = request.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .complete(req)
                .await
                .map(PyCompletionResponse::from)
                .map_err(convert_error)
        })
    }

    /// Make a streaming completion request (async iterator).
    ///
    /// Args:
    ///     request: The completion request (should have streaming enabled)
    ///
    /// Returns:
    ///     AsyncStreamIterator: Async iterator yielding StreamChunk objects
    ///
    /// Raises:
    ///     ModelSuiteError: If the request fails
    ///
    /// Example:
    ///     async for chunk in client.complete_stream(request):
    ///         if chunk.text:
    ///             print(chunk.text, end="", flush=True)
    fn complete_stream<'py>(
        &self,
        py: Python<'py>,
        request: PyCompletionRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let mut req = request.inner.clone();

        // Ensure streaming is enabled
        if !req.stream {
            req = req.with_streaming();
        }

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = inner.complete_stream(req).await.map_err(convert_error)?;

            Ok(PyAsyncStreamIterator::new(stream))
        })
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
    ///     ModelSuiteError: If the request fails
    fn complete_with_provider<'py>(
        &self,
        py: Python<'py>,
        provider_name: String,
        request: PyCompletionRequest,
    ) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let req = request.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .complete_with_provider(&provider_name, req)
                .await
                .map(PyCompletionResponse::from)
                .map_err(convert_error)
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

    /// Count tokens for a request (async).
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
    ///     ModelSuiteError: If the request fails
    ///
    /// Example:
    /// ```python
    /// from modelsuite import AsyncModelSuiteClient, TokenCountRequest, Message
    ///
    /// client = AsyncModelSuiteClient.from_env()
    /// request = TokenCountRequest(
    ///     model="claude-sonnet-4-20250514",
    ///     messages=[Message.user("Hello, how are you?")],
    ///     system="You are a helpful assistant",
    /// )
    /// result = await client.count_tokens(request)
    /// print(f"Input tokens: {result.input_tokens}")
    /// ```
    fn count_tokens<'py>(
        &self,
        py: Python<'py>,
        request: PyTokenCountRequest,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let req = request.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .count_tokens(req)
                .await
                .map(PyTokenCountResult::from)
                .map_err(convert_error)
        })
    }

    // ==================== Batch Processing ====================

    /// Create a batch processing job (async).
    fn create_batch<'py>(
        &self,
        py: Python<'py>,
        requests: Vec<PyBatchRequest>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let reqs: Vec<_> = requests.into_iter().map(|r| r.inner).collect();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .create_batch(reqs)
                .await
                .map(PyBatchJob::from)
                .map_err(convert_error)
        })
    }

    /// Get the status of a batch job (async).
    fn get_batch<'py>(
        &self,
        py: Python<'py>,
        provider_name: String,
        batch_id: String,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .get_batch(&provider_name, &batch_id)
                .await
                .map(PyBatchJob::from)
                .map_err(convert_error)
        })
    }

    /// Get the results of a completed batch job (async).
    fn get_batch_results<'py>(
        &self,
        py: Python<'py>,
        provider_name: String,
        batch_id: String,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .get_batch_results(&provider_name, &batch_id)
                .await
                .map(|results| {
                    results
                        .into_iter()
                        .map(PyBatchResult::from)
                        .collect::<Vec<_>>()
                })
                .map_err(convert_error)
        })
    }

    /// Cancel a batch job (async).
    fn cancel_batch<'py>(
        &self,
        py: Python<'py>,
        provider_name: String,
        batch_id: String,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .cancel_batch(&provider_name, &batch_id)
                .await
                .map(PyBatchJob::from)
                .map_err(convert_error)
        })
    }

    /// List batch jobs for a provider (async).
    #[pyo3(signature = (provider_name, limit=None))]
    fn list_batches<'py>(
        &self,
        py: Python<'py>,
        provider_name: String,
        limit: Option<u32>,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .list_batches(&provider_name, limit)
                .await
                .map(|jobs| jobs.into_iter().map(PyBatchJob::from).collect::<Vec<_>>())
                .map_err(convert_error)
        })
    }

    // ==================== Embeddings ====================

    /// Generate embeddings for text (async).
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
    ///     ModelSuiteError: If the request fails
    ///
    /// Example:
    /// ```python
    /// from modelsuite import AsyncModelSuiteClient, EmbeddingRequest
    ///
    /// client = AsyncModelSuiteClient.from_env()
    ///
    /// # Single text embedding
    /// response = await client.embed(EmbeddingRequest("text-embedding-3-small", "Hello, world!"))
    /// print(f"Dimensions: {response.dimensions}")
    ///
    /// # Batch embeddings
    /// response = await client.embed(EmbeddingRequest.batch(
    ///     "text-embedding-3-small",
    ///     ["Hello", "World", "How are you?"]
    /// ))
    /// for emb in response.embeddings:
    ///     print(f"Embedding {emb.index}: {emb.dimensions} dimensions")
    /// ```
    fn embed<'py>(
        &self,
        py: Python<'py>,
        request: PyEmbeddingRequest,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let req = request.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .embed(req)
                .await
                .map(PyEmbeddingResponse::from)
                .map_err(convert_error)
        })
    }

    /// Generate embeddings with a specific provider (async).
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
    ///     ModelSuiteError: If the request fails
    fn embed_with_provider<'py>(
        &self,
        py: Python<'py>,
        provider_name: String,
        request: PyEmbeddingRequest,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let req = request.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            inner
                .embed_with_provider(&provider_name, req)
                .await
                .map(PyEmbeddingResponse::from)
                .map_err(convert_error)
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

    // ==================== Audio APIs ====================

    /// Transcribe audio to text (async).
    ///
    /// Args:
    ///     request: TranscriptionRequest with audio data
    ///
    /// Returns:
    ///     TranscribeResponse with transcript text
    fn transcribe_audio<'py>(
        &self,
        py: Python<'py>,
        request: PyTranscriptionRequest,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "deepgram/nova-2".to_string());
        let audio_input =
            modelsuite::AudioInput::bytes(request.audio_bytes.clone(), "audio.mp3", "audio/mpeg");
        let core_request = modelsuite::TranscriptionRequest::new(model, audio_input);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner.transcribe(core_request).await;
            match result {
                Ok(response) => {
                    let words = response
                        .words
                        .unwrap_or_default()
                        .into_iter()
                        .map(|w| PyWord {
                            word: w.word,
                            start: w.start as f64,
                            end: w.end as f64,
                            confidence: 1.0,
                            speaker: None,
                        })
                        .collect();

                    Ok(PyTranscribeResponse {
                        transcript: response.text,
                        confidence: None,
                        words,
                        duration: response.duration.map(|d| d as f64),
                        metadata: None,
                    })
                }
                Err(e) => Err(convert_error(e)),
            }
        })
    }

    /// Synthesize text to speech (async).
    ///
    /// Args:
    ///     request: SynthesisRequest with text and voice settings
    ///
    /// Returns:
    ///     SynthesizeResponse with audio data
    fn synthesize_speech<'py>(
        &self,
        py: Python<'py>,
        request: PySynthesisRequest,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "elevenlabs/eleven_monolingual_v1".to_string());
        let voice = request
            .voice_id
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let core_request = modelsuite::SpeechRequest::new(model, request.text.clone(), voice);

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner.speech(core_request).await;
            match result {
                Ok(response) => {
                    let format = match response.format {
                        modelsuite::AudioFormat::Mp3 => "mp3",
                        modelsuite::AudioFormat::Opus => "opus",
                        modelsuite::AudioFormat::Aac => "aac",
                        modelsuite::AudioFormat::Flac => "flac",
                        modelsuite::AudioFormat::Wav => "wav",
                        modelsuite::AudioFormat::Pcm => "pcm",
                    };

                    Ok(PySynthesizeResponse {
                        audio_bytes: response.audio,
                        format: format.to_string(),
                        duration: response.duration_seconds.map(|d| d as f64),
                    })
                }
                Err(e) => Err(convert_error(e)),
            }
        })
    }

    // ==================== Image APIs ====================

    /// Generate images from a text prompt (async).
    ///
    /// Args:
    ///     request: ImageGenerationRequest with prompt and parameters
    ///
    /// Returns:
    ///     ImageGenerationResponse with generated images
    fn generate_image<'py>(
        &self,
        py: Python<'py>,
        request: PyImageGenerationRequest,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let mut core_request =
            modelsuite::ImageGenerationRequest::new(request.model.clone(), request.prompt.clone());

        if let Some(n) = request.n {
            core_request = core_request.with_n(n);
        }
        if let Some(size) = request.size {
            let image_size = match size {
                PyImageSize::Square256 => modelsuite::ImageSize::Square256,
                PyImageSize::Square512 => modelsuite::ImageSize::Square512,
                PyImageSize::Square1024 => modelsuite::ImageSize::Square1024,
                PyImageSize::Portrait1024x1792 => modelsuite::ImageSize::Portrait1024x1792,
                PyImageSize::Landscape1792x1024 => modelsuite::ImageSize::Landscape1792x1024,
            };
            core_request = core_request.with_size(image_size);
        }
        if let Some(quality) = request.quality {
            let image_quality = match quality {
                PyImageQuality::Hd => modelsuite::ImageQuality::Hd,
                PyImageQuality::Standard => modelsuite::ImageQuality::Standard,
            };
            core_request = core_request.with_quality(image_quality);
        }

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner.generate_image(core_request).await;
            match result {
                Ok(response) => {
                    let images = response
                        .images
                        .into_iter()
                        .map(|img| PyGeneratedImage {
                            url: img.url,
                            b64_json: img.b64_json,
                            revised_prompt: img.revised_prompt,
                        })
                        .collect();

                    Ok(PyImageGenerationResponse {
                        created: response.created,
                        images,
                    })
                }
                Err(e) => Err(convert_error(e)),
            }
        })
    }

    // ==================== Video APIs ====================

    /// Generate video from a prompt (async).
    ///
    /// Args:
    ///     request: VideoGenerationRequest with prompt and parameters
    ///
    /// Returns:
    ///     VideoGenerationResponse with video URL or job ID
    fn generate_video<'py>(
        &self,
        py: Python<'py>,
        request: PyVideoGenerationRequest,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "runwayml/gen-3".to_string());
        let mut core_request =
            modelsuite::VideoGenerationRequest::new(model, request.prompt.clone());

        if let Some(duration) = request.duration {
            core_request = core_request.with_duration(duration);
        }

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner.generate_video(core_request).await;
            match result {
                Ok(response) => {
                    let (status, video_url, duration) = match response.status {
                        modelsuite::VideoJobStatus::Queued => ("queued".to_string(), None, None),
                        modelsuite::VideoJobStatus::Processing { .. } => {
                            ("processing".to_string(), None, None)
                        }
                        modelsuite::VideoJobStatus::Completed {
                            video_url,
                            duration_seconds,
                            ..
                        } => (
                            "completed".to_string(),
                            Some(video_url),
                            duration_seconds.map(|d| d as f64),
                        ),
                        modelsuite::VideoJobStatus::Failed { error, .. } => {
                            (format!("failed:{}", error), None, None)
                        }
                        modelsuite::VideoJobStatus::Cancelled => {
                            ("cancelled".to_string(), None, None)
                        }
                    };

                    Ok(PyVideoGenerationResponse {
                        video_bytes: None,
                        video_url,
                        format: "mp4".to_string(),
                        duration,
                        width: None,
                        height: None,
                        task_id: Some(response.job_id),
                        status: Some(status),
                    })
                }
                Err(e) => Err(convert_error(e)),
            }
        })
    }

    // ==================== Specialized APIs ====================

    /// Rank documents by relevance to a query (async).
    ///
    /// Args:
    ///     request: RankingRequest with query and documents
    ///
    /// Returns:
    ///     RankingResponse with ranked documents
    fn rank_documents<'py>(
        &self,
        py: Python<'py>,
        request: PyRankingRequest,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let mut core_request = modelsuite::RankingRequest::new(
            request.model.clone(),
            request.query.clone(),
            request.documents.clone(),
        );

        if let Some(top_k) = request.top_k {
            core_request = core_request.with_top_k(top_k);
        }
        core_request = core_request.with_documents();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner.rank(core_request).await;
            match result {
                Ok(response) => {
                    let results = response
                        .results
                        .into_iter()
                        .map(|r| PyRankedDocument {
                            index: r.index,
                            document: r.document.unwrap_or_default(),
                            score: r.score as f64,
                        })
                        .collect();

                    Ok(PyRankingResponse { results })
                }
                Err(e) => Err(convert_error(e)),
            }
        })
    }

    /// Rerank search results for semantic relevance (async).
    ///
    /// Args:
    ///     request: RerankingRequest with query and documents
    ///
    /// Returns:
    ///     RerankingResponse with reranked results
    fn rerank_results<'py>(
        &self,
        py: Python<'py>,
        request: PyRerankingRequest,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let mut core_request = modelsuite::RankingRequest::new(
            request.model.clone(),
            request.query.clone(),
            request.documents.clone(),
        );

        if let Some(top_n) = request.top_n {
            core_request = core_request.with_top_k(top_n);
        }
        core_request = core_request.with_documents();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner.rank(core_request).await;
            match result {
                Ok(response) => {
                    let results = response
                        .results
                        .into_iter()
                        .map(|r| PyRerankedResult {
                            index: r.index,
                            document: r.document.unwrap_or_default(),
                            relevance_score: r.score as f64,
                        })
                        .collect();

                    Ok(PyRerankingResponse { results })
                }
                Err(e) => Err(convert_error(e)),
            }
        })
    }

    /// Check content for policy violations (async).
    ///
    /// Args:
    ///     request: ModerationRequest with text to check
    ///
    /// Returns:
    ///     ModerationResponse with flagged status and scores
    fn moderate_text<'py>(
        &self,
        py: Python<'py>,
        request: PyModerationRequest,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let core_request =
            modelsuite::ModerationRequest::new(request.model.clone(), request.text.clone());

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner.moderate(core_request).await;
            match result {
                Ok(response) => {
                    let scores = crate::specialized::PyModerationScores {
                        hate: response.category_scores.hate as f64,
                        hate_threatening: 0.0,
                        harassment: response.category_scores.harassment as f64,
                        harassment_threatening: 0.0,
                        self_harm: response.category_scores.self_harm as f64,
                        self_harm_intent: 0.0,
                        self_harm_instructions: 0.0,
                        sexual: response.category_scores.sexual as f64,
                        sexual_minors: 0.0,
                        violence: response.category_scores.violence as f64,
                        violence_graphic: 0.0,
                    };

                    Ok(PyModerationResponse {
                        flagged: response.flagged,
                        scores,
                    })
                }
                Err(e) => Err(convert_error(e)),
            }
        })
    }

    /// Classify text into provided labels (async).
    ///
    /// Args:
    ///     request: ClassificationRequest with text and labels
    ///
    /// Returns:
    ///     ClassificationResponse with classifications
    fn classify_text<'py>(
        &self,
        py: Python<'py>,
        request: PyClassificationRequest,
    ) -> PyResult<Bound<'py, pyo3::PyAny>> {
        let inner = self.inner.clone();
        let core_request = modelsuite::ClassificationRequest::new(
            request.model.clone(),
            request.text.clone(),
            request.labels.clone(),
        );

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let result = inner.classify(core_request).await;
            match result {
                Ok(response) => {
                    let results = response
                        .predictions
                        .into_iter()
                        .map(|p| crate::specialized::PyClassificationResult {
                            label: p.label,
                            confidence: p.score as f64,
                        })
                        .collect();

                    Ok(PyClassificationResponse { results })
                }
                Err(e) => Err(convert_error(e)),
            }
        })
    }

    fn __repr__(&self) -> String {
        let providers = self.providers();
        format!("AsyncModelSuiteClient(providers={:?})", providers)
    }
}

// Helper methods (non-Python)
impl PyAsyncModelSuiteClient {
    /// Apply retry configuration to the builder.
    ///
    /// - None: Use default retry (with_default_retry())
    /// - PyRetryConfig: Use custom retry configuration
    /// - False: Disable retry entirely
    fn apply_retry_config(
        py: Python<'_>,
        builder: modelsuite::ClientBuilder,
        retry_config: Option<Py<PyAny>>,
    ) -> PyResult<modelsuite::ClientBuilder> {
        match retry_config {
            None => {
                // Default: use production retry config
                Ok(builder.with_default_retry())
            }
            Some(config) => {
                // Check if it's False (bool)
                if let Ok(false_val) = config.extract::<bool>(py) {
                    if !false_val {
                        // retry_config=False means no retry
                        return Ok(builder);
                    }
                }

                // Try to extract as PyRetryConfig
                if let Ok(retry) = config.extract::<PyRetryConfig>(py) {
                    Ok(builder.with_retry(retry.inner))
                } else {
                    Err(pyo3::exceptions::PyTypeError::new_err(
                        "retry_config must be RetryConfig, False, or None",
                    ))
                }
            }
        }
    }

    /// Add a provider to the builder based on the provider name and configuration.
    fn add_provider_to_builder(
        builder: modelsuite::ClientBuilder,
        provider_name: &str,
        config: ProviderConfigDict,
        runtime: &tokio::runtime::Runtime,
    ) -> PyResult<modelsuite::ClientBuilder> {
        use modelsuite::providers::chat::azure::AzureConfig;

        let err = |e: modelsuite::Error| pyo3::exceptions::PyValueError::new_err(e.to_string());

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
                // Vertex now uses ADC (Application Default Credentials)
                // Service account file path can be set via GOOGLE_APPLICATION_CREDENTIALS
                // or use gcloud auth application-default login
                let project = config.project.ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err(
                        "vertex requires 'project'. Also set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'",
                    )
                })?;
                let location = config
                    .location
                    .or(config.region)
                    .unwrap_or_else(|| "us-central1".to_string());

                // Set environment variables for Vertex config
                std::env::set_var("GOOGLE_CLOUD_PROJECT", &project);
                std::env::set_var("GOOGLE_CLOUD_LOCATION", &location);

                // Use async from_env (ADC) - returns builder directly (silently skips if no creds)
                Ok(runtime.block_on(async { builder.with_vertex_from_env().await }))
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
