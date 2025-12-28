//! Synchronous LLMKit client for Python

use std::sync::Arc;

use llmkit::LLMKitClient;
use pyo3::prelude::*;

use crate::errors::convert_error;
use crate::types::request::PyCompletionRequest;
use crate::types::response::PyCompletionResponse;
use crate::types::stream::PyStreamIterator;

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
    /// Create a new LLMKit client with explicit API keys.
    ///
    /// Args:
    ///     anthropic_api_key: Optional Anthropic API key
    ///     openai_api_key: Optional OpenAI API key
    ///     groq_api_key: Optional Groq API key
    ///     mistral_api_key: Optional Mistral API key
    ///     default_provider: Optional default provider name
    ///
    /// Returns:
    ///     LLMKitClient: A new sync client instance
    #[new]
    #[pyo3(signature = (
        anthropic_api_key = None,
        openai_api_key = None,
        groq_api_key = None,
        mistral_api_key = None,
        default_provider = None,
    ))]
    fn new(
        anthropic_api_key: Option<String>,
        openai_api_key: Option<String>,
        groq_api_key: Option<String>,
        mistral_api_key: Option<String>,
        default_provider: Option<String>,
    ) -> PyResult<Self> {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );

        let mut builder = LLMKitClient::builder();

        // Add providers with explicit keys
        if let Some(key) = anthropic_api_key {
            builder = builder
                .with_anthropic(key)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }
        if let Some(key) = openai_api_key {
            builder = builder
                .with_openai(key)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }
        if let Some(key) = groq_api_key {
            builder = builder
                .with_groq(key)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }
        if let Some(key) = mistral_api_key {
            builder = builder
                .with_mistral(key)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
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
    /// Automatically detects and configures providers from environment variables:
    /// - ANTHROPIC_API_KEY
    /// - OPENAI_API_KEY
    /// - GROQ_API_KEY
    /// - MISTRAL_API_KEY
    /// - etc.
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

        let client = LLMKitClient::builder()
            .with_anthropic_from_env()
            .with_openai_from_env()
            .with_groq_from_env()
            .with_mistral_from_env()
            .with_default_retry()
            .build()
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

    fn __repr__(&self) -> String {
        let providers = self.providers();
        format!("LLMKitClient(providers={:?})", providers)
    }
}
