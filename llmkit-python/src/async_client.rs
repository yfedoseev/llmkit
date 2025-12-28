//! Asynchronous LLMKit client for Python

use std::sync::Arc;

use llmkit::LLMKitClient;
use pyo3::prelude::*;

use crate::errors::convert_error;
use crate::types::request::PyCompletionRequest;
use crate::types::response::PyCompletionResponse;
use crate::types::stream::PyAsyncStreamIterator;

/// Asynchronous LLMKit client.
///
/// This client provides async/await API calls. For sync usage, use `LLMKitClient`.
///
/// Example:
/// ```python
/// import asyncio
/// from llmkit import AsyncLLMKitClient, CompletionRequest, Message
///
/// async def main():
///     # Create client from environment variables
///     client = AsyncLLMKitClient.from_env()
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
#[pyclass(name = "AsyncLLMKitClient")]
pub struct PyAsyncLLMKitClient {
    inner: Arc<LLMKitClient>,
}

#[pymethods]
impl PyAsyncLLMKitClient {
    /// Create a new async LLMKit client with explicit API keys.
    ///
    /// Args:
    ///     anthropic_api_key: Optional Anthropic API key
    ///     openai_api_key: Optional OpenAI API key
    ///     groq_api_key: Optional Groq API key
    ///     mistral_api_key: Optional Mistral API key
    ///     default_provider: Optional default provider name
    ///
    /// Returns:
    ///     AsyncLLMKitClient: A new async client instance
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
    ///     AsyncLLMKitClient: A new async client instance
    #[staticmethod]
    fn from_env() -> PyResult<Self> {
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
    ///     LLMKitError: If the request fails
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
    ///     LLMKitError: If the request fails
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
    ///     LLMKitError: If the request fails
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

    fn __repr__(&self) -> String {
        let providers = self.providers();
        format!("AsyncLLMKitClient(providers={:?})", providers)
    }
}
