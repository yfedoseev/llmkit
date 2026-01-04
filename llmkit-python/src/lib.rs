//! LLMKit Python Bindings
//!
//! Python bindings for the LLMKit unified LLM API client.
//! Provides both synchronous (`LLMKitClient`) and asynchronous (`AsyncLLMKitClient`) clients.

use pyo3::prelude::*;

mod async_client;
mod audio;
mod client;
mod embedding;
mod errors;
mod models;
mod tools;
mod types;

use async_client::PyAsyncLLMKitClient;
use audio::*;
use client::PyLLMKitClient;
use embedding::*;
use errors::*;
use models::*;
use tools::*;
use types::enums::*;
use types::message::*;
use types::request::*;
use types::response::*;
use types::stream::*;

/// LLMKit: Unified LLM API client for Python
///
/// Provides access to 30+ LLM providers through a single interface.
///
/// Example:
/// ```python
/// from llmkit import LLMKitClient, CompletionRequest, Message
///
/// client = LLMKitClient.from_env()
/// response = client.complete(CompletionRequest(
///     model="claude-sonnet-4-20250514",
///     messages=[Message.user("Hello!")],
/// ))
/// print(response.text_content())
/// ```
#[pymodule]
fn _llmkit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Enums
    m.add_class::<PyRole>()?;
    m.add_class::<PyStopReason>()?;
    m.add_class::<PyStreamEventType>()?;
    m.add_class::<PyCacheControl>()?;
    m.add_class::<PyThinkingType>()?;
    m.add_class::<PyBatchStatus>()?;

    // Message types
    m.add_class::<PyContentBlock>()?;
    m.add_class::<PyMessage>()?;
    m.add_class::<PyCacheBreakpoint>()?;
    m.add_class::<PyThinkingConfig>()?;
    m.add_class::<PyStructuredOutput>()?;

    // Request/Response types
    m.add_class::<PyCompletionRequest>()?;
    m.add_class::<PyCompletionResponse>()?;
    m.add_class::<PyUsage>()?;
    m.add_class::<PyTokenCountRequest>()?;
    m.add_class::<PyTokenCountResult>()?;

    // Batch processing types
    m.add_class::<PyBatchRequest>()?;
    m.add_class::<PyBatchJob>()?;
    m.add_class::<PyBatchRequestCounts>()?;
    m.add_class::<PyBatchResult>()?;
    m.add_class::<PyBatchError>()?;

    // Streaming types
    m.add_class::<PyStreamChunk>()?;
    m.add_class::<PyContentDelta>()?;
    m.add_class::<PyStreamIterator>()?;
    m.add_class::<PyAsyncStreamIterator>()?;

    // Tools
    m.add_class::<PyToolDefinition>()?;
    m.add_class::<PyToolBuilder>()?;

    // Clients
    m.add_class::<PyLLMKitClient>()?;
    m.add_class::<PyAsyncLLMKitClient>()?;

    // Model Registry types
    m.add_class::<PyProvider>()?;
    m.add_class::<PyModelStatus>()?;
    m.add_class::<PyModelPricing>()?;
    m.add_class::<PyModelCapabilities>()?;
    m.add_class::<PyModelBenchmarks>()?;
    m.add_class::<PyRegistryStats>()?;
    m.add_class::<PyModelInfo>()?;

    // Model Registry functions
    m.add_function(wrap_pyfunction!(get_model_info, m)?)?;
    m.add_function(wrap_pyfunction!(get_all_models, m)?)?;
    m.add_function(wrap_pyfunction!(get_models_by_provider, m)?)?;
    m.add_function(wrap_pyfunction!(get_current_models, m)?)?;
    m.add_function(wrap_pyfunction!(get_classifier_models, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_models, m)?)?;
    m.add_function(wrap_pyfunction!(get_models_with_capability, m)?)?;
    m.add_function(wrap_pyfunction!(get_cheapest_model, m)?)?;
    m.add_function(wrap_pyfunction!(supports_structured_output, m)?)?;
    m.add_function(wrap_pyfunction!(get_registry_stats, m)?)?;
    m.add_function(wrap_pyfunction!(list_providers, m)?)?;

    // Embedding types
    m.add_class::<PyEncodingFormat>()?;
    m.add_class::<PyEmbeddingInputType>()?;
    m.add_class::<PyEmbeddingRequest>()?;
    m.add_class::<PyEmbedding>()?;
    m.add_class::<PyEmbeddingUsage>()?;
    m.add_class::<PyEmbeddingResponse>()?;

    // Audio types - Speech-to-Text
    m.add_class::<PyDeepgramVersion>()?;
    m.add_class::<PyTranscribeOptions>()?;
    m.add_class::<PyWord>()?;
    m.add_class::<PyTranscribeResponse>()?;
    m.add_class::<PyAudioLanguage>()?;
    m.add_class::<PyTranscriptionConfig>()?;
    m.add_class::<PyTranscriptionRequest>()?;

    // Audio types - Text-to-Speech
    m.add_class::<PyLatencyMode>()?;
    m.add_class::<PyVoiceSettings>()?;
    m.add_class::<PySynthesizeOptions>()?;
    m.add_class::<PyVoice>()?;
    m.add_class::<PySynthesizeResponse>()?;
    m.add_class::<PySynthesisRequest>()?;

    // Exceptions
    m.add("LLMKitError", m.py().get_type::<LLMKitError>())?;
    m.add(
        "ProviderNotFoundError",
        m.py().get_type::<ProviderNotFoundError>(),
    )?;
    m.add(
        "ConfigurationError",
        m.py().get_type::<ConfigurationError>(),
    )?;
    m.add(
        "AuthenticationError",
        m.py().get_type::<AuthenticationError>(),
    )?;
    m.add("RateLimitError", m.py().get_type::<RateLimitError>())?;
    m.add(
        "InvalidRequestError",
        m.py().get_type::<InvalidRequestError>(),
    )?;
    m.add(
        "ModelNotFoundError",
        m.py().get_type::<ModelNotFoundError>(),
    )?;
    m.add(
        "ContentFilteredError",
        m.py().get_type::<ContentFilteredError>(),
    )?;
    m.add(
        "ContextLengthError",
        m.py().get_type::<ContextLengthError>(),
    )?;
    m.add("NetworkError", m.py().get_type::<NetworkError>())?;
    m.add("StreamError", m.py().get_type::<StreamError>())?;
    m.add("TimeoutError", m.py().get_type::<TimeoutError>())?;
    m.add("ServerError", m.py().get_type::<ServerError>())?;
    m.add("NotSupportedError", m.py().get_type::<NotSupportedError>())?;

    Ok(())
}
