//! CompletionRequest and TokenCountRequest types for Python bindings

use llmkit::types::{
    BatchError, BatchJob, BatchRequest, BatchRequestCounts, BatchResult, BatchStatus,
    CompletionRequest, TokenCountRequest, TokenCountResult,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::enums::{PyBatchStatus, PyThinkingEffort};
use super::message::{PyMessage, PyStructuredOutput, PyThinkingConfig};
use super::response::PyCompletionResponse;
use crate::tools::PyToolDefinition;

/// Request to complete a conversation.
///
/// Can be constructed with all options in the constructor, or using builder methods
/// for method chaining.
///
/// Example:
/// ```python
/// # Constructor with options
/// request = CompletionRequest(
///     model="claude-sonnet-4-20250514",
///     messages=[Message.user("Hello")],
///     system="You are helpful",
///     max_tokens=1024,
/// )
///
/// # Builder pattern
/// request = (CompletionRequest("claude-sonnet-4-20250514", [Message.user("Hello")])
///     .with_system("You are helpful")
///     .with_max_tokens(1024))
/// ```
#[pyclass(name = "CompletionRequest")]
#[derive(Clone)]
pub struct PyCompletionRequest {
    pub(crate) inner: CompletionRequest,
}

#[pymethods]
impl PyCompletionRequest {
    /// Create a new completion request.
    ///
    /// Args:
    ///     model: Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o")
    ///     messages: Conversation messages
    ///     system: Optional system prompt
    ///     max_tokens: Optional maximum tokens to generate
    ///     temperature: Optional sampling temperature (0.0 to 2.0)
    ///     top_p: Optional nucleus sampling parameter
    ///     tools: Optional list of tool definitions
    ///     stop_sequences: Optional list of stop sequences
    ///     stream: Whether to stream the response (default: False)
    ///     thinking_budget: Optional token budget for extended thinking
    #[new]
    #[pyo3(signature = (
        model,
        messages,
        system = None,
        max_tokens = None,
        temperature = None,
        top_p = None,
        tools = None,
        stop_sequences = None,
        stream = false,
        thinking_budget = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: String,
        messages: Vec<PyMessage>,
        system: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        tools: Option<Vec<PyToolDefinition>>,
        stop_sequences: Option<Vec<String>>,
        stream: bool,
        thinking_budget: Option<u32>,
    ) -> PyResult<Self> {
        let mut request =
            CompletionRequest::new(model, messages.into_iter().map(|m| m.inner).collect());

        if let Some(s) = system {
            request = request.with_system(s);
        }
        if let Some(t) = max_tokens {
            request = request.with_max_tokens(t);
        }
        if let Some(t) = temperature {
            request = request.with_temperature(t);
        }
        if let Some(p) = top_p {
            request = request.with_top_p(p);
        }
        if let Some(t) = tools {
            request = request.with_tools(t.into_iter().map(|t| t.inner).collect());
        }
        if let Some(s) = stop_sequences {
            request = request.with_stop_sequences(s);
        }
        if stream {
            request = request.with_streaming();
        }
        if let Some(budget) = thinking_budget {
            request = request.with_thinking(budget);
        }

        Ok(Self { inner: request })
    }

    // ==================== Builder Methods ====================
    // All builder methods return a new PyCompletionRequest for method chaining

    /// Builder method: Set the system prompt.
    ///
    /// Args:
    ///     system: System prompt text
    ///
    /// Returns:
    ///     CompletionRequest: New request with system prompt set
    fn with_system(&self, system: String) -> Self {
        Self {
            inner: self.inner.clone().with_system(system),
        }
    }

    /// Builder method: Set max tokens.
    ///
    /// Args:
    ///     max_tokens: Maximum tokens to generate
    ///
    /// Returns:
    ///     CompletionRequest: New request with max tokens set
    fn with_max_tokens(&self, max_tokens: u32) -> Self {
        Self {
            inner: self.inner.clone().with_max_tokens(max_tokens),
        }
    }

    /// Builder method: Set temperature.
    ///
    /// Args:
    ///     temperature: Sampling temperature (0.0 to 2.0)
    ///
    /// Returns:
    ///     CompletionRequest: New request with temperature set
    fn with_temperature(&self, temperature: f32) -> Self {
        Self {
            inner: self.inner.clone().with_temperature(temperature),
        }
    }

    /// Builder method: Set top_p (nucleus sampling).
    ///
    /// Args:
    ///     top_p: Nucleus sampling parameter (0.0 to 1.0)
    ///
    /// Returns:
    ///     CompletionRequest: New request with top_p set
    fn with_top_p(&self, top_p: f32) -> Self {
        Self {
            inner: self.inner.clone().with_top_p(top_p),
        }
    }

    /// Builder method: Set tools.
    ///
    /// Args:
    ///     tools: List of tool definitions
    ///
    /// Returns:
    ///     CompletionRequest: New request with tools set
    fn with_tools(&self, tools: Vec<PyToolDefinition>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_tools(tools.into_iter().map(|t| t.inner).collect()),
        }
    }

    /// Builder method: Set stop sequences.
    ///
    /// Args:
    ///     stop_sequences: List of sequences that stop generation
    ///
    /// Returns:
    ///     CompletionRequest: New request with stop sequences set
    fn with_stop_sequences(&self, stop_sequences: Vec<String>) -> Self {
        Self {
            inner: self.inner.clone().with_stop_sequences(stop_sequences),
        }
    }

    /// Builder method: Enable streaming.
    ///
    /// Returns:
    ///     CompletionRequest: New request with streaming enabled
    fn with_streaming(&self) -> Self {
        Self {
            inner: self.inner.clone().with_streaming(),
        }
    }

    /// Builder method: Enable extended thinking with token budget.
    ///
    /// Args:
    ///     budget_tokens: Maximum tokens for thinking
    ///
    /// Returns:
    ///     CompletionRequest: New request with thinking enabled
    fn with_thinking(&self, budget_tokens: u32) -> Self {
        Self {
            inner: self.inner.clone().with_thinking(budget_tokens),
        }
    }

    /// Builder method: Set thinking configuration.
    ///
    /// Args:
    ///     config: ThinkingConfig instance
    ///
    /// Returns:
    ///     CompletionRequest: New request with thinking config set
    fn with_thinking_config(&self, config: PyThinkingConfig) -> Self {
        Self {
            inner: self.inner.clone().with_thinking_config(config.inner),
        }
    }

    /// Builder method: Disable thinking/reasoning.
    ///
    /// Useful for getting faster, cheaper responses from reasoning models
    /// like Qwen3, DeepSeek-R1, or when using OpenRouter's reasoning control.
    ///
    /// Returns:
    ///     CompletionRequest: New request with thinking disabled
    fn without_thinking(&self) -> Self {
        Self {
            inner: self.inner.clone().without_thinking(),
        }
    }

    /// Builder method: Set thinking effort level.
    ///
    /// Controls how much reasoning effort the model uses.
    /// Supported by OpenRouter and similar providers.
    ///
    /// Args:
    ///     effort: ThinkingEffort level (Low, Medium, High, Max)
    ///
    /// Returns:
    ///     CompletionRequest: New request with thinking effort set
    fn with_thinking_effort(&self, effort: PyThinkingEffort) -> Self {
        Self {
            inner: self.inner.clone().with_thinking_effort(effort.into()),
        }
    }

    /// Builder method: Set JSON schema for structured output.
    ///
    /// Args:
    ///     name: Schema name
    ///     schema: JSON schema as dictionary
    ///
    /// Returns:
    ///     CompletionRequest: New request with JSON schema set
    fn with_json_schema(&self, name: String, schema: Bound<'_, PyDict>) -> PyResult<Self> {
        let schema_value: serde_json::Value = pythonize::depythonize(&schema)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: self.inner.clone().with_json_schema(name, schema_value),
        })
    }

    /// Builder method: Set response format.
    ///
    /// Args:
    ///     format: StructuredOutput instance
    ///
    /// Returns:
    ///     CompletionRequest: New request with response format set
    fn with_response_format(&self, format: PyStructuredOutput) -> Self {
        Self {
            inner: self.inner.clone().with_response_format(format.inner),
        }
    }

    /// Builder method: Enable JSON object output.
    ///
    /// Returns:
    ///     CompletionRequest: New request with JSON output enabled
    fn with_json_output(&self) -> Self {
        Self {
            inner: self.inner.clone().with_json_output(),
        }
    }

    /// Builder method: Set predicted output for speculative decoding.
    ///
    /// Args:
    ///     predicted_content: Expected output content
    ///
    /// Returns:
    ///     CompletionRequest: New request with prediction set
    fn with_prediction(&self, predicted_content: String) -> Self {
        Self {
            inner: self.inner.clone().with_prediction(predicted_content),
        }
    }

    /// Builder method: Enable system prompt caching (ephemeral).
    ///
    /// Returns:
    ///     CompletionRequest: New request with system caching enabled
    fn with_system_caching(&self) -> Self {
        Self {
            inner: self.inner.clone().with_system_caching(),
        }
    }

    /// Builder method: Enable system prompt caching (extended, 1-hour TTL).
    ///
    /// Returns:
    ///     CompletionRequest: New request with extended system caching
    fn with_system_caching_extended(&self) -> Self {
        Self {
            inner: self.inner.clone().with_system_caching_extended(),
        }
    }

    /// Builder method: Enable extended output (128k tokens, Anthropic beta).
    ///
    /// Returns:
    ///     CompletionRequest: New request with extended output enabled
    fn with_extended_output(&self) -> Self {
        Self {
            inner: self.inner.clone().with_extended_output(),
        }
    }

    /// Builder method: Enable interleaved thinking (Anthropic beta).
    ///
    /// Returns:
    ///     CompletionRequest: New request with interleaved thinking enabled
    fn with_interleaved_thinking(&self) -> Self {
        Self {
            inner: self.inner.clone().with_interleaved_thinking(),
        }
    }

    /// Builder method: Set extra provider-specific options.
    ///
    /// Args:
    ///     extra: Dictionary of extra options
    ///
    /// Returns:
    ///     CompletionRequest: New request with extra options set
    fn with_extra(&self, extra: Bound<'_, PyDict>) -> PyResult<Self> {
        let extra_value: serde_json::Value = pythonize::depythonize(&extra)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: self.inner.clone().with_extra(extra_value),
        })
    }

    // ==================== Properties ====================

    /// The model identifier.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    /// The conversation messages.
    #[getter]
    fn messages(&self) -> Vec<PyMessage> {
        self.inner
            .messages
            .iter()
            .cloned()
            .map(PyMessage::from)
            .collect()
    }

    /// The system prompt (if set).
    #[getter]
    fn system(&self) -> Option<&str> {
        self.inner.system.as_deref()
    }

    /// Maximum tokens to generate.
    #[getter]
    fn max_tokens(&self) -> Option<u32> {
        self.inner.max_tokens
    }

    /// Sampling temperature.
    #[getter]
    fn temperature(&self) -> Option<f32> {
        self.inner.temperature
    }

    /// Whether streaming is enabled.
    #[getter]
    fn stream(&self) -> bool {
        self.inner.stream
    }

    // ==================== Helper Methods ====================

    /// Check if the request has caching enabled.
    fn has_caching(&self) -> bool {
        self.inner.has_caching()
    }

    /// Check if the request has thinking enabled.
    fn has_thinking(&self) -> bool {
        self.inner.has_thinking()
    }

    /// Check if the request has structured output.
    fn has_structured_output(&self) -> bool {
        self.inner.has_structured_output()
    }

    fn __repr__(&self) -> String {
        format!(
            "CompletionRequest(model={:?}, messages=[...{}], stream={})",
            self.inner.model,
            self.inner.messages.len(),
            self.inner.stream
        )
    }
}

impl From<CompletionRequest> for PyCompletionRequest {
    fn from(request: CompletionRequest) -> Self {
        Self { inner: request }
    }
}

impl From<PyCompletionRequest> for CompletionRequest {
    fn from(py_request: PyCompletionRequest) -> Self {
        py_request.inner
    }
}

// ============================================================================
// Token Counting Types
// ============================================================================

/// Request to count tokens for a model.
///
/// This allows estimation of token counts before making a completion request,
/// useful for cost estimation and context window management.
///
/// Example:
/// ```python
/// # Create from model and messages
/// request = TokenCountRequest(
///     model="claude-sonnet-4-20250514",
///     messages=[Message.user("Hello, how are you?")],
///     system="You are a helpful assistant",
/// )
///
/// # Count tokens
/// result = client.count_tokens(request)
/// print(f"Input tokens: {result.input_tokens}")
///
/// # Or create from existing completion request
/// request = TokenCountRequest.from_completion_request(completion_request)
/// ```
#[pyclass(name = "TokenCountRequest")]
#[derive(Clone)]
pub struct PyTokenCountRequest {
    pub(crate) inner: TokenCountRequest,
}

#[pymethods]
impl PyTokenCountRequest {
    /// Create a new token count request.
    ///
    /// Args:
    ///     model: Model identifier (e.g., "claude-sonnet-4-20250514")
    ///     messages: Conversation messages
    ///     system: Optional system prompt
    ///     tools: Optional list of tool definitions
    #[new]
    #[pyo3(signature = (model, messages, system = None, tools = None))]
    fn new(
        model: String,
        messages: Vec<PyMessage>,
        system: Option<String>,
        tools: Option<Vec<PyToolDefinition>>,
    ) -> Self {
        let mut request =
            TokenCountRequest::new(model, messages.into_iter().map(|m| m.inner).collect());

        if let Some(s) = system {
            request = request.with_system(s);
        }
        if let Some(t) = tools {
            request = request.with_tools(t.into_iter().map(|t| t.inner).collect());
        }

        Self { inner: request }
    }

    /// Create a token count request from an existing completion request.
    ///
    /// Args:
    ///     request: CompletionRequest to convert
    ///
    /// Returns:
    ///     TokenCountRequest with the same model, messages, system, and tools
    #[staticmethod]
    fn from_completion_request(request: &PyCompletionRequest) -> Self {
        Self {
            inner: TokenCountRequest::from_completion_request(&request.inner),
        }
    }

    /// Builder method: Set the system prompt.
    fn with_system(&self, system: String) -> Self {
        Self {
            inner: self.inner.clone().with_system(system),
        }
    }

    /// Builder method: Set the tools.
    fn with_tools(&self, tools: Vec<PyToolDefinition>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_tools(tools.into_iter().map(|t| t.inner).collect()),
        }
    }

    /// The model identifier.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    /// The conversation messages.
    #[getter]
    fn messages(&self) -> Vec<PyMessage> {
        self.inner
            .messages
            .iter()
            .cloned()
            .map(PyMessage::from)
            .collect()
    }

    /// The system prompt (if set).
    #[getter]
    fn system(&self) -> Option<&str> {
        self.inner.system.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "TokenCountRequest(model={:?}, messages=[...{}])",
            self.inner.model,
            self.inner.messages.len(),
        )
    }
}

impl From<TokenCountRequest> for PyTokenCountRequest {
    fn from(request: TokenCountRequest) -> Self {
        Self { inner: request }
    }
}

impl From<PyTokenCountRequest> for TokenCountRequest {
    fn from(py_request: PyTokenCountRequest) -> Self {
        py_request.inner
    }
}

/// Result of a token counting request.
///
/// Example:
/// ```python
/// result = client.count_tokens(request)
/// print(f"Input tokens: {result.input_tokens}")
/// ```
#[pyclass(name = "TokenCountResult")]
#[derive(Clone, Copy)]
pub struct PyTokenCountResult {
    pub(crate) inner: TokenCountResult,
}

#[pymethods]
impl PyTokenCountResult {
    /// Total number of input tokens.
    #[getter]
    fn input_tokens(&self) -> u32 {
        self.inner.input_tokens
    }

    fn __repr__(&self) -> String {
        format!("TokenCountResult(input_tokens={})", self.inner.input_tokens)
    }
}

impl From<TokenCountResult> for PyTokenCountResult {
    fn from(result: TokenCountResult) -> Self {
        Self { inner: result }
    }
}

// ============================================================================
// Batch Processing Types
// ============================================================================

/// A single request within a batch.
///
/// Example:
/// ```python
/// batch_req = BatchRequest(
///     custom_id="req-001",
///     request=CompletionRequest(
///         model="claude-sonnet-4-20250514",
///         messages=[Message.user("Hello")]
///     )
/// )
/// ```
#[pyclass(name = "BatchRequest")]
#[derive(Clone)]
pub struct PyBatchRequest {
    pub(crate) inner: BatchRequest,
}

#[pymethods]
impl PyBatchRequest {
    /// Create a new batch request.
    ///
    /// Args:
    ///     custom_id: Unique identifier for this request (used to match results)
    ///     request: The completion request
    #[new]
    fn new(custom_id: String, request: PyCompletionRequest) -> Self {
        Self {
            inner: BatchRequest::new(custom_id, request.inner),
        }
    }

    /// The custom ID for this request.
    #[getter]
    fn custom_id(&self) -> &str {
        &self.inner.custom_id
    }

    /// The completion request.
    #[getter]
    fn request(&self) -> PyCompletionRequest {
        PyCompletionRequest::from(self.inner.request.clone())
    }

    fn __repr__(&self) -> String {
        format!("BatchRequest(custom_id={:?})", self.inner.custom_id)
    }
}

impl From<BatchRequest> for PyBatchRequest {
    fn from(request: BatchRequest) -> Self {
        Self { inner: request }
    }
}

/// Request counts for a batch job.
#[pyclass(name = "BatchRequestCounts")]
#[derive(Clone)]
pub struct PyBatchRequestCounts {
    inner: BatchRequestCounts,
}

#[pymethods]
impl PyBatchRequestCounts {
    /// Total number of requests.
    #[getter]
    fn total(&self) -> u32 {
        self.inner.total
    }

    /// Number of succeeded requests.
    #[getter]
    fn succeeded(&self) -> u32 {
        self.inner.succeeded
    }

    /// Number of failed requests.
    #[getter]
    fn failed(&self) -> u32 {
        self.inner.failed
    }

    /// Number of pending requests.
    #[getter]
    fn pending(&self) -> u32 {
        self.inner.pending
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchRequestCounts(total={}, succeeded={}, failed={}, pending={})",
            self.inner.total, self.inner.succeeded, self.inner.failed, self.inner.pending
        )
    }
}

impl From<BatchRequestCounts> for PyBatchRequestCounts {
    fn from(counts: BatchRequestCounts) -> Self {
        Self { inner: counts }
    }
}

/// Information about a batch job.
///
/// Example:
/// ```python
/// job = client.create_batch([batch_request])
/// print(f"Batch ID: {job.id}, Status: {job.status}")
///
/// # Check status
/// if job.status.is_processing:
///     print("Batch still processing...")
/// elif job.status.is_success:
///     print("Batch completed!")
/// ```
#[pyclass(name = "BatchJob")]
#[derive(Clone)]
pub struct PyBatchJob {
    pub(crate) inner: BatchJob,
}

#[pymethods]
impl PyBatchJob {
    /// Unique batch ID.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Current status of the batch.
    #[getter]
    fn status(&self) -> PyBatchStatus {
        match self.inner.status {
            BatchStatus::Validating => PyBatchStatus::Validating,
            BatchStatus::InProgress => PyBatchStatus::InProgress,
            BatchStatus::Finalizing => PyBatchStatus::Finalizing,
            BatchStatus::Completed => PyBatchStatus::Completed,
            BatchStatus::Failed => PyBatchStatus::Failed,
            BatchStatus::Expired => PyBatchStatus::Expired,
            BatchStatus::Cancelled => PyBatchStatus::Cancelled,
        }
    }

    /// When the batch was created (ISO 8601).
    #[getter]
    fn created_at(&self) -> Option<&str> {
        self.inner.created_at.as_deref()
    }

    /// When the batch started processing (ISO 8601).
    #[getter]
    fn started_at(&self) -> Option<&str> {
        self.inner.started_at.as_deref()
    }

    /// When the batch finished (ISO 8601).
    #[getter]
    fn ended_at(&self) -> Option<&str> {
        self.inner.ended_at.as_deref()
    }

    /// When the batch expires (ISO 8601).
    #[getter]
    fn expires_at(&self) -> Option<&str> {
        self.inner.expires_at.as_deref()
    }

    /// Request counts.
    #[getter]
    fn request_counts(&self) -> PyBatchRequestCounts {
        PyBatchRequestCounts::from(self.inner.request_counts.clone())
    }

    /// Error message if the batch failed.
    #[getter]
    fn error(&self) -> Option<&str> {
        self.inner.error.as_deref()
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchJob(id={:?}, status={:?})",
            self.inner.id, self.inner.status
        )
    }
}

impl From<BatchJob> for PyBatchJob {
    fn from(job: BatchJob) -> Self {
        Self { inner: job }
    }
}

/// Error from a batch request.
#[pyclass(name = "BatchError")]
#[derive(Clone)]
pub struct PyBatchError {
    inner: BatchError,
}

#[pymethods]
impl PyBatchError {
    /// Error type.
    #[getter]
    fn error_type(&self) -> &str {
        &self.inner.error_type
    }

    /// Error message.
    #[getter]
    fn message(&self) -> &str {
        &self.inner.message
    }

    fn __repr__(&self) -> String {
        format!(
            "BatchError(type={:?}, message={:?})",
            self.inner.error_type, self.inner.message
        )
    }
}

impl From<BatchError> for PyBatchError {
    fn from(error: BatchError) -> Self {
        Self { inner: error }
    }
}

/// Result of a single request in a batch.
///
/// Example:
/// ```python
/// results = client.get_batch_results("anthropic", batch_id)
/// for result in results:
///     if result.is_success:
///         print(f"{result.custom_id}: {result.response.text_content()}")
///     else:
///         print(f"{result.custom_id} failed: {result.error.message}")
/// ```
#[pyclass(name = "BatchResult")]
#[derive(Clone)]
pub struct PyBatchResult {
    pub(crate) inner: BatchResult,
}

#[pymethods]
impl PyBatchResult {
    /// Custom ID matching the request.
    #[getter]
    fn custom_id(&self) -> &str {
        &self.inner.custom_id
    }

    /// The completion response (if successful).
    #[getter]
    fn response(&self) -> Option<PyCompletionResponse> {
        self.inner.response.clone().map(PyCompletionResponse::from)
    }

    /// Error details (if failed).
    #[getter]
    fn error(&self) -> Option<PyBatchError> {
        self.inner.error.clone().map(PyBatchError::from)
    }

    /// Check if this result was successful.
    #[getter]
    fn is_success(&self) -> bool {
        self.inner.is_success()
    }

    /// Check if this result was an error.
    #[getter]
    fn is_error(&self) -> bool {
        self.inner.is_error()
    }

    fn __repr__(&self) -> String {
        if self.inner.is_success() {
            format!(
                "BatchResult(custom_id={:?}, success=True)",
                self.inner.custom_id
            )
        } else {
            format!(
                "BatchResult(custom_id={:?}, success=False)",
                self.inner.custom_id
            )
        }
    }
}

impl From<BatchResult> for PyBatchResult {
    fn from(result: BatchResult) -> Self {
        Self { inner: result }
    }
}
