//! CompletionResponse and Usage types for Python bindings

use llmkit::types::{CompletionResponse, Usage};
use pyo3::prelude::*;

use super::enums::PyStopReason;
use super::message::PyContentBlock;

/// Token usage information.
#[pyclass(name = "Usage")]
#[derive(Clone)]
pub struct PyUsage {
    pub(crate) inner: Usage,
}

#[pymethods]
impl PyUsage {
    /// Number of tokens in the prompt.
    #[getter]
    fn input_tokens(&self) -> u32 {
        self.inner.input_tokens
    }

    /// Number of tokens in the completion.
    #[getter]
    fn output_tokens(&self) -> u32 {
        self.inner.output_tokens
    }

    /// Cache creation tokens (if applicable).
    #[getter]
    fn cache_creation_input_tokens(&self) -> u32 {
        self.inner.cache_creation_input_tokens
    }

    /// Cache read tokens (if applicable).
    #[getter]
    fn cache_read_input_tokens(&self) -> u32 {
        self.inner.cache_read_input_tokens
    }

    /// Total tokens used (input + output).
    fn total_tokens(&self) -> u32 {
        self.inner.total_tokens()
    }

    fn __repr__(&self) -> String {
        format!(
            "Usage(input_tokens={}, output_tokens={}, total={})",
            self.inner.input_tokens,
            self.inner.output_tokens,
            self.inner.total_tokens()
        )
    }
}

impl From<Usage> for PyUsage {
    fn from(usage: Usage) -> Self {
        Self { inner: usage }
    }
}

impl From<PyUsage> for Usage {
    fn from(py_usage: PyUsage) -> Self {
        py_usage.inner
    }
}

/// Response from a completion request.
#[pyclass(name = "CompletionResponse")]
#[derive(Clone)]
pub struct PyCompletionResponse {
    pub(crate) inner: CompletionResponse,
}

#[pymethods]
impl PyCompletionResponse {
    /// Unique response ID.
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }

    /// Model that generated the response.
    #[getter]
    fn model(&self) -> &str {
        &self.inner.model
    }

    /// Content blocks in the response.
    #[getter]
    fn content(&self) -> Vec<PyContentBlock> {
        self.inner
            .content
            .iter()
            .cloned()
            .map(PyContentBlock::from)
            .collect()
    }

    /// Reason the model stopped.
    #[getter]
    fn stop_reason(&self) -> PyStopReason {
        self.inner.stop_reason.into()
    }

    /// Token usage information.
    #[getter]
    fn usage(&self) -> PyUsage {
        self.inner.usage.into()
    }

    /// Get all text content from the response concatenated.
    ///
    /// Returns:
    ///     str: All text content joined together
    fn text_content(&self) -> String {
        self.inner.text_content()
    }

    /// Extract all tool use blocks from the response.
    ///
    /// Returns:
    ///     List[ContentBlock]: List of tool use content blocks
    fn tool_uses(&self) -> Vec<PyContentBlock> {
        self.inner
            .tool_uses()
            .into_iter()
            .cloned()
            .map(PyContentBlock::from)
            .collect()
    }

    /// Check if the response contains tool use.
    ///
    /// Returns:
    ///     bool: True if there are tool use blocks
    fn has_tool_use(&self) -> bool {
        self.inner.has_tool_use()
    }

    fn __repr__(&self) -> String {
        let content_preview = self.inner.text_content();
        let preview = if content_preview.len() > 50 {
            format!("{}...", &content_preview[..50])
        } else {
            content_preview
        };
        format!(
            "CompletionResponse(id={:?}, model={:?}, content={:?}, stop_reason={:?})",
            self.inner.id, self.inner.model, preview, self.inner.stop_reason
        )
    }
}

impl From<CompletionResponse> for PyCompletionResponse {
    fn from(response: CompletionResponse) -> Self {
        Self { inner: response }
    }
}

impl From<PyCompletionResponse> for CompletionResponse {
    fn from(py_response: PyCompletionResponse) -> Self {
        py_response.inner
    }
}
