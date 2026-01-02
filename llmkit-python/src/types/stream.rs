//! Streaming types for Python bindings

use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use llmkit::types::{ContentDelta, StreamChunk};
use llmkit::Result;
use pyo3::prelude::*;
use tokio::sync::Mutex;

use super::enums::{PyStopReason, PyStreamEventType};
use super::response::PyUsage;

/// Delta content for streaming responses.
#[pyclass(name = "ContentDelta")]
#[derive(Clone)]
pub struct PyContentDelta {
    pub(crate) inner: ContentDelta,
}

#[pymethods]
impl PyContentDelta {
    /// Get text if this is a text delta.
    #[getter]
    fn text(&self) -> Option<String> {
        match &self.inner {
            ContentDelta::TextDelta { text } => Some(text.clone()),
            _ => None,
        }
    }

    /// Get thinking content if this is a thinking delta.
    #[getter]
    fn thinking(&self) -> Option<String> {
        match &self.inner {
            ContentDelta::ThinkingDelta { thinking } => Some(thinking.clone()),
            _ => None,
        }
    }

    /// True if this is a text delta.
    #[getter]
    fn is_text(&self) -> bool {
        matches!(self.inner, ContentDelta::TextDelta { .. })
    }

    /// True if this is a tool use delta.
    #[getter]
    fn is_tool_use(&self) -> bool {
        matches!(self.inner, ContentDelta::ToolUseDelta { .. })
    }

    /// True if this is a thinking delta.
    #[getter]
    fn is_thinking(&self) -> bool {
        matches!(self.inner, ContentDelta::ThinkingDelta { .. })
    }

    /// Get tool use delta details.
    ///
    /// Returns:
    ///     Optional dict with 'id', 'name', 'input_json_delta' keys (all optional)
    fn as_tool_use_delta(&self, py: Python<'_>) -> Option<PyObject> {
        match &self.inner {
            ContentDelta::ToolUseDelta {
                id,
                name,
                input_json_delta,
            } => {
                let dict = pyo3::types::PyDict::new(py);
                if let Some(id) = id {
                    let _ = dict.set_item("id", id);
                }
                if let Some(name) = name {
                    let _ = dict.set_item("name", name);
                }
                if let Some(delta) = input_json_delta {
                    let _ = dict.set_item("input_json_delta", delta);
                }
                Some(dict.into())
            }
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ContentDelta::TextDelta { text } => {
                let preview = if text.len() > 30 {
                    format!("{}...", &text[..30])
                } else {
                    text.clone()
                };
                format!("ContentDelta.text({:?})", preview)
            }
            ContentDelta::ToolUseDelta { name, .. } => {
                format!("ContentDelta.tool_use({:?})", name)
            }
            ContentDelta::ThinkingDelta { .. } => "ContentDelta.thinking(...)".to_string(),
        }
    }
}

impl From<ContentDelta> for PyContentDelta {
    fn from(delta: ContentDelta) -> Self {
        Self { inner: delta }
    }
}

/// A chunk from a streaming response.
#[pyclass(name = "StreamChunk")]
#[derive(Clone)]
pub struct PyStreamChunk {
    pub(crate) inner: StreamChunk,
}

#[pymethods]
impl PyStreamChunk {
    /// The type of stream event.
    #[getter]
    fn event_type(&self) -> PyStreamEventType {
        self.inner.event_type.into()
    }

    /// Index of the content block being updated.
    #[getter]
    fn index(&self) -> Option<usize> {
        self.inner.index
    }

    /// The delta content (if applicable).
    #[getter]
    fn delta(&self) -> Option<PyContentDelta> {
        self.inner.delta.clone().map(PyContentDelta::from)
    }

    /// Convenience: Get text from delta if present.
    #[getter]
    fn text(&self) -> Option<String> {
        self.inner.delta.as_ref().and_then(|d| match d {
            ContentDelta::TextDelta { text } => Some(text.clone()),
            _ => None,
        })
    }

    /// Stop reason (only on message_stop).
    #[getter]
    fn stop_reason(&self) -> Option<PyStopReason> {
        self.inner.stop_reason.map(PyStopReason::from)
    }

    /// Usage information (may be partial or final).
    #[getter]
    fn usage(&self) -> Option<PyUsage> {
        self.inner.usage.map(PyUsage::from)
    }

    /// True if this is a message stop event.
    #[getter]
    fn is_done(&self) -> bool {
        matches!(
            self.inner.event_type,
            llmkit::types::StreamEventType::MessageStop
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "StreamChunk(event_type={:?}, text={:?})",
            self.inner.event_type,
            self.text()
        )
    }
}

impl From<StreamChunk> for PyStreamChunk {
    fn from(chunk: StreamChunk) -> Self {
        Self { inner: chunk }
    }
}

/// Synchronous stream iterator for blocking contexts.
#[pyclass(name = "StreamIterator")]
pub struct PyStreamIterator {
    #[allow(clippy::type_complexity)]
    stream: Arc<Mutex<Pin<Box<dyn futures::Stream<Item = Result<StreamChunk>> + Send>>>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

impl PyStreamIterator {
    pub fn new(
        stream: Pin<Box<dyn futures::Stream<Item = Result<StreamChunk>> + Send>>,
        runtime: Arc<tokio::runtime::Runtime>,
    ) -> Self {
        Self {
            stream: Arc::new(Mutex::new(stream)),
            runtime,
        }
    }
}

#[pymethods]
impl PyStreamIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyStreamChunk>> {
        let stream = self.stream.clone();

        py.allow_threads(|| {
            self.runtime.block_on(async {
                let mut guard = stream.lock().await;
                match guard.next().await {
                    Some(Ok(chunk)) => Ok(Some(PyStreamChunk::from(chunk))),
                    Some(Err(e)) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
                    None => Ok(None),
                }
            })
        })
    }
}

/// Async stream iterator for async contexts.
#[pyclass(name = "AsyncStreamIterator")]
pub struct PyAsyncStreamIterator {
    #[allow(clippy::type_complexity)]
    stream: Arc<Mutex<Pin<Box<dyn futures::Stream<Item = Result<StreamChunk>> + Send>>>>,
}

impl PyAsyncStreamIterator {
    pub fn new(stream: Pin<Box<dyn futures::Stream<Item = Result<StreamChunk>> + Send>>) -> Self {
        Self {
            stream: Arc::new(Mutex::new(stream)),
        }
    }
}

#[pymethods]
impl PyAsyncStreamIterator {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let stream = self.stream.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = stream.lock().await;
            match guard.next().await {
                Some(Ok(chunk)) => Ok(Some(PyStreamChunk::from(chunk))),
                Some(Err(e)) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
                None => Err(pyo3::exceptions::PyStopAsyncIteration::new_err(())),
            }
        })
    }
}
