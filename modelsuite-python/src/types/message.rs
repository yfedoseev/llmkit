//! Message and ContentBlock types for Python bindings

use modelsuite::types::{
    CacheBreakpoint, CacheControl, ContentBlock, Message, Role, StructuredOutput,
    StructuredOutputType, ThinkingConfig,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::enums::{PyCacheControl, PyRole, PyThinkingType};

/// A block of content within a message.
///
/// Use the static factory methods to create instances:
/// - `ContentBlock.text("Hello")`
/// - `ContentBlock.image("image/png", base64_data)`
/// - `ContentBlock.tool_use(id, name, input_dict)`
#[pyclass(name = "ContentBlock")]
#[derive(Clone)]
pub struct PyContentBlock {
    pub(crate) inner: ContentBlock,
}

#[pymethods]
impl PyContentBlock {
    // ==================== Factory Methods ====================

    /// Create a text content block.
    ///
    /// Args:
    ///     text: The text content
    ///
    /// Returns:
    ///     ContentBlock: A text content block
    #[staticmethod]
    fn text(text: String) -> Self {
        Self {
            inner: ContentBlock::text(text),
        }
    }

    /// Create an image content block from base64 data.
    ///
    /// Args:
    ///     media_type: MIME type (e.g., "image/png", "image/jpeg")
    ///     data: Base64-encoded image data
    ///
    /// Returns:
    ///     ContentBlock: An image content block
    #[staticmethod]
    fn image(media_type: String, data: String) -> Self {
        Self {
            inner: ContentBlock::image(media_type, data),
        }
    }

    /// Create an image content block from URL.
    ///
    /// Args:
    ///     url: URL of the image
    ///
    /// Returns:
    ///     ContentBlock: An image URL content block
    #[staticmethod]
    fn image_url(url: String) -> Self {
        Self {
            inner: ContentBlock::image_url(url),
        }
    }

    /// Create a tool use content block.
    ///
    /// Args:
    ///     id: Unique identifier for this tool use
    ///     name: Name of the tool to use
    ///     input: Tool input as a dictionary
    ///
    /// Returns:
    ///     ContentBlock: A tool use content block
    #[staticmethod]
    fn tool_use(
        _py: Python<'_>,
        id: String,
        name: String,
        input: Bound<'_, PyDict>,
    ) -> PyResult<Self> {
        let input_value: serde_json::Value = pythonize::depythonize(&input)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: ContentBlock::tool_use(id, name, input_value),
        })
    }

    /// Create a tool result content block.
    ///
    /// Args:
    ///     tool_use_id: ID of the tool use this is responding to
    ///     content: Result content as string
    ///     is_error: Whether this result represents an error
    ///
    /// Returns:
    ///     ContentBlock: A tool result content block
    #[staticmethod]
    #[pyo3(signature = (tool_use_id, content, is_error = false))]
    fn tool_result(tool_use_id: String, content: String, is_error: bool) -> Self {
        Self {
            inner: ContentBlock::tool_result(tool_use_id, content, is_error),
        }
    }

    /// Create a thinking content block.
    ///
    /// Args:
    ///     thinking: The thinking/reasoning content
    ///
    /// Returns:
    ///     ContentBlock: A thinking content block
    #[staticmethod]
    fn thinking(thinking: String) -> Self {
        Self {
            inner: ContentBlock::thinking(thinking),
        }
    }

    /// Create a PDF document content block.
    ///
    /// Args:
    ///     data: Base64-encoded PDF data
    ///
    /// Returns:
    ///     ContentBlock: A PDF document content block
    #[staticmethod]
    fn pdf(data: String) -> Self {
        Self {
            inner: ContentBlock::pdf(data),
        }
    }

    /// Create a text content block with ephemeral caching.
    ///
    /// Args:
    ///     text: The text content
    ///
    /// Returns:
    ///     ContentBlock: A cached text content block
    #[staticmethod]
    fn text_cached(text: String) -> Self {
        Self {
            inner: ContentBlock::text_cached_ephemeral(text),
        }
    }

    // ==================== Type Checking Properties ====================

    /// True if this is a text block.
    #[getter]
    fn is_text(&self) -> bool {
        self.inner.is_text()
    }

    /// True if this is a tool use block.
    #[getter]
    fn is_tool_use(&self) -> bool {
        self.inner.is_tool_use()
    }

    /// True if this is a tool result block.
    #[getter]
    fn is_tool_result(&self) -> bool {
        self.inner.is_tool_result()
    }

    /// True if this is a document block.
    #[getter]
    fn is_document(&self) -> bool {
        self.inner.is_document()
    }

    /// True if this is a thinking block.
    #[getter]
    fn is_thinking(&self) -> bool {
        matches!(self.inner, ContentBlock::Thinking { .. })
    }

    /// True if this is an image block.
    #[getter]
    fn is_image(&self) -> bool {
        matches!(
            self.inner,
            ContentBlock::Image { .. } | ContentBlock::ImageUrl { .. }
        )
    }

    // ==================== Data Access Properties ====================

    /// Get text content if this is a text block.
    #[getter(text_value)]
    fn get_text(&self) -> Option<String> {
        self.inner.as_text().map(|s| s.to_string())
    }

    /// Get thinking content if this is a thinking block.
    #[getter]
    fn thinking_content(&self) -> Option<String> {
        match &self.inner {
            ContentBlock::Thinking { thinking } => Some(thinking.clone()),
            _ => None,
        }
    }

    /// Get tool use details if this is a tool use block.
    ///
    /// Returns:
    ///     Optional tuple of (id, name, input_dict) or None
    fn as_tool_use(&self, py: Python<'_>) -> PyResult<Option<(String, String, PyObject)>> {
        match &self.inner {
            ContentBlock::ToolUse { id, name, input } => {
                let py_input = pythonize::pythonize(py, input)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                Ok(Some((id.clone(), name.clone(), py_input.into())))
            }
            _ => Ok(None),
        }
    }

    /// Get tool result details if this is a tool result block.
    ///
    /// Returns:
    ///     Optional tuple of (tool_use_id, content, is_error) or None
    fn as_tool_result(&self) -> Option<(String, String, bool)> {
        match &self.inner {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => Some((tool_use_id.clone(), content.clone(), *is_error)),
            _ => None,
        }
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            ContentBlock::Text { text } => {
                let preview = if text.len() > 50 {
                    format!("{}...", &text[..50])
                } else {
                    text.clone()
                };
                format!("ContentBlock.text({:?})", preview)
            }
            ContentBlock::Image { media_type, .. } => {
                format!("ContentBlock.image({:?}, ...)", media_type)
            }
            ContentBlock::ImageUrl { url } => {
                format!("ContentBlock.image_url({:?})", url)
            }
            ContentBlock::ToolUse { id, name, .. } => {
                format!("ContentBlock.tool_use({:?}, {:?}, ...)", id, name)
            }
            ContentBlock::ToolResult {
                tool_use_id,
                is_error,
                ..
            } => {
                format!(
                    "ContentBlock.tool_result({:?}, is_error={})",
                    tool_use_id, is_error
                )
            }
            ContentBlock::Thinking { .. } => "ContentBlock.thinking(...)".to_string(),
            ContentBlock::Document { .. } => "ContentBlock.document(...)".to_string(),
            ContentBlock::TextWithCache { .. } => "ContentBlock.text_cached(...)".to_string(),
        }
    }
}

impl From<ContentBlock> for PyContentBlock {
    fn from(content_block: ContentBlock) -> Self {
        Self {
            inner: content_block,
        }
    }
}

impl From<PyContentBlock> for ContentBlock {
    fn from(py_content_block: PyContentBlock) -> Self {
        py_content_block.inner
    }
}

/// A message in a conversation.
///
/// Use the static factory methods to create instances:
/// - `Message.system("You are helpful")`
/// - `Message.user("Hello")`
/// - `Message.assistant("Hi there")`
#[pyclass(name = "Message")]
#[derive(Clone)]
pub struct PyMessage {
    pub(crate) inner: Message,
}

#[pymethods]
impl PyMessage {
    // ==================== Factory Methods ====================

    /// Create a system message with text content.
    ///
    /// Args:
    ///     text: The system prompt text
    ///
    /// Returns:
    ///     Message: A system message
    #[staticmethod]
    fn system(text: String) -> Self {
        Self {
            inner: Message::system(text),
        }
    }

    /// Create a user message with text content.
    ///
    /// Args:
    ///     text: The user message text
    ///
    /// Returns:
    ///     Message: A user message
    #[staticmethod]
    fn user(text: String) -> Self {
        Self {
            inner: Message::user(text),
        }
    }

    /// Create an assistant message with text content.
    ///
    /// Args:
    ///     text: The assistant response text
    ///
    /// Returns:
    ///     Message: An assistant message
    #[staticmethod]
    fn assistant(text: String) -> Self {
        Self {
            inner: Message::assistant(text),
        }
    }

    /// Create a user message with multiple content blocks.
    ///
    /// Args:
    ///     content: List of content blocks
    ///
    /// Returns:
    ///     Message: A user message with the given content
    #[staticmethod]
    fn user_with_content(content: Vec<PyContentBlock>) -> Self {
        Self {
            inner: Message::user_with_content(content.into_iter().map(|c| c.inner).collect()),
        }
    }

    /// Create an assistant message with multiple content blocks.
    ///
    /// Args:
    ///     content: List of content blocks
    ///
    /// Returns:
    ///     Message: An assistant message with the given content
    #[staticmethod]
    fn assistant_with_content(content: Vec<PyContentBlock>) -> Self {
        Self {
            inner: Message::assistant_with_content(content.into_iter().map(|c| c.inner).collect()),
        }
    }

    /// Create a user message with tool results.
    ///
    /// Args:
    ///     results: List of tool result content blocks
    ///
    /// Returns:
    ///     Message: A user message containing tool results
    #[staticmethod]
    fn tool_results(results: Vec<PyContentBlock>) -> Self {
        Self {
            inner: Message::tool_results(results.into_iter().map(|c| c.inner).collect()),
        }
    }

    // ==================== Properties ====================

    /// The role of the message sender.
    #[getter]
    fn role(&self) -> PyRole {
        self.inner.role.into()
    }

    /// The content blocks in this message.
    #[getter]
    fn content(&self) -> Vec<PyContentBlock> {
        self.inner
            .content
            .iter()
            .cloned()
            .map(PyContentBlock::from)
            .collect()
    }

    // ==================== Methods ====================

    /// Get all text content from the message concatenated.
    ///
    /// Returns:
    ///     str: All text content joined together
    fn text_content(&self) -> String {
        self.inner.text_content()
    }

    /// Check if the message contains any tool use blocks.
    ///
    /// Returns:
    ///     bool: True if there are tool use blocks
    fn has_tool_use(&self) -> bool {
        self.inner.has_tool_use()
    }

    /// Extract all tool use blocks from the message.
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

    fn __repr__(&self) -> String {
        let role = match self.inner.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        let content_preview = self.inner.text_content();
        let preview = if content_preview.len() > 50 {
            format!("{}...", &content_preview[..50])
        } else {
            content_preview
        };
        format!("Message.{}({:?})", role, preview)
    }
}

impl From<Message> for PyMessage {
    fn from(message: Message) -> Self {
        Self { inner: message }
    }
}

impl From<PyMessage> for Message {
    fn from(py_message: PyMessage) -> Self {
        py_message.inner
    }
}

/// Cache breakpoint configuration for prompt caching.
#[pyclass(name = "CacheBreakpoint")]
#[derive(Clone)]
pub struct PyCacheBreakpoint {
    pub(crate) inner: CacheBreakpoint,
}

#[pymethods]
impl PyCacheBreakpoint {
    /// Create an ephemeral cache breakpoint (5-minute TTL).
    #[staticmethod]
    fn ephemeral() -> Self {
        Self {
            inner: CacheBreakpoint::ephemeral(),
        }
    }

    /// Create an extended cache breakpoint (1-hour TTL, Anthropic beta).
    #[staticmethod]
    fn extended() -> Self {
        Self {
            inner: CacheBreakpoint::extended(),
        }
    }

    /// The cache control type.
    #[getter]
    fn cache_control(&self) -> PyCacheControl {
        self.inner.cache_control.clone().into()
    }

    fn __repr__(&self) -> String {
        match self.inner.cache_control {
            CacheControl::Ephemeral => "CacheBreakpoint.ephemeral()".to_string(),
            CacheControl::Extended => "CacheBreakpoint.extended()".to_string(),
        }
    }
}

impl From<CacheBreakpoint> for PyCacheBreakpoint {
    fn from(cache_breakpoint: CacheBreakpoint) -> Self {
        Self {
            inner: cache_breakpoint,
        }
    }
}

impl From<PyCacheBreakpoint> for CacheBreakpoint {
    fn from(py_cache_breakpoint: PyCacheBreakpoint) -> Self {
        py_cache_breakpoint.inner
    }
}

/// Configuration for extended thinking mode.
#[pyclass(name = "ThinkingConfig")]
#[derive(Clone)]
pub struct PyThinkingConfig {
    pub(crate) inner: ThinkingConfig,
}

#[pymethods]
impl PyThinkingConfig {
    /// Enable extended thinking with a token budget.
    ///
    /// Args:
    ///     budget_tokens: Maximum tokens for thinking
    ///
    /// Returns:
    ///     ThinkingConfig: Enabled thinking configuration
    #[staticmethod]
    fn enabled(budget_tokens: u32) -> Self {
        Self {
            inner: ThinkingConfig::enabled(budget_tokens),
        }
    }

    /// Disable extended thinking.
    ///
    /// Returns:
    ///     ThinkingConfig: Disabled thinking configuration
    #[staticmethod]
    fn disabled() -> Self {
        Self {
            inner: ThinkingConfig::disabled(),
        }
    }

    /// The thinking type (enabled or disabled).
    #[getter]
    fn thinking_type(&self) -> PyThinkingType {
        self.inner.thinking_type.into()
    }

    /// The token budget for thinking (if enabled).
    #[getter]
    fn budget_tokens(&self) -> Option<u32> {
        self.inner.budget_tokens
    }

    /// Check if thinking is enabled.
    #[getter]
    fn is_enabled(&self) -> bool {
        self.inner.is_enabled()
    }

    fn __repr__(&self) -> String {
        if self.inner.is_enabled() {
            format!(
                "ThinkingConfig.enabled({})",
                self.inner.budget_tokens.unwrap_or(0)
            )
        } else {
            "ThinkingConfig.disabled()".to_string()
        }
    }
}

impl From<ThinkingConfig> for PyThinkingConfig {
    fn from(thinking_config: ThinkingConfig) -> Self {
        Self {
            inner: thinking_config,
        }
    }
}

impl From<PyThinkingConfig> for ThinkingConfig {
    fn from(py_thinking_config: PyThinkingConfig) -> Self {
        py_thinking_config.inner
    }
}

/// Configuration for structured output format.
#[pyclass(name = "StructuredOutput")]
#[derive(Clone)]
pub struct PyStructuredOutput {
    pub(crate) inner: StructuredOutput,
}

#[pymethods]
impl PyStructuredOutput {
    /// Create a JSON schema structured output.
    ///
    /// Args:
    ///     name: Name of the schema
    ///     schema: JSON schema as a dictionary
    ///
    /// Returns:
    ///     StructuredOutput: JSON schema configuration
    #[staticmethod]
    fn json_schema(_py: Python<'_>, name: String, schema: Bound<'_, PyDict>) -> PyResult<Self> {
        let schema_value: serde_json::Value = pythonize::depythonize(&schema)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: StructuredOutput::json_schema(name, schema_value),
        })
    }

    /// Create a JSON object structured output (no specific schema).
    ///
    /// Returns:
    ///     StructuredOutput: JSON object configuration
    #[staticmethod]
    fn json_object() -> Self {
        Self {
            inner: StructuredOutput::json_object(),
        }
    }

    /// Create a text structured output.
    ///
    /// Returns:
    ///     StructuredOutput: Text configuration
    #[staticmethod]
    fn text() -> Self {
        Self {
            inner: StructuredOutput::text(),
        }
    }

    fn __repr__(&self) -> String {
        match self.inner.format_type {
            StructuredOutputType::Text => "StructuredOutput.text()".to_string(),
            StructuredOutputType::JsonObject => "StructuredOutput.json_object()".to_string(),
            StructuredOutputType::JsonSchema => {
                if let Some(ref schema) = self.inner.json_schema {
                    format!("StructuredOutput.json_schema({:?}, ...)", schema.name)
                } else {
                    "StructuredOutput.json_schema(...)".to_string()
                }
            }
        }
    }
}

impl From<StructuredOutput> for PyStructuredOutput {
    fn from(structured_output: StructuredOutput) -> Self {
        Self {
            inner: structured_output,
        }
    }
}

impl From<PyStructuredOutput> for StructuredOutput {
    fn from(py_structured_output: PyStructuredOutput) -> Self {
        py_structured_output.inner
    }
}
