//! Message and ContentBlock types for JavaScript bindings

use modelsuite::types::{
    CacheBreakpoint, ContentBlock, Message, StructuredOutput, StructuredOutputType, ThinkingConfig,
};
use napi_derive::napi;

use super::enums::{JsCacheControl, JsRole, JsThinkingType};

/// A block of content within a message.
///
/// Use the static factory methods to create instances:
/// - `ContentBlock.text("Hello")`
/// - `ContentBlock.image("image/png", base64Data)`
/// - `ContentBlock.toolUse(id, name, inputObj)`
#[napi]
#[derive(Clone)]
pub struct JsContentBlock {
    pub(crate) inner: ContentBlock,
}

#[napi]
impl JsContentBlock {
    // ==================== Factory Methods ====================

    /// Create a text content block.
    #[napi(factory)]
    pub fn text(text: String) -> Self {
        Self {
            inner: ContentBlock::text(text),
        }
    }

    /// Create an image content block from base64 data.
    #[napi(factory)]
    pub fn image(media_type: String, data: String) -> Self {
        Self {
            inner: ContentBlock::image(media_type, data),
        }
    }

    /// Create an image content block from URL.
    #[napi(factory)]
    pub fn image_url(url: String) -> Self {
        Self {
            inner: ContentBlock::image_url(url),
        }
    }

    /// Create a tool use content block.
    #[napi(factory)]
    pub fn tool_use(id: String, name: String, input: serde_json::Value) -> Self {
        Self {
            inner: ContentBlock::tool_use(id, name, input),
        }
    }

    /// Create a tool result content block.
    #[napi(factory)]
    pub fn tool_result(tool_use_id: String, content: String, is_error: Option<bool>) -> Self {
        Self {
            inner: ContentBlock::tool_result(tool_use_id, content, is_error.unwrap_or(false)),
        }
    }

    /// Create a thinking content block.
    #[napi(factory)]
    pub fn thinking(thinking: String) -> Self {
        Self {
            inner: ContentBlock::thinking(thinking),
        }
    }

    /// Create a PDF document content block.
    #[napi(factory)]
    pub fn pdf(data: String) -> Self {
        Self {
            inner: ContentBlock::pdf(data),
        }
    }

    /// Create a text content block with ephemeral caching.
    #[napi(factory)]
    pub fn text_cached(text: String) -> Self {
        Self {
            inner: ContentBlock::text_cached_ephemeral(text),
        }
    }

    // ==================== Type Checking Properties ====================

    /// True if this is a text block.
    #[napi(getter)]
    pub fn is_text(&self) -> bool {
        self.inner.is_text()
    }

    /// True if this is a tool use block.
    #[napi(getter)]
    pub fn is_tool_use(&self) -> bool {
        self.inner.is_tool_use()
    }

    /// True if this is a tool result block.
    #[napi(getter)]
    pub fn is_tool_result(&self) -> bool {
        self.inner.is_tool_result()
    }

    /// True if this is a document block.
    #[napi(getter)]
    pub fn is_document(&self) -> bool {
        self.inner.is_document()
    }

    /// True if this is a thinking block.
    #[napi(getter)]
    pub fn is_thinking(&self) -> bool {
        matches!(self.inner, ContentBlock::Thinking { .. })
    }

    /// True if this is an image block.
    #[napi(getter)]
    pub fn is_image(&self) -> bool {
        matches!(
            self.inner,
            ContentBlock::Image { .. } | ContentBlock::ImageUrl { .. }
        )
    }

    // ==================== Data Access Properties ====================

    /// Get text content if this is a text block.
    #[napi(getter)]
    pub fn text_value(&self) -> Option<String> {
        self.inner.as_text().map(|s| s.to_string())
    }

    /// Get thinking content if this is a thinking block.
    #[napi(getter)]
    pub fn thinking_content(&self) -> Option<String> {
        match &self.inner {
            ContentBlock::Thinking { thinking } => Some(thinking.clone()),
            _ => None,
        }
    }

    /// Get tool use details if this is a tool use block.
    /// Returns an object with id, name, and input properties.
    #[napi]
    pub fn as_tool_use(&self) -> Option<JsToolUseInfo> {
        match &self.inner {
            ContentBlock::ToolUse { id, name, input } => Some(JsToolUseInfo {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            }),
            _ => None,
        }
    }

    /// Get tool result details if this is a tool result block.
    /// Returns an object with toolUseId, content, and isError properties.
    #[napi]
    pub fn as_tool_result(&self) -> Option<JsToolResultInfo> {
        match &self.inner {
            ContentBlock::ToolResult {
                tool_use_id,
                content,
                is_error,
            } => Some(JsToolResultInfo {
                tool_use_id: tool_use_id.clone(),
                content: content.clone(),
                is_error: *is_error,
            }),
            _ => None,
        }
    }
}

impl From<ContentBlock> for JsContentBlock {
    fn from(content_block: ContentBlock) -> Self {
        Self {
            inner: content_block,
        }
    }
}

impl From<JsContentBlock> for ContentBlock {
    fn from(js_content_block: JsContentBlock) -> Self {
        js_content_block.inner
    }
}

/// Tool use information returned from asToolUse().
#[napi(object)]
pub struct JsToolUseInfo {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Tool result information returned from asToolResult().
#[napi(object)]
pub struct JsToolResultInfo {
    pub tool_use_id: String,
    pub content: String,
    pub is_error: bool,
}

/// A message in a conversation.
///
/// Use the static factory methods to create instances:
/// - `Message.system("You are helpful")`
/// - `Message.user("Hello")`
/// - `Message.assistant("Hi there")`
#[napi]
#[derive(Clone)]
pub struct JsMessage {
    pub(crate) inner: Message,
}

#[napi]
impl JsMessage {
    // ==================== Factory Methods ====================

    /// Create a system message with text content.
    #[napi(factory)]
    pub fn system(text: String) -> Self {
        Self {
            inner: Message::system(text),
        }
    }

    /// Create a user message with text content.
    #[napi(factory)]
    pub fn user(text: String) -> Self {
        Self {
            inner: Message::user(text),
        }
    }

    /// Create an assistant message with text content.
    #[napi(factory)]
    pub fn assistant(text: String) -> Self {
        Self {
            inner: Message::assistant(text),
        }
    }

    /// Create a user message with multiple content blocks.
    #[napi(factory)]
    pub fn user_with_content(content: Vec<&JsContentBlock>) -> Self {
        Self {
            inner: Message::user_with_content(
                content.into_iter().map(|c| c.inner.clone()).collect(),
            ),
        }
    }

    /// Create an assistant message with multiple content blocks.
    #[napi(factory)]
    pub fn assistant_with_content(content: Vec<&JsContentBlock>) -> Self {
        Self {
            inner: Message::assistant_with_content(
                content.into_iter().map(|c| c.inner.clone()).collect(),
            ),
        }
    }

    /// Create a user message with tool results.
    #[napi(factory)]
    pub fn tool_results(results: Vec<&JsContentBlock>) -> Self {
        Self {
            inner: Message::tool_results(results.into_iter().map(|c| c.inner.clone()).collect()),
        }
    }

    // ==================== Properties ====================

    /// The role of the message sender.
    #[napi(getter)]
    pub fn role(&self) -> JsRole {
        self.inner.role.into()
    }

    /// The content blocks in this message.
    #[napi(getter)]
    pub fn content(&self) -> Vec<JsContentBlock> {
        self.inner
            .content
            .iter()
            .cloned()
            .map(JsContentBlock::from)
            .collect()
    }

    // ==================== Methods ====================

    /// Get all text content from the message concatenated.
    #[napi]
    pub fn text_content(&self) -> String {
        self.inner.text_content()
    }

    /// Check if the message contains any tool use blocks.
    #[napi]
    pub fn has_tool_use(&self) -> bool {
        self.inner.has_tool_use()
    }

    /// Extract all tool use blocks from the message.
    #[napi]
    pub fn tool_uses(&self) -> Vec<JsContentBlock> {
        self.inner
            .tool_uses()
            .into_iter()
            .cloned()
            .map(JsContentBlock::from)
            .collect()
    }
}

impl From<Message> for JsMessage {
    fn from(message: Message) -> Self {
        Self { inner: message }
    }
}

impl From<JsMessage> for Message {
    fn from(js_message: JsMessage) -> Self {
        js_message.inner
    }
}

/// Cache breakpoint configuration for prompt caching.
#[napi]
#[derive(Clone)]
pub struct JsCacheBreakpoint {
    pub(crate) inner: CacheBreakpoint,
}

#[napi]
impl JsCacheBreakpoint {
    /// Create an ephemeral cache breakpoint (5-minute TTL).
    #[napi(factory)]
    pub fn ephemeral() -> Self {
        Self {
            inner: CacheBreakpoint::ephemeral(),
        }
    }

    /// Create an extended cache breakpoint (1-hour TTL, Anthropic beta).
    #[napi(factory)]
    pub fn extended() -> Self {
        Self {
            inner: CacheBreakpoint::extended(),
        }
    }

    /// The cache control type.
    #[napi(getter)]
    pub fn cache_control(&self) -> JsCacheControl {
        self.inner.cache_control.clone().into()
    }
}

impl From<CacheBreakpoint> for JsCacheBreakpoint {
    fn from(cache_breakpoint: CacheBreakpoint) -> Self {
        Self {
            inner: cache_breakpoint,
        }
    }
}

impl From<JsCacheBreakpoint> for CacheBreakpoint {
    fn from(js_cache_breakpoint: JsCacheBreakpoint) -> Self {
        js_cache_breakpoint.inner
    }
}

/// Configuration for extended thinking mode.
#[napi]
#[derive(Clone)]
pub struct JsThinkingConfig {
    pub(crate) inner: ThinkingConfig,
}

#[napi]
impl JsThinkingConfig {
    /// Enable extended thinking with a token budget.
    #[napi(factory)]
    pub fn enabled(budget_tokens: u32) -> Self {
        Self {
            inner: ThinkingConfig::enabled(budget_tokens),
        }
    }

    /// Disable extended thinking.
    #[napi(factory)]
    pub fn disabled() -> Self {
        Self {
            inner: ThinkingConfig::disabled(),
        }
    }

    /// The thinking type (enabled or disabled).
    #[napi(getter)]
    pub fn thinking_type(&self) -> JsThinkingType {
        self.inner.thinking_type.into()
    }

    /// The token budget for thinking (if enabled).
    #[napi(getter)]
    pub fn budget_tokens(&self) -> Option<u32> {
        self.inner.budget_tokens
    }

    /// Check if thinking is enabled.
    #[napi(getter)]
    pub fn is_enabled(&self) -> bool {
        self.inner.is_enabled()
    }
}

impl From<ThinkingConfig> for JsThinkingConfig {
    fn from(thinking_config: ThinkingConfig) -> Self {
        Self {
            inner: thinking_config,
        }
    }
}

impl From<JsThinkingConfig> for ThinkingConfig {
    fn from(js_thinking_config: JsThinkingConfig) -> Self {
        js_thinking_config.inner
    }
}

/// Configuration for structured output format.
#[napi]
#[derive(Clone)]
pub struct JsStructuredOutput {
    pub(crate) inner: StructuredOutput,
}

#[napi]
impl JsStructuredOutput {
    /// Create a JSON schema structured output.
    #[napi(factory)]
    pub fn json_schema(name: String, schema: serde_json::Value) -> Self {
        Self {
            inner: StructuredOutput::json_schema(name, schema),
        }
    }

    /// Create a JSON object structured output (no specific schema).
    #[napi(factory)]
    pub fn json_object() -> Self {
        Self {
            inner: StructuredOutput::json_object(),
        }
    }

    /// Create a text structured output.
    #[napi(factory)]
    pub fn text() -> Self {
        Self {
            inner: StructuredOutput::text(),
        }
    }

    /// Check if this is a JSON schema output.
    #[napi(getter)]
    pub fn is_json_schema(&self) -> bool {
        matches!(self.inner.format_type, StructuredOutputType::JsonSchema)
    }

    /// Check if this is a JSON object output.
    #[napi(getter)]
    pub fn is_json_object(&self) -> bool {
        matches!(self.inner.format_type, StructuredOutputType::JsonObject)
    }

    /// Check if this is a text output.
    #[napi(getter)]
    pub fn is_text_output(&self) -> bool {
        matches!(self.inner.format_type, StructuredOutputType::Text)
    }
}

impl From<StructuredOutput> for JsStructuredOutput {
    fn from(structured_output: StructuredOutput) -> Self {
        Self {
            inner: structured_output,
        }
    }
}

impl From<JsStructuredOutput> for StructuredOutput {
    fn from(js_structured_output: JsStructuredOutput) -> Self {
        js_structured_output.inner
    }
}
