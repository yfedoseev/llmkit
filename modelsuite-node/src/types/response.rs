//! CompletionResponse and Usage types for JavaScript bindings

use modelsuite::types::{CompletionResponse, Usage};
use napi_derive::napi;

use super::enums::JsStopReason;
use super::message::JsContentBlock;

/// Token usage information.
#[napi]
#[derive(Clone)]
pub struct JsUsage {
    pub(crate) inner: Usage,
}

#[napi]
impl JsUsage {
    /// Number of tokens in the prompt.
    #[napi(getter)]
    pub fn input_tokens(&self) -> u32 {
        self.inner.input_tokens
    }

    /// Number of tokens in the completion.
    #[napi(getter)]
    pub fn output_tokens(&self) -> u32 {
        self.inner.output_tokens
    }

    /// Cache creation tokens (if applicable).
    #[napi(getter)]
    pub fn cache_creation_input_tokens(&self) -> u32 {
        self.inner.cache_creation_input_tokens
    }

    /// Cache read tokens (if applicable).
    #[napi(getter)]
    pub fn cache_read_input_tokens(&self) -> u32 {
        self.inner.cache_read_input_tokens
    }

    /// Total tokens used (input + output).
    #[napi]
    pub fn total_tokens(&self) -> u32 {
        self.inner.total_tokens()
    }
}

impl From<Usage> for JsUsage {
    fn from(usage: Usage) -> Self {
        Self { inner: usage }
    }
}

impl From<JsUsage> for Usage {
    fn from(js_usage: JsUsage) -> Self {
        js_usage.inner
    }
}

/// Response from a completion request.
#[napi]
#[derive(Clone)]
pub struct JsCompletionResponse {
    pub(crate) inner: CompletionResponse,
}

#[napi]
impl JsCompletionResponse {
    /// Unique response ID.
    #[napi(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// Model that generated the response.
    #[napi(getter)]
    pub fn model(&self) -> String {
        self.inner.model.clone()
    }

    /// Content blocks in the response.
    #[napi(getter)]
    pub fn content(&self) -> Vec<JsContentBlock> {
        self.inner
            .content
            .iter()
            .cloned()
            .map(JsContentBlock::from)
            .collect()
    }

    /// Reason the model stopped.
    #[napi(getter)]
    pub fn stop_reason(&self) -> JsStopReason {
        self.inner.stop_reason.into()
    }

    /// Token usage information.
    #[napi(getter)]
    pub fn usage(&self) -> JsUsage {
        self.inner.usage.into()
    }

    /// Get all text content from the response concatenated.
    #[napi]
    pub fn text_content(&self) -> String {
        self.inner.text_content()
    }

    /// Extract all tool use blocks from the response.
    #[napi]
    pub fn tool_uses(&self) -> Vec<JsContentBlock> {
        self.inner
            .tool_uses()
            .into_iter()
            .cloned()
            .map(JsContentBlock::from)
            .collect()
    }

    /// Check if the response contains tool use.
    #[napi]
    pub fn has_tool_use(&self) -> bool {
        self.inner.has_tool_use()
    }

    /// Get thinking content from the response if present.
    #[napi]
    pub fn thinking_content(&self) -> Option<String> {
        // Iterate through content blocks to find thinking content
        for block in &self.inner.content {
            if let modelsuite::types::ContentBlock::Thinking { thinking } = block {
                return Some(thinking.clone());
            }
        }
        None
    }
}

impl From<CompletionResponse> for JsCompletionResponse {
    fn from(response: CompletionResponse) -> Self {
        Self { inner: response }
    }
}

impl From<JsCompletionResponse> for CompletionResponse {
    fn from(js_response: JsCompletionResponse) -> Self {
        js_response.inner
    }
}
