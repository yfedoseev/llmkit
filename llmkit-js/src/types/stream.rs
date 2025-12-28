//! Streaming types for JavaScript bindings

use llmkit::types::{ContentDelta, StreamChunk, StreamEventType};
use napi_derive::napi;

use super::enums::{JsStopReason, JsStreamEventType};
use super::response::JsUsage;

/// Tool use delta information.
#[napi(object)]
pub struct JsToolUseDelta {
    pub id: Option<String>,
    pub name: Option<String>,
    pub input_json_delta: Option<String>,
}

/// Delta content for streaming responses.
#[napi]
#[derive(Clone)]
pub struct JsContentDelta {
    pub(crate) inner: ContentDelta,
}

#[napi]
impl JsContentDelta {
    /// Get text if this is a text delta.
    #[napi(getter)]
    pub fn text(&self) -> Option<String> {
        match &self.inner {
            ContentDelta::TextDelta { text } => Some(text.clone()),
            _ => None,
        }
    }

    /// Get thinking content if this is a thinking delta.
    #[napi(getter)]
    pub fn thinking(&self) -> Option<String> {
        match &self.inner {
            ContentDelta::ThinkingDelta { thinking } => Some(thinking.clone()),
            _ => None,
        }
    }

    /// True if this is a text delta.
    #[napi(getter)]
    pub fn is_text(&self) -> bool {
        matches!(self.inner, ContentDelta::TextDelta { .. })
    }

    /// True if this is a tool use delta.
    #[napi(getter)]
    pub fn is_tool_use(&self) -> bool {
        matches!(self.inner, ContentDelta::ToolUseDelta { .. })
    }

    /// True if this is a thinking delta.
    #[napi(getter)]
    pub fn is_thinking(&self) -> bool {
        matches!(self.inner, ContentDelta::ThinkingDelta { .. })
    }

    /// Get tool use delta details.
    /// Returns an object with optional id, name, and inputJsonDelta properties.
    #[napi]
    pub fn as_tool_use_delta(&self) -> Option<JsToolUseDelta> {
        match &self.inner {
            ContentDelta::ToolUseDelta {
                id,
                name,
                input_json_delta,
            } => Some(JsToolUseDelta {
                id: id.clone(),
                name: name.clone(),
                input_json_delta: input_json_delta.clone(),
            }),
            _ => None,
        }
    }
}

impl From<ContentDelta> for JsContentDelta {
    fn from(delta: ContentDelta) -> Self {
        Self { inner: delta }
    }
}

/// A chunk from a streaming response.
#[napi]
#[derive(Clone)]
pub struct JsStreamChunk {
    pub(crate) inner: StreamChunk,
}

#[napi]
impl JsStreamChunk {
    /// The type of stream event.
    #[napi(getter)]
    pub fn event_type(&self) -> JsStreamEventType {
        self.inner.event_type.into()
    }

    /// Index of the content block being updated.
    #[napi(getter)]
    pub fn index(&self) -> Option<u32> {
        self.inner.index.map(|i| i as u32)
    }

    /// The delta content (if applicable).
    #[napi(getter)]
    pub fn delta(&self) -> Option<JsContentDelta> {
        self.inner.delta.clone().map(JsContentDelta::from)
    }

    /// Convenience: Get text from delta if present.
    #[napi(getter)]
    pub fn text(&self) -> Option<String> {
        self.inner.delta.as_ref().and_then(|d| match d {
            ContentDelta::TextDelta { text } => Some(text.clone()),
            _ => None,
        })
    }

    /// Stop reason (only on message_stop).
    #[napi(getter)]
    pub fn stop_reason(&self) -> Option<JsStopReason> {
        self.inner.stop_reason.map(JsStopReason::from)
    }

    /// Usage information (may be partial or final).
    #[napi(getter)]
    pub fn usage(&self) -> Option<JsUsage> {
        self.inner.usage.map(JsUsage::from)
    }

    /// True if this is a message stop event.
    #[napi(getter)]
    pub fn is_done(&self) -> bool {
        matches!(self.inner.event_type, StreamEventType::MessageStop)
    }
}

impl From<StreamChunk> for JsStreamChunk {
    fn from(chunk: StreamChunk) -> Self {
        Self { inner: chunk }
    }
}
