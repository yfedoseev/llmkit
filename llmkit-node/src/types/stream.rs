//! Streaming types for JavaScript bindings

use std::sync::Arc;

use llmkit::types::{ContentDelta, StreamChunk, StreamEventType};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokio::sync::Mutex;

use super::enums::{JsStopReason, JsStreamEventType};
use super::response::JsUsage;
use crate::stream_internal::{StreamHandler, StreamResult};

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

/// Async iterator for streaming completion responses.
///
/// Use with manual iteration by calling `next()`:
///
/// @example
/// ```typescript
/// const stream = await client.stream(request);
/// let chunk;
/// while ((chunk = await stream.next()) !== null) {
///   if (chunk.text) {
///     process.stdout.write(chunk.text);
///   }
///   if (chunk.isDone) break;
/// }
/// ```
///
/// Or use the callback-based `completeStream` for simpler consumption.
#[napi]
pub struct JsAsyncStreamIterator {
    handler: Arc<Mutex<StreamHandler>>,
    done: Arc<Mutex<bool>>,
}

impl JsAsyncStreamIterator {
    /// Create a new async stream iterator from a StreamHandler.
    pub fn from_handler(handler: StreamHandler) -> Self {
        Self {
            handler: Arc::new(Mutex::new(handler)),
            done: Arc::new(Mutex::new(false)),
        }
    }
}

#[napi]
impl JsAsyncStreamIterator {
    /// Get the next chunk from the stream.
    ///
    /// Returns the next StreamChunk, or null when the stream is complete.
    /// Check `chunk.isDone` to determine when streaming is complete.
    #[napi]
    pub async fn next(&self) -> Result<Option<JsStreamChunk>> {
        // Check if already done
        {
            let done = self.done.lock().await;
            if *done {
                return Ok(None);
            }
        }

        // Get next chunk from handler
        let result = {
            let mut handler = self.handler.lock().await;
            handler.next().await
        };

        match result {
            Some(StreamResult::Chunk(chunk)) => {
                let js_chunk = JsStreamChunk::from(chunk);
                let is_done = js_chunk.is_done();

                if is_done {
                    let mut done = self.done.lock().await;
                    *done = true;
                }

                Ok(Some(js_chunk))
            }
            Some(StreamResult::Error(e)) => {
                let mut done = self.done.lock().await;
                *done = true;
                Err(Error::from_reason(e))
            }
            Some(StreamResult::Done) | None => {
                let mut done = self.done.lock().await;
                *done = true;
                Ok(None)
            }
        }
    }

    /// Check if the stream is done.
    #[napi(getter)]
    pub async fn is_finished(&self) -> bool {
        let done = self.done.lock().await;
        *done
    }
}
