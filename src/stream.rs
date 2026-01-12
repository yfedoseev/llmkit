//! Streaming utilities for handling LLM response streams.

use std::pin::Pin;
use std::task::{Context, Poll};

use futures::Stream;
use pin_project_lite::pin_project;

use crate::error::Result;
use crate::types::{
    CompletionResponse, ContentBlock, ContentDelta, StopReason, StreamChunk, StreamEventType, Usage,
};

pin_project! {
    /// A stream wrapper that collects chunks into a final response.
    pub struct CollectingStream<S> {
        #[pin]
        inner: S,
        response_id: Option<String>,
        model: Option<String>,
        content_blocks: Vec<ContentBlockBuilder>,
        current_block_index: Option<usize>,
        stop_reason: Option<StopReason>,
        usage: Usage,
    }
}

/// Builder for accumulating content block data from stream deltas.
#[derive(Debug, Clone)]
enum ContentBlockBuilder {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input_json: String,
    },
    Thinking(String),
}

impl ContentBlockBuilder {
    fn into_content_block(self) -> Result<ContentBlock> {
        match self {
            ContentBlockBuilder::Text(text) => Ok(ContentBlock::Text { text }),
            ContentBlockBuilder::ToolUse {
                id,
                name,
                input_json,
            } => {
                let input = serde_json::from_str(&input_json)
                    .unwrap_or_else(|_| serde_json::Value::Object(serde_json::Map::new()));
                Ok(ContentBlock::ToolUse { id, name, input })
            }
            ContentBlockBuilder::Thinking(thinking) => Ok(ContentBlock::Thinking { thinking }),
        }
    }
}

impl<S> CollectingStream<S>
where
    S: Stream<Item = Result<StreamChunk>>,
{
    /// Create a new collecting stream.
    pub fn new(inner: S) -> Self {
        Self {
            inner,
            response_id: None,
            model: None,
            content_blocks: Vec::new(),
            current_block_index: None,
            stop_reason: None,
            usage: Usage::default(),
        }
    }

    /// Convert accumulated state into a final response.
    pub fn into_response(self) -> Result<CompletionResponse> {
        let content: Vec<ContentBlock> = self
            .content_blocks
            .into_iter()
            .filter_map(|b| b.into_content_block().ok())
            .collect();

        Ok(CompletionResponse {
            id: self
                .response_id
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            model: self.model.unwrap_or_default(),
            content,
            stop_reason: self.stop_reason.unwrap_or(StopReason::EndTurn),
            usage: self.usage,
        })
    }
}

impl<S> Stream for CollectingStream<S>
where
    S: Stream<Item = Result<StreamChunk>>,
{
    type Item = Result<StreamChunk>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.project();

        match this.inner.poll_next(cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                // Process the chunk to accumulate state
                match chunk.event_type {
                    StreamEventType::ContentBlockStart => {
                        if let Some(index) = chunk.index {
                            *this.current_block_index = Some(index);
                            // Ensure we have enough space
                            while this.content_blocks.len() <= index {
                                this.content_blocks
                                    .push(ContentBlockBuilder::Text(String::new()));
                            }

                            // Initialize based on delta type if provided
                            if let Some(delta) = &chunk.delta {
                                match delta {
                                    ContentDelta::Text { .. } => {
                                        this.content_blocks[index] =
                                            ContentBlockBuilder::Text(String::new());
                                    }
                                    ContentDelta::ToolUse { id, name, .. } => {
                                        this.content_blocks[index] = ContentBlockBuilder::ToolUse {
                                            id: id.clone().unwrap_or_default(),
                                            name: name.clone().unwrap_or_default(),
                                            input_json: String::new(),
                                        };
                                    }
                                    ContentDelta::Thinking { .. } => {
                                        this.content_blocks[index] =
                                            ContentBlockBuilder::Thinking(String::new());
                                    }
                                }
                            }
                        }
                    }
                    StreamEventType::ContentBlockDelta => {
                        if let (Some(index), Some(delta)) =
                            (chunk.index.or(*this.current_block_index), &chunk.delta)
                        {
                            if index < this.content_blocks.len() {
                                match delta {
                                    ContentDelta::Text { text } => {
                                        if let ContentBlockBuilder::Text(ref mut s) =
                                            this.content_blocks[index]
                                        {
                                            s.push_str(text);
                                        }
                                    }
                                    ContentDelta::ToolUse {
                                        id,
                                        name,
                                        input_json_delta,
                                    } => {
                                        if let ContentBlockBuilder::ToolUse {
                                            id: ref mut block_id,
                                            name: ref mut block_name,
                                            input_json: ref mut json,
                                        } = this.content_blocks[index]
                                        {
                                            if let Some(new_id) = id {
                                                *block_id = new_id.clone();
                                            }
                                            if let Some(new_name) = name {
                                                *block_name = new_name.clone();
                                            }
                                            if let Some(delta_json) = input_json_delta {
                                                json.push_str(delta_json);
                                            }
                                        }
                                    }
                                    ContentDelta::Thinking { thinking } => {
                                        if let ContentBlockBuilder::Thinking(ref mut s) =
                                            this.content_blocks[index]
                                        {
                                            s.push_str(thinking);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    StreamEventType::ContentBlockStop => {
                        *this.current_block_index = None;
                    }
                    StreamEventType::MessageDelta | StreamEventType::MessageStop => {
                        if let Some(stop) = chunk.stop_reason {
                            *this.stop_reason = Some(stop);
                        }
                    }
                    _ => {}
                }

                if let Some(usage) = chunk.usage {
                    *this.usage = usage;
                }

                Poll::Ready(Some(Ok(chunk)))
            }
            Poll::Ready(Some(Err(e))) => Poll::Ready(Some(Err(e))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Collect a stream into a final response.
pub async fn collect_stream<S>(stream: S) -> Result<CompletionResponse>
where
    S: Stream<Item = Result<StreamChunk>> + Unpin,
{
    use futures::StreamExt;

    let mut collecting = CollectingStream::new(stream);

    // Consume all chunks
    while let Some(result) = collecting.next().await {
        result?; // Propagate errors
    }

    collecting.into_response()
}

/// Helper to create a simple text stream chunk.
pub fn text_chunk(text: impl Into<String>, index: usize) -> StreamChunk {
    StreamChunk {
        event_type: StreamEventType::ContentBlockDelta,
        index: Some(index),
        delta: Some(ContentDelta::Text { text: text.into() }),
        stop_reason: None,
        usage: None,
    }
}

/// Helper to create a message stop chunk.
pub fn stop_chunk(stop_reason: StopReason, usage: Usage) -> StreamChunk {
    StreamChunk {
        event_type: StreamEventType::MessageStop,
        index: None,
        delta: None,
        stop_reason: Some(stop_reason),
        usage: Some(usage),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    #[tokio::test]
    async fn test_collect_text_stream() {
        let chunks = vec![
            Ok(StreamChunk {
                event_type: StreamEventType::ContentBlockStart,
                index: Some(0),
                delta: Some(ContentDelta::Text {
                    text: String::new(),
                }),
                stop_reason: None,
                usage: None,
            }),
            Ok(text_chunk("Hello", 0)),
            Ok(text_chunk(" world", 0)),
            Ok(stop_chunk(
                StopReason::EndTurn,
                Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    ..Default::default()
                },
            )),
        ];

        let stream = stream::iter(chunks);
        let response = collect_stream(stream).await.unwrap();

        assert_eq!(response.text_content(), "Hello world");
        assert_eq!(response.stop_reason, StopReason::EndTurn);
        assert_eq!(response.usage.input_tokens, 10);
        assert_eq!(response.usage.output_tokens, 5);
    }
}
