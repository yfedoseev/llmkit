//! Internal stream handling - separated to avoid napi macro interference.
//!
//! This module is not processed by napi macros, so it can use llmkit::Error freely.

use std::pin::Pin;

use futures::{Stream, StreamExt};
use llmkit::types::StreamChunk;

/// A stream result that can be either a chunk or an error message.
pub enum StreamResult {
    Chunk(StreamChunk),
    Error(String),
    Done,
}

/// Internal stream handler that wraps the boxed stream.
pub struct StreamHandler {
    receiver: tokio::sync::mpsc::Receiver<StreamResult>,
}

impl StreamHandler {
    /// Create a new stream handler from a boxed LLMKit stream.
    ///
    /// This spawns a background task that reads from the stream and sends
    /// results through a channel.
    pub fn new(
        stream: Pin<Box<dyn Stream<Item = Result<StreamChunk, llmkit::Error>> + Send>>,
    ) -> Self {
        let (tx, rx) = tokio::sync::mpsc::channel::<StreamResult>(32);

        // Spawn background task to read from stream
        tokio::spawn(async move {
            let mut stream = stream;
            while let Some(result) = stream.next().await {
                let msg = match result {
                    Ok(chunk) => StreamResult::Chunk(chunk),
                    Err(e) => StreamResult::Error(e.to_string()),
                };
                if tx.send(msg).await.is_err() {
                    break; // Receiver dropped
                }
            }
            // Send done signal
            let _ = tx.send(StreamResult::Done).await;
        });

        Self { receiver: rx }
    }

    /// Get the next result from the stream.
    pub async fn next(&mut self) -> Option<StreamResult> {
        self.receiver.recv().await
    }
}
