//! Streaming Example
//!
//! Demonstrates real-time streaming of completion responses.
//!
//! Requirements:
//! - Set ANTHROPIC_API_KEY environment variable (or another provider's key)
//!
//! Run with:
//!     cargo run --example streaming

use futures::StreamExt;
use llmkit::{CompletionRequest, ContentDelta, LLMKitClient, Message};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .with_default_retry()
        .build()?;

    // Create a request with streaming enabled
    let request = CompletionRequest::new(
        "claude-sonnet-4-20250514",
        vec![Message::user(
            "Write a short poem about programming. 4 lines maximum.",
        )],
    )
    .with_max_tokens(200)
    .with_streaming();

    println!("Streaming response:\n");

    // Stream the response
    let mut stream = client.complete_stream(request).await?;

    while let Some(result) = stream.next().await {
        let chunk = result?;

        // Print text chunks as they arrive
        if let Some(ContentDelta::TextDelta { text }) = &chunk.delta {
            print!("{}", text);
            io::stdout().flush().unwrap();
        }

        // Check for completion
        if chunk.stop_reason.is_some() {
            println!("\n\n[Stream complete]");

            // Print final usage if available
            if let Some(usage) = &chunk.usage {
                println!("Total tokens: {}", usage.total_tokens());
            }
            break;
        }
    }

    Ok(())
}
