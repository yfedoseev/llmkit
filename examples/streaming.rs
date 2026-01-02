//! Streaming Example
//!
//! Demonstrates real-time streaming of completion responses.
//!
//! Requirements:
//! - Set OPENAI_API_KEY environment variable
//!
//! Run with:
//!     cargo run --example streaming --features openai

use futures::StreamExt;
use llmkit::{CompletionRequest, ContentDelta, LLMKitClient, Message};
use std::io::{self, Write};

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_openai_from_env()
        .with_default_retry()
        .build()?;

    // Create a request with streaming enabled
    // Use "provider/model" format for explicit provider routing
    let request = CompletionRequest::new(
        "openai/gpt-4o",
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
