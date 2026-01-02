//! Simple Completion Example
//!
//! Demonstrates the basic usage of LLMKit for making completion requests.
//!
//! Requirements:
//! - Set ANTHROPIC_API_KEY environment variable (or another provider's key)
//!
//! Run with:
//!     cargo run --example simple_completion

use llmkit::{CompletionRequest, LLMKitClient, Message};

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    // Create client from environment variables
    // Automatically detects configured providers
    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .with_default_retry()
        .build()?;

    // Create a simple completion request
    let request = CompletionRequest::new(
        "claude-sonnet-4-20250514",
        vec![Message::user(
            "What is the capital of France? Reply in one word.",
        )],
    )
    .with_max_tokens(100);

    // Make the request
    println!("Sending request...");
    let response = client.complete(request).await?;

    // Print the response
    println!("\nResponse: {}", response.text_content());
    println!("Model: {}", response.model);
    println!("Stop reason: {:?}", response.stop_reason);

    // Print token usage
    println!("\nToken usage:");
    println!("  Input tokens: {}", response.usage.input_tokens);
    println!("  Output tokens: {}", response.usage.output_tokens);
    println!("  Total tokens: {}", response.usage.total_tokens());

    Ok(())
}
