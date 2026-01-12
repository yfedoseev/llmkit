//! Extended Thinking (Reasoning Mode) Example
//!
//! Demonstrates using reasoning models for complex tasks.
//! QwQ-32B is a reasoning model that shows its thinking process.
//!
//! Requirements:
//! - Set OPENROUTER_API_KEY environment variable
//!
//! Run:
//!     cargo run --example 06_extended_thinking

use llmkit::{CompletionRequest, LLMKitClient, Message};

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_openrouter_from_env()
        .build()
        .await?;

    // Using QwQ-32B - a reasoning model that shows its thinking process
    // The model outputs its reasoning steps as part of the response
    let request = CompletionRequest::new(
        "openrouter/qwen/qwq-32b",
        vec![Message::user(
            "Solve this step by step: \
            A train travels from City A to City B at 60 mph. \
            Another train leaves City B towards City A at 40 mph at the same time. \
            The cities are 200 miles apart. \
            Where do they meet and after how long?",
        )],
    )
    .with_system("Think through the problem carefully and show your reasoning.")
    .with_max_tokens(2000);

    println!("Solving with QwQ-32B reasoning model...");
    println!("(This may take a moment)\n");

    let response = client.complete(request).await?;

    // QwQ-32B outputs its reasoning as part of the response
    println!("{}", "=".repeat(50));
    println!("RESPONSE (with reasoning):");
    println!("{}", "=".repeat(50));
    println!("{}", response.text_content());

    // Usage info
    println!("\n{}", "=".repeat(50));
    println!("Tokens used: {}", response.usage.total_tokens());

    Ok(())
}
