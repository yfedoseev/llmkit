//! Error Handling Example
//!
//! Demonstrates how to handle various errors when using the LLMKit library.
//!
//! Requirements:
//! - Set OPENROUTER_API_KEY environment variable
//!
//! Run:
//!   cargo run --example 08_error_handling

use llmkit::{CompletionRequest, LLMKitClient, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Error Handling Examples\n");

    // Example 1: Handle missing API key gracefully
    println!("1. Handling missing API key:");
    match LLMKitClient::builder().build().await {
        Ok(_) => println!("   Client created (no providers configured)"),
        Err(e) => println!("   Expected: {}", e),
    }

    // Example 2: Handle invalid model name
    println!("\n2. Handling invalid model name:");
    let client = LLMKitClient::builder()
        .with_openrouter_from_env()
        .build()
        .await?;

    let request = CompletionRequest::new(
        "openrouter/invalid/nonexistent-model",
        vec![Message::user("Hello")],
    )
    .with_max_tokens(50);

    match client.complete(request).await {
        Ok(response) => println!("   Response: {}", response.text_content()),
        Err(e) => println!("   Error (expected): {}", e),
    }

    // Example 3: Handle empty messages
    println!("\n3. Handling empty messages:");
    let request = CompletionRequest::new("openrouter/qwen/qwen3-32b", vec![]).with_max_tokens(50);

    match client.complete(request).await {
        Ok(response) => println!("   Response: {}", response.text_content()),
        Err(e) => println!("   Error (expected): {}", e),
    }

    // Example 4: Successful request for comparison
    println!("\n4. Successful request:");
    let request = CompletionRequest::new(
        "openrouter/qwen/qwen3-32b",
        vec![Message::user("Say 'Hello, World!' and nothing else.")],
    )
    .with_max_tokens(50);

    match client.complete(request).await {
        Ok(response) => println!("   Success: {}", response.text_content().trim()),
        Err(e) => println!("   Error: {}", e),
    }

    // Example 5: Retry pattern with exponential backoff
    println!("\n5. Retry pattern example:");
    let max_retries = 3;
    let mut attempt = 0;

    loop {
        attempt += 1;
        let request = CompletionRequest::new(
            "openrouter/qwen/qwen3-32b",
            vec![Message::user("Say 'retry test' only.")],
        )
        .with_max_tokens(50);

        match client.complete(request).await {
            Ok(response) => {
                println!(
                    "   Attempt {}: Success - {}",
                    attempt,
                    response.text_content().trim()
                );
                break;
            }
            Err(e) => {
                println!("   Attempt {}: Error - {}", attempt, e);
                if attempt >= max_retries {
                    println!("   Max retries reached");
                    break;
                }
                let delay = std::time::Duration::from_millis(100 * 2_u64.pow(attempt as u32));
                println!("   Retrying in {:?}...", delay);
                tokio::time::sleep(delay).await;
            }
        }
    }

    println!("\nError handling examples completed!");
    Ok(())
}
