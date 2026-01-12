//! Batch Processing Example
//!
//! Demonstrates processing multiple requests efficiently.
//!
//! Requirements:
//! - Set OPENROUTER_API_KEY environment variable
//!
//! Run:
//!     cargo run --example 10_batch_processing

use futures::future::join_all;
use llmkit::{CompletionRequest, CompletionResponse, LLMKitClient, Message};
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    // Wrap client in Arc for sharing across tasks
    let client = Arc::new(
        LLMKitClient::builder()
            .with_openrouter_from_env()
            .build()
            .await?,
    );

    // Items to process in batch
    let items = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Portugal?",
    ];

    println!("Processing {} items...\n", items.len());

    // Sequential processing (for comparison)
    println!("Sequential processing:");
    let start = Instant::now();

    for (i, item) in items.iter().enumerate() {
        let request =
            CompletionRequest::new("openrouter/qwen/qwen3-32b", vec![Message::user(*item)])
                .with_system("Answer in one word only.")
                .with_max_tokens(50);

        let response = client.complete(request).await?;
        println!("  {}: {}", i + 1, response.text_content().trim());
    }

    let sequential_time = start.elapsed();
    println!("Sequential time: {:?}\n", sequential_time);

    // Parallel processing using Arc
    println!("Parallel processing:");
    let start = Instant::now();

    let futures: Vec<_> = items
        .iter()
        .map(|item| {
            let client = Arc::clone(&client);
            let item = item.to_string();
            async move {
                let request =
                    CompletionRequest::new("openrouter/qwen/qwen3-32b", vec![Message::user(&item)])
                        .with_system("Answer in one word only.")
                        .with_max_tokens(50);

                client.complete(request).await
            }
        })
        .collect();

    let results: Vec<llmkit::Result<CompletionResponse>> = join_all(futures).await;

    for (i, result) in results.iter().enumerate() {
        match result {
            Ok(response) => println!("  {}: {}", i + 1, response.text_content().trim()),
            Err(e) => println!("  {}: Error - {}", i + 1, e),
        }
    }

    let parallel_time = start.elapsed();
    println!("Parallel time: {:?}", parallel_time);
    println!(
        "\nSpeedup: {:.2}x",
        sequential_time.as_secs_f64() / parallel_time.as_secs_f64()
    );

    Ok(())
}
