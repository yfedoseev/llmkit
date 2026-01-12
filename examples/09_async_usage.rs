//! Async Usage Example
//!
//! Demonstrates asynchronous patterns with LLMKit in Rust.
//! Shows concurrent execution, rate limiting, and async error handling.
//!
//! Requirements:
//! - Set OPENROUTER_API_KEY environment variable
//!
//! Run:
//!     cargo run --example 09_async_usage

use futures::future::join_all;
use llmkit::{CompletionRequest, LLMKitClient, Message};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    println!("{}", "=".repeat(50));
    println!("Example 1: Basic Async Completion");
    println!("{}", "=".repeat(50));
    basic_async_completion().await?;

    println!("\n{}", "=".repeat(50));
    println!("Example 2: Concurrent Requests");
    println!("{}", "=".repeat(50));
    concurrent_requests().await?;

    println!("\n{}", "=".repeat(50));
    println!("Example 3: Rate-Limited Batch");
    println!("{}", "=".repeat(50));
    rate_limited_batch().await?;

    println!("\n{}", "=".repeat(50));
    println!("Example 4: Async with Timeout");
    println!("{}", "=".repeat(50));
    async_with_timeout().await?;

    Ok(())
}

async fn basic_async_completion() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_openrouter_from_env()
        .build()
        .await?;

    let request = CompletionRequest::new(
        "openrouter/qwen/qwen3-32b",
        vec![Message::user("What is the capital of Japan?")],
    )
    .with_max_tokens(100);

    println!("Making async request...");
    let response = client.complete(request).await?;
    println!("Response: {}", response.text_content());

    Ok(())
}

async fn concurrent_requests() -> llmkit::Result<()> {
    let client = Arc::new(
        LLMKitClient::builder()
            .with_openrouter_from_env()
            .build()
            .await?,
    );

    let questions = ["What is Python?", "What is Rust?", "What is JavaScript?"];

    println!("Making concurrent requests...");

    let futures: Vec<_> = questions
        .iter()
        .map(|q| {
            let client = Arc::clone(&client);
            let question = q.to_string();
            async move {
                let request = CompletionRequest::new(
                    "openrouter/qwen/qwen3-32b",
                    vec![Message::user(&question)],
                )
                .with_max_tokens(100);
                (question, client.complete(request).await)
            }
        })
        .collect();

    let results = join_all(futures).await;

    for (question, result) in results {
        println!("\nQ: {}", question);
        match result {
            Ok(response) => {
                let text = response.text_content();
                let truncated: String = text.chars().take(100).collect();
                println!("A: {}...", truncated);
            }
            Err(e) => println!("A: Error - {}", e),
        }
    }

    Ok(())
}

async fn rate_limited_batch() -> llmkit::Result<()> {
    let client = Arc::new(
        LLMKitClient::builder()
            .with_openrouter_from_env()
            .build()
            .await?,
    );

    let questions = [
        "What is 1+1?",
        "What is 2+2?",
        "What is 3+3?",
        "What is 4+4?",
        "What is 5+5?",
    ];

    // Semaphore to limit concurrent requests
    let semaphore = Arc::new(Semaphore::new(2));

    println!("Processing with rate limiting (max 2 concurrent)...");

    let futures: Vec<_> = questions
        .iter()
        .map(|q| {
            let client = Arc::clone(&client);
            let semaphore = Arc::clone(&semaphore);
            let question = q.to_string();
            async move {
                let _permit = semaphore.acquire().await.unwrap();
                println!("Processing: {}", question);
                let request = CompletionRequest::new(
                    "openrouter/qwen/qwen3-32b",
                    vec![Message::user(&question)],
                )
                .with_max_tokens(20);
                let result = client.complete(request).await;
                (question, result)
            }
        })
        .collect();

    let results = join_all(futures).await;

    println!("\nResults:");
    for (question, result) in results {
        match result {
            Ok(response) => {
                println!("  {} -> {}", question, response.text_content().trim());
            }
            Err(e) => println!("  {} -> Error: {}", question, e),
        }
    }

    Ok(())
}

async fn async_with_timeout() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_openrouter_from_env()
        .build()
        .await?;

    let request = CompletionRequest::new(
        "openrouter/qwen/qwen3-32b",
        vec![Message::user("Say hello briefly")],
    )
    .with_max_tokens(50);

    // Wrap the request with a timeout
    match tokio::time::timeout(Duration::from_secs(30), client.complete(request)).await {
        Ok(Ok(response)) => {
            println!("Got response: {}", response.text_content());
        }
        Ok(Err(e)) => {
            println!("Request error: {}", e);
        }
        Err(_) => {
            println!("Request timed out!");
        }
    }

    Ok(())
}
