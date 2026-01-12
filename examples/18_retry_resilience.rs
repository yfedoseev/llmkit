//! Retry & Resilience Example
//!
//! Demonstrates configuring automatic retry behavior for handling
//! transient failures like rate limits, timeouts, and server errors.
//!
//! Requirements:
//! - Set OPENROUTER_API_KEY environment variable
//!
//! Run:
//!     cargo run --example 18_retry_resilience

use llmkit::{CompletionRequest, LLMKitClient, Message, RetryConfig};
use std::time::Duration;

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    println!("Retry & Resilience Configuration Example");
    println!("{}", "=".repeat(50));

    // ========================================
    // Example 1: Default Retry (Recommended)
    // ========================================
    println!("\n1. Default Retry Configuration");
    println!("{}", "-".repeat(40));
    println!("Settings: 10 retries, 1s-5min exponential backoff, jitter enabled");

    let client = LLMKitClient::builder()
        .with_openrouter_from_env()
        .with_default_retry() // Enables default retry for all providers
        .build()
        .await?;

    let request = CompletionRequest::new(
        "openrouter/qwen/qwen3-32b",
        vec![Message::user("Say 'retry test passed' in exactly 3 words.")],
    )
    .without_thinking()
    .with_max_tokens(50);

    let response = client.complete(request).await?;
    println!("Response: {}", response.text_content().trim());

    // ========================================
    // Example 2: Production Retry Config
    // ========================================
    println!("\n2. Production Retry Configuration");
    println!("{}", "-".repeat(40));

    let production_config = RetryConfig::production();
    println!("Settings:");
    println!("  Max retries: {}", production_config.max_retries);
    println!("  Initial delay: {:?}", production_config.initial_delay);
    println!("  Max delay: {:?}", production_config.max_delay);
    println!(
        "  Backoff multiplier: {}",
        production_config.backoff_multiplier
    );
    println!("  Jitter: {}", production_config.jitter);

    let client = LLMKitClient::builder()
        .with_openrouter_from_env()
        .with_retry(production_config)
        .build()
        .await?;

    let request = CompletionRequest::new(
        "openrouter/qwen/qwen3-32b",
        vec![Message::user("Say 'production config works'")],
    )
    .without_thinking()
    .with_max_tokens(50);

    let response = client.complete(request).await?;
    println!("Response: {}", response.text_content().trim());

    // ========================================
    // Example 3: Conservative Retry Config
    // ========================================
    println!("\n3. Conservative Retry Configuration");
    println!("{}", "-".repeat(40));

    let conservative_config = RetryConfig::conservative();
    println!("Settings:");
    println!("  Max retries: {}", conservative_config.max_retries);
    println!("  Initial delay: {:?}", conservative_config.initial_delay);
    println!("  Max delay: {:?}", conservative_config.max_delay);
    println!("Best for: Quick failures, interactive applications");

    let client = LLMKitClient::builder()
        .with_openrouter_from_env()
        .with_retry(conservative_config)
        .build()
        .await?;

    let request = CompletionRequest::new(
        "openrouter/qwen/qwen3-32b",
        vec![Message::user("Say 'conservative works'")],
    )
    .without_thinking()
    .with_max_tokens(50);

    let response = client.complete(request).await?;
    println!("Response: {}", response.text_content().trim());

    // ========================================
    // Example 4: Custom Retry Config
    // ========================================
    println!("\n4. Custom Retry Configuration");
    println!("{}", "-".repeat(40));

    let custom_config = RetryConfig::new(5) // 5 max retries
        .with_initial_delay(Duration::from_millis(500))
        .with_max_delay(Duration::from_secs(10))
        .with_backoff_multiplier(1.5)
        .with_jitter(true);

    println!("Custom settings:");
    println!("  Max retries: 5");
    println!("  Initial delay: 500ms");
    println!("  Max delay: 10s");
    println!("  Backoff multiplier: 1.5x");
    println!("  Jitter: enabled");

    let client = LLMKitClient::builder()
        .with_openrouter_from_env()
        .with_retry(custom_config)
        .build()
        .await?;

    let request = CompletionRequest::new(
        "openrouter/qwen/qwen3-32b",
        vec![Message::user("Say 'custom config works'")],
    )
    .without_thinking()
    .with_max_tokens(50);

    let response = client.complete(request).await?;
    println!("Response: {}", response.text_content().trim());

    // ========================================
    // Example 5: No Retry (for testing/debugging)
    // ========================================
    println!("\n5. No Retry Configuration");
    println!("{}", "-".repeat(40));
    println!("Useful for: Testing, debugging, fail-fast scenarios");

    let client = LLMKitClient::builder()
        .with_openrouter_from_env()
        .with_retry(RetryConfig::none())
        .build()
        .await?;

    let request = CompletionRequest::new(
        "openrouter/qwen/qwen3-32b",
        vec![Message::user("Say 'no retry works'")],
    )
    .without_thinking()
    .with_max_tokens(50);

    let response = client.complete(request).await?;
    println!("Response: {}", response.text_content().trim());

    // Summary
    println!("\n{}", "=".repeat(50));
    println!("Summary: Retry Configurations");
    println!("{}", "=".repeat(50));
    println!(
        "
Retry handles these transient errors automatically:
  - Rate limits (429)
  - Server errors (5xx)
  - Timeouts
  - Connection errors

Configuration options:
  - with_default_retry()      -> Default production settings
  - with_retry(config)        -> Custom RetryConfig
  - RetryConfig::production() -> Aggressive (10 retries, 5min max)
  - RetryConfig::conservative() -> Quick fail (3 retries, 30s max)
  - RetryConfig::none()       -> No retries (fail immediately)
"
    );

    Ok(())
}
