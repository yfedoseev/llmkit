//! Response Caching Example
//!
//! Demonstrates client-side response caching to reduce API costs
//! by caching identical requests. Uses qwen/qwen3-32b on OpenRouter.
//!
//! Cache hit = no API call = free!
//!
//! Requirements:
//! - Set OPENROUTER_API_KEY environment variable
//!
//! Run:
//!     cargo run --example 17_response_caching

use llmkit::cache::{CacheConfig, CachingProvider, InMemoryCache};
use llmkit::providers::OpenRouterProvider;
use llmkit::{CompletionRequest, LLMKitClient, Message};
use std::sync::Arc;
use std::time::Duration;

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    // Create the base provider
    let provider = OpenRouterProvider::from_env()?;

    // Wrap with caching - caches identical requests for 1 hour
    let cache = InMemoryCache::new(
        CacheConfig::new()
            .with_ttl(Duration::from_secs(3600))
            .with_max_entries(1000),
    );
    let cached_provider = Arc::new(CachingProvider::new(provider, cache));

    // Create client with the cached provider
    let client = LLMKitClient::builder()
        .with_provider("openrouter", cached_provider.clone())
        .build()
        .await?;

    let model = "openrouter/qwen/qwen3-32b";

    println!("Response Caching Example");
    println!("{}", "=".repeat(50));

    // First request - will hit the API
    println!("\n1. First request (cache MISS - API call):");
    println!("{}", "-".repeat(40));

    let request =
        CompletionRequest::new(model, vec![Message::user("What is the capital of France?")])
            .with_system("Answer briefly in one sentence.")
            .without_thinking()
            .with_max_tokens(100);

    let response = client.complete(request.clone()).await?;
    let stats1 = cached_provider.stats();
    println!("Response: {}", response.text_content().trim());
    println!(
        "Cache stats: {} hits, {} misses",
        stats1.hits, stats1.misses
    );

    // Second identical request - should hit the cache
    println!("\n2. Identical request (cache HIT - no API call!):");
    println!("{}", "-".repeat(40));

    let request =
        CompletionRequest::new(model, vec![Message::user("What is the capital of France?")])
            .with_system("Answer briefly in one sentence.")
            .without_thinking()
            .with_max_tokens(100);

    let response = client.complete(request).await?;
    let stats2 = cached_provider.stats();
    println!("Response: {}", response.text_content().trim());
    println!(
        "Cache stats: {} hits, {} misses",
        stats2.hits, stats2.misses
    );

    // Third request with different question - cache miss
    println!("\n3. Different question (cache MISS - API call):");
    println!("{}", "-".repeat(40));

    let request = CompletionRequest::new(
        model,
        vec![Message::user("What is the capital of Germany?")],
    )
    .with_system("Answer briefly in one sentence.")
    .without_thinking()
    .with_max_tokens(100);

    let response = client.complete(request).await?;
    let stats3 = cached_provider.stats();
    println!("Response: {}", response.text_content().trim());
    println!(
        "Cache stats: {} hits, {} misses",
        stats3.hits, stats3.misses
    );

    // Fourth request - same as first (cache hit)
    println!("\n4. Same as first question (cache HIT):");
    println!("{}", "-".repeat(40));

    let request =
        CompletionRequest::new(model, vec![Message::user("What is the capital of France?")])
            .with_system("Answer briefly in one sentence.")
            .without_thinking()
            .with_max_tokens(100);

    let response = client.complete(request).await?;
    let stats4 = cached_provider.stats();
    println!("Response: {}", response.text_content().trim());
    println!(
        "Cache stats: {} hits, {} misses",
        stats4.hits, stats4.misses
    );

    // Summary
    println!("\n{}", "=".repeat(50));
    println!("Summary:");
    println!("  Total requests: 4");
    println!("  API calls made: {} (misses)", stats4.misses);
    println!("  Free responses: {} (hits)", stats4.hits);
    println!("  Hit rate: {:.0}%", stats4.hit_rate() * 100.0);
    println!("\nCaching saves API costs on repeated identical requests!");

    Ok(())
}
