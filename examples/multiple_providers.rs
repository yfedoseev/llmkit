//! Multiple Providers Example
//!
//! Demonstrates how to configure and use multiple LLM providers.
//!
//! Requirements:
//! - Set multiple provider API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
//!
//! Run with:
//!     cargo run --example multiple_providers

use llmkit::{CompletionRequest, LLMKitClient, Message};

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    // Example 1: Auto-detect from environment
    println!("{}", "=".repeat(50));
    println!("Example 1: Auto-detect from Environment");
    println!("{}", "=".repeat(50));
    using_from_env().await?;

    // Example 2: Switch between providers
    println!("\n{}", "=".repeat(50));
    println!("Example 2: Switch Between Providers");
    println!("{}", "=".repeat(50));
    switch_between_providers().await?;

    // Example 3: Provider fallback
    println!("\n{}", "=".repeat(50));
    println!("Example 3: Provider Fallback");
    println!("{}", "=".repeat(50));
    provider_fallback().await?;

    Ok(())
}

async fn using_from_env() -> llmkit::Result<()> {
    // Build client with all available providers from environment
    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .with_openai_from_env()
        .with_default_retry()
        .build()?;

    // List all detected providers
    let providers = client.providers();
    println!("Detected providers: {:?}", providers);

    // Use the default provider (Anthropic if available)
    let response = client
        .complete(
            CompletionRequest::new("claude-sonnet-4-20250514", vec![Message::user("Say hello")])
                .with_max_tokens(50),
        )
        .await?;

    println!("\nDefault provider response: {}", response.text_content());

    Ok(())
}

async fn switch_between_providers() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .with_openai_from_env()
        .with_default_retry()
        .build()?;

    let providers = client.providers();
    println!("Available: {:?}\n", providers);

    let prompt = "What's 2+2? Answer with just the number.";

    // Model mapping - only use default providers (anthropic and openai)
    let models = [
        ("anthropic", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4o"),
    ];

    // Try different providers if available
    for (provider, model) in models {
        if !providers.iter().any(|p| p == &provider) {
            println!("{}: Not configured", provider);
            continue;
        }

        match client
            .complete_with_provider(
                provider,
                CompletionRequest::new(model, vec![Message::user(prompt)]).with_max_tokens(20),
            )
            .await
        {
            Ok(response) => {
                println!(
                    "{} ({}): {}",
                    provider,
                    model,
                    response.text_content().trim()
                );
            }
            Err(e) => {
                println!("{}: Error - {}", provider, e);
            }
        }
    }

    Ok(())
}

async fn provider_fallback() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .with_openai_from_env()
        .with_default_retry()
        .build()?;

    // Order providers by preference
    let provider_priority = [
        ("anthropic", "claude-sonnet-4-20250514"),
        ("openai", "gpt-4o"),
    ];

    let available: std::collections::HashSet<_> = client.providers().into_iter().collect();

    for (provider, model) in provider_priority {
        if !available.contains(provider) {
            println!("Skipping {} (not configured)", provider);
            continue;
        }

        print!("Trying {}... ", provider);
        match client
            .complete_with_provider(
                provider,
                CompletionRequest::new(model, vec![Message::user("What is Python?")])
                    .with_max_tokens(100),
            )
            .await
        {
            Ok(response) => {
                println!("Success!");
                println!(
                    "Response: {}...",
                    &response.text_content()[..100.min(response.text_content().len())]
                );
                return Ok(());
            }
            Err(e) => {
                println!("Failed: {}", e);
                continue;
            }
        }
    }

    println!("All providers failed!");
    Ok(())
}
