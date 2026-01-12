//! OpenAI-Compatible Provider Example
//!
//! Demonstrates using OpenRouter through the generic OpenAI-compatible
//! provider interface. This shows that any OpenAI-compatible API can be
//! used with LLMKit.
//!
//! Note: The dedicated OpenRouterProvider is recommended for Qwen3 models
//! because it automatically handles the `/no_think` injection. When using
//! the OpenAI-compatible endpoint, you need to manually add `/no_think`
//! to prompts or use higher max_tokens.
//!
//! Requirements:
//! - Set OPENROUTER_API_KEY environment variable
//!
//! Run:
//!     cargo run --example 19_openai_compatible

use llmkit::providers::OpenAICompatibleProvider;
use llmkit::{CompletionRequest, LLMKitClient, Message};
use std::sync::Arc;

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    let api_key = std::env::var("OPENROUTER_API_KEY")
        .expect("OPENROUTER_API_KEY environment variable not set");

    // Create OpenRouter provider using the OpenAI-compatible interface
    // Note: Provider name cannot contain dashes due to model format parsing
    let provider = OpenAICompatibleProvider::custom(
        "openroutercompat",
        "https://openrouter.ai/api/v1",
        Some(api_key),
    )?;

    let client = LLMKitClient::builder()
        .with_provider("openroutercompat", Arc::new(provider))
        .build()
        .await?;

    // Model format is "provider/model" - everything after first "/" is the model name
    let model = "openroutercompat/qwen/qwen3-32b";

    println!("OpenAI-Compatible Provider Example");
    println!("{}", "=".repeat(50));
    println!("Using OpenRouter via OpenAI-compatible endpoint\n");

    // Since we're not using OpenRouterProvider, we need to manually
    // add /no_think to disable Qwen3's reasoning mode
    println!("Request with manual /no_think:");
    println!("{}", "-".repeat(40));

    let request = CompletionRequest::new(
        model,
        vec![Message::user("What is 2 + 2? Answer briefly. /no_think")],
    )
    .with_system("You are a helpful assistant.")
    .with_max_tokens(100);

    let response = client.complete(request).await?;
    println!("Response: {}\n", response.text_content().trim());

    // Second request
    println!("Another request:");
    println!("{}", "-".repeat(40));

    let request = CompletionRequest::new(model, vec![Message::user("Name 3 colors. /no_think")])
        .with_system("Be concise.")
        .with_max_tokens(100);

    let response = client.complete(request).await?;
    println!("Response: {}\n", response.text_content().trim());

    println!("{}", "=".repeat(50));
    println!("OpenRouter works via OpenAI-compatible endpoint!");
    println!("Note: Manual /no_think required for Qwen3 models.");

    Ok(())
}
