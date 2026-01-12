//! Vision / Image Analysis Example
//!
//! Demonstrates image input capabilities with LLMKit.
//!
//! Requirements:
//! - Set ANTHROPIC_API_KEY environment variable (or use OpenAI's GPT-4V)
//!
//! Run with:
//!     cargo run --example vision

use llmkit::{CompletionRequest, ContentBlock, LLMKitClient, Message};

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .with_default_retry()
        .build()
        .await?;

    // Example 1: Analyze image from URL
    println!("{}", "=".repeat(50));
    println!("Example 1: Analyze image from URL");
    println!("{}", "=".repeat(50));
    analyze_image_from_url(&client).await?;

    // Example 2: Compare multiple images
    println!("\n{}", "=".repeat(50));
    println!("Example 2: Compare multiple images");
    println!("{}", "=".repeat(50));
    multi_image_comparison(&client).await?;

    Ok(())
}

async fn analyze_image_from_url(client: &LLMKitClient) -> llmkit::Result<()> {
    // Create a message with an image URL
    let message = Message::user_with_content(vec![
        ContentBlock::Text {
            text: "What do you see in this image? Describe it briefly.".to_string(),
        },
        ContentBlock::ImageUrl {
            url: "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/\
                  Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
                .to_string(),
        },
    ]);

    // Use "provider/model" format for explicit provider routing
    let request = CompletionRequest::new("anthropic/claude-sonnet-4-20250514", vec![message])
        .with_max_tokens(500);

    println!("Analyzing image from URL...");
    let response = client.complete(request).await?;
    println!("\nDescription:\n{}", response.text_content());

    Ok(())
}

async fn multi_image_comparison(client: &LLMKitClient) -> llmkit::Result<()> {
    // Example with multiple image URLs
    let message = Message::user_with_content(vec![
        ContentBlock::Text {
            text: "I'm going to show you two images. \
                   Please compare them and describe the differences."
                .to_string(),
        },
        ContentBlock::ImageUrl {
            url: "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/\
                  Cat03.jpg/120px-Cat03.jpg"
                .to_string(),
        },
        ContentBlock::ImageUrl {
            url: "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/\
                  YellowLabradorLooking_new.jpg/120px-YellowLabradorLooking_new.jpg"
                .to_string(),
        },
    ]);

    let request = CompletionRequest::new("anthropic/claude-sonnet-4-20250514", vec![message])
        .with_max_tokens(500);

    println!("Comparing two images...");
    let response = client.complete(request).await?;
    println!("\nComparison:\n{}", response.text_content());

    Ok(())
}
