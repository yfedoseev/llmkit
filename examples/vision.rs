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
        .build()?;

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

#[allow(dead_code)]
async fn analyze_local_image(client: &LLMKitClient, image_path: &str) -> llmkit::Result<()> {
    use std::fs;

    // Check if file exists
    if !std::path::Path::new(image_path).exists() {
        println!("Image not found: {}", image_path);
        return Ok(());
    }

    // Detect media type
    let ext = image_path
        .split('.')
        .next_back()
        .unwrap_or("png")
        .to_lowercase();
    let media_type = match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => "image/png",
    };

    // Read and encode
    let image_bytes = fs::read(image_path)
        .map_err(|e| llmkit::Error::Configuration(format!("Failed to read image: {}", e)))?;
    let image_data =
        base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &image_bytes);

    // Create message with image
    let message = Message::user_with_content(vec![
        ContentBlock::Text {
            text: "Analyze this image. What's in it?".to_string(),
        },
        ContentBlock::Image {
            media_type: media_type.to_string(),
            data: image_data,
        },
    ]);

    let request = CompletionRequest::new("anthropic/claude-sonnet-4-20250514", vec![message])
        .with_max_tokens(500);

    println!("Analyzing local image: {}", image_path);
    let response = client.complete(request).await?;
    println!("\nAnalysis:\n{}", response.text_content());

    Ok(())
}
