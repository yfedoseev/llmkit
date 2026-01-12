//! Image Generation Example
//!
//! Demonstrates image generation from text prompts using various providers.
//!
//! Requirements:
//! - Set OPENAI_API_KEY environment variable (for DALL-E)
//!
//! Run:
//!     cargo run --example 14_image_generation

use llmkit::{ImageGenerationRequest, ImageQuality, ImageSize, ImageStyle, LLMKitClient};

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_openai_from_env()
        .build()
        .await?;

    println!("=== Image Generation Example ===\n");

    // Example 1: Simple image generation
    println!("--- Example 1: Simple Image Generation ---");
    let request = ImageGenerationRequest::new(
        "openai/dall-e-3",
        "A serene landscape with mountains and a clear blue lake at sunset",
    );

    let response = client.generate_image(request).await?;
    println!("Generated {} image(s)", response.images.len());
    if let Some(image) = response.images.first() {
        if let Some(url) = &image.url {
            println!("Image URL: {}", url);
        }
        if let Some(revised) = &image.revised_prompt {
            println!("Revised prompt: {}", revised);
        }
    }

    // Example 2: High-quality image with vivid style
    println!("\n--- Example 2: High-Quality Vivid Image ---");
    let request = ImageGenerationRequest::new(
        "openai/dall-e-3",
        "A portrait of a Renaissance noble in ornate clothing",
    )
    .with_quality(ImageQuality::Hd)
    .with_style(ImageStyle::Vivid);

    let response = client.generate_image(request).await?;
    println!("Generated HD image");
    if let Some(image) = response.images.first() {
        if let Some(url) = &image.url {
            println!("Image URL: {}", url);
        }
    }

    // Example 3: Natural style image
    println!("\n--- Example 3: Natural Style Image ---");
    let request = ImageGenerationRequest::new(
        "openai/dall-e-3",
        "A photograph of a coffee shop interior with warm lighting",
    )
    .with_style(ImageStyle::Natural);

    let response = client.generate_image(request).await?;
    println!("Generated natural-style image");
    if let Some(image) = response.images.first() {
        if let Some(url) = &image.url {
            println!("Image URL: {}", url);
        }
    }

    // Example 4: Landscape orientation
    println!("\n--- Example 4: Landscape Orientation ---");
    let request = ImageGenerationRequest::new(
        "openai/dall-e-3",
        "A panoramic view of a futuristic city skyline at night",
    )
    .with_size(ImageSize::Landscape1792x1024);

    let response = client.generate_image(request).await?;
    println!("Generated landscape image (1792x1024)");
    if let Some(image) = response.images.first() {
        if let Some(url) = &image.url {
            println!("Image URL: {}", url);
        }
    }

    // Example 5: Portrait orientation
    println!("\n--- Example 5: Portrait Orientation ---");
    let request = ImageGenerationRequest::new(
        "openai/dall-e-3",
        "A detailed character portrait of a fantasy elf warrior",
    )
    .with_size(ImageSize::Portrait1024x1792)
    .with_quality(ImageQuality::Hd);

    let response = client.generate_image(request).await?;
    println!("Generated portrait image (1024x1792)");
    if let Some(image) = response.images.first() {
        if let Some(url) = &image.url {
            println!("Image URL: {}", url);
        }
    }

    // Example 6: Square image
    println!("\n--- Example 6: Square Image ---");
    let request = ImageGenerationRequest::new(
        "openai/dall-e-3",
        "An abstract geometric pattern with bold colors",
    )
    .with_size(ImageSize::Square1024);

    let response = client.generate_image(request).await?;
    println!("Generated square image (1024x1024)");
    if let Some(image) = response.images.first() {
        if let Some(url) = &image.url {
            println!("Image URL: {}", url);
        }
    }

    // Example 7: Multiple use cases
    println!("\n--- Example 7: Different Use Cases ---");
    let use_cases = [
        (
            "Marketing",
            "A modern product packaging design for organic tea",
        ),
        ("Education", "An illustrated diagram of the solar system"),
        (
            "Entertainment",
            "A fantasy dragon flying over a medieval castle",
        ),
        ("Art", "An impressionist painting of a garden in spring"),
    ];

    for (category, prompt) in use_cases {
        let request = ImageGenerationRequest::new("openai/dall-e-3", prompt);
        let response = client.generate_image(request).await?;
        println!(
            "  {}: Generated image for '{}'",
            category,
            &prompt[..40.min(prompt.len())]
        );
        if let Some(image) = response.images.first() {
            if let Some(url) = &image.url {
                println!("    URL: {}", url);
            }
        }
    }

    println!("\nImage generation examples completed!");
    println!("\nNote: Images are hosted temporarily. Download them if you want to keep them.");

    Ok(())
}
