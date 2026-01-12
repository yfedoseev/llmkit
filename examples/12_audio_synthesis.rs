//! Audio Synthesis (Text-to-Speech) Example
//!
//! Demonstrates speech generation from text using various providers.
//!
//! Requirements:
//! - Set OPENAI_API_KEY environment variable (for OpenAI TTS)
//!   Or ELEVENLABS_API_KEY for ElevenLabs
//!
//! Run:
//!     cargo run --example 12_audio_synthesis

use llmkit::{AudioFormat, LLMKitClient, SpeechRequest};
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LLMKitClient::builder()
        .with_openai_from_env()
        .build()
        .await?;

    println!("=== Audio Synthesis Example ===\n");

    // Example 1: Basic speech synthesis
    println!("--- Example 1: Basic Speech Synthesis ---");
    let text = "Hello! This is a demonstration of text-to-speech synthesis using LLMKit.";
    println!("Input text: {}", text);

    let request = SpeechRequest::new("openai/tts-1", text, "alloy");
    let response = client.speech(request).await?;

    let output_path = "output_basic.mp3";
    response.save(output_path)?;
    println!("Audio saved to: {}", output_path);
    println!("Audio size: {} bytes", response.audio.len());
    println!("Format: {:?}", response.format);

    // Example 2: High-quality speech with different voice
    println!("\n--- Example 2: High-Quality Speech ---");
    let text = "This is high-quality audio synthesis with the 'nova' voice.";

    let request = SpeechRequest::new("openai/tts-1-hd", text, "nova");
    let response = client.speech(request).await?;

    let output_path = "output_hd.mp3";
    response.save(output_path)?;
    println!("HD audio saved to: {}", output_path);
    println!("Audio size: {} bytes", response.audio.len());

    // Example 3: Different audio format
    println!("\n--- Example 3: Different Audio Format ---");
    let text = "Testing different audio formats.";

    let request =
        SpeechRequest::new("openai/tts-1", text, "shimmer").with_format(AudioFormat::Opus);
    let response = client.speech(request).await?;

    let output_path = "output_opus.opus";
    response.save(output_path)?;
    println!("Opus audio saved to: {}", output_path);
    println!("Format: {:?}", response.format);

    // Example 4: Adjusted speech speed
    println!("\n--- Example 4: Adjusted Speech Speed ---");
    let text = "This speech is generated at a faster pace.";

    let request = SpeechRequest::new("openai/tts-1", text, "echo").with_speed(1.25);
    let response = client.speech(request).await?;

    let output_path = "output_fast.mp3";
    response.save(output_path)?;
    println!("Fast audio saved to: {}", output_path);

    // Example 5: Slow speech
    println!("\n--- Example 5: Slow Speech ---");
    let text = "This speech is generated at a slower, more deliberate pace.";

    let request = SpeechRequest::new("openai/tts-1", text, "fable").with_speed(0.8);
    let response = client.speech(request).await?;

    let output_path = "output_slow.mp3";
    response.save(output_path)?;
    println!("Slow audio saved to: {}", output_path);

    // Example 6: Multiple voices comparison
    println!("\n--- Example 6: Voice Comparison ---");
    let voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"];
    let text = "The quick brown fox jumps over the lazy dog.";

    for voice in voices {
        let request = SpeechRequest::new("openai/tts-1", text, voice);
        let response = client.speech(request).await?;

        let output_path = format!("output_{}.mp3", voice);
        response.save(&output_path)?;
        println!("  Voice '{}': {} bytes", voice, response.audio.len());
    }

    // Clean up example files
    println!("\n--- Cleanup ---");
    let files_to_clean = [
        "output_basic.mp3",
        "output_hd.mp3",
        "output_opus.opus",
        "output_fast.mp3",
        "output_slow.mp3",
        "output_alloy.mp3",
        "output_echo.mp3",
        "output_fable.mp3",
        "output_onyx.mp3",
        "output_nova.mp3",
        "output_shimmer.mp3",
    ];

    for file in &files_to_clean {
        if Path::new(file).exists() {
            std::fs::remove_file(file)?;
            println!("Cleaned up: {}", file);
        }
    }

    println!("\nAudio synthesis examples completed!");

    Ok(())
}
