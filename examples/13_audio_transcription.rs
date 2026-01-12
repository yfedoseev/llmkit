//! Audio Transcription (Speech-to-Text) Example
//!
//! Demonstrates audio transcription using various providers.
//!
//! Requirements:
//! - Set OPENAI_API_KEY environment variable (for Whisper)
//!   Or DEEPGRAM_API_KEY for Deepgram
//!
//! Run:
//!     cargo run --example 13_audio_transcription -- <audio_file>

use llmkit::{AudioInput, LLMKitClient, TranscriptionRequest};
use std::env;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get audio file from command line
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --example 13_audio_transcription -- <audio_file>");
        eprintln!("\nExample:");
        eprintln!("  cargo run --example 13_audio_transcription -- speech.mp3");
        std::process::exit(1);
    }

    let audio_path = &args[1];
    if !Path::new(audio_path).exists() {
        eprintln!("Error: Audio file not found: {}", audio_path);
        std::process::exit(1);
    }

    let client = LLMKitClient::builder()
        .with_openai_from_env()
        .build()
        .await?;

    println!("=== Audio Transcription Example ===\n");
    println!("Audio file: {}", audio_path);

    // Example 1: Basic transcription
    println!("\n--- Example 1: Basic Transcription ---");
    let request = TranscriptionRequest::new("openai/whisper-1", AudioInput::file(audio_path));

    let response = client.transcribe(request).await?;

    println!("Transcript:");
    println!("  {}", response.text);
    if let Some(lang) = &response.language {
        println!("Detected language: {}", lang);
    }
    if let Some(duration) = response.duration {
        println!("Duration: {:.2}s", duration);
    }

    // Example 2: Transcription with language hint
    println!("\n--- Example 2: With Language Hint ---");
    let request = TranscriptionRequest::new("openai/whisper-1", AudioInput::file(audio_path))
        .with_language("en");

    let response = client.transcribe(request).await?;
    println!("Transcript (en): {}", response.text);

    // Example 3: Transcription with prompt guidance
    println!("\n--- Example 3: With Prompt Guidance ---");
    let request = TranscriptionRequest::new("openai/whisper-1", AudioInput::file(audio_path))
        .with_prompt("This is a technical discussion about programming and software development.");

    let response = client.transcribe(request).await?;
    println!("Transcript (guided): {}", response.text);

    // Example 4: Transcription with word timestamps
    println!("\n--- Example 4: With Word Timestamps ---");
    let request = TranscriptionRequest::new("openai/whisper-1", AudioInput::file(audio_path))
        .with_word_timestamps();

    let response = client.transcribe(request).await?;
    println!("Transcript: {}", response.text);

    if let Some(words) = &response.words {
        println!("\nWord-level timing (first 10 words):");
        for word in words.iter().take(10) {
            println!("  [{:.2}s - {:.2}s] {}", word.start, word.end, word.word);
        }
        if words.len() > 10 {
            println!("  ... and {} more words", words.len() - 10);
        }
    }

    // Example 5: Transcription with segment timestamps
    println!("\n--- Example 5: With Segment Timestamps ---");
    let request = TranscriptionRequest::new("openai/whisper-1", AudioInput::file(audio_path))
        .with_segment_timestamps();

    let response = client.transcribe(request).await?;

    if let Some(segments) = &response.segments {
        println!("Segments:");
        for segment in segments.iter().take(5) {
            println!(
                "  [{:.2}s - {:.2}s] {}",
                segment.start, segment.end, segment.text
            );
        }
        if segments.len() > 5 {
            println!("  ... and {} more segments", segments.len() - 5);
        }
    }

    // Example 6: Transcription from bytes
    println!("\n--- Example 6: Transcription from Bytes ---");
    let audio_bytes = std::fs::read(audio_path)?;
    let filename = Path::new(audio_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("audio.mp3");

    let media_type = match Path::new(audio_path).extension().and_then(|e| e.to_str()) {
        Some("mp3") => "audio/mpeg",
        Some("wav") => "audio/wav",
        Some("m4a") => "audio/mp4",
        Some("ogg") => "audio/ogg",
        _ => "audio/mpeg",
    };

    let request = TranscriptionRequest::new(
        "openai/whisper-1",
        AudioInput::bytes(audio_bytes, filename, media_type),
    );

    let response = client.transcribe(request).await?;
    println!("Transcript from bytes: {}", response.text);

    println!("\nAudio transcription examples completed!");

    Ok(())
}
