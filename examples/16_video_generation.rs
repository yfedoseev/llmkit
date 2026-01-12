//! Video Generation Example
//!
//! Demonstrates video generation from text prompts using various providers.
//!
//! Requirements:
//! - Set RUNWAYML_API_KEY environment variable
//!
//! Run:
//!     cargo run --example 16_video_generation

use llmkit::{CameraMotion, LLMKitClient, VideoGenerationRequest, VideoJobStatus, VideoResolution};
use std::time::Duration;

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_runwayml_from_env()
        .build()
        .await?;

    println!("=== Video Generation Example ===\n");

    // Example 1: Simple text-to-video
    println!("--- Example 1: Simple Text-to-Video ---");
    let request = VideoGenerationRequest::new(
        "runwayml/gen-3-alpha",
        "A serene mountain landscape with clouds slowly drifting across the sky",
    );

    let response = client.generate_video(request).await?;
    println!("Job started: {}", response.job_id);
    println!("Initial status: {:?}", response.status);

    // Poll for completion
    poll_video_job(&client, "runwayml", &response.job_id).await?;

    // Example 2: Video with specific duration and aspect ratio
    println!("\n--- Example 2: Custom Duration and Aspect Ratio ---");
    let request = VideoGenerationRequest::new(
        "runwayml/gen-3-alpha",
        "A bustling city street at night with neon lights reflecting on wet pavement",
    )
    .with_duration(5)
    .with_aspect_ratio("16:9");

    let response = client.generate_video(request).await?;
    println!("Job started: {}", response.job_id);
    poll_video_job(&client, "runwayml", &response.job_id).await?;

    // Example 3: Video with camera motion
    println!("\n--- Example 3: With Camera Motion ---");
    let request = VideoGenerationRequest::new(
        "runwayml/gen-3-alpha",
        "A beautiful sunset over the ocean with gentle waves",
    )
    .with_camera_motion(CameraMotion::ZoomOut)
    .with_duration(4);

    let response = client.generate_video(request).await?;
    println!("Job started: {}", response.job_id);
    println!("Camera motion: zoom out");
    poll_video_job(&client, "runwayml", &response.job_id).await?;

    // Example 4: High-resolution video
    println!("\n--- Example 4: High-Resolution Video ---");
    let request = VideoGenerationRequest::new(
        "runwayml/gen-3-alpha",
        "A professional product showcase of a luxury watch rotating slowly",
    )
    .with_resolution(VideoResolution::Hd1080)
    .with_fps(30);

    let response = client.generate_video(request).await?;
    println!("Job started: {}", response.job_id);
    println!("Resolution: 1080p, 30 fps");
    poll_video_job(&client, "runwayml", &response.job_id).await?;

    // Example 5: Video with seed for reproducibility
    println!("\n--- Example 5: Reproducible Video (with seed) ---");
    let request =
        VideoGenerationRequest::new("runwayml/gen-3-alpha", "A cat playing with a ball of yarn")
            .with_seed(12345);

    let response = client.generate_video(request).await?;
    println!("Job started: {}", response.job_id);
    println!("Seed: 12345");
    poll_video_job(&client, "runwayml", &response.job_id).await?;

    // Example 6: Video with motion amount control
    println!("\n--- Example 6: Controlled Motion Amount ---");
    let request = VideoGenerationRequest::new(
        "runwayml/gen-3-alpha",
        "A field of flowers gently swaying in the breeze",
    )
    .with_motion_amount(0.3) // Subtle motion
    .with_duration(6);

    let response = client.generate_video(request).await?;
    println!("Job started: {}", response.job_id);
    println!("Motion amount: 0.3 (subtle)");
    poll_video_job(&client, "runwayml", &response.job_id).await?;

    // Example 7: Different camera motions
    println!("\n--- Example 7: Various Camera Motions ---");
    let motions = [
        (CameraMotion::Static, "Static camera"),
        (CameraMotion::ZoomIn, "Zoom in"),
        (CameraMotion::PanLeft, "Pan left"),
        (CameraMotion::TiltUp, "Tilt up"),
        (CameraMotion::Orbit, "Orbit"),
    ];

    for (motion, description) in motions.iter().take(2) {
        // Only demo 2 for time
        let request = VideoGenerationRequest::new(
            "runwayml/gen-3-alpha",
            "A beautiful landscape with mountains and a lake",
        )
        .with_camera_motion(*motion)
        .with_duration(3);

        let response = client.generate_video(request).await?;
        println!("  {} - Job: {}", description, response.job_id);
    }

    println!("\nVideo generation examples completed!");
    println!(
        "\nNote: Video generation is asynchronous. Jobs may take several minutes to complete."
    );
    println!("Use the job ID to poll for status and retrieve the final video URL.");

    Ok(())
}

/// Poll for video job completion.
async fn poll_video_job(client: &LLMKitClient, provider: &str, job_id: &str) -> llmkit::Result<()> {
    let max_polls = 60; // Max 5 minutes (5 seconds * 60)
    let mut poll_count = 0;

    loop {
        let status = client.get_video_status(provider, job_id).await?;

        match status {
            VideoJobStatus::Completed {
                video_url,
                duration_seconds,
                ..
            } => {
                println!("  Video ready!");
                println!("  URL: {}", video_url);
                if let Some(duration) = duration_seconds {
                    println!("  Duration: {:.1}s", duration);
                }
                return Ok(());
            }
            VideoJobStatus::Processing { progress, stage } => {
                let progress_str = progress
                    .map(|p| format!("{}%", p))
                    .unwrap_or_else(|| "...".to_string());
                let stage_str = stage.unwrap_or_else(|| "processing".to_string());
                println!("  Status: {} ({})", stage_str, progress_str);
            }
            VideoJobStatus::Queued => {
                println!("  Status: queued");
            }
            VideoJobStatus::Failed { error, code } => {
                println!("  Failed: {} (code: {:?})", error, code);
                return Ok(());
            }
            VideoJobStatus::Cancelled => {
                println!("  Job was cancelled");
                return Ok(());
            }
        }

        poll_count += 1;
        if poll_count >= max_polls {
            println!("  Timed out waiting for video. Check job {} later.", job_id);
            return Ok(());
        }

        tokio::time::sleep(Duration::from_secs(5)).await;
    }
}
