//! Video generation API for creating videos from text prompts and images.
//!
//! This module provides a unified interface for video generation across various
//! providers including RunwayML, Pika, Luma, Kling, and others.
//!
//! # Text-to-Video Example
//!
//! ```ignore
//! use modelsuite::{VideoProvider, VideoGenerationRequest};
//!
//! // Create provider
//! let provider = RunwayProvider::from_env()?;
//!
//! // Start video generation
//! let request = VideoGenerationRequest::new(
//!     "gen-3",
//!     "A serene mountain landscape with clouds rolling through",
//! );
//!
//! let job = provider.generate_video(request).await?;
//! println!("Job started: {}", job.job_id);
//!
//! // Poll for completion
//! loop {
//!     let status = provider.get_video_status(&job.job_id).await?;
//!     match status {
//!         VideoJobStatus::Completed { video_url, .. } => {
//!             println!("Video ready: {}", video_url);
//!             break;
//!         }
//!         VideoJobStatus::Processing { progress } => {
//!             println!("Progress: {}%", progress.unwrap_or(0));
//!         }
//!         VideoJobStatus::Failed { error } => {
//!             return Err(error.into());
//!         }
//!         _ => {}
//!     }
//!     tokio::time::sleep(Duration::from_secs(5)).await;
//! }
//! ```
//!
//! # Image-to-Video Example
//!
//! ```ignore
//! use modelsuite::{VideoProvider, VideoGenerationRequest, VideoInput};
//!
//! let request = VideoGenerationRequest::new("gen-3", "Camera slowly zooms out")
//!     .with_image(VideoInput::file("input_frame.png"));
//!
//! let job = provider.generate_video(request).await?;
//! ```

use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

// ============================================================================
// Video Generation Request/Response Types
// ============================================================================

/// Request for generating a video.
#[derive(Debug, Clone)]
pub struct VideoGenerationRequest {
    /// The model to use for generation.
    pub model: String,
    /// Text prompt describing the video to generate.
    pub prompt: String,
    /// Optional negative prompt (things to avoid).
    pub negative_prompt: Option<String>,
    /// Optional input image for image-to-video generation.
    pub image: Option<VideoInput>,
    /// Optional input video for video-to-video generation.
    pub input_video: Option<VideoInput>,
    /// Video duration in seconds.
    pub duration: Option<u32>,
    /// Aspect ratio (e.g., "16:9", "9:16", "1:1").
    pub aspect_ratio: Option<String>,
    /// Video resolution/quality preset.
    pub resolution: Option<VideoResolution>,
    /// Frames per second.
    pub fps: Option<u32>,
    /// Seed for reproducibility.
    pub seed: Option<u64>,
    /// Motion intensity/amount (0.0 to 1.0).
    pub motion_amount: Option<f32>,
    /// Camera movement type.
    pub camera_motion: Option<CameraMotion>,
}

impl VideoGenerationRequest {
    /// Create a new video generation request.
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            prompt: prompt.into(),
            negative_prompt: None,
            image: None,
            input_video: None,
            duration: None,
            aspect_ratio: None,
            resolution: None,
            fps: None,
            seed: None,
            motion_amount: None,
            camera_motion: None,
        }
    }

    /// Set a negative prompt.
    pub fn with_negative_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.negative_prompt = Some(prompt.into());
        self
    }

    /// Set an input image for image-to-video generation.
    pub fn with_image(mut self, image: VideoInput) -> Self {
        self.image = Some(image);
        self
    }

    /// Set an input video for video-to-video generation.
    pub fn with_input_video(mut self, video: VideoInput) -> Self {
        self.input_video = Some(video);
        self
    }

    /// Set the video duration in seconds.
    pub fn with_duration(mut self, duration: u32) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set the aspect ratio (e.g., "16:9", "9:16", "1:1").
    pub fn with_aspect_ratio(mut self, aspect_ratio: impl Into<String>) -> Self {
        self.aspect_ratio = Some(aspect_ratio.into());
        self
    }

    /// Set the video resolution.
    pub fn with_resolution(mut self, resolution: VideoResolution) -> Self {
        self.resolution = Some(resolution);
        self
    }

    /// Set the frames per second.
    pub fn with_fps(mut self, fps: u32) -> Self {
        self.fps = Some(fps);
        self
    }

    /// Set a seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the motion amount (0.0 to 1.0).
    pub fn with_motion_amount(mut self, amount: f32) -> Self {
        self.motion_amount = Some(amount.clamp(0.0, 1.0));
        self
    }

    /// Set the camera motion type.
    pub fn with_camera_motion(mut self, motion: CameraMotion) -> Self {
        self.camera_motion = Some(motion);
        self
    }
}

/// Input source for video generation.
#[derive(Debug, Clone)]
pub enum VideoInput {
    /// Path to a local file.
    File(PathBuf),
    /// Binary data in memory.
    Bytes {
        data: Vec<u8>,
        filename: String,
        media_type: String,
    },
    /// URL to a file.
    Url(String),
    /// Base64-encoded data.
    Base64 { data: String, media_type: String },
}

impl VideoInput {
    /// Create an input from a file path.
    pub fn file(path: impl Into<PathBuf>) -> Self {
        VideoInput::File(path.into())
    }

    /// Create an input from bytes.
    pub fn bytes(
        data: Vec<u8>,
        filename: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Self {
        VideoInput::Bytes {
            data,
            filename: filename.into(),
            media_type: media_type.into(),
        }
    }

    /// Create an input from a URL.
    pub fn url(url: impl Into<String>) -> Self {
        VideoInput::Url(url.into())
    }

    /// Create an input from base64-encoded data.
    pub fn base64(data: impl Into<String>, media_type: impl Into<String>) -> Self {
        VideoInput::Base64 {
            data: data.into(),
            media_type: media_type.into(),
        }
    }
}

/// Video resolution presets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VideoResolution {
    /// 480p (SD)
    Sd480,
    /// 720p (HD)
    Hd720,
    /// 1080p (Full HD)
    #[default]
    Hd1080,
    /// 4K (Ultra HD)
    Uhd4k,
    /// Custom resolution
    Custom { width: u32, height: u32 },
}

impl VideoResolution {
    /// Get width and height as a tuple.
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            VideoResolution::Sd480 => (854, 480),
            VideoResolution::Hd720 => (1280, 720),
            VideoResolution::Hd1080 => (1920, 1080),
            VideoResolution::Uhd4k => (3840, 2160),
            VideoResolution::Custom { width, height } => (*width, *height),
        }
    }
}

/// Camera motion types for video generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraMotion {
    /// No camera movement.
    Static,
    /// Camera zooms in.
    ZoomIn,
    /// Camera zooms out.
    ZoomOut,
    /// Camera pans left.
    PanLeft,
    /// Camera pans right.
    PanRight,
    /// Camera tilts up.
    TiltUp,
    /// Camera tilts down.
    TiltDown,
    /// Camera moves forward.
    DollyIn,
    /// Camera moves backward.
    DollyOut,
    /// Camera orbits around subject.
    Orbit,
    /// Dynamic/automatic camera movement.
    Dynamic,
}

/// Response from starting a video generation job.
#[derive(Debug, Clone)]
pub struct VideoGenerationResponse {
    /// Unique job identifier for polling status.
    pub job_id: String,
    /// Initial job status.
    pub status: VideoJobStatus,
    /// Estimated time to completion (if known).
    pub estimated_duration: Option<Duration>,
}

impl VideoGenerationResponse {
    /// Create a new video generation response.
    pub fn new(job_id: impl Into<String>, status: VideoJobStatus) -> Self {
        Self {
            job_id: job_id.into(),
            status,
            estimated_duration: None,
        }
    }

    /// Set the estimated duration.
    pub fn with_estimated_duration(mut self, duration: Duration) -> Self {
        self.estimated_duration = Some(duration);
        self
    }
}

/// Status of a video generation job.
#[derive(Debug, Clone)]
pub enum VideoJobStatus {
    /// Job is queued and waiting to start.
    Queued,
    /// Job is currently being processed.
    Processing {
        /// Progress percentage (0-100), if available.
        progress: Option<u8>,
        /// Current processing stage.
        stage: Option<String>,
    },
    /// Job completed successfully.
    Completed {
        /// URL to the generated video.
        video_url: String,
        /// Duration of the generated video in seconds.
        duration_seconds: Option<f32>,
        /// Thumbnail URL (if available).
        thumbnail_url: Option<String>,
    },
    /// Job failed with an error.
    Failed {
        /// Error message.
        error: String,
        /// Error code (if available).
        code: Option<String>,
    },
    /// Job was cancelled.
    Cancelled,
}

impl VideoJobStatus {
    /// Check if the job is still in progress.
    pub fn is_pending(&self) -> bool {
        matches!(
            self,
            VideoJobStatus::Queued | VideoJobStatus::Processing { .. }
        )
    }

    /// Check if the job is complete (success or failure).
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            VideoJobStatus::Completed { .. }
                | VideoJobStatus::Failed { .. }
                | VideoJobStatus::Cancelled
        )
    }

    /// Get the video URL if completed.
    pub fn video_url(&self) -> Option<&str> {
        match self {
            VideoJobStatus::Completed { video_url, .. } => Some(video_url),
            _ => None,
        }
    }

    /// Get the error message if failed.
    pub fn error(&self) -> Option<&str> {
        match self {
            VideoJobStatus::Failed { error, .. } => Some(error),
            _ => None,
        }
    }
}

// ============================================================================
// Video Provider Trait
// ============================================================================

/// Trait for providers that support video generation.
#[async_trait]
pub trait VideoProvider: Send + Sync {
    /// Get the provider name.
    fn name(&self) -> &str;

    /// Start a video generation job.
    ///
    /// Returns a job ID that can be used to poll for status.
    async fn generate_video(
        &self,
        request: VideoGenerationRequest,
    ) -> Result<VideoGenerationResponse>;

    /// Get the status of a video generation job.
    async fn get_video_status(&self, job_id: &str) -> Result<VideoJobStatus>;

    /// Cancel a video generation job.
    async fn cancel_video(&self, _job_id: &str) -> Result<()> {
        Err(Error::not_supported("Video cancellation"))
    }

    /// Wait for a video job to complete with polling.
    async fn wait_for_video(
        &self,
        job_id: &str,
        poll_interval: Duration,
        timeout: Duration,
    ) -> Result<VideoJobStatus> {
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                return Err(Error::Timeout);
            }

            let status = self.get_video_status(job_id).await?;

            if status.is_terminal() {
                return Ok(status);
            }

            tokio::time::sleep(poll_interval).await;
        }
    }

    /// Get supported aspect ratios for this provider.
    fn supported_aspect_ratios(&self) -> &[&str] {
        &["16:9", "9:16", "1:1"]
    }

    /// Get supported video durations in seconds.
    fn supported_durations(&self) -> &[u32] {
        &[4, 6, 8]
    }

    /// Get maximum video duration in seconds.
    fn max_duration(&self) -> u32 {
        10
    }

    /// Get the default model for this provider.
    fn default_video_model(&self) -> Option<&str> {
        None
    }

    /// Check if the provider supports image-to-video generation.
    fn supports_image_to_video(&self) -> bool {
        true
    }

    /// Check if the provider supports video-to-video generation.
    fn supports_video_to_video(&self) -> bool {
        false
    }
}

// ============================================================================
// Video Model Registry
// ============================================================================

/// Information about a video generation model.
#[derive(Debug, Clone)]
pub struct VideoModelInfo {
    /// Model ID/name.
    pub id: &'static str,
    /// Provider that offers this model.
    pub provider: &'static str,
    /// Maximum video duration in seconds.
    pub max_duration: u32,
    /// Supported aspect ratios.
    pub aspect_ratios: &'static [&'static str],
    /// Supports image-to-video.
    pub supports_i2v: bool,
    /// Supports video-to-video.
    pub supports_v2v: bool,
    /// Approximate price per second of video (USD).
    pub price_per_second: f64,
}

/// Registry of known video generation models.
pub static VIDEO_MODELS: &[VideoModelInfo] = &[
    // Runway
    VideoModelInfo {
        id: "gen-3",
        provider: "runway",
        max_duration: 10,
        aspect_ratios: &["16:9", "9:16", "1:1"],
        supports_i2v: true,
        supports_v2v: false,
        price_per_second: 0.05,
    },
    VideoModelInfo {
        id: "gen-4",
        provider: "runway",
        max_duration: 10,
        aspect_ratios: &["16:9", "9:16", "1:1", "4:5"],
        supports_i2v: true,
        supports_v2v: true,
        price_per_second: 0.10,
    },
    // Pika
    VideoModelInfo {
        id: "pika-1.0",
        provider: "pika",
        max_duration: 4,
        aspect_ratios: &["16:9", "9:16", "1:1"],
        supports_i2v: true,
        supports_v2v: true,
        price_per_second: 0.04,
    },
    // Luma
    VideoModelInfo {
        id: "dream-machine",
        provider: "luma",
        max_duration: 5,
        aspect_ratios: &["16:9", "9:16", "1:1"],
        supports_i2v: true,
        supports_v2v: false,
        price_per_second: 0.05,
    },
    // Kling
    VideoModelInfo {
        id: "kling-2.0",
        provider: "kling",
        max_duration: 10,
        aspect_ratios: &["16:9", "9:16", "1:1"],
        supports_i2v: true,
        supports_v2v: false,
        price_per_second: 0.03,
    },
    // Minimax
    VideoModelInfo {
        id: "hailuo-video",
        provider: "minimax",
        max_duration: 6,
        aspect_ratios: &["16:9", "9:16"],
        supports_i2v: true,
        supports_v2v: false,
        price_per_second: 0.025,
    },
];

/// Get video model info by ID.
pub fn get_video_model_info(model_id: &str) -> Option<&'static VideoModelInfo> {
    VIDEO_MODELS.iter().find(|m| m.id == model_id)
}

/// Get all video models for a specific provider.
pub fn get_video_models_by_provider(provider: &str) -> Vec<&'static VideoModelInfo> {
    VIDEO_MODELS
        .iter()
        .filter(|m| m.provider.eq_ignore_ascii_case(provider))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_request_builder() {
        let request = VideoGenerationRequest::new("gen-3", "A cat playing")
            .with_duration(6)
            .with_aspect_ratio("16:9")
            .with_camera_motion(CameraMotion::ZoomIn)
            .with_motion_amount(0.5);

        assert_eq!(request.model, "gen-3");
        assert_eq!(request.prompt, "A cat playing");
        assert_eq!(request.duration, Some(6));
        assert_eq!(request.aspect_ratio, Some("16:9".to_string()));
        assert_eq!(request.camera_motion, Some(CameraMotion::ZoomIn));
        assert_eq!(request.motion_amount, Some(0.5));
    }

    #[test]
    fn test_motion_amount_clamping() {
        let request = VideoGenerationRequest::new("gen-3", "test").with_motion_amount(2.0);
        assert_eq!(request.motion_amount, Some(1.0));

        let request = VideoGenerationRequest::new("gen-3", "test").with_motion_amount(-0.5);
        assert_eq!(request.motion_amount, Some(0.0));
    }

    #[test]
    fn test_video_resolution() {
        assert_eq!(VideoResolution::Hd1080.dimensions(), (1920, 1080));
        assert_eq!(VideoResolution::Uhd4k.dimensions(), (3840, 2160));
        assert_eq!(
            VideoResolution::Custom {
                width: 1280,
                height: 720
            }
            .dimensions(),
            (1280, 720)
        );
    }

    #[test]
    fn test_video_input() {
        let file_input = VideoInput::file("video.mp4");
        assert!(matches!(file_input, VideoInput::File(_)));

        let url_input = VideoInput::url("https://example.com/video.mp4");
        assert!(matches!(url_input, VideoInput::Url(_)));
    }

    #[test]
    fn test_job_status() {
        let queued = VideoJobStatus::Queued;
        assert!(queued.is_pending());
        assert!(!queued.is_terminal());

        let processing = VideoJobStatus::Processing {
            progress: Some(50),
            stage: Some("rendering".to_string()),
        };
        assert!(processing.is_pending());
        assert!(!processing.is_terminal());

        let completed = VideoJobStatus::Completed {
            video_url: "https://example.com/video.mp4".to_string(),
            duration_seconds: Some(6.0),
            thumbnail_url: None,
        };
        assert!(!completed.is_pending());
        assert!(completed.is_terminal());
        assert_eq!(completed.video_url(), Some("https://example.com/video.mp4"));

        let failed = VideoJobStatus::Failed {
            error: "Generation failed".to_string(),
            code: None,
        };
        assert!(!failed.is_pending());
        assert!(failed.is_terminal());
        assert_eq!(failed.error(), Some("Generation failed"));
    }

    #[test]
    fn test_video_model_registry() {
        let model = get_video_model_info("gen-3");
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.provider, "runway");
        assert!(model.supports_i2v);
    }

    #[test]
    fn test_get_models_by_provider() {
        let runway_models = get_video_models_by_provider("runway");
        assert!(!runway_models.is_empty());
        assert!(runway_models.iter().all(|m| m.provider == "runway"));
    }
}
