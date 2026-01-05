//! Video API bindings for Python.
//!
//! Provides access to ModelSuite video functionality for video generation
//! from text prompts using various providers (Runware, DiffusionRouter).

use pyo3::prelude::*;

// ============================================================================
// VIDEO GENERATION MODELS
// ============================================================================

/// Video generation models supported by Runware.
#[pyclass(name = "VideoModel", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum PyVideoModel {
    /// RunwayML Gen-4.5
    #[default]
    RunwayGen45 = 0,
    /// Kling Video Generation
    Kling20 = 1,
    /// Pika 1.0
    Pika10 = 2,
    /// Hailuo Mini Video
    HailuoMini = 3,
    /// Leonardo Diffusion Ultra
    LeonardoUltra = 4,
}

/// Options for video generation.
#[pyclass(name = "VideoGenerationOptions")]
#[derive(Clone)]
pub struct PyVideoGenerationOptions {
    pub model: Option<String>,
    pub duration: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub quality: Option<String>,
}

#[pymethods]
impl PyVideoGenerationOptions {
    /// Create a new VideoGenerationOptions with defaults.
    #[new]
    pub fn new() -> Self {
        Self {
            model: None,
            duration: None,
            width: None,
            height: None,
            quality: None,
        }
    }

    /// Set the video model to use.
    ///
    /// Args:
    ///     model: Model name (e.g., "runway-gen-4.5", "kling-2.0")
    ///
    /// Returns:
    ///     Self for method chaining.
    pub fn with_model(&self, model: String) -> Self {
        let mut opts = self.clone();
        opts.model = Some(model);
        opts
    }

    /// Set the video duration in seconds.
    pub fn with_duration(&self, duration: u32) -> Self {
        let mut opts = self.clone();
        opts.duration = Some(duration);
        opts
    }

    /// Set the video width in pixels.
    pub fn with_width(&self, width: u32) -> Self {
        let mut opts = self.clone();
        opts.width = Some(width);
        opts
    }

    /// Set the video height in pixels.
    pub fn with_height(&self, height: u32) -> Self {
        let mut opts = self.clone();
        opts.height = Some(height);
        opts
    }

    /// Set the video quality.
    ///
    /// Args:
    ///     quality: Quality level (e.g., "standard", "high", "premium")
    pub fn with_quality(&self, quality: String) -> Self {
        let mut opts = self.clone();
        opts.quality = Some(quality);
        opts
    }

    #[getter]
    pub fn model(&self) -> Option<String> {
        self.model.clone()
    }

    #[getter]
    pub fn duration(&self) -> Option<u32> {
        self.duration
    }

    #[getter]
    pub fn width(&self) -> Option<u32> {
        self.width
    }

    #[getter]
    pub fn height(&self) -> Option<u32> {
        self.height
    }

    #[getter]
    pub fn quality(&self) -> Option<String> {
        self.quality.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoGenerationOptions(model={:?}, duration={:?}, {}x{}, quality={:?})",
            self.model,
            self.duration,
            self.width.unwrap_or(0),
            self.height.unwrap_or(0),
            self.quality
        )
    }
}

/// Response from a video generation request.
#[pyclass(name = "VideoGenerationResponse")]
#[derive(Clone)]
pub struct PyVideoGenerationResponse {
    pub video_bytes: Option<Vec<u8>>,
    pub video_url: Option<String>,
    pub format: String,
    pub duration: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub task_id: Option<String>,
    pub status: Option<String>,
}

#[pymethods]
impl PyVideoGenerationResponse {
    /// The generated video data as bytes (if available).
    #[getter]
    pub fn video_bytes(&self) -> Option<Vec<u8>> {
        self.video_bytes.clone()
    }

    /// URL to download the video (for async tasks).
    #[getter]
    pub fn video_url(&self) -> Option<String> {
        self.video_url.clone()
    }

    /// The video format (e.g., "mp4", "webm").
    #[getter]
    pub fn format(&self) -> &str {
        &self.format
    }

    /// Duration of the video in seconds.
    #[getter]
    pub fn duration(&self) -> Option<f64> {
        self.duration
    }

    /// Video width in pixels.
    #[getter]
    pub fn width(&self) -> Option<u32> {
        self.width
    }

    /// Video height in pixels.
    #[getter]
    pub fn height(&self) -> Option<u32> {
        self.height
    }

    /// Task ID for tracking async video generation.
    #[getter]
    pub fn task_id(&self) -> Option<String> {
        self.task_id.clone()
    }

    /// Current status of video generation.
    #[getter]
    pub fn status(&self) -> Option<String> {
        self.status.clone()
    }

    /// Size of the video in bytes (if available).
    #[getter]
    pub fn size(&self) -> usize {
        self.video_bytes.as_ref().map(|b| b.len()).unwrap_or(0)
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoGenerationResponse(format='{}', duration={:?}, status={:?}, size={})",
            self.format,
            self.duration,
            self.status,
            self.size()
        )
    }
}

// ============================================================================
// REQUEST WRAPPER TYPES (for client method signatures)
// ============================================================================

/// Request for video generation.
#[pyclass(name = "VideoGenerationRequest")]
#[derive(Clone)]
pub struct PyVideoGenerationRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub duration: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

#[pymethods]
impl PyVideoGenerationRequest {
    /// Create a new video generation request.
    ///
    /// Args:
    ///     prompt: Text description of the video to generate (required)
    ///     model: The video generation model to use (optional)
    ///     duration: Video duration in seconds (optional)
    ///     width: Video width in pixels (optional)
    ///     height: Video height in pixels (optional)
    #[new]
    pub fn new(prompt: String) -> Self {
        Self {
            prompt,
            model: None,
            duration: None,
            width: None,
            height: None,
        }
    }

    pub fn with_model(&self, model: String) -> Self {
        let mut req = self.clone();
        req.model = Some(model);
        req
    }

    pub fn with_duration(&self, duration: u32) -> Self {
        let mut req = self.clone();
        req.duration = Some(duration);
        req
    }

    pub fn with_width(&self, width: u32) -> Self {
        let mut req = self.clone();
        req.width = Some(width);
        req
    }

    pub fn with_height(&self, height: u32) -> Self {
        let mut req = self.clone();
        req.height = Some(height);
        req
    }

    #[getter]
    pub fn prompt(&self) -> &str {
        &self.prompt
    }

    #[getter]
    pub fn model(&self) -> Option<String> {
        self.model.clone()
    }

    #[getter]
    pub fn duration(&self) -> Option<u32> {
        self.duration
    }

    #[getter]
    pub fn width(&self) -> Option<u32> {
        self.width
    }

    #[getter]
    pub fn height(&self) -> Option<u32> {
        self.height
    }

    fn __repr__(&self) -> String {
        format!(
            "VideoGenerationRequest(prompt='{}...', model={:?}, duration={:?})",
            &self.prompt[..std::cmp::min(50, self.prompt.len())],
            self.model,
            self.duration
        )
    }
}
