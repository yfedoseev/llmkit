//! Video API bindings for Node.js/TypeScript.
//!
//! Provides access to ModelSuite video functionality for video generation
//! from text prompts using various providers (Runware, DiffusionRouter).

use napi_derive::napi;

// ============================================================================
// VIDEO GENERATION MODELS
// ============================================================================

/// Video generation models supported by Runware.
#[napi]
pub enum JsVideoModel {
    /// RunwayML Gen-4.5
    RunwayGen45,
    /// Kling Video Generation
    Kling20,
    /// Pika 1.0
    Pika10,
    /// Hailuo Mini Video
    HailuoMini,
    /// Leonardo Diffusion Ultra
    LeonardoUltra,
}

/// Options for video generation.
#[napi]
pub struct JsVideoGenerationOptions {
    pub model: Option<String>,
    pub duration: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub quality: Option<String>,
}

#[napi]
impl JsVideoGenerationOptions {
    /// Create a new VideoGenerationOptions with defaults.
    #[napi(constructor)]
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
    #[napi]
    pub fn with_model(&self, model: String) -> Self {
        Self {
            model: Some(model),
            ..(*self).clone()
        }
    }

    /// Set the video duration in seconds.
    #[napi]
    pub fn with_duration(&self, duration: u32) -> Self {
        Self {
            duration: Some(duration),
            ..(*self).clone()
        }
    }

    /// Set the video width in pixels.
    #[napi]
    pub fn with_width(&self, width: u32) -> Self {
        Self {
            width: Some(width),
            ..(*self).clone()
        }
    }

    /// Set the video height in pixels.
    #[napi]
    pub fn with_height(&self, height: u32) -> Self {
        Self {
            height: Some(height),
            ..(*self).clone()
        }
    }

    /// Set the video quality.
    #[napi]
    pub fn with_quality(&self, quality: String) -> Self {
        Self {
            quality: Some(quality),
            ..(*self).clone()
        }
    }
}

impl Default for JsVideoGenerationOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for JsVideoGenerationOptions {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            duration: self.duration,
            width: self.width,
            height: self.height,
            quality: self.quality.clone(),
        }
    }
}

/// Response from a video generation request.
#[napi]
pub struct JsVideoGenerationResponse {
    pub video_bytes: Option<Vec<u8>>,
    pub video_url: Option<String>,
    pub format: String,
    pub duration: Option<f64>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub task_id: Option<String>,
    pub status: Option<String>,
}

#[napi]
impl JsVideoGenerationResponse {
    /// Size of the video in bytes (if available).
    #[napi(getter)]
    pub fn size(&self) -> u32 {
        self.video_bytes
            .as_ref()
            .map(|b| b.len() as u32)
            .unwrap_or(0)
    }
}

// ============================================================================
// REQUEST WRAPPER TYPES (for client method signatures)
// ============================================================================

/// Request for video generation.
#[napi]
pub struct JsVideoGenerationRequest {
    pub prompt: String,
    pub model: Option<String>,
    pub duration: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

#[napi]
impl JsVideoGenerationRequest {
    /// Create a new video generation request.
    #[napi(constructor)]
    pub fn new(prompt: String) -> Self {
        Self {
            prompt,
            model: None,
            duration: None,
            width: None,
            height: None,
        }
    }

    #[napi]
    pub fn with_model(&self, model: String) -> Self {
        Self {
            model: Some(model),
            ..(*self).clone()
        }
    }

    #[napi]
    pub fn with_duration(&self, duration: u32) -> Self {
        Self {
            duration: Some(duration),
            ..(*self).clone()
        }
    }

    #[napi]
    pub fn with_width(&self, width: u32) -> Self {
        Self {
            width: Some(width),
            ..(*self).clone()
        }
    }

    #[napi]
    pub fn with_height(&self, height: u32) -> Self {
        Self {
            height: Some(height),
            ..(*self).clone()
        }
    }
}

impl Clone for JsVideoGenerationRequest {
    fn clone(&self) -> Self {
        Self {
            prompt: self.prompt.clone(),
            model: self.model.clone(),
            duration: self.duration,
            width: self.width,
            height: self.height,
        }
    }
}
