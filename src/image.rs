//! Image generation API for creating images from text prompts.
//!
//! This module provides a unified interface for generating images
//! from various providers including OpenAI (DALL-E), Replicate (Stable Diffusion), and others.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::{ImageProvider, ImageGenerationRequest, ImageSize};
//!
//! // Create provider
//! let provider = OpenAIProvider::from_env()?;
//!
//! // Generate an image
//! let request = ImageGenerationRequest::new(
//!     "dall-e-3",
//!     "A serene mountain landscape at sunset",
//! );
//!
//! let response = provider.generate_image(request).await?;
//! println!("Generated {} images", response.images.len());
//! ```
//!
//! # Async Polling for Long-Running Jobs
//!
//! Some providers (like Replicate) use async job polling:
//!
//! ```ignore
//! use llmkit::{AsyncImageProvider, ImageGenerationRequest};
//!
//! // Start generation
//! let job_id = provider.start_generation(request).await?;
//!
//! // Poll until complete
//! loop {
//!     match provider.poll_status(&job_id).await? {
//!         JobStatus::Completed => break,
//!         JobStatus::Running => tokio::time::sleep(Duration::from_secs(1)).await,
//!         JobStatus::Failed(e) => return Err(e),
//!     }
//! }
//!
//! // Get result
//! let response = provider.get_result(&job_id).await?;
//! ```

use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Request for generating images.
#[derive(Debug, Clone)]
pub struct ImageGenerationRequest {
    /// The prompt to generate images from.
    pub prompt: String,
    /// The model to use for generation.
    pub model: String,
    /// Number of images to generate.
    pub n: Option<u8>,
    /// Size of the generated images.
    pub size: Option<ImageSize>,
    /// Quality of the generated images.
    pub quality: Option<ImageQuality>,
    /// Style of the generated images.
    pub style: Option<ImageStyle>,
    /// Response format (URL or base64).
    pub response_format: Option<ImageFormat>,
    /// Negative prompt (for models that support it).
    pub negative_prompt: Option<String>,
    /// Seed for reproducibility (for models that support it).
    pub seed: Option<u64>,
}

impl ImageGenerationRequest {
    /// Create a new image generation request.
    pub fn new(model: impl Into<String>, prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            model: model.into(),
            n: None,
            size: None,
            quality: None,
            style: None,
            response_format: None,
            negative_prompt: None,
            seed: None,
        }
    }

    /// Set the number of images to generate.
    pub fn with_n(mut self, n: u8) -> Self {
        self.n = Some(n);
        self
    }

    /// Set the image size.
    pub fn with_size(mut self, size: ImageSize) -> Self {
        self.size = Some(size);
        self
    }

    /// Set the image quality.
    pub fn with_quality(mut self, quality: ImageQuality) -> Self {
        self.quality = Some(quality);
        self
    }

    /// Set the image style.
    pub fn with_style(mut self, style: ImageStyle) -> Self {
        self.style = Some(style);
        self
    }

    /// Set the response format.
    pub fn with_format(mut self, format: ImageFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Set a negative prompt.
    pub fn with_negative_prompt(mut self, negative_prompt: impl Into<String>) -> Self {
        self.negative_prompt = Some(negative_prompt.into());
        self
    }

    /// Set the seed for reproducibility.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Image size options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImageSize {
    /// 256x256 pixels (DALL-E 2 only).
    Square256,
    /// 512x512 pixels (DALL-E 2 only).
    Square512,
    /// 1024x1024 pixels.
    Square1024,
    /// 1024x1792 pixels (portrait).
    Portrait1024x1792,
    /// 1792x1024 pixels (landscape).
    Landscape1792x1024,
    /// Custom dimensions.
    Custom { width: u32, height: u32 },
}

impl ImageSize {
    /// Get width and height as a tuple.
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            ImageSize::Square256 => (256, 256),
            ImageSize::Square512 => (512, 512),
            ImageSize::Square1024 => (1024, 1024),
            ImageSize::Portrait1024x1792 => (1024, 1792),
            ImageSize::Landscape1792x1024 => (1792, 1024),
            ImageSize::Custom { width, height } => (*width, *height),
        }
    }

    /// Convert to OpenAI API size string.
    pub fn to_openai_string(&self) -> String {
        let (w, h) = self.dimensions();
        format!("{}x{}", w, h)
    }
}

impl Default for ImageSize {
    fn default() -> Self {
        ImageSize::Square1024
    }
}

/// Image quality options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageQuality {
    /// Standard quality (faster, cheaper).
    #[default]
    Standard,
    /// HD quality (more detail).
    Hd,
}

/// Image style options (DALL-E 3 only).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageStyle {
    /// Natural-looking images.
    #[default]
    Natural,
    /// More dramatic, vivid style.
    Vivid,
}

/// Response format for generated images.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFormat {
    /// Return URL to the image.
    #[default]
    Url,
    /// Return base64-encoded image data.
    B64Json,
}

/// Response from an image generation request.
#[derive(Debug, Clone)]
pub struct ImageGenerationResponse {
    /// Timestamp when the images were created.
    pub created: u64,
    /// The generated images.
    pub images: Vec<GeneratedImage>,
}

impl ImageGenerationResponse {
    /// Get the first image (convenience for n=1 requests).
    pub fn first(&self) -> Option<&GeneratedImage> {
        self.images.first()
    }
}

/// A generated image.
#[derive(Debug, Clone)]
pub struct GeneratedImage {
    /// URL to the generated image (if format is Url).
    pub url: Option<String>,
    /// Base64-encoded image data (if format is B64Json).
    pub b64_json: Option<String>,
    /// The revised prompt used by the model (DALL-E 3 only).
    pub revised_prompt: Option<String>,
}

impl GeneratedImage {
    /// Create a new generated image from URL.
    pub fn from_url(url: impl Into<String>) -> Self {
        Self {
            url: Some(url.into()),
            b64_json: None,
            revised_prompt: None,
        }
    }

    /// Create a new generated image from base64 data.
    pub fn from_b64(data: impl Into<String>) -> Self {
        Self {
            url: None,
            b64_json: Some(data.into()),
            revised_prompt: None,
        }
    }

    /// Set the revised prompt.
    pub fn with_revised_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.revised_prompt = Some(prompt.into());
        self
    }
}

/// Request for editing an existing image.
#[derive(Debug, Clone)]
pub struct ImageEditRequest {
    /// The image to edit.
    pub image: ImageInput,
    /// The prompt describing the edit.
    pub prompt: String,
    /// Mask image defining areas to edit.
    pub mask: Option<ImageInput>,
    /// The model to use.
    pub model: String,
    /// Number of images to generate.
    pub n: Option<u8>,
    /// Size of the output images.
    pub size: Option<ImageSize>,
    /// Response format.
    pub response_format: Option<ImageFormat>,
}

/// Request for creating image variations.
#[derive(Debug, Clone)]
pub struct ImageVariationRequest {
    /// The image to create variations of.
    pub image: ImageInput,
    /// The model to use.
    pub model: String,
    /// Number of variations to generate.
    pub n: Option<u8>,
    /// Size of the output images.
    pub size: Option<ImageSize>,
    /// Response format.
    pub response_format: Option<ImageFormat>,
}

/// Input image source.
#[derive(Debug, Clone)]
pub enum ImageInput {
    /// Path to a local file.
    File(PathBuf),
    /// Base64-encoded image data.
    Base64 { data: String, media_type: String },
    /// URL to an image.
    Url(String),
}

/// Trait for providers that support image generation.
#[async_trait]
pub trait ImageProvider: Send + Sync {
    /// Get the provider name.
    fn name(&self) -> &str;

    /// Generate images from a prompt.
    async fn generate_image(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<ImageGenerationResponse>;

    /// Edit an existing image (optional).
    async fn edit_image(&self, _request: ImageEditRequest) -> Result<ImageGenerationResponse> {
        Err(Error::not_supported("Image editing"))
    }

    /// Create variations of an image (optional).
    async fn create_variation(
        &self,
        _request: ImageVariationRequest,
    ) -> Result<ImageGenerationResponse> {
        Err(Error::not_supported("Image variations"))
    }

    /// Get supported image sizes for this provider.
    fn supported_sizes(&self) -> &[ImageSize];

    /// Get maximum number of images per request.
    fn max_images_per_request(&self) -> u8 {
        4
    }

    /// Get the default model for this provider.
    fn default_image_model(&self) -> Option<&str> {
        None
    }

    /// Get supported image models.
    fn supported_image_models(&self) -> Option<&[&str]> {
        None
    }
}

/// Job ID for async image generation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JobId(pub String);

impl JobId {
    /// Create a new job ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Status of an async image generation job.
#[derive(Debug, Clone)]
pub enum JobStatus {
    /// Job is queued but not yet started.
    Queued,
    /// Job is currently running.
    Running,
    /// Job completed successfully.
    Completed,
    /// Job failed with an error message.
    Failed(String),
    /// Job was cancelled.
    Cancelled,
}

/// Trait for async image providers that use job polling.
#[async_trait]
pub trait AsyncImageProvider: ImageProvider {
    /// Start an image generation job.
    async fn start_generation(&self, request: ImageGenerationRequest) -> Result<JobId>;

    /// Poll the status of a job.
    async fn poll_status(&self, job_id: &JobId) -> Result<JobStatus>;

    /// Get the result of a completed job.
    async fn get_result(&self, job_id: &JobId) -> Result<ImageGenerationResponse>;

    /// Wait for a job to complete with polling.
    async fn wait_for_completion(
        &self,
        job_id: &JobId,
        poll_interval: Duration,
        timeout: Duration,
    ) -> Result<ImageGenerationResponse> {
        let start = std::time::Instant::now();

        loop {
            if start.elapsed() > timeout {
                return Err(Error::Timeout);
            }

            match self.poll_status(job_id).await? {
                JobStatus::Completed => return self.get_result(job_id).await,
                JobStatus::Failed(msg) => return Err(Error::other(msg)),
                JobStatus::Cancelled => return Err(Error::other("Image generation was cancelled")),
                JobStatus::Queued | JobStatus::Running => {
                    tokio::time::sleep(poll_interval).await;
                }
            }
        }
    }
}

/// Information about an image generation model.
#[derive(Debug, Clone)]
pub struct ImageModelInfo {
    /// Model ID/name.
    pub id: &'static str,
    /// Provider that offers this model.
    pub provider: &'static str,
    /// Supported sizes.
    pub sizes: &'static [ImageSize],
    /// Maximum images per request.
    pub max_images: u8,
    /// Whether the model supports image editing.
    pub supports_editing: bool,
    /// Whether the model supports variations.
    pub supports_variations: bool,
    /// Price per image (USD).
    pub price_per_image: f64,
}

/// Registry of known image generation models.
pub static IMAGE_MODELS: &[ImageModelInfo] = &[
    // OpenAI DALL-E
    ImageModelInfo {
        id: "dall-e-3",
        provider: "openai",
        sizes: &[
            ImageSize::Square1024,
            ImageSize::Portrait1024x1792,
            ImageSize::Landscape1792x1024,
        ],
        max_images: 1,
        supports_editing: false,
        supports_variations: false,
        price_per_image: 0.04, // Standard 1024x1024
    },
    ImageModelInfo {
        id: "dall-e-2",
        provider: "openai",
        sizes: &[
            ImageSize::Square256,
            ImageSize::Square512,
            ImageSize::Square1024,
        ],
        max_images: 10,
        supports_editing: true,
        supports_variations: true,
        price_per_image: 0.02, // 1024x1024
    },
];

/// Get image model info by ID.
pub fn get_image_model_info(model_id: &str) -> Option<&'static ImageModelInfo> {
    IMAGE_MODELS.iter().find(|m| m.id == model_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_size_dimensions() {
        assert_eq!(ImageSize::Square1024.dimensions(), (1024, 1024));
        assert_eq!(ImageSize::Portrait1024x1792.dimensions(), (1024, 1792));
        assert_eq!(
            ImageSize::Custom {
                width: 800,
                height: 600
            }
            .dimensions(),
            (800, 600)
        );
    }

    #[test]
    fn test_image_size_to_openai_string() {
        assert_eq!(ImageSize::Square1024.to_openai_string(), "1024x1024");
        assert_eq!(
            ImageSize::Landscape1792x1024.to_openai_string(),
            "1792x1024"
        );
    }

    #[test]
    fn test_image_request_builder() {
        let request = ImageGenerationRequest::new("dall-e-3", "A cat")
            .with_size(ImageSize::Square1024)
            .with_quality(ImageQuality::Hd)
            .with_style(ImageStyle::Vivid)
            .with_n(2);

        assert_eq!(request.model, "dall-e-3");
        assert_eq!(request.prompt, "A cat");
        assert_eq!(request.size, Some(ImageSize::Square1024));
        assert_eq!(request.quality, Some(ImageQuality::Hd));
        assert_eq!(request.style, Some(ImageStyle::Vivid));
        assert_eq!(request.n, Some(2));
    }

    #[test]
    fn test_generated_image() {
        let img = GeneratedImage::from_url("https://example.com/image.png")
            .with_revised_prompt("A cute cat sleeping on a couch");

        assert_eq!(img.url, Some("https://example.com/image.png".to_string()));
        assert!(img.b64_json.is_none());
        assert!(img.revised_prompt.is_some());
    }

    #[test]
    fn test_job_id() {
        let job_id = JobId::new("job-123");
        assert_eq!(job_id.to_string(), "job-123");
    }

    #[test]
    fn test_image_model_registry() {
        let model = get_image_model_info("dall-e-3");
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.provider, "openai");
        assert_eq!(model.max_images, 1);
    }
}
