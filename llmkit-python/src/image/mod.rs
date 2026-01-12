//! Image generation API bindings for Python.
//!
//! Provides access to image generation from text prompts using
//! various providers (FAL AI, Recraft, RunwayML, Stability AI).

use pyo3::prelude::*;

// ============================================================================
// IMAGE SIZE ENUM
// ============================================================================

/// Image size options.
#[pyclass(name = "ImageSize", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyImageSize {
    /// 256x256 pixels (DALL-E 2 only).
    Square256 = 0,
    /// 512x512 pixels (DALL-E 2 only).
    Square512 = 1,
    /// 1024x1024 pixels (default).
    Square1024 = 2,
    /// 1024x1792 pixels (portrait).
    Portrait1024x1792 = 3,
    /// 1792x1024 pixels (landscape).
    Landscape1792x1024 = 4,
}

#[pymethods]
impl PyImageSize {
    /// Get width and height as a tuple.
    #[pyo3(text_signature = "(self)")]
    pub fn dimensions(&self) -> (u32, u32) {
        match self {
            PyImageSize::Square256 => (256, 256),
            PyImageSize::Square512 => (512, 512),
            PyImageSize::Square1024 => (1024, 1024),
            PyImageSize::Portrait1024x1792 => (1024, 1792),
            PyImageSize::Landscape1792x1024 => (1792, 1024),
        }
    }

    fn __repr__(&self) -> String {
        match self {
            PyImageSize::Square256 => "ImageSize.Square256".to_string(),
            PyImageSize::Square512 => "ImageSize.Square512".to_string(),
            PyImageSize::Square1024 => "ImageSize.Square1024".to_string(),
            PyImageSize::Portrait1024x1792 => "ImageSize.Portrait1024x1792".to_string(),
            PyImageSize::Landscape1792x1024 => "ImageSize.Landscape1792x1024".to_string(),
        }
    }
}

// ============================================================================
// IMAGE QUALITY ENUM
// ============================================================================

/// Image quality options.
#[pyclass(name = "ImageQuality", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyImageQuality {
    /// Standard quality (faster, cheaper).
    Standard = 0,
    /// HD quality (more detail).
    Hd = 1,
}

#[pymethods]
impl PyImageQuality {
    fn __repr__(&self) -> String {
        match self {
            PyImageQuality::Standard => "ImageQuality.Standard".to_string(),
            PyImageQuality::Hd => "ImageQuality.Hd".to_string(),
        }
    }
}

// ============================================================================
// IMAGE STYLE ENUM
// ============================================================================

/// Image style options (DALL-E 3 only).
#[pyclass(name = "ImageStyle", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyImageStyle {
    /// Natural-looking images.
    Natural = 0,
    /// More dramatic, vivid style.
    Vivid = 1,
}

#[pymethods]
impl PyImageStyle {
    fn __repr__(&self) -> String {
        match self {
            PyImageStyle::Natural => "ImageStyle.Natural".to_string(),
            PyImageStyle::Vivid => "ImageStyle.Vivid".to_string(),
        }
    }
}

// ============================================================================
// IMAGE FORMAT ENUM
// ============================================================================

/// Response format for generated images.
#[pyclass(name = "ImageFormat", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyImageFormat {
    /// Return URL to the image.
    Url = 0,
    /// Return base64-encoded image data.
    B64Json = 1,
}

#[pymethods]
impl PyImageFormat {
    fn __repr__(&self) -> String {
        match self {
            PyImageFormat::Url => "ImageFormat.Url".to_string(),
            PyImageFormat::B64Json => "ImageFormat.B64Json".to_string(),
        }
    }
}

// ============================================================================
// IMAGE GENERATION REQUEST
// ============================================================================

/// Request for generating images.
#[pyclass(name = "ImageGenerationRequest")]
pub struct PyImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    pub n: Option<u8>,
    pub size: Option<PyImageSize>,
    pub quality: Option<PyImageQuality>,
    pub style: Option<PyImageStyle>,
    pub response_format: Option<PyImageFormat>,
    pub negative_prompt: Option<String>,
    pub seed: Option<u64>,
}

#[pymethods]
impl PyImageGenerationRequest {
    /// Create a new image generation request.
    #[new]
    pub fn new(model: String, prompt: String) -> Self {
        Self {
            model,
            prompt,
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
    #[pyo3(text_signature = "(self, n)")]
    pub fn with_n(&self, n: u8) -> Self {
        Self {
            n: Some(n),
            ..(*self).clone()
        }
    }

    /// Set the image size.
    #[pyo3(text_signature = "(self, size)")]
    pub fn with_size(&self, size: PyImageSize) -> Self {
        Self {
            size: Some(size),
            ..(*self).clone()
        }
    }

    /// Set the image quality.
    #[pyo3(text_signature = "(self, quality)")]
    pub fn with_quality(&self, quality: PyImageQuality) -> Self {
        Self {
            quality: Some(quality),
            ..(*self).clone()
        }
    }

    /// Set the image style.
    #[pyo3(text_signature = "(self, style)")]
    pub fn with_style(&self, style: PyImageStyle) -> Self {
        Self {
            style: Some(style),
            ..(*self).clone()
        }
    }

    /// Set the response format.
    #[pyo3(text_signature = "(self, response_format)")]
    pub fn with_format(&self, response_format: PyImageFormat) -> Self {
        Self {
            response_format: Some(response_format),
            ..(*self).clone()
        }
    }

    /// Set a negative prompt.
    #[pyo3(text_signature = "(self, negative_prompt)")]
    pub fn with_negative_prompt(&self, negative_prompt: String) -> Self {
        Self {
            negative_prompt: Some(negative_prompt),
            ..(*self).clone()
        }
    }

    /// Set the seed for reproducibility.
    #[pyo3(text_signature = "(self, seed)")]
    pub fn with_seed(&self, seed: u64) -> Self {
        Self {
            seed: Some(seed),
            ..(*self).clone()
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageGenerationRequest(model='{}', prompt='{}', n={:?}, size={:?}, quality={:?}, style={:?})",
            self.model,
            if self.prompt.len() > 50 {
                format!("{}...", &self.prompt[..50])
            } else {
                self.prompt.clone()
            },
            self.n,
            self.size,
            self.quality,
            self.style
        )
    }
}

impl Clone for PyImageGenerationRequest {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            prompt: self.prompt.clone(),
            n: self.n,
            size: self.size,
            quality: self.quality,
            style: self.style,
            response_format: self.response_format,
            negative_prompt: self.negative_prompt.clone(),
            seed: self.seed,
        }
    }
}

// ============================================================================
// GENERATED IMAGE
// ============================================================================

/// A generated image.
#[pyclass(name = "GeneratedImage")]
pub struct PyGeneratedImage {
    pub url: Option<String>,
    pub b64_json: Option<String>,
    pub revised_prompt: Option<String>,
}

#[pymethods]
impl PyGeneratedImage {
    /// Create a new generated image from URL.
    #[staticmethod]
    pub fn from_url(url: String) -> Self {
        Self {
            url: Some(url),
            b64_json: None,
            revised_prompt: None,
        }
    }

    /// Create a new generated image from base64 data.
    #[staticmethod]
    pub fn from_b64(data: String) -> Self {
        Self {
            url: None,
            b64_json: Some(data),
            revised_prompt: None,
        }
    }

    /// Set the revised prompt.
    #[pyo3(text_signature = "(self, revised_prompt)")]
    pub fn with_revised_prompt(&self, revised_prompt: String) -> Self {
        Self {
            revised_prompt: Some(revised_prompt),
            ..(*self).clone()
        }
    }

    /// URL of the generated image.
    #[getter]
    pub fn url(&self) -> Option<String> {
        self.url.clone()
    }

    /// Base64-encoded image data.
    #[getter]
    pub fn b64_json(&self) -> Option<String> {
        self.b64_json.clone()
    }

    /// The revised prompt used by the model.
    #[getter]
    pub fn revised_prompt(&self) -> Option<String> {
        self.revised_prompt.clone()
    }

    /// Get the size of the image data if available (b64_json).
    #[getter]
    pub fn size(&self) -> usize {
        self.b64_json.as_ref().map(|b| b.len()).unwrap_or(0)
    }

    fn __repr__(&self) -> String {
        format!(
            "GeneratedImage(url={}, b64_size={}, revised_prompt={})",
            self.url.is_some(),
            self.size(),
            self.revised_prompt.is_some()
        )
    }
}

impl Clone for PyGeneratedImage {
    fn clone(&self) -> Self {
        Self {
            url: self.url.clone(),
            b64_json: self.b64_json.clone(),
            revised_prompt: self.revised_prompt.clone(),
        }
    }
}

// ============================================================================
// IMAGE GENERATION RESPONSE
// ============================================================================

/// Response from an image generation request.
#[pyclass(name = "ImageGenerationResponse")]
pub struct PyImageGenerationResponse {
    pub created: u64,
    pub images: Vec<PyGeneratedImage>,
}

#[pymethods]
impl PyImageGenerationResponse {
    /// Create a new image generation response.
    #[new]
    pub fn new() -> Self {
        Self {
            created: 0,
            images: Vec::new(),
        }
    }

    /// Get the first image (convenience for n=1 requests).
    #[pyo3(text_signature = "(self)")]
    pub fn first(&self) -> Option<PyGeneratedImage> {
        self.images.first().cloned()
    }

    /// Timestamp of when the images were created.
    #[getter]
    pub fn created(&self) -> u64 {
        self.created
    }

    #[setter]
    pub fn set_created(&mut self, value: u64) {
        self.created = value;
    }

    /// List of generated images.
    #[getter]
    pub fn images(&self) -> Vec<PyGeneratedImage> {
        self.images.clone()
    }

    #[setter]
    pub fn set_images(&mut self, value: Vec<PyGeneratedImage>) {
        self.images = value;
    }

    /// Get the number of images generated.
    #[getter]
    pub fn count(&self) -> usize {
        self.images.len()
    }

    /// Get total size of all image data (for b64 images).
    #[getter]
    pub fn total_size(&self) -> usize {
        self.images.iter().map(|img| img.size()).sum()
    }

    fn __repr__(&self) -> String {
        format!(
            "ImageGenerationResponse(created={}, count={}, total_size={})",
            self.created,
            self.count(),
            self.total_size()
        )
    }
}

impl Clone for PyImageGenerationResponse {
    fn clone(&self) -> Self {
        Self {
            created: self.created,
            images: self.images.clone(),
        }
    }
}
