//! Image generation API bindings for Node.js/TypeScript.
//!
//! Provides access to image generation from text prompts using
//! various providers (FAL AI, Recraft, RunwayML, Stability AI).

use napi::bindgen_prelude::*;
use napi_derive::napi;

// ============================================================================
// IMAGE SIZE ENUM
// ============================================================================

/// Image size options.
#[napi]
#[derive(Clone, Copy)]
pub enum JsImageSize {
    /// 256x256 pixels (DALL-E 2 only).
    Square256,
    /// 512x512 pixels (DALL-E 2 only).
    Square512,
    /// 1024x1024 pixels (default).
    Square1024,
    /// 1024x1792 pixels (portrait).
    Portrait1024x1792,
    /// 1792x1024 pixels (landscape).
    Landscape1792x1024,
}

/// Image quality options.
#[napi]
#[derive(Clone, Copy)]
pub enum JsImageQuality {
    /// Standard quality (faster, cheaper).
    Standard,
    /// HD quality (more detail).
    Hd,
}

/// Image style options (DALL-E 3 only).
#[napi]
#[derive(Clone, Copy)]
pub enum JsImageStyle {
    /// Natural-looking images.
    Natural,
    /// More dramatic, vivid style.
    Vivid,
}

/// Response format for generated images.
#[napi]
#[derive(Clone, Copy)]
pub enum JsImageFormat {
    /// Return URL to the image.
    Url,
    /// Return base64-encoded image data.
    B64Json,
}

// ============================================================================
// IMAGE GENERATION REQUEST
// ============================================================================

/// Request for generating images.
#[napi]
pub struct JsImageGenerationRequest {
    pub model: String,
    pub prompt: String,
    pub n: Option<u8>,
    pub size: Option<JsImageSize>,
    pub quality: Option<JsImageQuality>,
    pub style: Option<JsImageStyle>,
    pub response_format: Option<JsImageFormat>,
    pub negative_prompt: Option<String>,
    pub seed: Option<i64>,
}

#[napi]
impl JsImageGenerationRequest {
    /// Create a new image generation request.
    #[napi(constructor)]
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
    #[napi]
    pub fn with_n(&self, n: u8) -> Self {
        Self {
            n: Some(n),
            ..(*self).clone()
        }
    }

    /// Set the image size.
    #[napi]
    pub fn with_size(&self, size: JsImageSize) -> Self {
        Self {
            size: Some(size),
            ..(*self).clone()
        }
    }

    /// Set the image quality.
    #[napi]
    pub fn with_quality(&self, quality: JsImageQuality) -> Self {
        Self {
            quality: Some(quality),
            ..(*self).clone()
        }
    }

    /// Set the image style.
    #[napi]
    pub fn with_style(&self, style: JsImageStyle) -> Self {
        Self {
            style: Some(style),
            ..(*self).clone()
        }
    }

    /// Set the response format.
    #[napi]
    pub fn with_format(&self, response_format: JsImageFormat) -> Self {
        Self {
            response_format: Some(response_format),
            ..(*self).clone()
        }
    }

    /// Set a negative prompt.
    #[napi]
    pub fn with_negative_prompt(&self, negative_prompt: String) -> Self {
        Self {
            negative_prompt: Some(negative_prompt),
            ..(*self).clone()
        }
    }

    /// Set the seed for reproducibility.
    #[napi]
    pub fn with_seed(&self, seed: i64) -> Self {
        Self {
            seed: Some(seed),
            ..(*self).clone()
        }
    }
}

impl Clone for JsImageGenerationRequest {
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
#[napi]
pub struct JsGeneratedImage {
    pub url: Option<String>,
    pub b64_json: Option<String>,
    pub revised_prompt: Option<String>,
}

#[napi]
impl JsGeneratedImage {
    /// Create a new generated image from URL.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            url: None,
            b64_json: None,
            revised_prompt: None,
        }
    }

    /// Create a new generated image from URL.
    #[napi(js_name = "fromUrl")]
    pub fn from_url(url: String) -> Self {
        Self {
            url: Some(url),
            b64_json: None,
            revised_prompt: None,
        }
    }

    /// Create a new generated image from base64 data.
    #[napi(js_name = "fromB64")]
    pub fn from_b64(data: String) -> Self {
        Self {
            url: None,
            b64_json: Some(data),
            revised_prompt: None,
        }
    }

    /// Set the revised prompt.
    #[napi(js_name = "withRevisedPrompt")]
    pub fn with_revised_prompt(&self, revised_prompt: String) -> Self {
        Self {
            revised_prompt: Some(revised_prompt),
            ..(*self).clone()
        }
    }

    /// Get the size of the image data if available (b64_json).
    #[napi(getter)]
    pub fn size(&self) -> u32 {
        self.b64_json.as_ref().map(|b| b.len() as u32).unwrap_or(0)
    }
}

impl Default for JsGeneratedImage {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for JsGeneratedImage {
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
#[napi(object)]
#[derive(Clone)]
pub struct JsImageGenerationResponse {
    pub created: i64,
    pub images: Vec<JsGeneratedImage>,
}

// Manual FromNapiValue implementations for component types
// These are stubs since they're only created by Rust, never deserialized from JS
impl FromNapiValue for JsGeneratedImage {
    unsafe fn from_napi_value(
        _env: napi::sys::napi_env,
        _val: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        Err(napi::Error::new(
            napi::Status::InvalidArg,
            "JsGeneratedImage cannot be constructed from JavaScript",
        ))
    }
}
