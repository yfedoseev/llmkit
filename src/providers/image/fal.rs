//! Fal AI API provider implementation.
//!
//! This module provides access to Fal AI's inference platform, including
//! image generation, LLM inference, and other AI models.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::providers::FalProvider;
//!
//! // From environment variable
//! let provider = FalProvider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = FalProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! ## LLM Models
//! - `fal-ai/llavav15-13b` - LLaVA vision-language model
//! - `fal-ai/any-llm` - Router to various LLMs
//!
//! ## Image Generation
//! - `fal-ai/flux/schnell` - Fast FLUX image generation
//! - `fal-ai/flux/dev` - Development FLUX model
//! - `fal-ai/stable-diffusion-v3` - Stable Diffusion 3
//!
//! # Environment Variables
//!
//! - `FAL_KEY` - Your Fal AI API key

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

const FAL_API_URL: &str = "https://fal.run";

/// Fal AI API provider.
///
/// Provides access to Fal AI's inference platform for LLMs and image generation.
pub struct FalProvider {
    config: ProviderConfig,
    client: Client,
}

impl FalProvider {
    /// Create a new Fal AI provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Key {}", key)
                    .parse()
                    .map_err(|_| Error::config("Invalid API key format"))?,
            );
        }

        headers.insert(
            reqwest::header::CONTENT_TYPE,
            "application/json".parse().unwrap(),
        );

        let client = Client::builder()
            .timeout(config.timeout)
            .default_headers(headers)
            .build()?;

        Ok(Self { config, client })
    }

    /// Create a new Fal AI provider from environment variable.
    ///
    /// Reads the API key from `FAL_KEY`.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("FAL_KEY");
        Self::new(config)
    }

    /// Create a new Fal AI provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn model_url(&self, model: &str) -> String {
        format!(
            "{}/{}",
            self.config.base_url.as_deref().unwrap_or(FAL_API_URL),
            model
        )
    }

    /// Check if a model is an image generation model.
    fn is_image_model(model: &str) -> bool {
        model.contains("flux")
            || model.contains("stable-diffusion")
            || model.contains("sdxl")
            || model.contains("kandinsky")
            || model.contains("midjourney")
    }

    /// Generate an image using Fal AI.
    pub async fn generate_image(&self, model: &str, prompt: &str) -> Result<FalImageResponse> {
        let request = FalImageRequest {
            prompt: prompt.to_string(),
            image_size: Some("landscape_4_3".to_string()),
            num_inference_steps: Some(4),
            num_images: Some(1),
            enable_safety_checker: Some(true),
        };

        let response = self
            .client
            .post(self.model_url(model))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Fal AI API error {}: {}", status, error_text),
            ));
        }

        let api_response: FalImageResponse = response.json().await?;
        Ok(api_response)
    }

    /// Run LLM inference using Fal AI.
    async fn llm_inference(&self, model: &str, request: &CompletionRequest) -> Result<String> {
        // Extract messages
        let messages: Vec<FalMessage> = request
            .messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::System => "system",
                };
                let content = m
                    .content
                    .iter()
                    .filter_map(|b| {
                        if let ContentBlock::Text { text } = b {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                FalMessage {
                    role: role.to_string(),
                    content,
                }
            })
            .collect();

        let fal_request = FalLlmRequest {
            messages,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
        };

        let response = self
            .client
            .post(self.model_url(model))
            .json(&fal_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Fal AI API error {}: {}", status, error_text),
            ));
        }

        let api_response: FalLlmResponse = response.json().await?;
        Ok(api_response.output.unwrap_or_default())
    }
}

#[async_trait]
impl Provider for FalProvider {
    fn name(&self) -> &str {
        "fal"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        // Check if this is an image generation request
        if Self::is_image_model(&request.model) {
            // Extract prompt from last user message
            let prompt = request
                .messages
                .iter()
                .rfind(|m| matches!(m.role, Role::User))
                .and_then(|m| {
                    m.content.iter().find_map(|block| {
                        if let ContentBlock::Text { text } = block {
                            Some(text.clone())
                        } else {
                            None
                        }
                    })
                })
                .ok_or_else(|| Error::invalid_request("No prompt found for image generation"))?;

            let result = self.generate_image(&request.model, &prompt).await?;

            // Format result with image URLs
            let content = if let Some(images) = result.images {
                let urls: Vec<String> = images.into_iter().map(|img| img.url).collect();
                format!("Generated {} image(s):\n{}", urls.len(), urls.join("\n"))
            } else {
                "Image generation completed but no images returned".to_string()
            };

            return Ok(CompletionResponse {
                id: uuid::Uuid::new_v4().to_string(),
                model: request.model,
                content: vec![ContentBlock::Text { text: content }],
                stop_reason: StopReason::EndTurn,
                usage: Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            });
        }

        // LLM inference
        let output = self.llm_inference(&request.model, &request).await?;

        Ok(CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: request.model,
            content: vec![ContentBlock::Text { text: output }],
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        })
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // Fal AI streaming is model-specific, fall back to regular completion
        let response = self.complete(request).await?;

        let stream = async_stream::try_stream! {
            yield StreamChunk {
                event_type: StreamEventType::ContentBlockStart,
                index: Some(0),
                delta: None,
                stop_reason: None,
                usage: None,
            };

            for block in response.content {
                if let ContentBlock::Text { text } = block {
                    yield StreamChunk {
                        event_type: StreamEventType::ContentBlockDelta,
                        index: Some(0),
                        delta: Some(ContentDelta::Text { text }),
                        stop_reason: None,
                        usage: None,
                    };
                }
            }

            yield StreamChunk {
                event_type: StreamEventType::MessageStop,
                index: None,
                delta: None,
                stop_reason: Some(StopReason::EndTurn),
                usage: None,
            };
        };

        Ok(Box::pin(stream))
    }
}

// Fal AI API types

#[derive(Debug, Serialize)]
struct FalImageRequest {
    prompt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_size: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_inference_steps: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_images: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    enable_safety_checker: Option<bool>,
}

/// Response from Fal AI image generation.
#[derive(Debug, Deserialize)]
pub struct FalImageResponse {
    /// Generated images.
    pub images: Option<Vec<FalImage>>,
    /// Timing information.
    pub timings: Option<Value>,
    /// Seed used for generation.
    pub seed: Option<u64>,
}

/// A generated image.
#[derive(Debug, Deserialize)]
pub struct FalImage {
    /// URL to the generated image.
    pub url: String,
    /// Width of the image.
    pub width: Option<u32>,
    /// Height of the image.
    pub height: Option<u32>,
    /// Content type.
    pub content_type: Option<String>,
}

#[derive(Debug, Serialize)]
struct FalLlmRequest {
    messages: Vec<FalMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
}

#[derive(Debug, Serialize)]
struct FalMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct FalLlmResponse {
    output: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = FalProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.name(), "fal");
    }

    #[test]
    fn test_provider_with_api_key() {
        let provider = FalProvider::with_api_key("test-key").unwrap();
        assert_eq!(provider.name(), "fal");
    }

    #[test]
    fn test_model_url() {
        let provider = FalProvider::new(ProviderConfig::new("test-key")).unwrap();
        let url = provider.model_url("fal-ai/flux/schnell");
        assert_eq!(url, "https://fal.run/fal-ai/flux/schnell");
    }

    #[test]
    fn test_model_url_custom_base() {
        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.fal.ai".to_string());
        let provider = FalProvider::new(config).unwrap();
        let url = provider.model_url("fal-ai/flux/schnell");
        assert_eq!(url, "https://custom.fal.ai/fal-ai/flux/schnell");
    }

    #[test]
    fn test_is_image_model() {
        // FLUX models
        assert!(FalProvider::is_image_model("fal-ai/flux/schnell"));
        assert!(FalProvider::is_image_model("fal-ai/flux/dev"));
        assert!(FalProvider::is_image_model("fal-ai/flux-pro"));

        // Stable Diffusion models
        assert!(FalProvider::is_image_model("fal-ai/stable-diffusion-v3"));
        assert!(FalProvider::is_image_model("fal-ai/sdxl"));

        // Other image models
        assert!(FalProvider::is_image_model("fal-ai/kandinsky"));
        assert!(FalProvider::is_image_model("fal-ai/midjourney"));

        // Non-image models
        assert!(!FalProvider::is_image_model("fal-ai/any-llm"));
        assert!(!FalProvider::is_image_model("fal-ai/llavav15-13b"));
    }

    #[test]
    fn test_image_request_serialization() {
        let request = FalImageRequest {
            prompt: "A beautiful sunset".to_string(),
            image_size: Some("landscape_4_3".to_string()),
            num_inference_steps: Some(4),
            num_images: Some(1),
            enable_safety_checker: Some(true),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("A beautiful sunset"));
        assert!(json.contains("landscape_4_3"));
        assert!(json.contains("4"));
        assert!(json.contains("enable_safety_checker"));
    }

    #[test]
    fn test_image_response_deserialization() {
        let json = r#"{
            "images": [{
                "url": "https://fal.ai/image1.png",
                "width": 1024,
                "height": 768,
                "content_type": "image/png"
            }],
            "seed": 12345
        }"#;

        let response: FalImageResponse = serde_json::from_str(json).unwrap();
        assert!(response.images.is_some());
        let images = response.images.unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].url, "https://fal.ai/image1.png");
        assert_eq!(images[0].width, Some(1024));
        assert_eq!(images[0].height, Some(768));
        assert_eq!(response.seed, Some(12345));
    }

    #[test]
    fn test_llm_request_serialization() {
        let request = FalLlmRequest {
            messages: vec![FalMessage {
                role: "user".to_string(),
                content: "Hello".to_string(),
            }],
            max_tokens: Some(1000),
            temperature: Some(0.7),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("Hello"));
        assert!(json.contains("1000"));
        assert!(json.contains("0.7"));
    }

    #[test]
    fn test_llm_response_deserialization() {
        let json = r#"{"output": "Hello! How can I help you?"}"#;
        let response: FalLlmResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            response.output,
            Some("Hello! How can I help you?".to_string())
        );
    }

    #[test]
    fn test_fal_image_deserialization() {
        let json = r#"{
            "url": "https://example.com/img.png",
            "width": 512,
            "height": 512,
            "content_type": "image/png"
        }"#;

        let image: FalImage = serde_json::from_str(json).unwrap();
        assert_eq!(image.url, "https://example.com/img.png");
        assert_eq!(image.width, Some(512));
        assert_eq!(image.height, Some(512));
        assert_eq!(image.content_type, Some("image/png".to_string()));
    }

    #[test]
    fn test_fal_message_serialization() {
        let message = FalMessage {
            role: "assistant".to_string(),
            content: "I can help you with that!".to_string(),
        };

        let json = serde_json::to_string(&message).unwrap();
        assert!(json.contains("assistant"));
        assert!(json.contains("I can help you with that!"));
    }
}
