//! Embedding API for generating text embeddings across multiple providers.
//!
//! This module provides a unified interface for generating text embeddings
//! from various providers including OpenAI, Voyage, Jina, Cohere, and others.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::{EmbeddingProvider, EmbeddingRequest, EmbeddingInput};
//!
//! // Create provider (OpenAI example)
//! let provider = OpenAIProvider::from_env()?;
//!
//! // Create embedding request
//! let request = EmbeddingRequest::new("text-embedding-3-small", "Hello, world!");
//!
//! // Get embeddings
//! let response = provider.embed(request).await?;
//! println!("Embedding dimensions: {}", response.embeddings[0].values.len());
//! ```
//!
//! # Batch Embeddings
//!
//! ```ignore
//! // Batch embed multiple texts
//! let request = EmbeddingRequest::batch(
//!     "text-embedding-3-small",
//!     vec!["First text", "Second text", "Third text"],
//! );
//!
//! let response = provider.embed(request).await?;
//! for embedding in &response.embeddings {
//!     println!("Index {}: {} dimensions", embedding.index, embedding.values.len());
//! }
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Request for generating embeddings.
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// The embedding model to use (e.g., "text-embedding-3-small").
    pub model: String,
    /// The input text(s) to embed.
    pub input: EmbeddingInput,
    /// Optional: Number of dimensions for the output embedding.
    /// Only supported by some models (e.g., OpenAI text-embedding-3-*).
    pub dimensions: Option<usize>,
    /// Optional: Output encoding format.
    pub encoding_format: Option<EncodingFormat>,
    /// Optional: Input type hint for optimized embeddings.
    /// Supported by some providers like Voyage AI.
    pub input_type: Option<EmbeddingInputType>,
}

impl EmbeddingRequest {
    /// Create a new embedding request for a single text.
    pub fn new(model: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: EmbeddingInput::Single(text.into()),
            dimensions: None,
            encoding_format: None,
            input_type: None,
        }
    }

    /// Create a new embedding request for multiple texts (batch).
    pub fn batch(model: impl Into<String>, texts: Vec<impl Into<String>>) -> Self {
        Self {
            model: model.into(),
            input: EmbeddingInput::Batch(texts.into_iter().map(|t| t.into()).collect()),
            dimensions: None,
            encoding_format: None,
            input_type: None,
        }
    }

    /// Set the output dimensions (for models that support dimension reduction).
    pub fn with_dimensions(mut self, dimensions: usize) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    /// Set the encoding format.
    pub fn with_encoding_format(mut self, format: EncodingFormat) -> Self {
        self.encoding_format = Some(format);
        self
    }

    /// Set the input type hint.
    pub fn with_input_type(mut self, input_type: EmbeddingInputType) -> Self {
        self.input_type = Some(input_type);
        self
    }

    /// Get the number of texts to embed.
    pub fn text_count(&self) -> usize {
        match &self.input {
            EmbeddingInput::Single(_) => 1,
            EmbeddingInput::Batch(texts) => texts.len(),
        }
    }

    /// Get all input texts as a vector.
    pub fn texts(&self) -> Vec<&str> {
        match &self.input {
            EmbeddingInput::Single(text) => vec![text.as_str()],
            EmbeddingInput::Batch(texts) => texts.iter().map(|s| s.as_str()).collect(),
        }
    }
}

/// Input for embedding requests.
#[derive(Debug, Clone)]
pub enum EmbeddingInput {
    /// Single text to embed.
    Single(String),
    /// Multiple texts to embed in a batch.
    Batch(Vec<String>),
}

/// Output encoding format for embeddings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EncodingFormat {
    /// Float32 array (default).
    #[default]
    Float,
    /// Base64-encoded binary.
    Base64,
}

/// Input type hint for embedding optimization.
///
/// Some providers (like Voyage AI) optimize embeddings based on the input type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingInputType {
    /// The input is a search query.
    Query,
    /// The input is a document to be indexed.
    Document,
}

/// Response from an embedding request.
#[derive(Debug, Clone)]
pub struct EmbeddingResponse {
    /// The model used for embedding.
    pub model: String,
    /// The generated embeddings.
    pub embeddings: Vec<Embedding>,
    /// Token usage information.
    pub usage: EmbeddingUsage,
}

impl EmbeddingResponse {
    /// Get the first embedding (convenience for single-text requests).
    pub fn first(&self) -> Option<&Embedding> {
        self.embeddings.first()
    }

    /// Get embedding values as a flat vector (for single-text requests).
    pub fn values(&self) -> Option<&[f32]> {
        self.first().map(|e| e.values.as_slice())
    }

    /// Get the embedding dimensions.
    pub fn dimensions(&self) -> usize {
        self.embeddings.first().map(|e| e.values.len()).unwrap_or(0)
    }
}

/// A single embedding vector.
#[derive(Debug, Clone)]
pub struct Embedding {
    /// The index of this embedding in the batch.
    pub index: usize,
    /// The embedding vector values.
    pub values: Vec<f32>,
}

impl Embedding {
    /// Create a new embedding.
    pub fn new(index: usize, values: Vec<f32>) -> Self {
        Self { index, values }
    }

    /// Get the number of dimensions.
    pub fn dimensions(&self) -> usize {
        self.values.len()
    }

    /// Compute cosine similarity with another embedding.
    pub fn cosine_similarity(&self, other: &Embedding) -> f32 {
        if self.values.len() != other.values.len() {
            return 0.0;
        }

        let dot_product: f32 = self
            .values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm_a: f32 = self.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = other.values.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot_product / (norm_a * norm_b)
    }

    /// Compute dot product with another embedding.
    pub fn dot_product(&self, other: &Embedding) -> f32 {
        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Compute Euclidean distance to another embedding.
    pub fn euclidean_distance(&self, other: &Embedding) -> f32 {
        if self.values.len() != other.values.len() {
            return f32::INFINITY;
        }

        self.values
            .iter()
            .zip(other.values.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            .sqrt()
    }
}

/// Token usage for embedding requests.
#[derive(Debug, Clone, Default)]
pub struct EmbeddingUsage {
    /// Number of tokens in the input.
    pub prompt_tokens: u32,
    /// Total tokens processed.
    pub total_tokens: u32,
}

impl EmbeddingUsage {
    /// Create new usage stats.
    pub fn new(prompt_tokens: u32, total_tokens: u32) -> Self {
        Self {
            prompt_tokens,
            total_tokens,
        }
    }
}

/// Trait for providers that support text embeddings.
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Get the provider name.
    fn name(&self) -> &str;

    /// Generate embeddings for the given request.
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse>;

    /// Get the default embedding dimensions for a model.
    ///
    /// Returns `None` if the model is unknown or dimensions are variable.
    fn embedding_dimensions(&self, _model: &str) -> Option<usize> {
        None
    }

    /// Get the default embedding model for this provider.
    fn default_embedding_model(&self) -> Option<&str> {
        None
    }

    /// Get the maximum batch size supported by this provider.
    fn max_batch_size(&self) -> usize {
        2048
    }

    /// Check if a model supports dimension reduction.
    fn supports_dimensions(&self, _model: &str) -> bool {
        false
    }

    /// Get all supported embedding models.
    fn supported_embedding_models(&self) -> Option<&[&str]> {
        None
    }
}

/// Information about an embedding model.
#[derive(Debug, Clone)]
pub struct EmbeddingModelInfo {
    /// Model ID/name.
    pub id: &'static str,
    /// Provider that offers this model.
    pub provider: &'static str,
    /// Default output dimensions.
    pub dimensions: usize,
    /// Maximum input tokens.
    pub max_tokens: usize,
    /// Price per 1K tokens (USD).
    pub pricing_per_1k: f64,
    /// Whether the model supports dimension reduction.
    pub supports_dimensions: bool,
}

/// Registry of known embedding models.
pub static EMBEDDING_MODELS: &[EmbeddingModelInfo] = &[
    // OpenAI
    EmbeddingModelInfo {
        id: "text-embedding-3-small",
        provider: "openai",
        dimensions: 1536,
        max_tokens: 8191,
        pricing_per_1k: 0.00002,
        supports_dimensions: true,
    },
    EmbeddingModelInfo {
        id: "text-embedding-3-large",
        provider: "openai",
        dimensions: 3072,
        max_tokens: 8191,
        pricing_per_1k: 0.00013,
        supports_dimensions: true,
    },
    EmbeddingModelInfo {
        id: "text-embedding-ada-002",
        provider: "openai",
        dimensions: 1536,
        max_tokens: 8191,
        pricing_per_1k: 0.0001,
        supports_dimensions: false,
    },
    // Voyage AI
    EmbeddingModelInfo {
        id: "voyage-3",
        provider: "voyage",
        dimensions: 1024,
        max_tokens: 32000,
        pricing_per_1k: 0.00006,
        supports_dimensions: false,
    },
    EmbeddingModelInfo {
        id: "voyage-3-lite",
        provider: "voyage",
        dimensions: 512,
        max_tokens: 32000,
        pricing_per_1k: 0.00002,
        supports_dimensions: false,
    },
    EmbeddingModelInfo {
        id: "voyage-code-3",
        provider: "voyage",
        dimensions: 1024,
        max_tokens: 32000,
        pricing_per_1k: 0.00006,
        supports_dimensions: false,
    },
    // Jina AI
    EmbeddingModelInfo {
        id: "jina-embeddings-v3",
        provider: "jina",
        dimensions: 1024,
        max_tokens: 8192,
        pricing_per_1k: 0.00002,
        supports_dimensions: true,
    },
    EmbeddingModelInfo {
        id: "jina-clip-v2",
        provider: "jina",
        dimensions: 1024,
        max_tokens: 8192,
        pricing_per_1k: 0.00002,
        supports_dimensions: false,
    },
    // Cohere
    EmbeddingModelInfo {
        id: "embed-english-v3.0",
        provider: "cohere",
        dimensions: 1024,
        max_tokens: 512,
        pricing_per_1k: 0.0001,
        supports_dimensions: false,
    },
    EmbeddingModelInfo {
        id: "embed-multilingual-v3.0",
        provider: "cohere",
        dimensions: 1024,
        max_tokens: 512,
        pricing_per_1k: 0.0001,
        supports_dimensions: false,
    },
    EmbeddingModelInfo {
        id: "embed-english-light-v3.0",
        provider: "cohere",
        dimensions: 384,
        max_tokens: 512,
        pricing_per_1k: 0.0001,
        supports_dimensions: false,
    },
    // Google
    EmbeddingModelInfo {
        id: "textembedding-gecko@003",
        provider: "google",
        dimensions: 768,
        max_tokens: 3072,
        pricing_per_1k: 0.000025,
        supports_dimensions: false,
    },
    EmbeddingModelInfo {
        id: "text-embedding-004",
        provider: "google",
        dimensions: 768,
        max_tokens: 2048,
        pricing_per_1k: 0.000025,
        supports_dimensions: true,
    },
];

/// Get embedding model info by ID.
pub fn get_embedding_model_info(model_id: &str) -> Option<&'static EmbeddingModelInfo> {
    EMBEDDING_MODELS.iter().find(|m| m.id == model_id)
}

/// Get all embedding models for a provider.
pub fn get_embedding_models_by_provider(provider: &str) -> Vec<&'static EmbeddingModelInfo> {
    EMBEDDING_MODELS
        .iter()
        .filter(|m| m.provider == provider)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_request_single() {
        let request = EmbeddingRequest::new("text-embedding-3-small", "Hello, world!");
        assert_eq!(request.model, "text-embedding-3-small");
        assert_eq!(request.text_count(), 1);
        assert_eq!(request.texts(), vec!["Hello, world!"]);
    }

    #[test]
    fn test_embedding_request_batch() {
        let request =
            EmbeddingRequest::batch("text-embedding-3-small", vec!["First", "Second", "Third"]);
        assert_eq!(request.text_count(), 3);
        assert_eq!(request.texts(), vec!["First", "Second", "Third"]);
    }

    #[test]
    fn test_embedding_request_with_dimensions() {
        let request = EmbeddingRequest::new("text-embedding-3-small", "test").with_dimensions(256);
        assert_eq!(request.dimensions, Some(256));
    }

    #[test]
    fn test_cosine_similarity() {
        let e1 = Embedding::new(0, vec![1.0, 0.0, 0.0]);
        let e2 = Embedding::new(1, vec![1.0, 0.0, 0.0]);
        let e3 = Embedding::new(2, vec![0.0, 1.0, 0.0]);

        assert!((e1.cosine_similarity(&e2) - 1.0).abs() < 0.0001);
        assert!((e1.cosine_similarity(&e3) - 0.0).abs() < 0.0001);
    }

    #[test]
    fn test_euclidean_distance() {
        let e1 = Embedding::new(0, vec![0.0, 0.0]);
        let e2 = Embedding::new(1, vec![3.0, 4.0]);

        assert!((e1.euclidean_distance(&e2) - 5.0).abs() < 0.0001);
    }

    #[test]
    fn test_dot_product() {
        let e1 = Embedding::new(0, vec![1.0, 2.0, 3.0]);
        let e2 = Embedding::new(1, vec![4.0, 5.0, 6.0]);

        // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert!((e1.dot_product(&e2) - 32.0).abs() < 0.0001);
    }

    #[test]
    fn test_embedding_model_registry() {
        let model = get_embedding_model_info("text-embedding-3-small");
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.provider, "openai");
        assert_eq!(model.dimensions, 1536);
        assert!(model.supports_dimensions);
    }

    #[test]
    fn test_get_models_by_provider() {
        let voyage_models = get_embedding_models_by_provider("voyage");
        assert!(!voyage_models.is_empty());
        assert!(voyage_models.iter().all(|m| m.provider == "voyage"));
    }

    #[test]
    fn test_embedding_response() {
        let response = EmbeddingResponse {
            model: "test-model".to_string(),
            embeddings: vec![
                Embedding::new(0, vec![0.1, 0.2, 0.3]),
                Embedding::new(1, vec![0.4, 0.5, 0.6]),
            ],
            usage: EmbeddingUsage::new(10, 10),
        };

        assert_eq!(response.dimensions(), 3);
        assert!(response.first().is_some());
        assert_eq!(response.values().unwrap(), &[0.1, 0.2, 0.3]);
    }
}
