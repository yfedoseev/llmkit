//! Embedding API bindings for Node.js/TypeScript.
//!
//! Provides access to the ModelSuite embedding functionality for generating
//! text embeddings from various providers (OpenAI, Cohere, etc.).

use modelsuite::embedding::{
    Embedding, EmbeddingInputType, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    EncodingFormat,
};
use napi_derive::napi;

// ============================================================================
// ENUMS
// ============================================================================

/// Output encoding format for embeddings.
#[napi]
pub enum JsEncodingFormat {
    /// Float32 array (default).
    Float,
    /// Base64-encoded binary.
    Base64,
}

impl From<JsEncodingFormat> for EncodingFormat {
    fn from(value: JsEncodingFormat) -> Self {
        match value {
            JsEncodingFormat::Float => EncodingFormat::Float,
            JsEncodingFormat::Base64 => EncodingFormat::Base64,
        }
    }
}

impl From<EncodingFormat> for JsEncodingFormat {
    fn from(value: EncodingFormat) -> Self {
        match value {
            EncodingFormat::Float => JsEncodingFormat::Float,
            EncodingFormat::Base64 => JsEncodingFormat::Base64,
        }
    }
}

/// Input type hint for embedding optimization.
#[napi]
pub enum JsEmbeddingInputType {
    /// The input is a search query.
    Query,
    /// The input is a document to be indexed.
    Document,
}

impl From<JsEmbeddingInputType> for EmbeddingInputType {
    fn from(value: JsEmbeddingInputType) -> Self {
        match value {
            JsEmbeddingInputType::Query => EmbeddingInputType::Query,
            JsEmbeddingInputType::Document => EmbeddingInputType::Document,
        }
    }
}

// ============================================================================
// REQUEST
// ============================================================================

/// Request for generating embeddings.
///
/// @example
/// ```typescript
/// // Single text
/// const request = new EmbeddingRequest("text-embedding-3-small", "Hello, world!");
///
/// // Batch
/// const request = EmbeddingRequest.batch("text-embedding-3-small", ["Hello", "World"]);
/// ```
#[napi]
pub struct JsEmbeddingRequest {
    pub(crate) inner: EmbeddingRequest,
}

#[napi]
impl JsEmbeddingRequest {
    /// Create a new embedding request for a single text.
    ///
    /// @param model - The embedding model to use (e.g., "text-embedding-3-small")
    /// @param text - The text to embed
    #[napi(constructor)]
    pub fn new(model: String, text: String) -> Self {
        Self {
            inner: EmbeddingRequest::new(model, text),
        }
    }

    /// Create a new embedding request for multiple texts (batch).
    ///
    /// @param model - The embedding model to use
    /// @param texts - List of texts to embed
    #[napi(factory)]
    pub fn batch(model: String, texts: Vec<String>) -> Self {
        Self {
            inner: EmbeddingRequest::batch(model, texts),
        }
    }

    /// Set the output dimensions (for models that support dimension reduction).
    ///
    /// @param dimensions - The number of dimensions for the output embedding
    /// @returns Self for method chaining
    #[napi]
    pub fn with_dimensions(&self, dimensions: u32) -> Self {
        Self {
            inner: self.inner.clone().with_dimensions(dimensions as usize),
        }
    }

    /// Set the encoding format.
    ///
    /// @param format - The encoding format (Float or Base64)
    /// @returns Self for method chaining
    #[napi]
    pub fn with_encoding_format(&self, format: JsEncodingFormat) -> Self {
        Self {
            inner: self.inner.clone().with_encoding_format(format.into()),
        }
    }

    /// Set the input type hint for optimized embeddings.
    ///
    /// @param inputType - The input type (Query or Document)
    /// @returns Self for method chaining
    #[napi]
    pub fn with_input_type(&self, input_type: JsEmbeddingInputType) -> Self {
        Self {
            inner: self.inner.clone().with_input_type(input_type.into()),
        }
    }

    /// Get the model name.
    #[napi(getter)]
    pub fn model(&self) -> String {
        self.inner.model.clone()
    }

    /// Get the number of texts to embed.
    #[napi(getter)]
    pub fn text_count(&self) -> u32 {
        self.inner.text_count() as u32
    }

    /// Get all input texts as an array.
    #[napi]
    pub fn texts(&self) -> Vec<String> {
        self.inner
            .texts()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Get the dimensions setting.
    #[napi(getter)]
    pub fn dimensions(&self) -> Option<u32> {
        self.inner.dimensions.map(|d| d as u32)
    }
}

// ============================================================================
// RESPONSE
// ============================================================================

/// A single embedding vector.
#[napi]
pub struct JsEmbedding {
    inner: Embedding,
}

#[napi]
impl JsEmbedding {
    /// The index of this embedding in the batch.
    #[napi(getter)]
    pub fn index(&self) -> u32 {
        self.inner.index as u32
    }

    /// The embedding vector values.
    #[napi(getter)]
    pub fn values(&self) -> Vec<f64> {
        self.inner.values.iter().map(|v| *v as f64).collect()
    }

    /// Get the number of dimensions.
    #[napi(getter, js_name = "dimensionCount")]
    pub fn dimension_count(&self) -> u32 {
        self.inner.dimensions() as u32
    }

    /// Compute cosine similarity with another embedding.
    ///
    /// @param other - Another embedding to compare with
    /// @returns Cosine similarity score (-1 to 1)
    #[napi]
    pub fn cosine_similarity(&self, other: &JsEmbedding) -> f64 {
        self.inner.cosine_similarity(&other.inner) as f64
    }

    /// Compute dot product with another embedding.
    ///
    /// @param other - Another embedding to compute dot product with
    /// @returns Dot product value
    #[napi]
    pub fn dot_product(&self, other: &JsEmbedding) -> f64 {
        self.inner.dot_product(&other.inner) as f64
    }

    /// Compute Euclidean distance to another embedding.
    ///
    /// @param other - Another embedding to compute distance to
    /// @returns Euclidean distance
    #[napi]
    pub fn euclidean_distance(&self, other: &JsEmbedding) -> f64 {
        self.inner.euclidean_distance(&other.inner) as f64
    }
}

impl From<Embedding> for JsEmbedding {
    fn from(inner: Embedding) -> Self {
        Self { inner }
    }
}

/// Token usage for embedding requests.
#[napi]
pub struct JsEmbeddingUsage {
    inner: EmbeddingUsage,
}

#[napi]
impl JsEmbeddingUsage {
    /// Number of tokens in the input.
    #[napi(getter)]
    pub fn prompt_tokens(&self) -> u32 {
        self.inner.prompt_tokens
    }

    /// Total tokens processed.
    #[napi(getter)]
    pub fn total_tokens(&self) -> u32 {
        self.inner.total_tokens
    }
}

impl From<EmbeddingUsage> for JsEmbeddingUsage {
    fn from(inner: EmbeddingUsage) -> Self {
        Self { inner }
    }
}

/// Response from an embedding request.
#[napi]
pub struct JsEmbeddingResponse {
    inner: EmbeddingResponse,
}

#[napi]
impl JsEmbeddingResponse {
    /// The model used for embedding.
    #[napi(getter)]
    pub fn model(&self) -> String {
        self.inner.model.clone()
    }

    /// The generated embeddings.
    #[napi(getter)]
    pub fn embeddings(&self) -> Vec<JsEmbedding> {
        self.inner
            .embeddings
            .iter()
            .cloned()
            .map(JsEmbedding::from)
            .collect()
    }

    /// Token usage information.
    #[napi(getter)]
    pub fn usage(&self) -> JsEmbeddingUsage {
        self.inner.usage.clone().into()
    }

    /// Get the first embedding (convenience for single-text requests).
    #[napi]
    pub fn first(&self) -> Option<JsEmbedding> {
        self.inner.first().cloned().map(JsEmbedding::from)
    }

    /// Get embedding values as a flat array (for single-text requests).
    #[napi]
    pub fn values(&self) -> Option<Vec<f64>> {
        self.inner
            .values()
            .map(|v| v.iter().map(|f| *f as f64).collect())
    }

    /// Get the embedding dimensions.
    #[napi(getter, js_name = "dimensionCount")]
    pub fn dimension_count(&self) -> u32 {
        self.inner.dimensions() as u32
    }
}

impl From<EmbeddingResponse> for JsEmbeddingResponse {
    fn from(inner: EmbeddingResponse) -> Self {
        Self { inner }
    }
}
