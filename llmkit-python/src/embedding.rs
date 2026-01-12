//! Embedding API bindings for Python.
//!
//! Provides access to the LLMKit embedding functionality for generating
//! text embeddings from various providers (OpenAI, Cohere, etc.).

use llmkit::embedding::{
    Embedding, EmbeddingInputType, EmbeddingRequest, EmbeddingResponse, EmbeddingUsage,
    EncodingFormat,
};
use pyo3::prelude::*;

// ============================================================================
// ENUMS
// ============================================================================

/// Output encoding format for embeddings.
#[pyclass(name = "EncodingFormat", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyEncodingFormat {
    /// Float32 array (default).
    Float,
    /// Base64-encoded binary.
    Base64,
}

impl From<PyEncodingFormat> for EncodingFormat {
    fn from(value: PyEncodingFormat) -> Self {
        match value {
            PyEncodingFormat::Float => EncodingFormat::Float,
            PyEncodingFormat::Base64 => EncodingFormat::Base64,
        }
    }
}

impl From<EncodingFormat> for PyEncodingFormat {
    fn from(value: EncodingFormat) -> Self {
        match value {
            EncodingFormat::Float => PyEncodingFormat::Float,
            EncodingFormat::Base64 => PyEncodingFormat::Base64,
        }
    }
}

/// Input type hint for embedding optimization.
#[pyclass(name = "EmbeddingInputType", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyEmbeddingInputType {
    /// The input is a search query.
    Query,
    /// The input is a document to be indexed.
    Document,
}

impl From<PyEmbeddingInputType> for EmbeddingInputType {
    fn from(value: PyEmbeddingInputType) -> Self {
        match value {
            PyEmbeddingInputType::Query => EmbeddingInputType::Query,
            PyEmbeddingInputType::Document => EmbeddingInputType::Document,
        }
    }
}

// ============================================================================
// REQUEST
// ============================================================================

/// Request for generating embeddings.
#[pyclass(name = "EmbeddingRequest")]
#[derive(Clone)]
pub struct PyEmbeddingRequest {
    pub inner: EmbeddingRequest,
}

#[pymethods]
impl PyEmbeddingRequest {
    /// Create a new embedding request for a single text.
    ///
    /// Args:
    ///     model: The embedding model to use (e.g., "text-embedding-3-small").
    ///     text: The text to embed.
    ///
    /// Example:
    ///     >>> request = EmbeddingRequest("text-embedding-3-small", "Hello, world!")
    #[new]
    #[pyo3(signature = (model, text))]
    pub fn new(model: String, text: String) -> Self {
        Self {
            inner: EmbeddingRequest::new(model, text),
        }
    }

    /// Create a new embedding request for multiple texts (batch).
    ///
    /// Args:
    ///     model: The embedding model to use.
    ///     texts: List of texts to embed.
    ///
    /// Example:
    ///     >>> request = EmbeddingRequest.batch("text-embedding-3-small", ["Hello", "World"])
    #[staticmethod]
    pub fn batch(model: String, texts: Vec<String>) -> Self {
        Self {
            inner: EmbeddingRequest::batch(model, texts),
        }
    }

    /// Set the output dimensions (for models that support dimension reduction).
    ///
    /// Args:
    ///     dimensions: The number of dimensions for the output embedding.
    ///
    /// Returns:
    ///     Self for method chaining.
    pub fn with_dimensions(&self, dimensions: usize) -> Self {
        Self {
            inner: self.inner.clone().with_dimensions(dimensions),
        }
    }

    /// Set the encoding format.
    ///
    /// Args:
    ///     format: The encoding format (Float or Base64).
    ///
    /// Returns:
    ///     Self for method chaining.
    pub fn with_encoding_format(&self, format: PyEncodingFormat) -> Self {
        Self {
            inner: self.inner.clone().with_encoding_format(format.into()),
        }
    }

    /// Set the input type hint for optimized embeddings.
    ///
    /// Args:
    ///     input_type: The input type (Query or Document).
    ///
    /// Returns:
    ///     Self for method chaining.
    pub fn with_input_type(&self, input_type: PyEmbeddingInputType) -> Self {
        Self {
            inner: self.inner.clone().with_input_type(input_type.into()),
        }
    }

    /// Get the model name.
    #[getter]
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// Get the number of texts to embed.
    #[getter]
    pub fn text_count(&self) -> usize {
        self.inner.text_count()
    }

    /// Get all input texts as a list.
    pub fn texts(&self) -> Vec<String> {
        self.inner
            .texts()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Get the dimensions setting.
    #[getter]
    pub fn dimensions(&self) -> Option<usize> {
        self.inner.dimensions
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingRequest(model='{}', text_count={})",
            self.inner.model,
            self.inner.text_count()
        )
    }
}

// ============================================================================
// RESPONSE
// ============================================================================

/// A single embedding vector.
#[pyclass(name = "Embedding")]
#[derive(Clone)]
pub struct PyEmbedding {
    inner: Embedding,
}

#[pymethods]
impl PyEmbedding {
    /// The index of this embedding in the batch.
    #[getter]
    pub fn index(&self) -> usize {
        self.inner.index
    }

    /// The embedding vector values.
    #[getter]
    pub fn values(&self) -> Vec<f32> {
        self.inner.values.clone()
    }

    /// Get the number of dimensions.
    #[getter]
    pub fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    /// Compute cosine similarity with another embedding.
    ///
    /// Args:
    ///     other: Another embedding to compare with.
    ///
    /// Returns:
    ///     Cosine similarity score (-1 to 1).
    pub fn cosine_similarity(&self, other: &PyEmbedding) -> f32 {
        self.inner.cosine_similarity(&other.inner)
    }

    /// Compute dot product with another embedding.
    ///
    /// Args:
    ///     other: Another embedding to compute dot product with.
    ///
    /// Returns:
    ///     Dot product value.
    pub fn dot_product(&self, other: &PyEmbedding) -> f32 {
        self.inner.dot_product(&other.inner)
    }

    /// Compute Euclidean distance to another embedding.
    ///
    /// Args:
    ///     other: Another embedding to compute distance to.
    ///
    /// Returns:
    ///     Euclidean distance.
    pub fn euclidean_distance(&self, other: &PyEmbedding) -> f32 {
        self.inner.euclidean_distance(&other.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "Embedding(index={}, dimensions={})",
            self.inner.index,
            self.dimensions()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.values.len()
    }
}

impl From<Embedding> for PyEmbedding {
    fn from(inner: Embedding) -> Self {
        Self { inner }
    }
}

/// Token usage for embedding requests.
#[pyclass(name = "EmbeddingUsage")]
#[derive(Clone)]
pub struct PyEmbeddingUsage {
    inner: EmbeddingUsage,
}

#[pymethods]
impl PyEmbeddingUsage {
    /// Number of tokens in the input.
    #[getter]
    pub fn prompt_tokens(&self) -> u32 {
        self.inner.prompt_tokens
    }

    /// Total tokens processed.
    #[getter]
    pub fn total_tokens(&self) -> u32 {
        self.inner.total_tokens
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingUsage(prompt_tokens={}, total_tokens={})",
            self.inner.prompt_tokens, self.inner.total_tokens
        )
    }
}

impl From<EmbeddingUsage> for PyEmbeddingUsage {
    fn from(inner: EmbeddingUsage) -> Self {
        Self { inner }
    }
}

/// Response from an embedding request.
#[pyclass(name = "EmbeddingResponse")]
#[derive(Clone)]
pub struct PyEmbeddingResponse {
    inner: EmbeddingResponse,
}

#[pymethods]
impl PyEmbeddingResponse {
    /// The model used for embedding.
    #[getter]
    pub fn model(&self) -> &str {
        &self.inner.model
    }

    /// The generated embeddings.
    #[getter]
    pub fn embeddings(&self) -> Vec<PyEmbedding> {
        self.inner
            .embeddings
            .iter()
            .cloned()
            .map(PyEmbedding::from)
            .collect()
    }

    /// Token usage information.
    #[getter]
    pub fn usage(&self) -> PyEmbeddingUsage {
        self.inner.usage.clone().into()
    }

    /// Get the first embedding (convenience for single-text requests).
    pub fn first(&self) -> Option<PyEmbedding> {
        self.inner.first().cloned().map(PyEmbedding::from)
    }

    /// Get embedding values as a flat list (for single-text requests).
    pub fn values(&self) -> Option<Vec<f32>> {
        self.inner.values().map(|v| v.to_vec())
    }

    /// Get the embedding dimensions.
    #[getter]
    pub fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    fn __repr__(&self) -> String {
        format!(
            "EmbeddingResponse(model='{}', embeddings={}, dimensions={})",
            self.inner.model,
            self.inner.embeddings.len(),
            self.dimensions()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.embeddings.len()
    }
}

impl From<EmbeddingResponse> for PyEmbeddingResponse {
    fn from(inner: EmbeddingResponse) -> Self {
        Self { inner }
    }
}
