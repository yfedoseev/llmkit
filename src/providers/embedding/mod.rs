//! Embedding and reranking providers.
//!
//! This module contains providers specialized in generating text embeddings
//! and reranking search results for semantic search and RAG applications.

#[cfg(feature = "voyage")]
pub mod voyage;

#[cfg(feature = "jina")]
pub mod jina;

#[cfg(feature = "mistral-embeddings")]
pub mod mistral_embeddings;

// Re-exports

#[cfg(feature = "voyage")]
pub use voyage::VoyageProvider;

#[cfg(feature = "jina")]
pub use jina::JinaProvider;

#[cfg(feature = "mistral-embeddings")]
pub use mistral_embeddings::{EmbeddingData, MistralEmbeddingsProvider};
