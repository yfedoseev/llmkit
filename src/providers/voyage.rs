//! Voyage AI API provider implementation.
//!
//! This module provides access to Voyage AI's embedding and reranking models.
//! Voyage AI specializes in high-quality embeddings for semantic search and RAG.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::VoyageProvider;
//!
//! // From environment variable
//! let provider = VoyageProvider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = VoyageProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! ## Embedding Models
//! - `voyage-3` - Latest general-purpose embedding model
//! - `voyage-3-lite` - Lighter, faster variant
//! - `voyage-code-3` - Code-optimized embeddings
//! - `voyage-finance-2` - Finance domain
//! - `voyage-law-2` - Legal domain
//!
//! ## Reranking Models
//! - `rerank-2` - General reranking
//! - `rerank-2-lite` - Faster reranking
//!
//! # Environment Variables
//!
//! - `VOYAGE_API_KEY` - Your Voyage AI API key

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::{Provider, ProviderConfig};
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

const VOYAGE_API_URL: &str = "https://api.voyageai.com/v1";

/// Voyage AI API provider.
///
/// Provides access to Voyage AI's embedding and reranking models.
/// Note: Voyage AI is primarily an embedding provider, but this implementation
/// wraps embedding calls in a chat-like interface for unified access.
pub struct VoyageProvider {
    config: ProviderConfig,
    client: Client,
}

impl VoyageProvider {
    /// Create a new Voyage AI provider with the given configuration.
    pub fn new(config: ProviderConfig) -> Result<Self> {
        let mut headers = reqwest::header::HeaderMap::new();

        if let Some(ref key) = config.api_key {
            headers.insert(
                reqwest::header::AUTHORIZATION,
                format!("Bearer {}", key)
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

    /// Create a new Voyage AI provider from environment variable.
    ///
    /// Reads the API key from `VOYAGE_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("VOYAGE_API_KEY");
        Self::new(config)
    }

    /// Create a new Voyage AI provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn embeddings_url(&self) -> String {
        format!(
            "{}/embeddings",
            self.config.base_url.as_deref().unwrap_or(VOYAGE_API_URL)
        )
    }

    fn rerank_url(&self) -> String {
        format!(
            "{}/rerank",
            self.config.base_url.as_deref().unwrap_or(VOYAGE_API_URL)
        )
    }

    /// Get embeddings for the given texts.
    pub async fn embed(&self, model: &str, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = VoyageEmbedRequest {
            model: model.to_string(),
            input: texts,
            input_type: None,
            truncation: Some(true),
        };

        let response = self
            .client
            .post(&self.embeddings_url())
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Voyage AI API error {}: {}", status, error_text),
            ));
        }

        let api_response: VoyageEmbedResponse = response.json().await?;
        Ok(api_response.data.into_iter().map(|d| d.embedding).collect())
    }

    /// Rerank documents based on a query.
    pub async fn rerank(
        &self,
        model: &str,
        query: &str,
        documents: Vec<String>,
        top_k: Option<usize>,
    ) -> Result<Vec<VoyageRerankResult>> {
        let request = VoyageRerankRequest {
            model: model.to_string(),
            query: query.to_string(),
            documents,
            top_k,
            return_documents: Some(true),
        };

        let response = self
            .client
            .post(&self.rerank_url())
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Voyage AI API error {}: {}", status, error_text),
            ));
        }

        let api_response: VoyageRerankResponse = response.json().await?;
        Ok(api_response.data)
    }
}

#[async_trait]
impl Provider for VoyageProvider {
    fn name(&self) -> &str {
        "voyage"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        // Extract text from the last user message
        let text = request
            .messages
            .iter()
            .filter(|m| matches!(m.role, Role::User))
            .last()
            .and_then(|m| {
                m.content.iter().find_map(|block| {
                    if let ContentBlock::Text { text } = block {
                        Some(text.clone())
                    } else {
                        None
                    }
                })
            })
            .ok_or_else(|| Error::invalid_request("No user message found"))?;

        // Check if this is a rerank request (model starts with "rerank")
        if request.model.starts_with("rerank") {
            // For rerank, we expect the text to be formatted as "query\n---\ndoc1\ndoc2\n..."
            let parts: Vec<&str> = text.split("\n---\n").collect();
            if parts.len() >= 2 {
                let query = parts[0];
                let documents: Vec<String> = parts[1].lines().map(|s| s.to_string()).collect();
                let results = self.rerank(&request.model, query, documents, None).await?;

                let result_text = results
                    .iter()
                    .map(|r| {
                        format!(
                            "{}: {} (score: {:.4})",
                            r.index,
                            r.document.as_deref().unwrap_or(""),
                            r.relevance_score
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                return Ok(CompletionResponse {
                    id: uuid::Uuid::new_v4().to_string(),
                    model: request.model,
                    content: vec![ContentBlock::Text { text: result_text }],
                    stop_reason: StopReason::EndTurn,
                    usage: Usage {
                        input_tokens: 0,
                        output_tokens: 0,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                    },
                });
            }
        }

        // Default: embedding request
        let embeddings = self.embed(&request.model, vec![text]).await?;

        let embedding_text = embeddings
            .first()
            .map(|e| {
                format!(
                    "[{}]",
                    e.iter()
                        .take(10)
                        .map(|v| format!("{:.6}", v))
                        .collect::<Vec<_>>()
                        .join(", ")
                ) + &format!("... ({} dimensions)", e.len())
            })
            .unwrap_or_else(|| "No embedding generated".to_string());

        Ok(CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: request.model,
            content: vec![ContentBlock::Text {
                text: embedding_text,
            }],
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
        // Voyage AI doesn't support streaming, fall back to regular completion
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
                        delta: Some(ContentDelta::TextDelta { text }),
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

// Voyage AI API types

#[derive(Debug, Serialize)]
struct VoyageEmbedRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    input_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncation: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct VoyageEmbedResponse {
    data: Vec<VoyageEmbedding>,
}

#[derive(Debug, Deserialize)]
struct VoyageEmbedding {
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct VoyageRerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_documents: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct VoyageRerankResponse {
    data: Vec<VoyageRerankResult>,
}

/// Result from a rerank operation.
#[derive(Debug, Deserialize)]
pub struct VoyageRerankResult {
    /// Original index of the document.
    pub index: usize,
    /// Relevance score (higher is more relevant).
    pub relevance_score: f64,
    /// The document text (if return_documents was true).
    pub document: Option<String>,
}

// ============================================================================
// EmbeddingProvider Implementation
// ============================================================================

use crate::embedding::{
    Embedding, EmbeddingInput, EmbeddingInputType, EmbeddingProvider, EmbeddingRequest,
    EmbeddingResponse, EmbeddingUsage,
};

#[async_trait]
impl EmbeddingProvider for VoyageProvider {
    fn name(&self) -> &str {
        "voyage"
    }

    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let texts = match &request.input {
            EmbeddingInput::Single(text) => vec![text.clone()],
            EmbeddingInput::Batch(texts) => texts.clone(),
        };

        let input_type = request.input_type.map(|t| match t {
            EmbeddingInputType::Query => "query".to_string(),
            EmbeddingInputType::Document => "document".to_string(),
        });

        let api_request = VoyageEmbedRequest {
            model: request.model.clone(),
            input: texts,
            input_type,
            truncation: Some(true),
        };

        let response = self
            .client
            .post(&self.embeddings_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Voyage AI API error {}: {}", status, error_text),
            ));
        }

        let api_response: VoyageEmbedResponseFull = response.json().await?;

        let embeddings = api_response
            .data
            .into_iter()
            .enumerate()
            .map(|(i, e)| Embedding::new(i, e.embedding))
            .collect();

        Ok(EmbeddingResponse {
            model: request.model,
            embeddings,
            usage: EmbeddingUsage::new(
                api_response.usage.total_tokens,
                api_response.usage.total_tokens,
            ),
        })
    }

    fn embedding_dimensions(&self, model: &str) -> Option<usize> {
        match model {
            "voyage-3" => Some(1024),
            "voyage-3-lite" => Some(512),
            "voyage-code-3" => Some(1024),
            "voyage-finance-2" => Some(1024),
            "voyage-law-2" => Some(1024),
            _ => None,
        }
    }

    fn default_embedding_model(&self) -> Option<&str> {
        Some("voyage-3")
    }

    fn max_batch_size(&self) -> usize {
        128
    }

    fn supported_embedding_models(&self) -> Option<&[&str]> {
        Some(&[
            "voyage-3",
            "voyage-3-lite",
            "voyage-code-3",
            "voyage-finance-2",
            "voyage-law-2",
        ])
    }
}

#[derive(Debug, Deserialize)]
struct VoyageEmbedResponseFull {
    data: Vec<VoyageEmbedding>,
    usage: VoyageUsage,
}

#[derive(Debug, Deserialize)]
struct VoyageUsage {
    total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Provider;

    #[test]
    fn test_provider_creation() {
        let provider = VoyageProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(Provider::name(&provider), "voyage");
    }

    #[test]
    fn test_embedding_dimensions() {
        let provider = VoyageProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.embedding_dimensions("voyage-3"), Some(1024));
        assert_eq!(provider.embedding_dimensions("voyage-3-lite"), Some(512));
    }
}
