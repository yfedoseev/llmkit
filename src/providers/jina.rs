//! Jina AI API provider implementation.
//!
//! This module provides access to Jina AI's embedding, reranking, and reader models.
//! Jina AI offers high-quality embeddings and document processing capabilities.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::JinaProvider;
//!
//! // From environment variable
//! let provider = JinaProvider::from_env()?;
//!
//! // Or with explicit API key
//! let provider = JinaProvider::with_api_key("your-api-key")?;
//! ```
//!
//! # Supported Models
//!
//! ## Embedding Models
//! - `jina-embeddings-v3` - Latest multilingual embeddings
//! - `jina-embeddings-v2-base-en` - English embeddings
//! - `jina-embeddings-v2-base-code` - Code embeddings
//!
//! ## Reranker Models
//! - `jina-reranker-v2-base-multilingual` - Multilingual reranking
//! - `jina-colbert-v2` - ColBERT-based reranking
//!
//! ## Reader Models
//! - `jina-reader` - Web page reader/extractor
//!
//! # Environment Variables
//!
//! - `JINA_API_KEY` - Your Jina AI API key

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

const JINA_API_URL: &str = "https://api.jina.ai/v1";
const JINA_READER_URL: &str = "https://r.jina.ai";

/// Jina AI API provider.
///
/// Provides access to Jina AI's embedding, reranking, and reader models.
pub struct JinaProvider {
    config: ProviderConfig,
    client: Client,
}

impl JinaProvider {
    /// Create a new Jina AI provider with the given configuration.
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

    /// Create a new Jina AI provider from environment variable.
    ///
    /// Reads the API key from `JINA_API_KEY`.
    pub fn from_env() -> Result<Self> {
        let config = ProviderConfig::from_env("JINA_API_KEY");
        Self::new(config)
    }

    /// Create a new Jina AI provider with an API key.
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> {
        let config = ProviderConfig::new(api_key);
        Self::new(config)
    }

    fn embeddings_url(&self) -> String {
        format!(
            "{}/embeddings",
            self.config.base_url.as_deref().unwrap_or(JINA_API_URL)
        )
    }

    fn rerank_url(&self) -> String {
        format!(
            "{}/rerank",
            self.config.base_url.as_deref().unwrap_or(JINA_API_URL)
        )
    }

    /// Get embeddings for the given texts.
    pub async fn embed(&self, model: &str, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let request = JinaEmbedRequest {
            model: model.to_string(),
            input: texts,
            embedding_type: None,
        };

        let response = self
            .client
            .post(self.embeddings_url())
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Jina AI API error {}: {}", status, error_text),
            ));
        }

        let api_response: JinaEmbedResponse = response.json().await?;
        Ok(api_response.data.into_iter().map(|d| d.embedding).collect())
    }

    /// Rerank documents based on a query.
    pub async fn rerank(
        &self,
        model: &str,
        query: &str,
        documents: Vec<String>,
        top_n: Option<usize>,
    ) -> Result<Vec<JinaRerankResult>> {
        let request = JinaRerankRequest {
            model: model.to_string(),
            query: query.to_string(),
            documents,
            top_n,
            return_documents: Some(true),
        };

        let response = self
            .client
            .post(self.rerank_url())
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Jina AI API error {}: {}", status, error_text),
            ));
        }

        let api_response: JinaRerankResponse = response.json().await?;
        Ok(api_response.results)
    }

    /// Read and extract content from a URL using Jina Reader.
    pub async fn read_url(&self, url: &str) -> Result<String> {
        let reader_url = format!("{}/{}", JINA_READER_URL, url);

        let response = self.client.get(&reader_url).send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Jina Reader API error {}: {}", status, error_text),
            ));
        }

        Ok(response.text().await?)
    }
}

#[async_trait]
impl Provider for JinaProvider {
    fn name(&self) -> &str {
        "jina"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        // Extract text from the last user message
        let text = request
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
            .ok_or_else(|| Error::invalid_request("No user message found"))?;

        // Check if this is a reader request
        if request.model == "jina-reader" || request.model.starts_with("reader") {
            // Assume text is a URL
            let content = self.read_url(&text).await?;
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

        // Check if this is a rerank request
        if request.model.contains("rerank") || request.model.contains("colbert") {
            // For rerank, expect format "query\n---\ndoc1\ndoc2\n..."
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
                            r.document
                                .as_ref()
                                .and_then(|d| d.text.as_ref())
                                .unwrap_or(&String::new()),
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
        // Jina AI doesn't support streaming for embeddings, fall back to regular completion
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

// Jina AI API types

#[derive(Debug, Serialize)]
struct JinaEmbedRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    embedding_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct JinaEmbedResponse {
    data: Vec<JinaEmbedding>,
}

#[derive(Debug, Deserialize)]
struct JinaEmbedding {
    embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
struct JinaRerankRequest {
    model: String,
    query: String,
    documents: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_n: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    return_documents: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct JinaRerankResponse {
    results: Vec<JinaRerankResult>,
}

/// Result from a rerank operation.
#[derive(Debug, Deserialize)]
pub struct JinaRerankResult {
    /// Original index of the document.
    pub index: usize,
    /// Relevance score (higher is more relevant).
    pub relevance_score: f64,
    /// The document (if return_documents was true).
    pub document: Option<JinaDocument>,
}

/// Document in rerank result.
#[derive(Debug, Deserialize)]
pub struct JinaDocument {
    /// The document text.
    pub text: Option<String>,
}

// ============================================================================
// EmbeddingProvider Implementation
// ============================================================================

use crate::embedding::{
    Embedding, EmbeddingInput, EmbeddingProvider, EmbeddingRequest, EmbeddingResponse,
    EmbeddingUsage,
};

#[async_trait]
impl EmbeddingProvider for JinaProvider {
    fn name(&self) -> &str {
        "jina"
    }

    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        let texts = match &request.input {
            EmbeddingInput::Single(text) => vec![text.clone()],
            EmbeddingInput::Batch(texts) => texts.clone(),
        };

        let api_request = JinaEmbedRequestFull {
            model: request.model.clone(),
            input: texts,
            dimensions: request.dimensions,
            task: request.input_type.map(|t| match t {
                crate::embedding::EmbeddingInputType::Query => "retrieval.query".to_string(),
                crate::embedding::EmbeddingInputType::Document => "retrieval.passage".to_string(),
            }),
        };

        let response = self
            .client
            .post(self.embeddings_url())
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(Error::server(
                status.as_u16(),
                format!("Jina AI API error {}: {}", status, error_text),
            ));
        }

        let api_response: JinaEmbedResponseFull = response.json().await?;

        let embeddings = api_response
            .data
            .into_iter()
            .enumerate()
            .map(|(i, e)| Embedding::new(i, e.embedding))
            .collect();

        let usage = api_response.usage.map_or_else(
            || EmbeddingUsage::new(0, 0),
            |u| EmbeddingUsage::new(u.total_tokens, u.total_tokens),
        );

        Ok(EmbeddingResponse {
            model: request.model,
            embeddings,
            usage,
        })
    }

    fn embedding_dimensions(&self, model: &str) -> Option<usize> {
        match model {
            "jina-embeddings-v3" => Some(1024),
            "jina-embeddings-v2-base-en" => Some(768),
            "jina-embeddings-v2-base-code" => Some(768),
            "jina-clip-v2" => Some(1024),
            _ => None,
        }
    }

    fn default_embedding_model(&self) -> Option<&str> {
        Some("jina-embeddings-v3")
    }

    fn max_batch_size(&self) -> usize {
        2048
    }

    fn supports_dimensions(&self, model: &str) -> bool {
        model == "jina-embeddings-v3"
    }

    fn supported_embedding_models(&self) -> Option<&[&str]> {
        Some(&[
            "jina-embeddings-v3",
            "jina-embeddings-v2-base-en",
            "jina-embeddings-v2-base-code",
            "jina-clip-v2",
        ])
    }
}

#[derive(Debug, Serialize)]
struct JinaEmbedRequestFull {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task: Option<String>,
}

#[derive(Debug, Deserialize)]
struct JinaEmbedResponseFull {
    data: Vec<JinaEmbedding>,
    #[serde(default)]
    usage: Option<JinaUsage>,
}

#[derive(Debug, Deserialize)]
struct JinaUsage {
    total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::Provider;

    #[test]
    fn test_provider_creation() {
        let provider = JinaProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(Provider::name(&provider), "jina");
    }

    #[test]
    fn test_provider_with_api_key() {
        let provider = JinaProvider::with_api_key("test-key").unwrap();
        assert_eq!(Provider::name(&provider), "jina");
    }

    #[test]
    fn test_embeddings_url() {
        let provider = JinaProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(
            provider.embeddings_url(),
            "https://api.jina.ai/v1/embeddings"
        );
    }

    #[test]
    fn test_rerank_url() {
        let provider = JinaProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.rerank_url(), "https://api.jina.ai/v1/rerank");
    }

    #[test]
    fn test_embeddings_url_custom_base() {
        let mut config = ProviderConfig::new("test-key");
        config.base_url = Some("https://custom.jina.ai".to_string());
        let provider = JinaProvider::new(config).unwrap();
        assert_eq!(
            provider.embeddings_url(),
            "https://custom.jina.ai/embeddings"
        );
    }

    #[test]
    fn test_embedding_dimensions() {
        let provider = JinaProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(
            provider.embedding_dimensions("jina-embeddings-v3"),
            Some(1024)
        );
        assert_eq!(
            provider.embedding_dimensions("jina-embeddings-v2-base-en"),
            Some(768)
        );
        assert_eq!(
            provider.embedding_dimensions("jina-embeddings-v2-base-code"),
            Some(768)
        );
        assert_eq!(provider.embedding_dimensions("jina-clip-v2"), Some(1024));
        assert_eq!(provider.embedding_dimensions("unknown-model"), None);
    }

    #[test]
    fn test_default_embedding_model() {
        let provider = JinaProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(
            provider.default_embedding_model(),
            Some("jina-embeddings-v3")
        );
    }

    #[test]
    fn test_max_batch_size() {
        let provider = JinaProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert_eq!(provider.max_batch_size(), 2048);
    }

    #[test]
    fn test_supports_dimensions() {
        let provider = JinaProvider::new(ProviderConfig::new("test-key")).unwrap();
        assert!(provider.supports_dimensions("jina-embeddings-v3"));
        assert!(!provider.supports_dimensions("jina-embeddings-v2-base-en"));
    }

    #[test]
    fn test_supported_embedding_models() {
        let provider = JinaProvider::new(ProviderConfig::new("test-key")).unwrap();
        let models = provider.supported_embedding_models();
        assert!(models.is_some());
        let models = models.unwrap();
        assert!(models.contains(&"jina-embeddings-v3"));
        assert!(models.contains(&"jina-embeddings-v2-base-en"));
        assert!(models.contains(&"jina-clip-v2"));
    }

    #[test]
    fn test_embed_request_serialization() {
        let request = JinaEmbedRequest {
            model: "jina-embeddings-v3".to_string(),
            input: vec!["Hello".to_string(), "World".to_string()],
            embedding_type: None,
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("jina-embeddings-v3"));
        assert!(json.contains("Hello"));
        assert!(json.contains("World"));
    }

    #[test]
    fn test_rerank_request_serialization() {
        let request = JinaRerankRequest {
            model: "jina-reranker-v2-base-multilingual".to_string(),
            query: "What is AI?".to_string(),
            documents: vec!["AI is...".to_string(), "Machine learning...".to_string()],
            top_n: Some(5),
            return_documents: Some(true),
        };

        let json = serde_json::to_string(&request).unwrap();
        assert!(json.contains("jina-reranker"));
        assert!(json.contains("What is AI?"));
    }

    #[test]
    fn test_rerank_result_deserialization() {
        let json = r#"{
            "index": 0,
            "relevance_score": 0.95,
            "document": {"text": "AI is..."}
        }"#;

        let result: JinaRerankResult = serde_json::from_str(json).unwrap();
        assert_eq!(result.index, 0);
        assert_eq!(result.relevance_score, 0.95);
        assert!(result.document.is_some());
        assert_eq!(result.document.unwrap().text, Some("AI is...".to_string()));
    }
}
