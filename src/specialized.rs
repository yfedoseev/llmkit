//! Specialized AI APIs for ranking, moderation, and classification.
//!
//! This module provides unified interfaces for specialized AI tasks that go beyond
//! text generation, including document ranking/reranking, content moderation,
//! and text classification.
//!
//! # Ranking Example
//!
//! ```ignore
//! use modelsuite::{RankingProvider, RankingRequest};
//!
//! let provider = CohereProvider::from_env()?;
//!
//! let request = RankingRequest::new(
//!     "rerank-english-v3.0",
//!     "What is the capital of France?",
//!     vec![
//!         "Paris is the capital of France.",
//!         "Berlin is the capital of Germany.",
//!         "London is the capital of England.",
//!     ],
//! );
//!
//! let response = provider.rank(request).await?;
//! for result in &response.results {
//!     println!("Score: {:.4} - {}", result.score, result.document.as_ref().unwrap());
//! }
//! ```
//!
//! # Moderation Example
//!
//! ```ignore
//! use modelsuite::{ModerationProvider, ModerationRequest};
//!
//! let provider = OpenAIModerationProvider::from_env()?;
//!
//! let request = ModerationRequest::new("omni-moderation-latest", "Some user content to check");
//!
//! let response = provider.moderate(request).await?;
//! if response.flagged {
//!     println!("Content was flagged!");
//!     for category in response.flagged_categories() {
//!         println!("  - {}", category);
//!     }
//! }
//! ```
//!
//! # Classification Example
//!
//! ```ignore
//! use modelsuite::{ClassificationProvider, ClassificationRequest};
//!
//! let provider = CohereClassifyProvider::from_env()?;
//!
//! let request = ClassificationRequest::new(
//!     "embed-english-v3.0",
//!     "I love this product!",
//!     vec!["positive", "negative", "neutral"],
//! );
//!
//! let response = provider.classify(request).await?;
//! let top = &response.predictions[0];
//! println!("Predicted: {} (score: {:.4})", top.label, top.score);
//! ```

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::Result;

// ============================================================================
// Ranking / Reranking
// ============================================================================

/// Request for ranking/reranking documents.
#[derive(Debug, Clone)]
pub struct RankingRequest {
    /// The model to use for ranking.
    pub model: String,
    /// The query to rank documents against.
    pub query: String,
    /// The documents to rank.
    pub documents: Vec<String>,
    /// Maximum number of results to return.
    pub top_k: Option<usize>,
    /// Whether to return document content in results.
    pub return_documents: Option<bool>,
    /// Maximum number of tokens per document (for truncation).
    pub max_chunks_per_doc: Option<usize>,
}

impl RankingRequest {
    /// Create a new ranking request.
    pub fn new(
        model: impl Into<String>,
        query: impl Into<String>,
        documents: Vec<impl Into<String>>,
    ) -> Self {
        Self {
            model: model.into(),
            query: query.into(),
            documents: documents.into_iter().map(|d| d.into()).collect(),
            top_k: None,
            return_documents: None,
            max_chunks_per_doc: None,
        }
    }

    /// Set the maximum number of results to return.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    /// Include document content in results.
    pub fn with_documents(mut self) -> Self {
        self.return_documents = Some(true);
        self
    }

    /// Set max chunks per document for long document handling.
    pub fn with_max_chunks_per_doc(mut self, max_chunks: usize) -> Self {
        self.max_chunks_per_doc = Some(max_chunks);
        self
    }
}

/// Response from a ranking request.
#[derive(Debug, Clone)]
pub struct RankingResponse {
    /// Ranked results, sorted by relevance (highest first).
    pub results: Vec<RankedDocument>,
    /// The model used for ranking.
    pub model: String,
    /// API metadata (billing, etc.)
    pub meta: Option<RankingMeta>,
}

impl RankingResponse {
    /// Create a new ranking response.
    pub fn new(model: impl Into<String>, results: Vec<RankedDocument>) -> Self {
        Self {
            model: model.into(),
            results,
            meta: None,
        }
    }

    /// Get the top-ranked document.
    pub fn top(&self) -> Option<&RankedDocument> {
        self.results.first()
    }

    /// Get the indices of ranked documents in order.
    pub fn ranked_indices(&self) -> Vec<usize> {
        self.results.iter().map(|r| r.index).collect()
    }
}

/// A ranked document with its relevance score.
#[derive(Debug, Clone)]
pub struct RankedDocument {
    /// Original index in the input documents array.
    pub index: usize,
    /// Relevance score (higher is more relevant).
    pub score: f32,
    /// The document content (if return_documents was true).
    pub document: Option<String>,
}

impl RankedDocument {
    /// Create a new ranked document.
    pub fn new(index: usize, score: f32) -> Self {
        Self {
            index,
            score,
            document: None,
        }
    }

    /// Set the document content.
    pub fn with_document(mut self, document: impl Into<String>) -> Self {
        self.document = Some(document.into());
        self
    }
}

/// Metadata from ranking API response.
#[derive(Debug, Clone, Default)]
pub struct RankingMeta {
    /// Billing tokens used.
    pub billed_units: Option<u64>,
    /// API version.
    pub api_version: Option<String>,
}

/// Trait for providers that support document ranking/reranking.
#[async_trait]
pub trait RankingProvider: Send + Sync {
    /// Get the provider name.
    fn name(&self) -> &str;

    /// Rank documents by relevance to a query.
    async fn rank(&self, request: RankingRequest) -> Result<RankingResponse>;

    /// Get the default model for ranking.
    fn default_ranking_model(&self) -> Option<&str> {
        None
    }

    /// Get the maximum number of documents that can be ranked in one request.
    fn max_documents(&self) -> usize {
        1000
    }

    /// Get the maximum query length in characters.
    fn max_query_length(&self) -> usize {
        2048
    }
}

// ============================================================================
// Moderation
// ============================================================================

/// Request for content moderation.
#[derive(Debug, Clone)]
pub struct ModerationRequest {
    /// The model to use for moderation.
    pub model: String,
    /// The text content to moderate.
    pub input: String,
    /// Additional inputs (for multi-modal moderation).
    pub inputs: Option<Vec<ModerationInput>>,
}

impl ModerationRequest {
    /// Create a new moderation request.
    pub fn new(model: impl Into<String>, input: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            inputs: None,
        }
    }

    /// Add multiple inputs for moderation.
    pub fn with_inputs(mut self, inputs: Vec<ModerationInput>) -> Self {
        self.inputs = Some(inputs);
        self
    }
}

/// Input types for multi-modal moderation.
#[derive(Debug, Clone)]
pub enum ModerationInput {
    /// Text content.
    Text(String),
    /// Image URL.
    ImageUrl(String),
    /// Base64-encoded image.
    ImageBase64 { data: String, media_type: String },
}

/// Response from a moderation request.
#[derive(Debug, Clone)]
pub struct ModerationResponse {
    /// Whether any content was flagged.
    pub flagged: bool,
    /// Category flags indicating which categories were triggered.
    pub categories: ModerationCategories,
    /// Category scores (0.0 to 1.0).
    pub category_scores: ModerationScores,
    /// Model used for moderation.
    pub model: String,
}

impl ModerationResponse {
    /// Create a new moderation response.
    pub fn new(flagged: bool) -> Self {
        Self {
            flagged,
            categories: ModerationCategories::default(),
            category_scores: ModerationScores::default(),
            model: String::new(),
        }
    }

    /// Set the model name.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = model.into();
        self
    }

    /// Set category flags.
    pub fn with_categories(mut self, categories: ModerationCategories) -> Self {
        self.categories = categories;
        self
    }

    /// Set category scores.
    pub fn with_scores(mut self, scores: ModerationScores) -> Self {
        self.category_scores = scores;
        self
    }

    /// Get a list of flagged category names.
    pub fn flagged_categories(&self) -> Vec<&'static str> {
        let mut result = Vec::new();
        if self.categories.hate {
            result.push("hate");
        }
        if self.categories.hate_threatening {
            result.push("hate/threatening");
        }
        if self.categories.harassment {
            result.push("harassment");
        }
        if self.categories.harassment_threatening {
            result.push("harassment/threatening");
        }
        if self.categories.self_harm {
            result.push("self-harm");
        }
        if self.categories.self_harm_intent {
            result.push("self-harm/intent");
        }
        if self.categories.self_harm_instructions {
            result.push("self-harm/instructions");
        }
        if self.categories.sexual {
            result.push("sexual");
        }
        if self.categories.sexual_minors {
            result.push("sexual/minors");
        }
        if self.categories.violence {
            result.push("violence");
        }
        if self.categories.violence_graphic {
            result.push("violence/graphic");
        }
        if self.categories.illicit {
            result.push("illicit");
        }
        if self.categories.illicit_violent {
            result.push("illicit/violent");
        }
        result
    }
}

/// Moderation category flags.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModerationCategories {
    /// Hate speech.
    pub hate: bool,
    /// Hate speech with threatening language.
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: bool,
    /// Harassment.
    pub harassment: bool,
    /// Harassment with threatening language.
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: bool,
    /// Self-harm content.
    #[serde(rename = "self-harm")]
    pub self_harm: bool,
    /// Self-harm with intent.
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: bool,
    /// Self-harm instructions.
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: bool,
    /// Sexual content.
    pub sexual: bool,
    /// Sexual content involving minors.
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: bool,
    /// Violent content.
    pub violence: bool,
    /// Graphic violence.
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: bool,
    /// Illicit activity.
    #[serde(default)]
    pub illicit: bool,
    /// Violent illicit activity.
    #[serde(default, rename = "illicit/violent")]
    pub illicit_violent: bool,
}

/// Moderation category confidence scores (0.0 to 1.0).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModerationScores {
    /// Hate speech score.
    pub hate: f32,
    /// Hate/threatening score.
    #[serde(rename = "hate/threatening")]
    pub hate_threatening: f32,
    /// Harassment score.
    pub harassment: f32,
    /// Harassment/threatening score.
    #[serde(rename = "harassment/threatening")]
    pub harassment_threatening: f32,
    /// Self-harm score.
    #[serde(rename = "self-harm")]
    pub self_harm: f32,
    /// Self-harm/intent score.
    #[serde(rename = "self-harm/intent")]
    pub self_harm_intent: f32,
    /// Self-harm/instructions score.
    #[serde(rename = "self-harm/instructions")]
    pub self_harm_instructions: f32,
    /// Sexual score.
    pub sexual: f32,
    /// Sexual/minors score.
    #[serde(rename = "sexual/minors")]
    pub sexual_minors: f32,
    /// Violence score.
    pub violence: f32,
    /// Violence/graphic score.
    #[serde(rename = "violence/graphic")]
    pub violence_graphic: f32,
    /// Illicit score.
    #[serde(default)]
    pub illicit: f32,
    /// Illicit/violent score.
    #[serde(default, rename = "illicit/violent")]
    pub illicit_violent: f32,
}

/// Trait for providers that support content moderation.
#[async_trait]
pub trait ModerationProvider: Send + Sync {
    /// Get the provider name.
    fn name(&self) -> &str;

    /// Moderate content for policy violations.
    async fn moderate(&self, request: ModerationRequest) -> Result<ModerationResponse>;

    /// Get the default model for moderation.
    fn default_moderation_model(&self) -> Option<&str> {
        None
    }

    /// Check if the provider supports multi-modal moderation (images).
    fn supports_multimodal(&self) -> bool {
        false
    }
}

// ============================================================================
// Classification
// ============================================================================

/// Request for text classification.
#[derive(Debug, Clone)]
pub struct ClassificationRequest {
    /// The model to use for classification.
    pub model: String,
    /// The text to classify.
    pub input: String,
    /// The possible labels/classes.
    pub labels: Vec<String>,
    /// Whether to allow multiple labels per input.
    pub multi_label: Option<bool>,
    /// Optional examples for few-shot classification.
    pub examples: Option<Vec<ClassificationExample>>,
}

impl ClassificationRequest {
    /// Create a new classification request.
    pub fn new(
        model: impl Into<String>,
        input: impl Into<String>,
        labels: Vec<impl Into<String>>,
    ) -> Self {
        Self {
            model: model.into(),
            input: input.into(),
            labels: labels.into_iter().map(|l| l.into()).collect(),
            multi_label: None,
            examples: None,
        }
    }

    /// Enable multi-label classification.
    pub fn with_multi_label(mut self) -> Self {
        self.multi_label = Some(true);
        self
    }

    /// Add examples for few-shot classification.
    pub fn with_examples(mut self, examples: Vec<ClassificationExample>) -> Self {
        self.examples = Some(examples);
        self
    }
}

/// Example for few-shot classification.
#[derive(Debug, Clone)]
pub struct ClassificationExample {
    /// The example text.
    pub text: String,
    /// The label for this example.
    pub label: String,
}

impl ClassificationExample {
    /// Create a new classification example.
    pub fn new(text: impl Into<String>, label: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            label: label.into(),
        }
    }
}

/// Response from a classification request.
#[derive(Debug, Clone)]
pub struct ClassificationResponse {
    /// Predictions sorted by confidence (highest first).
    pub predictions: Vec<ClassificationPrediction>,
    /// The model used for classification.
    pub model: String,
}

impl ClassificationResponse {
    /// Create a new classification response.
    pub fn new(model: impl Into<String>, predictions: Vec<ClassificationPrediction>) -> Self {
        Self {
            model: model.into(),
            predictions,
        }
    }

    /// Get the top prediction.
    pub fn top(&self) -> Option<&ClassificationPrediction> {
        self.predictions.first()
    }

    /// Get the predicted label.
    pub fn label(&self) -> Option<&str> {
        self.predictions.first().map(|p| p.label.as_str())
    }

    /// Get the confidence score for a specific label.
    pub fn score_for(&self, label: &str) -> Option<f32> {
        self.predictions
            .iter()
            .find(|p| p.label == label)
            .map(|p| p.score)
    }
}

/// A classification prediction with confidence score.
#[derive(Debug, Clone)]
pub struct ClassificationPrediction {
    /// The predicted label.
    pub label: String,
    /// Confidence score (0.0 to 1.0).
    pub score: f32,
}

impl ClassificationPrediction {
    /// Create a new classification prediction.
    pub fn new(label: impl Into<String>, score: f32) -> Self {
        Self {
            label: label.into(),
            score,
        }
    }
}

/// Trait for providers that support text classification.
#[async_trait]
pub trait ClassificationProvider: Send + Sync {
    /// Get the provider name.
    fn name(&self) -> &str;

    /// Classify text into one or more labels.
    async fn classify(&self, request: ClassificationRequest) -> Result<ClassificationResponse>;

    /// Get the default model for classification.
    fn default_classification_model(&self) -> Option<&str> {
        None
    }

    /// Get the maximum number of labels supported.
    fn max_labels(&self) -> usize {
        100
    }

    /// Check if few-shot classification is supported.
    fn supports_few_shot(&self) -> bool {
        false
    }
}

// ============================================================================
// Model Registry
// ============================================================================

/// Information about a ranking model.
#[derive(Debug, Clone)]
pub struct RankingModelInfo {
    /// Model ID.
    pub id: &'static str,
    /// Provider.
    pub provider: &'static str,
    /// Max documents per request.
    pub max_documents: usize,
    /// Max query tokens.
    pub max_query_tokens: usize,
    /// Price per 1000 searches (USD).
    pub price_per_1k_searches: f64,
}

/// Registry of known ranking models.
pub static RANKING_MODELS: &[RankingModelInfo] = &[
    // Cohere
    RankingModelInfo {
        id: "rerank-english-v3.0",
        provider: "cohere",
        max_documents: 1000,
        max_query_tokens: 2048,
        price_per_1k_searches: 2.00,
    },
    RankingModelInfo {
        id: "rerank-multilingual-v3.0",
        provider: "cohere",
        max_documents: 1000,
        max_query_tokens: 2048,
        price_per_1k_searches: 2.00,
    },
    // Voyage
    RankingModelInfo {
        id: "rerank-2",
        provider: "voyage",
        max_documents: 1000,
        max_query_tokens: 4000,
        price_per_1k_searches: 0.05,
    },
    RankingModelInfo {
        id: "rerank-lite-2",
        provider: "voyage",
        max_documents: 1000,
        max_query_tokens: 4000,
        price_per_1k_searches: 0.02,
    },
    // Jina
    RankingModelInfo {
        id: "jina-reranker-v2-base-multilingual",
        provider: "jina",
        max_documents: 500,
        max_query_tokens: 8192,
        price_per_1k_searches: 0.02,
    },
];

/// Information about a moderation model.
#[derive(Debug, Clone)]
pub struct ModerationModelInfo {
    /// Model ID.
    pub id: &'static str,
    /// Provider.
    pub provider: &'static str,
    /// Supports multi-modal (images).
    pub supports_images: bool,
    /// Price per 1000 requests (USD).
    pub price_per_1k_requests: f64,
}

/// Registry of known moderation models.
pub static MODERATION_MODELS: &[ModerationModelInfo] = &[
    // OpenAI
    ModerationModelInfo {
        id: "omni-moderation-latest",
        provider: "openai",
        supports_images: true,
        price_per_1k_requests: 0.0, // Free
    },
    ModerationModelInfo {
        id: "text-moderation-latest",
        provider: "openai",
        supports_images: false,
        price_per_1k_requests: 0.0, // Free
    },
    ModerationModelInfo {
        id: "text-moderation-stable",
        provider: "openai",
        supports_images: false,
        price_per_1k_requests: 0.0, // Free
    },
];

/// Get ranking model info by ID.
pub fn get_ranking_model_info(model_id: &str) -> Option<&'static RankingModelInfo> {
    RANKING_MODELS.iter().find(|m| m.id == model_id)
}

/// Get moderation model info by ID.
pub fn get_moderation_model_info(model_id: &str) -> Option<&'static ModerationModelInfo> {
    MODERATION_MODELS.iter().find(|m| m.id == model_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Ranking tests
    #[test]
    fn test_ranking_request_builder() {
        let request = RankingRequest::new(
            "rerank-english-v3.0",
            "What is the capital?",
            vec!["Paris is the capital", "Berlin is a city"],
        )
        .with_top_k(5)
        .with_documents();

        assert_eq!(request.model, "rerank-english-v3.0");
        assert_eq!(request.query, "What is the capital?");
        assert_eq!(request.documents.len(), 2);
        assert_eq!(request.top_k, Some(5));
        assert_eq!(request.return_documents, Some(true));
    }

    #[test]
    fn test_ranking_response() {
        let results = vec![
            RankedDocument::new(1, 0.95).with_document("Top doc"),
            RankedDocument::new(0, 0.8),
        ];
        let response = RankingResponse::new("rerank-english-v3.0", results);

        assert_eq!(response.top().unwrap().score, 0.95);
        assert_eq!(response.ranked_indices(), vec![1, 0]);
    }

    // Moderation tests
    #[test]
    fn test_moderation_request() {
        let request = ModerationRequest::new("omni-moderation-latest", "Some text to check");
        assert_eq!(request.model, "omni-moderation-latest");
        assert_eq!(request.input, "Some text to check");
    }

    #[test]
    fn test_moderation_response() {
        let categories = ModerationCategories {
            hate: true,
            violence: true,
            ..Default::default()
        };

        let response = ModerationResponse::new(true)
            .with_model("omni-moderation-latest")
            .with_categories(categories);

        assert!(response.flagged);
        let flagged = response.flagged_categories();
        assert!(flagged.contains(&"hate"));
        assert!(flagged.contains(&"violence"));
        assert!(!flagged.contains(&"sexual"));
    }

    // Classification tests
    #[test]
    fn test_classification_request_builder() {
        let request = ClassificationRequest::new(
            "embed-english-v3.0",
            "I love this product!",
            vec!["positive", "negative", "neutral"],
        )
        .with_multi_label()
        .with_examples(vec![
            ClassificationExample::new("Great!", "positive"),
            ClassificationExample::new("Terrible", "negative"),
        ]);

        assert_eq!(request.model, "embed-english-v3.0");
        assert_eq!(request.input, "I love this product!");
        assert_eq!(request.labels.len(), 3);
        assert_eq!(request.multi_label, Some(true));
        assert_eq!(request.examples.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_classification_response() {
        let predictions = vec![
            ClassificationPrediction::new("positive", 0.92),
            ClassificationPrediction::new("neutral", 0.06),
            ClassificationPrediction::new("negative", 0.02),
        ];
        let response = ClassificationResponse::new("model", predictions);

        assert_eq!(response.label(), Some("positive"));
        assert_eq!(response.top().unwrap().score, 0.92);
        assert_eq!(response.score_for("neutral"), Some(0.06));
        assert_eq!(response.score_for("unknown"), None);
    }

    // Registry tests
    #[test]
    fn test_ranking_model_registry() {
        let model = get_ranking_model_info("rerank-english-v3.0");
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.provider, "cohere");
        assert_eq!(model.max_documents, 1000);
    }

    #[test]
    fn test_moderation_model_registry() {
        let model = get_moderation_model_info("omni-moderation-latest");
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.provider, "openai");
        assert!(model.supports_images);
    }
}
