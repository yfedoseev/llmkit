//! Specialized APIs for Node.js/TypeScript.
//!
//! Provides access to specialized NLP services:
//! - Ranking: Document relevance ranking
//! - Reranking: Semantic search result reranking
//! - Moderation: Content safety and policy checking
//! - Classification: Text classification and sentiment analysis

use napi::bindgen_prelude::*;
use napi_derive::napi;

// ============================================================================
// RANKING API
// ============================================================================

/// Request for document ranking.
#[napi]
pub struct JsRankingRequest {
    pub query: String,
    pub documents: Vec<String>,
    pub top_k: Option<u32>,
}

#[napi]
impl JsRankingRequest {
    /// Create a new ranking request.
    #[napi(constructor)]
    pub fn new(query: String, documents: Vec<String>) -> Self {
        Self {
            query,
            documents,
            top_k: None,
        }
    }

    /// Set the number of top results to return.
    #[napi(js_name = "withTopK")]
    pub fn with_top_k(&self, top_k: u32) -> Self {
        Self {
            top_k: Some(top_k),
            ..(*self).clone()
        }
    }
}

impl Clone for JsRankingRequest {
    fn clone(&self) -> Self {
        Self {
            query: self.query.clone(),
            documents: self.documents.clone(),
            top_k: self.top_k,
        }
    }
}

/// Ranking result with score.
#[napi]
pub struct JsRankedDocument {
    pub index: u32,
    pub document: String,
    pub score: f64,
}

#[napi]
impl JsRankedDocument {
    /// Create a new ranked document.
    #[napi(constructor)]
    pub fn new(index: u32, document: String, score: f64) -> Self {
        Self {
            index,
            document,
            score,
        }
    }
}

impl Clone for JsRankedDocument {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            document: self.document.clone(),
            score: self.score,
        }
    }
}

/// Response from ranking request.
#[napi(object)]
#[derive(Clone)]
pub struct JsRankingResponse {
    pub results: Vec<JsRankedDocument>,
}

// ============================================================================
// RERANKING API
// ============================================================================

/// Request for reranking search results.
#[napi]
pub struct JsRerankingRequest {
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<u32>,
}

#[napi]
impl JsRerankingRequest {
    /// Create a new reranking request.
    #[napi(constructor)]
    pub fn new(query: String, documents: Vec<String>) -> Self {
        Self {
            query,
            documents,
            top_n: None,
        }
    }

    /// Set the number of top results to return.
    #[napi(js_name = "withTopN")]
    pub fn with_top_n(&self, top_n: u32) -> Self {
        Self {
            top_n: Some(top_n),
            ..(*self).clone()
        }
    }
}

impl Clone for JsRerankingRequest {
    fn clone(&self) -> Self {
        Self {
            query: self.query.clone(),
            documents: self.documents.clone(),
            top_n: self.top_n,
        }
    }
}

/// Reranked result.
#[napi]
pub struct JsRerankedResult {
    pub index: u32,
    pub document: String,
    pub relevance_score: f64,
}

#[napi]
impl JsRerankedResult {
    /// Create a new reranked result.
    #[napi(constructor)]
    pub fn new(index: u32, document: String, relevance_score: f64) -> Self {
        Self {
            index,
            document,
            relevance_score,
        }
    }
}

impl Clone for JsRerankedResult {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            document: self.document.clone(),
            relevance_score: self.relevance_score,
        }
    }
}

/// Response from reranking request.
#[napi(object)]
#[derive(Clone)]
pub struct JsRerankingResponse {
    pub results: Vec<JsRerankedResult>,
}

// ============================================================================
// MODERATION API
// ============================================================================

/// Request for content moderation.
#[napi]
pub struct JsModerationRequest {
    pub text: String,
}

#[napi]
impl JsModerationRequest {
    /// Create a new moderation request.
    #[napi(constructor)]
    pub fn new(text: String) -> Self {
        Self { text }
    }
}

impl Clone for JsModerationRequest {
    fn clone(&self) -> Self {
        Self {
            text: self.text.clone(),
        }
    }
}

/// Moderation category scores.
#[napi(object)]
#[derive(Clone)]
pub struct JsModerationScores {
    pub hate: f64,
    pub hate_threatening: f64,
    pub harassment: f64,
    pub harassment_threatening: f64,
    pub self_harm: f64,
    pub self_harm_intent: f64,
    pub self_harm_instructions: f64,
    pub sexual: f64,
    pub sexual_minors: f64,
    pub violence: f64,
    pub violence_graphic: f64,
}

/// Response from moderation request.
#[napi]
pub struct JsModerationResponse {
    pub flagged: bool,
    pub scores: JsModerationScores,
}

#[napi]
impl JsModerationResponse {
    /// Create a new moderation response.
    #[napi(constructor)]
    pub fn new(flagged: bool) -> Self {
        Self {
            flagged,
            scores: JsModerationScores {
                hate: 0.0,
                hate_threatening: 0.0,
                harassment: 0.0,
                harassment_threatening: 0.0,
                self_harm: 0.0,
                self_harm_intent: 0.0,
                self_harm_instructions: 0.0,
                sexual: 0.0,
                sexual_minors: 0.0,
                violence: 0.0,
                violence_graphic: 0.0,
            },
        }
    }
}

impl Clone for JsModerationResponse {
    fn clone(&self) -> Self {
        Self {
            flagged: self.flagged,
            scores: self.scores.clone(),
        }
    }
}

// ============================================================================
// CLASSIFICATION API
// ============================================================================

/// Request for text classification.
#[napi]
pub struct JsClassificationRequest {
    pub text: String,
    pub labels: Vec<String>,
}

#[napi]
impl JsClassificationRequest {
    /// Create a new classification request.
    #[napi(constructor)]
    pub fn new(text: String, labels: Vec<String>) -> Self {
        Self { text, labels }
    }
}

impl Clone for JsClassificationRequest {
    fn clone(&self) -> Self {
        Self {
            text: self.text.clone(),
            labels: self.labels.clone(),
        }
    }
}

/// Classification result with label and confidence.
#[napi(object)]
#[derive(Clone)]
pub struct JsClassificationResult {
    pub label: String,
    pub confidence: f64,
}

/// Response from classification request.
#[napi(object)]
#[derive(Clone)]
pub struct JsClassificationResponse {
    pub results: Vec<JsClassificationResult>,
}

// Manual FromNapiValue implementations for component types
// These are stubs since they're only created by Rust, never deserialized from JS
impl FromNapiValue for JsRankedDocument {
    unsafe fn from_napi_value(
        _env: napi::sys::napi_env,
        _val: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        Err(napi::Error::new(
            napi::Status::InvalidArg,
            "JsRankedDocument cannot be constructed from JavaScript",
        ))
    }
}

impl FromNapiValue for JsRerankedResult {
    unsafe fn from_napi_value(
        _env: napi::sys::napi_env,
        _val: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        Err(napi::Error::new(
            napi::Status::InvalidArg,
            "JsRerankedResult cannot be constructed from JavaScript",
        ))
    }
}
