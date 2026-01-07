//! Specialized APIs for Python bindings.
//!
//! Provides access to specialized NLP services:
//! - Ranking: Document relevance ranking
//! - Reranking: Semantic search result reranking
//! - Moderation: Content safety and policy checking
//! - Classification: Text classification and sentiment analysis

use pyo3::prelude::*;

// ============================================================================
// RANKING API
// ============================================================================

/// Request for document ranking.
#[pyclass(name = "RankingRequest")]
pub struct PyRankingRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    pub top_k: Option<usize>,
}

#[pymethods]
impl PyRankingRequest {
    /// Create a new ranking request.
    ///
    /// Args:
    ///     model: Model identifier in "provider/model" format (e.g., "cohere/rerank-english-v3.0")
    ///     query: The query to rank documents against
    ///     documents: List of documents to rank
    #[new]
    pub fn new(model: String, query: String, documents: Vec<String>) -> Self {
        Self {
            model,
            query,
            documents,
            top_k: None,
        }
    }

    /// Set the number of top results to return.
    #[pyo3(text_signature = "(self, top_k)")]
    pub fn with_top_k(&self, top_k: usize) -> Self {
        Self {
            top_k: Some(top_k),
            ..(*self).clone()
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RankingRequest(model='{}', query='{}', documents={}, top_k={:?})",
            self.model,
            if self.query.len() > 50 {
                format!("{}...", &self.query[..50])
            } else {
                self.query.clone()
            },
            self.documents.len(),
            self.top_k
        )
    }
}

impl Clone for PyRankingRequest {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            query: self.query.clone(),
            documents: self.documents.clone(),
            top_k: self.top_k,
        }
    }
}

/// Ranking result with score.
#[pyclass(name = "RankedDocument")]
pub struct PyRankedDocument {
    pub index: usize,
    pub document: String,
    pub score: f64,
}

#[pymethods]
impl PyRankedDocument {
    /// Create a new ranked document.
    #[new]
    pub fn new(index: usize, document: String, score: f64) -> Self {
        Self {
            index,
            document,
            score,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RankedDocument(index={}, score={:.4}, doc_len={})",
            self.index,
            self.score,
            self.document.len()
        )
    }
}

impl Clone for PyRankedDocument {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            document: self.document.clone(),
            score: self.score,
        }
    }
}

/// Response from ranking request.
#[pyclass(name = "RankingResponse")]
pub struct PyRankingResponse {
    pub results: Vec<PyRankedDocument>,
}

#[pymethods]
impl PyRankingResponse {
    /// Create a new ranking response.
    #[new]
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Get number of ranked documents.
    #[getter]
    pub fn count(&self) -> usize {
        self.results.len()
    }

    /// Get top result.
    #[pyo3(text_signature = "(self)")]
    pub fn first(&self) -> Option<PyRankedDocument> {
        self.results.first().cloned()
    }

    fn __repr__(&self) -> String {
        format!("RankingResponse(count={})", self.count())
    }
}

impl Clone for PyRankingResponse {
    fn clone(&self) -> Self {
        Self {
            results: self.results.clone(),
        }
    }
}

// ============================================================================
// RERANKING API
// ============================================================================

/// Request for reranking search results.
#[pyclass(name = "RerankingRequest")]
pub struct PyRerankingRequest {
    pub model: String,
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<usize>,
}

#[pymethods]
impl PyRerankingRequest {
    /// Create a new reranking request.
    ///
    /// Args:
    ///     model: Model identifier in "provider/model" format (e.g., "voyage/rerank-2")
    ///     query: The query to rerank documents against
    ///     documents: List of documents to rerank
    #[new]
    pub fn new(model: String, query: String, documents: Vec<String>) -> Self {
        Self {
            model,
            query,
            documents,
            top_n: None,
        }
    }

    /// Set the number of top results to return.
    #[pyo3(text_signature = "(self, top_n)")]
    pub fn with_top_n(&self, top_n: usize) -> Self {
        Self {
            top_n: Some(top_n),
            ..(*self).clone()
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RerankingRequest(model='{}', query='{}', documents={}, top_n={:?})",
            self.model,
            if self.query.len() > 50 {
                format!("{}...", &self.query[..50])
            } else {
                self.query.clone()
            },
            self.documents.len(),
            self.top_n
        )
    }
}

impl Clone for PyRerankingRequest {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            query: self.query.clone(),
            documents: self.documents.clone(),
            top_n: self.top_n,
        }
    }
}

/// Reranked result.
#[pyclass(name = "RerankedResult")]
pub struct PyRerankedResult {
    pub index: usize,
    pub document: String,
    pub relevance_score: f64,
}

#[pymethods]
impl PyRerankedResult {
    /// Create a new reranked result.
    #[new]
    pub fn new(index: usize, document: String, relevance_score: f64) -> Self {
        Self {
            index,
            document,
            relevance_score,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RerankedResult(index={}, score={:.4})",
            self.index, self.relevance_score
        )
    }
}

impl Clone for PyRerankedResult {
    fn clone(&self) -> Self {
        Self {
            index: self.index,
            document: self.document.clone(),
            relevance_score: self.relevance_score,
        }
    }
}

/// Response from reranking request.
#[pyclass(name = "RerankingResponse")]
pub struct PyRerankingResponse {
    pub results: Vec<PyRerankedResult>,
}

#[pymethods]
impl PyRerankingResponse {
    /// Create a new reranking response.
    #[new]
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Get number of results.
    #[getter]
    pub fn count(&self) -> usize {
        self.results.len()
    }

    /// Get top result.
    #[pyo3(text_signature = "(self)")]
    pub fn first(&self) -> Option<PyRerankedResult> {
        self.results.first().cloned()
    }

    fn __repr__(&self) -> String {
        format!("RerankingResponse(count={})", self.count())
    }
}

impl Clone for PyRerankingResponse {
    fn clone(&self) -> Self {
        Self {
            results: self.results.clone(),
        }
    }
}

// ============================================================================
// MODERATION API
// ============================================================================

/// Request for content moderation.
#[pyclass(name = "ModerationRequest")]
pub struct PyModerationRequest {
    pub model: String,
    pub text: String,
}

#[pymethods]
impl PyModerationRequest {
    /// Create a new moderation request.
    ///
    /// Args:
    ///     model: Model identifier in "provider/model" format (e.g., "openai/omni-moderation-latest")
    ///     text: The text content to moderate
    #[new]
    pub fn new(model: String, text: String) -> Self {
        Self { model, text }
    }

    fn __repr__(&self) -> String {
        format!(
            "ModerationRequest(model='{}', text_len={})",
            self.model,
            self.text.len()
        )
    }
}

impl Clone for PyModerationRequest {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            text: self.text.clone(),
        }
    }
}

/// Moderation category scores.
#[pyclass(name = "ModerationScores")]
pub struct PyModerationScores {
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

#[pymethods]
impl PyModerationScores {
    /// Create new moderation scores.
    #[new]
    pub fn new() -> Self {
        Self {
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
        }
    }

    /// Get highest category score.
    #[pyo3(text_signature = "(self)")]
    pub fn max_score(&self) -> f64 {
        *[
            self.hate,
            self.hate_threatening,
            self.harassment,
            self.harassment_threatening,
            self.self_harm,
            self.self_harm_intent,
            self.self_harm_instructions,
            self.sexual,
            self.sexual_minors,
            self.violence,
            self.violence_graphic,
        ]
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(&0.0)
    }

    fn __repr__(&self) -> String {
        format!(
            "ModerationScores(max={:.4}, flagged={})",
            self.max_score(),
            self.max_score() > 0.5
        )
    }
}

impl Clone for PyModerationScores {
    fn clone(&self) -> Self {
        Self {
            hate: self.hate,
            hate_threatening: self.hate_threatening,
            harassment: self.harassment,
            harassment_threatening: self.harassment_threatening,
            self_harm: self.self_harm,
            self_harm_intent: self.self_harm_intent,
            self_harm_instructions: self.self_harm_instructions,
            sexual: self.sexual,
            sexual_minors: self.sexual_minors,
            violence: self.violence,
            violence_graphic: self.violence_graphic,
        }
    }
}

/// Response from moderation request.
#[pyclass(name = "ModerationResponse")]
pub struct PyModerationResponse {
    pub flagged: bool,
    pub scores: PyModerationScores,
}

#[pymethods]
impl PyModerationResponse {
    /// Create a new moderation response.
    #[new]
    pub fn new(flagged: bool) -> Self {
        Self {
            flagged,
            scores: PyModerationScores::new(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ModerationResponse(flagged={}, max_score={:.4})",
            self.flagged,
            self.scores.max_score()
        )
    }
}

impl Clone for PyModerationResponse {
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
#[pyclass(name = "ClassificationRequest")]
pub struct PyClassificationRequest {
    pub model: String,
    pub text: String,
    pub labels: Vec<String>,
}

#[pymethods]
impl PyClassificationRequest {
    /// Create a new classification request.
    ///
    /// Args:
    ///     model: Model identifier in "provider/model" format (e.g., "cohere/classify")
    ///     text: The text to classify
    ///     labels: List of possible classification labels
    #[new]
    pub fn new(model: String, text: String, labels: Vec<String>) -> Self {
        Self { model, text, labels }
    }

    fn __repr__(&self) -> String {
        format!(
            "ClassificationRequest(model='{}', text_len={}, labels={})",
            self.model,
            self.text.len(),
            self.labels.len()
        )
    }
}

impl Clone for PyClassificationRequest {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            text: self.text.clone(),
            labels: self.labels.clone(),
        }
    }
}

/// Classification result with label and confidence.
#[pyclass(name = "ClassificationResult")]
pub struct PyClassificationResult {
    pub label: String,
    pub confidence: f64,
}

#[pymethods]
impl PyClassificationResult {
    /// Create a new classification result.
    #[new]
    pub fn new(label: String, confidence: f64) -> Self {
        Self { label, confidence }
    }

    fn __repr__(&self) -> String {
        format!(
            "ClassificationResult(label='{}', confidence={:.4})",
            self.label, self.confidence
        )
    }
}

impl Clone for PyClassificationResult {
    fn clone(&self) -> Self {
        Self {
            label: self.label.clone(),
            confidence: self.confidence,
        }
    }
}

/// Response from classification request.
#[pyclass(name = "ClassificationResponse")]
pub struct PyClassificationResponse {
    pub results: Vec<PyClassificationResult>,
}

#[pymethods]
impl PyClassificationResponse {
    /// Create a new classification response.
    #[new]
    pub fn new() -> Self {
        Self {
            results: Vec::new(),
        }
    }

    /// Get top classification.
    #[pyo3(text_signature = "(self)")]
    pub fn top(&self) -> Option<PyClassificationResult> {
        self.results.first().cloned()
    }

    /// Get number of classifications.
    #[getter]
    pub fn count(&self) -> usize {
        self.results.len()
    }

    fn __repr__(&self) -> String {
        format!("ClassificationResponse(count={})", self.count())
    }
}

impl Clone for PyClassificationResponse {
    fn clone(&self) -> Self {
        Self {
            results: self.results.clone(),
        }
    }
}
