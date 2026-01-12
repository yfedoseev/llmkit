"""Unit tests for LLMKit Python specialized API bindings."""

import os

import pytest

from llmkit import (  # type: ignore[attr-defined]
    ClassificationRequest,
    ClassificationResponse,
    ClassificationResult,
    LLMKitClient,
    ModerationRequest,
    ModerationResponse,
    ModerationScores,
    RankedDocument,
    RankingRequest,
    RankingResponse,
    RerankedResult,
    RerankingRequest,
    RerankingResponse,
)


def has_ranking_provider() -> bool:
    """Check if any ranking provider API key is configured."""
    return bool(os.environ.get("COHERE_API_KEY") or os.environ.get("VOYAGE_API_KEY"))


def has_moderation_provider() -> bool:
    """Check if any moderation provider API key is configured."""
    return bool(os.environ.get("OPENAI_API_KEY"))


def has_classification_provider() -> bool:
    """Check if any classification provider API key is configured."""
    return bool(os.environ.get("COHERE_API_KEY"))


class TestRankingAPI:
    """Test document ranking API."""

    def test_create_ranking_request(self):
        """Test creating a ranking request."""
        req = RankingRequest("cohere/rerank-english-v3.0", "python programming", ["doc1", "doc2", "doc3"])
        assert req is not None
        assert req.query == "python programming"
        assert len(req.documents) == 3

    def test_ranking_with_top_k(self):
        """Test ranking request with top_k parameter."""
        req = RankingRequest("cohere/rerank-english-v3.0", "query", ["d1", "d2", "d3"]).with_top_k(2)
        assert req.top_k == 2

    def test_ranked_document(self):
        """Test RankedDocument class."""
        doc = RankedDocument(0, "Sample document", 0.95)
        assert doc.index == 0
        assert doc.score == 0.95
        assert len(doc.document) > 0

    def test_ranking_response(self):
        """Test RankingResponse class."""
        resp = RankingResponse()
        assert resp.count == 0

        resp.results = [
            RankedDocument(0, "doc1", 0.9),
            RankedDocument(1, "doc2", 0.8),
        ]
        assert resp.count == 2
        assert resp.first() is not None
        assert resp.first().score == 0.9

    @pytest.mark.skipif(not has_ranking_provider(), reason="No ranking provider API key configured")
    def test_client_rank_documents(self):
        """Test client.rank_documents method."""
        client = LLMKitClient.from_env()
        req = RankingRequest(
            "cohere/rerank-english-v3.0",
            "best python library",
            ["Django", "Flask", "FastAPI", "Tornado"],
        )
        response = client.rank_documents(req)  # type: ignore[attr-defined]

        assert isinstance(response, RankingResponse)
        assert response.count > 0
        assert response.first() is not None


class TestRerankingAPI:
    """Test semantic search reranking API."""

    def test_create_reranking_request(self):
        """Test creating a reranking request."""
        docs = ["result1", "result2", "result3"]
        req = RerankingRequest("voyage/rerank-2", "machine learning", docs)
        assert req is not None
        assert req.query == "machine learning"

    def test_reranking_with_top_n(self):
        """Test reranking with top_n parameter."""
        req = RerankingRequest("voyage/rerank-2", "query", ["d1", "d2"]).with_top_n(1)
        assert req.top_n == 1

    def test_reranked_result(self):
        """Test RerankedResult class."""
        result = RerankedResult(0, "content", 0.92)
        assert result.index == 0
        assert result.relevance_score == 0.92

    def test_reranking_response(self):
        """Test RerankingResponse class."""
        resp = RerankingResponse()
        assert resp.count == 0

        resp.results = [
            RerankedResult(0, "r1", 0.95),
            RerankedResult(1, "r2", 0.80),
        ]
        assert resp.count == 2

    @pytest.mark.skipif(not has_ranking_provider(), reason="No ranking provider API key configured")
    def test_client_rerank_results(self):
        """Test client.rerank_results method."""
        client = LLMKitClient.from_env()
        docs = [
            "Python is a programming language",
            "Django is a web framework",
            "Python snakes are reptiles",
        ]
        req = RerankingRequest("voyage/rerank-2", "Python web development", docs).with_top_n(2)
        response = client.rerank_results(req)  # type: ignore[attr-defined]

        assert isinstance(response, RerankingResponse)
        assert response.count > 0


class TestModerationAPI:
    """Test content moderation API."""

    def test_create_moderation_request(self):
        """Test creating a moderation request."""
        req = ModerationRequest("openai/omni-moderation-latest", "This is acceptable content")
        assert req is not None
        assert len(req.text) > 0

    def test_moderation_scores(self):
        """Test ModerationScores class."""
        scores = ModerationScores()
        assert scores.hate == 0.0
        assert scores.max_score() == 0.0

        scores.sexual = 0.8
        assert scores.max_score() == 0.8

    def test_moderation_response(self):
        """Test ModerationResponse class."""
        resp = ModerationResponse(False)
        assert resp.flagged is False
        assert resp.scores.max_score() == 0.0

        resp2 = ModerationResponse(True)
        # Get scores, modify, and set back (getter returns a clone)
        scores = resp2.scores
        scores.violence = 0.9
        resp2.scores = scores
        assert resp2.flagged is True
        assert resp2.scores.max_score() == 0.9

    @pytest.mark.skipif(not has_moderation_provider(), reason="No moderation provider API key configured")
    def test_client_moderate_text(self):
        """Test client.moderate_text method."""
        client = LLMKitClient.from_env()
        req = ModerationRequest("openai/omni-moderation-latest", "This is normal content")
        response = client.moderate_text(req)  # type: ignore[attr-defined]

        assert isinstance(response, ModerationResponse)
        assert hasattr(response, "flagged")
        assert hasattr(response, "scores")


class TestClassificationAPI:
    """Test text classification API."""

    def test_create_classification_request(self):
        """Test creating a classification request."""
        labels = ["positive", "negative", "neutral"]
        req = ClassificationRequest("cohere/classify", "This movie was great!", labels)
        assert req is not None
        assert len(req.labels) == 3

    def test_classification_result(self):
        """Test ClassificationResult class."""
        result = ClassificationResult("positive", 0.98)
        assert result.label == "positive"
        assert result.confidence == 0.98

    def test_classification_response(self):
        """Test ClassificationResponse class."""
        resp = ClassificationResponse()
        assert resp.count == 0

        resp.results = [
            ClassificationResult("positive", 0.98),
            ClassificationResult("negative", 0.01),
            ClassificationResult("neutral", 0.01),
        ]
        assert resp.count == 3
        assert resp.top() is not None
        assert resp.top().label == "positive"

    @pytest.mark.skipif(not has_classification_provider(), reason="No classification provider API key configured")
    def test_client_classify_text(self):
        """Test client.classify_text method."""
        client = LLMKitClient.from_env()
        labels = ["product_feedback", "bug_report", "feature_request"]
        req = ClassificationRequest("cohere/classify", "The search feature doesn't work properly", labels)
        response = client.classify_text(req)  # type: ignore[attr-defined]

        assert isinstance(response, ClassificationResponse)
        assert response.count > 0
        assert response.top() is not None


class TestSpecializedImports:
    """Test that all specialized types can be imported."""

    def test_import_ranking_types(self):
        """Test importing ranking types."""
        from llmkit import (  # type: ignore[attr-defined]
            RankedDocument,
            RankingRequest,
            RankingResponse,
        )

        assert RankingRequest is not None
        assert RankedDocument is not None
        assert RankingResponse is not None

    def test_import_reranking_types(self):
        """Test importing reranking types."""
        from llmkit import (  # type: ignore[attr-defined]
            RerankedResult,
            RerankingRequest,
            RerankingResponse,
        )

        assert RerankingRequest is not None
        assert RerankedResult is not None
        assert RerankingResponse is not None

    def test_import_moderation_types(self):
        """Test importing moderation types."""
        from llmkit import (  # type: ignore[attr-defined]
            ModerationRequest,
            ModerationResponse,
            ModerationScores,
        )

        assert ModerationRequest is not None
        assert ModerationScores is not None
        assert ModerationResponse is not None

    def test_import_classification_types(self):
        """Test importing classification types."""
        from llmkit import (  # type: ignore[attr-defined]
            ClassificationRequest,
            ClassificationResponse,
            ClassificationResult,
        )

        assert ClassificationRequest is not None
        assert ClassificationResult is not None
        assert ClassificationResponse is not None


class TestSpecializedIntegration:
    """Integration tests for specialized APIs."""

    @pytest.mark.skipif(not has_ranking_provider(), reason="No ranking provider API key configured")
    def test_ranking_workflow(self):
        """Test complete ranking workflow."""
        client = LLMKitClient.from_env()

        documents = [
            "Python is a powerful programming language",
            "Java is used for enterprise applications",
            "Python was created by Guido van Rossum",
        ]

        req = RankingRequest("cohere/rerank-english-v3.0", "Python programming", documents).with_top_k(2)
        response = client.rank_documents(req)  # type: ignore[attr-defined]

        assert response.count > 0
        assert response.first() is not None

    @pytest.mark.skipif(not has_ranking_provider(), reason="No ranking provider API key configured")
    def test_reranking_workflow(self):
        """Test complete reranking workflow."""
        client = LLMKitClient.from_env()

        results = ["First search result", "Second result", "Third result"]
        req = RerankingRequest("voyage/rerank-2", "search query", results).with_top_n(2)
        response = client.rerank_results(req)  # type: ignore[attr-defined]

        assert response.count > 0

    @pytest.mark.skipif(not has_moderation_provider(), reason="No moderation provider API key configured")
    def test_moderation_workflow(self):
        """Test complete moderation workflow."""
        client = LLMKitClient.from_env()

        texts = [
            "This is acceptable content",
            "This is another comment",
        ]

        for text in texts:
            req = ModerationRequest("openai/omni-moderation-latest", text)
            response = client.moderate_text(req)  # type: ignore[attr-defined]
            assert isinstance(response, ModerationResponse)

    @pytest.mark.skipif(not has_classification_provider(), reason="No classification provider API key configured")
    def test_classification_workflow(self):
        """Test complete classification workflow."""
        client = LLMKitClient.from_env()

        labels = ["spam", "ham"]
        texts = [
            "Buy cheap products now!!!",
            "Meeting scheduled for tomorrow at 2 PM",
        ]

        for text in texts:
            req = ClassificationRequest("cohere/classify", text, labels)
            response = client.classify_text(req)  # type: ignore[attr-defined]
            assert response.count > 0
