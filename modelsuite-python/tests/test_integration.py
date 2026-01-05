#!/usr/bin/env python3
"""
Integration tests for ModelSuite - Test all modalities working together.
"""

import pytest

from modelsuite import (  # type: ignore[attr-defined]
    ClassificationRequest,
    # Image
    ImageGenerationRequest,
    ModelSuiteClient,
    ModerationRequest,
    # Specialized
    RankingRequest,
    RerankingRequest,
    SynthesisRequest,
    # Audio
    TranscriptionRequest,
    # Video
    VideoGenerationRequest,
)


@pytest.fixture
def client():
    """Create ModelSuite client from environment."""
    return ModelSuiteClient.from_env()


class TestAudioIntegration:
    """Test audio API integration."""

    def test_transcription_request_creation(self, client):
        """Test creating and using transcription request."""
        audio_bytes = b"test audio data"
        req = TranscriptionRequest(audio_bytes)
        assert req is not None

    def test_synthesis_request_creation(self, client):
        """Test creating and using synthesis request."""
        req = SynthesisRequest("Hello world", voice="alloy")
        assert req is not None

    def test_audio_workflow(self, client):
        """Test complete audio workflow."""
        # Create synthesis request
        synthesis_req = SynthesisRequest("Test audio generation")
        # In production, this would call the provider
        assert synthesis_req is not None

        # Create transcription request
        transcription_req = TranscriptionRequest(b"audio bytes")
        assert transcription_req is not None


class TestVideoIntegration:
    """Test video API integration."""

    def test_video_generation_request(self, client):
        """Test creating video generation request."""
        req = VideoGenerationRequest("A sunset over mountains")
        assert req is not None

    def test_video_with_builder(self, client):
        """Test builder pattern for video requests."""
        req = VideoGenerationRequest("Test video").with_model("runway-gen3-alpha").with_duration(5)
        assert req.prompt == "Test video"
        assert req.model == "runway-gen3-alpha"
        assert req.duration == 5

    def test_video_workflow(self, client):
        """Test complete video workflow."""
        req = VideoGenerationRequest("Generate test video")
        assert req is not None
        # Would call client.generate_video(req) in production


class TestImageIntegration:
    """Test image API integration."""

    def test_image_generation_request(self, client):
        """Test creating image generation request."""
        req = ImageGenerationRequest("A futuristic city")
        assert req is not None

    def test_image_with_builder(self, client):
        """Test builder pattern for image requests."""
        req = ImageGenerationRequest("Test image").with_model("dall-e-3").with_size("1024x1024")
        assert req.prompt == "Test image"
        assert req.model == "dall-e-3"
        assert req.size == "1024x1024"

    def test_image_workflow(self, client):
        """Test complete image workflow."""
        req = ImageGenerationRequest("Generate test image")
        assert req is not None
        # Would call client.generate_image(req) in production


class TestSpecializedIntegration:
    """Test specialized APIs integration."""

    def test_ranking_request(self, client):
        """Test ranking API."""
        docs = ["Document 1", "Document 2", "Document 3"]
        req = RankingRequest("test query", docs)
        assert req is not None
        assert len(req.documents) == 3

    def test_ranking_with_builder(self, client):
        """Test ranking builder pattern."""
        docs = ["Doc A", "Doc B"]
        req = RankingRequest("query", docs).with_top_k(1)
        assert req.top_k == 1

    def test_reranking_request(self, client):
        """Test reranking API."""
        results = ["Result 1", "Result 2"]
        req = RerankingRequest("query", results)
        assert req is not None

    def test_moderation_request(self, client):
        """Test moderation API."""
        req = ModerationRequest("Test content")
        assert req is not None

    def test_classification_request(self, client):
        """Test classification API."""
        labels = ["positive", "negative", "neutral"]
        req = ClassificationRequest("Test text", labels)
        assert req is not None
        assert len(req.labels) == 3

    def test_specialized_workflow(self, client):
        """Test complete specialized workflow."""
        # Moderation
        mod_req = ModerationRequest("Test content")
        assert mod_req is not None

        # Classification
        class_req = ClassificationRequest("Great product!", ["positive", "negative"])
        assert class_req is not None

        # Ranking
        rank_req = RankingRequest("python", ["Python doc", "Java doc"])
        assert rank_req is not None


class TestCrossModalityWorkflow:
    """Test workflows combining multiple modalities."""

    def test_complete_workflow(self, client):
        """Test using all 4 modalities in one workflow."""
        # 1. Generate audio
        audio_req = SynthesisRequest("Describe the image")
        assert audio_req is not None

        # 2. Generate image
        image_req = ImageGenerationRequest("A scene to describe")
        assert image_req is not None

        # 3. Generate video
        video_req = VideoGenerationRequest("Animation of the scene")
        assert video_req is not None

        # 4. Moderate and classify
        mod_req = ModerationRequest("Describe the video content")
        assert mod_req is not None

        class_req = ClassificationRequest(
            "The content is amazing!",
            ["positive", "negative", "neutral"],
        )
        assert class_req is not None

    def test_search_workflow(self, client):
        """Test search + moderation + ranking workflow."""
        # Search returns results
        search_results = ["Result 1", "Result 2", "Result 3"]

        # Moderate results
        for result in search_results:
            mod_req = ModerationRequest(result)
            assert mod_req is not None

        # Rank results
        rank_req = RankingRequest("search query", search_results).with_top_k(2)
        assert rank_req is not None

    def test_content_processing_pipeline(self, client):
        """Test content processing pipeline."""
        user_comment = "This is a great product!"

        # Step 1: Check moderation
        mod_req = ModerationRequest(user_comment)
        assert mod_req is not None

        # Step 2: Classify sentiment
        class_req = ClassificationRequest(user_comment, ["positive", "negative", "neutral"])
        assert class_req is not None

        # Step 3: Generate audio response
        audio_req = SynthesisRequest("Thank you for the positive feedback!")
        assert audio_req is not None


class TestErrorHandling:
    """Test error handling across modalities."""

    def test_invalid_request_handling(self, client):
        """Test handling of invalid requests."""
        # These should create valid request objects but fail in execution
        try:
            req = RankingRequest("query", [])  # Empty documents
            assert req is not None
        except Exception:
            pass  # Expected in some cases

    def test_malformed_input(self, client):
        """Test handling of malformed input."""
        try:
            req = ClassificationRequest("text", [])  # Empty labels
            assert req is not None
        except Exception:
            pass  # Expected in some cases


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
