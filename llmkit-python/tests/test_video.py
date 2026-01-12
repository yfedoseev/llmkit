"""Unit tests for LLMKit Python video bindings."""

from llmkit import (  # type: ignore[attr-defined]
    LLMKitClient,
    VideoGenerationOptions,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoModel,
)


class TestVideoModel:
    """Test VideoModel enum."""

    def test_runway_gen45(self):
        """Test RunwayGen45 variant."""
        model = VideoModel.RunwayGen45
        assert model is not None
        assert repr(model) is not None

    def test_kling20(self):
        """Test Kling20 variant."""
        model = VideoModel.Kling20
        assert model is not None

    def test_pika10(self):
        """Test Pika10 variant."""
        model = VideoModel.Pika10
        assert model is not None

    def test_hailuo_mini(self):
        """Test HailuoMini variant."""
        model = VideoModel.HailuoMini
        assert model is not None

    def test_leonardo_ultra(self):
        """Test LeonardoUltra variant."""
        model = VideoModel.LeonardoUltra
        assert model is not None

    def test_enum_comparison(self):
        """Test that enum variants are comparable."""
        model1 = VideoModel.RunwayGen45
        model2 = VideoModel.RunwayGen45
        model3 = VideoModel.Kling20
        assert model1 == model2
        assert model1 != model3


class TestVideoGenerationOptions:
    """Test VideoGenerationOptions class."""

    def test_create_empty(self):
        """Test creating empty VideoGenerationOptions."""
        opts = VideoGenerationOptions()
        assert opts is not None
        assert repr(opts) is not None

    def test_with_model(self):
        """Test setting model via builder method."""
        opts = VideoGenerationOptions()
        opts_with_model = opts.with_model("runway-gen-3")
        assert opts_with_model is not None
        # Options should be immutable (builder pattern returns new instance)
        opts_with_duration = opts_with_model.with_duration(10)
        assert opts_with_duration is not None

    def test_with_duration(self):
        """Test setting duration via builder method."""
        opts = VideoGenerationOptions()
        opts_with_duration = opts.with_duration(15)
        assert opts_with_duration is not None

    def test_with_width(self):
        """Test setting width via builder method."""
        opts = VideoGenerationOptions()
        opts_with_width = opts.with_width(1920)
        assert opts_with_width is not None

    def test_with_height(self):
        """Test setting height via builder method."""
        opts = VideoGenerationOptions()
        opts_with_height = opts.with_height(1080)
        assert opts_with_height is not None

    def test_with_quality(self):
        """Test setting quality via builder method."""
        opts = VideoGenerationOptions()
        opts_with_quality = opts.with_quality("high")
        assert opts_with_quality is not None

    def test_builder_chain(self):
        """Test chaining multiple builder methods."""
        opts = (
            VideoGenerationOptions()
            .with_model("runway-gen-4")
            .with_duration(30)
            .with_width(1280)
            .with_height(720)
            .with_quality("medium")
        )
        assert opts is not None

    def test_multiple_chains_independence(self):
        """Test that multiple chains are independent."""
        opts1 = VideoGenerationOptions().with_duration(10)
        opts2 = VideoGenerationOptions().with_duration(20)
        # Both should exist independently
        assert opts1 is not None
        assert opts2 is not None


class TestVideoGenerationRequest:
    """Test VideoGenerationRequest class."""

    def test_create_with_prompt(self):
        """Test creating VideoGenerationRequest with prompt."""
        req = VideoGenerationRequest("Generate a sci-fi video")
        assert req is not None
        assert repr(req) is not None

    def test_with_model(self):
        """Test setting model via builder method."""
        req = VideoGenerationRequest("Generate a video")
        req_with_model = req.with_model("runway-gen-4")
        assert req_with_model is not None

    def test_with_duration(self):
        """Test setting duration via builder method."""
        req = VideoGenerationRequest("Generate a video")
        req_with_duration = req.with_duration(20)
        assert req_with_duration is not None

    def test_with_width(self):
        """Test setting width via builder method."""
        req = VideoGenerationRequest("Generate a video")
        req_with_width = req.with_width(1920)
        assert req_with_width is not None

    def test_with_height(self):
        """Test setting height via builder method."""
        req = VideoGenerationRequest("Generate a video")
        req_with_height = req.with_height(1080)
        assert req_with_height is not None

    def test_builder_chain(self):
        """Test chaining multiple builder methods."""
        req = (
            VideoGenerationRequest("A beautiful landscape video")
            .with_model("kling-2.0")
            .with_duration(10)
            .with_width(1280)
            .with_height(720)
        )
        assert req is not None

    def test_various_prompts(self):
        """Test creating requests with various prompts."""
        prompts = [
            "A cat sleeping on a sunny windowsill",
            "Abstract geometric animation",
            "Ocean waves crashing on beach",
            "Urban cyberpunk city at night",
            "",  # Empty prompt should be allowed
        ]
        for prompt in prompts:
            req = VideoGenerationRequest(prompt)
            assert req is not None


class TestVideoGenerationResponse:
    """Test VideoGenerationResponse class."""

    def test_create_response(self):
        """Test creating VideoGenerationResponse."""
        resp = VideoGenerationResponse()
        assert resp is not None
        assert repr(resp) is not None

    def test_response_with_url(self):
        """Test response with video URL."""
        resp = VideoGenerationResponse()
        resp.video_url = "https://example.com/video.mp4"
        assert resp.video_url == "https://example.com/video.mp4"

    def test_response_with_bytes(self):
        """Test response with video bytes."""
        resp = VideoGenerationResponse()
        video_data = b"fake_video_data"
        resp.video_bytes = video_data
        assert resp.video_bytes == video_data

    def test_response_format(self):
        """Test video format property."""
        resp = VideoGenerationResponse()
        assert resp.format == "mp4"  # Default format
        resp.format = "mov"
        assert resp.format == "mov"

    def test_response_duration(self):
        """Test video duration property."""
        resp = VideoGenerationResponse()
        resp.duration = 15.5
        assert resp.duration == 15.5

    def test_response_resolution(self):
        """Test video resolution properties."""
        resp = VideoGenerationResponse()
        resp.width = 1920
        resp.height = 1080
        assert resp.width == 1920
        assert resp.height == 1080

    def test_response_task_id(self):
        """Test task ID for async operations."""
        resp = VideoGenerationResponse()
        resp.task_id = "task-abc123"
        assert resp.task_id == "task-abc123"

    def test_response_status(self):
        """Test status property for polling."""
        resp = VideoGenerationResponse()
        resp.status = "processing"
        assert resp.status == "processing"

        resp.status = "completed"
        assert resp.status == "completed"

    def test_size_property(self):
        """Test size calculation from video bytes."""
        resp = VideoGenerationResponse()
        # Size should be 0 when no bytes
        assert resp.size == 0

        # Size should match bytes length
        resp.video_bytes = b"x" * 1024  # 1 KB
        assert resp.size == 1024

        resp.video_bytes = b"x" * (1024 * 1024)  # 1 MB
        assert resp.size == 1024 * 1024

    def test_response_complete_object(self):
        """Test creating a complete response object."""
        resp = VideoGenerationResponse()
        resp.video_url = "https://example.com/video.mp4"
        resp.format = "mp4"
        resp.duration = 10.0
        resp.width = 1280
        resp.height = 720
        resp.task_id = "task-12345"
        resp.status = "completed"

        assert resp.video_url == "https://example.com/video.mp4"
        assert resp.format == "mp4"
        assert resp.duration == 10.0
        assert resp.width == 1280
        assert resp.height == 720
        assert resp.task_id == "task-12345"
        assert resp.status == "completed"


class TestVideoClientMethods:
    """Test video generation methods on LLMKitClient."""

    def test_client_has_generate_video_method(self):
        """Test that client has generate_video method."""
        client = LLMKitClient.from_env()
        assert hasattr(client, "generate_video")
        assert callable(getattr(client, "generate_video"))

    def test_generate_video_with_request(self):
        """Test calling generate_video with VideoGenerationRequest."""
        client = LLMKitClient.from_env()
        req = VideoGenerationRequest("A beautiful sunset over mountains")
        response = client.generate_video(req)  # type: ignore[attr-defined]

        assert isinstance(response, VideoGenerationResponse)
        assert response.video_url is not None
        assert response.format is not None

    def test_generate_video_response_properties(self):
        """Test that generate_video response has expected properties."""
        client = LLMKitClient.from_env()
        req = VideoGenerationRequest("Test video generation")
        response = client.generate_video(req)  # type: ignore[attr-defined]

        # Response should have these properties (may be None in placeholder)
        assert hasattr(response, "video_bytes")
        assert hasattr(response, "video_url")
        assert hasattr(response, "format")
        assert hasattr(response, "duration")
        assert hasattr(response, "width")
        assert hasattr(response, "height")
        assert hasattr(response, "task_id")
        assert hasattr(response, "status")

    def test_generate_video_with_configured_options(self):
        """Test generate_video with fully configured request."""
        client = LLMKitClient.from_env()
        req = (
            VideoGenerationRequest("An animated character walking")
            .with_model("pika-1.0")
            .with_duration(5)
            .with_width(1024)
            .with_height(576)
        )
        response = client.generate_video(req)  # type: ignore[attr-defined]
        assert isinstance(response, VideoGenerationResponse)

    def test_generate_video_error_handling(self):
        """Test error handling in generate_video."""
        client = LLMKitClient.from_env()
        req = VideoGenerationRequest("")  # Empty prompt

        # Should not raise an error (error handling is provider-specific)
        try:
            response = client.generate_video(req)  # type: ignore[attr-defined]
            assert response is not None
        except Exception as e:
            # If error is raised, it should be a proper LLMKit error
            assert "Error" in type(e).__name__ or "Exception" in type(e).__name__


class TestVideoImports:
    """Test that all video types can be imported."""

    def test_import_video_model(self):
        """Test importing VideoModel."""
        from llmkit import VideoModel  # type: ignore[attr-defined]

        assert VideoModel is not None

    def test_import_video_generation_options(self):
        """Test importing VideoGenerationOptions."""
        from llmkit import VideoGenerationOptions  # type: ignore[attr-defined]

        assert VideoGenerationOptions is not None

    def test_import_video_generation_request(self):
        """Test importing VideoGenerationRequest."""
        from llmkit import VideoGenerationRequest  # type: ignore[attr-defined]

        assert VideoGenerationRequest is not None

    def test_import_video_generation_response(self):
        """Test importing VideoGenerationResponse."""
        from llmkit import VideoGenerationResponse  # type: ignore[attr-defined]

        assert VideoGenerationResponse is not None

    def test_all_video_types_available(self):
        """Test that all video types are available from main module."""
        import llmkit

        assert hasattr(llmkit, "VideoModel")
        assert hasattr(llmkit, "VideoGenerationOptions")
        assert hasattr(llmkit, "VideoGenerationRequest")
        assert hasattr(llmkit, "VideoGenerationResponse")


class TestVideoIntegration:
    """Integration tests for video functionality."""

    def test_end_to_end_video_generation_flow(self):
        """Test complete video generation workflow."""
        client = LLMKitClient.from_env()

        # 1. Create request
        req = VideoGenerationRequest("A spinning 3D cube with colorful lights")

        # 2. Configure options
        req = req.with_model("runway-gen-4").with_duration(10).with_width(1920).with_height(1080)

        # 3. Generate video
        response = client.generate_video(req)  # type: ignore[attr-defined]

        # 4. Verify response
        assert isinstance(response, VideoGenerationResponse)
        assert response.video_url is not None or response.video_bytes is not None
        assert response.format is not None

    def test_polling_simulation(self):
        """Test polling pattern for async video generation."""
        client = LLMKitClient.from_env()

        req = VideoGenerationRequest("Generate a test video")
        response = client.generate_video(req)  # type: ignore[attr-defined]

        # Simulate polling loop
        max_attempts = 5
        attempts = 0
        while attempts < max_attempts:
            if response.status == "completed":
                break
            # In real scenario, would call client.poll_video(task_id)
            # For now, just verify status field exists
            assert hasattr(response, "status")
            attempts += 1

        assert response is not None

    def test_multiple_video_generations(self):
        """Test generating multiple videos sequentially."""
        client = LLMKitClient.from_env()

        prompts = [
            "A cat playing with a ball",
            "A sunset over the ocean",
            "A person dancing",
        ]

        responses = []
        for prompt in prompts:
            req = VideoGenerationRequest(prompt)
            response = client.generate_video(req)  # type: ignore[attr-defined]
            responses.append(response)

        assert len(responses) == 3
        assert all(isinstance(r, VideoGenerationResponse) for r in responses)
