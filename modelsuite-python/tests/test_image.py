"""Unit tests for ModelSuite Python image bindings."""

from modelsuite import (  # type: ignore[attr-defined]
    GeneratedImage,
    ImageFormat,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageQuality,
    ImageSize,
    ImageStyle,
    ModelSuiteClient,
)


class TestImageSize:
    """Test ImageSize enum."""

    def test_square256(self):
        """Test Square256 variant."""
        size = ImageSize.Square256
        assert size is not None
        assert repr(size) is not None

    def test_square512(self):
        """Test Square512 variant."""
        size = ImageSize.Square512
        assert size is not None

    def test_square1024(self):
        """Test Square1024 variant."""
        size = ImageSize.Square1024
        assert size is not None

    def test_portrait1024x1792(self):
        """Test Portrait1024x1792 variant."""
        size = ImageSize.Portrait1024x1792
        assert size is not None

    def test_landscape1792x1024(self):
        """Test Landscape1792x1024 variant."""
        size = ImageSize.Landscape1792x1024
        assert size is not None

    def test_dimensions(self):
        """Test getting dimensions from size."""
        assert ImageSize.Square256.dimensions() == (256, 256)
        assert ImageSize.Square512.dimensions() == (512, 512)
        assert ImageSize.Square1024.dimensions() == (1024, 1024)
        assert ImageSize.Portrait1024x1792.dimensions() == (1024, 1792)
        assert ImageSize.Landscape1792x1024.dimensions() == (1792, 1024)


class TestImageQuality:
    """Test ImageQuality enum."""

    def test_standard(self):
        """Test Standard variant."""
        quality = ImageQuality.Standard
        assert quality is not None
        assert repr(quality) is not None

    def test_hd(self):
        """Test HD variant."""
        quality = ImageQuality.Hd
        assert quality is not None


class TestImageStyle:
    """Test ImageStyle enum."""

    def test_natural(self):
        """Test Natural variant."""
        style = ImageStyle.Natural
        assert style is not None
        assert repr(style) is not None

    def test_vivid(self):
        """Test Vivid variant."""
        style = ImageStyle.Vivid
        assert style is not None


class TestImageFormat:
    """Test ImageFormat enum."""

    def test_url(self):
        """Test Url variant."""
        fmt = ImageFormat.Url
        assert fmt is not None
        assert repr(fmt) is not None

    def test_b64json(self):
        """Test B64Json variant."""
        fmt = ImageFormat.B64Json
        assert fmt is not None


class TestImageGenerationRequest:
    """Test ImageGenerationRequest class."""

    def test_create_with_model_and_prompt(self):
        """Test creating ImageGenerationRequest with model and prompt."""
        req = ImageGenerationRequest("dall-e-3", "A serene landscape")
        assert req is not None
        assert repr(req) is not None

    def test_with_n(self):
        """Test setting number of images."""
        req = ImageGenerationRequest("dall-e-3", "A cat")
        req_with_n = req.with_n(3)
        assert req_with_n is not None

    def test_with_size(self):
        """Test setting image size."""
        req = ImageGenerationRequest("dall-e-3", "A dog")
        req_with_size = req.with_size(ImageSize.Square1024)
        assert req_with_size is not None

    def test_with_quality(self):
        """Test setting image quality."""
        req = ImageGenerationRequest("dall-e-3", "A bird")
        req_with_quality = req.with_quality(ImageQuality.Hd)
        assert req_with_quality is not None

    def test_with_style(self):
        """Test setting image style."""
        req = ImageGenerationRequest("dall-e-3", "A car")
        req_with_style = req.with_style(ImageStyle.Vivid)
        assert req_with_style is not None

    def test_with_format(self):
        """Test setting response format."""
        req = ImageGenerationRequest("dall-e-3", "A house")
        req_with_format = req.with_format(ImageFormat.B64Json)
        assert req_with_format is not None

    def test_with_negative_prompt(self):
        """Test setting negative prompt."""
        req = ImageGenerationRequest("stability-ai", "A sunset")
        req_with_neg = req.with_negative_prompt("blurry, low quality")
        assert req_with_neg is not None

    def test_with_seed(self):
        """Test setting seed for reproducibility."""
        req = ImageGenerationRequest("fal-ai", "A mountain")
        req_with_seed = req.with_seed(12345)
        assert req_with_seed is not None

    def test_builder_chain(self):
        """Test chaining multiple builder methods."""
        req = (
            ImageGenerationRequest("dall-e-3", "A magical forest")
            .with_n(2)
            .with_size(ImageSize.Landscape1792x1024)
            .with_quality(ImageQuality.Hd)
            .with_style(ImageStyle.Vivid)
            .with_format(ImageFormat.Url)
        )
        assert req is not None

    def test_various_prompts(self):
        """Test creating requests with various prompts."""
        prompts = [
            "A serene mountain landscape",
            "Portrait of a Renaissance noble",
            "Abstract geometric pattern",
            "Futuristic cyberpunk city",
            "",  # Empty prompt should be allowed
        ]
        for prompt in prompts:
            req = ImageGenerationRequest("dall-e-3", prompt)
            assert req is not None

    def test_various_models(self):
        """Test creating requests with various models."""
        models = [
            "dall-e-3",
            "dall-e-2",
            "fal-ai/flux/dev",
            "fal-ai/flux/schnell",
            "stability-ai/stable-diffusion-xl",
        ]
        for model in models:
            req = ImageGenerationRequest(model, "Test prompt")
            assert req is not None


class TestGeneratedImage:
    """Test GeneratedImage class."""

    def test_create_empty(self):
        """Test creating empty GeneratedImage."""
        img = GeneratedImage.from_url("https://example.com/image.png")
        assert img is not None
        assert repr(img) is not None

    def test_from_url(self):
        """Test creating image from URL."""
        url = "https://example.com/image.png"
        img = GeneratedImage.from_url(url)
        assert img.url == url

    def test_from_b64(self):
        """Test creating image from base64 data."""
        b64_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
        img = GeneratedImage.from_b64(b64_data)
        assert img.b64_json == b64_data

    def test_with_revised_prompt(self):
        """Test setting revised prompt."""
        img = GeneratedImage.from_url("https://example.com/image.png")
        img_with_prompt = img.with_revised_prompt("A revised description")
        assert img_with_prompt is not None
        assert img_with_prompt.revised_prompt == "A revised description"

    def test_size_property(self):
        """Test size property."""
        img = GeneratedImage.from_url("https://example.com/image.png")
        # URL images have no b64_json, so size should be 0
        assert img.size == 0

        # b64 images should have size
        img_b64 = GeneratedImage.from_b64("x" * 1000)
        assert img_b64.size == 1000


class TestImageGenerationResponse:
    """Test ImageGenerationResponse class."""

    def test_create_response(self):
        """Test creating ImageGenerationResponse."""
        resp = ImageGenerationResponse()
        assert resp is not None
        assert repr(resp) is not None

    def test_count_property(self):
        """Test count of images."""
        resp = ImageGenerationResponse()
        assert resp.count == 0

        # Add images
        resp.images = [
            GeneratedImage.from_url("https://example.com/image1.png"),
            GeneratedImage.from_url("https://example.com/image2.png"),
        ]
        assert resp.count == 2

    def test_first_image(self):
        """Test getting first image."""
        resp = ImageGenerationResponse()
        assert resp.first() is None

        resp.images = [GeneratedImage.from_url("https://example.com/image1.png")]
        first = resp.first()
        assert first is not None

    def test_total_size_property(self):
        """Test total size of all image data."""
        resp = ImageGenerationResponse()
        assert resp.total_size == 0

        # Add b64 images
        resp.images = [
            GeneratedImage.from_b64("x" * 1000),
            GeneratedImage.from_b64("y" * 2000),
        ]
        assert resp.total_size == 3000

    def test_created_timestamp(self):
        """Test created timestamp."""
        resp = ImageGenerationResponse()
        resp.created = 1609459200  # 2021-01-01 00:00:00 UTC
        assert resp.created == 1609459200

    def test_complete_response(self):
        """Test creating a complete response with images."""
        resp = ImageGenerationResponse()
        resp.created = 1609459200
        resp.images = [
            GeneratedImage.from_url("https://example.com/image1.png").with_revised_prompt(
                "A serene landscape with mountains"
            ),
            GeneratedImage.from_url("https://example.com/image2.png"),
        ]

        assert resp.count == 2
        assert resp.first() is not None
        assert resp.first().revised_prompt == "A serene landscape with mountains"


class TestImageClientMethods:
    """Test image generation methods on ModelSuiteClient."""

    def test_client_has_generate_image_method(self):
        """Test that client has generate_image method."""
        client = ModelSuiteClient.from_env()
        assert hasattr(client, "generate_image")
        assert callable(getattr(client, "generate_image"))

    def test_generate_image_with_request(self):
        """Test calling generate_image with ImageGenerationRequest."""
        client = ModelSuiteClient.from_env()
        req = ImageGenerationRequest("dall-e-3", "A beautiful sunset over mountains")
        response = client.generate_image(req)  # type: ignore[attr-defined]

        assert isinstance(response, ImageGenerationResponse)
        assert response.count > 0

    def test_generate_image_response_properties(self):
        """Test that generate_image response has expected properties."""
        client = ModelSuiteClient.from_env()
        req = ImageGenerationRequest("dall-e-3", "Test image generation")
        response = client.generate_image(req)  # type: ignore[attr-defined]

        # Response should have these properties
        assert hasattr(response, "created")
        assert hasattr(response, "images")
        assert hasattr(response, "count")
        assert hasattr(response, "total_size")

    def test_generate_image_with_configured_options(self):
        """Test generate_image with fully configured request."""
        client = ModelSuiteClient.from_env()
        req = (
            ImageGenerationRequest("dall-e-3", "An animated scene")
            .with_n(2)
            .with_size(ImageSize.Square1024)
            .with_quality(ImageQuality.Hd)
            .with_style(ImageStyle.Vivid)
        )
        response = client.generate_image(req)  # type: ignore[attr-defined]
        assert isinstance(response, ImageGenerationResponse)

    def test_generate_image_with_negative_prompt(self):
        """Test generate_image with negative prompt."""
        client = ModelSuiteClient.from_env()
        req = (
            ImageGenerationRequest("stability-ai/stable-diffusion-xl", "A landscape")
            .with_negative_prompt("blurry, low quality, watermark")
            .with_n(1)
        )
        response = client.generate_image(req)  # type: ignore[attr-defined]
        assert response is not None

    def test_generate_image_error_handling(self):
        """Test error handling in generate_image."""
        client = ModelSuiteClient.from_env()
        req = ImageGenerationRequest("dall-e-3", "")  # Empty prompt

        # Should not raise an error (error handling is provider-specific)
        try:
            response = client.generate_image(req)  # type: ignore[attr-defined]
            assert response is not None
        except Exception as e:
            # If error is raised, it should be a proper ModelSuite error
            assert "Error" in type(e).__name__ or "Exception" in type(e).__name__


class TestImageImports:
    """Test that all image types can be imported."""

    def test_import_image_size(self):
        """Test importing ImageSize."""
        from modelsuite import ImageSize  # type: ignore[attr-defined]

        assert ImageSize is not None

    def test_import_image_quality(self):
        """Test importing ImageQuality."""
        from modelsuite import ImageQuality  # type: ignore[attr-defined]

        assert ImageQuality is not None

    def test_import_image_style(self):
        """Test importing ImageStyle."""
        from modelsuite import ImageStyle  # type: ignore[attr-defined]

        assert ImageStyle is not None

    def test_import_image_format(self):
        """Test importing ImageFormat."""
        from modelsuite import ImageFormat  # type: ignore[attr-defined]

        assert ImageFormat is not None

    def test_import_image_generation_request(self):
        """Test importing ImageGenerationRequest."""
        from modelsuite import ImageGenerationRequest  # type: ignore[attr-defined]

        assert ImageGenerationRequest is not None

    def test_import_generated_image(self):
        """Test importing GeneratedImage."""
        from modelsuite import GeneratedImage  # type: ignore[attr-defined]

        assert GeneratedImage is not None

    def test_import_image_generation_response(self):
        """Test importing ImageGenerationResponse."""
        from modelsuite import ImageGenerationResponse  # type: ignore[attr-defined]

        assert ImageGenerationResponse is not None

    def test_all_image_types_available(self):
        """Test that all image types are available from main module."""
        import modelsuite

        assert hasattr(modelsuite, "ImageSize")
        assert hasattr(modelsuite, "ImageQuality")
        assert hasattr(modelsuite, "ImageStyle")
        assert hasattr(modelsuite, "ImageFormat")
        assert hasattr(modelsuite, "ImageGenerationRequest")
        assert hasattr(modelsuite, "GeneratedImage")
        assert hasattr(modelsuite, "ImageGenerationResponse")


class TestImageIntegration:
    """Integration tests for image functionality."""

    def test_end_to_end_image_generation_flow(self):
        """Test complete image generation workflow."""
        client = ModelSuiteClient.from_env()

        # 1. Create request
        req = ImageGenerationRequest("dall-e-3", "A serene Japanese garden with temple")

        # 2. Configure options
        req = (
            req.with_n(1)
            .with_size(ImageSize.Landscape1792x1024)
            .with_quality(ImageQuality.Hd)
            .with_style(ImageStyle.Natural)
        )

        # 3. Generate image
        response = client.generate_image(req)  # type: ignore[attr-defined]

        # 4. Verify response
        assert isinstance(response, ImageGenerationResponse)
        assert response.count > 0
        assert response.first() is not None

    def test_multiple_image_generations(self):
        """Test generating multiple images with different prompts."""
        client = ModelSuiteClient.from_env()

        prompts = [
            ("dall-e-3", "A cat playing with yarn"),
            ("fal-ai/flux/dev", "A sunset over ocean"),
            ("stability-ai/stable-diffusion-xl", "A futuristic city"),
        ]

        responses = []
        for model, prompt in prompts:
            req = ImageGenerationRequest(model, prompt)
            response = client.generate_image(req)  # type: ignore[attr-defined]
            responses.append(response)

        assert len(responses) == 3
        assert all(isinstance(r, ImageGenerationResponse) for r in responses)

    def test_image_size_variations(self):
        """Test generating images with different sizes."""
        client = ModelSuiteClient.from_env()

        sizes = [
            ImageSize.Square256,
            ImageSize.Square512,
            ImageSize.Square1024,
            ImageSize.Portrait1024x1792,
            ImageSize.Landscape1792x1024,
        ]

        for size in sizes:
            req = ImageGenerationRequest("dall-e-3", "Test image").with_size(size)
            response = client.generate_image(req)  # type: ignore[attr-defined]
            assert isinstance(response, ImageGenerationResponse)
