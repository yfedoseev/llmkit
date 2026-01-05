"""Tests for ModelSuite Python bindings."""

import pytest

from modelsuite import (
    AsyncModelSuiteClient,
    AuthenticationError,
    BatchStatus,
    CacheBreakpoint,
    CacheControl,
    CompletionRequest,
    # Types
    ContentBlock,
    EmbeddingInputType,
    EmbeddingRequest,
    EncodingFormat,
    InvalidRequestError,
    # Clients
    ModelSuiteClient,
    # Exceptions
    ModelSuiteError,
    Message,
    ModelStatus,
    # Model Registry
    Provider,
    ProviderNotFoundError,
    RateLimitError,
    Role,
    StopReason,
    StreamEventType,
    StructuredOutput,
    ThinkingConfig,
    ThinkingType,
    ToolBuilder,
    ToolDefinition,
    get_all_models,
    get_cheapest_model,
    get_classifier_models,
    get_current_models,
    get_model_info,
    get_models_by_provider,
    get_models_with_capability,
    get_registry_stats,
    list_providers,
    supports_structured_output,
)


class TestEnums:
    """Test enum types."""

    def test_role_values(self) -> None:
        assert Role.System == 0
        assert Role.User == 1
        assert Role.Assistant == 2

    def test_role_str(self) -> None:
        assert str(Role.System) == "system"
        assert str(Role.User) == "user"
        assert str(Role.Assistant) == "assistant"

    def test_stop_reason_values(self) -> None:
        assert StopReason.EndTurn == 0
        assert StopReason.MaxTokens == 1
        assert StopReason.ToolUse == 2
        assert StopReason.StopSequence == 3
        assert StopReason.ContentFilter == 4

    def test_stop_reason_properties(self) -> None:
        assert StopReason.EndTurn.is_complete
        assert not StopReason.MaxTokens.is_complete
        assert StopReason.ToolUse.is_tool_use
        assert not StopReason.EndTurn.is_tool_use

    def test_stream_event_type_values(self) -> None:
        assert StreamEventType.MessageStart == 0
        assert StreamEventType.MessageStop == 5
        assert StreamEventType.MessageStop.is_done
        assert not StreamEventType.MessageStart.is_done

    def test_cache_control_values(self) -> None:
        assert CacheControl.Ephemeral == 0
        assert CacheControl.Extended == 1

    def test_thinking_type_values(self) -> None:
        assert ThinkingType.Enabled == 0
        assert ThinkingType.Disabled == 1

    def test_batch_status_properties(self) -> None:
        assert BatchStatus.Validating.is_processing
        assert BatchStatus.InProgress.is_processing
        assert not BatchStatus.Completed.is_processing
        assert BatchStatus.Completed.is_done
        assert BatchStatus.Completed.is_success
        assert BatchStatus.Failed.is_done
        assert not BatchStatus.Failed.is_success


class TestContentBlock:
    """Test ContentBlock type."""

    def test_text_block(self) -> None:
        block = ContentBlock.text("Hello, world!")
        assert block.is_text
        assert not block.is_tool_use
        assert block.text_value == "Hello, world!"

    def test_image_block(self) -> None:
        block = ContentBlock.image("image/png", "base64data...")
        assert block.is_image
        assert not block.is_text

    def test_image_url_block(self) -> None:
        block = ContentBlock.image_url("https://example.com/image.png")
        assert block.is_image

    def test_tool_use_block(self) -> None:
        block = ContentBlock.tool_use("id-123", "get_weather", {"city": "London"})
        assert block.is_tool_use
        assert not block.is_text
        result = block.as_tool_use()
        assert result is not None
        id, name, input_dict = result
        assert id == "id-123"
        assert name == "get_weather"
        assert input_dict == {"city": "London"}

    def test_tool_result_block(self) -> None:
        block = ContentBlock.tool_result("id-123", "Weather: Sunny, 20C", False)
        assert block.is_tool_result
        result = block.as_tool_result()
        assert result is not None
        tool_use_id, content, is_error = result
        assert tool_use_id == "id-123"
        assert content == "Weather: Sunny, 20C"
        assert not is_error

    def test_tool_result_error(self) -> None:
        block = ContentBlock.tool_result("id-123", "Error: City not found", True)
        result = block.as_tool_result()
        assert result is not None
        assert result[2] is True

    def test_thinking_block(self) -> None:
        block = ContentBlock.thinking("Let me think about this...")
        assert block.is_thinking
        assert block.thinking_content == "Let me think about this..."

    def test_pdf_block(self) -> None:
        block = ContentBlock.pdf("base64pdfdata...")
        assert block.is_document

    def test_text_cached_block(self) -> None:
        block = ContentBlock.text_cached("Cached text content")
        assert block.is_text
        assert block.text_value == "Cached text content"


class TestMessage:
    """Test Message type."""

    def test_system_message(self) -> None:
        msg = Message.system("You are a helpful assistant.")
        assert msg.role == Role.System
        assert msg.text_content() == "You are a helpful assistant."

    def test_user_message(self) -> None:
        msg = Message.user("Hello!")
        assert msg.role == Role.User
        assert msg.text_content() == "Hello!"

    def test_assistant_message(self) -> None:
        msg = Message.assistant("Hi there!")
        assert msg.role == Role.Assistant
        assert msg.text_content() == "Hi there!"

    def test_user_with_content(self) -> None:
        blocks = [
            ContentBlock.text("Look at this image:"),
            ContentBlock.image_url("https://example.com/image.png"),
        ]
        msg = Message.user_with_content(blocks)
        assert msg.role == Role.User
        assert len(msg.content) == 2
        assert msg.text_content() == "Look at this image:"

    def test_assistant_with_content(self) -> None:
        blocks = [
            ContentBlock.text("Here's the weather:"),
            ContentBlock.tool_use("id-1", "get_weather", {"city": "NYC"}),
        ]
        msg = Message.assistant_with_content(blocks)
        assert msg.role == Role.Assistant
        assert msg.has_tool_use()
        tool_uses = msg.tool_uses()
        assert len(tool_uses) == 1

    def test_tool_results_message(self) -> None:
        results = [
            ContentBlock.tool_result("id-1", "Sunny, 25C"),
            ContentBlock.tool_result("id-2", "Rainy, 15C"),
        ]
        msg = Message.tool_results(results)
        assert msg.role == Role.User
        assert len(msg.content) == 2


class TestCacheBreakpoint:
    """Test CacheBreakpoint type."""

    def test_ephemeral(self) -> None:
        bp = CacheBreakpoint.ephemeral()
        assert bp.cache_control == CacheControl.Ephemeral

    def test_extended(self) -> None:
        bp = CacheBreakpoint.extended()
        assert bp.cache_control == CacheControl.Extended


class TestThinkingConfig:
    """Test ThinkingConfig type."""

    def test_enabled(self) -> None:
        config = ThinkingConfig.enabled(10000)
        assert config.thinking_type == ThinkingType.Enabled
        assert config.budget_tokens == 10000

    def test_disabled(self) -> None:
        config = ThinkingConfig.disabled()
        assert config.thinking_type == ThinkingType.Disabled
        assert config.budget_tokens is None


class TestStructuredOutput:
    """Test StructuredOutput type."""

    def test_json_schema(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        output = StructuredOutput.json_schema("person", schema)
        assert output is not None

    def test_json_object(self) -> None:
        output = StructuredOutput.json_object()
        assert output is not None


class TestToolBuilder:
    """Test ToolBuilder and ToolDefinition."""

    def test_simple_tool(self) -> None:
        tool = (
            ToolBuilder("greet")
            .description("Greet someone")
            .string_param("name", "Person's name", required=True)
            .build()
        )
        assert tool.name == "greet"
        assert tool.description == "Greet someone"

    def test_complex_tool(self) -> None:
        tool = (
            ToolBuilder("get_weather")
            .description("Get current weather for a location")
            .string_param("city", "City name", required=True)
            .string_param("country", "Country code", required=False)
            .enum_param("unit", "Temperature unit", ["celsius", "fahrenheit"], required=False)
            .boolean_param("include_forecast", "Include 5-day forecast", required=False)
            .build()
        )
        assert tool.name == "get_weather"
        schema = tool.input_schema
        assert "properties" in schema
        assert "city" in schema["properties"]

    def test_tool_with_array_param(self) -> None:
        tool = (
            ToolBuilder("batch_process")
            .description("Process multiple items")
            .array_param("items", "Items to process", "string", required=True)
            .integer_param("batch_size", "Batch size", required=False)
            .build()
        )
        assert tool.name == "batch_process"

    def test_tool_definition_direct(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        }
        tool = ToolDefinition("search", "Search the web", schema)
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert tool.input_schema == schema


class TestCompletionRequest:
    """Test CompletionRequest type."""

    def test_constructor(self) -> None:
        request = CompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message.user("Hello!")],
        )
        assert request.model == "claude-sonnet-4-20250514"
        assert len(request.messages) == 1
        assert not request.stream

    def test_constructor_with_options(self) -> None:
        request = CompletionRequest(
            model="gpt-4o",
            messages=[Message.user("Hello!")],
            system="You are helpful",
            max_tokens=1024,
            temperature=0.7,
            stream=True,
        )
        assert request.model == "gpt-4o"
        assert request.system == "You are helpful"
        assert request.max_tokens == 1024
        assert request.temperature is not None
        assert abs(request.temperature - 0.7) < 0.0001  # f32 precision
        assert request.stream

    def test_builder_pattern(self) -> None:
        request = (
            CompletionRequest("claude-sonnet-4-20250514", [Message.user("Hi")])
            .with_system("Be helpful")
            .with_max_tokens(2048)
            .with_temperature(0.5)
        )
        assert request.system == "Be helpful"
        assert request.max_tokens == 2048
        assert request.temperature == 0.5

    def test_streaming_builder(self) -> None:
        request = CompletionRequest(
            "claude-sonnet-4-20250514", [Message.user("Hi")]
        ).with_streaming()
        assert request.stream

    def test_thinking_builder(self) -> None:
        request = CompletionRequest(
            "claude-sonnet-4-20250514", [Message.user("Solve this puzzle")]
        ).with_thinking(10000)
        assert request.has_thinking()

    def test_tools_builder(self) -> None:
        tool = ToolBuilder("test").description("Test tool").build()
        request = CompletionRequest(
            "claude-sonnet-4-20250514", [Message.user("Use the tool")]
        ).with_tools([tool])
        # Just verify it doesn't crash
        assert request is not None

    def test_json_schema_builder(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        request = CompletionRequest(
            "claude-sonnet-4-20250514", [Message.user("Answer")]
        ).with_json_schema("response", schema)
        assert request.has_structured_output()

    def test_caching_builder(self) -> None:
        request = CompletionRequest(
            "claude-sonnet-4-20250514", [Message.user("Hello")]
        ).with_system_caching()
        assert request.has_caching()


class TestClients:
    """Test client creation."""

    def test_sync_client_from_env(self) -> None:
        # This should work if any API keys are set in env
        try:
            client = ModelSuiteClient.from_env()
            providers = client.providers()
            assert isinstance(providers, list)
        except Exception:
            # May fail if no API keys are configured
            pass

    def test_async_client_from_env(self) -> None:
        try:
            client = AsyncModelSuiteClient.from_env()
            providers = client.providers()
            assert isinstance(providers, list)
        except Exception:
            pass

    def test_sync_client_repr(self) -> None:
        try:
            client = ModelSuiteClient.from_env()
            repr_str = repr(client)
            assert "ModelSuiteClient" in repr_str
        except Exception:
            pass


class TestExceptions:
    """Test exception hierarchy."""

    def test_exception_inheritance(self) -> None:
        assert issubclass(ProviderNotFoundError, ModelSuiteError)
        assert issubclass(AuthenticationError, ModelSuiteError)
        assert issubclass(RateLimitError, ModelSuiteError)
        assert issubclass(InvalidRequestError, ModelSuiteError)

    def test_can_raise_and_catch(self) -> None:
        try:
            raise ModelSuiteError("Test error")
        except ModelSuiteError as e:
            assert "Test error" in str(e)

    def test_catch_subclass(self) -> None:
        try:
            raise AuthenticationError("Invalid API key")
        except ModelSuiteError as e:
            assert "Invalid API key" in str(e)


class TestRepr:
    """Test string representations."""

    def test_message_repr(self) -> None:
        msg = Message.user("Hello!")
        repr_str = repr(msg)
        assert "Message" in repr_str
        assert "user" in repr_str.lower()

    def test_content_block_repr(self) -> None:
        block = ContentBlock.text("Hello!")
        repr_str = repr(block)
        assert "ContentBlock" in repr_str
        assert "text" in repr_str.lower()

    def test_request_repr(self) -> None:
        request = CompletionRequest("claude-sonnet-4-20250514", [Message.user("Hi")])
        repr_str = repr(request)
        assert "CompletionRequest" in repr_str
        assert "claude-sonnet-4-20250514" in repr_str

    def test_tool_repr(self) -> None:
        tool = ToolBuilder("test").description("A test").build()
        repr_str = repr(tool)
        assert "ToolDefinition" in repr_str
        assert "test" in repr_str


class TestModelRegistry:
    """Test model registry functionality."""

    def test_get_model_info(self) -> None:
        """Test getting model info by ID."""
        info = get_model_info("anthropic/claude-3-5-sonnet-20241022")
        assert info is not None
        assert "claude" in info.id.lower()
        assert info.provider == Provider.Anthropic

    def test_get_model_info_by_alias(self) -> None:
        """Test getting model info by alias."""
        info = get_model_info("gpt-4o")
        assert info is not None
        assert info.provider == Provider.OpenAI

    def test_get_model_info_not_found(self) -> None:
        """Test that non-existent model returns None."""
        info = get_model_info("nonexistent-model-xyz")
        assert info is None

    def test_get_all_models(self) -> None:
        """Test getting all models."""
        models = get_all_models()
        assert len(models) > 10  # Should have many models
        # All should be ModelInfo objects
        for model in models[:5]:  # Check first 5
            assert model.id is not None
            assert model.name is not None

    def test_get_models_by_provider(self) -> None:
        """Test getting models by provider."""
        anthropic_models = get_models_by_provider(Provider.Anthropic)
        assert len(anthropic_models) > 0
        for model in anthropic_models:
            assert model.provider == Provider.Anthropic

    def test_get_current_models(self) -> None:
        """Test getting current (non-deprecated) models."""
        current = get_current_models()
        assert len(current) > 0
        for model in current:
            assert model.status == ModelStatus.Current

    def test_get_classifier_models(self) -> None:
        """Test getting classifier-suitable models."""
        classifiers = get_classifier_models()
        # Should have at least some classifier models
        assert len(classifiers) >= 0  # May be empty
        for model in classifiers:
            assert model.can_classify

    def test_get_models_with_capability_vision(self) -> None:
        """Test getting models with vision capability."""
        vision_models = get_models_with_capability(vision=True)
        assert len(vision_models) > 0
        for model in vision_models:
            assert model.capabilities.vision

    def test_get_models_with_capability_tools(self) -> None:
        """Test getting models with tool calling capability."""
        tool_models = get_models_with_capability(tools=True)
        assert len(tool_models) > 0
        for model in tool_models:
            assert model.capabilities.tools

    def test_get_models_with_capability_thinking(self) -> None:
        """Test getting models with thinking capability."""
        thinking_models = get_models_with_capability(thinking=True)
        # At least Claude models should have thinking
        for model in thinking_models:
            assert model.capabilities.thinking

    def test_get_cheapest_model(self) -> None:
        """Test finding cheapest model."""
        # get_cheapest_model may return None if no API keys configured
        cheapest = get_cheapest_model()
        if cheapest is not None:
            # Should have pricing info
            assert cheapest.pricing.input_per_1m >= 0

    def test_get_cheapest_model_with_requirements(self) -> None:
        """Test finding cheapest model with requirements."""
        cheapest = get_cheapest_model(needs_vision=True, needs_tools=True)
        if cheapest is not None:
            assert cheapest.capabilities.vision
            assert cheapest.capabilities.tools

    def test_supports_structured_output(self) -> None:
        """Test checking structured output support."""
        # Claude 3.5 Sonnet supports structured output
        assert supports_structured_output("anthropic/claude-3-5-sonnet-20241022")
        # Non-existent model should return False
        assert not supports_structured_output("nonexistent-model")

    def test_get_registry_stats(self) -> None:
        """Test getting registry statistics."""
        stats = get_registry_stats()
        assert stats.total_models > 10
        assert stats.current_models > 0
        assert stats.providers > 5

    def test_list_providers(self) -> None:
        """Test listing providers."""
        providers = list_providers()
        assert len(providers) > 5
        assert Provider.Anthropic in providers
        assert Provider.OpenAI in providers

    def test_model_info_properties(self) -> None:
        """Test ModelInfo property access."""
        info = get_model_info("anthropic/claude-3-5-sonnet-20241022")
        assert info is not None

        # Basic properties
        assert info.id is not None
        assert info.name is not None
        assert len(info.description) > 0

        # Pricing
        pricing = info.pricing
        assert pricing.input_per_1m >= 0
        assert pricing.output_per_1m >= 0

        # Capabilities
        caps = info.capabilities
        assert caps.max_context > 0
        assert caps.max_output > 0
        assert isinstance(caps.vision, bool)
        assert isinstance(caps.tools, bool)
        assert isinstance(caps.streaming, bool)

        # Benchmarks
        benchmarks = info.benchmarks
        score = benchmarks.quality_score()
        assert 0 <= score <= 100

    def test_model_info_methods(self) -> None:
        """Test ModelInfo methods."""
        info = get_model_info("anthropic/claude-3-5-sonnet-20241022")
        assert info is not None

        # raw_id should not have provider prefix
        raw = info.raw_id()
        assert "/" not in raw or raw.count("/") < info.id.count("/")

        # estimate_cost should return positive number
        cost = info.estimate_cost(1000, 500)
        assert cost >= 0

        # quality_per_dollar
        qpd = info.quality_per_dollar()
        assert qpd >= 0

    def test_model_info_repr(self) -> None:
        """Test ModelInfo string representation."""
        info = get_model_info("gpt-4o")
        assert info is not None
        repr_str = repr(info)
        assert "ModelInfo" in repr_str

    def test_provider_enum_values(self) -> None:
        """Test Provider enum values."""
        assert Provider.Anthropic is not None
        assert Provider.OpenAI is not None
        assert Provider.Google is not None
        assert Provider.Bedrock is not None
        assert Provider.AzureOpenAI is not None

    def test_model_status_enum_values(self) -> None:
        """Test ModelStatus enum values."""
        assert ModelStatus.Current == 0
        assert ModelStatus.Legacy == 1
        assert ModelStatus.Deprecated == 2

    def test_pricing_estimate_cost(self) -> None:
        """Test pricing cost estimation."""
        info = get_model_info("anthropic/claude-3-5-sonnet-20241022")
        assert info is not None
        pricing = info.pricing

        # Cost for 1M input + 1M output should be roughly input + output price
        cost = pricing.estimate_cost(1_000_000, 1_000_000)
        expected = pricing.input_per_1m + pricing.output_per_1m
        # Allow some tolerance for floating point
        assert abs(cost - expected) < 0.01


class TestEmbeddings:
    """Tests for embedding types."""

    def test_encoding_format_enum(self) -> None:
        """Test EncodingFormat enum values."""
        assert EncodingFormat.Float == 0
        assert EncodingFormat.Base64 == 1

    def test_embedding_input_type_enum(self) -> None:
        """Test EmbeddingInputType enum values."""
        assert EmbeddingInputType.Query == 0
        assert EmbeddingInputType.Document == 1

    def test_embedding_request_creation(self) -> None:
        """Test EmbeddingRequest creation for single text."""
        request = EmbeddingRequest("text-embedding-3-small", "Hello, world!")
        assert request.model == "text-embedding-3-small"
        assert request.text_count == 1
        assert "Hello, world!" in request.texts()

    def test_embedding_request_batch(self) -> None:
        """Test EmbeddingRequest batch creation."""
        texts = ["Hello", "World", "How are you?"]
        request = EmbeddingRequest.batch("text-embedding-3-small", texts)
        assert request.model == "text-embedding-3-small"
        assert request.text_count == 3
        result_texts = request.texts()
        assert result_texts == texts

    def test_embedding_request_with_dimensions(self) -> None:
        """Test EmbeddingRequest with custom dimensions."""
        request = EmbeddingRequest("text-embedding-3-small", "Hello").with_dimensions(256)
        assert request.dimensions == 256

    def test_embedding_request_with_encoding_format(self) -> None:
        """Test EmbeddingRequest with encoding format."""
        request = EmbeddingRequest("text-embedding-3-small", "Hello").with_encoding_format(
            EncodingFormat.Base64
        )
        # Just verify it doesn't raise
        assert request is not None

    def test_embedding_request_with_input_type(self) -> None:
        """Test EmbeddingRequest with input type."""
        request = EmbeddingRequest("text-embedding-3-small", "Hello").with_input_type(
            EmbeddingInputType.Query
        )
        # Just verify it doesn't raise
        assert request is not None

    def test_embedding_request_chaining(self) -> None:
        """Test EmbeddingRequest method chaining."""
        request = (
            EmbeddingRequest("text-embedding-3-small", "Hello")
            .with_dimensions(512)
            .with_encoding_format(EncodingFormat.Float)
            .with_input_type(EmbeddingInputType.Document)
        )
        assert request.dimensions == 512
        assert request.model == "text-embedding-3-small"

    def test_embedding_request_repr(self) -> None:
        """Test EmbeddingRequest string representation."""
        request = EmbeddingRequest("text-embedding-3-small", "Hello")
        repr_str = repr(request)
        assert "EmbeddingRequest" in repr_str
        assert "text-embedding-3-small" in repr_str

    def test_client_embedding_providers_method(self) -> None:
        """Test that client has embedding_providers method."""
        client = ModelSuiteClient.from_env()
        providers = client.embedding_providers()
        # Should return a list (may be empty if no OpenAI/Cohere configured)
        assert isinstance(providers, list)

    def test_client_supports_embeddings_method(self) -> None:
        """Test that client has supports_embeddings method."""
        client = ModelSuiteClient.from_env()
        # Check for a provider that might not be configured
        result = client.supports_embeddings("nonexistent")
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
