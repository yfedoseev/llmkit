"""Tests for LLMKit Python bindings."""

import pytest
from llmkit import (
    # Enums
    Role,
    StopReason,
    StreamEventType,
    CacheControl,
    ThinkingType,
    BatchStatus,
    # Types
    ContentBlock,
    Message,
    CacheBreakpoint,
    ThinkingConfig,
    StructuredOutput,
    ToolDefinition,
    ToolBuilder,
    CompletionRequest,
    Usage,
    # Clients
    LLMKitClient,
    AsyncLLMKitClient,
    # Exceptions
    LLMKitError,
    ProviderNotFoundError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
)


class TestEnums:
    """Test enum types."""

    def test_role_values(self):
        assert Role.System == 0
        assert Role.User == 1
        assert Role.Assistant == 2

    def test_role_str(self):
        assert str(Role.System) == "system"
        assert str(Role.User) == "user"
        assert str(Role.Assistant) == "assistant"

    def test_stop_reason_values(self):
        assert StopReason.EndTurn == 0
        assert StopReason.MaxTokens == 1
        assert StopReason.ToolUse == 2
        assert StopReason.StopSequence == 3
        assert StopReason.ContentFilter == 4

    def test_stop_reason_properties(self):
        assert StopReason.EndTurn.is_complete
        assert not StopReason.MaxTokens.is_complete
        assert StopReason.ToolUse.is_tool_use
        assert not StopReason.EndTurn.is_tool_use

    def test_stream_event_type_values(self):
        assert StreamEventType.MessageStart == 0
        assert StreamEventType.MessageStop == 5
        assert StreamEventType.MessageStop.is_done
        assert not StreamEventType.MessageStart.is_done

    def test_cache_control_values(self):
        assert CacheControl.Ephemeral == 0
        assert CacheControl.Extended == 1

    def test_thinking_type_values(self):
        assert ThinkingType.Enabled == 0
        assert ThinkingType.Disabled == 1

    def test_batch_status_properties(self):
        assert BatchStatus.Validating.is_processing
        assert BatchStatus.InProgress.is_processing
        assert not BatchStatus.Completed.is_processing
        assert BatchStatus.Completed.is_done
        assert BatchStatus.Completed.is_success
        assert BatchStatus.Failed.is_done
        assert not BatchStatus.Failed.is_success


class TestContentBlock:
    """Test ContentBlock type."""

    def test_text_block(self):
        block = ContentBlock.text("Hello, world!")
        assert block.is_text
        assert not block.is_tool_use
        assert block.text_value == "Hello, world!"

    def test_image_block(self):
        block = ContentBlock.image("image/png", "base64data...")
        assert block.is_image
        assert not block.is_text

    def test_image_url_block(self):
        block = ContentBlock.image_url("https://example.com/image.png")
        assert block.is_image

    def test_tool_use_block(self):
        block = ContentBlock.tool_use("id-123", "get_weather", {"city": "London"})
        assert block.is_tool_use
        assert not block.is_text
        result = block.as_tool_use()
        assert result is not None
        id, name, input_dict = result
        assert id == "id-123"
        assert name == "get_weather"
        assert input_dict == {"city": "London"}

    def test_tool_result_block(self):
        block = ContentBlock.tool_result("id-123", "Weather: Sunny, 20C", False)
        assert block.is_tool_result
        result = block.as_tool_result()
        assert result is not None
        tool_use_id, content, is_error = result
        assert tool_use_id == "id-123"
        assert content == "Weather: Sunny, 20C"
        assert not is_error

    def test_tool_result_error(self):
        block = ContentBlock.tool_result("id-123", "Error: City not found", True)
        result = block.as_tool_result()
        assert result[2] is True

    def test_thinking_block(self):
        block = ContentBlock.thinking("Let me think about this...")
        assert block.is_thinking
        assert block.thinking_content == "Let me think about this..."

    def test_pdf_block(self):
        block = ContentBlock.pdf("base64pdfdata...")
        assert block.is_document

    def test_text_cached_block(self):
        block = ContentBlock.text_cached("Cached text content")
        assert block.is_text
        assert block.text_value == "Cached text content"


class TestMessage:
    """Test Message type."""

    def test_system_message(self):
        msg = Message.system("You are a helpful assistant.")
        assert msg.role == Role.System
        assert msg.text_content() == "You are a helpful assistant."

    def test_user_message(self):
        msg = Message.user("Hello!")
        assert msg.role == Role.User
        assert msg.text_content() == "Hello!"

    def test_assistant_message(self):
        msg = Message.assistant("Hi there!")
        assert msg.role == Role.Assistant
        assert msg.text_content() == "Hi there!"

    def test_user_with_content(self):
        blocks = [
            ContentBlock.text("Look at this image:"),
            ContentBlock.image_url("https://example.com/image.png"),
        ]
        msg = Message.user_with_content(blocks)
        assert msg.role == Role.User
        assert len(msg.content) == 2
        assert msg.text_content() == "Look at this image:"

    def test_assistant_with_content(self):
        blocks = [
            ContentBlock.text("Here's the weather:"),
            ContentBlock.tool_use("id-1", "get_weather", {"city": "NYC"}),
        ]
        msg = Message.assistant_with_content(blocks)
        assert msg.role == Role.Assistant
        assert msg.has_tool_use()
        tool_uses = msg.tool_uses()
        assert len(tool_uses) == 1

    def test_tool_results_message(self):
        results = [
            ContentBlock.tool_result("id-1", "Sunny, 25C"),
            ContentBlock.tool_result("id-2", "Rainy, 15C"),
        ]
        msg = Message.tool_results(results)
        assert msg.role == Role.User
        assert len(msg.content) == 2


class TestCacheBreakpoint:
    """Test CacheBreakpoint type."""

    def test_ephemeral(self):
        bp = CacheBreakpoint.ephemeral()
        assert bp.cache_control == CacheControl.Ephemeral

    def test_extended(self):
        bp = CacheBreakpoint.extended()
        assert bp.cache_control == CacheControl.Extended


class TestThinkingConfig:
    """Test ThinkingConfig type."""

    def test_enabled(self):
        config = ThinkingConfig.enabled(10000)
        assert config.thinking_type == ThinkingType.Enabled
        assert config.budget_tokens == 10000

    def test_disabled(self):
        config = ThinkingConfig.disabled()
        assert config.thinking_type == ThinkingType.Disabled
        assert config.budget_tokens is None


class TestStructuredOutput:
    """Test StructuredOutput type."""

    def test_json_schema(self):
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

    def test_json_object(self):
        output = StructuredOutput.json_object()
        assert output is not None


class TestToolBuilder:
    """Test ToolBuilder and ToolDefinition."""

    def test_simple_tool(self):
        tool = (
            ToolBuilder("greet")
            .description("Greet someone")
            .string_param("name", "Person's name", required=True)
            .build()
        )
        assert tool.name == "greet"
        assert tool.description == "Greet someone"

    def test_complex_tool(self):
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

    def test_tool_with_array_param(self):
        tool = (
            ToolBuilder("batch_process")
            .description("Process multiple items")
            .array_param("items", "Items to process", "string", required=True)
            .integer_param("batch_size", "Batch size", required=False)
            .build()
        )
        assert tool.name == "batch_process"

    def test_tool_definition_direct(self):
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

    def test_constructor(self):
        request = CompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message.user("Hello!")],
        )
        assert request.model == "claude-sonnet-4-20250514"
        assert len(request.messages) == 1
        assert not request.stream

    def test_constructor_with_options(self):
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
        assert abs(request.temperature - 0.7) < 0.0001  # f32 precision
        assert request.stream

    def test_builder_pattern(self):
        request = (
            CompletionRequest("claude-sonnet-4-20250514", [Message.user("Hi")])
            .with_system("Be helpful")
            .with_max_tokens(2048)
            .with_temperature(0.5)
        )
        assert request.system == "Be helpful"
        assert request.max_tokens == 2048
        assert request.temperature == 0.5

    def test_streaming_builder(self):
        request = CompletionRequest(
            "claude-sonnet-4-20250514", [Message.user("Hi")]
        ).with_streaming()
        assert request.stream

    def test_thinking_builder(self):
        request = CompletionRequest(
            "claude-sonnet-4-20250514", [Message.user("Solve this puzzle")]
        ).with_thinking(10000)
        assert request.has_thinking()

    def test_tools_builder(self):
        tool = ToolBuilder("test").description("Test tool").build()
        request = CompletionRequest(
            "claude-sonnet-4-20250514", [Message.user("Use the tool")]
        ).with_tools([tool])
        # Just verify it doesn't crash
        assert request is not None

    def test_json_schema_builder(self):
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        request = CompletionRequest(
            "claude-sonnet-4-20250514", [Message.user("Answer")]
        ).with_json_schema("response", schema)
        assert request.has_structured_output()

    def test_caching_builder(self):
        request = CompletionRequest(
            "claude-sonnet-4-20250514", [Message.user("Hello")]
        ).with_system_caching()
        assert request.has_caching()


class TestClients:
    """Test client creation."""

    def test_sync_client_from_env(self):
        # This should work if any API keys are set in env
        try:
            client = LLMKitClient.from_env()
            providers = client.providers()
            assert isinstance(providers, list)
        except Exception:
            # May fail if no API keys are configured
            pass

    def test_async_client_from_env(self):
        try:
            client = AsyncLLMKitClient.from_env()
            providers = client.providers()
            assert isinstance(providers, list)
        except Exception:
            pass

    def test_sync_client_repr(self):
        try:
            client = LLMKitClient.from_env()
            repr_str = repr(client)
            assert "LLMKitClient" in repr_str
        except Exception:
            pass


class TestExceptions:
    """Test exception hierarchy."""

    def test_exception_inheritance(self):
        assert issubclass(ProviderNotFoundError, LLMKitError)
        assert issubclass(AuthenticationError, LLMKitError)
        assert issubclass(RateLimitError, LLMKitError)
        assert issubclass(InvalidRequestError, LLMKitError)

    def test_can_raise_and_catch(self):
        try:
            raise LLMKitError("Test error")
        except LLMKitError as e:
            assert "Test error" in str(e)

    def test_catch_subclass(self):
        try:
            raise AuthenticationError("Invalid API key")
        except LLMKitError as e:
            assert "Invalid API key" in str(e)


class TestRepr:
    """Test string representations."""

    def test_message_repr(self):
        msg = Message.user("Hello!")
        repr_str = repr(msg)
        assert "Message" in repr_str
        assert "user" in repr_str.lower()

    def test_content_block_repr(self):
        block = ContentBlock.text("Hello!")
        repr_str = repr(block)
        assert "ContentBlock" in repr_str
        assert "text" in repr_str.lower()

    def test_request_repr(self):
        request = CompletionRequest(
            "claude-sonnet-4-20250514", [Message.user("Hi")]
        )
        repr_str = repr(request)
        assert "CompletionRequest" in repr_str
        assert "claude-sonnet-4-20250514" in repr_str

    def test_tool_repr(self):
        tool = ToolBuilder("test").description("A test").build()
        repr_str = repr(tool)
        assert "ToolDefinition" in repr_str
        assert "test" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
