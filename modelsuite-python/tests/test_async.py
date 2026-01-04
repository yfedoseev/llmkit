"""Async integration tests for LLMKit Python bindings.

These tests make actual API calls and require valid API keys.
Tests are automatically skipped if the required API key is not set.

Set the following environment variables to enable tests:
- ANTHROPIC_API_KEY: Anthropic tests
- OPENAI_API_KEY: OpenAI tests
"""

import os

import pytest

from modelsuite import (
    AsyncLLMKitClient,
    CompletionRequest,
    ContentBlock,
    LLMKitClient,
    Message,
    StopReason,
    TokenCountRequest,
    ToolBuilder,
)


def has_env(key: str) -> bool:
    return bool(os.environ.get(key))


# =============================================================================
# Anthropic Async Tests
# =============================================================================


@pytest.mark.skipif(
    not has_env("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestAnthropicAsync:
    """Async tests for Anthropic provider."""

    @pytest.fixture
    def client(self) -> AsyncLLMKitClient:
        return AsyncLLMKitClient.from_env()

    @pytest.mark.asyncio
    async def test_simple_completion(self, client) -> None:
        """Test a simple async completion request."""
        request = CompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message.user("What is 2+2? Reply with just the number.")],
            max_tokens=50,
        )

        response = await client.complete(request)

        assert response.id is not None
        assert "claude" in response.model.lower()
        assert "4" in response.text_content()
        assert response.stop_reason == StopReason.EndTurn
        assert response.usage is not None
        assert response.usage.input_tokens > 0
        assert response.usage.output_tokens > 0

    @pytest.mark.asyncio
    async def test_system_prompt(self, client) -> None:
        """Test completion with system prompt."""
        request = (
            CompletionRequest(
                model="claude-sonnet-4-20250514",
                messages=[Message.user("What are you?")],
            )
            .with_system("You are a friendly robot named R2D2. Always introduce yourself.")
            .with_max_tokens(100)
        )

        response = await client.complete(request)
        text = response.text_content().lower()
        assert "r2d2" in text or "robot" in text

    @pytest.mark.asyncio
    async def test_streaming(self, client) -> None:
        """Test async streaming response."""
        request = (
            CompletionRequest(
                model="claude-sonnet-4-20250514",
                messages=[Message.user("Count from 1 to 5, one number per line.")],
            )
            .with_max_tokens(100)
            .with_streaming()
        )

        chunks = []
        async for chunk in client.complete_stream(request):
            if chunk.text:
                chunks.append(chunk.text)
            if chunk.is_done:
                break

        full_text = "".join(chunks)
        assert "1" in full_text
        assert "5" in full_text

    @pytest.mark.asyncio
    async def test_tool_use(self, client) -> None:
        """Test tool use in async context."""
        tool = (
            ToolBuilder("get_weather")
            .description("Get the current weather in a city")
            .string_param("city", "The city name", required=True)
            .build()
        )

        request = (
            CompletionRequest(
                model="claude-sonnet-4-20250514",
                messages=[Message.user("What is the weather in Paris?")],
            )
            .with_tools([tool])
            .with_max_tokens(200)
        )

        response = await client.complete(request)

        assert response.has_tool_use()
        assert response.stop_reason == StopReason.ToolUse

        tool_uses = response.tool_uses()
        assert len(tool_uses) > 0

        tool_info = tool_uses[0].as_tool_use()
        assert tool_info is not None
        id, name, input_dict = tool_info
        assert name == "get_weather"
        assert "city" in input_dict

    @pytest.mark.asyncio
    async def test_token_counting(self, client) -> None:
        """Test async token counting."""
        request = TokenCountRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message.user("Hello, how are you?")],
            system="You are a helpful assistant.",
        )

        result = await client.count_tokens(request)
        assert result.input_tokens > 0
        assert result.input_tokens < 100

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, client) -> None:
        """Test multi-turn conversation in async context."""
        request = CompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[
                Message.user("My name is Alice."),
                Message.assistant("Hello Alice! Nice to meet you."),
                Message.user("What is my name?"),
            ],
            max_tokens=50,
        )

        response = await client.complete(request)
        assert "alice" in response.text_content().lower()

    @pytest.mark.asyncio
    async def test_vision(self, client) -> None:
        """Test vision capability with image URL."""
        # Use a simple, public test image
        request = CompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[
                Message.user_with_content(
                    [
                        ContentBlock.text("What do you see in this image? Be brief."),
                        ContentBlock.image_url(
                            "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
                        ),
                    ]
                )
            ],
            max_tokens=100,
        )

        response = await client.complete(request)
        # Should get some description
        assert len(response.text_content()) > 10


# =============================================================================
# OpenAI Async Tests
# =============================================================================


@pytest.mark.skipif(
    not has_env("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set",
)
class TestOpenAIAsync:
    """Async tests for OpenAI provider."""

    @pytest.fixture
    def client(self) -> AsyncLLMKitClient:
        return AsyncLLMKitClient.from_env()

    @pytest.mark.asyncio
    async def test_simple_completion(self, client) -> None:
        """Test a simple async completion request."""
        request = CompletionRequest(
            model="gpt-4o-mini",
            messages=[Message.user("What is 3+3? Reply with just the number.")],
            max_tokens=50,
        )

        response = await client.complete(request)

        assert response.id is not None
        assert "gpt-4o-mini" in response.model.lower()
        assert "6" in response.text_content()

    @pytest.mark.asyncio
    async def test_json_output(self, client) -> None:
        """Test JSON output mode."""
        request = (
            CompletionRequest(
                model="gpt-4o-mini",
                messages=[
                    Message.user("Return a JSON object with a 'greeting' field saying 'hello'")
                ],
            )
            .with_json_output()
            .with_max_tokens(100)
        )

        response = await client.complete(request)
        text = response.text_content()

        import json

        parsed = json.loads(text)
        assert "greeting" in parsed

    @pytest.mark.asyncio
    async def test_structured_output(self, client) -> None:
        """Test structured output with JSON schema."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
            "additionalProperties": False,
        }

        request = (
            CompletionRequest(
                model="gpt-4o-mini",
                messages=[Message.user("Generate info for a person named Bob who is 25 years old")],
            )
            .with_json_schema("person", schema)
            .with_max_tokens(100)
        )

        response = await client.complete(request)

        import json

        parsed = json.loads(response.text_content())
        assert parsed["name"] == "Bob"
        assert parsed["age"] == 25

    @pytest.mark.asyncio
    async def test_streaming(self, client) -> None:
        """Test async streaming response."""
        request = (
            CompletionRequest(
                model="gpt-4o-mini",
                messages=[Message.user("Say 'hello world'")],
            )
            .with_max_tokens(50)
            .with_streaming()
        )

        chunks = []
        async for chunk in client.complete_stream(request):
            if chunk.text:
                chunks.append(chunk.text)
            if chunk.is_done:
                break

        full_text = "".join(chunks).lower()
        assert "hello" in full_text


# =============================================================================
# Multi-Provider Async Tests
# =============================================================================


@pytest.mark.skipif(
    not (has_env("ANTHROPIC_API_KEY") and has_env("OPENAI_API_KEY")),
    reason="Both ANTHROPIC_API_KEY and OPENAI_API_KEY required",
)
class TestMultiProviderAsync:
    """Test using multiple providers in async context."""

    @pytest.fixture
    def client(self) -> AsyncLLMKitClient:
        return AsyncLLMKitClient.from_env()

    @pytest.mark.asyncio
    async def test_switch_providers(self, client) -> None:
        """Test switching between providers."""
        # Anthropic
        anthropic_request = CompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message.user("Say 'Anthropic'")],
            max_tokens=20,
        )
        anthropic_response = await client.complete(anthropic_request)
        assert "claude" in anthropic_response.model.lower()

        # OpenAI
        openai_request = CompletionRequest(
            model="gpt-4o-mini",
            messages=[Message.user("Say 'OpenAI'")],
            max_tokens=20,
        )
        openai_response = await client.complete(openai_request)
        assert "gpt" in openai_response.model.lower()

    @pytest.mark.asyncio
    async def test_complete_with_provider(self, client) -> None:
        """Test explicit provider selection."""
        request = CompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message.user("Say 'test'")],
            max_tokens=20,
        )

        response = await client.complete_with_provider("anthropic", request)
        assert "claude" in response.model.lower()


# =============================================================================
# Error Handling Async Tests
# =============================================================================


@pytest.mark.skipif(
    not has_env("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestErrorHandlingAsync:
    """Test error handling in async context."""

    @pytest.fixture
    def client(self) -> AsyncLLMKitClient:
        return AsyncLLMKitClient.from_env()

    @pytest.mark.asyncio
    async def test_invalid_model(self, client) -> None:
        """Test handling of invalid model."""
        request = CompletionRequest(
            model="non-existent-model-12345",
            messages=[Message.user("Hello")],
            max_tokens=50,
        )

        with pytest.raises(Exception):
            await client.complete(request)

    @pytest.mark.asyncio
    async def test_empty_messages(self, client) -> None:
        """Test handling of empty messages."""
        request = CompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[],
            max_tokens=50,
        )

        with pytest.raises(Exception):
            await client.complete(request)


# =============================================================================
# Sync vs Async Comparison
# =============================================================================


@pytest.mark.skipif(
    not has_env("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
class TestSyncAsyncComparison:
    """Compare sync and async behavior."""

    @pytest.mark.asyncio
    async def test_same_result(self) -> None:
        """Verify sync and async produce same results."""
        sync_client = LLMKitClient.from_env()
        async_client = AsyncLLMKitClient.from_env()

        request = CompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message.user("What is 1+1? Reply with just the number.")],
            max_tokens=10,
            temperature=0,  # Deterministic
        )

        sync_response = sync_client.complete(request)
        async_response = await async_client.complete(request)

        # Both should contain "2"
        assert "2" in sync_response.text_content()
        assert "2" in async_response.text_content()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
