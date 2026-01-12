"""
Response Caching Example

Demonstrates client-side response caching to reduce API costs
by caching identical requests. Uses qwen/qwen3-32b on OpenRouter.

Cache hit = no API call = free!

Note: We use ThinkingConfig.disabled() to disable the model's reasoning mode,
which allows using lower max_tokens values efficiently.

Requirements:
- Set OPENROUTER_API_KEY environment variable
- uv sync

Run:
    uv run python 17_response_caching.py
"""

from llmkit import LLMKitClient, Message, CompletionRequest, ThinkingConfig


def main():
    # Note: Python binding may not expose CachingProvider directly.
    # This example demonstrates the concept using simple memoization.
    # For full caching support, use the Rust or Node.js examples.

    client = LLMKitClient.from_env()
    model = "openrouter/qwen/qwen3-32b"

    # Simple in-memory cache for demonstration
    cache = {}

    def cached_complete(request_key: str, request: CompletionRequest):
        """Complete with simple caching."""
        if request_key in cache:
            print("  [CACHE HIT - no API call]")
            return cache[request_key]
        else:
            print("  [CACHE MISS - API call]")
            response = client.complete(request)
            cache[request_key] = response
            return response

    print("Response Caching Example")
    print("=" * 50)

    # First request - will hit the API
    print("\n1. First request (France):")
    print("-" * 40)

    request1 = CompletionRequest(
        model=model,
        messages=[Message.user("What is the capital of France?")],
        system="Answer briefly in one sentence.",
        max_tokens=100,
    ).with_thinking_config(ThinkingConfig.disabled())
    response = cached_complete("france", request1)
    print(f"Response: {response.text_content().strip()}")

    # Second identical request - should hit the cache
    print("\n2. Same request again (France):")
    print("-" * 40)

    request2 = CompletionRequest(
        model=model,
        messages=[Message.user("What is the capital of France?")],
        system="Answer briefly in one sentence.",
        max_tokens=100,
    ).with_thinking_config(ThinkingConfig.disabled())
    response = cached_complete("france", request2)
    print(f"Response: {response.text_content().strip()}")

    # Third request with different question - cache miss
    print("\n3. Different question (Germany):")
    print("-" * 40)

    request3 = CompletionRequest(
        model=model,
        messages=[Message.user("What is the capital of Germany?")],
        system="Answer briefly in one sentence.",
        max_tokens=100,
    ).with_thinking_config(ThinkingConfig.disabled())
    response = cached_complete("germany", request3)
    print(f"Response: {response.text_content().strip()}")

    # Fourth request - same as first (cache hit)
    print("\n4. First question again (France):")
    print("-" * 40)

    request4 = CompletionRequest(
        model=model,
        messages=[Message.user("What is the capital of France?")],
        system="Answer briefly in one sentence.",
        max_tokens=100,
    ).with_thinking_config(ThinkingConfig.disabled())
    response = cached_complete("france", request4)
    print(f"Response: {response.text_content().strip()}")

    # Summary
    print("\n" + "=" * 50)
    print("Summary:")
    print("  Total requests: 4")
    print("  API calls made: 2 (misses)")
    print("  Free responses: 2 (hits)")
    print("  Hit rate: 50%")
    print("\nCaching saves API costs on repeated identical requests!")


if __name__ == "__main__":
    main()
