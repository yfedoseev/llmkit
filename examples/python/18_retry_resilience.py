"""
Retry & Resilience Example

Demonstrates configuring automatic retry behavior for handling
transient failures like rate limits, timeouts, and server errors.

Requirements:
- Set OPENROUTER_API_KEY environment variable
- uv sync

Run:
    uv run python 18_retry_resilience.py
"""

from llmkit import LLMKitClient, Message, CompletionRequest, RetryConfig


def main():
    print("Retry & Resilience Configuration Example")
    print("=" * 50)

    # ========================================
    # Example 1: Default Retry (Recommended)
    # ========================================
    print("\n1. Default Retry Configuration")
    print("-" * 40)
    print("Settings: 10 retries, 1s-5min exponential backoff, jitter enabled")

    client = LLMKitClient.from_env()  # Uses default retry

    request = CompletionRequest(
        model="openrouter/qwen/qwen3-32b",
        messages=[Message.user("Say 'retry test passed' in exactly 3 words.")],
        max_tokens=50,
    ).without_thinking()

    response = client.complete(request)
    print(f"Response: {response.text_content().strip()}")

    # ========================================
    # Example 2: Production Retry Config
    # ========================================
    print("\n2. Production Retry Configuration")
    print("-" * 40)

    production_config = RetryConfig.production()
    print(f"Settings:")
    print(f"  Max retries: {production_config.max_retries}")
    print(f"  Initial delay: {production_config.initial_delay_ms}ms")
    print(f"  Max delay: {production_config.max_delay_ms}ms")
    print(f"  Backoff multiplier: {production_config.backoff_multiplier}")
    print(f"  Jitter: {production_config.jitter}")

    client = LLMKitClient.from_env(retry_config=production_config)

    request = CompletionRequest(
        model="openrouter/qwen/qwen3-32b",
        messages=[Message.user("Say 'production config works'")],
        max_tokens=50,
    ).without_thinking()

    response = client.complete(request)
    print(f"Response: {response.text_content().strip()}")

    # ========================================
    # Example 3: Conservative Retry Config
    # ========================================
    print("\n3. Conservative Retry Configuration")
    print("-" * 40)

    conservative_config = RetryConfig.conservative()
    print(f"Settings:")
    print(f"  Max retries: {conservative_config.max_retries}")
    print(f"  Initial delay: {conservative_config.initial_delay_ms}ms")
    print(f"  Max delay: {conservative_config.max_delay_ms}ms")
    print("Best for: Quick failures, interactive applications")

    client = LLMKitClient.from_env(retry_config=conservative_config)

    request = CompletionRequest(
        model="openrouter/qwen/qwen3-32b",
        messages=[Message.user("Say 'conservative works'")],
        max_tokens=50,
    ).without_thinking()

    response = client.complete(request)
    print(f"Response: {response.text_content().strip()}")

    # ========================================
    # Example 4: Custom Retry Config
    # ========================================
    print("\n4. Custom Retry Configuration")
    print("-" * 40)

    custom_config = RetryConfig(
        max_retries=5,
        initial_delay_ms=500,
        max_delay_ms=10000,
        backoff_multiplier=1.5,
        jitter=True,
    )

    print("Custom settings:")
    print(f"  Max retries: {custom_config.max_retries}")
    print(f"  Initial delay: {custom_config.initial_delay_ms}ms")
    print(f"  Max delay: {custom_config.max_delay_ms}ms")
    print(f"  Backoff multiplier: {custom_config.backoff_multiplier}")
    print(f"  Jitter: {custom_config.jitter}")

    client = LLMKitClient.from_env(retry_config=custom_config)

    request = CompletionRequest(
        model="openrouter/qwen/qwen3-32b",
        messages=[Message.user("Say 'custom config works'")],
        max_tokens=50,
    ).without_thinking()

    response = client.complete(request)
    print(f"Response: {response.text_content().strip()}")

    # ========================================
    # Example 5: No Retry (for testing/debugging)
    # ========================================
    print("\n5. No Retry Configuration")
    print("-" * 40)
    print("Useful for: Testing, debugging, fail-fast scenarios")

    # Option 1: Using RetryConfig.none()
    client = LLMKitClient.from_env(retry_config=RetryConfig.none())

    # Option 2: Using False (Python only)
    # client = LLMKitClient.from_env(retry_config=False)

    request = CompletionRequest(
        model="openrouter/qwen/qwen3-32b",
        messages=[Message.user("Say 'no retry works'")],
        max_tokens=50,
    ).without_thinking()

    response = client.complete(request)
    print(f"Response: {response.text_content().strip()}")

    # Summary
    print("\n" + "=" * 50)
    print("Summary: Retry Configurations")
    print("=" * 50)
    print("""
Retry handles these transient errors automatically:
  - Rate limits (429)
  - Server errors (5xx)
  - Timeouts
  - Connection errors

Configuration options:
  - Default (None)              -> 10 retries, 5min max
  - RetryConfig.production()    -> Same as default
  - RetryConfig.conservative()  -> 3 retries, 30s max
  - RetryConfig.none()          -> No retries
  - RetryConfig(...)            -> Custom settings
  - False                       -> No retries (Python only)
""")


if __name__ == "__main__":
    main()
