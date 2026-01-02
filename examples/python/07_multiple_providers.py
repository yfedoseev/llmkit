"""
Multiple Providers Example

Demonstrates how to configure and use multiple LLM providers.

Requirements:
- Set multiple provider API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)

Run:
    python 07_multiple_providers.py
"""

from llmkit import LLMKitClient, Message, CompletionRequest


def using_from_env():
    """Auto-detect providers from environment variables."""
    client = LLMKitClient.from_env()

    # List all detected providers
    providers = client.providers()
    print(f"Detected providers: {providers}")
    print(f"Default provider: {client.default_provider}")

    # Use the default provider
    response = client.complete(
        CompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message.user("Say hello")],
            max_tokens=50,
        )
    )
    print(f"\nDefault provider response: {response.text_content()}")


def explicit_provider_config():
    """Configure providers explicitly."""
    # Configure specific providers
    client = LLMKitClient(
        providers={
            "anthropic": {"api_key": "your-anthropic-key"},
            "openai": {"api_key": "your-openai-key"},
            # Azure requires additional config
            "azure": {
                "api_key": "your-azure-key",
                "endpoint": "https://your-resource.openai.azure.com",
                "deployment": "gpt-4",
            },
            # Bedrock uses AWS credentials
            "bedrock": {"region": "us-east-1"},
            # Local Ollama
            "ollama": {"base_url": "http://localhost:11434"},
        },
        default_provider="anthropic",
    )

    print(f"Configured providers: {client.providers()}")


def switch_between_providers():
    """Switch between providers for different tasks."""
    client = LLMKitClient.from_env()

    providers = client.providers()
    print(f"Available: {providers}\n")

    prompt = "What's 2+2? Answer with just the number."

    # Try different providers if available
    for provider in ["anthropic", "openai", "groq"]:
        if provider not in providers:
            print(f"{provider}: Not configured")
            continue

        # Map provider to a model
        models = {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-4o",
            "groq": "llama-3.3-70b-versatile",
        }

        model = models.get(provider)
        if not model:
            continue

        try:
            response = client.complete_with_provider(
                provider,
                CompletionRequest(
                    model=model,
                    messages=[Message.user(prompt)],
                    max_tokens=20,
                ),
            )
            print(f"{provider} ({model}): {response.text_content().strip()}")
        except Exception as e:
            print(f"{provider}: Error - {e}")


def cost_aware_routing():
    """Route requests to cheaper providers when appropriate."""
    from llmkit import get_model_info, get_cheapest_model

    # Find the cheapest model that meets requirements
    cheapest = get_cheapest_model(
        min_context=None,
        needs_vision=False,
        needs_tools=True,
    )

    if cheapest:
        print(f"Cheapest model with tools: {cheapest.name}")
        print(f"  Provider: {cheapest.provider}")
        print(f"  Price: ${cheapest.pricing.input_per_1m}/1M input tokens")

    # Compare costs for a specific model
    models_to_compare = [
        "claude-sonnet-4-20250514",
        "gpt-4o",
        "gpt-4o-mini",
    ]

    print("\nModel cost comparison:")
    for model_id in models_to_compare:
        info = get_model_info(model_id)
        if info:
            cost = info.estimate_cost(1000, 500)  # 1000 input, 500 output
            print(
                f"  {info.name}: ${cost:.6f} "
                f"(${info.pricing.input_per_1m}/1M in, ${info.pricing.output_per_1m}/1M out)"
            )


def provider_fallback():
    """Implement fallback between providers."""
    client = LLMKitClient.from_env()

    # Order providers by preference
    provider_priority = ["anthropic", "openai", "groq"]

    request = CompletionRequest(
        model="",  # Will set per provider
        messages=[Message.user("What is Python?")],
        max_tokens=100,
    )

    # Model mapping
    model_map = {
        "anthropic": "claude-sonnet-4-20250514",
        "openai": "gpt-4o",
        "groq": "llama-3.3-70b-versatile",
    }

    available = set(client.providers())

    for provider in provider_priority:
        if provider not in available:
            print(f"Skipping {provider} (not configured)")
            continue

        model = model_map.get(provider)
        if not model:
            continue

        try:
            print(f"Trying {provider}...")
            response = client.complete_with_provider(
                provider,
                CompletionRequest(
                    model=model,
                    messages=request.messages,
                    max_tokens=request.max_tokens,
                ),
            )
            print(f"Success with {provider}!")
            print(f"Response: {response.text_content()[:100]}...")
            return
        except Exception as e:
            print(f"Failed with {provider}: {e}")
            continue

    print("All providers failed!")


def main():
    print("=" * 50)
    print("Example 1: Auto-detect from Environment")
    print("=" * 50)
    using_from_env()

    print("\n" + "=" * 50)
    print("Example 2: Switch Between Providers")
    print("=" * 50)
    switch_between_providers()

    print("\n" + "=" * 50)
    print("Example 3: Cost-Aware Routing")
    print("=" * 50)
    cost_aware_routing()

    print("\n" + "=" * 50)
    print("Example 4: Provider Fallback")
    print("=" * 50)
    provider_fallback()


if __name__ == "__main__":
    main()
