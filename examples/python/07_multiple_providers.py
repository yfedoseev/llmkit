"""
Multiple Providers Example

Demonstrates how to configure and use multiple LLM providers.

Requirements:
- Set multiple provider API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)

Run:
    python 07_multiple_providers.py
"""

from modelsuite import LLMKitClient, Message, CompletionRequest


def using_from_env():
    """Auto-detect providers from environment variables."""
    client = LLMKitClient.from_env()

    # List all detected providers
    providers = client.providers()
    print(f"Detected providers: {providers}")
    print(f"Default provider: {client.default_provider}")

    # Use the default provider with explicit provider/model format
    response = client.complete(
        CompletionRequest(
            model="anthropic/claude-sonnet-4-20250514",
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

    # Using the unified "provider/model" format - much cleaner!
    models = [
        "anthropic/claude-sonnet-4-20250514",
        "openai/gpt-4o",
    ]

    for model in models:
        provider = model.split("/")[0]
        if provider not in providers:
            print(f"{model}: Not configured")
            continue

        try:
            # Use unified format - provider is embedded in model string
            response = client.complete(
                CompletionRequest(
                    model=model,
                    messages=[Message.user(prompt)],
                    max_tokens=20,
                ),
            )
            print(f"{model}: {response.text_content().strip()}")
        except Exception as e:
            print(f"{model}: Error - {e}")


def cost_aware_routing():
    """Route requests to cheaper providers when appropriate."""
    from modelsuite import get_model_info, get_cheapest_model

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

    # Order providers by preference using unified "provider/model" format
    model_priority = [
        "anthropic/claude-sonnet-4-20250514",
        "openai/gpt-4o",
    ]

    available = set(client.providers())

    for model in model_priority:
        provider = model.split("/")[0]
        if provider not in available:
            print(f"Skipping {model} (not configured)")
            continue

        try:
            print(f"Trying {model}...")
            response = client.complete(
                CompletionRequest(
                    model=model,
                    messages=[Message.user("What is Python?")],
                    max_tokens=100,
                ),
            )
            print(f"Success with {model}!")
            print(f"Response: {response.text_content()[:100]}...")
            return
        except Exception as e:
            print(f"Failed with {model}: {e}")
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
