"""
OpenAI-Compatible Provider Example

Demonstrates using OpenRouter through the generic OpenAI-compatible
provider interface. This shows that any OpenAI-compatible API can be
used with LLMKit.

Note: The dedicated OpenRouterProvider is recommended for Qwen3 models
because it automatically handles the `/no_think` injection. When using
the OpenAI-compatible endpoint, you need to manually add `/no_think`
to prompts or use higher max_tokens.

Requirements:
- Set OPENROUTER_API_KEY environment variable
- uv sync

Run:
    uv run python 19_openai_compatible.py
"""

import os
from llmkit import LLMKitClient, Message, CompletionRequest


def main():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable not set")

    # Create client with OpenAI-compatible provider pointing to OpenRouter
    # Note: Provider name (model_id) cannot contain dashes due to model format parsing
    client = LLMKitClient(
        providers={
            "openai_compatible": {
                "api_key": api_key,
                "base_url": "https://openrouter.ai/api/v1",
                "model_id": "openroutercompat",  # Provider name for routing
            }
        }
    )

    # Model format is "provider/model" - everything after first "/" is the model name
    model = "openroutercompat/qwen/qwen3-32b"

    print("OpenAI-Compatible Provider Example")
    print("=" * 50)
    print("Using OpenRouter via OpenAI-compatible endpoint\n")

    # Since we're not using OpenRouterProvider, we need to manually
    # add /no_think to disable Qwen3's reasoning mode
    print("Request with manual /no_think:")
    print("-" * 40)

    request = CompletionRequest(
        model=model,
        messages=[Message.user("What is 2 + 2? Answer briefly. /no_think")],
        system="You are a helpful assistant.",
        max_tokens=100,
    )

    response = client.complete(request)
    print(f"Response: {response.text_content().strip()}\n")

    # Second request
    print("Another request:")
    print("-" * 40)

    request = CompletionRequest(
        model=model,
        messages=[Message.user("Name 3 colors. /no_think")],
        system="Be concise.",
        max_tokens=100,
    )

    response = client.complete(request)
    print(f"Response: {response.text_content().strip()}\n")

    print("=" * 50)
    print("OpenRouter works via OpenAI-compatible endpoint!")
    print("Note: Manual /no_think required for Qwen3 models.")


if __name__ == "__main__":
    main()
