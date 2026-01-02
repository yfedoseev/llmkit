"""
Simple Completion Example

Demonstrates the basic usage of LLMKit for making completion requests.

Requirements:
- Set ANTHROPIC_API_KEY environment variable (or another provider's key)

Run:
    python 01_simple_completion.py
"""

from llmkit import LLMKitClient, Message, CompletionRequest


def main():
    # Create client from environment variables
    # Automatically detects configured providers
    client = LLMKitClient.from_env()

    # List available providers
    providers = client.providers()
    print(f"Available providers: {providers}")

    # Create a simple completion request
    request = CompletionRequest(
        model="claude-sonnet-4-20250514",
        messages=[
            Message.user("What is the capital of France? Reply in one word.")
        ],
        max_tokens=100,
    )

    # Make the request
    print("\nSending request...")
    response = client.complete(request)

    # Print the response
    print(f"\nResponse: {response.text_content()}")
    print(f"Model: {response.model}")
    print(f"Stop reason: {response.stop_reason}")

    # Print token usage
    if response.usage:
        print(f"\nToken usage:")
        print(f"  Input tokens: {response.usage.input_tokens}")
        print(f"  Output tokens: {response.usage.output_tokens}")
        print(f"  Total tokens: {response.usage.total_tokens()}")


if __name__ == "__main__":
    main()
