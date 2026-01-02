"""
Streaming Example

Demonstrates real-time streaming of completion responses.

Requirements:
- Set OPENAI_API_KEY environment variable

Run:
    python 02_streaming.py
"""

from llmkit import LLMKitClient, Message, CompletionRequest


def main():
    client = LLMKitClient.from_env()

    # Create a request with streaming enabled
    # Use "provider/model" format for explicit provider routing
    request = CompletionRequest(
        model="openai/gpt-4o",
        messages=[
            Message.user("Write a short poem about programming. 4 lines maximum.")
        ],
        max_tokens=200,
    ).with_streaming()

    print("Streaming response:\n")

    # Stream the response
    for chunk in client.complete_stream(request):
        # Print text chunks as they arrive
        if chunk.text:
            print(chunk.text, end="", flush=True)

        # Check if streaming is complete
        if chunk.is_done:
            print("\n\n[Stream complete]")

            # Print final usage if available
            if chunk.usage:
                print(f"Total tokens: {chunk.usage.total_tokens()}")
            break

    print()


def stream_with_events():
    """Alternative: inspect all stream events."""
    client = LLMKitClient.from_env()

    request = CompletionRequest(
        model="openai/gpt-4o",
        messages=[Message.user("Say hello in 3 languages")],
        max_tokens=100,
    ).with_streaming()

    print("\nStreaming with event inspection:\n")

    for chunk in client.complete_stream(request):
        # Inspect event type
        event_type = chunk.event_type
        print(f"[Event: {event_type}]", end=" ")

        if chunk.text:
            print(f"Text: {chunk.text!r}")
        elif chunk.is_done:
            print(f"Stop reason: {chunk.stop_reason}")
        else:
            print()


if __name__ == "__main__":
    main()
    # Uncomment to see event-level streaming:
    # stream_with_events()
