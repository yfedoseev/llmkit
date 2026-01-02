"""
Extended Thinking (Reasoning Mode) Example

Demonstrates how to enable extended thinking for complex reasoning tasks.

Requirements:
- Set ANTHROPIC_API_KEY environment variable

Run:
    python 06_extended_thinking.py
"""

from llmkit import LLMKitClient, Message, CompletionRequest


def basic_thinking():
    """Enable thinking with a token budget."""
    client = LLMKitClient.from_env()

    # Enable thinking with a budget of 5000 tokens for reasoning
    request = CompletionRequest(
        model="claude-sonnet-4-20250514",
        messages=[
            Message.user(
                "Solve this step by step: "
                "A train travels from City A to City B at 60 mph. "
                "Another train leaves City B towards City A at 40 mph at the same time. "
                "The cities are 200 miles apart. "
                "Where do they meet and after how long?"
            )
        ],
        max_tokens=2000,
    ).with_thinking(budget_tokens=5000)

    print("Solving with extended thinking enabled...")
    print("(This may take a moment)\n")

    response = client.complete(request)

    # Get the thinking/reasoning content
    thinking = response.thinking_content()
    if thinking:
        print("=" * 50)
        print("THINKING PROCESS:")
        print("=" * 50)
        print(thinking)
        print()

    # Get the final answer
    print("=" * 50)
    print("FINAL ANSWER:")
    print("=" * 50)
    print(response.text_content())

    # Usage info
    if response.usage:
        print(f"\nTokens used: {response.usage.total_tokens()}")


def complex_reasoning():
    """A more complex reasoning task."""
    client = LLMKitClient.from_env()

    request = CompletionRequest(
        model="claude-sonnet-4-20250514",
        messages=[
            Message.user(
                "Consider this logic puzzle:\n\n"
                "Three friends - Alice, Bob, and Carol - each have a different pet "
                "(cat, dog, fish) and live in different colored houses (red, blue, green).\n\n"
                "Clues:\n"
                "1. Alice doesn't live in the red house.\n"
                "2. The person with the cat lives in the blue house.\n"
                "3. Bob doesn't have a fish.\n"
                "4. Carol lives in the green house.\n"
                "5. The dog owner doesn't live in the green house.\n\n"
                "Who owns what pet and lives in which house?"
            )
        ],
        max_tokens=2000,
    ).with_thinking(budget_tokens=8000)

    print("Solving logic puzzle with extended thinking...\n")

    response = client.complete(request)

    thinking = response.thinking_content()
    if thinking:
        # Show first part of thinking
        print("THINKING (first 500 chars):")
        print("-" * 40)
        print(thinking[:500] + "..." if len(thinking) > 500 else thinking)
        print()

    print("SOLUTION:")
    print("-" * 40)
    print(response.text_content())


def streaming_with_thinking():
    """Stream responses with thinking enabled."""
    client = LLMKitClient.from_env()

    request = CompletionRequest(
        model="claude-sonnet-4-20250514",
        messages=[
            Message.user(
                "What's the best strategy for playing tic-tac-toe? "
                "Think through the possible moves."
            )
        ],
        max_tokens=1500,
    ).with_thinking(budget_tokens=3000).with_streaming()

    print("Streaming with thinking enabled...\n")

    thinking_content = []
    answer_content = []
    current_section = "thinking"

    for chunk in client.complete_stream(request):
        # Check for thinking content
        delta = chunk.delta
        if delta:
            if delta.is_thinking:
                thinking_text = delta.thinking
                if thinking_text:
                    thinking_content.append(thinking_text)
                    # Print thinking with indicator
                    print(f"[T] {thinking_text}", end="", flush=True)
            elif delta.is_text:
                text = delta.text
                if text:
                    if current_section == "thinking":
                        current_section = "answer"
                        print("\n\n[Answer]:")
                    answer_content.append(text)
                    print(text, end="", flush=True)

        if chunk.is_done:
            break

    print("\n")


def main():
    print("=" * 60)
    print("Example 1: Basic Math Problem with Thinking")
    print("=" * 60)
    basic_thinking()

    print("\n" + "=" * 60)
    print("Example 2: Complex Logic Puzzle")
    print("=" * 60)
    complex_reasoning()

    # Uncomment to test streaming with thinking:
    # print("\n" + "=" * 60)
    # print("Example 3: Streaming with Thinking")
    # print("=" * 60)
    # streaming_with_thinking()


if __name__ == "__main__":
    main()
