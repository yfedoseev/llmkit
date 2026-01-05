"""
Async Usage Example

Demonstrates asynchronous operations with ModelSuite.

Requirements:
- Set GROQ_API_KEY environment variable (Groq has fast inference)

Run:
    python 09_async_usage.py
"""

import asyncio
from modelsuite import AsyncModelSuiteClient, Message, CompletionRequest


async def basic_async_completion():
    """Basic async completion request."""
    client = AsyncModelSuiteClient.from_env()

    # Use "provider/model" format for explicit provider routing
    request = CompletionRequest(
        model="groq/llama-3.3-70b-versatile",
        messages=[Message.user("What is the capital of Japan?")],
        max_tokens=100,
    )

    print("Making async request...")
    response = await client.complete(request)
    print(f"Response: {response.text_content()}")

    return response


async def async_streaming():
    """Async streaming responses."""
    client = AsyncModelSuiteClient.from_env()

    request = CompletionRequest(
        model="groq/llama-3.3-70b-versatile",
        messages=[Message.user("Count from 1 to 5 slowly")],
        max_tokens=200,
    ).with_streaming()

    print("Streaming async response:")
    async for chunk in await client.complete_stream(request):
        if chunk.text:
            print(chunk.text, end="", flush=True)
        if chunk.is_done:
            break
    print()


async def concurrent_requests():
    """Make multiple requests concurrently."""
    client = AsyncModelSuiteClient.from_env()

    # Define multiple requests
    questions = [
        "What is Python?",
        "What is Rust?",
        "What is JavaScript?",
    ]

    requests = [
        CompletionRequest(
            model="groq/llama-3.3-70b-versatile",
            messages=[Message.user(q)],
            max_tokens=100,
        )
        for q in questions
    ]

    print("Making concurrent requests...")

    # Run all requests concurrently
    responses = await asyncio.gather(
        *[client.complete(req) for req in requests],
        return_exceptions=True,
    )

    for question, response in zip(questions, responses):
        if isinstance(response, Exception):
            print(f"\nQ: {question}")
            print(f"A: Error - {response}")
        else:
            print(f"\nQ: {question}")
            print(f"A: {response.text_content()[:100]}...")


async def rate_limited_batch():
    """Process requests with rate limiting."""
    client = AsyncModelSuiteClient.from_env()

    questions = [
        "What is 1+1?",
        "What is 2+2?",
        "What is 3+3?",
        "What is 4+4?",
        "What is 5+5?",
    ]

    # Process with semaphore to limit concurrency
    semaphore = asyncio.Semaphore(2)  # Max 2 concurrent requests

    async def limited_request(question):
        async with semaphore:
            print(f"Processing: {question}")
            request = CompletionRequest(
                model="groq/llama-3.3-70b-versatile",
                messages=[Message.user(question)],
                max_tokens=20,
            )
            response = await client.complete(request)
            return question, response.text_content().strip()

    print("Processing with rate limiting (max 2 concurrent)...")
    results = await asyncio.gather(*[limited_request(q) for q in questions])

    print("\nResults:")
    for question, answer in results:
        print(f"  {question} -> {answer}")


async def async_with_timeout():
    """Handle timeouts in async requests."""
    client = AsyncModelSuiteClient.from_env()

    request = CompletionRequest(
        model="groq/llama-3.3-70b-versatile",
        messages=[Message.user("Write a very long story")],
        max_tokens=2000,
    )

    try:
        # Set a timeout
        response = await asyncio.wait_for(
            client.complete(request),
            timeout=30.0,  # 30 second timeout
        )
        print(f"Got response: {response.text_content()[:100]}...")
    except asyncio.TimeoutError:
        print("Request timed out!")


async def async_error_handling():
    """Error handling in async context."""
    from modelsuite import (
        ModelSuiteError,
        AuthenticationError,
        RateLimitError,
    )

    client = AsyncModelSuiteClient.from_env()

    request = CompletionRequest(
        model="groq/llama-3.3-70b-versatile",
        messages=[Message.user("Hello!")],
        max_tokens=100,
    )

    try:
        response = await client.complete(request)
        print(f"Success: {response.text_content()}")
    except AuthenticationError:
        print("Auth error - check API key")
    except RateLimitError as e:
        print(f"Rate limited: {e}")
    except ModelSuiteError as e:
        print(f"ModelSuite error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


async def main():
    print("=" * 50)
    print("Example 1: Basic Async Completion")
    print("=" * 50)
    await basic_async_completion()

    print("\n" + "=" * 50)
    print("Example 2: Async Streaming")
    print("=" * 50)
    await async_streaming()

    print("\n" + "=" * 50)
    print("Example 3: Concurrent Requests")
    print("=" * 50)
    await concurrent_requests()

    print("\n" + "=" * 50)
    print("Example 4: Rate-Limited Batch")
    print("=" * 50)
    await rate_limited_batch()

    print("\n" + "=" * 50)
    print("Example 5: Async Error Handling")
    print("=" * 50)
    await async_error_handling()


if __name__ == "__main__":
    asyncio.run(main())
