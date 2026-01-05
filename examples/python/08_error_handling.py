"""
Error Handling Example

Demonstrates how to handle various error conditions with ModelSuite.

Requirements:
- Set MISTRAL_API_KEY environment variable

Run:
    python 08_error_handling.py
"""

from modelsuite import (
    ModelSuiteClient,
    Message,
    CompletionRequest,
    # Error types
    ModelSuiteError,
    ProviderNotFoundError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ContextLengthError,
    TimeoutError,
)


def basic_error_handling():
    """Basic try/except pattern for ModelSuite errors."""
    client = ModelSuiteClient.from_env()

    # Use "provider/model" format for explicit provider routing
    request = CompletionRequest(
        model="mistral/mistral-large-latest",
        messages=[Message.user("Hello!")],
        max_tokens=100,
    )

    try:
        response = client.complete(request)
        print(f"Success: {response.text_content()}")
    except ModelSuiteError as e:
        print(f"ModelSuite error: {e}")


def handle_specific_errors():
    """Handle specific error types differently."""
    client = ModelSuiteClient.from_env()

    request = CompletionRequest(
        model="mistral/mistral-large-latest",
        messages=[Message.user("Hello!")],
        max_tokens=100,
    )

    try:
        response = client.complete(request)
        print(f"Success: {response.text_content()}")

    except AuthenticationError as e:
        # Invalid or expired API key
        print(f"Authentication failed: {e}")
        print("Please check your API key and try again.")

    except RateLimitError as e:
        # Too many requests
        print(f"Rate limited: {e}")
        # Check if retry_after is available
        if hasattr(e, "retry_after_seconds") and e.retry_after_seconds:
            print(f"Retry after {e.retry_after_seconds} seconds")

    except ContextLengthError as e:
        # Input too long for model
        print(f"Context too long: {e}")
        print("Try reducing your input or using a model with larger context.")

    except InvalidRequestError as e:
        # Bad request parameters
        print(f"Invalid request: {e}")
        print("Check your request parameters.")

    except ProviderNotFoundError as e:
        # Provider not configured
        print(f"Provider not found: {e}")
        print("Make sure the provider is configured with an API key.")

    except TimeoutError as e:
        # Request timed out
        print(f"Request timed out: {e}")
        print("The request took too long. Try again or reduce complexity.")

    except ModelSuiteError as e:
        # Catch-all for other ModelSuite errors
        print(f"ModelSuite error: {e}")


def demonstrate_provider_not_found():
    """Show what happens with unconfigured provider."""
    client = ModelSuiteClient.from_env()

    try:
        # Try to use a provider that's not configured
        response = client.complete_with_provider(
            "definitely_not_a_real_provider",
            CompletionRequest(
                model="some-model",
                messages=[Message.user("Hello!")],
                max_tokens=100,
            ),
        )
    except ProviderNotFoundError as e:
        print(f"Expected error: {e}")
        print(f"Available providers: {client.providers()}")


def demonstrate_invalid_request():
    """Show what happens with invalid parameters."""
    client = ModelSuiteClient.from_env()

    try:
        # Invalid max_tokens (negative)
        response = client.complete(
            CompletionRequest(
                model="mistral/mistral-large-latest",
                messages=[Message.user("Hello!")],
                max_tokens=-1,  # Invalid!
            )
        )
    except (InvalidRequestError, ModelSuiteError) as e:
        print(f"Invalid request error: {e}")


def retry_on_rate_limit():
    """Implement retry logic for rate limits."""
    import time

    client = ModelSuiteClient.from_env()

    request = CompletionRequest(
        model="mistral/mistral-large-latest",
        messages=[Message.user("Hello!")],
        max_tokens=100,
    )

    max_retries = 3
    retry_delay = 1  # Start with 1 second

    for attempt in range(max_retries):
        try:
            response = client.complete(request)
            print(f"Success on attempt {attempt + 1}")
            print(f"Response: {response.text_content()}")
            return response

        except RateLimitError as e:
            if attempt < max_retries - 1:
                wait_time = getattr(e, "retry_after_seconds", retry_delay)
                print(f"Rate limited. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                retry_delay *= 2  # Exponential backoff
            else:
                print("Max retries exceeded")
                raise

        except ModelSuiteError as e:
            print(f"Error: {e}")
            raise


def safe_complete(client, request, default_response="Unable to generate response"):
    """Wrapper function that never raises exceptions."""
    try:
        response = client.complete(request)
        return response.text_content()
    except AuthenticationError:
        return "Error: Invalid API key"
    except RateLimitError:
        return "Error: Too many requests, please try again later"
    except ContextLengthError:
        return "Error: Input too long"
    except InvalidRequestError as e:
        return f"Error: Invalid request - {e}"
    except ProviderNotFoundError:
        return "Error: Provider not available"
    except TimeoutError:
        return "Error: Request timed out"
    except ModelSuiteError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"


def main():
    print("=" * 50)
    print("Example 1: Basic Error Handling")
    print("=" * 50)
    basic_error_handling()

    print("\n" + "=" * 50)
    print("Example 2: Specific Error Types")
    print("=" * 50)
    handle_specific_errors()

    print("\n" + "=" * 50)
    print("Example 3: Provider Not Found")
    print("=" * 50)
    demonstrate_provider_not_found()

    print("\n" + "=" * 50)
    print("Example 4: Safe Completion Wrapper")
    print("=" * 50)
    client = ModelSuiteClient.from_env()
    result = safe_complete(
        client,
        CompletionRequest(
            model="mistral/mistral-large-latest",
            messages=[Message.user("Say hello briefly")],
            max_tokens=50,
        ),
    )
    print(f"Safe result: {result}")


if __name__ == "__main__":
    main()
