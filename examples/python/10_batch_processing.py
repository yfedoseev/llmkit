"""
Batch Processing Example

Demonstrates the batch API for processing multiple requests asynchronously.

Requirements:
- Set ANTHROPIC_API_KEY environment variable

Run:
    python 10_batch_processing.py
"""

import time
from llmkit import (
    LLMKitClient,
    Message,
    CompletionRequest,
    BatchRequest,
)


def create_batch():
    """Create and submit a batch job."""
    client = LLMKitClient.from_env()

    # Create multiple completion requests
    questions = [
        ("q1", "What is 2+2?"),
        ("q2", "What is the capital of France?"),
        ("q3", "What color is the sky?"),
        ("q4", "How many days in a week?"),
        ("q5", "What is H2O?"),
    ]

    # Build batch requests
    batch_requests = []
    for custom_id, question in questions:
        request = CompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message.user(question)],
            max_tokens=100,
        )
        batch_requests.append(BatchRequest(custom_id, request))

    print(f"Submitting batch with {len(batch_requests)} requests...")

    # Submit the batch
    batch_job = client.create_batch(batch_requests)

    print(f"Batch created:")
    print(f"  ID: {batch_job.id}")
    print(f"  Status: {batch_job.status}")
    print(f"  Total requests: {batch_job.request_counts.total}")

    return batch_job.id


def check_batch_status(batch_id: str):
    """Check the status of a batch job."""
    client = LLMKitClient.from_env()

    batch_job = client.get_batch("anthropic", batch_id)

    print(f"Batch {batch_id}:")
    print(f"  Status: {batch_job.status}")
    print(f"  Total: {batch_job.request_counts.total}")
    print(f"  Succeeded: {batch_job.request_counts.succeeded}")
    print(f"  Failed: {batch_job.request_counts.failed}")
    print(f"  Pending: {batch_job.request_counts.pending}")
    print(f"  Is complete: {batch_job.is_complete()}")

    if batch_job.created_at:
        print(f"  Created: {batch_job.created_at}")
    if batch_job.ended_at:
        print(f"  Ended: {batch_job.ended_at}")

    return batch_job


def get_batch_results(batch_id: str):
    """Get results from a completed batch."""
    client = LLMKitClient.from_env()

    results = client.get_batch_results("anthropic", batch_id)

    print(f"\nBatch results ({len(results)} items):")
    for result in results:
        print(f"\n  {result.custom_id}:")
        if result.is_success():
            response = result.response
            print(f"    Status: Success")
            print(f"    Response: {response.text_content()[:100]}...")
            if response.usage:
                print(f"    Tokens: {response.usage.total_tokens()}")
        else:
            error = result.error
            print(f"    Status: Error")
            print(f"    Error type: {error.error_type}")
            print(f"    Message: {error.message}")


def cancel_batch(batch_id: str):
    """Cancel a batch job."""
    client = LLMKitClient.from_env()

    batch_job = client.cancel_batch("anthropic", batch_id)
    print(f"Batch {batch_id} cancelled. Status: {batch_job.status}")


def list_batches():
    """List all batch jobs."""
    client = LLMKitClient.from_env()

    batches = client.list_batches("anthropic", limit=10)

    print(f"Found {len(batches)} batches:")
    for batch in batches:
        print(f"  {batch.id}: {batch.status}")
        print(f"    Requests: {batch.request_counts.total}")
        print(f"    Created: {batch.created_at}")


def wait_for_batch(batch_id: str, timeout: int = 300):
    """Wait for a batch to complete."""
    client = LLMKitClient.from_env()

    print(f"Waiting for batch {batch_id} to complete...")

    start = time.time()
    while True:
        batch_job = client.get_batch("anthropic", batch_id)

        if batch_job.is_complete():
            print(f"Batch completed with status: {batch_job.status}")
            return batch_job

        elapsed = time.time() - start
        if elapsed > timeout:
            print(f"Timeout waiting for batch")
            return batch_job

        print(
            f"  Status: {batch_job.status}, "
            f"Progress: {batch_job.request_counts.succeeded}/{batch_job.request_counts.total}"
        )
        time.sleep(5)  # Poll every 5 seconds


def full_batch_workflow():
    """Complete batch processing workflow."""
    client = LLMKitClient.from_env()

    # 1. Create batch requests
    print("Step 1: Creating batch requests...")
    batch_requests = [
        BatchRequest(
            "translate-1",
            CompletionRequest(
                model="claude-sonnet-4-20250514",
                messages=[Message.user("Translate 'Hello' to Spanish")],
                max_tokens=50,
            ),
        ),
        BatchRequest(
            "translate-2",
            CompletionRequest(
                model="claude-sonnet-4-20250514",
                messages=[Message.user("Translate 'Goodbye' to French")],
                max_tokens=50,
            ),
        ),
        BatchRequest(
            "translate-3",
            CompletionRequest(
                model="claude-sonnet-4-20250514",
                messages=[Message.user("Translate 'Thank you' to Japanese")],
                max_tokens=50,
            ),
        ),
    ]

    # 2. Submit batch
    print("\nStep 2: Submitting batch...")
    batch_job = client.create_batch(batch_requests)
    print(f"Batch ID: {batch_job.id}")

    # 3. Wait for completion
    print("\nStep 3: Waiting for completion...")
    final_job = wait_for_batch(batch_job.id, timeout=120)

    if not final_job.is_complete():
        print("Batch did not complete in time")
        return

    # 4. Get results
    print("\nStep 4: Getting results...")
    results = client.get_batch_results("anthropic", batch_job.id)

    translations = {}
    for result in results:
        if result.is_success():
            translations[result.custom_id] = result.response.text_content()

    print("\nTranslations:")
    for request_id, translation in translations.items():
        print(f"  {request_id}: {translation}")


def main():
    print("=" * 50)
    print("Example 1: List Existing Batches")
    print("=" * 50)
    list_batches()

    # Uncomment to run full workflow (requires API credits):
    # print("\n" + "=" * 50)
    # print("Example 2: Full Batch Workflow")
    # print("=" * 50)
    # full_batch_workflow()


if __name__ == "__main__":
    main()
