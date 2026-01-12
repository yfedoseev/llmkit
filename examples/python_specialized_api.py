#!/usr/bin/env python3
"""
Example: Specialized APIs with LLMKit Python bindings

Demonstrates: Ranking, Reranking, Moderation, Classification
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llmkit-python'))

from llmkit import (
    LLMKitClient,
    RankingRequest,
    RerankingRequest,
    ModerationRequest,
    ClassificationRequest,
)


def main():
    print("🔧 LLMKit Specialized APIs Example")
    print("=" * 50)

    client = LLMKitClient.from_env()

    # 1. RANKING
    print("\n--- Example 1: Document Ranking ---")
    documents = [
        "Python is a general-purpose programming language",
        "Java is used for enterprise software",
        "Python was created by Guido van Rossum",
    ]
    rank_req = RankingRequest("Python programming", documents)
    rank_resp = client.rank_documents(rank_req)
    print(f"Query: {rank_req.query}")
    print(f"Top result: {rank_resp.first().document} (score: {rank_resp.first().score:.3f})")

    # 2. RERANKING
    print("\n--- Example 2: Search Result Reranking ---")
    search_results = [
        "Random document A",
        "Python best practices guide",
        "Unrelated content",
    ]
    rerank_req = RerankingRequest("Python best practices", search_results).with_top_n(2)
    rerank_resp = client.rerank_results(rerank_req)
    print(f"Query: {rerank_req.query}")
    for result in rerank_resp.results:
        print(f"  {result.document}: {result.relevance_score:.3f}")

    # 3. MODERATION
    print("\n--- Example 3: Content Moderation ---")
    texts_to_moderate = [
        "This is a great product!",
        "I really enjoyed the experience",
        "This service is acceptable",
    ]
    for text in texts_to_moderate:
        mod_req = ModerationRequest(text)
        mod_resp = client.moderate_text(mod_req)
        status = "✓ OK" if not mod_resp.flagged else "✗ FLAGGED"
        print(f"{status}: {text[:40]}...")

    # 4. CLASSIFICATION
    print("\n--- Example 4: Text Classification ---")
    labels = ["positive", "negative", "neutral"]
    reviews = [
        "Amazing product, highly recommend!",
        "Terrible quality, very disappointed",
        "It's okay, nothing special",
    ]
    for review in reviews:
        class_req = ClassificationRequest(review, labels)
        class_resp = client.classify_text(class_req)
        top = class_resp.top()
        print(f"{top.label:10} | {review} ({top.confidence:.2%})")

    # 5. COMPLETE WORKFLOW
    print("\n--- Example 5: Complete Workflow ---")
    print("Processing user comments...")

    user_comments = [
        "This is fantastic! Best purchase ever!",
        "Very disappointed with quality",
        "It works as described",
    ]

    for comment in user_comments:
        # Check moderation
        mod_req = ModerationRequest(comment)
        mod_resp = client.moderate_text(mod_req)

        if mod_resp.flagged:
            print(f"⚠️  REJECTED: {comment[:40]}... (violates policy)")
            continue

        # Classify sentiment
        class_req = ClassificationRequest(comment, ["positive", "negative", "neutral"])
        class_resp = client.classify_text(class_req)
        sentiment = class_resp.top()

        print(
            f"✓ {sentiment.label.upper():8} | {comment[:40]}... "
            f"({sentiment.confidence:.1%})"
        )

    print("\n" + "=" * 50)
    print("✓ Specialized API examples completed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
