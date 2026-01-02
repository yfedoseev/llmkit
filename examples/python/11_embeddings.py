"""
Embeddings Example

Demonstrates text embedding generation and similarity computation.

Requirements:
- Set OPENAI_API_KEY environment variable (for text-embedding-3-small)
  Or COHERE_API_KEY for Cohere embeddings

Run:
    python 11_embeddings.py
"""

from llmkit import LLMKitClient, EmbeddingRequest


def basic_embedding():
    """Generate embedding for a single text."""
    client = LLMKitClient.from_env()

    # Create embedding request
    request = EmbeddingRequest("text-embedding-3-small", "Hello, world!")

    print("Generating embedding...")
    response = client.embed(request)

    print(f"Model: {response.model}")
    print(f"Dimensions: {response.dimensions}")

    # Get the embedding values
    values = response.values()
    if values:
        print(f"First 5 values: {values[:5]}")
        print(f"Total values: {len(values)}")

    # Usage info
    print(f"Tokens used: {response.usage.total_tokens}")


def batch_embeddings():
    """Generate embeddings for multiple texts at once."""
    client = LLMKitClient.from_env()

    texts = [
        "The quick brown fox",
        "A lazy dog sleeps",
        "Python is a programming language",
        "Machine learning is fascinating",
    ]

    # Create batch request
    request = EmbeddingRequest.batch("text-embedding-3-small", texts)

    print(f"Generating embeddings for {len(texts)} texts...")
    response = client.embed(request)

    print(f"Got {len(response.embeddings)} embeddings")
    print(f"Dimensions: {response.dimensions}")

    for emb in response.embeddings:
        print(f"  Text {emb.index}: {emb.dimension_count} dimensions")


def compute_similarity():
    """Compute similarity between texts."""
    client = LLMKitClient.from_env()

    texts = [
        "I love programming in Python",
        "Python coding is my favorite",
        "The weather is nice today",
        "I enjoy writing code",
    ]

    print("Computing similarities...")

    # Generate embeddings
    request = EmbeddingRequest.batch("text-embedding-3-small", texts)
    response = client.embed(request)

    embeddings = response.embeddings

    # Compare first text to all others
    reference = embeddings[0]
    print(f"\nReference: '{texts[0]}'")
    print("\nSimilarities:")

    for i, emb in enumerate(embeddings[1:], start=1):
        similarity = reference.cosine_similarity(emb)
        print(f"  vs '{texts[i]}': {similarity:.4f}")


def semantic_search():
    """Simple semantic search example."""
    client = LLMKitClient.from_env()

    # Document corpus
    documents = [
        "Python is a high-level programming language",
        "Machine learning uses algorithms to learn from data",
        "The Eiffel Tower is located in Paris, France",
        "Deep learning is a subset of machine learning",
        "JavaScript is commonly used for web development",
        "Natural language processing deals with text and speech",
    ]

    # Search query
    query = "What is artificial intelligence?"

    print(f"Query: '{query}'")
    print("\nSearching documents...")

    # Embed query
    query_response = client.embed(
        EmbeddingRequest("text-embedding-3-small", query)
    )
    query_embedding = query_response.embeddings[0]

    # Embed documents
    doc_response = client.embed(
        EmbeddingRequest.batch("text-embedding-3-small", documents)
    )

    # Compute similarities and rank
    results = []
    for i, doc_emb in enumerate(doc_response.embeddings):
        similarity = query_embedding.cosine_similarity(doc_emb)
        results.append((similarity, documents[i]))

    # Sort by similarity (highest first)
    results.sort(reverse=True)

    print("\nResults (by relevance):")
    for similarity, doc in results:
        print(f"  [{similarity:.4f}] {doc}")


def different_embedding_models():
    """Use different embedding models/dimensions."""
    client = LLMKitClient.from_env()

    text = "Hello, world!"

    # OpenAI small model
    print("OpenAI text-embedding-3-small:")
    try:
        response = client.embed(
            EmbeddingRequest("text-embedding-3-small", text)
        )
        print(f"  Dimensions: {response.dimensions}")
    except Exception as e:
        print(f"  Error: {e}")

    # OpenAI small model with reduced dimensions
    print("\nOpenAI text-embedding-3-small (256 dims):")
    try:
        response = client.embed(
            EmbeddingRequest("text-embedding-3-small", text)
            .with_dimensions(256)
        )
        print(f"  Dimensions: {response.dimensions}")
    except Exception as e:
        print(f"  Error: {e}")

    # OpenAI large model
    print("\nOpenAI text-embedding-3-large:")
    try:
        response = client.embed(
            EmbeddingRequest("text-embedding-3-large", text)
        )
        print(f"  Dimensions: {response.dimensions}")
    except Exception as e:
        print(f"  Error: {e}")


def embedding_with_input_type():
    """Use input type hints for optimized embeddings."""
    from llmkit import EmbeddingInputType

    client = LLMKitClient.from_env()

    # For search queries
    query_request = (
        EmbeddingRequest("text-embedding-3-small", "What is Python?")
        .with_input_type(EmbeddingInputType.Query)
    )

    # For documents to be indexed
    doc_request = (
        EmbeddingRequest("text-embedding-3-small", "Python is a programming language.")
        .with_input_type(EmbeddingInputType.Document)
    )

    print("Query embedding:")
    query_response = client.embed(query_request)
    print(f"  Dimensions: {query_response.dimensions}")

    print("\nDocument embedding:")
    doc_response = client.embed(doc_request)
    print(f"  Dimensions: {doc_response.dimensions}")


def distance_metrics():
    """Compare different distance metrics."""
    client = LLMKitClient.from_env()

    texts = ["Hello world", "Hello there"]

    response = client.embed(
        EmbeddingRequest.batch("text-embedding-3-small", texts)
    )

    emb1 = response.embeddings[0]
    emb2 = response.embeddings[1]

    print(f"Text 1: '{texts[0]}'")
    print(f"Text 2: '{texts[1]}'")
    print()
    print("Distance metrics:")
    print(f"  Cosine similarity: {emb1.cosine_similarity(emb2):.6f}")
    print(f"  Dot product: {emb1.dot_product(emb2):.6f}")
    print(f"  Euclidean distance: {emb1.euclidean_distance(emb2):.6f}")


def main():
    print("=" * 50)
    print("Example 1: Basic Embedding")
    print("=" * 50)
    basic_embedding()

    print("\n" + "=" * 50)
    print("Example 2: Batch Embeddings")
    print("=" * 50)
    batch_embeddings()

    print("\n" + "=" * 50)
    print("Example 3: Similarity Computation")
    print("=" * 50)
    compute_similarity()

    print("\n" + "=" * 50)
    print("Example 4: Semantic Search")
    print("=" * 50)
    semantic_search()

    print("\n" + "=" * 50)
    print("Example 5: Distance Metrics")
    print("=" * 50)
    distance_metrics()


if __name__ == "__main__":
    main()
