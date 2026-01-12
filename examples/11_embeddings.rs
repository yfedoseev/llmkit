//! Embeddings Example
//!
//! Demonstrates text embedding generation and similarity computation.
//!
//! Requirements:
//! - Set OPENAI_API_KEY environment variable (for text-embedding-3-small)
//!
//! Run:
//!     cargo run --example 11_embeddings

use llmkit::{EmbeddingRequest, LLMKitClient};

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_openai_from_env()
        .build()
        .await?;

    // Example 1: Basic embedding
    println!("=== Example 1: Basic Embedding ===");
    let request = EmbeddingRequest::new("openai/text-embedding-3-small", "Hello, world!");
    let response = client.embed(request).await?;

    println!("Model: {}", response.model);
    println!("Dimensions: {}", response.dimensions());
    if let Some(values) = response.values() {
        println!("First 5 values: {:?}", &values[..5.min(values.len())]);
    }
    println!("Tokens used: {}", response.usage.total_tokens);

    // Example 2: Batch embeddings
    println!("\n=== Example 2: Batch Embeddings ===");
    let texts = vec![
        "The quick brown fox",
        "A lazy dog sleeps",
        "Python is a programming language",
        "Machine learning is fascinating",
    ];

    let request = EmbeddingRequest::batch("openai/text-embedding-3-small", texts.clone());
    let response = client.embed(request).await?;

    println!("Got {} embeddings", response.embeddings.len());
    println!("Dimensions: {}", response.dimensions());
    for emb in &response.embeddings {
        println!("  Text {}: {} dimensions", emb.index, emb.values.len());
    }

    // Example 3: Compute similarity
    println!("\n=== Example 3: Similarity Computation ===");
    let similarity_texts = vec![
        "I love programming in Python",
        "Python coding is my favorite",
        "The weather is nice today",
        "I enjoy writing code",
    ];

    let request =
        EmbeddingRequest::batch("openai/text-embedding-3-small", similarity_texts.clone());
    let response = client.embed(request).await?;

    let reference = &response.embeddings[0];
    println!("Reference: '{}'", similarity_texts[0]);
    println!("\nSimilarities:");

    for (i, emb) in response.embeddings.iter().skip(1).enumerate() {
        let similarity = reference.cosine_similarity(emb);
        println!("  vs '{}': {:.4}", similarity_texts[i + 1], similarity);
    }

    // Example 4: Semantic search
    println!("\n=== Example 4: Semantic Search ===");
    let documents = vec![
        "Python is a high-level programming language",
        "Machine learning uses algorithms to learn from data",
        "The Eiffel Tower is located in Paris, France",
        "Deep learning is a subset of machine learning",
        "JavaScript is commonly used for web development",
        "Natural language processing deals with text and speech",
    ];

    let query = "What is artificial intelligence?";
    println!("Query: '{}'", query);

    // Embed query
    let query_response = client
        .embed(EmbeddingRequest::new(
            "openai/text-embedding-3-small",
            query,
        ))
        .await?;
    let query_embedding = &query_response.embeddings[0];

    // Embed documents
    let doc_response = client
        .embed(EmbeddingRequest::batch(
            "openai/text-embedding-3-small",
            documents.clone(),
        ))
        .await?;

    // Compute similarities and rank
    let mut results: Vec<(f32, &str)> = doc_response
        .embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (query_embedding.cosine_similarity(emb), documents[i]))
        .collect();

    results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

    println!("\nResults (by relevance):");
    for (similarity, doc) in results {
        println!("  [{:.4}] {}", similarity, doc);
    }

    // Example 5: Distance metrics
    println!("\n=== Example 5: Distance Metrics ===");
    let metric_texts = vec!["Hello world", "Hello there"];

    let response = client
        .embed(EmbeddingRequest::batch(
            "openai/text-embedding-3-small",
            metric_texts.clone(),
        ))
        .await?;

    let emb1 = &response.embeddings[0];
    let emb2 = &response.embeddings[1];

    println!("Text 1: '{}'", metric_texts[0]);
    println!("Text 2: '{}'", metric_texts[1]);
    println!();
    println!("Distance metrics:");
    println!("  Cosine similarity: {:.6}", emb1.cosine_similarity(emb2));
    println!("  Dot product: {:.6}", emb1.dot_product(emb2));
    println!("  Euclidean distance: {:.6}", emb1.euclidean_distance(emb2));

    Ok(())
}
