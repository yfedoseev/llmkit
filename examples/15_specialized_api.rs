//! Specialized APIs Example
//!
//! Demonstrates ranking, reranking, moderation, and classification APIs.
//!
//! Requirements:
//! - Set COHERE_API_KEY environment variable (for ranking/reranking/classification)
//! - Set OPENAI_API_KEY environment variable (for moderation)
//!
//! Run:
//!     cargo run --example 15_specialized_api

use llmkit::{ClassificationRequest, LLMKitClient, ModerationRequest, RankingRequest};

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    let client = LLMKitClient::builder()
        .with_cohere_from_env()
        .with_openai_from_env()
        .build()
        .await?;

    println!("=== Specialized APIs Example ===\n");

    // Example 1: Document Ranking
    println!("--- Example 1: Document Ranking ---");
    let documents = vec![
        "Python is a general-purpose programming language",
        "Java is used for enterprise software",
        "Python was created by Guido van Rossum",
        "JavaScript is used for web development",
    ];

    let request = RankingRequest::new(
        "cohere/rerank-english-v3.0",
        "Python programming",
        documents.clone(),
    );
    let response = client.rank(request).await?;

    println!("Query: 'Python programming'");
    println!("Ranked results:");
    for result in &response.results {
        println!("  [{:.3}] {}", result.score, documents[result.index]);
    }

    // Example 2: Search Result Reranking
    println!("\n--- Example 2: Search Result Reranking ---");
    let search_results = vec![
        "Random document about cats",
        "Python best practices and coding standards",
        "Weather forecast for today",
        "Advanced Python programming techniques",
        "Python beginner tutorial",
    ];

    let request = RankingRequest::new(
        "cohere/rerank-english-v3.0",
        "Python best practices",
        search_results.clone(),
    )
    .with_top_k(3);
    let response = client.rank(request).await?;

    println!("Query: 'Python best practices'");
    println!("Top 3 reranked results:");
    for result in &response.results {
        println!("  [{:.3}] {}", result.score, search_results[result.index]);
    }

    // Example 3: Content Moderation
    println!("\n--- Example 3: Content Moderation ---");
    let texts_to_moderate = [
        "This is a great product! I highly recommend it.",
        "I really enjoyed the experience, thank you!",
        "The service was acceptable, nothing special.",
    ];

    for text in texts_to_moderate {
        let request = ModerationRequest::new("openai/omni-moderation-latest", text);
        let response = client.moderate(request).await?;

        let status = if response.flagged { "FLAGGED" } else { "OK" };
        println!("  [{}] {}", status, text);
    }

    // Example 4: Text Classification
    println!("\n--- Example 4: Text Classification ---");
    let labels = vec!["positive", "negative", "neutral"];
    let reviews = [
        "Amazing product, highly recommend!",
        "Terrible quality, very disappointed",
        "It's okay, nothing special",
        "Best purchase I ever made!",
        "Waste of money, don't buy",
    ];

    for review in reviews {
        let request =
            ClassificationRequest::new("cohere/embed-english-v3.0", review, labels.clone());
        let response = client.classify(request).await?;

        if let Some(prediction) = response.predictions.first() {
            println!(
                "  {:10} | {} ({:.0}%)",
                prediction.label,
                review,
                prediction.score * 100.0
            );
        }
    }

    // Example 5: Complete Content Processing Workflow
    println!("\n--- Example 5: Complete Workflow ---");
    println!("Processing user comments through moderation and classification...\n");

    let user_comments = [
        "This is fantastic! Best purchase ever!",
        "Very disappointed with the quality",
        "It works as described, nothing more",
        "Absolutely love it, exceeded expectations",
    ];

    for comment in user_comments {
        // Step 1: Check moderation
        let mod_request = ModerationRequest::new("openai/omni-moderation-latest", comment);
        let mod_response = client.moderate(mod_request).await?;

        if mod_response.flagged {
            println!("  REJECTED: {} (violates policy)", comment);
            continue;
        }

        // Step 2: Classify sentiment
        let class_request = ClassificationRequest::new(
            "cohere/embed-english-v3.0",
            comment,
            vec!["positive", "negative", "neutral"],
        );
        let class_response = client.classify(class_request).await?;

        if let Some(prediction) = class_response.predictions.first() {
            println!(
                "  {:10} | {} ({:.0}%)",
                prediction.label.to_uppercase(),
                comment,
                prediction.score * 100.0
            );
        }
    }

    println!("\nSpecialized API examples completed!");

    Ok(())
}
