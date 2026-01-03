//! Integration tests for LLMKit
//!
//! These tests make actual API calls and require valid API keys.
//! Tests are automatically skipped if the required API key is not set.
//!
//! Set the following environment variables to enable tests:
//! - ANTHROPIC_API_KEY: Anthropic tests
//! - OPENAI_API_KEY: OpenAI tests
//! - GROQ_API_KEY: Groq tests
//! - MISTRAL_API_KEY: Mistral tests
//!
//! To run all integration tests:
//! ```bash
//! cargo test --features all-providers --test integration_tests
//! ```
//!
//! To run specific provider tests:
//! ```bash
//! ANTHROPIC_API_KEY=sk-... cargo test --features anthropic --test integration_tests anthropic
//! ```

use futures::StreamExt;
use llmkit::{
    tools::ToolBuilder,
    types::{
        CompletionRequest, ContentBlock, ContentDelta, Message, StopReason, StreamEventType,
        ThinkingConfig,
    },
    LLMKitClient,
};
use std::env;

/// Check if an environment variable is set and non-empty
fn has_env(key: &str) -> bool {
    env::var(key).map(|v| !v.is_empty()).unwrap_or(false)
}

/// Extract text from a stream chunk's delta
fn extract_text(delta: &Option<ContentDelta>) -> Option<String> {
    match delta {
        Some(ContentDelta::Text { text }) => Some(text.clone()),
        _ => None,
    }
}

/// Check if a stream event indicates the stream is done
fn is_stream_done(event_type: &StreamEventType) -> bool {
    matches!(event_type, StreamEventType::MessageStop)
}

// =============================================================================
// Anthropic Tests
// =============================================================================

#[tokio::test]
async fn test_anthropic_simple_completion() {
    if !has_env("ANTHROPIC_API_KEY") {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "anthropic/claude-sonnet-4-20250514",
        vec![Message::user("What is 2+2? Reply with just the number.")],
    )
    .with_max_tokens(50);

    let response = client.complete(request).await.expect("Request failed");

    assert!(!response.id.is_empty());
    assert!(response.model.to_lowercase().contains("claude"));
    assert!(response.text_content().contains('4'));
    assert!(response.stop_reason == StopReason::EndTurn);
    assert!(response.usage.input_tokens > 0);
    assert!(response.usage.output_tokens > 0);
}

#[tokio::test]
async fn test_anthropic_system_prompt() {
    if !has_env("ANTHROPIC_API_KEY") {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "anthropic/claude-sonnet-4-20250514",
        vec![Message::user("What are you?")],
    )
    .with_system("You are a friendly robot named R2D2. Always introduce yourself.")
    .with_max_tokens(100);

    let response = client.complete(request).await.expect("Request failed");
    let text = response.text_content().to_lowercase();

    assert!(text.contains("r2d2") || text.contains("robot"));
}

#[tokio::test]
async fn test_anthropic_tool_use() {
    if !has_env("ANTHROPIC_API_KEY") {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()
        .expect("Failed to build client");

    let tool = ToolBuilder::new("get_weather")
        .description("Get the current weather in a city")
        .string_param("city", "The city name", true)
        .build();

    let request = CompletionRequest::new(
        "anthropic/claude-sonnet-4-20250514",
        vec![Message::user("What is the weather in Paris?")],
    )
    .with_tools(vec![tool])
    .with_max_tokens(200);

    let response = client.complete(request).await.expect("Request failed");

    assert!(response.has_tool_use());
    assert!(response.stop_reason == StopReason::ToolUse);

    let tool_uses = response.tool_uses();
    assert!(!tool_uses.is_empty());

    let first_tool = &tool_uses[0];
    if let ContentBlock::ToolUse { name, input, .. } = first_tool {
        assert_eq!(name, "get_weather");
        assert!(input.get("city").is_some());
    } else {
        panic!("Expected ToolUse content block");
    }
}

#[tokio::test]
async fn test_anthropic_multi_turn_conversation() {
    if !has_env("ANTHROPIC_API_KEY") {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "anthropic/claude-sonnet-4-20250514",
        vec![
            Message::user("My name is Alice."),
            Message::assistant("Hello Alice! Nice to meet you."),
            Message::user("What is my name?"),
        ],
    )
    .with_max_tokens(50);

    let response = client.complete(request).await.expect("Request failed");

    assert!(response.text_content().to_lowercase().contains("alice"));
}

#[tokio::test]
async fn test_anthropic_vision() {
    if !has_env("ANTHROPIC_API_KEY") {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "anthropic/claude-sonnet-4-20250514",
        vec![Message::user_with_content(vec![
            ContentBlock::text("What do you see in this image? Be brief."),
            ContentBlock::image_url(
                "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
            ),
        ])],
    )
    .with_max_tokens(100);

    let response = client.complete(request).await.expect("Request failed");

    // Should get some description
    assert!(response.text_content().len() > 10);
}

#[tokio::test]
async fn test_anthropic_extended_thinking() {
    if !has_env("ANTHROPIC_API_KEY") {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "anthropic/claude-sonnet-4-20250514",
        vec![Message::user("What is 15 * 23?")],
    )
    .with_thinking_config(ThinkingConfig::enabled(1024))
    .with_max_tokens(500);

    let response = client.complete(request).await.expect("Request failed");

    // Should get the correct answer
    assert!(response.text_content().contains("345"));

    // Check for thinking blocks in content
    let has_thinking = response
        .content
        .iter()
        .any(|block| matches!(block, ContentBlock::Thinking { .. }));
    assert!(has_thinking, "Expected thinking blocks in response");
}

#[tokio::test]
async fn test_anthropic_streaming() {
    if !has_env("ANTHROPIC_API_KEY") {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "anthropic/claude-sonnet-4-20250514",
        vec![Message::user("Count from 1 to 5, one number per line.")],
    )
    .with_max_tokens(100)
    .with_streaming();

    let mut stream = client
        .complete_stream(request)
        .await
        .expect("Failed to start stream");

    let mut chunks: Vec<String> = Vec::new();
    let mut is_done = false;

    while let Some(result) = stream.next().await {
        let chunk = result.expect("Stream error");

        if let Some(text) = extract_text(&chunk.delta) {
            chunks.push(text);
        }

        if is_stream_done(&chunk.event_type) {
            is_done = true;
            break;
        }
    }

    assert!(is_done);
    assert!(!chunks.is_empty());

    let full_text = chunks.join("");
    assert!(full_text.contains('1'));
    assert!(full_text.contains('5'));
}

#[tokio::test]
async fn test_anthropic_streaming_event_types() {
    if !has_env("ANTHROPIC_API_KEY") {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "anthropic/claude-sonnet-4-20250514",
        vec![Message::user("Hi")],
    )
    .with_max_tokens(20)
    .with_streaming();

    let mut stream = client
        .complete_stream(request)
        .await
        .expect("Failed to start stream");

    let mut event_types = Vec::new();

    while let Some(result) = stream.next().await {
        let chunk = result.expect("Stream error");
        event_types.push(chunk.event_type);

        if is_stream_done(&chunk.event_type) {
            break;
        }
    }

    // Should have received at least ContentBlockDelta events
    assert!(event_types
        .iter()
        .any(|e| matches!(e, StreamEventType::ContentBlockDelta)));
}

// =============================================================================
// OpenAI Tests
// =============================================================================

#[cfg(feature = "openai")]
#[tokio::test]
async fn test_openai_simple_completion() {
    if !has_env("OPENAI_API_KEY") {
        eprintln!("Skipping: OPENAI_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_openai_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "openai/gpt-4o-mini",
        vec![Message::user("What is 3+3? Reply with just the number.")],
    )
    .with_max_tokens(50);

    let response = client.complete(request).await.expect("Request failed");

    assert!(!response.id.is_empty());
    assert!(response.model.to_lowercase().contains("openai/gpt-4o-mini"));
    assert!(response.text_content().contains('6'));
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn test_openai_json_output() {
    if !has_env("OPENAI_API_KEY") {
        eprintln!("Skipping: OPENAI_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_openai_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "openai/gpt-4o-mini",
        vec![Message::user(
            "Return a JSON object with a 'greeting' field saying 'hello'",
        )],
    )
    .with_json_output()
    .with_max_tokens(100);

    let response = client.complete(request).await.expect("Request failed");
    let text = response.text_content();

    // Should be valid JSON
    let parsed: serde_json::Value =
        serde_json::from_str(&text).expect("Response should be valid JSON");

    assert!(parsed.get("greeting").is_some());
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn test_openai_structured_output() {
    if !has_env("OPENAI_API_KEY") {
        eprintln!("Skipping: OPENAI_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_openai_from_env()
        .build()
        .expect("Failed to build client");

    let schema = serde_json::json!({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"],
        "additionalProperties": false
    });

    let request = CompletionRequest::new(
        "openai/gpt-4o-mini",
        vec![Message::user(
            "Generate info for a person named Bob who is 25 years old",
        )],
    )
    .with_json_schema("person", schema)
    .with_max_tokens(100);

    let response = client.complete(request).await.expect("Request failed");
    let parsed: serde_json::Value =
        serde_json::from_str(&response.text_content()).expect("Response should be valid JSON");

    assert_eq!(parsed["name"], "Bob");
    assert_eq!(parsed["age"], 25);
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn test_openai_tool_use() {
    if !has_env("OPENAI_API_KEY") {
        eprintln!("Skipping: OPENAI_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_openai_from_env()
        .build()
        .expect("Failed to build client");

    let tool = ToolBuilder::new("get_weather")
        .description("Get the current weather in a city")
        .string_param("city", "The city name", true)
        .build();

    let request = CompletionRequest::new(
        "openai/gpt-4o-mini",
        vec![Message::user("What is the weather in London?")],
    )
    .with_tools(vec![tool])
    .with_max_tokens(200);

    let response = client.complete(request).await.expect("Request failed");

    assert!(response.has_tool_use());
    assert!(response.stop_reason == StopReason::ToolUse);
}

#[cfg(feature = "openai")]
#[tokio::test]
async fn test_openai_streaming() {
    if !has_env("OPENAI_API_KEY") {
        eprintln!("Skipping: OPENAI_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_openai_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "openai/gpt-4o-mini",
        vec![Message::user("Say 'hello world'")],
    )
    .with_max_tokens(50)
    .with_streaming();

    let mut stream = client
        .complete_stream(request)
        .await
        .expect("Failed to start stream");

    let mut chunks: Vec<String> = Vec::new();

    while let Some(result) = stream.next().await {
        let chunk = result.expect("Stream error");

        if let Some(text) = extract_text(&chunk.delta) {
            chunks.push(text);
        }

        if is_stream_done(&chunk.event_type) {
            break;
        }
    }

    let full_text = chunks.join("").to_lowercase();
    assert!(full_text.contains("hello"));
}

// =============================================================================
// Groq Tests
// =============================================================================

#[cfg(feature = "groq")]
#[tokio::test]
async fn test_groq_simple_completion() {
    if !has_env("GROQ_API_KEY") {
        eprintln!("Skipping: GROQ_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_groq_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "groq/llama-3.3-70b-versatile",
        vec![Message::user(
            "What is the capital of France? Reply with just the city name.",
        )],
    )
    .with_max_tokens(50);

    let response = client.complete(request).await.expect("Request failed");

    assert!(!response.id.is_empty());
    assert!(response.text_content().to_lowercase().contains("paris"));
}

#[cfg(feature = "groq")]
#[tokio::test]
async fn test_groq_streaming() {
    if !has_env("GROQ_API_KEY") {
        eprintln!("Skipping: GROQ_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_groq_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "groq/llama-3.3-70b-versatile",
        vec![Message::user("Count from 1 to 3.")],
    )
    .with_max_tokens(50)
    .with_streaming();

    let mut stream = client
        .complete_stream(request)
        .await
        .expect("Failed to start stream");

    let mut chunks: Vec<String> = Vec::new();

    while let Some(result) = stream.next().await {
        let chunk = result.expect("Stream error");

        if let Some(text) = extract_text(&chunk.delta) {
            chunks.push(text);
        }

        if is_stream_done(&chunk.event_type) {
            break;
        }
    }

    assert!(!chunks.is_empty());
}

// =============================================================================
// Mistral Tests
// =============================================================================

#[cfg(feature = "mistral")]
#[tokio::test]
async fn test_mistral_simple_completion() {
    if !has_env("MISTRAL_API_KEY") {
        eprintln!("Skipping: MISTRAL_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_mistral_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "mistral/mistral-small-latest",
        vec![Message::user("What is 5+5? Reply with just the number.")],
    )
    .with_max_tokens(50);

    let response = client.complete(request).await.expect("Request failed");

    assert!(!response.id.is_empty());
    assert!(response.text_content().contains("10"));
}

// =============================================================================
// Multi-Provider Tests
// =============================================================================

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[tokio::test]
async fn test_multi_provider_switching() {
    if !has_env("ANTHROPIC_API_KEY") || !has_env("OPENAI_API_KEY") {
        eprintln!("Skipping: Both ANTHROPIC_API_KEY and OPENAI_API_KEY required");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .with_openai_from_env()
        .build()
        .expect("Failed to build client");

    // Test with Anthropic
    let anthropic_request = CompletionRequest::new(
        "anthropic/claude-sonnet-4-20250514",
        vec![Message::user("Say 'Anthropic'")],
    )
    .with_max_tokens(20);

    let anthropic_response = client
        .complete(anthropic_request)
        .await
        .expect("Anthropic request failed");
    assert!(anthropic_response.model.to_lowercase().contains("claude"));

    // Test with OpenAI
    let openai_request =
        CompletionRequest::new("openai/gpt-4o-mini", vec![Message::user("Say 'OpenAI'")])
            .with_max_tokens(20);

    let openai_response = client
        .complete(openai_request)
        .await
        .expect("OpenAI request failed");
    assert!(openai_response.model.to_lowercase().contains("gpt"));
}

#[tokio::test]
async fn test_complete_with_provider() {
    if !has_env("ANTHROPIC_API_KEY") {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "anthropic/claude-sonnet-4-20250514",
        vec![Message::user("Say 'test'")],
    )
    .with_max_tokens(20);

    let response = client
        .complete_with_provider("anthropic", request)
        .await
        .expect("Request failed");

    assert!(response.model.to_lowercase().contains("claude"));
}

#[cfg(all(feature = "anthropic", feature = "openai"))]
#[tokio::test]
async fn test_multi_provider_streaming() {
    if !has_env("ANTHROPIC_API_KEY") || !has_env("OPENAI_API_KEY") {
        eprintln!("Skipping: Both ANTHROPIC_API_KEY and OPENAI_API_KEY required");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .with_openai_from_env()
        .build()
        .expect("Failed to build client");

    // Test streaming with Anthropic
    let anthropic_request = CompletionRequest::new(
        "anthropic/claude-sonnet-4-20250514",
        vec![Message::user("Say 'A'")],
    )
    .with_max_tokens(10)
    .with_streaming();

    let mut anthropic_stream = client
        .complete_stream(anthropic_request)
        .await
        .expect("Failed to start Anthropic stream");

    let mut anthropic_text = String::new();
    while let Some(result) = anthropic_stream.next().await {
        let chunk = result.expect("Stream error");
        if let Some(text) = extract_text(&chunk.delta) {
            anthropic_text.push_str(&text);
        }
        if is_stream_done(&chunk.event_type) {
            break;
        }
    }

    assert!(!anthropic_text.is_empty());

    // Test streaming with OpenAI
    let openai_request =
        CompletionRequest::new("openai/gpt-4o-mini", vec![Message::user("Say 'B'")])
            .with_max_tokens(10)
            .with_streaming();

    let mut openai_stream = client
        .complete_stream(openai_request)
        .await
        .expect("Failed to start OpenAI stream");

    let mut openai_text = String::new();
    while let Some(result) = openai_stream.next().await {
        let chunk = result.expect("Stream error");
        if let Some(text) = extract_text(&chunk.delta) {
            openai_text.push_str(&text);
        }
        if is_stream_done(&chunk.event_type) {
            break;
        }
    }

    assert!(!openai_text.is_empty());
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[tokio::test]
async fn test_invalid_model_error() {
    if !has_env("ANTHROPIC_API_KEY") {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()
        .expect("Failed to build client");

    let request = CompletionRequest::new(
        "anthropic/non-existent-model-12345",
        vec![Message::user("Hello")],
    )
    .with_max_tokens(50);

    let result = client.complete(request).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_empty_messages_error() {
    if !has_env("ANTHROPIC_API_KEY") {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()
        .expect("Failed to build client");

    let request =
        CompletionRequest::new("anthropic/claude-sonnet-4-20250514", vec![]).with_max_tokens(50);

    let result = client.complete(request).await;
    assert!(result.is_err());
}

#[tokio::test]
async fn test_streaming_invalid_model() {
    if !has_env("ANTHROPIC_API_KEY") {
        eprintln!("Skipping: ANTHROPIC_API_KEY not set");
        return;
    }

    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()
        .expect("Failed to build client");

    let request =
        CompletionRequest::new("anthropic/invalid-model-xyz", vec![Message::user("Hello")])
            .with_max_tokens(50)
            .with_streaming();

    let result = client.complete_stream(request).await;
    assert!(result.is_err());
}
