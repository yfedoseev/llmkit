# Getting Started with ModelSuite (Rust)

ModelSuite is a unified LLM API client that provides a single interface to 48+ LLM providers and 120+ models including Anthropic, OpenAI, Azure, AWS Bedrock, Google Vertex AI, and many more.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
llmkit = { version = "0.1", features = ["anthropic", "openai"] }
tokio = { version = "1", features = ["full"] }
```

### Feature Flags

Select only the providers you need:

```toml
[dependencies]
# Minimal - just Anthropic
llmkit = { version = "0.1", features = ["anthropic"] }

# Common providers
llmkit = { version = "0.1", features = ["anthropic", "openai", "groq"] }

# All providers
llmkit = { version = "0.1", features = ["all-providers"] }
```

Available feature flags:
- `anthropic` - Anthropic Claude (default)
- `openai` - OpenAI GPT (default)
- `azure` - Azure OpenAI
- `bedrock` - AWS Bedrock
- `vertex` - Google Vertex AI
- `google` - Google AI (Gemini)
- `groq` - Groq
- `mistral` - Mistral AI
- `cohere` - Cohere
- `deepseek` - DeepSeek
- `openrouter` - OpenRouter
- `ollama` - Ollama (local)
- And 25+ more...
- `all-providers` - Enable all providers

## Quick Start

```rust
use llmkit::{LLMKitClient, Message, CompletionRequest};

#[tokio::main]
async fn main() -> llmkit::Result<()> {
    // Create client from environment variables
    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()?;

    // Make a completion request
    let request = CompletionRequest::new(
        "claude-sonnet-4-20250514",
        vec![Message::user("What is the capital of France?")]
    );

    let response = client.complete(request).await?;
    println!("{}", response.text_content());

    Ok(())
}
```

## Environment Setup

Set one or more provider API keys:

```bash
# Core providers
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Cloud providers
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=gpt-4
export AWS_REGION=us-east-1  # For Bedrock

# Fast inference
export GROQ_API_KEY=...
export MISTRAL_API_KEY=...
```

## Explicit Configuration

Configure providers explicitly:

```rust
use llmkit::LLMKitClient;

let client = LLMKitClient::builder()
    .with_anthropic("your-api-key")
    .with_openai("your-openai-key")
    .with_azure("your-azure-key", "endpoint", "deployment")
    .with_default_retry()
    .build()?;
```

## Streaming

Stream responses in real-time:

```rust
use futures::StreamExt;

let request = CompletionRequest::new(
    "claude-sonnet-4-20250514",
    vec![Message::user("Write a haiku about programming")]
).with_stream(true);

let mut stream = client.complete_stream(request).await?;

while let Some(result) = stream.next().await {
    let chunk = result?;
    if let Some(text) = chunk.text() {
        print!("{}", text);
    }
}
println!();
```

## Tool Calling (Function Calling)

Define and use tools:

```rust
use llmkit::{ToolDefinition, ContentBlock};
use serde_json::json;

// Define a tool
let weather_tool = ToolDefinition {
    name: "get_weather".to_string(),
    description: "Get current weather for a city".to_string(),
    input_schema: json!({
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"]
            }
        },
        "required": ["city"]
    }),
};

// Make request with tools
let request = CompletionRequest::new(
    "claude-sonnet-4-20250514",
    vec![Message::user("What's the weather in Paris?")]
).with_tools(vec![weather_tool]);

let response = client.complete(request).await?;

// Check if the model wants to use a tool
if response.has_tool_use() {
    for content in &response.content {
        if let ContentBlock::ToolUse { id, name, input } = content {
            println!("Tool: {}", name);
            println!("Input: {}", input);

            // Execute the tool and send results back
            let tool_result = ContentBlock::ToolResult {
                tool_use_id: id.clone(),
                content: r#"{"temperature": 22, "unit": "celsius"}"#.to_string(),
                is_error: false,
            };

            // Continue the conversation
            let mut messages = vec![
                Message::user("What's the weather in Paris?"),
                Message::assistant_with_content(response.content.clone()),
                Message::user_with_content(vec![tool_result]),
            ];

            let final_response = client.complete(
                CompletionRequest::new("claude-sonnet-4-20250514", messages)
            ).await?;

            println!("{}", final_response.text_content());
        }
    }
}
```

## Structured Output

Get JSON responses with schema validation:

```rust
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize)]
struct Person {
    name: String,
    age: u32,
    city: String,
}

let schema = json!({
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"}
    },
    "required": ["name", "age", "city"]
});

let request = CompletionRequest::new(
    "claude-sonnet-4-20250514",
    vec![Message::user("Generate a fake person's info")]
).with_json_schema("person", schema);

let response = client.complete(request).await?;
let person: Person = serde_json::from_str(&response.text_content())?;
println!("{:?}", person);
```

## Extended Thinking

Enable reasoning mode for complex tasks:

```rust
let request = CompletionRequest::new(
    "claude-sonnet-4-20250514",
    vec![Message::user("Solve this puzzle: ...")]
).with_thinking(5000);  // 5000 token budget

let response = client.complete(request).await?;

// Get thinking content (reasoning process)
if let Some(thinking) = response.thinking_content() {
    println!("Thinking: {}", thinking);
}

println!("Answer: {}", response.text_content());
```

## Vision / Image Analysis

Analyze images:

```rust
use std::fs;
use base64::Engine;

// From file
let image_bytes = fs::read("image.png")?;
let image_data = base64::engine::general_purpose::STANDARD.encode(&image_bytes);

let message = Message::user_with_content(vec![
    ContentBlock::Text { text: "What's in this image?".to_string() },
    ContentBlock::Image {
        media_type: "image/png".to_string(),
        data: image_data,
    },
]);

let response = client.complete(
    CompletionRequest::new("claude-sonnet-4-20250514", vec![message])
).await?;

println!("{}", response.text_content());
```

## Error Handling

Handle errors gracefully:

```rust
use llmkit::error::Error;

match client.complete(request).await {
    Ok(response) => {
        println!("{}", response.text_content());
    }
    Err(Error::Authentication(msg)) => {
        eprintln!("Invalid API key: {}", msg);
    }
    Err(Error::RateLimit { retry_after, .. }) => {
        eprintln!("Rate limited. Retry after: {:?}", retry_after);
    }
    Err(Error::ContextLength { max, actual, .. }) => {
        eprintln!("Input too long: {} tokens (max: {})", actual, max);
    }
    Err(Error::InvalidRequest(msg)) => {
        eprintln!("Invalid request: {}", msg);
    }
    Err(Error::ProviderNotFound(provider)) => {
        eprintln!("Provider not configured: {}", provider);
    }
    Err(Error::Timeout) => {
        eprintln!("Request timed out");
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## Multiple Providers

Use different providers for different tasks:

```rust
// Configure multiple providers
let client = LLMKitClient::builder()
    .with_anthropic_from_env()
    .with_openai_from_env()
    .with_groq_from_env()
    .build()?;

// Use a specific provider by model prefix
let response = client.complete(
    CompletionRequest::new("gpt-4o", vec![Message::user("Hello!")])
).await?;

// Or explicitly
let response = client.complete_with_provider(
    "openai",
    CompletionRequest::new("gpt-4o", vec![Message::user("Hello!")])
).await?;
```

## Prompt Caching

Cache frequently used prompts (Anthropic):

```rust
let request = CompletionRequest::new(
    "claude-sonnet-4-20250514",
    vec![Message::user("Summarize this document: ...")]
)
.with_system("You are a document summarizer.")
.with_cache_control(CacheControl::Ephemeral);  // 5-minute cache

let response = client.complete(request).await?;

// Check cache usage
if let Some(usage) = &response.usage {
    println!("Cache creation: {:?}", usage.cache_creation_input_tokens);
    println!("Cache read: {:?}", usage.cache_read_input_tokens);
}
```

## Model Registry

Query available models:

```rust
use llmkit::model_registry::{
    get_model_info,
    get_all_models,
    get_models_by_provider,
    get_available_models,
    Provider,
};

// Get info about a specific model
if let Some(info) = get_model_info("claude-sonnet-4-20250514") {
    println!("Name: {}", info.name);
    println!("Price: ${}/1M input tokens", info.pricing.input_per_1m);
    println!("Max context: {}", info.capabilities.max_context);
    println!("Supports vision: {}", info.capabilities.vision);
}

// Get all Anthropic models
let anthropic_models = get_models_by_provider(Provider::Anthropic);
for model in anthropic_models {
    println!("{}: {}", model.name, model.description);
}

// Get available models (with configured API keys)
let available = get_available_models();
println!("{} models available", available.len());
```

## Async Runtime

LLMKit requires an async runtime. We recommend Tokio:

```rust
// Using tokio::main macro
#[tokio::main]
async fn main() -> llmkit::Result<()> {
    // Your code here
    Ok(())
}

// Or build a runtime manually
fn main() -> llmkit::Result<()> {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;

    rt.block_on(async {
        // Your async code here
        Ok(())
    })
}
```

## Performance Tips

1. **Reuse clients**: Create one client and share it across requests
2. **Use feature flags**: Only enable providers you need
3. **Enable streaming**: For long responses, streaming reduces time-to-first-token
4. **Use prompt caching**: Cache system prompts to reduce costs and latency

```rust
use std::sync::Arc;

// Create a shared client
let client = Arc::new(LLMKitClient::builder()
    .with_anthropic_from_env()
    .build()?);

// Clone Arc for each task
let client_clone = client.clone();
tokio::spawn(async move {
    let response = client_clone.complete(request).await?;
    // ...
});
```

## Next Steps

- Check out the [examples](../examples/) for more complete code samples
- Run examples with `cargo run --example simple_completion`
- See the API docs with `cargo doc --open`
