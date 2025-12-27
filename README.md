# LLMKit

Unified LLM API library for Rust - one interface for all providers.

## Features

- **Multi-provider**: Anthropic, OpenAI, OpenRouter, Ollama
- **Unified types**: Same message format works everywhere
- **Streaming**: First-class streaming support
- **Tool calling**: Abstract tool definitions
- **Builder pattern**: Fluent configuration

## Quick Start

```rust
use llmkit::{LLMKitClient, Message, Role, ContentBlock, CompletionRequest};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let client = LLMKitClient::builder()
        .with_anthropic_from_env()
        .build()?;

    let response = client.complete(CompletionRequest {
        model: "claude-sonnet-4-20250514".to_string(),
        messages: vec![Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: "Hello!".to_string()
            }],
        }],
        ..Default::default()
    }).await?;

    Ok(())
}
```

## Streaming

```rust
use futures::StreamExt;

let stream = client.complete_stream(request).await?;

while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    // Handle streaming chunk
}
```

## Tool Calling

```rust
use llmkit::ToolDefinition;
use serde_json::json;

let tools = vec![ToolDefinition {
    name: "get_weather".to_string(),
    description: "Get current weather".to_string(),
    input_schema: json!({
        "type": "object",
        "properties": {
            "location": {"type": "string"}
        },
        "required": ["location"]
    }),
}];

let request = CompletionRequest {
    model: "claude-sonnet-4-20250514".to_string(),
    messages: messages,
    tools: Some(tools),
    ..Default::default()
};
```

## Providers

| Provider | Environment Variable |
|----------|---------------------|
| Anthropic | `ANTHROPIC_API_KEY` |
| OpenAI | `OPENAI_API_KEY` |
| OpenRouter | `OPENROUTER_API_KEY` |
| Ollama | (local, no key) |
| Groq | `GROQ_API_KEY` |
| Mistral | `MISTRAL_API_KEY` |

## Features

```toml
[dependencies]
llmkit = { version = "0.1", features = ["anthropic", "openai"] }
```

| Feature | Description |
|---------|-------------|
| `anthropic` | Anthropic Claude models (default) |
| `openai` | OpenAI GPT models (default) |
| `openrouter` | OpenRouter (100+ models) |
| `ollama` | Local Ollama models |
| `groq` | Groq (ultra-fast inference) |
| `mistral` | Mistral AI models |
| `all-providers` | All providers |

## License

MIT OR Apache-2.0
