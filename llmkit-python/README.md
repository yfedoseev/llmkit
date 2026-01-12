# LLMKit Python

Python bindings for LLMKit - a unified LLM API client library.

## Installation

```bash
pip install llmkit
```

## Quick Start

```python
from llmkit import LLMKitClient, AsyncLLMKitClient, CompletionRequest, Message

# Sync client - use "provider/model" format
client = LLMKitClient.from_env()
response = client.complete(CompletionRequest(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[Message.user("Hello!")],
))
print(response.text_content())

# Async client
async def main():
    client = AsyncLLMKitClient.from_env()
    response = await client.complete(CompletionRequest(
        model="anthropic/claude-sonnet-4-20250514",
        messages=[Message.user("Hello!")],
    ))
    print(response.text_content())

# Streaming
for chunk in client.complete_stream(request):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

## Features

- Unified API for 70+ LLM providers
- Sync and async clients
- Streaming support
- Tool/function calling
- Extended thinking (reasoning)
- Prompt caching
- Structured output (JSON schema)

## License

MIT OR Apache-2.0
