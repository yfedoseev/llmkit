# LLMKit Python

**The production-grade LLM client for Python.** Native Rust performance with a Pythonic API.

[![PyPI](https://img.shields.io/pypi/v/llmkit-python.svg)](https://pypi.org/project/llmkit-python/)
[![Python](https://img.shields.io/pypi/pyversions/llmkit-python.svg)](https://pypi.org/project/llmkit-python/)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/yfedoseev/llmkit/blob/main/LICENSE)

## Why LLMKit?

- **Rust Core** — Native performance, memory safety, no GIL limitations
- **100+ Providers** — OpenAI, Anthropic, Google, AWS Bedrock, Azure, Groq, and more
- **11,000+ Models** — Built-in registry with pricing and capabilities
- **Prompt Caching** — Save up to 90% on API costs with native caching support
- **Extended Thinking** — Unified reasoning API across 5 providers
- **Production Ready** — No memory leaks, no worker restarts, runs forever

## Installation

```bash
pip install llmkit-python
```

## Quick Start

```python
from llmkit import LLMKitClient, CompletionRequest, Message

# Create client from environment variables
client = LLMKitClient.from_env()

# Make a completion request
response = client.complete(CompletionRequest(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[Message.user("Hello!")]
))
print(response.text_content())
```

## Async Support

```python
from llmkit import AsyncLLMKitClient, CompletionRequest, Message

async def main():
    client = AsyncLLMKitClient.from_env()
    response = await client.complete(CompletionRequest(
        model="openai/gpt-4o",
        messages=[Message.user("Hello!")]
    ))
    print(response.text_content())
```

## Streaming

```python
# Sync streaming
for chunk in client.complete_stream(request):
    if chunk.text:
        print(chunk.text, end="", flush=True)

# Async streaming
async for chunk in await async_client.complete_stream(request):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

## Tool Calling

```python
from llmkit import ToolBuilder

# Build tools with fluent API
weather_tool = ToolBuilder("get_weather") \
    .description("Get current weather for a location") \
    .string_param("city", "City name", required=True) \
    .enum_param("unit", "Temperature unit", ["celsius", "fahrenheit"]) \
    .build()

request = CompletionRequest(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[Message.user("What's the weather in Tokyo?")]
).with_tools([weather_tool])

response = client.complete(request)
for tool_call in response.tool_calls():
    print(f"Call {tool_call.name} with {tool_call.arguments}")
```

## Prompt Caching

Save up to 90% on repeated prompts:

```python
# Large system prompts are automatically cached
request = CompletionRequest(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[
        Message.system(large_system_prompt),  # Cached after first call
        Message.user("Question 1")
    ]
).with_cache()

# Subsequent calls reuse the cached system prompt
response = client.complete(request)
print(f"Cache savings: {response.usage.cache_read_tokens} tokens")
```

## Extended Thinking

Unified reasoning across Anthropic, OpenAI, Google, DeepSeek, and OpenRouter:

```python
request = CompletionRequest(
    model="anthropic/claude-sonnet-4-20250514",
    messages=[Message.user("Solve this step by step: ...")]
).with_thinking(budget_tokens=10000)

response = client.complete(request)
print("Reasoning:", response.thinking_content())
print("Answer:", response.text_content())
```

## Model Registry

11,000+ models with pricing and capabilities — no API calls needed:

```python
from llmkit import get_model_info, get_models_by_provider, get_models_with_capability

# Get model details instantly
info = get_model_info("anthropic/claude-sonnet-4-20250514")
print(f"Context: {info.context_window:,} tokens")
print(f"Input: ${info.input_price}/1M tokens")
print(f"Output: ${info.output_price}/1M tokens")

# Find models by provider
anthropic_models = get_models_by_provider("anthropic")

# Find models with specific capabilities
vision_models = get_models_with_capability(vision=True)
```

## Features

| Feature | Status |
|---------|--------|
| Chat Completions | Supported |
| Streaming | Supported |
| Tool Calling | Supported |
| Structured Output | Supported |
| Extended Thinking | Supported |
| Prompt Caching | Supported |
| Vision/Images | Supported |
| Embeddings | Supported |
| Image Generation | Supported |
| Audio STT/TTS | Supported |
| Video Generation | Supported |

## Documentation

- [Getting Started](https://github.com/yfedoseev/llmkit/blob/main/docs/getting-started-python.md)
- [Full Documentation](https://github.com/yfedoseev/llmkit/tree/main/docs)
- [Examples](https://github.com/yfedoseev/llmkit/tree/main/examples)

## License

MIT OR Apache-2.0
