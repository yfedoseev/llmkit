# Getting Started with ModelSuite (Python)

ModelSuite is a unified LLM API client that provides a single interface to 48+ LLM providers and 120+ models including Anthropic, OpenAI, Azure, AWS Bedrock, Google Vertex AI, and many more.

## Installation

```bash
pip install modelsuite
```

## Quick Start

```python
from modelsuite import ModelSuiteClient, Message, CompletionRequest

# Create client from environment variables
client = ModelSuiteClient.from_env()

# Make a completion request
response = client.complete(
    CompletionRequest(
        model="claude-sonnet-4-20250514",
        messages=[Message.user("What is the capital of France?")]
    )
)

print(response.text_content())
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
export GOOGLE_CLOUD_PROJECT=your-project  # For Vertex AI
export VERTEX_REGION=us-central1

# Fast inference providers
export GROQ_API_KEY=...
export MISTRAL_API_KEY=...
export CEREBRAS_API_KEY=...
export DEEPSEEK_API_KEY=...

# Other providers
export COHERE_API_KEY=...
export OPENROUTER_API_KEY=...
# ... and 30+ more
```

ModelSuite automatically detects which providers are configured from environment variables.

## Explicit Configuration

Instead of environment variables, you can configure providers explicitly:

```python
client = ModelSuiteClient(
    providers={
        "anthropic": {"api_key": "sk-ant-..."},
        "openai": {"api_key": "sk-..."},
        "azure": {
            "api_key": "...",
            "endpoint": "https://your-resource.openai.azure.com",
            "deployment": "gpt-4"
        },
        "bedrock": {"region": "us-east-1"},
    },
    default_provider="anthropic"
)
```

## Streaming

Stream responses in real-time:

```python
request = CompletionRequest(
    model="claude-sonnet-4-20250514",
    messages=[Message.user("Write a haiku about programming")]
).with_streaming()

for chunk in client.complete_stream(request):
    if chunk.text:
        print(chunk.text, end="", flush=True)
    if chunk.is_done:
        break
print()
```

## Async Usage

For async applications:

```python
import asyncio
from modelsuite import AsyncModelSuiteClient, Message, CompletionRequest

async def main():
    client = AsyncModelSuiteClient.from_env()

    response = await client.complete(
        CompletionRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message.user("Hello!")]
        )
    )
    print(response.text_content())

asyncio.run(main())
```

### Async Streaming

```python
async def stream_example():
    client = AsyncModelSuiteClient.from_env()

    request = CompletionRequest(
        model="claude-sonnet-4-20250514",
        messages=[Message.user("Tell me a story")]
    ).with_streaming()

    async for chunk in await client.complete_stream(request):
        if chunk.text:
            print(chunk.text, end="", flush=True)
```

## Tool Calling (Function Calling)

Define and use tools:

```python
from modelsuite import ToolBuilder, ContentBlock

# Define a tool
weather_tool = ToolBuilder("get_weather") \
    .description("Get current weather for a city") \
    .string_param("city", "City name", required=True) \
    .enum_param("unit", "Temperature unit", ["celsius", "fahrenheit"]) \
    .build()

# Make request with tools
request = CompletionRequest(
    model="claude-sonnet-4-20250514",
    messages=[Message.user("What's the weather in Paris?")]
).with_tools([weather_tool])

response = client.complete(request)

# Check if the model wants to use a tool
if response.has_tool_use():
    for tool_use in response.tool_uses():
        tool_info = tool_use.as_tool_use()
        print(f"Tool: {tool_info[1]}")  # name
        print(f"Input: {tool_info[2]}")  # input dict

        # Execute the tool and send results back
        result = ContentBlock.tool_result(
            tool_use_id=tool_info[0],  # id
            content='{"temperature": 22, "unit": "celsius"}',
            is_error=False
        )

        # Continue the conversation with tool results
        messages = [
            Message.user("What's the weather in Paris?"),
            Message.assistant_with_content(response.content),
            Message.tool_results([result])
        ]

        final_response = client.complete(
            CompletionRequest(model="claude-sonnet-4-20250514", messages=messages)
        )
        print(final_response.text_content())
```

## Structured Output

Get JSON responses with schema validation:

```python
import json

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "city": {"type": "string"}
    },
    "required": ["name", "age", "city"]
}

request = CompletionRequest(
    model="claude-sonnet-4-20250514",
    messages=[Message.user("Generate a fake person's info")]
).with_json_schema("person", schema)

response = client.complete(request)
data = json.loads(response.text_content())
print(data)  # {"name": "Alice", "age": 30, "city": "Paris"}
```

## Extended Thinking

Enable reasoning mode for complex tasks:

```python
request = CompletionRequest(
    model="claude-sonnet-4-20250514",
    messages=[Message.user("Solve this puzzle: ...")]
).with_thinking(budget_tokens=5000)

response = client.complete(request)

# Get thinking content (reasoning process)
if thinking := response.thinking_content():
    print("Thinking:", thinking)

print("Answer:", response.text_content())
```

## Vision / Image Analysis

Analyze images:

```python
import base64

# From file
with open("image.png", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

message = Message.user_with_content([
    ContentBlock.text("What's in this image?"),
    ContentBlock.image("image/png", image_data)
])

# Or from URL
message = Message.user_with_content([
    ContentBlock.text("Describe this image:"),
    ContentBlock.image_url("https://example.com/image.png")
])

response = client.complete(
    CompletionRequest(model="claude-sonnet-4-20250514", messages=[message])
)
print(response.text_content())
```

## Embeddings

Generate text embeddings:

```python
from modelsuite import EmbeddingRequest

# Single text
request = EmbeddingRequest("text-embedding-3-small", "Hello, world!")
response = client.embed(request)

print(f"Dimensions: {response.dimensions}")
print(f"First 5 values: {response.values()[:5]}")

# Batch embeddings
request = EmbeddingRequest.batch(
    "text-embedding-3-small",
    ["Hello", "World", "How are you?"]
)
response = client.embed(request)

for embedding in response.embeddings:
    print(f"Index {embedding.index}: {len(embedding.values)} dimensions")

# Compute similarity
emb1 = response.embeddings[0]
emb2 = response.embeddings[1]
similarity = emb1.cosine_similarity(emb2)
print(f"Similarity: {similarity}")
```

## Token Counting

Estimate token usage before making requests:

```python
from modelsuite import TokenCountRequest

count_request = TokenCountRequest(
    model="claude-sonnet-4-20250514",
    messages=[Message.user("Hello, how are you?")],
    system="You are a helpful assistant"
)

result = client.count_tokens(count_request)
print(f"Input tokens: {result.input_tokens}")
```

## Batch Processing

Process multiple requests asynchronously:

```python
from modelsuite import BatchRequest

# Create batch requests
batch_requests = [
    BatchRequest("request-1", CompletionRequest(
        model="claude-sonnet-4-20250514",
        messages=[Message.user("What is 2+2?")]
    )),
    BatchRequest("request-2", CompletionRequest(
        model="claude-sonnet-4-20250514",
        messages=[Message.user("What is 3+3?")]
    )),
]

# Submit batch
batch_job = client.create_batch(batch_requests)
print(f"Batch ID: {batch_job.id}")
print(f"Status: {batch_job.status}")

# Check status
batch_job = client.get_batch("anthropic", batch_job.id)
print(f"Status: {batch_job.status}")
print(f"Succeeded: {batch_job.request_counts.succeeded}")

# Get results when complete
if batch_job.is_complete():
    results = client.get_batch_results("anthropic", batch_job.id)
    for result in results:
        if result.is_success():
            print(f"{result.custom_id}: {result.response.text_content()}")
        else:
            print(f"{result.custom_id}: Error - {result.error.message}")
```

## Model Registry

Query available models:

```python
from modelsuite import (
    get_model_info,
    get_all_models,
    get_models_by_provider,
    get_available_models,
    get_cheapest_model,
    Provider
)

# Get info about a specific model
info = get_model_info("claude-sonnet-4-20250514")
if info:
    print(f"Name: {info.name}")
    print(f"Price: ${info.pricing.input_per_1m}/1M input tokens")
    print(f"Max context: {info.capabilities.max_context}")
    print(f"Supports vision: {info.capabilities.vision}")
    print(f"Supports tools: {info.capabilities.tools}")

# Get all Anthropic models
anthropic_models = get_models_by_provider(Provider.Anthropic)
for model in anthropic_models:
    print(f"{model.name}: {model.description}")

# Get available models (with configured API keys)
available = get_available_models()
print(f"{len(available)} models available")

# Find cheapest model with specific requirements
cheapest = get_cheapest_model(min_context=100000, needs_vision=True, needs_tools=True)
if cheapest:
    print(f"Cheapest: {cheapest.name}")
```

## Error Handling

Handle errors gracefully:

```python
from modelsuite import (
    ModelSuiteError,
    ProviderNotFoundError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ContextLengthError,
    TimeoutError,
)

try:
    response = client.complete(request)
except AuthenticationError:
    print("Invalid API key")
except RateLimitError as e:
    print(f"Rate limited. Retry after: {e.retry_after_seconds}s")
except ContextLengthError:
    print("Input too long")
except InvalidRequestError as e:
    print(f"Invalid request: {e}")
except ProviderNotFoundError:
    print("Provider not configured")
except TimeoutError:
    print("Request timed out")
except ModelSuiteError as e:
    print(f"ModelSuite error: {e}")
```

## Multiple Providers

Use different providers for different tasks:

```python
# List available providers
print(client.providers())  # ['anthropic', 'openai', ...]

# Use a specific provider
response = client.complete_with_provider(
    "openai",
    CompletionRequest(
        model="gpt-4o",
        messages=[Message.user("Hello!")]
    )
)

# Check default provider
print(client.default_provider)
```

## Prompt Caching

Cache frequently used prompts (Anthropic):

```python
# Enable caching on system prompt
request = CompletionRequest(
    model="claude-sonnet-4-20250514",
    messages=[Message.user("Summarize this document: ...")]
).with_system("You are a document summarizer.") \
 .with_system_caching()  # 5-minute cache

# Extended caching (1 hour)
request = request.with_system_caching_extended()

# Check cache usage in response
response = client.complete(request)
if response.usage:
    print(f"Cache creation: {response.usage.cache_creation_input_tokens}")
    print(f"Cache read: {response.usage.cache_read_input_tokens}")
```

## Next Steps

- Check out the [examples](../examples/python/) for more complete code samples
- See the [API Reference](./api-reference-python.md) for detailed documentation
- View the [Provider Guide](./providers.md) for provider-specific configuration
