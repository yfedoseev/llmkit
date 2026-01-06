# ModelSuite

**Unified LLM API library** - One interface for 70+ LLM providers and specialized APIs.

Rust core with Python and Node.js/TypeScript bindings.

## Features

- **70+ Providers & Specialized APIs**: OpenAI, Anthropic, Together, Fireworks, DeepSeek, Perplexity, Mistral, Groq, xAI, Azure, Bedrock, Vertex AI, Google, Cohere, HuggingFace, Replicate, Ollama, vLLM, LM Studio, Lambda Labs, and more (local & cloud)
- **Comprehensive Coverage**: 70+ providers with superior extended thinking and reasoning support
- **Extended Thinking Across 4 Providers**: Unified `ThinkingConfig` for OpenAI, Anthropic, Google Vertex, DeepSeek
- **Regional Providers**: GDPR-compliant endpoints (Mistral EU, Maritaca Brazil)
- **Real-Time Voice**: Deepgram v3, ElevenLabs with latency control (Speech-to-Text & Text-to-Speech)
- **Video Generation**: Runware aggregator supporting 5+ video models (Runway, Kling, Pika, Hailuo, Leonardo)
- **Image Generation**: 4 providers (OpenAI, FAL AI, Stability AI, Recraft with vector design)
- **Specialized APIs**: Ranking, Reranking, Moderation, Classification
- **Domain-Specific Models**: Med-PaLM 2 (healthcare), DeepSeek-R1 (scientific reasoning)
- **Unified Types**: Same message format works everywhere
- **Streaming**: First-class async streaming support
- **Tool Calling**: Abstract tool definitions with builder pattern
- **Prompt Caching**: Cache system prompts for cost savings
- **Structured Output**: JSON schema enforcement
- **Vision/Images**: Image analysis support
- **Embeddings**: Text embedding generation
- **Batch Processing**: Async batch API
- **Token Counting**: Estimate costs before requests
- **Model Registry**: 1,798+ models with pricing and capabilities

## Installation

### Python

```bash
pip install modelsuite
```

### Node.js/TypeScript

```bash
npm install modelsuite
```

### Rust

```toml
[dependencies]
modelsuite = { version = "0.1", features = ["anthropic", "openai"] }
```

## Quick Start

### Python

```python
from modelsuite import ModelSuiteClient, Message, CompletionRequest

client = ModelSuiteClient.from_env()
response = client.complete(
    CompletionRequest(
        model="anthropic/claude-sonnet-4-20250514",  # provider/model format
        messages=[Message.user("Hello!")]
    )
)
print(response.text_content())
```

### Node.js/TypeScript

```typescript
import { JsModelSuiteClient as ModelSuiteClient, JsMessage as Message, JsCompletionRequest as CompletionRequest } from 'modelsuite'

const client = ModelSuiteClient.fromEnv()
const response = await client.complete(
    CompletionRequest.create('openai/gpt-4o', [Message.user('Hello!')])
)
console.log(response.textContent())
```

### Rust

```rust
use modelsuite::{ModelSuiteClient, Message, CompletionRequest};

#[tokio::main]
async fn main() -> modelsuite::Result<()> {
    let client = ModelSuiteClient::from_env()?;

    let response = client.complete(
        CompletionRequest::new("groq/llama-3.3-70b-versatile", vec![Message::user("Hello!")])
    ).await?;

    println!("{}", response.text_content());
    Ok(())
}
```

## Model Format

ModelSuite uses a unified `"provider/model"` format for explicit provider routing:

```
anthropic/claude-sonnet-4-20250514
openai/gpt-4o
groq/llama-3.3-70b-versatile
mistral/mistral-large-latest
```

This format is self-documenting and eliminates ambiguity. The provider prefix is **required** for all models.

## Audio APIs

### Python

```python
from modelsuite import ModelSuiteClient, TranscriptionRequest, SynthesisRequest

client = ModelSuiteClient.from_env()

# Speech-to-Text
audio_file = open("speech.mp3", "rb")
transcript = client.transcribe_audio(TranscriptionRequest(audio_file.read()))
print(f"Transcribed: {transcript.text}")

# Text-to-Speech
speech = client.synthesize_speech(SynthesisRequest("Hello, world!", voice="alloy"))
with open("output.mp3", "wb") as f:
    f.write(speech.audio_bytes)
```

### Node.js/TypeScript

```typescript
import { ModelSuiteClient, TranscriptionRequest, SynthesisRequest } from 'modelsuite'

const client = ModelSuiteClient.fromEnv()

// Speech-to-Text
const audioBytes = fs.readFileSync("speech.mp3")
const transcript = await client.transcribeAudio(new TranscriptionRequest(audioBytes))
console.log(`Transcribed: ${transcript.text}`)

// Text-to-Speech
const speech = await client.synthesizeSpeech(new SynthesisRequest("Hello, world!", "alloy"))
fs.writeFileSync("output.mp3", speech.audioBytes)
```

## Video Generation

### Python

```python
from modelsuite import ModelSuiteClient, VideoGenerationRequest

client = ModelSuiteClient.from_env()

# Generate video from text
video = client.generate_video(
    VideoGenerationRequest("A serene sunset over mountains")
    .with_model("runway-gen3-alpha")
    .with_duration(10)
)
print(f"Video URL: {video.video_url}")
```

### Node.js/TypeScript

```typescript
import { ModelSuiteClient, VideoGenerationRequest } from 'modelsuite'

const client = ModelSuiteClient.fromEnv()

// Generate video
const video = await client.generateVideo(
    new VideoGenerationRequest("A serene sunset over mountains")
        .withModel("runway-gen3-alpha")
        .withDuration(10)
)
console.log(`Video URL: ${video.videoUrl}`)
```

## Image Generation

### Python

```python
from modelsuite import ModelSuiteClient, ImageGenerationRequest

client = ModelSuiteClient.from_env()

# Generate image
image = client.generate_image(
    ImageGenerationRequest("A futuristic city at sunset")
    .with_model("dall-e-3")
    .with_size("1024x1024")
)
print(f"Image URL: {image.images[0].url}")
```

### Node.js/TypeScript

```typescript
import { ModelSuiteClient, ImageGenerationRequest } from 'modelsuite'

const client = ModelSuiteClient.fromEnv()

// Generate image
const image = await client.generateImage(
    new ImageGenerationRequest("A futuristic city at sunset")
        .withModel("dall-e-3")
        .withSize("1024x1024")
)
console.log(`Image URL: ${image.images[0].url}`)
```

## Specialized APIs

### Python

```python
from modelsuite import ModelSuiteClient, ModerationRequest, ClassificationRequest

client = ModelSuiteClient.from_env()

# Content moderation
moderation = client.moderate_text(ModerationRequest("Is this content appropriate?"))
print(f"Flagged: {moderation.flagged}")

# Text classification
classification = client.classify_text(
    ClassificationRequest("Great product!", ["positive", "negative", "neutral"])
)
print(f"Sentiment: {classification.top().label}")
```

### Node.js/TypeScript

```typescript
import { ModelSuiteClient, ModerationRequest, ClassificationRequest } from 'modelsuite'

const client = ModelSuiteClient.fromEnv()

// Content moderation
const moderation = await client.moderateText(new ModerationRequest("Is this appropriate?"))
console.log(`Flagged: ${moderation.flagged}`)

// Text classification
const classification = await client.classifyText(
    new ClassificationRequest("Great product!", ["positive", "negative", "neutral"])
)
console.log(`Sentiment: ${classification.top()?.label}`)
```

## Streaming

### Python

```python
for chunk in client.complete_stream(request.with_streaming()):
    if chunk.text:
        print(chunk.text, end="", flush=True)
```

### Node.js

```typescript
const stream = await client.stream(request.withStreaming())
let chunk
while ((chunk = await stream.next()) !== null) {
    if (chunk.text) process.stdout.write(chunk.text)
    if (chunk.isDone) break
}
```

### Rust

```rust
use futures::StreamExt;

let mut stream = client.complete_stream(request).await?;
while let Some(chunk) = stream.next().await {
    if let Some(text) = chunk?.text() {
        print!("{}", text);
    }
}
```

## Tool Calling

### Python

```python
from modelsuite import ToolBuilder

weather_tool = ToolBuilder("get_weather") \
    .description("Get current weather") \
    .string_param("city", "City name") \
    .build()

request = CompletionRequest(...).with_tools([weather_tool])
```

### Node.js

```typescript
const weatherTool = new ToolBuilder('get_weather')
    .description('Get current weather')
    .stringParam('city', 'City name')
    .build()

const request = CompletionRequest.create(...).withTools([weatherTool])
```

## Providers

ModelSuite supports 70+ LLM providers:

| Category | Providers |
|----------|-----------|
| **Core** | Anthropic, OpenAI, Azure OpenAI |
| **Cloud** | AWS Bedrock, Google Vertex AI (Google, Anthropic, DeepSeek, Meta Llama, Mistral, AI21), Google AI (Gemini) |
| **Fast Inference** | Groq, Mistral, Cerebras, SambaNova, Fireworks, DeepSeek |
| **Enterprise** | Cohere, AI21 |
| **OpenAI-Compatible** | xAI (Grok), Meta Llama, Lambda Labs, Friendli, Volcengine |
| **Hosted** | Together, Perplexity, Anyscale, DeepInfra, Novita, Hyperbolic |
| **Platforms** | HuggingFace, Replicate, Baseten, RunPod |
| **Cloud ML** | Cloudflare, WatsonX, Databricks, DataRobot |
| **Local** | Ollama, LM Studio, vLLM, TGI, Llamafile |
| **Regional** | YandexGPT, GigaChat, Clova, Maritaca, Mistral (EU) |
| **Real-Time Voice** | Deepgram (v3), ElevenLabs, OpenAI Realtime |
| **Video Generation** | Runware (5+ models), DiffusionRouter (planned) |
| **Domain-Specific** | Med-PaLM 2 (medical), DeepSeek-R1 (scientific) |
| **Specialized** | Stability AI, Voyage, Jina, Fal |
| **Contingent (API Pending)** | LightOn (France), LatamGPT (Brazil), ChatLAW (Legal), Grok Realtime (xAI) |

### Environment Variables

Each provider is auto-detected from environment variables:

```bash
# Core providers
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Azure OpenAI
export AZURE_OPENAI_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
export AZURE_OPENAI_DEPLOYMENT=gpt-4

# AWS Bedrock
export AWS_REGION=us-east-1

# Google
export GOOGLE_API_KEY=...  # For Gemini
export GOOGLE_CLOUD_PROJECT=...  # For Vertex AI
export VERTEX_LOCATION=us-central1

# Fast inference
export GROQ_API_KEY=...
export MISTRAL_API_KEY=...
export CEREBRAS_API_KEY=...
export DEEPSEEK_API_KEY=...

# And 30+ more...
```

## Rust Features

Enable only the providers you need:

```toml
[dependencies]
# Just Anthropic
modelsuite = { version = "0.1", features = ["anthropic"] }

# Multiple providers
modelsuite = { version = "0.1", features = ["anthropic", "openai", "groq"] }

# All providers
modelsuite = { version = "0.1", features = ["all-providers"] }
```

| Feature | Description |
|---------|-------------|
| `anthropic` | Anthropic Claude (default) |
| `openai` | OpenAI GPT (default) |
| `azure` | Azure OpenAI |
| `bedrock` | AWS Bedrock |
| `vertex` | Google Vertex AI |
| `google` | Google AI (Gemini) |
| `groq` | Groq |
| `mistral` | Mistral AI |
| `cohere` | Cohere |
| `ollama` | Local Ollama |
| `all-providers` | Enable all |

## Advanced Features

### Extended Thinking (4 Providers)

```python
from modelsuite import ThinkingConfig

# Works on OpenAI, Anthropic, Google Vertex, DeepSeek
request = CompletionRequest("gemini-2.0-flash", ...).with_thinking(ThinkingConfig.enabled(5000))
response = client.complete(request)
print(response.thinking_content())  # Reasoning process
print(response.text_content())       # Final answer
```

### Regional Providers

```python
# GDPR-compliant endpoints for European markets
from modelsuite.providers import MistralRegion

request = CompletionRequest("mistral/mistral-large", ...)
# Automatically uses EU endpoint when MISTRAL_REGION=eu
```

### Real-Time Voice

```python
# Stream audio with latency control
from modelsuite.providers.audio.elevenlabs import LatencyMode

config = ElevenLabsConfig().with_latency(LatencyMode.Balanced)
# Options: LowestLatency, LowLatency, Balanced, HighQuality, HighestQuality
```

### Video Generation

```python
# Generate videos with Runware aggregator (5+ models)
response = client.complete(
    CompletionRequest("runware-video",
        messages=[Message.user("Generate a 5-second sunset video")])
)
```

### Domain-Specific Models

```python
# Medical AI with HIPAA compliance
from modelsuite.providers import VertexProvider

provider = VertexProvider.for_medical_domain(project_id, location, token)

# Scientific reasoning (71% AIME pass rate)
from modelsuite import ThinkingConfig
request = CompletionRequest("deepseek-reasoner", ...).with_thinking(
    ThinkingConfig.enabled(10000)  # Higher budget for complex problems
)
```

### Structured Output

```python
schema = {"type": "object", "properties": {...}}
request = CompletionRequest(...).with_json_schema("name", schema)
```

### Embeddings

```python
from modelsuite import EmbeddingRequest

response = client.embed(EmbeddingRequest("openai/text-embedding-3-small", "Hello"))
print(response.values())  # [0.123, -0.456, ...]
```

### Token Counting

```python
from modelsuite import TokenCountRequest

result = client.count_tokens(TokenCountRequest(model, messages))
print(result.input_tokens)
```

### Batch Processing

```python
from modelsuite import BatchRequest

batch = client.create_batch([BatchRequest("id-1", request1), ...])
results = client.get_batch_results("anthropic", batch.id)
```

## Documentation

- [Getting Started (Python)](docs/getting-started-python.md)
- [Getting Started (Node.js)](docs/getting-started-nodejs.md)
- [Getting Started (Rust)](docs/getting-started-rust.md)
- [Migration Guide (v0.0.x → v0.1.0)](MIGRATION.md)
- [Release Notes (v0.1.0)](RELEASE_NOTES.md)
- [Changelog](CHANGELOG.md)
- [Domain-Specific Models Guide](docs/domain_models.md) - Finance, Legal, Medical, Scientific
- [Scientific Benchmarks](docs/scientific_benchmarks.md) - Reasoning model performance
- [Models Registry](docs/MODELS_REGISTRY.md) - Complete provider/model reference
- [Examples](examples/)

## Examples

See the [`examples/`](examples/) directory:

- `examples/python/` - Python examples
- `examples/nodejs/` - Node.js/TypeScript examples
- `examples/` - Rust examples (run with `cargo run --example name`)

## License

MIT OR Apache-2.0
