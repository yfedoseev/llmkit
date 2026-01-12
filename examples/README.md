# LLMKit Examples

This directory contains examples demonstrating LLMKit usage across all supported languages.

## Structure

```
examples/
├── *.rs              # Rust examples (cargo examples)
├── python/           # Python examples
│   ├── pyproject.toml
│   └── *.py
└── nodejs/           # TypeScript/Node.js examples
    ├── package.json
    └── *.ts
```

## Rust Examples

Rust examples are configured as cargo examples and use the local crate.

```bash
# From the repository root
cargo run --example simple_completion --features anthropic
cargo run --example streaming --features openai
cargo run --example tool_calling --features anthropic
cargo run --example vision --features anthropic
cargo run --example structured_output --features openai
cargo run --example multiple_providers --features "anthropic,openai"
```

## Python Examples

See [python/README.md](python/README.md) for setup and usage.

```bash
cd examples/python
uv sync
uv run python 01_simple_completion.py
```

## TypeScript/Node.js Examples

See [nodejs/README.md](nodejs/README.md) for setup and usage.

```bash
cd examples/nodejs
npm install
npx ts-node 01-simple-completion.ts
```

## Prerequisites

Set the appropriate API keys as environment variables:

```bash
export ANTHROPIC_API_KEY=your-anthropic-key
export OPENAI_API_KEY=your-openai-key
export ELEVENLABS_API_KEY=your-elevenlabs-key  # For audio examples
# ... other providers as needed
```

## Example Topics

| Topic | Rust | Python | TypeScript |
|-------|------|--------|------------|
| Simple Completion | `simple_completion.rs` | `01_simple_completion.py` | `01-simple-completion.ts` |
| Streaming | `streaming.rs` | `02_streaming.py` | `02-streaming.ts` |
| Tool Calling | `tool_calling.rs` | `03_tool_calling.py` | `03-tool-calling.ts` |
| Vision | `vision.rs` | `04_vision.py` | `04-vision.ts` |
| Structured Output | `structured_output.rs` | `05_structured_output.py` | `05-structured-output.ts` |
| Extended Thinking | - | `06_extended_thinking.py` | `06-extended-thinking.ts` |
| Multiple Providers | `multiple_providers.rs` | `07_multiple_providers.py` | `07-multiple-providers.ts` |
| Error Handling | - | `08_error_handling.py` | `08-error-handling.ts` |
| Async Usage | - | `09_async_usage.py` | - |
| Batch Processing | - | `10_batch_processing.py` | `09-batch-processing.ts` |
| Embeddings | - | `11_embeddings.py` | `10-embeddings.ts` |
| Audio Synthesis | - | `12_audio_synthesis.py` | `11-audio-synthesis.ts` |
| Audio Transcription | - | `13_audio_transcription.py` | `12-audio-transcription.ts` |
| Image Generation | - | `14_image_generation.py` | `13-image-generation.ts` |
| Specialized APIs | - | `15_specialized_api.py` | `14-specialized-api.ts` |
| Video Generation | - | `16_video_generation.py` | `15-video-generation.ts` |
| Response Caching | - | `17_response_caching.py` | `16-response-caching.ts` |
| Retry Resilience | - | `18_retry_resilience.py` | `17-retry-resilience.ts` |
| OpenAI Compatible | - | `19_openai_compatible.py` | `18-openai-compatible.ts` |
