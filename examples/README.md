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
cargo run --example 01_simple_completion
cargo run --example 02_streaming
cargo run --example 03_tool_calling
# ... etc
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
export OPENROUTER_API_KEY=your-openrouter-key
export GROQ_API_KEY=your-groq-key
export ELEVENLABS_API_KEY=your-elevenlabs-key
export DEEPGRAM_API_KEY=your-deepgram-key
export COHERE_API_KEY=your-cohere-key
# ... other providers as needed
```

## Example Topics

19 examples are available. Rust has examples 01-10, Python and TypeScript have all 19.

| # | Topic | Rust | Python | TypeScript |
|---|-------|------|--------|------------|
| 01 | Simple Completion | `01_simple_completion.rs` | `01_simple_completion.py` | `01-simple-completion.ts` |
| 02 | Streaming | `02_streaming.rs` | `02_streaming.py` | `02-streaming.ts` |
| 03 | Tool Calling | `03_tool_calling.rs` | `03_tool_calling.py` | `03-tool-calling.ts` |
| 04 | Vision | `04_vision.rs` | `04_vision.py` | `04-vision.ts` |
| 05 | Structured Output | `05_structured_output.rs` | `05_structured_output.py` | `05-structured-output.ts` |
| 06 | Extended Thinking | `06_extended_thinking.rs` | `06_extended_thinking.py` | `06-extended-thinking.ts` |
| 07 | Multiple Providers | `07_multiple_providers.rs` | `07_multiple_providers.py` | `07-multiple-providers.ts` |
| 08 | Error Handling | `08_error_handling.rs` | `08_error_handling.py` | `08-error-handling.ts` |
| 09 | Async Usage | `09_async_usage.rs` | `09_async_usage.py` | `09-async-usage.ts` |
| 10 | Batch Processing | `10_batch_processing.rs` | `10_batch_processing.py` | `10-batch-processing.ts` |
| 11 | Embeddings | - | `11_embeddings.py` | `11-embeddings.ts` |
| 12 | Audio Synthesis | - | `12_audio_synthesis.py` | `12-audio-synthesis.ts` |
| 13 | Audio Transcription | - | `13_audio_transcription.py` | `13-audio-transcription.ts` |
| 14 | Image Generation | - | `14_image_generation.py` | `14-image-generation.ts` |
| 15 | Specialized APIs | - | `15_specialized_api.py` | `15-specialized-api.ts` |
| 16 | Video Generation | - | `16_video_generation.py` | `16-video-generation.ts` |
| 17 | Response Caching | - | `17_response_caching.py` | `17-response-caching.ts` |
| 18 | Retry Resilience | - | `18_retry_resilience.py` | `18-retry-resilience.ts` |
| 19 | OpenAI Compatible | - | `19_openai_compatible.py` | `19-openai-compatible.ts` |

**Note:** Rust examples 11-19 will be added as the corresponding APIs are implemented in the Rust crate.
