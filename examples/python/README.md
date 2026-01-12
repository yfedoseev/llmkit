# LLMKit Python Examples

Examples demonstrating LLMKit usage in Python.

## Setup

```bash
# From this directory
uv sync

# Or using pip (installs as llmkit-python)
pip install -e ../../llmkit-python
```

## Running Examples

```bash
# Set your API keys
export ANTHROPIC_API_KEY=your-key
export OPENAI_API_KEY=your-key

# Run an example
uv run python 01_simple_completion.py
```

## Examples

| File | Description |
|------|-------------|
| `01_simple_completion.py` | Basic completion request |
| `02_streaming.py` | Streaming responses |
| `03_tool_calling.py` | Function/tool calling |
| `04_vision.py` | Image input with vision models |
| `05_structured_output.py` | JSON schema outputs |
| `06_extended_thinking.py` | Claude extended thinking |
| `07_multiple_providers.py` | Using multiple providers |
| `08_error_handling.py` | Error handling patterns |
| `09_async_usage.py` | Async client usage |
| `10_batch_processing.py` | Batch API |
| `11_embeddings.py` | Text embeddings |
| `12_audio_synthesis.py` | Text-to-speech with ElevenLabs |
| `13_audio_transcription.py` | Speech-to-text with Deepgram |
| `14_image_generation.py` | Image generation with DALL-E |
| `15_specialized_api.py` | Specialized APIs (ranking, moderation) |
| `16_video_generation.py` | Video generation |
| `17_response_caching.py` | Client-side response caching |
| `18_retry_resilience.py` | Retry configuration and resilience |
| `19_openai_compatible.py` | OpenAI-compatible provider |
