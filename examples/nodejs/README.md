# LLMKit TypeScript/Node.js Examples

Examples demonstrating LLMKit usage in TypeScript/Node.js.

## Setup

```bash
# From this directory
npm install

# Build the local llmkit-node package first
cd ../../llmkit-node && npm run build && cd ../examples/nodejs
```

## Running Examples

```bash
# Set your API keys
export ANTHROPIC_API_KEY=your-key
export OPENAI_API_KEY=your-key
export OPENROUTER_API_KEY=your-key

# Run an example
npx ts-node 01-simple-completion.ts

# Or use the npm scripts
npm run example:simple
npm run example:streaming
```

## Examples

| File | Description |
|------|-------------|
| `01-simple-completion.ts` | Basic completion request |
| `02-streaming.ts` | Streaming responses |
| `03-tool-calling.ts` | Function/tool calling |
| `04-vision.ts` | Image input with vision models |
| `05-structured-output.ts` | JSON schema outputs |
| `06-extended-thinking.ts` | Claude extended thinking |
| `07-multiple-providers.ts` | Using multiple providers |
| `08-error-handling.ts` | Error handling patterns |
| `09-async-usage.ts` | Async patterns and concurrency |
| `10-batch-processing.ts` | Batch API |
| `11-embeddings.ts` | Text embeddings |
| `12-audio-synthesis.ts` | Text-to-speech with ElevenLabs |
| `13-audio-transcription.ts` | Speech-to-text with Deepgram |
| `14-image-generation.ts` | Image generation with DALL-E |
| `15-specialized-api.ts` | Specialized APIs (ranking, moderation) |
| `16-video-generation.ts` | Video generation |
| `17-response-caching.ts` | Client-side response caching |
| `18-retry-resilience.ts` | Retry configuration and resilience |
| `19-openai-compatible.ts` | OpenAI-compatible provider |
