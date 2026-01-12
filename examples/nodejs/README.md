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
| `09-batch-processing.ts` | Batch API |
| `10-embeddings.ts` | Text embeddings |
| `11-audio-synthesis.ts` | Text-to-speech with ElevenLabs |
| `12-audio-transcription.ts` | Speech-to-text with Deepgram |
| `13-image-generation.ts` | Image generation with DALL-E |
| `14-specialized-api.ts` | Specialized APIs (ranking, moderation) |
| `15-video-generation.ts` | Video generation |
| `16-response-caching.ts` | Client-side response caching |
| `17-retry-resilience.ts` | Retry configuration and resilience |
| `18-openai-compatible.ts` | OpenAI-compatible provider |
