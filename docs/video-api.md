# Video Generation API

Generate videos from text prompts using various providers and models.

## Quick Start

### Python

```python
from llmkit import LLMKitClient, VideoGenerationRequest

client = LLMKitClient.from_env()

# Generate a video from a text prompt
request = VideoGenerationRequest("A serene landscape with mountains")
response = client.generate_video(request)

print(f"Video URL: {response.video_url}")
print(f"Task ID: {response.task_id}")
print(f"Status: {response.status}")
```

### TypeScript

```typescript
import { LLMKitClient, VideoGenerationRequest } from 'llmkit'

const client = LLMKitClient.fromEnv()

// Generate a video from a text prompt
const request = new VideoGenerationRequest('A serene landscape with mountains')
const response = await client.generateVideo(request)

console.log(`Video URL: ${response.video_url}`)
console.log(`Task ID: ${response.task_id}`)
console.log(`Status: ${response.status}`)
```

## Supported Models

### Runware Provider

Runware provides access to multiple video generation models:

| Model | ID | Output Length | Capabilities |
|-------|----|----|---|
| Runway Gen-4 | `runway-gen-4` | Up to 30s | High quality, photorealistic |
| Kling 2.0 | `kling-2.0` | Up to 10s | Fast, detailed |
| Pika 1.0 | `pika-1.0` | Up to 10s | Smooth animation |
| Hailuo Mini | `hailuo-mini` | Up to 10s | Cost-effective |
| Leonardo Ultra | `leonardo-ultra` | Up to 15s | Premium quality |

### DiffusionRouter Provider

**Status:** Coming February 2026

- Stable Diffusion Video models
- Open-source video generation

## API Reference

### VideoGenerationRequest

Request object for video generation.

#### Constructor

**Python:**
```python
request = VideoGenerationRequest(prompt: str)
```

**TypeScript:**
```typescript
const request = new VideoGenerationRequest(prompt: string)
```

#### Parameters

- `prompt` (string, required): Description of the video to generate

#### Builder Methods

All builder methods return a new instance with the specified option set (fluent API).

##### with_model(model: str)

Set the video generation model to use.

**Python:**
```python
request = request.with_model("runway-gen-4")
```

**TypeScript:**
```typescript
request = request.with_model('runway-gen-4')
```

##### with_duration(seconds: int)

Set the video duration in seconds.

**Python:**
```python
request = request.with_duration(10)  # 10 seconds
```

**TypeScript:**
```typescript
request = request.with_duration(10)  // 10 seconds
```

##### with_width(pixels: int)

Set the video width in pixels.

**Python:**
```python
request = request.with_width(1920)  # Full HD width
```

**TypeScript:**
```typescript
request = request.with_width(1920)  // Full HD width
```

##### with_height(pixels: int)

Set the video height in pixels.

**Python:**
```python
request = request.with_height(1080)  # Full HD height
```

**TypeScript:**
```typescript
request = request.with_height(1080)  // Full HD height
```

#### Example

**Python:**
```python
request = (
    VideoGenerationRequest("A person dancing")
    .with_model("runway-gen-4")
    .with_duration(10)
    .with_width(1280)
    .with_height(720)
)
```

**TypeScript:**
```typescript
const request = new VideoGenerationRequest('A person dancing')
  .with_model('runway-gen-4')
  .with_duration(10)
  .with_width(1280)
  .with_height(720)
```

### VideoGenerationResponse

Response object containing video generation results.

#### Properties

- `video_url` (string, optional): URL to the generated video
- `video_bytes` (bytes, optional): Raw video data
- `format` (string): Video format (e.g., "mp4", "mov")
- `duration` (float, optional): Video duration in seconds
- `width` (int, optional): Video width in pixels
- `height` (int, optional): Video height in pixels
- `task_id` (string, optional): Task ID for async operations
- `status` (string, optional): Current status ("processing", "completed", "failed")
- `size` (int, read-only): Size of video data in bytes

#### Example

**Python:**
```python
response = client.generate_video(request)

print(f"Video URL: {response.video_url}")
print(f"Format: {response.format}")
print(f"Duration: {response.duration}s")
print(f"Resolution: {response.width}x{response.height}")
print(f"File size: {response.size} bytes")
print(f"Task ID: {response.task_id}")
print(f"Status: {response.status}")
```

**TypeScript:**
```typescript
const response = await client.generateVideo(request)

console.log(`Video URL: ${response.video_url}`)
console.log(`Format: ${response.format}`)
console.log(`Duration: ${response.duration}s`)
console.log(`Resolution: ${response.width}x${response.height}`)
console.log(`File size: ${response.size} bytes`)
console.log(`Task ID: ${response.task_id}`)
console.log(`Status: ${response.status}`)
```

### LLMKitClient.generate_video()

Generate a video from a text prompt.

#### Signature

**Python:**
```python
def generate_video(self, request: VideoGenerationRequest) -> VideoGenerationResponse
```

**TypeScript:**
```typescript
async generateVideo(request: VideoGenerationRequest): Promise<VideoGenerationResponse>
```

#### Returns

- `VideoGenerationResponse`: Object containing video generation results

#### Example

**Python:**
```python
request = VideoGenerationRequest("A serene landscape")
response = client.generate_video(request)
```

**TypeScript:**
```typescript
const request = new VideoGenerationRequest('A serene landscape')
const response = await client.generateVideo(request)
```

## Async Operations & Polling

Video generation is often asynchronous. The response includes a `task_id` and `status` that can be used to monitor progress.

### Status Values

- `"processing"`: Video is being generated
- `"completed"`: Video generation finished successfully
- `"failed"`: Video generation failed

### Polling Pattern

**Python:**
```python
import time

request = VideoGenerationRequest("Generate a video")
response = client.generate_video(request)

task_id = response.task_id
max_wait = 300  # 5 minutes

start_time = time.time()
while time.time() - start_time < max_wait:
    # In production, call client.poll_video(task_id)
    # response = client.poll_video(task_id)

    if response.status == "completed":
        print(f"Video ready: {response.video_url}")
        break
    elif response.status == "failed":
        print("Video generation failed")
        break

    print(f"Status: {response.status}, waiting...")
    time.sleep(5)
```

**TypeScript:**
```typescript
async function pollVideoGeneration(
  taskId: string,
  maxWaitMs: number = 300000
): Promise<string> {
  const startTime = Date.now()

  while (Date.now() - startTime < maxWaitMs) {
    // In production, call client.pollVideo(taskId)
    // const response = await client.pollVideo(taskId)

    // For now, simulate with sleep
    await sleep(5000)

    // Check response.status
    return 'https://example.com/video.mp4'
  }

  throw new Error('Video generation timeout')
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}
```

## Output Formats

### Default Format

Videos are returned in **MP4** format by default, which is widely compatible.

### Accessing Video Data

#### Option 1: Via URL

```python
response = client.generate_video(request)
if response.video_url:
    # Download from URL
    import requests
    r = requests.get(response.video_url)
    with open("video.mp4", "wb") as f:
        f.write(r.content)
```

#### Option 2: Via Raw Bytes

```python
response = client.generate_video(request)
if response.video_bytes:
    with open("video.mp4", "wb") as f:
        f.write(response.video_bytes)
```

## Resolution Guide

| Format | Resolution | Use Case |
|--------|-----------|----------|
| 576p | 1024x576 | Mobile, social media |
| HD | 1280x720 | Standard social media |
| Full HD | 1920x1080 | High quality, YouTube |
| 4K | 3840x2160 | Premium content |

## Duration Limits

Maximum duration varies by model:

- **Runway Gen-4**: Up to 30 seconds
- **Kling 2.0**: Up to 10 seconds
- **Pika 1.0**: Up to 10 seconds
- **Hailuo Mini**: Up to 10 seconds
- **Leonardo Ultra**: Up to 15 seconds

## Complete Working Examples

### Python: Generate and Save Video

```python
from llmkit import LLMKitClient, VideoGenerationRequest

def generate_and_save_video(prompt: str, output_file: str):
    client = LLMKitClient.from_env()

    request = VideoGenerationRequest(prompt).with_model("runway-gen-4")
    response = client.generate_video(request)

    if response.video_bytes:
        with open(output_file, "wb") as f:
            f.write(response.video_bytes)
        print(f"✓ Video saved to {output_file}")
    elif response.video_url:
        import requests
        r = requests.get(response.video_url)
        with open(output_file, "wb") as f:
            f.write(r.content)
        print(f"✓ Video downloaded to {output_file}")
    else:
        print("✗ No video data available")

# Usage
generate_and_save_video("A cat playing with yarn", "cat_video.mp4")
```

### TypeScript: Generate Video with Polling

```typescript
import { LLMKitClient, VideoGenerationRequest } from 'llmkit'
import * as fs from 'fs'

async function generateVideoWithPolling(
  prompt: string,
  maxWaitSeconds: number = 300
): Promise<string> {
  const client = LLMKitClient.fromEnv()

  const request = new VideoGenerationRequest(prompt)
    .with_model('runway-gen-4')
    .with_duration(10)

  const response = await client.generateVideo(request)
  console.log(`Task ID: ${response.task_id}`)
  console.log(`Initial status: ${response.status}`)

  // Poll for completion (in production)
  const startTime = Date.now()
  while (Date.now() - startTime < maxWaitSeconds * 1000) {
    if (response.status === 'completed') {
      console.log(`✓ Video ready: ${response.video_url}`)
      return response.video_url || ''
    }

    if (response.status === 'failed') {
      throw new Error('Video generation failed')
    }

    console.log(`Status: ${response.status}...`)
    await sleep(5000)
  }

  throw new Error('Video generation timeout')
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms))
}

// Usage
generateVideoWithPolling('A beautiful sunset over the ocean')
  .then(url => console.log(`Video: ${url}`))
  .catch(err => console.error(`Error: ${err.message}`))
```

## Error Handling

### Python

```python
try:
    request = VideoGenerationRequest("Generate a video")
    response = client.generate_video(request)
except Exception as e:
    print(f"Video generation failed: {e}")
```

### TypeScript

```typescript
try {
  const request = new VideoGenerationRequest('Generate a video')
  const response = await client.generateVideo(request)
} catch (error) {
  console.error(`Video generation failed: ${(error as Error).message}`)
}
```

## Environment Variables

### Required

- `LLMKIT_API_KEY`: Your LLMKit API key

### Optional

- `LLMKIT_VIDEO_PROVIDER`: Override default video provider
- `LLMKIT_VIDEO_TIMEOUT`: Video generation timeout in seconds

### Example

```bash
export LLMKIT_API_KEY="your-api-key"
export LLMKIT_VIDEO_TIMEOUT="600"  # 10 minutes
```

## Tips & Best Practices

### Prompts

Write detailed, descriptive prompts for better results:

**Good:**
> "A serene Japanese garden with a koi pond, cherry blossom trees, stone bridge, soft morning light, peaceful atmosphere"

**Not ideal:**
> "garden video"

### Model Selection

- **Runway Gen-4**: Best for photorealistic, high-quality output
- **Kling 2.0**: Fast generation, good for quick turnaround
- **Pika 1.0**: Smooth animations and movements
- **Leonardo Ultra**: Premium quality, most detailed

### Resolution Trade-offs

| Resolution | Processing Time | Quality | File Size |
|-----------|---|---|---|
| 576p | Fast | Standard | Small |
| 720p | Medium | Good | Medium |
| 1080p | Slower | High | Large |

### Rate Limiting

Video generation is computationally expensive. Implement appropriate delays between requests:

**Python:**
```python
import time

for prompt in prompts:
    request = VideoGenerationRequest(prompt)
    response = client.generate_video(request)
    time.sleep(10)  # 10 second delay between requests
```

**TypeScript:**
```typescript
for (const prompt of prompts) {
  const request = new VideoGenerationRequest(prompt)
  await client.generateVideo(request)
  await sleep(10000)  // 10 second delay
}
```

## Troubleshooting

### No Video Data in Response

**Issue:** `response.video_bytes` and `response.video_url` are both None/undefined

**Solutions:**
- Check that video generation completed (status == "completed")
- Verify task_id and use polling to wait for completion
- Check API quotas and rate limits

### Task Remains "Processing"

**Issue:** Video generation appears stuck

**Solutions:**
- Increase polling timeout
- Check for provider-specific limits
- Verify API key permissions

### Invalid Model Name

**Issue:** "Model not found" error

**Solutions:**
- Use model IDs from the Supported Models table above
- Check for typos in model names

## See Also

- [Audio API](./audio-api.md) - Speech recognition and synthesis
- [Image API](./image-api.md) - Image generation (coming soon)
- [Complete API Reference](../README.md)
