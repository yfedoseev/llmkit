# Image Generation API

Generate images from text prompts using various providers including DALL-E, Stable Diffusion, and more.

## Quick Start

### Python

```python
from llmkit import LLMKitClient, ImageGenerationRequest, ImageSize

client = LLMKitClient.from_env()

# Generate an image
request = ImageGenerationRequest("dall-e-3", "A serene mountain landscape")
response = client.generate_image(request)

print(f"Generated {response.count} image(s)")
print(f"Image URL: {response.first().url}")
```

### TypeScript

```typescript
import { LLMKitClient, ImageGenerationRequest, ImageSize } from 'llmkit'

const client = LLMKitClient.fromEnv()

// Generate an image
const request = new ImageGenerationRequest('dall-e-3', 'A serene mountain landscape')
const response = await client.generateImage(request)

console.log(`Generated ${response.count} image(s)`)
console.log(`Image URL: ${response.first()?.url}`)
```

## Supported Models

### OpenAI (DALL-E)

| Model | ID | Max Images | Features |
|-------|----|----|----------|
| DALL-E 3 | `dall-e-3` | 1 | Photorealistic, style control |
| DALL-E 2 | `dall-e-2` | 10 | Fast, flexible |

### FAL AI

| Model | ID | Features |
|-------|----|----|
| FLUX Dev | `fal-ai/flux/dev` | High quality, flexible |
| FLUX Schnell | `fal-ai/flux/schnell` | Fast, real-time |
| Stable Diffusion 3 | `fal-ai/stable-diffusion-3` | Latest SD model |

### Stability AI

| Model | ID | Features |
|-------|----|----|
| SDXL | `stability-ai/stable-diffusion-xl` | Highly capable |
| Stable Diffusion 3 | `stability-ai/stable-diffusion-3` | Latest release |

### Recraft

| Model | ID | Features |
|-------|----|----|
| Recraft V3 | `recraft-v3` | Vector/design focused |

### RunwayML

Multiple models for image generation and manipulation.

## API Reference

### ImageGenerationRequest

Request object for image generation.

#### Constructor

**Python:**
```python
request = ImageGenerationRequest(model: str, prompt: str)
```

**TypeScript:**
```typescript
const request = new ImageGenerationRequest(model: string, prompt: string)
```

#### Builder Methods

##### with_n(count: int)

Set number of images to generate (1-10).

```python
request = request.with_n(4)
```

##### with_size(size: ImageSize)

Set image dimensions.

**Available sizes:**
- `ImageSize.Square256` - 256x256
- `ImageSize.Square512` - 512x512
- `ImageSize.Square1024` - 1024x1024 (default)
- `ImageSize.Portrait1024x1792` - 1024x1792
- `ImageSize.Landscape1792x1024` - 1792x1024

```python
request = request.with_size(ImageSize.Landscape1792x1024)
```

##### with_quality(quality: ImageQuality)

Set image quality (DALL-E 3 only).

```python
request = request.with_quality(ImageQuality.Hd)
```

**Options:**
- `ImageQuality.Standard` - Faster, cheaper
- `ImageQuality.Hd` - More detail

##### with_style(style: ImageStyle)

Set image style (DALL-E 3 only).

```python
request = request.with_style(ImageStyle.Vivid)
```

**Options:**
- `ImageStyle.Natural` - Realistic
- `ImageStyle.Vivid` - Dramatic, vivid

##### with_format(format: ImageFormat)

Set response format.

```python
request = request.with_format(ImageFormat.B64Json)
```

**Options:**
- `ImageFormat.Url` - Return download URL
- `ImageFormat.B64Json` - Return base64-encoded data

##### with_negative_prompt(prompt: str)

Describe what NOT to include (Stability AI models).

```python
request = request.with_negative_prompt("blurry, low quality, watermark")
```

##### with_seed(seed: int)

Set seed for reproducible results.

```python
request = request.with_seed(12345)
```

#### Complete Example

**Python:**
```python
request = (
    ImageGenerationRequest("dall-e-3", "A Renaissance painting style portrait")
    .with_n(1)
    .with_size(ImageSize.Landscape1792x1024)
    .with_quality(ImageQuality.Hd)
    .with_style(ImageStyle.Vivid)
    .with_format(ImageFormat.Url)
)
```

**TypeScript:**
```typescript
const request = new ImageGenerationRequest(
  'dall-e-3',
  'A Renaissance painting style portrait'
)
  .with_n(1)
  .with_size(ImageSize.Landscape1792x1024)
  .with_quality(ImageQuality.Hd)
  .with_style(ImageStyle.Vivid)
  .with_format(ImageFormat.Url)
```

### ImageGenerationResponse

Response object containing generated images.

#### Properties

- `created` (int): Unix timestamp of creation
- `images` (List[GeneratedImage]): Array of generated images
- `count` (int, read-only): Number of images generated
- `total_size` (int, read-only): Total size of all image data

#### Methods

##### first()

Get the first image (convenience for n=1 requests).

```python
image = response.first()
if image:
    print(f"URL: {image.url}")
```

### GeneratedImage

Individual generated image.

#### Properties

- `url` (str, optional): Download URL for the image
- `b64_json` (str, optional): Base64-encoded image data
- `revised_prompt` (str, optional): Prompt used by the model

#### Methods

##### from_url(url: str)

Create image from URL (static method).

```python
image = GeneratedImage.from_url("https://example.com/image.png")
```

##### from_b64(data: str)

Create image from base64 data (static method).

```python
image = GeneratedImage.from_b64(b64_encoded_data)
```

##### with_revised_prompt(prompt: str)

Set the revised prompt.

```python
image = image.with_revised_prompt("Updated prompt description")
```

## LLMKitClient.generate_image()

Generate images from a text prompt.

#### Signature

**Python:**
```python
def generate_image(self, request: ImageGenerationRequest) -> ImageGenerationResponse
```

**TypeScript:**
```typescript
async generateImage(request: ImageGenerationRequest): Promise<ImageGenerationResponse>
```

#### Example

**Python:**
```python
request = ImageGenerationRequest("dall-e-3", "A sunset over mountains")
response = client.generate_image(request)
```

**TypeScript:**
```typescript
const request = new ImageGenerationRequest('dall-e-3', 'A sunset over mountains')
const response = await client.generateImage(request)
```

## Accessing Image Data

### Option 1: Via URL

```python
response = client.generate_image(request)
if response.first()?.url:
    import requests
    r = requests.get(response.first().url)
    with open("image.png", "wb") as f:
        f.write(r.content)
```

### Option 2: Via Base64

```python
response = client.generate_image(request.with_format(ImageFormat.B64Json))
if response.first()?.b64_json:
    import base64
    image_data = base64.b64decode(response.first().b64_json)
    with open("image.png", "wb") as f:
        f.write(image_data)
```

## Resolution Guide

| Name | Dimensions | Use Case |
|------|-----------|----------|
| Square 256 | 256x256 | Thumbnails, icons |
| Square 512 | 512x512 | Small avatars |
| Square 1024 | 1024x1024 | Standard quality |
| Portrait | 1024x1792 | Character portraits, posters |
| Landscape | 1792x1024 | Wide scenes, backgrounds |

## Cost and Speed Trade-offs

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| FLUX Schnell | Very Fast | Good | Low | Real-time apps |
| DALL-E 2 | Fast | Good | Low | General use |
| DALL-E 3 Standard | Medium | Excellent | Medium | Quality |
| DALL-E 3 HD | Slower | Premium | High | High-end |
| Stable Diffusion 3 | Medium | Excellent | Low | Open-source |

## Complete Working Examples

### Python: Save Generated Images

```python
from llmkit import LLMKitClient, ImageGenerationRequest, ImageSize, ImageQuality
import base64
from pathlib import Path

def generate_and_save_images(prompts: list, output_dir: str = "generated_images"):
    client = LLMKitClient.from_env()
    Path(output_dir).mkdir(exist_ok=True)

    for i, prompt in enumerate(prompts):
        request = (
            ImageGenerationRequest("dall-e-3", prompt)
            .with_size(ImageSize.Square1024)
            .with_quality(ImageQuality.Hd)
        )

        response = client.generate_image(request)

        if response.first()?.b64_json:
            image_data = base64.b64decode(response.first().b64_json)
            filepath = Path(output_dir) / f"image_{i}.png"
            filepath.write_bytes(image_data)
            print(f"✓ Saved: {filepath}")

# Usage
prompts = [
    "A serene mountain landscape",
    "A futuristic city at night",
    "A peaceful forest with sunlight",
]
generate_and_save_images(prompts)
```

### TypeScript: Batch Generation with Retry

```typescript
import { LLMKitClient, ImageGenerationRequest, ImageSize } from 'llmkit'
import * as fs from 'fs'

async function generateWithRetry(
  client: LLMKitClient,
  prompt: string,
  maxRetries: number = 3
) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const request = new ImageGenerationRequest('dall-e-3', prompt).with_size(
        ImageSize.Square1024
      )
      const response = await client.generateImage(request)
      return response
    } catch (error) {
      console.error(`Attempt ${attempt} failed:`, error)
      if (attempt < maxRetries) {
        await sleep(1000 * attempt) // Exponential backoff
      } else {
        throw error
      }
    }
  }
}

async function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// Usage
const client = LLMKitClient.fromEnv()
const response = await generateWithRetry(client, 'A cat in space')
```

## Error Handling

### Python

```python
try:
    request = ImageGenerationRequest("dall-e-3", "Generate image")
    response = client.generate_image(request)
except Exception as e:
    print(f"Image generation failed: {e}")
    if "quota" in str(e).lower():
        print("Rate limit reached, try again later")
```

### TypeScript

```typescript
try {
  const request = new ImageGenerationRequest('dall-e-3', 'Generate image')
  const response = await client.generateImage(request)
} catch (error) {
  console.error(`Image generation failed: ${(error as Error).message}`)
  if ((error as Error).message.includes('quota')) {
    console.log('Rate limit reached, try again later')
  }
}
```

## Environment Variables

Configure provider-specific API keys:

- `OPENAI_API_KEY`: For DALL-E models
- `FAL_API_KEY`: For Fal.ai Flux models
- `STABILITY_API_KEY`: For Stability AI models
- `REPLICATE_API_TOKEN`: For Replicate models

## Tips & Best Practices

### Writing Good Prompts

**Effective prompt:**
> "A serene Japanese Zen garden with a wooden bridge over a koi pond, cherry blossom trees, stone lanterns, morning mist, peaceful atmosphere, professional photography, high detail"

**Less effective:**
> "garden"

### Quality Settings

- Use `ImageQuality.Hd` for final products
- Use `ImageQuality.Standard` for iterations and testing
- HD quality takes longer and costs more

### Caching Results

For frequently requested images, cache the URLs:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def generate_cached_image(prompt: str):
    request = ImageGenerationRequest("dall-e-3", prompt)
    response = client.generate_image(request)
    return response.first().url
```

### Batch Processing

When generating many images, add delays:

```python
import time

for prompt in large_prompt_list:
    request = ImageGenerationRequest("dall-e-3", prompt)
    response = client.generate_image(request)
    time.sleep(2)  # Respect rate limits
```

## Troubleshooting

### No Image Data in Response

**Issue:** Both `url` and `b64_json` are None

**Solutions:**
- Verify the model name is correct
- Check your API quota
- Verify API key has image generation permissions

### "Quota Exceeded" Error

**Issue:** Rate limiting or quota limits

**Solutions:**
- Implement retry logic with exponential backoff
- Add delays between requests
- Use Standard quality instead of HD

### Image Quality Issues

**Issue:** Generated images are low quality or blurry

**Solutions:**
- Use more descriptive prompts
- Set `ImageQuality.Hd` for better quality
- Try a different model (DALL-E 3 vs FLUX)
- Adjust style (Vivid vs Natural)

## See Also

- [Audio API](./audio-api.md) - Speech recognition and synthesis
- [Video API](./video-api.md) - Video generation
- [Complete API Reference](../README.md)
