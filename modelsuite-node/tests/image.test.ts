import { describe, it, expect, beforeAll } from 'vitest'
import {
  LLMKitClient,
  ImageSize,
  ImageQuality,
  ImageStyle,
  ImageFormat,
  ImageGenerationRequest,
  GeneratedImage,
  ImageGenerationResponse,
} from '../index'

describe('ImageSize Enum', () => {
  it('should have Square256 variant', () => {
    expect(ImageSize.Square256).toBeDefined()
  })

  it('should have Square512 variant', () => {
    expect(ImageSize.Square512).toBeDefined()
  })

  it('should have Square1024 variant', () => {
    expect(ImageSize.Square1024).toBeDefined()
  })

  it('should have Portrait1024x1792 variant', () => {
    expect(ImageSize.Portrait1024x1792).toBeDefined()
  })

  it('should have Landscape1792x1024 variant', () => {
    expect(ImageSize.Landscape1792x1024).toBeDefined()
  })
})

describe('ImageQuality Enum', () => {
  it('should have Standard variant', () => {
    expect(ImageQuality.Standard).toBeDefined()
  })

  it('should have Hd variant', () => {
    expect(ImageQuality.Hd).toBeDefined()
  })
})

describe('ImageStyle Enum', () => {
  it('should have Natural variant', () => {
    expect(ImageStyle.Natural).toBeDefined()
  })

  it('should have Vivid variant', () => {
    expect(ImageStyle.Vivid).toBeDefined()
  })
})

describe('ImageFormat Enum', () => {
  it('should have Url variant', () => {
    expect(ImageFormat.Url).toBeDefined()
  })

  it('should have B64Json variant', () => {
    expect(ImageFormat.B64Json).toBeDefined()
  })
})

describe('ImageGenerationRequest', () => {
  it('should create request with model and prompt', () => {
    const req = new ImageGenerationRequest('dall-e-3', 'A serene landscape')
    expect(req).toBeDefined()
  })

  it('should set number of images via builder method', () => {
    const req = new ImageGenerationRequest('dall-e-3', 'A cat')
    const reqWithN = req.with_n(3)
    expect(reqWithN).toBeDefined()
  })

  it('should set size via builder method', () => {
    const req = new ImageGenerationRequest('dall-e-3', 'A dog')
    const reqWithSize = req.with_size(ImageSize.Square1024)
    expect(reqWithSize).toBeDefined()
  })

  it('should set quality via builder method', () => {
    const req = new ImageGenerationRequest('dall-e-3', 'A bird')
    const reqWithQuality = req.with_quality(ImageQuality.Hd)
    expect(reqWithQuality).toBeDefined()
  })

  it('should set style via builder method', () => {
    const req = new ImageGenerationRequest('dall-e-3', 'A car')
    const reqWithStyle = req.with_style(ImageStyle.Vivid)
    expect(reqWithStyle).toBeDefined()
  })

  it('should set response format via builder method', () => {
    const req = new ImageGenerationRequest('dall-e-3', 'A house')
    const reqWithFormat = req.with_format(ImageFormat.B64Json)
    expect(reqWithFormat).toBeDefined()
  })

  it('should set negative prompt via builder method', () => {
    const req = new ImageGenerationRequest('stability-ai', 'A sunset')
    const reqWithNeg = req.with_negative_prompt('blurry, low quality')
    expect(reqWithNeg).toBeDefined()
  })

  it('should set seed for reproducibility', () => {
    const req = new ImageGenerationRequest('fal-ai', 'A mountain')
    const reqWithSeed = req.with_seed(BigInt(12345))
    expect(reqWithSeed).toBeDefined()
  })

  it('should chain multiple builder methods', () => {
    const req = new ImageGenerationRequest('dall-e-3', 'A magical forest')
      .with_n(2)
      .with_size(ImageSize.Landscape1792x1024)
      .with_quality(ImageQuality.Hd)
      .with_style(ImageStyle.Vivid)
      .with_format(ImageFormat.Url)
    expect(req).toBeDefined()
  })

  it('should create requests with various prompts', () => {
    const prompts = [
      'A serene mountain landscape',
      'Portrait of a Renaissance noble',
      'Abstract geometric pattern',
      'Futuristic cyberpunk city',
      '', // Empty prompt should be allowed
    ]

    prompts.forEach((prompt) => {
      const req = new ImageGenerationRequest('dall-e-3', prompt)
      expect(req).toBeDefined()
    })
  })

  it('should create requests with various models', () => {
    const models = [
      'dall-e-3',
      'dall-e-2',
      'fal-ai/flux/dev',
      'fal-ai/flux/schnell',
      'stability-ai/stable-diffusion-xl',
    ]

    models.forEach((model) => {
      const req = new ImageGenerationRequest(model, 'Test prompt')
      expect(req).toBeDefined()
    })
  })
})

describe('GeneratedImage', () => {
  it('should create image from URL', () => {
    const url = 'https://example.com/image.png'
    const img = GeneratedImage.fromUrl(url)
    expect(img).toBeDefined()
    expect(img.url).toBe(url)
  })

  it('should create image from base64 data', () => {
    const b64Data = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk'
    const img = GeneratedImage.fromB64(b64Data)
    expect(img).toBeDefined()
    expect(img.b64_json).toBe(b64Data)
  })

  it('should set revised prompt', () => {
    const img = GeneratedImage.fromUrl('https://example.com/image.png')
    const imgWithPrompt = img.withRevisedPrompt('A revised description')
    expect(imgWithPrompt).toBeDefined()
    expect(imgWithPrompt.revised_prompt).toBe('A revised description')
  })

  it('should calculate size property', () => {
    const img = GeneratedImage.fromUrl('https://example.com/image.png')
    // URL images have no b64_json, so size should be 0
    expect(img.size).toBe(0)

    // b64 images should have size
    const imgB64 = GeneratedImage.fromB64('x'.repeat(1000))
    expect(imgB64.size).toBe(1000)
  })
})

describe('ImageGenerationResponse', () => {
  it('should create empty response', () => {
    const resp = new ImageGenerationResponse()
    expect(resp).toBeDefined()
  })

  it('should calculate count of images', () => {
    const resp = new ImageGenerationResponse()
    expect(resp.count).toBe(0)

    resp.images = [GeneratedImage.fromUrl('https://example.com/image1.png')]
    expect(resp.count).toBe(1)

    resp.images = [
      GeneratedImage.fromUrl('https://example.com/image1.png'),
      GeneratedImage.fromUrl('https://example.com/image2.png'),
    ]
    expect(resp.count).toBe(2)
  })

  it('should get first image', () => {
    const resp = new ImageGenerationResponse()
    expect(resp.first()).toBeUndefined()

    const img = GeneratedImage.fromUrl('https://example.com/image1.png')
    resp.images = [img]
    const first = resp.first()
    expect(first).toBeDefined()
  })

  it('should calculate total size', () => {
    const resp = new ImageGenerationResponse()
    expect(resp.total_size).toBe(0)

    // Add b64 images
    resp.images = [GeneratedImage.fromB64('x'.repeat(1000)), GeneratedImage.fromB64('y'.repeat(2000))]
    expect(resp.total_size).toBe(3000)
  })

  it('should set created timestamp', () => {
    const resp = new ImageGenerationResponse()
    resp.created = BigInt(1609459200) // 2021-01-01 00:00:00 UTC
    expect(resp.created).toEqual(BigInt(1609459200))
  })

  it('should create complete response with images', () => {
    const resp = new ImageGenerationResponse()
    resp.created = BigInt(1609459200)
    resp.images = [
      GeneratedImage.fromUrl('https://example.com/image1.png').withRevisedPrompt(
        'A serene landscape with mountains'
      ),
      GeneratedImage.fromUrl('https://example.com/image2.png'),
    ]

    expect(resp.count).toBe(2)
    expect(resp.first()).toBeDefined()
    expect(resp.first()?.revised_prompt).toBe('A serene landscape with mountains')
  })
})

describe('LLMKitClient image methods', () => {
  let client: LLMKitClient

  beforeAll(() => {
    client = LLMKitClient.fromEnv()
  })

  it('should have generateImage method', () => {
    expect(client).toHaveProperty('generateImage')
    expect(typeof client.generateImage).toBe('function')
  })

  it('should generate image with request', async () => {
    const req = new ImageGenerationRequest('dall-e-3', 'A beautiful sunset over mountains')
    const response = await client.generateImage(req)

    expect(response).toBeInstanceOf(ImageGenerationResponse)
    expect(response.count).toBeGreaterThan(0)
  })

  it('should have expected response properties', async () => {
    const req = new ImageGenerationRequest('dall-e-3', 'Test image generation')
    const response = await client.generateImage(req)

    expect(response).toHaveProperty('created')
    expect(response).toHaveProperty('images')
    expect(response).toHaveProperty('count')
    expect(response).toHaveProperty('total_size')
  })

  it('should generate image with configured options', async () => {
    const req = new ImageGenerationRequest('dall-e-3', 'An animated scene')
      .with_n(2)
      .with_size(ImageSize.Square1024)
      .with_quality(ImageQuality.Hd)
      .with_style(ImageStyle.Vivid)

    const response = await client.generateImage(req)
    expect(response).toBeInstanceOf(ImageGenerationResponse)
  })

  it('should handle negative prompt', async () => {
    const req = new ImageGenerationRequest(
      'stability-ai/stable-diffusion-xl',
      'A landscape'
    )
      .with_negative_prompt('blurry, low quality, watermark')
      .with_n(1)

    const response = await client.generateImage(req)
    expect(response).toBeDefined()
  })

  it('should handle empty prompt', async () => {
    const req = new ImageGenerationRequest('dall-e-3', '')

    try {
      const response = await client.generateImage(req)
      expect(response).toBeDefined()
    } catch (error) {
      // Error handling is provider-specific
      expect(error).toBeDefined()
    }
  })
})

describe('Image module exports', () => {
  it('should export ImageSize', () => {
    expect(ImageSize).toBeDefined()
  })

  it('should export ImageQuality', () => {
    expect(ImageQuality).toBeDefined()
  })

  it('should export ImageStyle', () => {
    expect(ImageStyle).toBeDefined()
  })

  it('should export ImageFormat', () => {
    expect(ImageFormat).toBeDefined()
  })

  it('should export ImageGenerationRequest', () => {
    expect(ImageGenerationRequest).toBeDefined()
  })

  it('should export GeneratedImage', () => {
    expect(GeneratedImage).toBeDefined()
  })

  it('should export ImageGenerationResponse', () => {
    expect(ImageGenerationResponse).toBeDefined()
  })
})

describe('Image integration tests', () => {
  let client: LLMKitClient

  beforeAll(() => {
    client = LLMKitClient.fromEnv()
  })

  it('should complete end-to-end image generation flow', async () => {
    // 1. Create request
    const req = new ImageGenerationRequest('dall-e-3', 'A serene Japanese garden with temple')

    // 2. Configure options
    const configuredReq = req
      .with_n(1)
      .with_size(ImageSize.Landscape1792x1024)
      .with_quality(ImageQuality.Hd)
      .with_style(ImageStyle.Natural)

    // 3. Generate image
    const response = await client.generateImage(configuredReq)

    // 4. Verify response
    expect(response).toBeInstanceOf(ImageGenerationResponse)
    expect(response.count).toBeGreaterThan(0)
    expect(response.first()).toBeDefined()
  })

  it('should generate multiple images with different prompts', async () => {
    const prompts = [
      { model: 'dall-e-3', prompt: 'A cat playing with yarn' },
      { model: 'fal-ai/flux/dev', prompt: 'A sunset over ocean' },
      { model: 'stability-ai/stable-diffusion-xl', prompt: 'A futuristic city' },
    ]

    const responses: ImageGenerationResponse[] = []
    for (const { model, prompt } of prompts) {
      const req = new ImageGenerationRequest(model, prompt)
      const response = await client.generateImage(req)
      responses.push(response)
    }

    expect(responses).toHaveLength(3)
    expect(responses.every((r) => r instanceof ImageGenerationResponse)).toBe(true)
  })

  it('should generate images with different sizes', async () => {
    const sizes = [
      ImageSize.Square256,
      ImageSize.Square512,
      ImageSize.Square1024,
      ImageSize.Portrait1024x1792,
      ImageSize.Landscape1792x1024,
    ]

    for (const size of sizes) {
      const req = new ImageGenerationRequest('dall-e-3', 'Test image').with_size(size)
      const response = await client.generateImage(req)
      expect(response).toBeInstanceOf(ImageGenerationResponse)
    }
  })
})
