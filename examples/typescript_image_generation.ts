/**
 * Example: Image Generation with ModelSuite TypeScript bindings
 *
 * This example demonstrates how to use ModelSuite to generate images
 * from text prompts using various providers.
 *
 * Providers:
 * - FAL AI: FLUX and Stable Diffusion 3 models
 * - Recraft: Vector and design-focused generation
 * - Stability AI: SDXL and other Stable Diffusion models
 * - RunwayML: Various image generation models
 *
 * Usage:
 *   npx ts-node examples/typescript_image_generation.ts
 */

import {
  ModelSuiteClient,
  ImageGenerationRequest,
  ImageSize,
  ImageQuality,
  ImageStyle,
  ImageFormat,
} from '../index'
import * as fs from 'fs'

async function main() {
  console.log('🎨 ModelSuite Image Generation Example')
  console.log('=' + '='.repeat(49))

  const client = ModelSuiteClient.fromEnv()
  console.log('✓ Client initialized from environment\n')

  // Example 1: Simple image generation
  console.log('--- Example 1: Simple Image Generation ---')
  const req1 = new ImageGenerationRequest('dall-e-3', 'A serene landscape with mountains')
  console.log(`Model: dall-e-3`)
  console.log(`Prompt: ${req1.prompt}`)

  const response1 = await client.generateImage(req1)
  console.log(`Generated ${response1.count} image(s)`)
  if (response1.first()) {
    console.log(`Image URL: ${response1.first()?.url}\n`)
  }

  // Example 2: High-quality image with style
  console.log('--- Example 2: High-Quality Image ---')
  const req2 = new ImageGenerationRequest('dall-e-3', 'A portrait of a Renaissance noble')
    .with_quality(ImageQuality.Hd)
    .with_style(ImageStyle.Vivid)

  console.log(`Prompt: ${req2.prompt}`)
  console.log('Quality: HD')
  console.log('Style: Vivid')

  const response2 = await client.generateImage(req2)
  console.log(`Generated ${response2.count} image(s)`)
  console.log(`Created at: ${response2.created}\n`)

  // Example 3: Multiple images with custom size
  console.log('--- Example 3: Multiple Images (Landscape) ---')
  const req3 = new ImageGenerationRequest('fal-ai/flux/dev', 'Abstract geometric patterns')
    .with_n(2)
    .with_size(ImageSize.Landscape1792x1024)

  console.log(`Model: fal-ai/flux/dev`)
  console.log(`Prompt: ${req3.prompt}`)
  console.log('Number of images: 2')
  console.log('Size: 1792x1024 (landscape)')

  const response3 = await client.generateImage(req3)
  console.log(`Generated ${response3.count} image(s)`)
  console.log(`Total size: ${response3.total_size} bytes\n`)

  // Example 4: Base64 encoded output
  console.log('--- Example 4: Base64 Encoded Output ---')
  const req4 = new ImageGenerationRequest('dall-e-3', 'A futuristic cityscape at night')
    .with_format(ImageFormat.B64Json)

  console.log(`Prompt: ${req4.prompt}`)
  console.log('Format: Base64-encoded')

  const response4 = await client.generateImage(req4)
  console.log(`Generated ${response4.count} image(s)`)
  if (response4.first()?.b64_json) {
    console.log(`B64 data size: ${response4.first()?.b64_json.length} characters\n`)
  }

  // Example 5: Stability AI with negative prompt
  console.log('--- Example 5: Using Negative Prompts ---')
  const req5 = new ImageGenerationRequest('stability-ai/stable-diffusion-xl', 'A serene ocean')
    .with_negative_prompt('blurry, watermark, low quality')
    .with_n(1)

  console.log(`Model: stability-ai/stable-diffusion-xl`)
  console.log(`Prompt: ${req5.prompt}`)
  console.log(`Negative prompt: ${req5.negative_prompt}`)

  const response5 = await client.generateImage(req5)
  console.log(`Generated ${response5.count} image(s)\n`)

  // Example 6: Portrait orientation
  console.log('--- Example 6: Portrait Orientation ---')
  const req6 = new ImageGenerationRequest(
    'dall-e-3',
    'A detailed portrait of a fantasy character'
  )
    .with_size(ImageSize.Portrait1024x1792)
    .with_quality(ImageQuality.Hd)

  console.log(`Prompt: ${req6.prompt}`)
  console.log('Size: 1024x1792 (portrait)')
  console.log('Quality: HD')

  const response6 = await client.generateImage(req6)
  console.log(`Generated ${response6.count} image(s)\n`)

  // Example 7: Recraft (vector/design)
  console.log('--- Example 7: Vector/Design Generation (Recraft) ---')
  const req7 = new ImageGenerationRequest('recraft-v3', 'A minimalist logo for a tech startup')
    .with_n(1)

  console.log('Model: recraft-v3 (vector generation)')
  console.log(`Prompt: ${req7.prompt}`)

  const response7 = await client.generateImage(req7)
  console.log(`Generated ${response7.count} image(s)\n`)

  // Example 8: Various sizes
  console.log('--- Example 8: Various Image Sizes ---')
  const sizes: [string, ImageSize][] = [
    ['Square (256x256)', ImageSize.Square256],
    ['Square (512x512)', ImageSize.Square512],
    ['Square (1024x1024)', ImageSize.Square1024],
    ['Portrait (1024x1792)', ImageSize.Portrait1024x1792],
    ['Landscape (1792x1024)', ImageSize.Landscape1792x1024],
  ]

  for (let i = 0; i < 2; i++) {
    const [sizeName, size] = sizes[i]
    const req = new ImageGenerationRequest('dall-e-3', 'A test image').with_size(size)
    const response = await client.generateImage(req)
    console.log(`✓ ${sizeName}: ${response.count} image(s)`)
  }

  console.log('')

  // Example 9: Batch generation
  console.log('--- Example 9: Batch Generation ---')
  const prompts = [
    { prompt: 'A red apple on a white table', model: 'dall-e-3' },
    { prompt: 'A galaxy seen from space', model: 'fal-ai/flux/dev' },
    { prompt: 'A steampunk airship', model: 'stability-ai/stable-diffusion-xl' },
  ]

  console.log(`Generating ${prompts.length} images with different prompts...`)
  const responses: any[] = []
  for (const { prompt, model } of prompts) {
    const req = new ImageGenerationRequest(model, prompt)
    const response = await client.generateImage(req)
    responses.push(response)
    console.log(`  ✓ ${prompt.substring(0, 40)}... (${model})`)
  }

  const totalImages = responses.reduce((sum, r) => sum + r.count, 0)
  console.log(`Total images generated: ${totalImages}\n`)

  // Example 10: Reproducible generation with seed
  console.log('--- Example 10: Reproducible Generation (with Seed) ---')
  const req10a = new ImageGenerationRequest(
    'stability-ai/stable-diffusion-xl',
    'A cat sitting'
  ).with_seed(BigInt(12345))

  const req10b = new ImageGenerationRequest(
    'stability-ai/stable-diffusion-xl',
    'A cat sitting'
  ).with_seed(BigInt(12345))

  console.log('Generating two images with the same seed (12345)...')
  console.log('Note: Same seed should produce identical or very similar images')

  const response10a = await client.generateImage(req10a)
  const response10b = await client.generateImage(req10b)

  console.log(`✓ Generated image 1: ${response10a.first()?.url || 'N/A'}`)
  console.log(`✓ Generated image 2: ${response10b.first()?.url || 'N/A'}`)

  console.log('')
  console.log('=' + '='.repeat(49))
  console.log('✓ Image generation examples completed!')
}

/**
 * Save image from response to a file
 */
async function saveImageToFile(response: any, filename: string): Promise<void> {
  if (response.first()?.b64_json) {
    const buffer = Buffer.from(response.first().b64_json, 'base64')
    fs.writeFileSync(filename, buffer)
    console.log(`✓ Image saved to ${filename}`)
  } else if (response.first()?.url) {
    console.log(`ℹ Image available at: ${response.first()?.url}`)
    console.log('Use fetch to download and save:')
    console.log(`  const res = await fetch('${response.first()?.url}')`)
    console.log(`  const buffer = await res.arrayBuffer()`)
    console.log(`  fs.writeFileSync('${filename}', Buffer.from(buffer))`)
  } else {
    console.log('✗ No image data available in response')
  }
}

// Run main function
main().catch((err) => {
  console.error('\n✗ Error:', err.message)
  process.exit(1)
})
