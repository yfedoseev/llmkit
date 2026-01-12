/**
 * Example: Video Generation with LLMKit TypeScript bindings
 *
 * This example demonstrates how to use LLMKit to generate videos
 * from text prompts using various providers (Runware, DiffusionRouter).
 *
 * Providers:
 * - Runware: Multiple video models (Runway Gen-3, Kling, Pika, Hailuo, Leonardo)
 * - DiffusionRouter: Stable Diffusion Video (coming February 2026)
 *
 * Usage:
 *   npx ts-node examples/typescript_video_generation.ts
 */

import { LLMKitClient, VideoGenerationRequest, VideoModel } from '../index'
import * as fs from 'fs'

async function main() {
  console.log('🎬 LLMKit Video Generation Example')
  console.log('=' + '='.repeat(49))

  // Initialize client from environment
  // Requires provider-specific API keys (e.g., RUNWAYML_API_KEY)
  const client = LLMKitClient.fromEnv()
  console.log('✓ Client initialized from environment\n')

  // Example 1: Simple video generation request
  console.log('--- Example 1: Simple Video Generation ---')
  const req1 = new VideoGenerationRequest(
    'A serene landscape with mountains and a clear sky'
  )
  console.log(`Prompt: ${req1.prompt}`)

  const response1 = await client.generateVideo(req1)
  console.log(`Video URL: ${response1.video_url}`)
  console.log(`Format: ${response1.format}`)
  console.log(`Task ID: ${response1.task_id}`)
  console.log(`Status: ${response1.status}\n`)

  // Example 2: Video generation with specific model
  console.log('--- Example 2: Video with Specific Model ---')
  const req2 = new VideoGenerationRequest(
    'An abstract geometric animation with rotating shapes'
  ).with_model('kling-2.0')

  console.log(`Prompt: ${req2.prompt}`)
  console.log('Model: Kling 2.0')

  const response2 = await client.generateVideo(req2)
  console.log(`Video URL: ${response2.video_url}`)
  console.log(`Status: ${response2.status}\n`)

  // Example 3: Fully configured video generation request
  console.log('--- Example 3: Fully Configured Request ---')
  const req3 = new VideoGenerationRequest(
    'A person dancing in a vibrant neon club'
  )
    .with_model('runway-gen-4')
    .with_duration(10)
    .with_width(1280)
    .with_height(720)

  console.log(`Prompt: ${req3.prompt}`)
  console.log('Model: Runway Gen-4')
  console.log('Duration: 10 seconds')
  console.log('Resolution: 1280x720')

  const response3 = await client.generateVideo(req3)
  console.log(`Video URL: ${response3.video_url}`)
  console.log(`Duration: ${response3.duration}s`)
  console.log(`Resolution: ${response3.width}x${response3.height}`)
  console.log(`Size: ${response3.size} bytes`)
  console.log(`Status: ${response3.status}\n`)

  // Example 4: Polling for async video generation
  console.log('--- Example 4: Polling for Task Completion ---')
  const req4 = new VideoGenerationRequest(
    'A spaceship flying through a wormhole in space'
  ).with_model('pika-1.0')

  const response4 = await client.generateVideo(req4)
  console.log(`Task ID: ${response4.task_id}`)
  console.log(`Initial status: ${response4.status}`)

  // Simulate polling (in production, use actual polling API)
  console.log('Simulating polling...')
  for (let i = 0; i < 5; i++) {
    await sleep(500) // Sleep 500ms for demonstration
    console.log(`  Poll ${i + 1}: Status would be checked via API`)
    if (i === 4) {
      console.log('  → Task would be completed')
    }
  }

  console.log('')

  // Example 5: Multiple video generation requests
  console.log('--- Example 5: Multiple Video Generations ---')
  const prompts = [
    'A cat playing with yarn in a cozy living room',
    'Ocean waves crashing on a rocky beach at sunset',
    'Urban street art mural with vibrant colors',
  ]

  const models = ['runway-gen-4', 'kling-2.0', 'pika-1.0']

  console.log(`Generating ${prompts.length} videos with different models...`)
  const responses: Array<typeof response1> = []

  for (let i = 0; i < prompts.length; i++) {
    const req = new VideoGenerationRequest(prompts[i]).with_model(models[i])
    const response = await client.generateVideo(req)
    responses.push(response)
    console.log(
      `  ✓ Generated: ${prompts[i].substring(0, 40)}... (${models[i]})`
    )
    console.log(`    → Task ID: ${response.task_id}`)
  }

  console.log('')

  // Example 6: Video generation with high quality settings
  console.log('--- Example 6: High Quality Video ---')
  const req6 = new VideoGenerationRequest(
    'Professional product demonstration video'
  )
    .with_model('leonardo-ultra')
    .with_duration(15)
    .with_width(1920)
    .with_height(1080)

  console.log(`Prompt: ${req6.prompt}`)
  console.log('Settings:')
  console.log('  - Model: Leonardo Ultra (highest quality)')
  console.log('  - Duration: 15 seconds')
  console.log('  - Resolution: 1920x1080 (Full HD)')

  const response6 = await client.generateVideo(req6)
  console.log(`Video URL: ${response6.video_url}`)
  console.log('Quality: high')
  console.log(`Size: ${response6.size} bytes\n`)

  // Example 7: Video generation for different use cases
  console.log('--- Example 7: Different Use Cases ---')
  const useCases = [
    {
      name: 'Marketing',
      prompt: 'Product launch celebration with confetti and champagne',
    },
    {
      name: 'Education',
      prompt: 'Animated physics demonstration of gravitational forces',
    },
    {
      name: 'Entertainment',
      prompt: 'Sci-fi spaceship battle with explosions and lasers',
    },
    {
      name: 'Art',
      prompt: 'Abstract surreal dreamscape with melting clocks',
    },
  ]

  for (const useCase of useCases) {
    const req = new VideoGenerationRequest(useCase.prompt)
    const response = await client.generateVideo(req)
    console.log(`✓ ${useCase.name}: ${useCase.prompt.substring(0, 40)}...`)
    console.log(`  → Status: ${response.status}`)
  }

  console.log('')
  console.log('=' + '='.repeat(49))
  console.log('✓ Video generation examples completed!')
  console.log('\nNote: In production, use polling to wait for async video generation:')
  console.log("  while (response.status !== 'completed') {")
  console.log('    response = await client.pollVideo(response.task_id)')
  console.log('    await sleep(5000)')
  console.log('  }')
}

/**
 * Save video from response to a file
 */
async function saveVideoToFile(response: any, filename: string): Promise<void> {
  if (response.video_bytes) {
    fs.writeFileSync(filename, Buffer.from(response.video_bytes))
    console.log(`✓ Video saved to ${filename}`)
  } else if (response.video_url) {
    console.log(`ℹ Video available at: ${response.video_url}`)
    console.log('Use a download tool or fetch to save:')
    console.log(`  const res = await fetch('${response.video_url}')`)
    console.log(`  const buffer = await res.arrayBuffer()`)
    console.log(`  fs.writeFileSync('${filename}', Buffer.from(buffer))`)
  } else {
    console.log('✗ No video data available in response')
  }
}

/**
 * Sleep utility function
 */
function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// Run main function
main().catch((err) => {
  console.error('\n✗ Error:', err.message)
  process.exit(1)
})
