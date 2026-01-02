/**
 * Vision / Image Analysis Example
 *
 * Demonstrates image input capabilities with LLMKit.
 *
 * Requirements:
 * - Set ANTHROPIC_API_KEY environment variable (or use OpenAI's GPT-4V)
 *
 * Run:
 *   npx ts-node 04-vision.ts
 */

import * as fs from 'fs'
import * as path from 'path'
import {
    JsLlmKitClient as LLMKitClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
    JsContentBlock as ContentBlock,
} from 'llmkit'

async function analyzeImageFromUrl() {
    const client = LLMKitClient.fromEnv()

    // Create a message with an image URL
    const message = Message.userWithContent([
        ContentBlock.text('What do you see in this image? Describe it briefly.'),
        ContentBlock.imageUrl(
            'https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/' +
            'Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg'
        ),
    ])

    const request = CompletionRequest
        .create('claude-sonnet-4-20250514', [message])
        .withMaxTokens(500)

    console.log('Analyzing image from URL...')
    const response = await client.complete(request)
    console.log('\nDescription:\n', response.textContent())
}

async function analyzeLocalImage(imagePath: string) {
    const client = LLMKitClient.fromEnv()

    // Check if file exists
    if (!fs.existsSync(imagePath)) {
        console.log(`Image not found: ${imagePath}`)
        console.log('Skipping local image analysis...')
        return
    }

    // Detect media type
    const ext = path.extname(imagePath).toLowerCase()
    const mediaTypes: Record<string, string> = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    const mediaType = mediaTypes[ext] ?? 'image/png'

    // Read and encode
    const imageData = fs.readFileSync(imagePath).toString('base64')

    // Create message with image
    const message = Message.userWithContent([
        ContentBlock.text('Analyze this image. What\'s in it?'),
        ContentBlock.image(mediaType, imageData),
    ])

    const request = CompletionRequest
        .create('claude-sonnet-4-20250514', [message])
        .withMaxTokens(500)

    console.log(`Analyzing local image: ${imagePath}`)
    const response = await client.complete(request)
    console.log('\nAnalysis:\n', response.textContent())
}

async function multiImageComparison() {
    const client = LLMKitClient.fromEnv()

    // Example with multiple image URLs
    const message = Message.userWithContent([
        ContentBlock.text(
            'I\'m going to show you two images. ' +
            'Please compare them and describe the differences.'
        ),
        ContentBlock.imageUrl(
            'https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/' +
            'Cat03.jpg/120px-Cat03.jpg'
        ),
        ContentBlock.imageUrl(
            'https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/' +
            'YellowLabradorLooking_new.jpg/120px-YellowLabradorLooking_new.jpg'
        ),
    ])

    const request = CompletionRequest
        .create('claude-sonnet-4-20250514', [message])
        .withMaxTokens(500)

    console.log('Comparing two images...')
    const response = await client.complete(request)
    console.log('\nComparison:\n', response.textContent())
}

async function main() {
    console.log('='.repeat(50))
    console.log('Vision Example 1: Analyze image from URL')
    console.log('='.repeat(50))
    await analyzeImageFromUrl()

    console.log('\n' + '='.repeat(50))
    console.log('Vision Example 2: Compare multiple images')
    console.log('='.repeat(50))
    await multiImageComparison()

    // Uncomment to test local image:
    // console.log('\n' + '='.repeat(50))
    // console.log('Vision Example 3: Analyze local image')
    // console.log('='.repeat(50))
    // await analyzeLocalImage('path/to/your/image.png')
}

main().catch(console.error)
