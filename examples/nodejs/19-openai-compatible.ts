/**
 * OpenAI-Compatible Provider Example
 *
 * Demonstrates using OpenRouter through the generic OpenAI-compatible
 * provider interface. This shows that any OpenAI-compatible API can be
 * used with LLMKit.
 *
 * Note: The dedicated OpenRouterProvider is recommended for Qwen3 models
 * because it automatically handles the `/no_think` injection. When using
 * the OpenAI-compatible endpoint, you need to manually add `/no_think`
 * to prompts or use higher max_tokens.
 *
 * Requirements:
 * - Set OPENROUTER_API_KEY environment variable
 * - npm install
 *
 * Run:
 *   npx ts-node 18-openai-compatible.ts
 */

import {
    JsLLMKitClient as LLMKitClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
} from 'llmkit'

async function main() {
    const apiKey = process.env.OPENROUTER_API_KEY
    if (!apiKey) {
        throw new Error('OPENROUTER_API_KEY environment variable not set')
    }

    // Create client with OpenAI-compatible provider pointing to OpenRouter
    // Note: Provider name (modelId) cannot contain dashes due to model format parsing
    const client = new LLMKitClient({
        providers: {
            openaiCompatible: {
                apiKey: apiKey,
                baseUrl: 'https://openrouter.ai/api/v1',
                modelId: 'openroutercompat',  // Provider name for routing
            }
        }
    })

    // Model format is "provider/model" - everything after first "/" is the model name
    const model = 'openroutercompat/qwen/qwen3-32b'

    console.log('OpenAI-Compatible Provider Example')
    console.log('='.repeat(50))
    console.log('Using OpenRouter via OpenAI-compatible endpoint\n')

    // Since we're not using OpenRouterProvider, we need to manually
    // add /no_think to disable Qwen3's reasoning mode
    console.log('Request with manual /no_think:')
    console.log('-'.repeat(40))

    let request = CompletionRequest
        .create(model, [Message.user('What is 2 + 2? Answer briefly. /no_think')])
        .withSystem('You are a helpful assistant.')
        .withMaxTokens(100)

    let response = await client.complete(request)
    console.log(`Response: ${response.textContent().trim()}\n`)

    // Second request
    console.log('Another request:')
    console.log('-'.repeat(40))

    request = CompletionRequest
        .create(model, [Message.user('Name 3 colors. /no_think')])
        .withSystem('Be concise.')
        .withMaxTokens(100)

    response = await client.complete(request)
    console.log(`Response: ${response.textContent().trim()}\n`)

    console.log('='.repeat(50))
    console.log('OpenRouter works via OpenAI-compatible endpoint!')
    console.log('Note: Manual /no_think required for Qwen3 models.')
}

main().catch(console.error)
