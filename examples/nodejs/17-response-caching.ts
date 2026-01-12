/**
 * Response Caching Example
 *
 * Demonstrates client-side response caching to reduce API costs
 * by caching identical requests. Uses qwen/qwen3-32b on OpenRouter.
 *
 * Cache hit = no API call = free!
 *
 * Note: We use .withoutThinking() to disable the model's reasoning mode,
 * which allows using lower max_tokens values efficiently.
 *
 * Requirements:
 * - Set OPENROUTER_API_KEY environment variable
 * - npm install
 *
 * Run:
 *   npx ts-node 16-response-caching.ts
 */

import {
    JsLLMKitClient as LLMKitClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
    JsCompletionResponse as CompletionResponse,
} from 'llmkit'

// Simple in-memory cache for demonstration
const cache = new Map<string, CompletionResponse>()

async function cachedComplete(
    client: InstanceType<typeof LLMKitClient>,
    cacheKey: string,
    request: InstanceType<typeof CompletionRequest>
): Promise<CompletionResponse> {
    if (cache.has(cacheKey)) {
        console.log('  [CACHE HIT - no API call]')
        return cache.get(cacheKey)!
    } else {
        console.log('  [CACHE MISS - API call]')
        const response = await client.complete(request)
        cache.set(cacheKey, response)
        return response
    }
}

async function main() {
    const client = LLMKitClient.fromEnv()
    const model = 'openrouter/qwen/qwen3-32b'

    console.log('Response Caching Example')
    console.log('='.repeat(50))

    // First request - will hit the API
    console.log('\n1. First request (France):')
    console.log('-'.repeat(40))

    let request = CompletionRequest
        .create(model, [Message.user('What is the capital of France?')])
        .withSystem('Answer briefly in one sentence.')
        .withoutThinking()
        .withMaxTokens(100)

    let response = await cachedComplete(client, 'france', request)
    console.log(`Response: ${response.textContent().trim()}`)

    // Second identical request - should hit the cache
    console.log('\n2. Same request again (France):')
    console.log('-'.repeat(40))

    request = CompletionRequest
        .create(model, [Message.user('What is the capital of France?')])
        .withSystem('Answer briefly in one sentence.')
        .withoutThinking()
        .withMaxTokens(100)

    response = await cachedComplete(client, 'france', request)
    console.log(`Response: ${response.textContent().trim()}`)

    // Third request with different question - cache miss
    console.log('\n3. Different question (Germany):')
    console.log('-'.repeat(40))

    request = CompletionRequest
        .create(model, [Message.user('What is the capital of Germany?')])
        .withSystem('Answer briefly in one sentence.')
        .withoutThinking()
        .withMaxTokens(100)

    response = await cachedComplete(client, 'germany', request)
    console.log(`Response: ${response.textContent().trim()}`)

    // Fourth request - same as first (cache hit)
    console.log('\n4. First question again (France):')
    console.log('-'.repeat(40))

    request = CompletionRequest
        .create(model, [Message.user('What is the capital of France?')])
        .withSystem('Answer briefly in one sentence.')
        .withoutThinking()
        .withMaxTokens(100)

    response = await cachedComplete(client, 'france', request)
    console.log(`Response: ${response.textContent().trim()}`)

    // Summary
    console.log('\n' + '='.repeat(50))
    console.log('Summary:')
    console.log('  Total requests: 4')
    console.log('  API calls made: 2 (misses)')
    console.log('  Free responses: 2 (hits)')
    console.log('  Hit rate: 50%')
    console.log('\nCaching saves API costs on repeated identical requests!')
}

main().catch(console.error)
