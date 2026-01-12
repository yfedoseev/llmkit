/**
 * Retry & Resilience Example
 *
 * Demonstrates configuring automatic retry behavior for handling
 * transient failures like rate limits, timeouts, and server errors.
 *
 * Requirements:
 * - Set OPENROUTER_API_KEY environment variable
 * - npm install
 *
 * Run:
 *   npx ts-node 17-retry-resilience.ts
 */

import {
    JsLLMKitClient as LLMKitClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
    JsRetryConfig as RetryConfig,
} from 'llmkit'

async function main() {
    console.log('Retry & Resilience Configuration Example')
    console.log('='.repeat(50))

    // ========================================
    // Example 1: Default Retry (Recommended)
    // ========================================
    console.log('\n1. Default Retry Configuration')
    console.log('-'.repeat(40))
    console.log('Settings: 10 retries, 1s-5min exponential backoff, jitter enabled')

    let client = LLMKitClient.fromEnv()  // Uses default retry

    let request = CompletionRequest
        .create('openrouter/qwen/qwen3-32b', [
            Message.user("Say 'retry test passed' in exactly 3 words.")
        ])
        .withoutThinking()
        .withMaxTokens(50)

    let response = await client.complete(request)
    console.log(`Response: ${response.textContent().trim()}`)

    // ========================================
    // Example 2: Production Retry Config
    // ========================================
    console.log('\n2. Production Retry Configuration')
    console.log('-'.repeat(40))

    const productionConfig = RetryConfig.production()
    console.log('Settings:')
    console.log(`  Max retries: ${productionConfig.maxRetries}`)
    console.log(`  Initial delay: ${productionConfig.initialDelayMs}ms`)
    console.log(`  Max delay: ${productionConfig.maxDelayMs}ms`)
    console.log(`  Backoff multiplier: ${productionConfig.backoffMultiplier}`)
    console.log(`  Jitter: ${productionConfig.jitter}`)

    client = LLMKitClient.fromEnv(productionConfig)

    request = CompletionRequest
        .create('openrouter/qwen/qwen3-32b', [
            Message.user("Say 'production config works'")
        ])
        .withoutThinking()
        .withMaxTokens(50)

    response = await client.complete(request)
    console.log(`Response: ${response.textContent().trim()}`)

    // ========================================
    // Example 3: Conservative Retry Config
    // ========================================
    console.log('\n3. Conservative Retry Configuration')
    console.log('-'.repeat(40))

    const conservativeConfig = RetryConfig.conservative()
    console.log('Settings:')
    console.log(`  Max retries: ${conservativeConfig.maxRetries}`)
    console.log(`  Initial delay: ${conservativeConfig.initialDelayMs}ms`)
    console.log(`  Max delay: ${conservativeConfig.maxDelayMs}ms`)
    console.log('Best for: Quick failures, interactive applications')

    client = LLMKitClient.fromEnv(conservativeConfig)

    request = CompletionRequest
        .create('openrouter/qwen/qwen3-32b', [
            Message.user("Say 'conservative works'")
        ])
        .withoutThinking()
        .withMaxTokens(50)

    response = await client.complete(request)
    console.log(`Response: ${response.textContent().trim()}`)

    // ========================================
    // Example 4: Custom Retry Config
    // ========================================
    console.log('\n4. Custom Retry Configuration')
    console.log('-'.repeat(40))

    const customConfig = new RetryConfig({
        maxRetries: 5,
        initialDelayMs: 500,
        maxDelayMs: 10000,
        backoffMultiplier: 1.5,
        jitter: true,
    })

    console.log('Custom settings:')
    console.log(`  Max retries: ${customConfig.maxRetries}`)
    console.log(`  Initial delay: ${customConfig.initialDelayMs}ms`)
    console.log(`  Max delay: ${customConfig.maxDelayMs}ms`)
    console.log(`  Backoff multiplier: ${customConfig.backoffMultiplier}`)
    console.log(`  Jitter: ${customConfig.jitter}`)

    client = LLMKitClient.fromEnv(customConfig)

    request = CompletionRequest
        .create('openrouter/qwen/qwen3-32b', [
            Message.user("Say 'custom config works'")
        ])
        .withoutThinking()
        .withMaxTokens(50)

    response = await client.complete(request)
    console.log(`Response: ${response.textContent().trim()}`)

    // ========================================
    // Example 5: No Retry (for testing/debugging)
    // ========================================
    console.log('\n5. No Retry Configuration')
    console.log('-'.repeat(40))
    console.log('Useful for: Testing, debugging, fail-fast scenarios')

    client = LLMKitClient.fromEnv(RetryConfig.none())

    request = CompletionRequest
        .create('openrouter/qwen/qwen3-32b', [
            Message.user("Say 'no retry works'")
        ])
        .withoutThinking()
        .withMaxTokens(50)

    response = await client.complete(request)
    console.log(`Response: ${response.textContent().trim()}`)

    // Summary
    console.log('\n' + '='.repeat(50))
    console.log('Summary: Retry Configurations')
    console.log('='.repeat(50))
    console.log(`
Retry handles these transient errors automatically:
  - Rate limits (429)
  - Server errors (5xx)
  - Timeouts
  - Connection errors

Configuration options:
  - Default (undefined)         -> 10 retries, 5min max
  - RetryConfig.production()    -> Same as default
  - RetryConfig.conservative()  -> 3 retries, 30s max
  - RetryConfig.none()          -> No retries
  - new RetryConfig({...})      -> Custom settings
`)
}

main().catch(console.error)
