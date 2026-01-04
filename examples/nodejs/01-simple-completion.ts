/**
 * Simple Completion Example
 *
 * Demonstrates the basic usage of LLMKit for making completion requests.
 *
 * Requirements:
 * - Set ANTHROPIC_API_KEY environment variable (or another provider's key)
 *
 * Run:
 *   npx ts-node 01-simple-completion.ts
 */

import {
    JsLlmKitClient as LLMKitClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
} from 'modelsuite'

async function main() {
    // Create client from environment variables
    // Automatically detects configured providers
    const client = LLMKitClient.fromEnv()

    // List available providers
    const providers = client.providers()
    console.log('Available providers:', providers)

    // Create a simple completion request
    // Use "provider/model" format for explicit provider routing
    const request = CompletionRequest
        .create('anthropic/claude-sonnet-4-20250514', [
            Message.user('What is the capital of France? Reply in one word.')
        ])
        .withMaxTokens(100)

    // Make the request
    console.log('\nSending request...')
    const response = await client.complete(request)

    // Print the response
    console.log('\nResponse:', response.textContent())
    console.log('Model:', response.model)
    console.log('Stop reason:', response.stopReason)

    // Print token usage
    const usage = response.usage
    if (usage) {
        console.log('\nToken usage:')
        console.log('  Input tokens:', usage.inputTokens)
        console.log('  Output tokens:', usage.outputTokens)
        console.log('  Total tokens:', usage.totalTokens())
    }
}

main().catch(console.error)
