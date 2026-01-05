/**
 * Error Handling Example
 *
 * Demonstrates how to handle various error conditions with ModelSuite.
 *
 * Requirements:
 * - Set MISTRAL_API_KEY environment variable
 *
 * Run:
 *   npx ts-node 08-error-handling.ts
 */

import {
    JsModelSuiteClient as ModelSuiteClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
} from 'modelsuite'

async function basicErrorHandling() {
    const client = ModelSuiteClient.fromEnv()

    // Use "provider/model" format for explicit provider routing
    const request = CompletionRequest
        .create('mistral/mistral-large-latest', [Message.user('Hello!')])
        .withMaxTokens(100)

    try {
        const response = await client.complete(request)
        console.log('Success:', response.textContent())
    } catch (error) {
        if (error instanceof Error) {
            console.log('Error:', error.message)
        }
    }
}

async function handleSpecificErrors() {
    const client = ModelSuiteClient.fromEnv()

    const request = CompletionRequest
        .create('mistral/mistral-large-latest', [Message.user('Hello!')])
        .withMaxTokens(100)

    try {
        const response = await client.complete(request)
        console.log('Success:', response.textContent())
    } catch (error) {
        if (error instanceof Error) {
            const message = error.message

            if (message.includes('Authentication')) {
                console.log('Authentication failed')
                console.log('Please check your API key and try again.')
            } else if (message.includes('RateLimit')) {
                console.log('Rate limited')
                console.log('Too many requests. Please wait and try again.')
            } else if (message.includes('ContextLength')) {
                console.log('Context too long')
                console.log('Try reducing your input or using a model with larger context.')
            } else if (message.includes('InvalidRequest')) {
                console.log('Invalid request:', message)
                console.log('Check your request parameters.')
            } else if (message.includes('ProviderNotFound')) {
                console.log('Provider not found')
                console.log('Make sure the provider is configured with an API key.')
            } else if (message.includes('Timeout')) {
                console.log('Request timed out')
                console.log('The request took too long. Try again or reduce complexity.')
            } else {
                console.log('Error:', message)
            }
        }
    }
}

async function demonstrateProviderNotFound() {
    const client = ModelSuiteClient.fromEnv()

    try {
        // Try to use a provider that's not configured
        await client.completeWithProvider(
            'definitely_not_a_real_provider',
            CompletionRequest
                .create('some-model', [Message.user('Hello!')])
                .withMaxTokens(100)
        )
    } catch (error) {
        if (error instanceof Error) {
            console.log('Expected error:', error.message)
            console.log('Available providers:', client.providers())
        }
    }
}

async function retryOnRateLimit() {
    const client = ModelSuiteClient.fromEnv()

    const request = CompletionRequest
        .create('mistral/mistral-large-latest', [Message.user('Hello!')])
        .withMaxTokens(100)

    const maxRetries = 3
    let retryDelay = 1000  // Start with 1 second

    for (let attempt = 0; attempt < maxRetries; attempt++) {
        try {
            const response = await client.complete(request)
            console.log(`Success on attempt ${attempt + 1}`)
            console.log('Response:', response.textContent())
            return response
        } catch (error) {
            if (error instanceof Error && error.message.includes('RateLimit')) {
                if (attempt < maxRetries - 1) {
                    console.log(`Rate limited. Waiting ${retryDelay}ms before retry...`)
                    await new Promise(resolve => setTimeout(resolve, retryDelay))
                    retryDelay *= 2  // Exponential backoff
                } else {
                    console.log('Max retries exceeded')
                    throw error
                }
            } else {
                throw error
            }
        }
    }
}

async function safeComplete(
    client: InstanceType<typeof ModelSuiteClient>,
    request: InstanceType<typeof CompletionRequest>,
    defaultResponse: string = 'Unable to generate response'
): Promise<string> {
    try {
        const response = await client.complete(request)
        return response.textContent()
    } catch (error) {
        if (error instanceof Error) {
            const message = error.message

            if (message.includes('Authentication')) {
                return 'Error: Invalid API key'
            }
            if (message.includes('RateLimit')) {
                return 'Error: Too many requests, please try again later'
            }
            if (message.includes('ContextLength')) {
                return 'Error: Input too long'
            }
            if (message.includes('InvalidRequest')) {
                return `Error: Invalid request - ${message}`
            }
            if (message.includes('ProviderNotFound')) {
                return 'Error: Provider not available'
            }
            if (message.includes('Timeout')) {
                return 'Error: Request timed out'
            }
            return `Error: ${message}`
        }
        return defaultResponse
    }
}

async function main() {
    console.log('='.repeat(50))
    console.log('Example 1: Basic Error Handling')
    console.log('='.repeat(50))
    await basicErrorHandling()

    console.log('\n' + '='.repeat(50))
    console.log('Example 2: Specific Error Types')
    console.log('='.repeat(50))
    await handleSpecificErrors()

    console.log('\n' + '='.repeat(50))
    console.log('Example 3: Provider Not Found')
    console.log('='.repeat(50))
    await demonstrateProviderNotFound()

    console.log('\n' + '='.repeat(50))
    console.log('Example 4: Safe Completion Wrapper')
    console.log('='.repeat(50))
    const client = ModelSuiteClient.fromEnv()
    const result = await safeComplete(
        client,
        CompletionRequest
            .create('mistral/mistral-large-latest', [Message.user('Say hello briefly')])
            .withMaxTokens(50)
    )
    console.log('Safe result:', result)
}

main().catch(console.error)
