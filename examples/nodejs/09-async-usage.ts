/**
 * Async Usage Example
 *
 * Demonstrates asynchronous patterns with LLMKit in Node.js/TypeScript.
 * Since Node.js is async by nature, this focuses on concurrent execution,
 * rate limiting, and Promise patterns.
 *
 * Requirements:
 * - Set GROQ_API_KEY environment variable (Groq has fast inference)
 * - npm install
 *
 * Run:
 *   npx ts-node 09-async-usage.ts
 */

import {
    JsLLMKitClient as LLMKitClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
} from 'llmkit'

async function basicAsyncCompletion() {
    const client = LLMKitClient.fromEnv()

    const request = CompletionRequest
        .create('groq/llama-3.3-70b-versatile', [
            Message.user('What is the capital of Japan?')
        ])
        .withMaxTokens(100)

    console.log('Making async request...')
    const response = await client.complete(request)
    console.log(`Response: ${response.textContent()}`)
}

async function asyncStreaming() {
    const client = LLMKitClient.fromEnv()

    const request = CompletionRequest
        .create('groq/llama-3.3-70b-versatile', [
            Message.user('Count from 1 to 5 slowly')
        ])
        .withMaxTokens(200)
        .withStreaming()

    console.log('Streaming async response:')
    const stream = await client.completeStream(request)
    for await (const chunk of stream) {
        if (chunk.text) {
            process.stdout.write(chunk.text)
        }
        if (chunk.isDone) break
    }
    console.log()
}

async function concurrentRequests() {
    const client = LLMKitClient.fromEnv()

    const questions = [
        'What is Python?',
        'What is Rust?',
        'What is JavaScript?',
    ]

    const requests = questions.map(q =>
        CompletionRequest
            .create('groq/llama-3.3-70b-versatile', [Message.user(q)])
            .withMaxTokens(100)
    )

    console.log('Making concurrent requests...')

    // Run all requests concurrently with Promise.all
    const responses = await Promise.all(
        requests.map(req => client.complete(req).catch(e => e))
    )

    for (let i = 0; i < questions.length; i++) {
        const response = responses[i]
        console.log(`\nQ: ${questions[i]}`)
        if (response instanceof Error) {
            console.log(`A: Error - ${response.message}`)
        } else {
            console.log(`A: ${response.textContent().slice(0, 100)}...`)
        }
    }
}

async function rateLimitedBatch() {
    const client = LLMKitClient.fromEnv()

    const questions = [
        'What is 1+1?',
        'What is 2+2?',
        'What is 3+3?',
        'What is 4+4?',
        'What is 5+5?',
    ]

    // Simple semaphore implementation for rate limiting
    const maxConcurrent = 2
    let running = 0
    const queue: (() => void)[] = []

    async function withLimit<T>(fn: () => Promise<T>): Promise<T> {
        while (running >= maxConcurrent) {
            await new Promise<void>(resolve => queue.push(resolve))
        }
        running++
        try {
            return await fn()
        } finally {
            running--
            const next = queue.shift()
            if (next) next()
        }
    }

    console.log('Processing with rate limiting (max 2 concurrent)...')

    const results = await Promise.all(
        questions.map(q =>
            withLimit(async () => {
                console.log(`Processing: ${q}`)
                const request = CompletionRequest
                    .create('groq/llama-3.3-70b-versatile', [Message.user(q)])
                    .withMaxTokens(20)
                const response = await client.complete(request)
                return { question: q, answer: response.textContent().trim() }
            })
        )
    )

    console.log('\nResults:')
    for (const { question, answer } of results) {
        console.log(`  ${question} -> ${answer}`)
    }
}

async function asyncWithTimeout() {
    const client = LLMKitClient.fromEnv()

    const request = CompletionRequest
        .create('groq/llama-3.3-70b-versatile', [
            Message.user('Write a very long story')
        ])
        .withMaxTokens(2000)

    // Helper function to add timeout to a promise
    function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
        return Promise.race([
            promise,
            new Promise<never>((_, reject) =>
                setTimeout(() => reject(new Error('Request timed out')), ms)
            )
        ])
    }

    try {
        const response = await withTimeout(client.complete(request), 30000)
        console.log(`Got response: ${response.textContent().slice(0, 100)}...`)
    } catch (error) {
        if (error instanceof Error && error.message === 'Request timed out') {
            console.log('Request timed out!')
        } else {
            throw error
        }
    }
}

async function asyncErrorHandling() {
    const client = LLMKitClient.fromEnv()

    const request = CompletionRequest
        .create('groq/llama-3.3-70b-versatile', [Message.user('Hello!')])
        .withMaxTokens(100)

    try {
        const response = await client.complete(request)
        console.log(`Success: ${response.textContent()}`)
    } catch (error) {
        if (error instanceof Error) {
            const message = error.message.toLowerCase()
            if (message.includes('auth') || message.includes('api key')) {
                console.log('Auth error - check API key')
            } else if (message.includes('rate limit')) {
                console.log(`Rate limited: ${error.message}`)
            } else {
                console.log(`LLMKit error: ${error.message}`)
            }
        } else {
            console.log(`Unexpected error: ${error}`)
        }
    }
}

async function main() {
    console.log('='.repeat(50))
    console.log('Example 1: Basic Async Completion')
    console.log('='.repeat(50))
    await basicAsyncCompletion()

    console.log('\n' + '='.repeat(50))
    console.log('Example 2: Async Streaming')
    console.log('='.repeat(50))
    await asyncStreaming()

    console.log('\n' + '='.repeat(50))
    console.log('Example 3: Concurrent Requests')
    console.log('='.repeat(50))
    await concurrentRequests()

    console.log('\n' + '='.repeat(50))
    console.log('Example 4: Rate-Limited Batch')
    console.log('='.repeat(50))
    await rateLimitedBatch()

    console.log('\n' + '='.repeat(50))
    console.log('Example 5: Async Error Handling')
    console.log('='.repeat(50))
    await asyncErrorHandling()
}

main().catch(console.error)
