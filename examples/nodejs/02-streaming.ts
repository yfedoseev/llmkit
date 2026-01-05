/**
 * Streaming Example
 *
 * Demonstrates real-time streaming of completion responses.
 *
 * Requirements:
 * - Set OPENAI_API_KEY environment variable
 *
 * Run:
 *   npx ts-node 02-streaming.ts
 */

import {
    JsModelSuiteClient as ModelSuiteClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
} from 'modelsuite'

async function streamWithAsyncIterator() {
    const client = ModelSuiteClient.fromEnv()

    // Create a request with streaming enabled
    // Use "provider/model" format for explicit provider routing
    const request = CompletionRequest
        .create('openai/gpt-4o', [
            Message.user('Write a short poem about programming. 4 lines maximum.')
        ])
        .withMaxTokens(200)
        .withStreaming()

    console.log('Streaming response (async iterator):\n')

    // Stream using async iterator
    const stream = await client.stream(request)

    let chunk
    while ((chunk = await stream.next()) !== null) {
        // Print text chunks as they arrive
        if (chunk.text) {
            process.stdout.write(chunk.text)
        }

        // Check if streaming is complete
        if (chunk.isDone) {
            console.log('\n\n[Stream complete]')

            // Print final usage if available
            if (chunk.usage) {
                console.log('Total tokens:', chunk.usage.totalTokens())
            }
            break
        }
    }
}

async function streamWithCallback() {
    const client = ModelSuiteClient.fromEnv()

    const request = CompletionRequest
        .create('openai/gpt-4o', [
            Message.user('Say hello in 3 languages')
        ])
        .withMaxTokens(100)
        .withStreaming()

    console.log('\nStreaming response (callback):\n')

    // Return a promise that resolves when streaming is done
    return new Promise<void>((resolve, reject) => {
        client.completeStream(request, (chunk, error) => {
            if (error) {
                console.error('Stream error:', error)
                reject(new Error(error))
                return
            }

            if (chunk?.text) {
                process.stdout.write(chunk.text)
            }

            if (chunk?.isDone) {
                console.log('\n\n[Stream complete]')
                resolve()
            }
        })
    })
}

async function main() {
    console.log('=' .repeat(50))
    console.log('Example 1: Async Iterator')
    console.log('=' .repeat(50))
    await streamWithAsyncIterator()

    console.log('\n' + '=' .repeat(50))
    console.log('Example 2: Callback')
    console.log('=' .repeat(50))
    await streamWithCallback()
}

main().catch(console.error)
