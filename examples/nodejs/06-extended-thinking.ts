/**
 * Extended Thinking (Reasoning Mode) Example
 *
 * Demonstrates how to enable extended thinking for complex reasoning tasks.
 *
 * Requirements:
 * - Set ANTHROPIC_API_KEY environment variable
 *
 * Run:
 *   npx ts-node 06-extended-thinking.ts
 */

import {
    JsLlmKitClient as LLMKitClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
} from 'llmkit'

async function basicThinking() {
    const client = LLMKitClient.fromEnv()

    // Enable thinking with a budget of 5000 tokens for reasoning
    // Use "provider/model" format for explicit provider routing
    const request = CompletionRequest
        .create('anthropic/claude-sonnet-4-20250514', [
            Message.user(
                'Solve this step by step: ' +
                'A train travels from City A to City B at 60 mph. ' +
                'Another train leaves City B towards City A at 40 mph at the same time. ' +
                'The cities are 200 miles apart. ' +
                'Where do they meet and after how long?'
            )
        ])
        .withMaxTokens(2000)
        .withThinking(5000)

    console.log('Solving with extended thinking enabled...')
    console.log('(This may take a moment)\n')

    const response = await client.complete(request)

    // Get the thinking/reasoning content
    const thinking = response.thinkingContent()
    if (thinking) {
        console.log('='.repeat(50))
        console.log('THINKING PROCESS:')
        console.log('='.repeat(50))
        console.log(thinking)
        console.log()
    }

    // Get the final answer
    console.log('='.repeat(50))
    console.log('FINAL ANSWER:')
    console.log('='.repeat(50))
    console.log(response.textContent())

    // Usage info
    if (response.usage) {
        console.log(`\nTokens used: ${response.usage.totalTokens()}`)
    }
}

async function complexReasoning() {
    const client = LLMKitClient.fromEnv()

    const request = CompletionRequest
        .create('anthropic/claude-sonnet-4-20250514', [
            Message.user(
                'Consider this logic puzzle:\n\n' +
                'Three friends - Alice, Bob, and Carol - each have a different pet ' +
                '(cat, dog, fish) and live in different colored houses (red, blue, green).\n\n' +
                'Clues:\n' +
                '1. Alice doesn\'t live in the red house.\n' +
                '2. The person with the cat lives in the blue house.\n' +
                '3. Bob doesn\'t have a fish.\n' +
                '4. Carol lives in the green house.\n' +
                '5. The dog owner doesn\'t live in the green house.\n\n' +
                'Who owns what pet and lives in which house?'
            )
        ])
        .withMaxTokens(2000)
        .withThinking(8000)

    console.log('Solving logic puzzle with extended thinking...\n')

    const response = await client.complete(request)

    const thinking = response.thinkingContent()
    if (thinking) {
        // Show first part of thinking
        console.log('THINKING (first 500 chars):')
        console.log('-'.repeat(40))
        console.log(thinking.length > 500 ? thinking.slice(0, 500) + '...' : thinking)
        console.log()
    }

    console.log('SOLUTION:')
    console.log('-'.repeat(40))
    console.log(response.textContent())
}

async function streamingWithThinking() {
    const client = LLMKitClient.fromEnv()

    const request = CompletionRequest
        .create('anthropic/claude-sonnet-4-20250514', [
            Message.user(
                'What\'s the best strategy for playing tic-tac-toe? ' +
                'Think through the possible moves.'
            )
        ])
        .withMaxTokens(1500)
        .withThinking(3000)
        .withStreaming()

    console.log('Streaming with thinking enabled...\n')

    const stream = await client.stream(request)

    let currentSection = 'thinking'

    let chunk
    while ((chunk = await stream.next()) !== null) {
        const delta = chunk.delta
        if (delta) {
            if (delta.isThinking) {
                const thinking = delta.thinking
                if (thinking) {
                    process.stdout.write(`[T] ${thinking}`)
                }
            } else if (delta.isText) {
                const text = delta.text
                if (text) {
                    if (currentSection === 'thinking') {
                        currentSection = 'answer'
                        console.log('\n\n[Answer]:')
                    }
                    process.stdout.write(text)
                }
            }
        }

        if (chunk.isDone) {
            break
        }
    }

    console.log('\n')
}

async function main() {
    console.log('='.repeat(60))
    console.log('Example 1: Basic Math Problem with Thinking')
    console.log('='.repeat(60))
    await basicThinking()

    console.log('\n' + '='.repeat(60))
    console.log('Example 2: Complex Logic Puzzle')
    console.log('='.repeat(60))
    await complexReasoning()

    // Uncomment to test streaming with thinking:
    // console.log('\n' + '='.repeat(60))
    // console.log('Example 3: Streaming with Thinking')
    // console.log('='.repeat(60))
    // await streamingWithThinking()
}

main().catch(console.error)
