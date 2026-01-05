/**
 * Batch Processing Example
 *
 * Demonstrates the batch API for processing multiple requests asynchronously.
 *
 * Requirements:
 * - Set ANTHROPIC_API_KEY environment variable
 *
 * Run:
 *   npx ts-node 09-batch-processing.ts
 */

import {
    JsModelSuiteClient as ModelSuiteClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
    JsBatchRequest as BatchRequest,
} from 'modelsuite'

async function createBatch(): Promise<string> {
    const client = ModelSuiteClient.fromEnv()

    // Create multiple completion requests
    const questions: [string, string][] = [
        ['q1', 'What is 2+2?'],
        ['q2', 'What is the capital of France?'],
        ['q3', 'What color is the sky?'],
        ['q4', 'How many days in a week?'],
        ['q5', 'What is H2O?'],
    ]

    // Build batch requests using "provider/model" format
    const batchRequests = questions.map(([customId, question]) =>
        BatchRequest.create(
            customId,
            CompletionRequest
                .create('anthropic/claude-sonnet-4-20250514', [Message.user(question)])
                .withMaxTokens(100)
        )
    )

    console.log(`Submitting batch with ${batchRequests.length} requests...`)

    // Submit the batch
    const batchJob = await client.createBatch(batchRequests)

    console.log('Batch created:')
    console.log('  ID:', batchJob.id)
    console.log('  Status:', batchJob.status)
    console.log('  Total requests:', batchJob.requestCounts.total)

    return batchJob.id
}

async function checkBatchStatus(batchId: string) {
    const client = ModelSuiteClient.fromEnv()

    const batchJob = await client.getBatch('anthropic', batchId)

    console.log(`Batch ${batchId}:`)
    console.log('  Status:', batchJob.status)
    console.log('  Total:', batchJob.requestCounts.total)
    console.log('  Succeeded:', batchJob.requestCounts.succeeded)
    console.log('  Failed:', batchJob.requestCounts.failed)
    console.log('  Pending:', batchJob.requestCounts.pending)
    console.log('  Is complete:', batchJob.isComplete())

    if (batchJob.createdAt) {
        console.log('  Created:', batchJob.createdAt)
    }
    if (batchJob.endedAt) {
        console.log('  Ended:', batchJob.endedAt)
    }

    return batchJob
}

async function getBatchResults(batchId: string) {
    const client = ModelSuiteClient.fromEnv()

    const results = await client.getBatchResults('anthropic', batchId)

    console.log(`\nBatch results (${results.length} items):`)
    for (const result of results) {
        console.log(`\n  ${result.customId}:`)
        if (result.isSuccess()) {
            const response = result.response
            console.log('    Status: Success')
            console.log('    Response:', response?.textContent().slice(0, 100) + '...')
            if (response?.usage) {
                console.log('    Tokens:', response.usage.totalTokens())
            }
        } else {
            const error = result.error
            console.log('    Status: Error')
            console.log('    Error type:', error?.errorType)
            console.log('    Message:', error?.message)
        }
    }
}

async function listBatches() {
    const client = ModelSuiteClient.fromEnv()

    const batches = await client.listBatches('anthropic', 10)

    console.log(`Found ${batches.length} batches:`)
    for (const batch of batches) {
        console.log(`  ${batch.id}: ${batch.status}`)
        console.log(`    Requests: ${batch.requestCounts.total}`)
        console.log(`    Created: ${batch.createdAt}`)
    }
}

async function waitForBatch(batchId: string, timeoutMs: number = 300000) {
    const client = ModelSuiteClient.fromEnv()

    console.log(`Waiting for batch ${batchId} to complete...`)

    const start = Date.now()
    while (true) {
        const batchJob = await client.getBatch('anthropic', batchId)

        if (batchJob.isComplete()) {
            console.log(`Batch completed with status: ${batchJob.status}`)
            return batchJob
        }

        const elapsed = Date.now() - start
        if (elapsed > timeoutMs) {
            console.log('Timeout waiting for batch')
            return batchJob
        }

        console.log(
            `  Status: ${batchJob.status}, ` +
            `Progress: ${batchJob.requestCounts.succeeded}/${batchJob.requestCounts.total}`
        )

        await new Promise(resolve => setTimeout(resolve, 5000))  // Poll every 5 seconds
    }
}

async function fullBatchWorkflow() {
    const client = ModelSuiteClient.fromEnv()

    // 1. Create batch requests
    console.log('Step 1: Creating batch requests...')
    const batchRequests = [
        BatchRequest.create(
            'translate-1',
            CompletionRequest
                .create('anthropic/claude-sonnet-4-20250514', [
                    Message.user("Translate 'Hello' to Spanish")
                ])
                .withMaxTokens(50)
        ),
        BatchRequest.create(
            'translate-2',
            CompletionRequest
                .create('anthropic/claude-sonnet-4-20250514', [
                    Message.user("Translate 'Goodbye' to French")
                ])
                .withMaxTokens(50)
        ),
        BatchRequest.create(
            'translate-3',
            CompletionRequest
                .create('anthropic/claude-sonnet-4-20250514', [
                    Message.user("Translate 'Thank you' to Japanese")
                ])
                .withMaxTokens(50)
        ),
    ]

    // 2. Submit batch
    console.log('\nStep 2: Submitting batch...')
    const batchJob = await client.createBatch(batchRequests)
    console.log('Batch ID:', batchJob.id)

    // 3. Wait for completion
    console.log('\nStep 3: Waiting for completion...')
    const finalJob = await waitForBatch(batchJob.id, 120000)

    if (!finalJob.isComplete()) {
        console.log('Batch did not complete in time')
        return
    }

    // 4. Get results
    console.log('\nStep 4: Getting results...')
    const results = await client.getBatchResults('anthropic', batchJob.id)

    const translations: Record<string, string> = {}
    for (const result of results) {
        if (result.isSuccess()) {
            translations[result.customId] = result.response?.textContent() ?? ''
        }
    }

    console.log('\nTranslations:')
    for (const [requestId, translation] of Object.entries(translations)) {
        console.log(`  ${requestId}: ${translation}`)
    }
}

async function main() {
    console.log('='.repeat(50))
    console.log('Example 1: List Existing Batches')
    console.log('='.repeat(50))
    await listBatches()

    // Uncomment to run full workflow (requires API credits):
    // console.log('\n' + '='.repeat(50))
    // console.log('Example 2: Full Batch Workflow')
    // console.log('='.repeat(50))
    // await fullBatchWorkflow()
}

main().catch(console.error)
