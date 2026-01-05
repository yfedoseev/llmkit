/**
 * Multiple Providers Example
 *
 * Demonstrates how to configure and use multiple LLM providers.
 *
 * Requirements:
 * - Set multiple provider API keys (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
 *
 * Run:
 *   npx ts-node 07-multiple-providers.ts
 */

import {
    JsModelSuiteClient as ModelSuiteClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
    getModelInfo,
    getCheapestModel,
} from 'modelsuite'

async function usingFromEnv() {
    const client = ModelSuiteClient.fromEnv()

    // List all detected providers
    const providers = client.providers()
    console.log('Detected providers:', providers)
    console.log('Default provider:', client.defaultProvider)

    // Use the default provider with explicit provider/model format
    const response = await client.complete(
        CompletionRequest
            .create('anthropic/claude-sonnet-4-20250514', [Message.user('Say hello')])
            .withMaxTokens(50)
    )
    console.log('\nDefault provider response:', response.textContent())
}

function explicitProviderConfig() {
    // Configure specific providers
    const client = new ModelSuiteClient({
        providers: {
            anthropic: { apiKey: 'your-anthropic-key' },
            openai: { apiKey: 'your-openai-key' },
            // Azure requires additional config
            azure: {
                apiKey: 'your-azure-key',
                endpoint: 'https://your-resource.openai.azure.com',
                deployment: 'gpt-4',
            },
            // Bedrock uses AWS credentials
            bedrock: { region: 'us-east-1' },
            // Local Ollama
            ollama: { baseUrl: 'http://localhost:11434' },
        },
        defaultProvider: 'anthropic',
    })

    console.log('Configured providers:', client.providers())
}

async function switchBetweenProviders() {
    const client = ModelSuiteClient.fromEnv()

    const providers = client.providers()
    console.log('Available:', providers, '\n')

    const prompt = "What's 2+2? Answer with just the number."

    // Using the unified "provider/model" format - much cleaner!
    const models = [
        'anthropic/claude-sonnet-4-20250514',
        'openai/gpt-4o',
    ]

    // Try different providers if available
    for (const model of models) {
        const provider = model.split('/')[0]
        if (!providers.includes(provider)) {
            console.log(`${model}: Not configured`)
            continue
        }

        try {
            // Use unified format - provider is embedded in model string
            const response = await client.complete(
                CompletionRequest
                    .create(model, [Message.user(prompt)])
                    .withMaxTokens(20)
            )
            console.log(`${model}: ${response.textContent().trim()}`)
        } catch (e) {
            console.log(`${model}: Error - ${e}`)
        }
    }
}

function costAwareRouting() {
    // Find the cheapest model that meets requirements
    const cheapest = getCheapestModel(null, false, true)

    if (cheapest) {
        console.log('Cheapest model with tools:', cheapest.name)
        console.log('  Provider:', cheapest.provider)
        console.log('  Price: $' + cheapest.pricing.inputPer1M + '/1M input tokens')
    }

    // Compare costs for specific models
    const modelsToCompare = [
        'claude-sonnet-4-20250514',
        'gpt-4o',
        'gpt-4o-mini',
    ]

    console.log('\nModel cost comparison:')
    for (const modelId of modelsToCompare) {
        const info = getModelInfo(modelId)
        if (info) {
            const cost = info.estimateCost(1000, 500)  // 1000 input, 500 output
            console.log(
                `  ${info.name}: $${cost.toFixed(6)} ` +
                `($${info.pricing.inputPer1M}/1M in, $${info.pricing.outputPer1M}/1M out)`
            )
        }
    }
}

async function providerFallback() {
    const client = ModelSuiteClient.fromEnv()

    // Order providers by preference using unified "provider/model" format
    const modelPriority = [
        'anthropic/claude-sonnet-4-20250514',
        'openai/gpt-4o',
    ]

    const available = new Set(client.providers())

    for (const model of modelPriority) {
        const provider = model.split('/')[0]
        if (!available.has(provider)) {
            console.log(`Skipping ${model} (not configured)`)
            continue
        }

        try {
            console.log(`Trying ${model}...`)
            const response = await client.complete(
                CompletionRequest
                    .create(model, [Message.user('What is Python?')])
                    .withMaxTokens(100)
            )
            console.log(`Success with ${model}!`)
            console.log(`Response: ${response.textContent().slice(0, 100)}...`)
            return
        } catch (e) {
            console.log(`Failed with ${model}: ${e}`)
            continue
        }
    }

    console.log('All providers failed!')
}

async function main() {
    console.log('='.repeat(50))
    console.log('Example 1: Auto-detect from Environment')
    console.log('='.repeat(50))
    await usingFromEnv()

    console.log('\n' + '='.repeat(50))
    console.log('Example 2: Switch Between Providers')
    console.log('='.repeat(50))
    await switchBetweenProviders()

    console.log('\n' + '='.repeat(50))
    console.log('Example 3: Cost-Aware Routing')
    console.log('='.repeat(50))
    costAwareRouting()

    console.log('\n' + '='.repeat(50))
    console.log('Example 4: Provider Fallback')
    console.log('='.repeat(50))
    await providerFallback()
}

main().catch(console.error)
