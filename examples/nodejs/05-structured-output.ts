/**
 * Structured Output (JSON Schema) Example
 *
 * Demonstrates how to get structured JSON responses using schema enforcement.
 *
 * Requirements:
 * - Set OPENAI_API_KEY environment variable
 *
 * Run:
 *   npx ts-node 05-structured-output.ts
 */

import {
    JsLlmKitClient as LLMKitClient,
    JsMessage as Message,
    JsCompletionRequest as CompletionRequest,
} from 'modelsuite'

async function simpleJsonOutput() {
    const client = LLMKitClient.fromEnv()

    // Use "provider/model" format for explicit provider routing
    const request = CompletionRequest
        .create('openai/gpt-4o', [
            Message.user(
                'Generate a JSON object representing a fictional book with ' +
                'title, author, year, and genre fields.'
            )
        ])
        .withMaxTokens(200)
        .withJsonOutput()

    console.log('Getting simple JSON output...')
    const response = await client.complete(request)

    try {
        const data = JSON.parse(response.textContent())
        console.log('\nParsed JSON:')
        console.log(JSON.stringify(data, null, 2))
    } catch {
        console.log('\nRaw response:', response.textContent())
    }
}

async function schemaEnforcedOutput() {
    const client = LLMKitClient.fromEnv()

    // Define the expected JSON schema
    const personSchema = {
        type: 'object',
        properties: {
            name: {
                type: 'string',
                description: 'Full name of the person',
            },
            age: {
                type: 'integer',
                description: 'Age in years',
                minimum: 0,
                maximum: 150,
            },
            email: {
                type: 'string',
                description: 'Email address',
            },
            occupation: {
                type: 'string',
                description: 'Current job title',
            },
            skills: {
                type: 'array',
                items: { type: 'string' },
                description: 'List of skills',
            },
        },
        required: ['name', 'age', 'email', 'occupation'],
    }

    const request = CompletionRequest
        .create('openai/gpt-4o', [
            Message.user(
                'Generate a fictional software engineer\'s profile. ' +
                'Make it realistic with appropriate skills.'
            )
        ])
        .withMaxTokens(500)
        .withJsonSchema('person_profile', personSchema)

    console.log('Getting schema-enforced JSON output...')
    const response = await client.complete(request)

    const data = JSON.parse(response.textContent())
    console.log('\nPerson profile:')
    console.log(JSON.stringify(data, null, 2))

    console.log('\nValidation:')
    console.log('  Name:', data.name ?? 'MISSING')
    console.log('  Age:', data.age ?? 'MISSING')
    console.log('  Email:', data.email ?? 'MISSING')
    console.log('  Occupation:', data.occupation ?? 'MISSING')
    if (data.skills) {
        console.log('  Skills:', data.skills.join(', '))
    }
}

async function complexNestedSchema() {
    const client = LLMKitClient.fromEnv()

    // Schema for a product catalog entry
    const productSchema = {
        type: 'object',
        properties: {
            product: {
                type: 'object',
                properties: {
                    id: { type: 'string' },
                    name: { type: 'string' },
                    price: { type: 'number' },
                    currency: { type: 'string', enum: ['USD', 'EUR', 'GBP'] },
                },
                required: ['id', 'name', 'price', 'currency'],
            },
            inventory: {
                type: 'object',
                properties: {
                    quantity: { type: 'integer' },
                    warehouse: { type: 'string' },
                    last_updated: { type: 'string' },
                },
                required: ['quantity', 'warehouse'],
            },
            categories: {
                type: 'array',
                items: { type: 'string' },
            },
        },
        required: ['product', 'inventory'],
    }

    const request = CompletionRequest
        .create('openai/gpt-4o', [
            Message.user(
                'Generate a product entry for a fictional electronics item.'
            )
        ])
        .withMaxTokens(500)
        .withJsonSchema('product_entry', productSchema)

    console.log('Getting complex nested JSON...')
    const response = await client.complete(request)

    const data = JSON.parse(response.textContent())
    console.log('\nProduct entry:')
    console.log(JSON.stringify(data, null, 2))
}

async function listExtraction() {
    const client = LLMKitClient.fromEnv()

    const schema = {
        type: 'object',
        properties: {
            items: {
                type: 'array',
                items: {
                    type: 'object',
                    properties: {
                        name: { type: 'string' },
                        quantity: { type: 'integer' },
                        category: {
                            type: 'string',
                            enum: ['produce', 'dairy', 'meat', 'pantry', 'other'],
                        },
                    },
                    required: ['name', 'quantity', 'category'],
                },
            },
        },
        required: ['items'],
    }

    const text = `
    I need to buy some groceries:
    - 2 pounds of chicken breast
    - A dozen eggs
    - Milk (1 gallon)
    - 5 apples
    - Rice (2 bags)
    - Cheddar cheese
    `

    const request = CompletionRequest
        .create('openai/gpt-4o', [
            Message.user(`Extract the shopping list from this text:\n${text}`)
        ])
        .withMaxTokens(500)
        .withJsonSchema('shopping_list', schema)

    console.log('Extracting structured list from text...')
    const response = await client.complete(request)

    const data = JSON.parse(response.textContent())
    console.log('\nExtracted shopping list:')
    for (const item of data.items ?? []) {
        console.log(`  - ${item.name}: ${item.quantity} (${item.category})`)
    }
}

async function main() {
    console.log('='.repeat(50))
    console.log('Example 1: Simple JSON Output')
    console.log('='.repeat(50))
    await simpleJsonOutput()

    console.log('\n' + '='.repeat(50))
    console.log('Example 2: Schema-Enforced Output')
    console.log('='.repeat(50))
    await schemaEnforcedOutput()

    console.log('\n' + '='.repeat(50))
    console.log('Example 3: Complex Nested Schema')
    console.log('='.repeat(50))
    await complexNestedSchema()

    console.log('\n' + '='.repeat(50))
    console.log('Example 4: List Extraction')
    console.log('='.repeat(50))
    await listExtraction()
}

main().catch(console.error)
