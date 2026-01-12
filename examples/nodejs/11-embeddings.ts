/**
 * Embeddings Example
 *
 * Demonstrates text embedding generation and similarity computation.
 *
 * Requirements:
 * - Set OPENAI_API_KEY environment variable (for text-embedding-3-small)
 *   Or COHERE_API_KEY for Cohere embeddings
 *
 * Run:
 *   npx ts-node 10-embeddings.ts
 */

import {
    JsLLMKitClient as LLMKitClient,
    JsEmbeddingRequest as EmbeddingRequest,
} from 'llmkit'

async function basicEmbedding() {
    const client = LLMKitClient.fromEnv()

    // Create embedding request using "provider/model" format
    const request = new EmbeddingRequest('openai/text-embedding-3-small', 'Hello, world!')

    console.log('Generating embedding...')
    const response = await client.embed(request)

    console.log('Model:', response.model)
    console.log('Dimensions:', response.dimensionCount)

    // Get the embedding values
    const values = response.values()
    if (values) {
        console.log('First 5 values:', values.slice(0, 5))
        console.log('Total values:', values.length)
    }

    // Usage info
    console.log('Tokens used:', response.usage.totalTokens)
}

async function batchEmbeddings() {
    const client = LLMKitClient.fromEnv()

    const texts = [
        'The quick brown fox',
        'A lazy dog sleeps',
        'Python is a programming language',
        'Machine learning is fascinating',
    ]

    // Create batch request using "provider/model" format
    const request = EmbeddingRequest.batch('openai/text-embedding-3-small', texts)

    console.log(`Generating embeddings for ${texts.length} texts...`)
    const response = await client.embed(request)

    console.log(`Got ${response.embeddings.length} embeddings`)
    console.log('Dimensions:', response.dimensionCount)

    for (const emb of response.embeddings) {
        console.log(`  Text ${emb.index}: ${emb.dimensionCount} dimensions`)
    }
}

async function computeSimilarity() {
    const client = LLMKitClient.fromEnv()

    const texts = [
        'I love programming in Python',
        'Python coding is my favorite',
        'The weather is nice today',
        'I enjoy writing code',
    ]

    console.log('Computing similarities...')

    // Generate embeddings using "provider/model" format
    const request = EmbeddingRequest.batch('openai/text-embedding-3-small', texts)
    const response = await client.embed(request)

    const embeddings = response.embeddings

    // Compare first text to all others
    const reference = embeddings[0]
    console.log(`\nReference: '${texts[0]}'`)
    console.log('\nSimilarities:')

    for (let i = 1; i < embeddings.length; i++) {
        const similarity = reference.cosineSimilarity(embeddings[i])
        console.log(`  vs '${texts[i]}': ${similarity.toFixed(4)}`)
    }
}

async function semanticSearch() {
    const client = LLMKitClient.fromEnv()

    // Document corpus
    const documents = [
        'Python is a high-level programming language',
        'Machine learning uses algorithms to learn from data',
        'The Eiffel Tower is located in Paris, France',
        'Deep learning is a subset of machine learning',
        'JavaScript is commonly used for web development',
        'Natural language processing deals with text and speech',
    ]

    // Search query
    const query = 'What is artificial intelligence?'

    console.log(`Query: '${query}'`)
    console.log('\nSearching documents...')

    // Embed query using "provider/model" format
    const queryResponse = await client.embed(
        new EmbeddingRequest('openai/text-embedding-3-small', query)
    )
    const queryEmbedding = queryResponse.embeddings[0]

    // Embed documents
    const docResponse = await client.embed(
        EmbeddingRequest.batch('openai/text-embedding-3-small', documents)
    )

    // Compute similarities and rank
    const results: [number, string][] = []
    for (let i = 0; i < docResponse.embeddings.length; i++) {
        const similarity = queryEmbedding.cosineSimilarity(docResponse.embeddings[i])
        results.push([similarity, documents[i]])
    }

    // Sort by similarity (highest first)
    results.sort((a, b) => b[0] - a[0])

    console.log('\nResults (by relevance):')
    for (const [similarity, doc] of results) {
        console.log(`  [${similarity.toFixed(4)}] ${doc}`)
    }
}

async function differentEmbeddingModels() {
    const client = LLMKitClient.fromEnv()

    const text = 'Hello, world!'

    // OpenAI small model using "provider/model" format
    console.log('OpenAI text-embedding-3-small:')
    try {
        const response = await client.embed(
            new EmbeddingRequest('openai/text-embedding-3-small', text)
        )
        console.log('  Dimensions:', response.dimensionCount)
    } catch (e) {
        console.log('  Error:', e)
    }

    // OpenAI small model with reduced dimensions
    console.log('\nOpenAI text-embedding-3-small (256 dims):')
    try {
        const response = await client.embed(
            new EmbeddingRequest('openai/text-embedding-3-small', text)
                .withDimensions(256)
        )
        console.log('  Dimensions:', response.dimensionCount)
    } catch (e) {
        console.log('  Error:', e)
    }

    // OpenAI large model
    console.log('\nOpenAI text-embedding-3-large:')
    try {
        const response = await client.embed(
            new EmbeddingRequest('openai/text-embedding-3-large', text)
        )
        console.log('  Dimensions:', response.dimensionCount)
    } catch (e) {
        console.log('  Error:', e)
    }
}

async function distanceMetrics() {
    const client = LLMKitClient.fromEnv()

    const texts = ['Hello world', 'Hello there']

    const response = await client.embed(
        EmbeddingRequest.batch('openai/text-embedding-3-small', texts)
    )

    const emb1 = response.embeddings[0]
    const emb2 = response.embeddings[1]

    console.log(`Text 1: '${texts[0]}'`)
    console.log(`Text 2: '${texts[1]}'`)
    console.log()
    console.log('Distance metrics:')
    console.log(`  Cosine similarity: ${emb1.cosineSimilarity(emb2).toFixed(6)}`)
    console.log(`  Dot product: ${emb1.dotProduct(emb2).toFixed(6)}`)
    console.log(`  Euclidean distance: ${emb1.euclideanDistance(emb2).toFixed(6)}`)
}

async function main() {
    console.log('='.repeat(50))
    console.log('Example 1: Basic Embedding')
    console.log('='.repeat(50))
    await basicEmbedding()

    console.log('\n' + '='.repeat(50))
    console.log('Example 2: Batch Embeddings')
    console.log('='.repeat(50))
    await batchEmbeddings()

    console.log('\n' + '='.repeat(50))
    console.log('Example 3: Similarity Computation')
    console.log('='.repeat(50))
    await computeSimilarity()

    console.log('\n' + '='.repeat(50))
    console.log('Example 4: Semantic Search')
    console.log('='.repeat(50))
    await semanticSearch()

    console.log('\n' + '='.repeat(50))
    console.log('Example 5: Distance Metrics')
    console.log('='.repeat(50))
    await distanceMetrics()
}

main().catch(console.error)
