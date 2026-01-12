/**
 * Example: Specialized APIs with LLMKit TypeScript bindings
 * Demonstrates: Ranking, Reranking, Moderation, Classification
 */

import {
  LLMKitClient,
  RankingRequest,
  RerankingRequest,
  ModerationRequest,
  ClassificationRequest,
} from '../index'

async function main() {
  console.log('🔧 LLMKit Specialized APIs Example')
  console.log('='.repeat(50))

  const client = LLMKitClient.fromEnv()

  // 1. RANKING
  console.log('\n--- Example 1: Document Ranking ---')
  const documents = [
    'Python is a general-purpose programming language',
    'Java is used for enterprise software',
    'Python was created by Guido van Rossum',
  ]
  const rankReq = new RankingRequest('Python programming', documents)
  const rankResp = await client.rankDocuments(rankReq)
  console.log(`Query: ${rankReq.query}`)
  const topRank = rankResp.first()
  if (topRank) {
    console.log(`Top result: ${topRank.document} (score: ${topRank.score.toFixed(3)})`)
  }

  // 2. RERANKING
  console.log('\n--- Example 2: Search Result Reranking ---')
  const searchResults = [
    'Random document A',
    'Python best practices guide',
    'Unrelated content',
  ]
  const rerankReq = new RerankingRequest('Python best practices', searchResults).with_top_n(
    2
  )
  const rerankResp = await client.rerankResults(rerankReq)
  console.log(`Query: ${rerankReq.query}`)
  for (const result of rerankResp.results) {
    console.log(`  ${result.document}: ${result.relevance_score.toFixed(3)}`)
  }

  // 3. MODERATION
  console.log('\n--- Example 3: Content Moderation ---')
  const textsToModerate = [
    'This is a great product!',
    'I really enjoyed the experience',
    'This service is acceptable',
  ]

  for (const text of textsToModerate) {
    const modReq = new ModerationRequest(text)
    const modResp = await client.moderateText(modReq)
    const status = !modResp.flagged ? '✓ OK' : '✗ FLAGGED'
    console.log(`${status}: ${text.substring(0, 40)}...`)
  }

  // 4. CLASSIFICATION
  console.log('\n--- Example 4: Text Classification ---')
  const labels = ['positive', 'negative', 'neutral']
  const reviews = [
    'Amazing product, highly recommend!',
    'Terrible quality, very disappointed',
    "It's okay, nothing special",
  ]

  for (const review of reviews) {
    const classReq = new ClassificationRequest(review, labels)
    const classResp = await client.classifyText(classReq)
    const top = classResp.top()
    if (top) {
      console.log(
        `${top.label.padEnd(10)} | ${review} (${(top.confidence * 100).toFixed(0)}%)`
      )
    }
  }

  // 5. COMPLETE WORKFLOW
  console.log('\n--- Example 5: Complete Workflow ---')
  console.log('Processing user comments...')

  const userComments = [
    'This is fantastic! Best purchase ever!',
    'Very disappointed with quality',
    'It works as described',
  ]

  for (const comment of userComments) {
    // Check moderation
    const modReq = new ModerationRequest(comment)
    const modResp = await client.moderateText(modReq)

    if (modResp.flagged) {
      console.log(`⚠️  REJECTED: ${comment.substring(0, 40)}... (violates policy)`)
      continue
    }

    // Classify sentiment
    const classReq = new ClassificationRequest(comment, [
      'positive',
      'negative',
      'neutral',
    ])
    const classResp = await client.classifyText(classReq)
    const sentiment = classResp.top()

    if (sentiment) {
      const percent = (sentiment.confidence * 100).toFixed(0)
      console.log(
        `✓ ${sentiment.label.toUpperCase().padEnd(8)} | ${comment.substring(0, 40)}... (${percent}%)`
      )
    }
  }

  console.log('\n' + '='.repeat(50))
  console.log('✓ Specialized API examples completed!')
}

main().catch((err) => {
  console.error('Error:', err.message)
  process.exit(1)
})
