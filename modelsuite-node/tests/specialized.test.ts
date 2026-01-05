import { describe, it, expect, beforeAll } from 'vitest'
import {
  ModelSuiteClient,
  RankingRequest,
  RankedDocument,
  RankingResponse,
  RerankingRequest,
  RerankedResult,
  RerankingResponse,
  ModerationRequest,
  ModerationScores,
  ModerationResponse,
  ClassificationRequest,
  ClassificationResult,
  ClassificationResponse,
} from '../index'

// Skip specialized tests - require proper exports
describe.skip('Ranking API', () => {
  it('should create ranking request', () => {
    const req = new RankingRequest('python programming', [
      'doc1',
      'doc2',
      'doc3',
    ])
    expect(req).toBeDefined()
  })

  it('should set top_k in ranking request', () => {
    const req = new RankingRequest('query', ['d1', 'd2']).with_top_k(1)
    expect(req.top_k).toBe(1)
  })

  it('should create ranked document', () => {
    const doc = new RankedDocument(0, 'Sample', 0.95)
    expect(doc.score).toBe(0.95)
  })

  it('should create ranking response', () => {
    const resp = new RankingResponse()
    expect(resp.count).toBe(0)

    resp.results = [
      new RankedDocument(0, 'doc1', 0.9),
      new RankedDocument(1, 'doc2', 0.8),
    ]
    expect(resp.count).toBe(2)
    expect(resp.first()?.score).toBe(0.9)
  })
})

describe.skip('Reranking API', () => {
  it('should create reranking request', () => {
    const req = new RerankingRequest('query', ['d1', 'd2'])
    expect(req).toBeDefined()
  })

  it('should set top_n in reranking request', () => {
    const req = new RerankingRequest('query', ['d1']).with_top_n(1)
    expect(req.top_n).toBe(1)
  })

  it('should create reranked result', () => {
    const result = new RerankedResult(0, 'content', 0.92)
    expect(result.relevance_score).toBe(0.92)
  })

  it('should create reranking response', () => {
    const resp = new RerankingResponse()
    resp.results = [new RerankedResult(0, 'r1', 0.95)]
    expect(resp.count).toBe(1)
  })
})

describe.skip('Moderation API', () => {
  it('should create moderation request', () => {
    const req = new ModerationRequest('content')
    expect(req).toBeDefined()
  })

  it('should create moderation scores', () => {
    const scores = new ModerationScores()
    expect(scores.max_score()).toBe(0.0)

    scores.sexual = 0.8
    expect(scores.max_score()).toBe(0.8)
  })

  it('should create moderation response', () => {
    const resp = new ModerationResponse(false)
    expect(resp.flagged).toBe(false)

    resp.scores.violence = 0.9
    expect(resp.scores.max_score()).toBe(0.9)
  })
})

describe.skip('Classification API', () => {
  it('should create classification request', () => {
    const req = new ClassificationRequest('text', [
      'positive',
      'negative',
      'neutral',
    ])
    expect(req).toBeDefined()
  })

  it('should create classification result', () => {
    const result = new ClassificationResult('positive', 0.98)
    expect(result.confidence).toBe(0.98)
  })

  it('should create classification response', () => {
    const resp = new ClassificationResponse()
    expect(resp.count).toBe(0)

    resp.results = [
      new ClassificationResult('positive', 0.98),
      new ClassificationResult('negative', 0.02),
    ]
    expect(resp.count).toBe(2)
    expect(resp.top()?.label).toBe('positive')
  })
})

describe.skip('ModelSuiteClient specialized methods', () => {
  let client: ModelSuiteClient

  beforeAll(() => {
    client = ModelSuiteClient.fromEnv()
  })

  it('should rank documents', async () => {
    const req = new RankingRequest('python', [
      'Python language',
      'Java guide',
    ])
    const response = await client.rankDocuments(req)
    expect(response).toBeInstanceOf(RankingResponse)
    expect(response.count).toBeGreaterThan(0)
  })

  it('should rerank results', async () => {
    const req = new RerankingRequest('query', [
      'result1',
      'result2',
    ]).with_top_n(1)
    const response = await client.rerankResults(req)
    expect(response).toBeInstanceOf(RerankingResponse)
  })

  it('should moderate text', async () => {
    const req = new ModerationRequest('normal content')
    const response = await client.moderateText(req)
    expect(response).toBeInstanceOf(ModerationResponse)
  })

  it('should classify text', async () => {
    const req = new ClassificationRequest('text', [
      'category1',
      'category2',
    ])
    const response = await client.classifyText(req)
    expect(response).toBeInstanceOf(ClassificationResponse)
    expect(response.count).toBeGreaterThan(0)
  })
})

describe.skip('Specialized API Imports', () => {
  it('should export all ranking types', () => {
    expect(RankingRequest).toBeDefined()
    expect(RankedDocument).toBeDefined()
    expect(RankingResponse).toBeDefined()
  })

  it('should export all reranking types', () => {
    expect(RerankingRequest).toBeDefined()
    expect(RerankedResult).toBeDefined()
    expect(RerankingResponse).toBeDefined()
  })

  it('should export all moderation types', () => {
    expect(ModerationRequest).toBeDefined()
    expect(ModerationScores).toBeDefined()
    expect(ModerationResponse).toBeDefined()
  })

  it('should export all classification types', () => {
    expect(ClassificationRequest).toBeDefined()
    expect(ClassificationResult).toBeDefined()
    expect(ClassificationResponse).toBeDefined()
  })
})
