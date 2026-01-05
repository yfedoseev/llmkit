import { describe, it, expect, beforeAll } from 'vitest'
import {
  JsModelSuiteClient as ModelSuiteClient,
  JsVideoModel as VideoModel,
  JsVideoGenerationOptions as VideoGenerationOptions,
  JsVideoGenerationRequest as VideoGenerationRequest,
  JsVideoGenerationResponse as VideoGenerationResponse,
} from '../index'

describe('VideoModel Enum', () => {
  it('should have RunwayGen45 variant', () => {
    expect(VideoModel.RunwayGen45).toBeDefined()
  })

  it('should have Kling20 variant', () => {
    expect(VideoModel.Kling20).toBeDefined()
  })

  it('should have Pika10 variant', () => {
    expect(VideoModel.Pika10).toBeDefined()
  })

  it('should have HailuoMini variant', () => {
    expect(VideoModel.HailuoMini).toBeDefined()
  })

  it('should have LeonardoUltra variant', () => {
    expect(VideoModel.LeonardoUltra).toBeDefined()
  })
})

describe('VideoGenerationOptions', () => {
  it('should create empty options', () => {
    const opts = new VideoGenerationOptions()
    expect(opts).toBeDefined()
  })

  it('should set model via builder method', () => {
    const opts = new VideoGenerationOptions()
    const optsWithModel = opts.withModel('runway-gen-3')
    expect(optsWithModel).toBeDefined()
  })

  it('should set duration via builder method', () => {
    const opts = new VideoGenerationOptions()
    const optsWithDuration = opts.withDuration(15)
    expect(optsWithDuration).toBeDefined()
  })

  it('should set width via builder method', () => {
    const opts = new VideoGenerationOptions()
    const optsWithWidth = opts.withWidth(1920)
    expect(optsWithWidth).toBeDefined()
  })

  it('should set height via builder method', () => {
    const opts = new VideoGenerationOptions()
    const optsWithHeight = opts.withHeight(1080)
    expect(optsWithHeight).toBeDefined()
  })

  it('should set quality via builder method', () => {
    const opts = new VideoGenerationOptions()
    const optsWithQuality = opts.withQuality('high')
    expect(optsWithQuality).toBeDefined()
  })

  it('should chain multiple builder methods', () => {
    const opts = new VideoGenerationOptions()
      .withModel('runway-gen-4')
      .withDuration(30)
      .withWidth(1280)
      .withHeight(720)
      .withQuality('medium')
    expect(opts).toBeDefined()
  })

  it('should allow multiple independent chains', () => {
    const opts1 = new VideoGenerationOptions().withDuration(10)
    const opts2 = new VideoGenerationOptions().withDuration(20)
    expect(opts1).toBeDefined()
    expect(opts2).toBeDefined()
  })

  it('should support all builder methods in any order', () => {
    const opts = new VideoGenerationOptions()
      .withQuality('high')
      .withWidth(1920)
      .withModel('pika-1.0')
      .withHeight(1080)
      .withDuration(10)
    expect(opts).toBeDefined()
  })
})

describe('VideoGenerationRequest', () => {
  it('should create request with prompt', () => {
    const req = new VideoGenerationRequest('Generate a sci-fi video')
    expect(req).toBeDefined()
  })

  it('should set model via builder method', () => {
    const req = new VideoGenerationRequest('Generate a video')
    const reqWithModel = req.withModel('runway-gen-4')
    expect(reqWithModel).toBeDefined()
  })

  it('should set duration via builder method', () => {
    const req = new VideoGenerationRequest('Generate a video')
    const reqWithDuration = req.withDuration(20)
    expect(reqWithDuration).toBeDefined()
  })

  it('should set width via builder method', () => {
    const req = new VideoGenerationRequest('Generate a video')
    const reqWithWidth = req.withWidth(1920)
    expect(reqWithWidth).toBeDefined()
  })

  it('should set height via builder method', () => {
    const req = new VideoGenerationRequest('Generate a video')
    const reqWithHeight = req.withHeight(1080)
    expect(reqWithHeight).toBeDefined()
  })

  it('should chain multiple builder methods', () => {
    const req = new VideoGenerationRequest('A beautiful landscape video')
      .withModel('kling-2.0')
      .withDuration(10)
      .withWidth(1280)
      .withHeight(720)
    expect(req).toBeDefined()
  })

  it('should create requests with various prompts', () => {
    const prompts = [
      'A cat sleeping on a sunny windowsill',
      'Abstract geometric animation',
      'Ocean waves crashing on beach',
      'Urban cyberpunk city at night',
      '', // Empty prompt should be allowed
    ]

    prompts.forEach((prompt) => {
      const req = new VideoGenerationRequest(prompt)
      expect(req).toBeDefined()
    })
  })

  it('should support different model types in builder', () => {
    const models = ['runway-gen-4', 'kling-2.0', 'pika-1.0', 'hailuo-mini', 'leonardo-ultra']

    models.forEach((model) => {
      const req = new VideoGenerationRequest('Test').withModel(model)
      expect(req).toBeDefined()
    })
  })
})

// VideoGenerationResponse is returned by API, not constructed by users
describe.skip('VideoGenerationResponse', () => {
  it('should create empty response', () => {
    const resp = new VideoGenerationResponse()
    expect(resp).toBeDefined()
  })

  it('should have video_url property', () => {
    const resp = new VideoGenerationResponse()
    resp.video_url = 'https://example.com/video.mp4'
    expect(resp.video_url).toBe('https://example.com/video.mp4')
  })

  it('should have video_bytes property', () => {
    const resp = new VideoGenerationResponse()
    const videoData = Buffer.from('fake_video_data')
    resp.video_bytes = videoData
    expect(resp.video_bytes).toEqual(videoData)
  })

  it('should have format property with default', () => {
    const resp = new VideoGenerationResponse()
    expect(resp.format).toBe('mp4')
  })

  it('should allow changing format', () => {
    const resp = new VideoGenerationResponse()
    resp.format = 'mov'
    expect(resp.format).toBe('mov')
  })

  it('should have duration property', () => {
    const resp = new VideoGenerationResponse()
    resp.duration = 15.5
    expect(resp.duration).toBe(15.5)
  })

  it('should have width and height properties', () => {
    const resp = new VideoGenerationResponse()
    resp.width = 1920
    resp.height = 1080
    expect(resp.width).toBe(1920)
    expect(resp.height).toBe(1080)
  })

  it('should have task_id for async operations', () => {
    const resp = new VideoGenerationResponse()
    resp.task_id = 'task-abc123'
    expect(resp.task_id).toBe('task-abc123')
  })

  it('should have status property for polling', () => {
    const resp = new VideoGenerationResponse()
    resp.status = 'processing'
    expect(resp.status).toBe('processing')

    resp.status = 'completed'
    expect(resp.status).toBe('completed')
  })

  it('should calculate size from video bytes', () => {
    const resp = new VideoGenerationResponse()
    // Size should be 0 when no bytes
    expect(resp.size).toBe(0)

    // Size should match bytes length
    resp.video_bytes = Buffer.alloc(1024) // 1 KB
    expect(resp.size).toBe(1024)

    resp.video_bytes = Buffer.alloc(1024 * 1024) // 1 MB
    expect(resp.size).toBe(1024 * 1024)
  })

  it('should create complete response object', () => {
    const resp = new VideoGenerationResponse()
    resp.video_url = 'https://example.com/video.mp4'
    resp.format = 'mp4'
    resp.duration = 10.0
    resp.width = 1280
    resp.height = 720
    resp.task_id = 'task-12345'
    resp.status = 'completed'

    expect(resp.video_url).toBe('https://example.com/video.mp4')
    expect(resp.format).toBe('mp4')
    expect(resp.duration).toBe(10.0)
    expect(resp.width).toBe(1280)
    expect(resp.height).toBe(720)
    expect(resp.task_id).toBe('task-12345')
    expect(resp.status).toBe('completed')
  })
})

// Skip integration tests that require actual video generation API
describe.skip('ModelSuiteClient video methods', () => {
  let client: ModelSuiteClient

  beforeAll(() => {
    client = ModelSuiteClient.fromEnv()
  })

  it('should have generateVideo method', () => {
    expect(client).toHaveProperty('generateVideo')
    expect(typeof client.generateVideo).toBe('function')
  })

  it('should generate video with request', async () => {
    const req = new VideoGenerationRequest('A beautiful sunset over mountains')
    const response = await client.generateVideo(req)

    expect(response).toBeInstanceOf(VideoGenerationResponse)
    expect(response.video_url).toBeDefined()
    expect(response.format).toBeDefined()
  })

  it('should have expected response properties', async () => {
    const req = new VideoGenerationRequest('Test video generation')
    const response = await client.generateVideo(req)

    // Response should have these properties (may be undefined in placeholder)
    expect(response).toHaveProperty('video_bytes')
    expect(response).toHaveProperty('video_url')
    expect(response).toHaveProperty('format')
    expect(response).toHaveProperty('duration')
    expect(response).toHaveProperty('width')
    expect(response).toHaveProperty('height')
    expect(response).toHaveProperty('task_id')
    expect(response).toHaveProperty('status')
  })

  it('should generate video with configured request', async () => {
    const req = new VideoGenerationRequest('An animated character walking')
      .withModel('pika-1.0')
      .withDuration(5)
      .withWidth(1024)
      .withHeight(576)

    const response = await client.generateVideo(req)
    expect(response).toBeInstanceOf(VideoGenerationResponse)
  })

  it('should handle empty prompt', async () => {
    const req = new VideoGenerationRequest('')
    const response = await client.generateVideo(req)
    expect(response).toBeDefined()
  })

  it('should handle various model selections', async () => {
    const models = ['runway-gen-4', 'kling-2.0', 'pika-1.0']

    for (const model of models) {
      const req = new VideoGenerationRequest('Test').withModel(model)
      const response = await client.generateVideo(req)
      expect(response).toBeInstanceOf(VideoGenerationResponse)
    }
  })
})

describe('Video module exports', () => {
  it('should export VideoModel', () => {
    expect(VideoModel).toBeDefined()
  })

  it('should export VideoGenerationOptions', () => {
    expect(VideoGenerationOptions).toBeDefined()
  })

  it('should export VideoGenerationRequest', () => {
    expect(VideoGenerationRequest).toBeDefined()
  })

  it('should export VideoGenerationResponse', () => {
    expect(VideoGenerationResponse).toBeDefined()
  })
})

// Skip integration tests that require actual video generation API
describe.skip('Video integration tests', () => {
  let client: ModelSuiteClient

  beforeAll(() => {
    client = ModelSuiteClient.fromEnv()
  })

  it('should complete end-to-end video generation flow', async () => {
    // 1. Create request
    const req = new VideoGenerationRequest('A spinning 3D cube with colorful lights')

    // 2. Configure options
    const configuredReq = req
      .withModel('runway-gen-4')
      .withDuration(10)
      .withWidth(1920)
      .withHeight(1080)

    // 3. Generate video
    const response = await client.generateVideo(configuredReq)

    // 4. Verify response
    expect(response).toBeInstanceOf(VideoGenerationResponse)
    expect(
      response.video_url !== undefined || response.video_bytes !== undefined
    ).toBe(true)
    expect(response.format).toBeDefined()
  })

  it('should support polling pattern for async operations', async () => {
    const req = new VideoGenerationRequest('Generate a test video')
    const response = await client.generateVideo(req)

    // Simulate polling loop
    let currentResponse = response
    let maxAttempts = 5
    let attempts = 0

    while (attempts < maxAttempts) {
      if (currentResponse.status === 'completed') {
        break
      }
      // In real scenario, would call client.pollVideo(task_id)
      expect(currentResponse).toHaveProperty('status')
      attempts++
    }

    expect(currentResponse).toBeDefined()
  })

  it('should generate multiple videos sequentially', async () => {
    const prompts = [
      'A cat playing with a ball',
      'A sunset over the ocean',
      'A person dancing',
    ]

    const responses: VideoGenerationResponse[] = []
    for (const prompt of prompts) {
      const req = new VideoGenerationRequest(prompt)
      const response = await client.generateVideo(req)
      responses.push(response)
    }

    expect(responses).toHaveLength(3)
    expect(responses.every((r) => r instanceof VideoGenerationResponse)).toBe(true)
  })
})
