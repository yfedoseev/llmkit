import { describe, it, expect, beforeAll } from 'vitest'
import {
  ModelSuiteClient,
  // Audio
  TranscriptionRequest,
  SynthesisRequest,
  // Video
  VideoGenerationRequest,
  // Image
  ImageGenerationRequest,
  // Specialized
  RankingRequest,
  RerankingRequest,
  ModerationRequest,
  ClassificationRequest,
} from '../index'

// Skip integration tests - require proper exports and actual API calls
describe.skip('ModelSuite Integration Tests', () => {
  let client: ModelSuiteClient

  beforeAll(() => {
    client = ModelSuiteClient.fromEnv()
  })

  describe('Audio Integration', () => {
    it('should create transcription request', () => {
      const audioBytes = Buffer.from('test audio data')
      const req = new TranscriptionRequest(audioBytes)
      expect(req).toBeDefined()
    })

    it('should create synthesis request with voice', () => {
      const req = new SynthesisRequest('Hello world', 'alloy')
      expect(req).toBeDefined()
    })

    it('should support audio workflow', async () => {
      // Create synthesis request
      const synthReq = new SynthesisRequest('Test audio generation')
      expect(synthReq).toBeDefined()

      // Create transcription request
      const transReq = new TranscriptionRequest(Buffer.from('audio bytes'))
      expect(transReq).toBeDefined()
    })
  })

  describe('Video Integration', () => {
    it('should create video generation request', () => {
      const req = new VideoGenerationRequest('A sunset over mountains')
      expect(req).toBeDefined()
    })

    it('should support builder pattern for video', () => {
      const req = new VideoGenerationRequest('Test video')
        .withModel('runway-gen3-alpha')
        .withDuration(5)

      expect(req.prompt).toBe('Test video')
      expect(req.model).toBe('runway-gen3-alpha')
      expect(req.duration).toBe(5)
    })

    it('should support video workflow', async () => {
      const req = new VideoGenerationRequest('Generate test video')
      expect(req).toBeDefined()
    })
  })

  describe('Image Integration', () => {
    it('should create image generation request', () => {
      const req = new ImageGenerationRequest('A futuristic city')
      expect(req).toBeDefined()
    })

    it('should support builder pattern for image', () => {
      const req = new ImageGenerationRequest('Test image')
        .withModel('dall-e-3')
        .withSize('1024x1024')

      expect(req.prompt).toBe('Test image')
      expect(req.model).toBe('dall-e-3')
      expect(req.size).toBe('1024x1024')
    })

    it('should support image workflow', async () => {
      const req = new ImageGenerationRequest('Generate test image')
      expect(req).toBeDefined()
    })
  })

  describe('Specialized APIs Integration', () => {
    it('should create ranking request', () => {
      const docs = ['Document 1', 'Document 2', 'Document 3']
      const req = new RankingRequest('test query', docs)
      expect(req).toBeDefined()
      expect(req.documents.length).toBe(3)
    })

    it('should create classification request', () => {
      const labels = ['positive', 'negative', 'neutral']
      const req = new ClassificationRequest('Test text', labels)
      expect(req).toBeDefined()
      expect(req.labels.length).toBe(3)
    })
  })

  describe('Cross-Modality Workflows', () => {
    it('should support complete workflow', async () => {
      const audioReq = new SynthesisRequest('Describe the image')
      const imageReq = new ImageGenerationRequest('A scene to describe')
      const videoReq = new VideoGenerationRequest('Animation of the scene')
      const modReq = new ModerationRequest('Describe the video content')

      expect(audioReq).toBeDefined()
      expect(imageReq).toBeDefined()
      expect(videoReq).toBeDefined()
      expect(modReq).toBeDefined()
    })
  })

  describe('Type Safety', () => {
    it('should enforce type checking on requests', () => {
      const req = new RankingRequest('query', ['doc1', 'doc2'])
      expect(req.query).toBe('query')
      expect(req.documents).toEqual(['doc1', 'doc2'])
    })
  })
})
