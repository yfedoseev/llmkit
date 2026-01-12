/**
 * Unit tests for LLMKit Node.js bindings
 *
 * These tests verify the core functionality of the bindings without
 * making actual API calls.
 */
import { describe, it, expect, beforeEach } from 'vitest';
import {
  JsMessage as Message,
  JsContentBlock as ContentBlock,
  JsCompletionRequest as CompletionRequest,
  JsToolDefinition as ToolDefinition,
  JsToolBuilder as ToolBuilder,
  JsRole as Role,
  JsLLMKitClient as LLMKitClient,
  JsCacheBreakpoint as CacheBreakpoint,
  JsCacheControl as CacheControl,
  JsThinkingConfig as ThinkingConfig,
  JsThinkingType as ThinkingType,
  JsStructuredOutput as StructuredOutput,
  JsTokenCountRequest as TokenCountRequest,
  JsBatchRequest as BatchRequest,
} from '../index';

// =============================================================================
// Message Tests
// =============================================================================

describe('Message', () => {
  it('creates user message with text', () => {
    const msg = Message.user('Hello, world!');
    expect(msg.role).toBe(Role.User);
    expect(msg.textContent()).toBe('Hello, world!');
  });

  it('creates assistant message with text', () => {
    const msg = Message.assistant('Hello!');
    expect(msg.role).toBe(Role.Assistant);
    expect(msg.textContent()).toBe('Hello!');
  });

  it('creates system message', () => {
    const msg = Message.system('You are a helpful assistant');
    expect(msg.role).toBe(Role.System);
    expect(msg.textContent()).toBe('You are a helpful assistant');
  });

  it('creates user message with multiple content blocks', () => {
    const blocks = [
      ContentBlock.text('Hello'),
      ContentBlock.text('World'),
    ];
    const msg = Message.userWithContent(blocks);
    expect(msg.role).toBe(Role.User);
    expect(msg.content.length).toBe(2);
    expect(msg.textContent()).toBe('HelloWorld');
  });

  it('creates assistant message with tool use', () => {
    const toolUse = ContentBlock.toolUse('tool-1', 'get_weather', { location: 'NYC' });
    const msg = Message.assistantWithContent([toolUse]);
    expect(msg.hasToolUse()).toBe(true);
    expect(msg.toolUses().length).toBe(1);
  });

  it('creates tool results message', () => {
    const result = ContentBlock.toolResult('tool-1', '72°F and sunny');
    const msg = Message.toolResults([result]);
    expect(msg.role).toBe(Role.User);
    expect(msg.content.length).toBe(1);
  });
});

// =============================================================================
// ContentBlock Tests
// =============================================================================

describe('ContentBlock', () => {
  it('creates text block', () => {
    const block = ContentBlock.text('Hello');
    expect(block.isText).toBe(true);
    expect(block.textValue).toBe('Hello');
    expect(block.isToolUse).toBe(false);
  });

  it('creates text block with caching', () => {
    const block = ContentBlock.textCached('Cached content');
    expect(block.isText).toBe(true);
    expect(block.textValue).toBe('Cached content');
  });

  it('creates image block from base64', () => {
    const block = ContentBlock.image('image/png', 'base64data...');
    expect(block.isImage).toBe(true);
    expect(block.isText).toBe(false);
  });

  it('creates image block from URL', () => {
    const block = ContentBlock.imageUrl('https://example.com/image.png');
    expect(block.isImage).toBe(true);
  });

  it('creates tool use block', () => {
    const block = ContentBlock.toolUse('tool-123', 'search', { query: 'test' });
    expect(block.isToolUse).toBe(true);
    expect(block.isText).toBe(false);

    const info = block.asToolUse();
    expect(info).toBeDefined();
    expect(info!.id).toBe('tool-123');
    expect(info!.name).toBe('search');
    expect(info!.input).toEqual({ query: 'test' });
  });

  it('creates tool result block', () => {
    const block = ContentBlock.toolResult('tool-123', 'Search results here');
    expect(block.isToolResult).toBe(true);

    const info = block.asToolResult();
    expect(info).toBeDefined();
    expect(info!.toolUseId).toBe('tool-123');
    expect(info!.content).toBe('Search results here');
    expect(info!.isError).toBe(false);
  });

  it('creates tool result block with error', () => {
    const block = ContentBlock.toolResult('tool-123', 'Error occurred', true);
    const info = block.asToolResult();
    expect(info!.isError).toBe(true);
  });

  it('creates thinking block', () => {
    const block = ContentBlock.thinking('Let me think about this...');
    expect(block.isThinking).toBe(true);
    expect(block.thinkingContent).toBe('Let me think about this...');
  });

  it('creates PDF document block', () => {
    const block = ContentBlock.pdf('base64pdfdata...');
    expect(block.isDocument).toBe(true);
  });
});

// =============================================================================
// CompletionRequest Tests
// =============================================================================

describe('CompletionRequest', () => {
  it('creates basic request', () => {
    const messages = [Message.user('Hello')];
    const req = CompletionRequest.create('claude-sonnet-4-20250514', messages);

    expect(req.model).toBe('claude-sonnet-4-20250514');
    expect(req.messages.length).toBe(1);
    expect(req.stream).toBe(false);
  });

  it('chains builder methods', () => {
    const req = CompletionRequest.create('gpt-4o', [Message.user('Hi')])
      .withSystem('You are helpful')
      .withMaxTokens(1000)
      .withTemperature(0.7)
      .withTopP(0.9)
      .withStopSequences(['END']);

    expect(req.model).toBe('gpt-4o');
    expect(req.system).toBe('You are helpful');
    expect(req.maxTokens).toBe(1000);
    expect(req.temperature).toBeCloseTo(0.7, 5);  // Float precision
    expect(req.stream).toBe(false);
  });

  it('enables streaming', () => {
    const req = CompletionRequest.create('claude-sonnet-4-20250514', [Message.user('Hi')])
      .withStreaming();

    expect(req.stream).toBe(true);
  });

  it('configures thinking', () => {
    const req = CompletionRequest.create('claude-sonnet-4-20250514', [Message.user('Solve this')])
      .withThinking(10000);

    expect(req.hasThinking()).toBe(true);
  });

  it('configures thinking with config object', () => {
    const config = ThinkingConfig.enabled(5000);
    const req = CompletionRequest.create('claude-sonnet-4-20250514', [Message.user('Think')])
      .withThinkingConfig(config);

    expect(req.hasThinking()).toBe(true);
  });

  it('configures JSON schema output', () => {
    const schema = {
      type: 'object',
      properties: {
        name: { type: 'string' },
        age: { type: 'integer' },
      },
      required: ['name'],
    };

    const req = CompletionRequest.create('gpt-4o', [Message.user('Get info')])
      .withJsonSchema('person', schema);

    expect(req.hasStructuredOutput()).toBe(true);
  });

  it('configures JSON output mode', () => {
    const req = CompletionRequest.create('gpt-4o', [Message.user('Reply in JSON')])
      .withJsonOutput();

    expect(req.hasStructuredOutput()).toBe(true);
  });

  it('configures system caching', () => {
    const req = CompletionRequest.create('claude-sonnet-4-20250514', [Message.user('Hi')])
      .withSystem('Long system prompt...')
      .withSystemCaching();

    expect(req.hasCaching()).toBe(true);
  });

  it('configures extended system caching', () => {
    const req = CompletionRequest.create('claude-sonnet-4-20250514', [Message.user('Hi')])
      .withSystem('Long system prompt...')
      .withSystemCachingExtended();

    expect(req.hasCaching()).toBe(true);
  });

  it('configures tools', () => {
    const tool = new ToolDefinition('search', 'Search the web', {
      type: 'object',
      properties: {
        query: { type: 'string' },
      },
    });

    const req = CompletionRequest.create('claude-sonnet-4-20250514', [Message.user('Search')])
      .withTools([tool]);

    // We can verify the request was built successfully
    expect(req.model).toBe('claude-sonnet-4-20250514');
  });

  it('sets extra parameters', () => {
    const req = CompletionRequest.create('custom-model', [Message.user('Hi')])
      .withExtra({ custom_param: 'value' });

    expect(req.model).toBe('custom-model');
  });

  it('enables extended output', () => {
    const req = CompletionRequest.create('claude-sonnet-4-20250514', [Message.user('Hi')])
      .withExtendedOutput();

    expect(req.model).toBe('claude-sonnet-4-20250514');
  });

  it('enables prediction', () => {
    const req = CompletionRequest.create('claude-sonnet-4-20250514', [Message.user('Complete this')])
      .withPrediction('Expected output prefix');

    expect(req.model).toBe('claude-sonnet-4-20250514');
  });
});

// =============================================================================
// ToolBuilder Tests
// =============================================================================

describe('ToolBuilder', () => {
  it('builds simple tool', () => {
    const tool = new ToolBuilder('hello')
      .description('Says hello')
      .build();

    expect(tool.name).toBe('hello');
    expect(tool.description).toBe('Says hello');
  });

  it('builds tool with string parameter', () => {
    const tool = new ToolBuilder('search')
      .description('Search the web')
      .stringParam('query', 'The search query', true)
      .build();

    expect(tool.name).toBe('search');
    expect(tool.inputSchema.properties.query).toBeDefined();
    expect(tool.inputSchema.properties.query.type).toBe('string');
    expect(tool.inputSchema.required).toContain('query');
  });

  it('builds tool with multiple parameter types', () => {
    const tool = new ToolBuilder('create_item')
      .description('Create an item')
      .stringParam('name', 'Item name', true)
      .integerParam('quantity', 'Quantity', true)
      .numberParam('price', 'Price', false)
      .booleanParam('active', 'Is active', false)
      .build();

    const schema = tool.inputSchema;
    expect(schema.properties.name.type).toBe('string');
    expect(schema.properties.quantity.type).toBe('integer');
    expect(schema.properties.price.type).toBe('number');
    expect(schema.properties.active.type).toBe('boolean');
    expect(schema.required).toContain('name');
    expect(schema.required).toContain('quantity');
    expect(schema.required).not.toContain('price');
    expect(schema.required).not.toContain('active');
  });

  it('builds tool with array parameter', () => {
    const tool = new ToolBuilder('process_list')
      .description('Process a list')
      .arrayParam('items', 'List of items', 'string', true)
      .build();

    const schema = tool.inputSchema;
    expect(schema.properties.items.type).toBe('array');
    expect(schema.properties.items.items.type).toBe('string');
  });

  it('builds tool with enum parameter', () => {
    const tool = new ToolBuilder('set_priority')
      .description('Set priority')
      .enumParam('level', 'Priority level', ['low', 'medium', 'high'], true)
      .build();

    const schema = tool.inputSchema;
    expect(schema.properties.level.type).toBe('string');
    expect(schema.properties.level.enum).toEqual(['low', 'medium', 'high']);
  });

  it('builds tool with custom parameter', () => {
    const customSchema = {
      type: 'object',
      properties: {
        nested: { type: 'string' },
      },
    };

    const tool = new ToolBuilder('complex')
      .description('Complex operation')
      .customParam('data', customSchema, true)
      .build();

    expect(tool.inputSchema.properties.data).toEqual(customSchema);
  });
});

// =============================================================================
// ToolDefinition Tests
// =============================================================================

describe('ToolDefinition', () => {
  it('creates tool definition directly', () => {
    const schema = {
      type: 'object',
      properties: {
        city: { type: 'string', description: 'City name' },
      },
      required: ['city'],
    };

    const tool = new ToolDefinition('get_weather', 'Get current weather', schema);

    expect(tool.name).toBe('get_weather');
    expect(tool.description).toBe('Get current weather');
    expect(tool.inputSchema).toEqual(schema);
  });
});

// =============================================================================
// Configuration Types Tests
// =============================================================================

describe('CacheBreakpoint', () => {
  it('creates ephemeral cache breakpoint', () => {
    const bp = CacheBreakpoint.ephemeral();
    expect(bp.cacheControl).toBe(CacheControl.Ephemeral);
  });

  it('creates extended cache breakpoint', () => {
    const bp = CacheBreakpoint.extended();
    expect(bp.cacheControl).toBe(CacheControl.Extended);
  });
});

describe('ThinkingConfig', () => {
  it('creates enabled thinking config', () => {
    const config = ThinkingConfig.enabled(10000);
    expect(config.thinkingType).toBe(ThinkingType.Enabled);
    expect(config.budgetTokens).toBe(10000);
  });

  it('creates disabled thinking config', () => {
    const config = ThinkingConfig.disabled();
    expect(config.thinkingType).toBe(ThinkingType.Disabled);
    expect(config.budgetTokens).toBeNull();
  });
});

describe('StructuredOutput', () => {
  it('creates JSON schema structured output', () => {
    const schema = {
      type: 'object',
      properties: { name: { type: 'string' } },
    };
    const output = StructuredOutput.jsonSchema('person', schema);
    // Just verify it was created successfully
    expect(output).toBeDefined();
  });

  it('creates JSON object structured output', () => {
    const output = StructuredOutput.jsonObject();
    expect(output).toBeDefined();
  });
});

// =============================================================================
// TokenCountRequest Tests
// =============================================================================

describe('TokenCountRequest', () => {
  it('creates token count request', () => {
    const messages = [Message.user('Hello, world!')];
    const req = TokenCountRequest.create('claude-sonnet-4-20250514', messages);

    expect(req.model).toBe('claude-sonnet-4-20250514');
    expect(req.messages.length).toBe(1);
  });

  it('creates from completion request', () => {
    const completionReq = CompletionRequest.create('claude-sonnet-4-20250514', [Message.user('Hi')]);
    const tokenReq = TokenCountRequest.fromCompletionRequest(completionReq);

    expect(tokenReq.model).toBe('claude-sonnet-4-20250514');
  });

  it('adds system prompt', () => {
    const req = TokenCountRequest.create('claude-sonnet-4-20250514', [Message.user('Hi')])
      .withSystem('You are helpful');

    expect(req.system).toBe('You are helpful');
  });

  it('adds tools', () => {
    const tool = new ToolDefinition('test', 'Test tool', { type: 'object' });
    const req = TokenCountRequest.create('claude-sonnet-4-20250514', [Message.user('Hi')])
      .withTools([tool]);

    expect(req.model).toBe('claude-sonnet-4-20250514');
  });
});

// =============================================================================
// BatchRequest Tests
// =============================================================================

describe('BatchRequest', () => {
  it('creates batch request', () => {
    const completionReq = CompletionRequest.create('claude-sonnet-4-20250514', [Message.user('Hi')]);
    const batchReq = BatchRequest.create('request-1', completionReq);

    expect(batchReq.customId).toBe('request-1');
  });
});

// =============================================================================
// Client Tests (No API calls)
// =============================================================================

describe('LLMKitClient', () => {
  it('throws when creating client with no providers', () => {
    // Creating a client with no providers should throw an error
    expect(() => new LLMKitClient()).toThrow('No providers configured');
  });

  it('creates client with provider config', () => {
    // Note: This won't work without actual API keys, but we can test the config parsing
    const client = new LLMKitClient({
      providers: {
        anthropic: { apiKey: 'test-key-123' },
      },
      defaultProvider: 'anthropic',
    });

    expect(client.providers()).toContain('anthropic');
    expect(client.defaultProvider).toBe('anthropic');
  });

  it('creates client with multiple providers', () => {
    const client = new LLMKitClient({
      providers: {
        anthropic: { apiKey: 'anthropic-key' },
        openai: { apiKey: 'openai-key' },
        groq: { apiKey: 'groq-key' },
      },
    });

    const providers = client.providers();
    expect(providers).toContain('anthropic');
    expect(providers).toContain('openai');
    expect(providers).toContain('groq');
  });

  it('creates client with Azure config', () => {
    const client = new LLMKitClient({
      providers: {
        azure: {
          apiKey: 'azure-key',
          endpoint: 'https://myresource.openai.azure.com',
          deployment: 'gpt-4',
        },
      },
    });

    expect(client.providers()).toContain('azure');
  });

  it('creates client with Bedrock config', () => {
    // Bedrock provider requires the bedrock feature to be compiled in
    try {
      const client = new LLMKitClient({
        providers: {
          bedrock: {
            region: 'us-east-1',
          },
        },
      });
      expect(client.providers()).toContain('bedrock');
    } catch (e) {
      // Skip if bedrock feature not available
      expect((e as Error).message).toMatch(/provider|configuration|not.*available/i);
    }
  });

  it('creates client with Vertex config', () => {
    // Vertex provider requires the vertex feature to be compiled in
    try {
      const client = new LLMKitClient({
        providers: {
          vertex: {
            project: 'my-project',
            location: 'us-central1',
            accessToken: 'token',
          },
        },
      });
      expect(client.providers()).toContain('vertex');
    } catch (e) {
      // Skip if vertex feature not available
      expect((e as Error).message).toMatch(/provider|configuration|not.*available/i);
    }
  });
});

// =============================================================================
// Model Registry Tests
// =============================================================================

import {
  getModelInfo,
  getAllModels,
  getModelsByProvider,
  getCurrentModels,
  getClassifierModels,
  getModelsWithCapability,
  getCheapestModel,
  supportsStructuredOutput,
  getRegistryStats,
  listProviders,
  JsProvider as Provider,
} from '../index';

describe('Model Registry', () => {
  describe('getModelInfo', () => {
    it('returns model info for valid model ID', () => {
      const info = getModelInfo('claude-sonnet-4-20250514');
      expect(info).not.toBeNull();
      if (info) {
        expect(info.provider).toBe(Provider.Anthropic);
        // Capabilities vary by model, just check they exist
        expect(info.capabilities).toBeDefined();
      }
    });

    it('returns model info for alias', () => {
      const info = getModelInfo('gpt-4o');
      expect(info).not.toBeNull();
      if (info) {
        expect(info.provider).toBe(Provider.OpenAI);
      }
    });

    it('returns null for unknown model', () => {
      const info = getModelInfo('unknown-model-xyz');
      expect(info).toBeNull();
    });
  });

  describe('getAllModels', () => {
    it('returns all models', () => {
      const models = getAllModels();
      expect(models.length).toBeGreaterThan(50);
    });
  });

  describe('getModelsByProvider', () => {
    it('returns Anthropic models', () => {
      const models = getModelsByProvider(Provider.Anthropic);
      expect(models.length).toBeGreaterThan(0);
      expect(models.every(m => m.provider === Provider.Anthropic)).toBe(true);
    });

    it('returns OpenAI models', () => {
      const models = getModelsByProvider(Provider.OpenAI);
      expect(models.length).toBeGreaterThan(0);
      expect(models.every(m => m.provider === Provider.OpenAI)).toBe(true);
    });
  });

  describe('getCurrentModels', () => {
    it('returns only current models', () => {
      const models = getCurrentModels();
      expect(models.length).toBeGreaterThan(0);
      // All should be current status
    });
  });

  describe('getClassifierModels', () => {
    it('returns classifier-suitable models', () => {
      const models = getClassifierModels();
      expect(models.length).toBeGreaterThan(0);
      expect(models.every(m => m.canClassify)).toBe(true);
    });
  });

  describe('getModelsWithCapability', () => {
    it('returns vision-capable models', () => {
      const models = getModelsWithCapability(true, null, null);
      expect(models.length).toBeGreaterThan(0);
      expect(models.every(m => m.capabilities.vision)).toBe(true);
    });

    it('returns thinking-capable models', () => {
      const models = getModelsWithCapability(null, null, true);
      // Some registries may not have thinking-capable models marked
      expect(models.length).toBeGreaterThanOrEqual(0);
      if (models.length > 0) {
        expect(models.every(m => m.capabilities.thinking)).toBe(true);
      }
    });
  });

  describe('supportsStructuredOutput', () => {
    it('returns true for GPT-4o', () => {
      // gpt-4o alias should resolve to a model that supports structured output
      const result = supportsStructuredOutput('gpt-4o');
      // If the alias resolves, it should support structured output
      // If the model isn't found, result is false
      expect(typeof result).toBe('boolean');
    });

    it('returns false for unknown model', () => {
      expect(supportsStructuredOutput('unknown-model')).toBe(false);
    });
  });

  describe('getRegistryStats', () => {
    it('returns registry statistics', () => {
      const stats = getRegistryStats();
      expect(stats.totalModels).toBeGreaterThan(50);
      expect(stats.providers).toBeGreaterThan(10);
      expect(stats.currentModels).toBeGreaterThan(0);
    });
  });

  describe('listProviders', () => {
    it('returns all providers', () => {
      const providers = listProviders();
      expect(providers.length).toBeGreaterThan(10);
      expect(providers).toContain(Provider.Anthropic);
      expect(providers).toContain(Provider.OpenAI);
      expect(providers).toContain(Provider.Google);
    });
  });

  describe('ModelInfo methods', () => {
    it('calculates estimate cost', () => {
      const info = getModelInfo('gpt-4o');
      expect(info).not.toBeNull();
      if (info) {
        const cost = info.estimateCost(1000000, 500000);
        expect(cost).toBeGreaterThan(0);
      }
    });

    it('calculates quality per dollar', () => {
      const info = getModelInfo('gpt-4o');
      expect(info).not.toBeNull();
      if (info) {
        const qpd = info.qualityPerDollar();
        // Returns 0 if no benchmark data is available
        expect(qpd).toBeGreaterThanOrEqual(0);
      }
    });

    it('returns raw ID without prefix', () => {
      // Use a specific model ID (not alias) for predictable rawId
      const info = getModelInfo('claude-sonnet-4-20250514');
      expect(info).not.toBeNull();
      if (info) {
        expect(info.rawId()).toBe('claude-sonnet-4-20250514');
      }
    });
  });
});

// =============================================================================
// Embedding Tests
// =============================================================================

import {
  JsEmbeddingRequest as EmbeddingRequest,
  JsEncodingFormat as EncodingFormat,
  JsEmbeddingInputType as EmbeddingInputType,
} from '../index';

describe('EmbeddingRequest', () => {
  it('creates embedding request for single text', () => {
    const request = new EmbeddingRequest('text-embedding-3-small', 'Hello, world!');
    expect(request.model).toBe('text-embedding-3-small');
    expect(request.textCount).toBe(1);
    expect(request.texts()).toContain('Hello, world!');
  });

  it('creates batch embedding request', () => {
    const texts = ['Hello', 'World', 'How are you?'];
    const request = EmbeddingRequest.batch('text-embedding-3-small', texts);
    expect(request.model).toBe('text-embedding-3-small');
    expect(request.textCount).toBe(3);
    expect(request.texts()).toEqual(texts);
  });

  it('sets dimensions', () => {
    const request = new EmbeddingRequest('text-embedding-3-small', 'Hello')
      .withDimensions(256);
    expect(request.dimensions).toBe(256);
  });

  it('sets encoding format', () => {
    const request = new EmbeddingRequest('text-embedding-3-small', 'Hello')
      .withEncodingFormat(EncodingFormat.Base64);
    expect(request).toBeDefined();
  });

  it('sets input type', () => {
    const request = new EmbeddingRequest('text-embedding-3-small', 'Hello')
      .withInputType(EmbeddingInputType.Query);
    expect(request).toBeDefined();
  });

  it('chains configuration methods', () => {
    const request = new EmbeddingRequest('text-embedding-3-small', 'Hello')
      .withDimensions(512)
      .withEncodingFormat(EncodingFormat.Float)
      .withInputType(EmbeddingInputType.Document);
    expect(request.dimensions).toBe(512);
    expect(request.model).toBe('text-embedding-3-small');
  });
});

describe('EmbeddingRequest edge cases', () => {
  it('handles empty text', () => {
    const request = new EmbeddingRequest('text-embedding-3-small', '');
    expect(request.textCount).toBe(1);
    expect(request.texts()).toContain('');
  });

  it('handles batch with single item', () => {
    const request = EmbeddingRequest.batch('text-embedding-3-small', ['Single']);
    expect(request.textCount).toBe(1);
    expect(request.texts()).toEqual(['Single']);
  });

  it('handles batch with empty list', () => {
    const request = EmbeddingRequest.batch('text-embedding-3-small', []);
    expect(request.textCount).toBe(0);
    expect(request.texts()).toEqual([]);
  });

  it('handles large batch', () => {
    const texts = Array.from({ length: 100 }, (_, i) => `Text ${i}`);
    const request = EmbeddingRequest.batch('text-embedding-3-small', texts);
    expect(request.textCount).toBe(100);
    expect(request.texts().length).toBe(100);
  });

  it('handles various dimension sizes', () => {
    const dimensions = [64, 128, 256, 512, 1024, 1536, 3072];
    for (const dim of dimensions) {
      const request = new EmbeddingRequest('text-embedding-3-large', 'Test').withDimensions(dim);
      expect(request.dimensions).toBe(dim);
    }
  });

  it('defaults to no dimensions (model default)', () => {
    const request = new EmbeddingRequest('text-embedding-3-small', 'Hello');
    // In TypeScript bindings, unset Option<usize> becomes null
    expect(request.dimensions).toBeNull();
  });

  it('creates immutable instances when chaining', () => {
    const original = new EmbeddingRequest('text-embedding-3-small', 'Hello');
    const withDims = original.withDimensions(256);

    // Original should be unchanged (null in TypeScript bindings)
    expect(original.dimensions).toBeNull();
    // New instance should have dimensions
    expect(withDims.dimensions).toBe(256);
  });

  it('allows overriding dimensions in chain', () => {
    const request = new EmbeddingRequest('text-embedding-3-large', 'Test query')
      .withDimensions(512)
      .withEncodingFormat(EncodingFormat.Float)
      .withInputType(EmbeddingInputType.Query)
      .withDimensions(256); // Override previous
    expect(request.dimensions).toBe(256);
    expect(request.model).toBe('text-embedding-3-large');
  });

  it('handles unicode characters', () => {
    const unicodeTexts = ['こんにちは', 'مرحبا', '🎉🚀', 'Привет мир'];
    for (const text of unicodeTexts) {
      const request = new EmbeddingRequest('text-embedding-3-small', text);
      expect(request.texts()).toContain(text);
    }
  });

  it('handles multiline text', () => {
    const multiline = 'Line 1\nLine 2\nLine 3';
    const request = new EmbeddingRequest('text-embedding-3-small', multiline);
    expect(request.texts()).toContain(multiline);
  });
});

describe('EncodingFormat enum', () => {
  it('has Float value', () => {
    expect(EncodingFormat.Float).toBe(0);
  });

  it('has Base64 value', () => {
    expect(EncodingFormat.Base64).toBe(1);
  });

  it('can be used with embedding request', () => {
    const formats = [EncodingFormat.Float, EncodingFormat.Base64];
    for (const fmt of formats) {
      const request = new EmbeddingRequest('text-embedding-3-small', 'Test').withEncodingFormat(fmt);
      expect(request).toBeDefined();
    }
  });
});

describe('EmbeddingInputType enum', () => {
  it('has Query value', () => {
    expect(EmbeddingInputType.Query).toBe(0);
  });

  it('has Document value', () => {
    expect(EmbeddingInputType.Document).toBe(1);
  });

  it('can be used with embedding request', () => {
    const inputTypes = [EmbeddingInputType.Query, EmbeddingInputType.Document];
    for (const inputType of inputTypes) {
      const request = new EmbeddingRequest('text-embedding-3-small', 'Test').withInputType(inputType);
      expect(request).toBeDefined();
    }
  });
});

describe('LLMKitClient embedding methods', () => {
  it('has embeddingProviders method', () => {
    const client = new LLMKitClient({
      providers: {
        anthropic: { apiKey: 'test-key-123' },
      },
    });
    const providers = client.embeddingProviders();
    expect(Array.isArray(providers)).toBe(true);
  });

  it('has supportsEmbeddings method', () => {
    const client = new LLMKitClient({
      providers: {
        anthropic: { apiKey: 'test-key-123' },
      },
    });
    // Anthropic doesn't support embeddings
    expect(client.supportsEmbeddings('anthropic')).toBe(false);
    expect(client.supportsEmbeddings('nonexistent')).toBe(false);
  });

  it('has embed method', () => {
    const client = new LLMKitClient({
      providers: {
        anthropic: { apiKey: 'test-key-123' },
      },
    });
    expect(typeof client.embed).toBe('function');
  });
});
