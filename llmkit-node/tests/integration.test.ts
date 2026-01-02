/**
 * Integration tests for LLMKit Node.js bindings
 *
 * These tests make actual API calls and require valid API keys.
 * Tests are automatically skipped if the required API key is not set.
 *
 * Set the following environment variables to enable tests:
 * - ANTHROPIC_API_KEY: Anthropic tests
 * - OPENAI_API_KEY: OpenAI tests
 * - GROQ_API_KEY: Groq tests
 * - MISTRAL_API_KEY: Mistral tests
 */
import { describe, it, expect, beforeAll } from 'vitest';
import {
  JsMessage as Message,
  JsCompletionRequest as CompletionRequest,
  JsToolBuilder as ToolBuilder,
  JsLlmKitClient as LLMKitClient,
  JsTokenCountRequest as TokenCountRequest,
  JsStopReason as StopReason,
} from '../index';

// =============================================================================
// Helper Functions
// =============================================================================

function hasEnv(key: string): boolean {
  return !!process.env[key];
}

// =============================================================================
// Anthropic Tests
// =============================================================================

describe.skipIf(!hasEnv('ANTHROPIC_API_KEY'))('Anthropic Integration', () => {
  let client: LLMKitClient;

  beforeAll(() => {
    client = LLMKitClient.fromEnv();
  });

  it('completes a simple request', async () => {
    const request = CompletionRequest.create('claude-sonnet-4-20250514', [
      Message.user('What is 2+2? Reply with just the number.'),
    ]).withMaxTokens(50);

    const response = await client.complete(request);

    expect(response.id).toBeDefined();
    expect(response.model).toContain('claude');
    expect(response.textContent()).toContain('4');
    expect(response.stopReason).toBe(StopReason.EndTurn);
    expect(response.usage).toBeDefined();
    expect(response.usage!.inputTokens).toBeGreaterThan(0);
    expect(response.usage!.outputTokens).toBeGreaterThan(0);
  });

  it('handles system prompt', async () => {
    const request = CompletionRequest.create('claude-sonnet-4-20250514', [
      Message.user('What are you?'),
    ])
      .withSystem('You are a friendly robot named R2D2. Always introduce yourself.')
      .withMaxTokens(100);

    const response = await client.complete(request);
    expect(response.textContent().toLowerCase()).toMatch(/r2d2|robot/);
  });

  it('streams response', async () => {
    const request = CompletionRequest.create('claude-sonnet-4-20250514', [
      Message.user('Count from 1 to 5, one number per line.'),
    ])
      .withMaxTokens(100)
      .withStreaming();

    const chunks: string[] = [];
    let isDone = false;

    await new Promise<void>((resolve, reject) => {
      client.completeStream(request, (chunk, error) => {
        if (error) {
          reject(new Error(error));
          return;
        }
        if (!chunk || chunk.isDone) {
          isDone = true;
          resolve();
          return;
        }
        if (chunk.text) {
          chunks.push(chunk.text);
        }
      });
    });

    expect(isDone).toBe(true);
    expect(chunks.length).toBeGreaterThan(0);
    const fullText = chunks.join('');
    expect(fullText).toContain('1');
    expect(fullText).toContain('5');
  });

  it('handles tool use', async () => {
    const tool = new ToolBuilder('get_weather')
      .description('Get the current weather in a city')
      .stringParam('city', 'The city name', true)
      .build();

    const request = CompletionRequest.create('claude-sonnet-4-20250514', [
      Message.user('What is the weather in Paris?'),
    ])
      .withTools([tool])
      .withMaxTokens(200);

    const response = await client.complete(request);

    expect(response.hasToolUse()).toBe(true);
    expect(response.stopReason).toBe(StopReason.ToolUse);

    const toolUses = response.toolUses();
    expect(toolUses.length).toBeGreaterThan(0);

    const toolUse = toolUses[0].asToolUse();
    expect(toolUse).toBeDefined();
    expect(toolUse!.name).toBe('get_weather');
    expect(toolUse!.input.city).toBeDefined();
  });

  it('counts tokens', async () => {
    const request = TokenCountRequest.create('claude-sonnet-4-20250514', [
      Message.user('Hello, how are you?'),
    ]).withSystem('You are a helpful assistant.');

    const result = await client.countTokens(request);
    expect(result.inputTokens).toBeGreaterThan(0);
    // Should be approximately 15-25 tokens
    expect(result.inputTokens).toBeLessThan(100);
  });

  it('handles multi-turn conversation', async () => {
    const request = CompletionRequest.create('claude-sonnet-4-20250514', [
      Message.user('My name is Alice.'),
      Message.assistant('Hello Alice! Nice to meet you.'),
      Message.user('What is my name?'),
    ]).withMaxTokens(50);

    const response = await client.complete(request);
    expect(response.textContent().toLowerCase()).toContain('alice');
  });
});

// =============================================================================
// OpenAI Tests
// =============================================================================

describe.skipIf(!hasEnv('OPENAI_API_KEY'))('OpenAI Integration', () => {
  let client: LLMKitClient;

  beforeAll(() => {
    client = LLMKitClient.fromEnv();
  });

  it('completes a simple request', async () => {
    const request = CompletionRequest.create('gpt-4o-mini', [
      Message.user('What is 3+3? Reply with just the number.'),
    ]).withMaxTokens(50);

    const response = await client.complete(request);

    expect(response.id).toBeDefined();
    expect(response.model).toContain('gpt-4o-mini');
    expect(response.textContent()).toContain('6');
    expect(response.usage).toBeDefined();
  });

  it('uses JSON output mode', async () => {
    const request = CompletionRequest.create('gpt-4o-mini', [
      Message.user('Return a JSON object with a "greeting" field saying "hello"'),
    ])
      .withJsonOutput()
      .withMaxTokens(100);

    const response = await client.complete(request);
    const text = response.textContent();

    // Should be valid JSON
    const parsed = JSON.parse(text);
    expect(parsed.greeting).toBeDefined();
  });

  it('uses structured output with JSON schema', async () => {
    const schema = {
      type: 'object',
      properties: {
        name: { type: 'string' },
        age: { type: 'integer' },
      },
      required: ['name', 'age'],
      additionalProperties: false,
    };

    const request = CompletionRequest.create('gpt-4o-mini', [
      Message.user('Generate info for a person named Bob who is 25 years old'),
    ])
      .withJsonSchema('person', schema)
      .withMaxTokens(100);

    const response = await client.complete(request);
    const parsed = JSON.parse(response.textContent());

    expect(parsed.name).toBe('Bob');
    expect(parsed.age).toBe(25);
  });

  it('streams response', async () => {
    const request = CompletionRequest.create('gpt-4o-mini', [
      Message.user('Say "hello world"'),
    ])
      .withMaxTokens(50)
      .withStreaming();

    const chunks: string[] = [];

    await new Promise<void>((resolve, reject) => {
      client.completeStream(request, (chunk, error) => {
        if (error) {
          reject(new Error(error));
          return;
        }
        if (!chunk || chunk.isDone) {
          resolve();
          return;
        }
        if (chunk.text) {
          chunks.push(chunk.text);
        }
      });
    });

    const fullText = chunks.join('').toLowerCase();
    expect(fullText).toContain('hello');
  });
});

// =============================================================================
// Groq Tests
// =============================================================================

describe.skipIf(!hasEnv('GROQ_API_KEY'))('Groq Integration', () => {
  let client: LLMKitClient;

  beforeAll(() => {
    client = LLMKitClient.fromEnv();
  });

  it('completes a simple request', async () => {
    const request = CompletionRequest.create('llama-3.3-70b-versatile', [
      Message.user('What is the capital of France? Reply with just the city name.'),
    ]).withMaxTokens(50);

    const response = await client.complete(request);

    expect(response.id).toBeDefined();
    expect(response.textContent().toLowerCase()).toContain('paris');
  });

  it('streams response', async () => {
    const request = CompletionRequest.create('llama-3.3-70b-versatile', [
      Message.user('Count from 1 to 3.'),
    ])
      .withMaxTokens(50)
      .withStreaming();

    const chunks: string[] = [];

    await new Promise<void>((resolve, reject) => {
      client.completeStream(request, (chunk, error) => {
        if (error) {
          reject(new Error(error));
          return;
        }
        if (!chunk || chunk.isDone) {
          resolve();
          return;
        }
        if (chunk.text) {
          chunks.push(chunk.text);
        }
      });
    });

    expect(chunks.length).toBeGreaterThan(0);
  });
});

// =============================================================================
// Mistral Tests
// =============================================================================

describe.skipIf(!hasEnv('MISTRAL_API_KEY'))('Mistral Integration', () => {
  let client: LLMKitClient;

  beforeAll(() => {
    client = LLMKitClient.fromEnv();
  });

  it('completes a simple request', async () => {
    const request = CompletionRequest.create('mistral-small-latest', [
      Message.user('What is 5+5? Reply with just the number.'),
    ]).withMaxTokens(50);

    const response = await client.complete(request);

    expect(response.id).toBeDefined();
    expect(response.textContent()).toContain('10');
  });
});

// =============================================================================
// Multi-Provider Tests
// =============================================================================

describe.skipIf(!hasEnv('ANTHROPIC_API_KEY') || !hasEnv('OPENAI_API_KEY'))(
  'Multi-Provider Integration',
  () => {
    let client: LLMKitClient;

    beforeAll(() => {
      client = LLMKitClient.fromEnv();
    });

    it('uses different providers in same client', async () => {
      // Test with Anthropic
      const anthropicRequest = CompletionRequest.create('claude-sonnet-4-20250514', [
        Message.user('Say "Anthropic"'),
      ]).withMaxTokens(20);

      const anthropicResponse = await client.complete(anthropicRequest);
      expect(anthropicResponse.model).toContain('claude');

      // Test with OpenAI
      const openaiRequest = CompletionRequest.create('gpt-4o-mini', [
        Message.user('Say "OpenAI"'),
      ]).withMaxTokens(20);

      const openaiResponse = await client.complete(openaiRequest);
      expect(openaiResponse.model).toContain('gpt');
    });

    it('uses completeWithProvider', async () => {
      const request = CompletionRequest.create('claude-sonnet-4-20250514', [
        Message.user('Say "test"'),
      ]).withMaxTokens(20);

      const response = await client.completeWithProvider('anthropic', request);
      expect(response.model).toContain('claude');
    });

    it('lists available providers', () => {
      const providers = client.providers();
      expect(providers).toContain('anthropic');
      expect(providers).toContain('openai');
    });
  }
);

// =============================================================================
// Error Handling Tests
// =============================================================================

describe.skipIf(!hasEnv('ANTHROPIC_API_KEY'))('Error Handling', () => {
  let client: LLMKitClient;

  beforeAll(() => {
    client = LLMKitClient.fromEnv();
  });

  it('handles invalid model gracefully', async () => {
    const request = CompletionRequest.create('non-existent-model-12345', [
      Message.user('Hello'),
    ]).withMaxTokens(50);

    await expect(client.complete(request)).rejects.toThrow();
  });

  it('handles empty messages gracefully', async () => {
    const request = CompletionRequest.create('claude-sonnet-4-20250514', [])
      .withMaxTokens(50);

    await expect(client.complete(request)).rejects.toThrow();
  });
});
