# Model Registry

LLMKit includes a built-in registry of **11,000+ models** with pricing, capabilities, and specifications. No API calls needed — all data is compiled into the binary for instant lookups.

## Quick Start

### Rust

```rust
use llmkit::{get_model_info, get_models_by_provider, get_models_with_capability};

// Get model details
let info = get_model_info("anthropic/claude-sonnet-4-20250514")?;
println!("Context window: {} tokens", info.context_window);
println!("Input price: ${}/1M tokens", info.input_price);
println!("Output price: ${}/1M tokens", info.output_price);

// Find models by provider
let anthropic_models = get_models_by_provider("anthropic");

// Find models with specific capabilities
let vision_models = get_models_with_capability(ModelCapability::Vision);
let thinking_models = get_models_with_capability(ModelCapability::Thinking);
```

### Python

```python
from llmkit import get_model_info, get_models_by_provider, get_models_with_capability

# Get model details
info = get_model_info("anthropic/claude-sonnet-4-20250514")
print(f"Context window: {info.context_window:,} tokens")
print(f"Input price: ${info.input_price}/1M tokens")
print(f"Output price: ${info.output_price}/1M tokens")

# Find models by provider
anthropic_models = get_models_by_provider("anthropic")

# Find models with specific capabilities
vision_models = get_models_with_capability(vision=True)
thinking_models = get_models_with_capability(thinking=True)
```

### Node.js

```typescript
import { getModelInfo, getModelsByProvider, getModelsWithCapability } from 'llmkit-node'

// Get model details
const info = getModelInfo('anthropic/claude-sonnet-4-20250514')
console.log(`Context window: ${info.contextWindow.toLocaleString()} tokens`)
console.log(`Input price: $${info.inputPrice}/1M tokens`)
console.log(`Output price: $${info.outputPrice}/1M tokens`)

// Find models by provider
const anthropicModels = getModelsByProvider('anthropic')

// Find models with specific capabilities
const visionModels = getModelsWithCapability({ vision: true })
const thinkingModels = getModelsWithCapability({ thinking: true })
```

## Model Information

Each model entry includes:

| Field | Description |
|-------|-------------|
| `id` | Unique identifier (`provider/model-name`) |
| `name` | Human-readable name |
| `provider` | Provider identifier |
| `context_window` | Maximum context length in tokens |
| `max_output_tokens` | Maximum output length |
| `input_price` | Price per 1M input tokens (USD) |
| `output_price` | Price per 1M output tokens (USD) |
| `supports_vision` | Image/vision input support |
| `supports_tools` | Function/tool calling support |
| `supports_streaming` | Streaming response support |
| `supports_json_mode` | JSON output mode |
| `supports_thinking` | Extended thinking/reasoning |
| `supports_caching` | Prompt caching support |

## Popular Models

### Anthropic Claude

| Model | Context | Input Price | Output Price | Features |
|-------|---------|-------------|--------------|----------|
| `anthropic/claude-sonnet-4-20250514` | 200K | $3.00 | $15.00 | Vision, Tools, Caching, Thinking |
| `anthropic/claude-opus-4-20250514` | 200K | $15.00 | $75.00 | Vision, Tools, Caching, Thinking |
| `anthropic/claude-3-5-haiku-20241022` | 200K | $0.80 | $4.00 | Vision, Tools, Caching |

### OpenAI GPT

| Model | Context | Input Price | Output Price | Features |
|-------|---------|-------------|--------------|----------|
| `openai/gpt-4o` | 128K | $2.50 | $10.00 | Vision, Tools, JSON |
| `openai/gpt-4o-mini` | 128K | $0.15 | $0.60 | Vision, Tools, JSON |
| `openai/o1` | 200K | $15.00 | $60.00 | Thinking |
| `openai/o1-mini` | 128K | $3.00 | $12.00 | Thinking |

### Google Gemini

| Model | Context | Input Price | Output Price | Features |
|-------|---------|-------------|--------------|----------|
| `google/gemini-2.0-flash` | 1M | $0.075 | $0.30 | Vision, Tools, Caching |
| `google/gemini-1.5-pro` | 2M | $1.25 | $5.00 | Vision, Tools, Caching |
| `google/gemini-1.5-flash` | 1M | $0.075 | $0.30 | Vision, Tools, Caching |

### DeepSeek

| Model | Context | Input Price | Output Price | Features |
|-------|---------|-------------|--------------|----------|
| `deepseek/deepseek-chat` | 64K | $0.14 | $0.28 | Tools, Caching |
| `deepseek/deepseek-reasoner` | 64K | $0.55 | $2.19 | Thinking, Caching |

### Mistral

| Model | Context | Input Price | Output Price | Features |
|-------|---------|-------------|--------------|----------|
| `mistral/mistral-large` | 128K | $2.00 | $6.00 | Vision, Tools |
| `mistral/mistral-small` | 32K | $0.20 | $0.60 | Tools |
| `mistral/codestral` | 32K | $0.20 | $0.60 | Code |

### Open Source (via Groq, Together, etc.)

| Model | Context | Provider | Features |
|-------|---------|----------|----------|
| `groq/llama-3.3-70b-versatile` | 128K | Groq | Ultra-fast, Tools |
| `together/meta-llama/Meta-Llama-3.1-405B` | 128K | Together | Tools, Vision |
| `fireworks/llama-v3p1-70b-instruct` | 128K | Fireworks | Tools |

## Capability Queries

### Find Vision Models

```python
vision_models = get_models_with_capability(vision=True)
for model in vision_models[:10]:
    print(f"{model.id}: {model.context_window:,} tokens")
```

### Find Thinking/Reasoning Models

```python
thinking_models = get_models_with_capability(thinking=True)
# Returns: claude-sonnet-4, claude-opus-4, o1, o1-mini, deepseek-reasoner, gemini-2.0-flash-thinking
```

### Find Models with Caching

```python
cache_models = get_models_with_capability(caching=True)
# Returns models supporting prompt caching for cost savings
```

### Find Budget-Friendly Models

```python
all_models = get_all_models()
budget_models = [m for m in all_models if m.input_price < 1.0]
budget_models.sort(key=lambda m: m.input_price)
```

## Cost Estimation

```python
from llmkit import get_model_info

def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    info = get_model_info(model_id)
    input_cost = (input_tokens / 1_000_000) * info.input_price
    output_cost = (output_tokens / 1_000_000) * info.output_price
    return input_cost + output_cost

# Example: 10K input, 2K output with Claude Sonnet
cost = estimate_cost("anthropic/claude-sonnet-4-20250514", 10000, 2000)
print(f"Estimated cost: ${cost:.4f}")  # $0.06
```

## Model ID Format

All models use the `provider/model-name` format:

```
anthropic/claude-sonnet-4-20250514
openai/gpt-4o
google/gemini-2.0-flash
groq/llama-3.3-70b-versatile
deepseek/deepseek-chat
mistral/mistral-large
bedrock/anthropic.claude-3-sonnet
azure/gpt-4o
vertex/gemini-pro
```

## Updating the Registry

The model registry is updated with each LLMKit release. To get the latest models and pricing, update to the newest version:

```bash
# Rust
cargo update llmkit

# Python
pip install --upgrade llmkit-python

# Node.js
npm update llmkit-node
```
