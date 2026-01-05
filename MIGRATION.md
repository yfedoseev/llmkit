# Migration Guide: v0.0.x → v0.1.0

**TL;DR:** No breaking changes. Your code works as-is. New features are opt-in.

---

## Quick Start

### If You're Already Using ModelSuite

✅ **Your code continues to work unchanged.**

```rust
// This still works exactly the same
let client = ModelSuiteClient::new("openai", "gpt-4", "api-key")?;
let response = client.complete(request).await?;
```

---

## What's New (Opt-In Features)

### 1. Extended Thinking (Available on 4 providers)

**Before (v0.0.x):**
```rust
// OpenAI-specific
let request = CompletionRequest::new("o1", messages)
    .with_reasoning_effort(ReasoningEffort::Medium);
```

**After (v0.1.0) - Unified across providers:**
```rust
use modelsuite::types::ThinkingConfig;

// Works on OpenAI, Anthropic, Google Vertex, DeepSeek
let request = CompletionRequest::new("gemini-2.0-flash", messages)
    .with_thinking(ThinkingConfig::enabled(5000));

let request = CompletionRequest::new("deepseek-reasoner", messages)
    .with_thinking(ThinkingConfig::enabled(5000));
```

**Providers supporting ThinkingConfig:**
- OpenAI: o3, o1-pro, o1-full
- Anthropic: claude-opus-4.1
- Google Vertex: gemini-2.0-flash-exp
- DeepSeek: deepseek-reasoner

---

### 2. Regional Provider Support (New)

**New in v0.1.0:**

```rust
use modelsuite::providers::chat::mistral::MistralRegion;

// EU-hosted endpoint (GDPR-compliant)
let provider = MistralProvider::new(
    MistralConfig::new("api-key")
        .with_region(MistralRegion::EU)
)?;

// Or via environment
std::env::set_var("MISTRAL_REGION", "eu");
let provider = MistralProvider::from_env()?;
```

**Supported regional providers:**
- **Mistral**: Global (api.mistral.ai) or EU (api.eu.mistral.ai)
- **Maritaca** (Brazil): Model discovery with `supported_models()`

---

### 3. Real-Time Voice Enhancements (New)

**Deepgram v3 Upgrade:**
```rust
use modelsuite::providers::audio::deepgram::DeepgramVersion;

// Use v3 features (nova-3 models)
let provider = DeepgramProvider::new(
    DeepgramConfig::new("api-key")
        .with_version(DeepgramVersion::V3)
)?;
```

**ElevenLabs Streaming Control:**
```rust
use modelsuite::providers::audio::elevenlabs::LatencyMode;

let config = ElevenLabsConfig::new("api-key")
    .with_streaming_latency(LatencyMode::Balanced);

// Options: LowestLatency, LowLatency, Balanced, HighQuality, HighestQuality
```

---

### 4. Video Generation (New Modality)

**New in v0.1.0:**

```rust
// Runware video generation supporting multiple models
let request = CompletionRequest::new("runware-video", vec![
    Message::user("Generate a 5-second video of a sunset")
]);

let response = client.complete(request).await?;
```

**Supported video models:**
- runway-gen-4.5
- kling-2.0
- pika-1.0
- hailuo-mini
- leonardo-ultra

---

### 5. Domain-Specific Models (New)

**Medical AI (Vertex):**
```rust
// Easier medical domain setup
let provider = VertexProvider::for_medical_domain(
    "my-project",
    "us-central1",
    "access-token"
)?;
```

**Scientific Reasoning:**
```rust
// For math, physics, chemistry, CS problems
let request = CompletionRequest::new("deepseek-reasoner", messages)
    .with_thinking(ThinkingConfig::enabled(10000));  // Higher budget for complex problems
```

See `docs/scientific_benchmarks.md` for performance data.

---

## Type Changes (Non-Breaking)

### New Enums (Import as Needed)

```rust
// Regional support
use modelsuite::providers::chat::mistral::MistralRegion;

// Voice versioning
use modelsuite::providers::audio::deepgram::DeepgramVersion;

// Audio latency control
use modelsuite::providers::audio::elevenlabs::LatencyMode;

// Video model selection
use modelsuite::providers::video::runware::VideoModel;

// Thinking configuration
use modelsuite::types::ThinkingConfig;
```

### Config Field Additions

**VertexConfig:**
```rust
// New optional field (defaults to None)
pub default_model: Option<String>

// Usage
let config = VertexConfig::new(...);
config.default_model = Some("medpalm-2".to_string());
```

**MistralConfig:**
```rust
// New optional field (defaults to Global)
pub region: Option<MistralRegion>
```

**DeepgramConfig:**
```rust
// New optional field (defaults to V1)
pub version: Option<DeepgramVersion>
```

---

## New Provider Skeletons

The following provider skeletons are ready for API integration:

- `LightOnProvider` (France) - awaiting partnership
- `LatamGPTProvider` (Latin America) - awaiting API launch
- `GrokRealtimeProvider` (xAI) - awaiting API access
- `ChatLawProvider` (Legal) - awaiting API approval

**Current status:** All return "API access pending" errors.

When APIs become available:
```rust
// These will be fully functional
let provider = LightOnProvider::from_env()?;
let provider = LatamGPTProvider::from_env()?;
let provider = GrokRealtimeProvider::from_env()?;
let provider = ChatLawProvider::from_env()?;
```

---

## Feature Flags (Same as Before)

Feature flags work the same way. No changes needed:

```toml
[dependencies]
modelsuite = { version = "0.1.0", features = [
    "openai",
    "anthropic",
    "google",
    "deepseek",
    "deepgram",
    "elevenlabs",
    "runware",  # NEW in v0.1.0
] }
```

---

## Python Binding Changes

### Same API Surface
```python
from modelsuite import ModelSuiteClient, ThinkingConfig

# New: Extended thinking
config = ThinkingConfig.enabled(budget_tokens=5000)
response = client.complete(
    model="gemini-2.0-flash",
    messages=[...],
    thinking=config
)

# New: Regional providers
from modelsuite.providers import MistralRegion
# Configuration automatically loads from MISTRAL_REGION env var

# New: Video generation
response = client.complete(
    model="runware-video",
    prompt="Generate a sunset video"
)
```

### No Breaking Changes
- All v0.0.x code still works
- New types are optional
- Existing functions unchanged

---

## TypeScript/Node.js Binding Changes

### Same API Surface
```typescript
import { ModelSuiteClient, ThinkingConfig } from 'modelsuite';

// New: Extended thinking
const config = ThinkingConfig.enabled({ budgetTokens: 5000 });
const response = await client.complete({
    model: "gemini-2.0-flash",
    messages: [...],
    thinking: config
});

// New: Regional providers
import { MistralRegion } from 'modelsuite/providers';
// Configuration from MISTRAL_REGION env var

// New: Video generation
const response = await client.complete({
    model: "runware-video",
    prompt: "Generate a sunset video"
});
```

### No Breaking Changes
- All v0.0.x code still works
- New types are optional
- Existing functions unchanged

---

## Testing Your Migration

### Step 1: Update Dependency
```toml
[dependencies]
modelsuite = "0.1.0"  # was "0.0.x"
```

### Step 2: Run Tests
```bash
cargo test
```

✅ All existing tests should pass

### Step 3: Try New Features (Optional)
```rust
// Try one new feature
let request = CompletionRequest::new("gemini-2.0-flash", messages)
    .with_thinking(ThinkingConfig::enabled(5000));
```

### Step 4: No Breaking Changes?
If existing code breaks, please file an issue:
https://github.com/yourorg/modelsuite/issues

---

## Common Patterns

### Adding Thinking to Your Code
```rust
// Before (if using o1/o3)
let request = CompletionRequest::new("o1", messages);

// After (unified across 4 providers)
use modelsuite::types::ThinkingConfig;

let request = CompletionRequest::new("gemini-2.0-flash", messages)
    .with_thinking(ThinkingConfig::enabled(5000));
```

### Using Regional Endpoints
```rust
// Mistral with EU compliance
use modelsuite::providers::chat::mistral::MistralRegion;

let config = MistralConfig::new(api_key)
    .with_region(MistralRegion::EU);

let provider = MistralProvider::with_config(config)?;
```

### Medical Domain
```rust
// Medical AI with one line
let provider = VertexProvider::for_medical_domain(
    project_id,
    location,
    access_token
)?;
```

---

## FAQ

**Q: Will my existing code break?**
A: No. All features are additions. Existing code works unchanged.

**Q: Do I need to update immediately?**
A: No rush. v0.1.0 is fully backward compatible.

**Q: Should I use the new ThinkingConfig or old reasoning_effort?**
A: Either works. ThinkingConfig is more portable across providers.

**Q: What about the video/ modality?**
A: It's new. Won't affect existing image or other code.

**Q: When will contingent providers be available?**
A: When their APIs approve access. Skeletons are ready.

**Q: Can I use v0.1.0 features in production?**
A: Yes. All new features are production-ready.

---

## Support

- **Documentation:** https://github.com/yourorg/modelsuite/docs
- **Issues:** https://github.com/yourorg/modelsuite/issues
- **Examples:** See `examples/` directory for v0.1.0 patterns

---

**Updated:** January 3, 2026
