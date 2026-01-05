# Phase 1: Extended Thinking Implementation Guide

**Phase:** Phase 1.1 & 1.2 - Extended Thinking Completion
**Timeline:** Jan 6-17, 2026 (Week 1-2)
**Priority:** CRITICAL (Establishes pattern for all future work)
**Effort:** 5 developer-days total (3 days + 2 days)
**Target:** 4/4 extended thinking providers working with 100% test pass rate

---

## Overview

Phase 1 adds deep thinking/reasoning capabilities to two more providers, completing the extended thinking quadrant. This phase is **critical path** because:

1. **Pattern Establishment:** The approach you take here becomes the template for domain-specific models (Phase 5)
2. **Infrastructure:** ThinkingConfig mapping is the foundation for all thinking-enabled providers
3. **Tests:** These tests establish the benchmark for other thinking providers

---

## Current State (Jan 3, 2026)

**Providers with Extended Thinking (2/4):**
- ✅ OpenAI (o3, o1) - `reasoning_effort` mapping implemented
- ✅ Anthropic (Claude) - `extended_thinking` field implemented

**Providers Without Extended Thinking Yet (2/4):**
- ⏳ Google Gemini (via Vertex) - TASK 1.1
- ⏳ DeepSeek-R1 - TASK 1.2

**Infrastructure:**
- ✅ `ThinkingConfig` struct exists in `src/types.rs`
- ✅ `ThinkingType::Enabled` and `ThinkingType::Disabled` exist
- ✅ Optional `budget_tokens` field for token-based budgeting

---

## Task 1.1: Google Gemini Deep Thinking (3 days)

### File to Modify
**Path:** `/home/yfedoseev/projects/modelsuite/src/providers/chat/vertex.rs`

### Research References
- **Gemini API Docs:** https://ai.google.dev/api/rest/v1beta/models/generateContent
- **Thinking Feature:** Look for `systemInstruction.thinking` in Vertex AI documentation
- **Pattern Reference:** See `src/providers/chat/openai.rs:153-172` for reasoning_effort mapping
- **Current Vertex File:** Already supports 40+ models, just needs thinking parameter

### Implementation Pattern

The Vertex provider needs to:

1. **Add thinking field to VertexRequest struct** (similar to how OpenAI adds reasoning_effort)
   - Field: `thinking: Option<VertexThinking>`
   - Only serialize if present (`#[serde(skip_serializing_if = "Option::is_none")]`)

2. **Create VertexThinking struct** (parallel to Vertex's system design)
   ```rust
   pub struct VertexThinking {
       enabled: bool,
       budget_tokens: Option<u32>,  // Optional budget (not all Vertex models use it)
   }
   ```

3. **Map ThinkingConfig → VertexThinking** in `convert_request()` method
   - Similar to OpenAI's mapping in openai.rs:161-172
   - Handle `Disabled` → don't include thinking field
   - Handle `Enabled` → set thinking.enabled = true
   - Map budget_tokens directly if provided

4. **Document supported models**
   - Gemini 2.0 supports thinking
   - Update docs to indicate thinking support
   - Add example showing thinking usage

### Code Location Hints

**Find these sections in vertex.rs:**
1. Line ~200-300: Look for VertexRequest struct definition
   - Add `thinking: Option<VertexThinking>` field here
2. Line ~400-500: Look for convert_request() method
   - Add thinking config mapping here (after `temperature`, before API call)
3. Search for "pub struct VertexRequest"
   - Add VertexThinking struct definition nearby

### Tests Required (3 test cases minimum)

**Test 1: Thinking config serialization**
```rust
#[test]
fn test_vertex_thinking_disabled() {
    // ThinkingConfig { thinking_type: Disabled } → request.thinking = None
    // Assert: serialized JSON has no "thinking" field
}

#[test]
fn test_vertex_thinking_enabled() {
    // ThinkingConfig { thinking_type: Enabled, budget_tokens: Some(5000) }
    // Assert: serialized JSON has thinking.enabled = true, budget_tokens = 5000
}

#[test]
fn test_vertex_thinking_enabled_no_budget() {
    // ThinkingConfig { thinking_type: Enabled, budget_tokens: None }
    // Assert: serialized JSON has thinking.enabled = true (use default budget)
}
```

**Integration Test (optional, requires Vertex API key):**
```rust
#[tokio::test]
#[ignore]  // Run manually with valid credentials
async fn test_gemini_deep_thinking_real_api() {
    let provider = VertexProvider::from_env().expect("Vertex credentials");

    let request = CompletionRequest::new("What is 7 + 8?")
        .with_thinking(ThinkingConfig::enabled());

    let response = provider.complete(&request).await;
    assert!(response.is_ok());
    // Verify response has reasoning prefix in content
}
```

### Acceptance Criteria for 1.1

- [ ] VertexThinking struct implemented
- [ ] Thinking field added to VertexRequest
- [ ] convert_request() maps ThinkingConfig to VertexThinking
- [ ] Serialization works correctly (no field if disabled)
- [ ] 3+ unit tests pass
- [ ] Integration test passes (with real credentials)
- [ ] Gemini 2.0 model documented as supporting thinking
- [ ] Example code in docs/PROVIDERS.md updated
- [ ] Zero clippy warnings
- [ ] Zero new test failures (634 tests still passing)

---

## Task 1.2: DeepSeek-R1 Thinking Support (2 days)

### File to Modify
**Path:** `/home/yfedoseev/projects/modelsuite/src/providers/chat/deepseek.rs`

### Research References
- **DeepSeek API:** https://api-docs.deepseek.com/
- **R1 Model:** `deepseek-reasoner` (already mentioned in deepseek.rs comments!)
- **Pattern Reference:** See `src/providers/chat/openai.rs` for reasoning_effort mapping
- **Current Support:** deepseek.rs already supports model selection

### Implementation Pattern

DeepSeek has a **simpler pattern** than OpenAI because model selection happens automatically:

1. **Key Insight:** DeepSeek already supports two models
   - `deepseek-chat` - Standard chat model (v3)
   - `deepseek-reasoner` - Reasoning model (R1)
   - These are already in comments in deepseek.rs!

2. **Implementation approach:** Model selection based on thinking config
   - When `ThinkingConfig.enabled = true` → use `deepseek-reasoner` model
   - When `ThinkingConfig.enabled = false` → use `deepseek-chat` model
   - Add optional thinking config to request

3. **Add thinking field to DSRequest struct** (the DeepSeek request)
   - Field: `thinking: Option<DSThinking>`
   - Only for completeness (DeepSeek may not expose thinking field)

4. **Implement model selection logic**
   ```rust
   fn select_model(&self, request: &CompletionRequest) -> String {
       if request.thinking.as_ref().map(|t| t.is_enabled()).unwrap_or(false) {
           "deepseek-reasoner".to_string()
       } else {
           "deepseek-chat".to_string()  // or use request.model if specified
       }
   }
   ```

5. **Update documentation**
   - Document that thinking automatically selects R1 model
   - Add examples of thinking config usage
   - Document R1 performance characteristics

### Code Location Hints

**Find these sections in deepseek.rs:**
1. Line ~30-50: Look at DSRequest struct
   - Check if thinking field exists
   - Add if missing: `thinking: Option<DSThinking>`
2. Line ~100-150: Look for model selection in build_request or similar
   - Add logic to auto-select deepseek-reasoner when thinking enabled
3. Line ~200+: Look for convert_request() method
   - Add thinking field mapping here

### Tests Required (2 test cases minimum)

**Test 1: Model selection with thinking**
```rust
#[test]
fn test_deepseek_model_selection_with_thinking() {
    let provider = DeepSeekProvider::with_api_key("test-key").unwrap();

    let request = CompletionRequest::new("Solve this math problem")
        .with_thinking(ThinkingConfig::enabled());

    let ds_request = provider.build_request(&request);
    assert_eq!(ds_request.model, "deepseek-reasoner");
}

#[test]
fn test_deepseek_model_selection_without_thinking() {
    let provider = DeepSeekProvider::with_api_key("test-key").unwrap();

    let request = CompletionRequest::new("What time is it?");

    let ds_request = provider.build_request(&request);
    assert_eq!(ds_request.model, "deepseek-chat");
}
```

**Integration Test (optional, requires DeepSeek API key):**
```rust
#[tokio::test]
#[ignore]  // Run manually with valid credentials
async fn test_deepseek_r1_real_api() {
    let provider = DeepSeekProvider::from_env().expect("DeepSeek credentials");

    let request = CompletionRequest::new("What is 7 + 8?")
        .with_thinking(ThinkingConfig::enabled());

    let response = provider.complete(&request).await;
    assert!(response.is_ok());
}
```

### Acceptance Criteria for 1.2

- [ ] Model selection logic implemented
- [ ] thinking field added to request (if needed)
- [ ] Auto-selection of deepseek-reasoner when thinking enabled
- [ ] 2+ unit tests pass
- [ ] Integration test passes (with real credentials)
- [ ] Both models documented (deepseek-chat and deepseek-reasoner)
- [ ] Example code shows R1 usage
- [ ] Zero clippy warnings
- [ ] Zero new test failures (634 tests still passing)

---

## Implementation Sequence (Recommended)

### Day 1-2 (Monday-Tuesday, Jan 6-7): Gemini Deep Thinking
```
Morning:   Review Vertex API docs + OpenAI pattern
Midday:    Implement VertexThinking struct + add field
Afternoon: Implement convert_request() mapping
Evening:   Write unit tests (3 cases)

Next Day:  Integration test (optional), documentation
```

### Day 3 (Wednesday, Jan 8): Voice team sync + any fixes
```
Morning:   Fix any failing tests from Day 1-2
Midday:    Code review from Dev 2 & 3
Afternoon: Prepare for architecture sync (1 hour)
```

### Day 4-5 (Thursday-Friday, Jan 9-10): DeepSeek-R1
```
Day 4:     Model selection + field implementation
Day 5:     Tests, integration, documentation, Checkpoint 1
```

---

## Common Implementation Gotchas

### Gotcha 1: Thinking vs Reasoning Terminology
- **OpenAI:** Uses `reasoning_effort` (qualitative levels: low/medium/high)
- **Anthropic:** Uses `extended_thinking` (boolean flag)
- **Google Gemini:** Uses `thinking` (boolean + optional budget)
- **DeepSeek:** Uses model selection (deepseek-reasoner vs deepseek-chat)

**Solution:** Map unified `ThinkingConfig` to each provider's specific format in convert_request()

### Gotcha 2: Response Handling
- Some providers return thinking content in response
- Some hide thinking internally
- Don't assume thinking content in CompletionResponse

**Solution:** Map what's available, document what's hidden

### Gotcha 3: API Rate Limits
- Thinking/reasoning requests are more expensive
- May have lower rate limits
- Don't exhaust quota during testing

**Solution:** Use mock tests for CI/CD, manual integration tests only

### Gotcha 4: Model Availability
- Not all models support thinking
- Regional variations in API endpoints
- Version updates may change support

**Solution:** Document explicitly which models support thinking

---

## Development Workflow

### 1. Setup (Before coding)
```bash
# Verify current state
cd /home/yfedoseev/projects/modelsuite
cargo test --lib --all-features  # Should see 634 passing

# Create feature branch
git checkout -b phase/1.1-gemini-thinking
```

### 2. Implement (During coding)
```bash
# Work on vertex.rs or deepseek.rs
# Test frequently:
cargo test --lib --all-features

# Check code quality:
cargo fmt --check
cargo clippy --all-targets
```

### 3. Commit (When feature is done)
```bash
git add src/providers/chat/vertex.rs
git commit -m "phase/1.1-gemini-thinking: Implement Google Gemini Deep Thinking support"
```

### 4. Code Review
- Post PR on GitHub
- Request review from Dev 2 & 3
- Fix any clippy/fmt issues
- Merge when approved

### 5. Integration Test (Optional)
```bash
# If you have credentials:
GOOGLE_CLOUD_PROJECT=... VERTEX_ACCESS_TOKEN=... \
cargo test test_gemini_deep_thinking_real_api -- --ignored
```

---

## Testing Strategy

### Unit Tests (Required, CI/CD)
- ✅ Test serialization with thinking enabled/disabled
- ✅ Test model selection logic
- ✅ Test config mapping
- Run with: `cargo test --lib`

### Mock Tests (Recommended, CI/CD)
- ✅ Mock API responses
- ✅ Verify request format
- ✅ Test error handling
- Use wiremock if available

### Integration Tests (Optional, Manual)
- ✅ Real API calls with real credentials
- ✅ Test end-to-end flow
- ✅ Verify response handling
- Mark with `#[ignore]`
- Run with: `cargo test -- --ignored`

### Benchmark Tests (Nice-to-have)
- ⏳ Thinking vs non-thinking latency comparison
- ⏳ Token usage comparison
- ⏳ Cost comparison
- For documentation, not CI/CD

---

## Documentation Updates

### Files to Update

1. **PROVIDERS.md** (add to Gemini + DeepSeek sections)
   ```markdown
   ### Thinking/Reasoning Support
   - ✅ Extended thinking supported
   - Model: gemini-2.0-flash-exp
   - Configuration: See example below

   Example:
   ```rust
   let request = CompletionRequest::new("Complex problem")
       .with_thinking(ThinkingConfig::enabled());
   ```
   ```

2. **docs/extended_thinking_guide.md** (create if needed)
   - Compare thinking support across 4 providers
   - Explain the mapping differences
   - Give examples for each

3. **Code examples** in repo
   - Add example showing Gemini thinking
   - Add example showing DeepSeek-R1
   - Add example showing thinking vs non-thinking

---

## Success Checklist

### Before Starting
- [ ] Read OpenAI pattern (openai.rs:153-172)
- [ ] Review Vertex API docs
- [ ] Review DeepSeek API docs
- [ ] Understand ThinkingConfig in types.rs

### During Implementation (Task 1.1)
- [ ] VertexThinking struct created
- [ ] thinking field added to VertexRequest
- [ ] Serialization works correctly
- [ ] Unit tests written (3+ cases)
- [ ] Integration test written (optional)
- [ ] Documentation updated
- [ ] cargo test passing
- [ ] cargo fmt passing
- [ ] cargo clippy passing

### During Implementation (Task 1.2)
- [ ] Model selection logic implemented
- [ ] thinking field added to DSRequest
- [ ] Unit tests written (2+ cases)
- [ ] Integration test written (optional)
- [ ] Documentation updated
- [ ] cargo test passing
- [ ] cargo fmt passing
- [ ] cargo clippy passing

### Checkpoint 1 (Friday, Jan 10)
- [ ] 4/4 extended thinking providers verified
- [ ] 634+ tests passing (100% pass rate)
- [ ] Build clean (0 warnings)
- [ ] Go/No-Go decision: PROCEED to Week 2 ✅

---

## Helpful Commands

```bash
# Run tests for specific provider
cargo test vertex --lib --all-features
cargo test deepseek --lib --all-features

# Run tests matching pattern
cargo test thinking --lib --all-features

# Run with backtrace for debugging
RUST_BACKTRACE=1 cargo test --lib --all-features

# Check individual file
cargo check --lib

# Format code
cargo fmt

# Lint code
cargo clippy --all-targets

# Run integration tests only
cargo test -- --ignored

# Profile test performance
cargo test --lib --release --all-features
```

---

## References & Resources

**Code References:**
- OpenAI thinking pattern: `src/providers/chat/openai.rs:153-172`
- ThinkingConfig: Search `types.rs` for `pub struct ThinkingConfig`
- Provider trait: `src/provider.rs`
- Test patterns: Look at existing tests in openai.rs, anthropic.rs

**API Documentation:**
- Gemini: https://ai.google.dev/api/rest/v1beta/models/generateContent
- Vertex: https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini
- DeepSeek: https://api-docs.deepseek.com/

**Learning Resources:**
- Serde serialization: https://serde.rs/
- Async Rust: https://tokio.rs/
- reqwest client: https://docs.rs/reqwest/

---

## When Stuck

**If VertexThinking serialization fails:**
- Check `#[serde(skip_serializing_if = "Option::is_none")]`
- Verify field names match API docs exactly
- Test with `serde_json::to_string()` to debug

**If tests fail:**
- Check that CompletionRequest has thinking method
- Verify ThinkingConfig struct fields
- Look at other provider tests for patterns

**If API returns errors:**
- Check that thinking field is correctly mapped
- Verify model names are correct
- Ensure credentials are valid

**Still stuck after 1 hour:**
- Post in #modelsuite-q1-2026 channel
- Tag Dev 2 or Dev 3 for help
- Share: error message + code snippet + what you've tried

---

## Next Phase

Once Phase 1 is complete:
- Extended Thinking pattern becomes template for Phase 5 (domain-specific)
- Same mapping technique used for blending providers by reasoning capability
- Infrastructure ready for future reasoning models (Grok, etc.)

---

**Document Created:** January 3, 2026
**Status:** ✅ READY FOR IMPLEMENTATION
**Start Date:** January 6, 2026
**Target Completion:** January 10, 2026
**Effort:** 5 developer-days
