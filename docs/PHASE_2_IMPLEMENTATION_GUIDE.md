# Phase 2: Regional Provider Expansion Implementation Guide

**Phase:** Phase 2.1-2.4 - Regional Provider Expansion
**Timeline:** Jan 13-24, 2026 (Week 2-3)
**Priority:** HIGH (Enables regional coverage, runs parallel with Phase 1)
**Effort:** 11 developer-days total (2+3+2+2 days)
**Target:** 4 new regional providers, 2+ guaranteed (Mistral EU + Maritaca)

---

## Overview

Phase 2 adds regional provider coverage for Europe, France, Brazil, and Latin America. This phase:

1. **Expands Geographic Coverage:** 6 → 7+ regions
2. **Adds Government-Backed Solutions:** LightOn (France) + LatamGPT (LatAm)
3. **Enhances Existing Providers:** Mistral EU endpoint + Maritaca-3 model
4. **Tests Contingency Planning:** Both LightOn and LatamGPT are contingent on external API access

---

## Guaranteed Implementations (2/4)

These WILL be completed in Week 2 regardless of external factors:

### Task 2.1: Mistral EU Regional Support (2 days)
**File:** `src/providers/chat/mistral.rs`
**Status:** ✅ GUARANTEED (API already available)

**What It Does:**
- Adds GDPR-compliant EU endpoint support to Mistral provider
- Users can specify `region=eu` to use `api.eu.mistral.ai`
- Same models available, just different endpoint

**Implementation Pattern:**
```rust
pub enum MistralRegion {
    Global,  // api.mistral.ai (default)
    EU,      // api.eu.mistral.ai (GDPR)
}
```

**Changes Needed:**
1. Add region enum to MistralProvider config
2. Update api_url() method to select endpoint based on region
3. Support MISTRAL_REGION environment variable
4. Add tests for region selection

**Acceptance Criteria:**
- [ ] Region enum implemented
- [ ] Both endpoints working
- [ ] Tests for region selection (2+ cases)
- [ ] Zero breaking changes (default = Global)
- [ ] 634+ tests still passing

---

### Task 2.3: Maritaca AI Enhancement (2 days)
**File:** `src/providers/chat/maritaca.rs` (existing, enhance)
**Status:** ✅ GUARANTEED (API already available)

**What It Does:**
- Adds Maritaca-3 model support to existing Maritaca provider
- Enhances Portuguese language optimization
- Improves Brazilian regional context handling

**Implementation Pattern:**
```rust
pub enum MaritacaModel {
    Maritaca2,  // Existing
    Maritaca3,  // New
    Maritaca3Plus,  // New variant if available
}
```

**Changes Needed:**
1. Add new model enum variants
2. Update model selection logic
3. Test new models with Portuguese content
4. Add regional configuration option
5. Document Brazilian context handling

**Acceptance Criteria:**
- [ ] New model variants implemented
- [ ] Model selection working
- [ ] Portuguese language tests pass (2+ cases)
- [ ] Documentation updated
- [ ] 634+ tests still passing

---

## Contingent Implementations (0-2/4)

These depend on external API access. Decision points at Jan 12 and Jan 14.

### Task 2.2: LightOn France Provider (3 days or 0.5 days)
**File:** `src/providers/chat/lighton.rs` (NEW)
**Status:** ⚠️ CONTINGENT on API access
**Decision Point:** Jan 12 (Day 6)

**Decision Matrix:**
```
IF API Available by Jan 12:
  → Full implementation (3 days)
  → Go to Week 2 as planned

IF No Response by Jan 12:
  → Skeleton implementation (0.5 days)
  → Document as "Partnership pending"
  → Defer to Phase 4 (Week 3)

IF Negative Response:
  → Skip entirely
  → Reallocate 3 days to other work
```

**What It Does (if full impl):**
- New VLM provider from France (LightOn)
- European market expansion
- Partnership with government-backed initiative

**Implementation Pattern:**
```rust
pub struct LightOnProvider {
    config: ProviderConfig,
    client: Client,
}

pub enum LightOnModel {
    VLM4,  // Vision-Language Model
    VLM4Light,  // Faster variant
}
```

**Files to Create:**
- `src/providers/chat/lighton.rs` (main)
- Tests in same file
- Documentation in PROVIDERS.md

**Acceptance Criteria (if full impl):**
- [ ] Provider struct and config
- [ ] API request/response conversion
- [ ] Model enumeration
- [ ] 3+ unit tests
- [ ] Integration test (optional)
- [ ] Documentation updated
- [ ] Zero clippy warnings

**Fallback (if no API):**
```rust
pub struct LightOnProvider;

impl LightOnProvider {
    pub fn from_env() -> Result<Self> {
        Err(Error::config(
            "LightOn API access pending partnership agreement"
        ))
    }
}
```

---

### Task 2.4: LatamGPT Regional Provider (2 days or 0.5 days)
**File:** `src/providers/chat/latamgpt.rs` (NEW)
**Status:** ⚠️ CONTINGENT on API launch timing
**Decision Point:** Jan 14 (Day 8)

**Decision Matrix:**
```
IF API Launches Jan 15-20:
  → Full implementation (2 days)
  → Go to Week 2 as planned

IF Launch Delayed to Late Jan/Feb:
  → Skeleton implementation (0.5 days)
  → Complete when API launches

IF No Timeline Available:
  → Skip for now
  → Re-evaluate in Feb
  → Reallocate 2 days to other work
```

**What It Does (if full impl):**
- New government-backed AI for Latin America
- Covers Chile + Brazil region
- Complement to Maritaca (Brazilian LLM)

**Implementation Pattern:**
```rust
pub struct LatamGPTProvider {
    config: ProviderConfig,
    client: Client,
}

pub enum LatamGPTRegion {
    Chile,
    Brazil,
}
```

**Files to Create:**
- `src/providers/chat/latamgpt.rs` (main)
- Tests in same file
- Documentation in PROVIDERS.md

**Acceptance Criteria (if full impl):**
- [ ] Provider struct and config
- [ ] Region selection
- [ ] API request/response conversion
- [ ] 2+ unit tests
- [ ] Documentation updated
- [ ] Zero clippy warnings

**Fallback (if delayed):**
```rust
pub struct LatamGPTProvider;

impl LatamGPTProvider {
    pub fn from_env() -> Result<Self> {
        Err(Error::config(
            "LatamGPT launching late January 2026"
        ))
    }
}
```

---

## Week-by-Week Timeline

### Week 2 (Jan 13-17): Regional Expansion Begins

**Monday Jan 13 - Tuesday Jan 14:**
- Start Task 2.1: Mistral EU (1.5 days)
- Start Task 2.3: Maritaca AI (1.5 days)
- Finalize Task 2.2 decision (LightOn: full or skeleton?)
- Finalize Task 2.4 decision (LatamGPT: full or skeleton?)

**Wednesday Jan 15:**
- Continue 2.1 + 2.3
- Mid-week sync (1 hour)
- Address any blockers

**Thursday Jan 16 - Friday Jan 17:**
- Finish 2.1 + 2.3
- Begin 2.2 (if full impl approved)
- Begin 2.4 (if full impl approved)
- Checkpoint 2 (Friday) - assess progress

**Deliverables by Jan 17:**
- Mistral EU: ✅ Complete
- Maritaca-3: ✅ Complete
- LightOn: Full OR Skeleton (depends on API)
- LatamGPT: Full OR Skeleton (depends on timing)

### Week 3 (Jan 20-24): Finalization

**Monday Jan 20 - Wednesday Jan 22:**
- Finish any 2.2 (LightOn) work
- Finish any 2.4 (LatamGPT) work
- Continuous integration testing
- Documentation finalization

**Thursday Jan 23 - Friday Jan 24:**
- Feature freeze begins
- Final testing
- Checkpoint 3 assessment

---

## Implementation Details

### Task 2.1: Mistral EU - Step by Step

**Step 1: Understanding the Change** (30 min)
- Review current Mistral provider structure
- Understand how API URL is currently set
- Check if region configuration exists elsewhere

**Step 2: Add Region Enum** (30 min)
```rust
// Add to mistral.rs, near imports
pub enum MistralRegion {
    Global,  // api.mistral.ai (default)
    EU,      // api.eu.mistral.ai (GDPR)
}
```

**Step 3: Update Config** (1 hour)
- Add region field to MistralProvider or MistralConfig
- Update constructor to accept region parameter
- Make region configurable via env var: MISTRAL_REGION

**Step 4: Update API URL Method** (1 hour)
```rust
fn api_url(&self) -> &str {
    match self.region {
        MistralRegion::Global => "https://api.mistral.ai/v1/chat/completions",
        MistralRegion::EU => "https://api.eu.mistral.ai/v1/chat/completions",
    }
}
```

**Step 5: Write Tests** (1 hour)
```rust
#[test]
fn test_mistral_global_url() {
    // Default should be global
    // assert_eq!(provider.api_url(), "https://api.mistral.ai/...");
}

#[test]
fn test_mistral_eu_url() {
    // With EU region set
    // assert_eq!(provider.api_url(), "https://api.eu.mistral.ai/...");
}
```

**Step 6: Documentation** (30 min)
- Update PROVIDERS.md with regional info
- Add example showing region selection
- Document GDPR compliance for EU region

---

### Task 2.3: Maritaca AI - Step by Step

**Step 1: Review Current Implementation** (30 min)
- Check existing Maritaca model enum
- Understand API model names
- Review current capabilities

**Step 2: Add New Model Variants** (1 hour)
```rust
// Update MaritacaModel enum
pub enum MaritacaModel {
    Maritaca2,     // Existing
    Maritaca3,     // New
    // Maritaca3Plus,  // If available
}
```

**Step 3: Update Model Selection** (1 hour)
- Ensure all models route correctly to API
- Test model name mapping

**Step 4: Add Portuguese Optimization** (1 hour)
- Document Portuguese-specific handling
- Add examples in Portuguese
- Document regional context

**Step 5: Write Tests** (1 hour)
```rust
#[test]
fn test_maritaca3_model_selection() {
    // Verify Maritaca3 is selectable
}

#[test]
fn test_maritaca_portuguese_content() {
    // Test Portuguese prompt handling
}
```

**Step 6: Documentation** (30 min)
- Update PROVIDERS.md with Maritaca-3
- Add Portuguese language example
- Document Brazil-specific features

---

### Task 2.2: LightOn - Full Implementation (if API available)

**File Structure:**
```rust
// src/providers/chat/lighton.rs

use crate::types::{CompletionRequest, CompletionResponse, ...};
use crate::provider::Provider;

pub struct LightOnProvider { ... }
pub struct LightOnConfig { ... }
pub enum LightOnModel { VLM4, VLM4Light }

impl LightOnProvider {
    pub fn from_env() -> Result<Self> { ... }
    pub fn with_api_key(api_key: impl Into<String>) -> Result<Self> { ... }
}

#[async_trait]
impl Provider for LightOnProvider { ... }
```

**Implementation Order:**
1. Provider struct + config
2. from_env() + with_api_key() constructors
3. Request/response conversion (follow OpenAI pattern)
4. API call implementation
5. Tests (unit + integration)
6. Documentation

**Estimated Effort:** 3 days
- Day 1: Struct + constructor + API call
- Day 2: Request/response conversion + tests
- Day 3: Integration test + documentation

---

### Task 2.4: LatamGPT - Full Implementation (if API available)

**File Structure:**
```rust
// src/providers/chat/latamgpt.rs

pub struct LatamGPTProvider { ... }
pub enum LatamGPTRegion { Chile, Brazil }

impl LatamGPTProvider {
    pub fn from_env() -> Result<Self> { ... }
}

#[async_trait]
impl Provider for LatamGPTProvider { ... }
```

**Implementation Order:**
1. Provider struct + config + region enum
2. Constructor methods
3. Request/response conversion
4. API call
5. Tests
6. Documentation

**Estimated Effort:** 2 days
- Day 1: Struct + constructor + API call
- Day 2: Tests + documentation

---

## Decision Points & Contingency Plan

### Decision Point 1: Jan 12 (Day 6, Week 2)

**LightOn API Access Status Check:**

```
EMAIL SENT:        Jan 3
FOLLOW-UP:         Jan 8
DECISION DATE:     Jan 12 (Friday end-of-day)

OUTCOMES:
1. Email response received ✅
   → Decision: Full implementation
   → Action: Proceed with Task 2.2 (3 days)
   → Timeline: Jan 15-17 (Thu-Fri + Mon spillover)

2. No response yet ⏳
   → Decision: Skeleton implementation
   → Action: Create skeleton (0.5 day)
   → Timeline: Jan 15 (Thu)
   → Defer: Full impl to Phase 4 (Week 3)

3. Negative response ❌
   → Decision: Skip entirely
   → Action: Reallocate 3 days
   → New plan: Extra work on 2.3 or Phase 5
```

**How to Execute Decision:**
- Check email response on Jan 12 morning
- If decision is "Full", start Task 2.2 on Jan 13
- If decision is "Skeleton", spend 0.5 days on skeleton, then pivot
- If decision is "Skip", notify team and reallocate

---

### Decision Point 2: Jan 14 (Day 8, Week 2)

**LatamGPT API Launch Status Check:**

```
EMAIL SENT:        Jan 4
PUBLIC ROADMAP:    Check latamgpt.com/roadmap
DECISION DATE:     Jan 14 (Tuesday end-of-day)

OUTCOMES:
1. API launching Jan 15-20 ✅
   → Decision: Full implementation
   → Action: Proceed with Task 2.4 (2 days)
   → Timeline: Jan 16-17 (Thu-Fri)

2. API launching late Jan/Feb ⏳
   → Decision: Skeleton implementation
   → Action: Create skeleton (0.5 day)
   → Timeline: Jan 16 (Thu)
   → Defer: Full impl to Phase 4 (Week 3)
   → Plan: Complete when API launches

3. API launch delayed >Feb 15 ❌
   → Decision: Skip entirely
   → Action: Reallocate 2 days
   → New plan: Extra work on other phases
```

**How to Execute Decision:**
- Check public roadmap on Jan 13
- Send follow-up email if timeline unclear
- Final decision by Jan 14 EOD
- Communicate decision to Dev 1 (Lead) immediately

---

## Testing Strategy

### Unit Tests (Required for all tasks)

**Mistral EU Tests (2 minimum):**
```rust
#[test]
fn test_mistral_region_enum() { }

#[test]
fn test_mistral_api_url_selection() { }
```

**Maritaca-3 Tests (2 minimum):**
```rust
#[test]
fn test_maritaca_model_enum() { }

#[test]
fn test_maritaca_request_conversion() { }
```

**LightOn Tests (3 minimum, if full impl):**
```rust
#[test]
fn test_lighton_request_conversion() { }

#[test]
fn test_lighton_model_selection() { }

#[test]
fn test_lighton_response_parsing() { }
```

**LatamGPT Tests (2 minimum, if full impl):**
```rust
#[test]
fn test_latamgpt_region_selection() { }

#[test]
fn test_latamgpt_request_conversion() { }
```

### Integration Tests (Optional, requires credentials)

```rust
#[tokio::test]
#[ignore]
async fn test_mistral_eu_real_api() { }

#[tokio::test]
#[ignore]
async fn test_maritaca3_real_api() { }

#[tokio::test]
#[ignore]
async fn test_lighton_real_api() { }  // If full impl

#[tokio::test]
#[ignore]
async fn test_latamgpt_real_api() { }  // If full impl
```

---

## Code Review Checklist

When submitting PRs for review:

**Before Review:**
- [ ] All tests pass locally
- [ ] `cargo fmt` done
- [ ] `cargo clippy` clean
- [ ] No warnings in build
- [ ] Documentation updated

**Code Quality:**
- [ ] No unwrap() without justification
- [ ] Proper error handling
- [ ] Consistent with existing patterns
- [ ] Comments for complex logic

**Testing:**
- [ ] Unit tests for new code
- [ ] No test failures from changes
- [ ] Optional: integration test written (even if not run)

**Documentation:**
- [ ] PROVIDERS.md updated
- [ ] Example code provided
- [ ] Region/model options documented
- [ ] README updated if user-facing

---

## References & Resources

**Code Patterns:**
- OpenAI provider: `src/providers/chat/openai.rs` (best pattern reference)
- Mistral existing: `src/providers/chat/mistral.rs` (add region here)
- Maritaca existing: `src/providers/chat/maritaca.rs` (enhance this)

**API Documentation:**
- Mistral: https://docs.mistral.ai/api/
- Maritaca: https://www.maritaca.ai/ (internal API docs)
- LightOn: https://lighton.ai/developer (pending API access)
- LatamGPT: https://latamgpt.com/ (pending API access)

**Learning Resources:**
- Enum patterns in Rust: https://doc.rust-lang.org/book/ch06-00-enums.html
- Async/await: https://tokio.rs/
- Serde serialization: https://serde.rs/

---

## Success Criteria Summary

### Week 2 End (Jan 17)

**Guaranteed Complete:**
- ✅ Mistral EU regional support
- ✅ Maritaca-3 model support

**Contingent on API Access:**
- ⏳ LightOn (full if API available by Jan 12, else skeleton)
- ⏳ LatamGPT (full if API ready by Jan 14, else skeleton)

**Quality Gates:**
- ✅ 2+ unit tests per task
- ✅ 100% test pass rate (634+ tests)
- ✅ Zero clippy warnings
- ✅ Documentation updated
- ✅ Code reviewed and approved

### Week 3 End (Jan 24)

- ✅ All contingencies resolved
- ✅ Full or skeleton implementations in place
- ✅ Integration testing complete
- ✅ Ready for Week 4 release prep

---

## Timeline & Effort Tracking

```
PHASE 2 EFFORT BREAKDOWN
═══════════════════════════════════════════════════════════════

Task 2.1: Mistral EU
├─ Day 1: Enum + config + constructor (1 day)
├─ Day 2: API URL selection + tests (1 day)
└─ Total: 2 days ✅ GUARANTEED

Task 2.3: Maritaca AI
├─ Day 1: Model enum + selection (1 day)
├─ Day 2: Tests + documentation (1 day)
└─ Total: 2 days ✅ GUARANTEED

Task 2.2: LightOn (Contingent)
├─ IF API available by Jan 12:
│  ├─ Day 1: Provider struct + constructor (1 day)
│  ├─ Day 2: Request/response conversion (1 day)
│  └─ Day 3: Tests + documentation (1 day)
│  └─ Total: 3 days ✅
├─ ELSE (No API by Jan 12):
│  └─ Skeleton: (0.5 days)
│  └─ Total: 0.5 days ✅

Task 2.4: LatamGPT (Contingent)
├─ IF API available by Jan 14:
│  ├─ Day 1: Provider struct + constructor (1 day)
│  └─ Day 2: Tests + documentation (1 day)
│  └─ Total: 2 days ✅
├─ ELSE (No API by Jan 14):
│  └─ Skeleton: (0.5 days)
│  └─ Total: 0.5 days ✅

TOTAL PHASE 2:
├─ Best case (all full): 2 + 2 + 3 + 2 = 9 days
├─ Realistic (1-2 deferred): 2 + 2 + 2 + 0.5 = 6.5 days
└─ Minimum (both deferred): 2 + 2 + 0.5 + 0.5 = 5 days

CONTINGENCY BUFFER: 4 days (if full → defer to Phase 4)
```

---

**Document Created:** January 3, 2026
**Status:** ✅ READY FOR IMPLEMENTATION
**Start Date:** January 13, 2026 (Week 2)
**Target Completion:** January 24, 2026 (Week 3)
**Effort:** 5-11 developer-days (depends on API access)
