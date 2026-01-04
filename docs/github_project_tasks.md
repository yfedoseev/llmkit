# LLMKit Q1 2026 Gap Closure - GitHub Project Board Tasks

**Created:** January 3, 2026
**For:** GitHub Projects "Q1 2026 Gap Closure"
**Total Tasks:** 23

---

## Project Board Columns

1. **Backlog** - All tasks not yet started
2. **Week 1** - Jan 6-10 (Extended Thinking focus)
3. **Week 2** - Jan 13-17 (Regional + Voice + Video)
4. **Week 3** - Jan 20-24 (Regional Phase 2 + Domain)
5. **Week 4** - Jan 27-31 (Polish + Release prep)
6. **In Progress** - Currently being worked on
7. **Done** - Completed and merged

---

## Phase 1: Extended Thinking (Critical Path)

### Task 1.1: Google Gemini Deep Thinking
```
Title: phase/1.1-gemini-thinking: Implement Google Gemini Deep Thinking support
Effort: 3 days
Priority: CRITICAL
Assignee: Developer 1 (Lead)
Status: Pending (Week 1)

Description:
Add deep thinking mode to Google Vertex provider for Gemini 2.0 models.
Follow OpenAI reasoning_effort pattern.

Files to Modify:
- src/providers/chat/vertex.rs

Tests Required:
- Unit: thinking config serialization (3 test cases)
- Integration: real Gemini API with thinking enabled
- Benchmark: latency vs OpenAI o3

Acceptance Criteria:
- Gemini 2.0 thinking mode callable
- Tests passing
- Zero warnings in cargo build
- Documentation updated (PROVIDERS.md)

Blockers: None
Notes: Pattern reference in src/providers/chat/openai.rs:150-193
```

### Task 1.2: DeepSeek-R1 Thinking Support
```
Title: phase/1.2-deepseek-r1: Add DeepSeek-R1 model variants with reasoning
Effort: 2 days
Priority: CRITICAL
Assignee: Developer 1 (Lead)
Status: Pending (Week 1)

Description:
Add R1 model selection logic to DeepSeek provider. Auto-select reasoning model
when ThinkingConfig.enabled = true.

Files to Modify:
- src/providers/chat/deepseek.rs

Tests Required:
- Unit: model selection logic (2 test cases)
- Integration: R1 API call with reasoning
- Benchmark: R1 vs standard on AIME problems

Acceptance Criteria:
- DeepSeek-R1 reasoning operational
- Tests passing
- Zero warnings
- Example code provided

Blockers: None
Notes: Model enum pattern already in deepseek.rs
```

### Task 1.3: Checkpoint 1 - Extended Thinking Verification
```
Title: week1-checkpoint: Verify extended thinking 4/4 providers + tests
Effort: 0.5 days
Priority: CRITICAL
Assignee: Developer 1 (Lead)
Status: Pending (Week 1)

Description:
End-of-week 1 verification that all 4 extended thinking providers are working.

Acceptance Criteria:
- OpenAI o3 + reasoning_effort ✅
- Anthropic Claude + extended_thinking ✅
- Google Gemini + thinking mode ✅
- DeepSeek-R1 + model selection ✅
- 634+ tests passing ✅
- Build clean (no warnings) ✅

Go/No-Go: MUST PASS to proceed to Week 2
```

---

## Phase 2: Regional Provider Expansion

### Task 2.1: Mistral EU Regional Support
```
Title: phase/2.1-mistral-eu: Add Mistral EU regional endpoint (GDPR-compliant)
Effort: 2 days
Priority: HIGH
Assignee: Developer 1 (Lead)
Status: Pending (Week 2)

Description:
Add region enum to Mistral provider. Support both global and EU-compliant
endpoints (api.eu.mistral.ai).

Files to Modify:
- src/providers/chat/mistral.rs

Tests Required:
- Unit: region enum and URL generation
- Integration: EU endpoint connectivity

Acceptance Criteria:
- Region selection working
- MISTRAL_REGION env var supported
- Tests passing
- Documentation updated

Blockers: None
Notes: No API changes required, EU has same models
```

### Task 2.2: LightOn France Provider
```
Title: phase/2.2-lighton-france: Integrate LightOn VLM-4 provider (France)
Effort: 3 days (or 0.5 days skeleton)
Priority: HIGH
Assignee: Developer 1 (Lead)
Status: Pending (Week 2, contingent on API access)

Description:
New provider for LightOn VLM-4. European market expansion.

Files to Modify:
- src/providers/chat/lighton.rs (NEW)

Contingency Decision Point: Day 6 of Week 2
- IF API available: Full implementation (3 days)
- IF delayed: Skeleton + feature flag (0.5 days)
- IF unavailable: Skip entirely, document alternatives

Tests Required (if full impl):
- Unit: request/response conversion
- Integration: real LightOn API

Blockers: ⚠️ API access (MEDIUM RISK)
Notes: Email sent Jan 3 to partnership@lighton.ai
Status Page: Check github.com/lighton-ai for updates
```

### Task 2.3: Maritaca AI Enhancement
```
Title: phase/2.3-maritaca-enhancement: Add Maritaca-3 model support
Effort: 2 days
Priority: HIGH
Assignee: Developer 1 (Lead)
Status: Pending (Week 2)

Description:
Enhance existing Maritaca provider with Maritaca-3 model. Portuguese language
optimization. Brazilian regional context.

Files to Modify:
- src/providers/chat/maritaca.rs

Tests Required:
- Unit: new model enum values
- Integration: Maritaca-3 API call

Acceptance Criteria:
- Maritaca-3 model selectable
- Portuguese language tests
- Tests passing

Blockers: None
Notes: Provider already implemented, just adding new model variants
```

### Task 2.4: LatamGPT Regional Provider
```
Title: phase/2.4-latamgpt-regional: Add LatamGPT Chile/Brazil provider
Effort: 2 days (or 0.5 days skeleton)
Priority: HIGH
Assignee: Developer 1 (Lead)
Status: Pending (Week 2-3, contingent on API launch)

Description:
New provider for LatamGPT. Government-backed Latin American initiative.
May launch late Jan or Feb.

Files to Modify:
- src/providers/chat/latamgpt.rs (NEW)

Contingency Decision Point: Day 8 of Week 2
- IF launched Jan 15-20: Full implementation (2 days)
- IF launching Feb: Skeleton now, complete in Phase 4
- IF delayed >Feb 15: Cancel, re-evaluate Q2

Tests Required (if full impl):
- Unit: request/response conversion
- Integration: LatamGPT API

Blockers: ⚠️ API launch timing (MEDIUM RISK)
Notes: Email sent Jan 4, check public roadmap at latamgpt.com
```

---

## Phase 3: Real-Time Voice Upgrade

### Task 3.1: Deepgram v3 Upgrade
```
Title: phase/3.1-deepgram-v3: Upgrade Deepgram to v3 with nova-3 models
Effort: 2 days
Priority: HIGH
Assignee: Developer 2 (Voice/Domain)
Status: Pending (Week 2)

Description:
Upgrade Deepgram provider from v2 to v3. Add nova-3 models with enhanced
real-time speech recognition.

Files to Modify:
- src/providers/audio/deepgram.rs

Tests Required:
- Unit: version enum and API version header
- Integration: v3 API calls
- Performance: latency measurements

Acceptance Criteria:
- nova-3 models selectable
- v2 backward compatibility maintained
- Tests passing
- Example code updated

Blockers: None
Notes: API docs reviewed, minimal changes needed
```

### Task 3.2: Enhanced ElevenLabs Streaming
```
Title: phase/3.3-elevenlabs-streaming: Add latency optimization config
Effort: 2 days
Priority: HIGH
Assignee: Developer 2 (Voice/Domain)
Status: Pending (Week 3)

Description:
Add latency enum to ElevenLabs. Support 5 latency tiers from fastest to
highest quality (0-4 scale).

Files to Modify:
- src/providers/audio/elevenlabs.rs

Tests Required:
- Unit: latency enum serialization
- Integration: streaming with different latency levels

Acceptance Criteria:
- Latency selection working
- Sub-100ms latency achievable with low setting
- Tests passing

Blockers: None
Notes: Enum pattern: LowestLatency=0 to HighestQuality=4
```

### Task 3.2: Grok Real-Time Voice (xAI)
```
Title: phase/3.2-grok-realtime: Implement Grok real-time voice (WebSocket)
Effort: 4 days (or fallback to skip)
Priority: HIGH
Assignee: Developer 2 (Voice/Domain)
Status: Pending (Week 2-3, HIGH RISK)

Description:
New real-time voice provider from xAI. WebSocket-based streaming API.
Follow OpenAI Realtime pattern.

Files to Modify:
- src/providers/audio/grok_realtime.rs (NEW)

Contingency Strategy (3-tier):
Tier 1 (Days 1-5): Partnership outreach to xAI
Tier 2 (Days 6-10): Generic WebSocket infrastructure using Deepgram pattern
Tier 3 (Days 11-12): IF no API access, ship with 2/3 voice providers

Tests Required (if API available):
- Unit: WebSocket frame serialization
- Integration: real Grok API
- Performance: sub-100ms latency

Minimum Viable: 2/3 providers (Deepgram + ElevenLabs) still achieves voice goal

Blockers: 🚧 xAI API access (HIGH RISK)
Notes: Email sent Jan 3, pattern ref: src/providers/specialized/openai_realtime.rs
Decision Point: Day 10 of Week 2
```

---

## Phase 4: Video Generation

### Task 4.1: Create Video Modality Structure
```
Title: phase/4-video-modality: Create src/providers/video/ directory structure
Effort: 1 day
Priority: HIGH
Assignee: Developer 3 (Video/Testing)
Status: Pending (Week 2)

Description:
Create new video modality directory and move RunwayML from image/ to video/.
Add feature flags and update Cargo.toml.

Files to Create:
- src/providers/video/mod.rs (NEW)
- src/providers/video/runwayml.rs (MOVED from image/)

Files to Modify:
- src/providers/mod.rs (add video module)
- src/providers/image/mod.rs (keep re-export for backward compat)
- Cargo.toml (video feature flag)

Breaking Changes:
- Import path changes to video/ (provide deprecation re-export)

Acceptance Criteria:
- New directory created
- RunwayML moved
- Re-export provides backward compatibility
- Tests still passing

Blockers: None
Notes: OK to have breaking changes (pre-1.0)
```

### Task 4.2: Runware Video Aggregator
```
Title: phase/4.1-runware-aggregator: Implement Runware video model aggregator
Effort: 3 days
Priority: HIGH
Assignee: Developer 3 (Video/Testing)
Status: Pending (Week 2)

Description:
New aggregator supporting 5+ video models: runway-gen-4.5, kling-2.0,
pika-1.0, leonardo-ultra, hailuo-mini.

Files to Modify:
- src/providers/video/runware.rs (NEW)

Tests Required (3-tier, cost-controlled):
Tier 1 (Unit): request serialization (free, CI/CD)
Tier 2 (Mock): wiremock video generation (free, CI/CD)
Tier 3 (Integration): real Runware API (manual, $10-20 budget)

Acceptance Criteria:
- Model selection working
- Request serialization correct
- Tests passing (unit + mock)
- Optional manual integration test
- Documentation updated

Budget: $50-100 for all manual testing (shared with Phase 5)
Notes: Pattern: model enum for switching
```

### Task 4.3: DiffusionRouter Skeleton
```
Title: phase/4.2-diffusion-router-skeleton: Create DiffusionRouter skeleton
Effort: 1 day
Priority: MEDIUM
Assignee: Developer 3 (Video/Testing)
Status: Pending (Week 2-3)

Description:
DiffusionRouter launches Feb 2026. Create skeleton for future implementation.

Files to Modify:
- src/providers/video/diffusion_router.rs (NEW, skeleton)

Content:
- Error return with "Launching Feb 2026" message
- Module structure in place
- Placeholder trait implementation

Acceptance Criteria:
- Code compiles
- Proper error message in docs
- Tests passing (compilation only)

Notes: Will be completed post-Q1 when API available
```

---

## Phase 5: Domain-Specific Models

### Task 5.1: BloombergGPT Documentation Framework
```
Title: phase/5.1-bloomberg-documentation: Document BloombergGPT alternatives
Effort: 0.5 days
Priority: MEDIUM
Assignee: Developer 2 (Voice/Domain)
Status: Pending (Week 2-3)

Description:
BloombergGPT is NOT publicly available. Skip implementation, document
framework for domain-specific models instead.

Files to Create:
- docs/domain_models.md (NEW)

Content:
- Finance domain: BloombergGPT status + alternatives (FinGPT, AdaptLLM)
- Framework for future domain specialization
- Example code for domain-specific model selection
- Enterprise partnership section

Acceptance Criteria:
- Documentation complete
- Alternatives researched and documented
- Example code working

Notes: 4 days saved by not implementing
```

### Task 5.2: Med-PaLM 2 Medical Integration
```
Title: phase/5.2-med-palm2: Add Med-PaLM 2 medical domain helper method
Effort: 1 day
Priority: MEDIUM
Assignee: Developer 2 (Voice/Domain)
Status: Pending (Week 3)

Description:
Med-PaLM 2 already available via Vertex AI. Add domain-specific helper method
and documentation for medical use cases.

Files to Modify:
- src/providers/chat/vertex.rs

Implementation:
```rust
pub fn for_medical_domain(...) -> Result<Self> {
    // Med-PaLM 2 by default
    // HIPAA compliance notes
}
```

Tests Required:
- Unit: helper method instantiation
- Integration: medical example queries

Acceptance Criteria:
- Helper method working
- Documentation updated
- HIPAA considerations documented
- Medical examples provided

Blockers: None
Notes: Minimal code change, documentation-heavy
```

### Task 5.3: ChatLAW Legal Provider
```
Title: phase/5.3-chatlaw: Integrate ChatLAW legal domain provider
Effort: 2 days (or 0.5 days documentation)
Priority: MEDIUM
Assignee: Developer 2 (Voice/Domain)
Status: Pending (Week 3, contingent on API access)

Description:
New provider for ChatLAW. Legal document analysis and reasoning.

Files to Modify:
- src/providers/chat/chatlaw.rs (NEW)

Contingency Decision Point: Week 1
- IF API available: Full implementation (2 days)
- IF delayed: Document as "pending API access" (0.5 days)

Tests Required (if full impl):
- Unit: legal document handling
- Integration: ChatLAW API

Blockers: ⚠️ API access (MEDIUM RISK)
Notes: Research required in Week 1 to confirm availability
```

### Task 5.4: Scientific Reasoning Benchmarks
```
Title: phase/5.4-scientific-benchmarks: Document scientific reasoning models
Effort: 1 day
Priority: MEDIUM
Assignee: Developer 2 (Voice/Domain)
Status: Pending (Week 3)

Description:
Leverage DeepSeek-R1 for scientific domain. Document benchmarks and
example usage.

Files to Create:
- docs/scientific_benchmarks.md (NEW)

Content:
- DeepSeek-R1 performance on AIME, Physics, Chemistry, CS
- Example scientific queries
- Benchmark comparison with other reasoning models
- Integration guide

Tests Required:
- Example code verification
- Benchmark documentation accuracy

Acceptance Criteria:
- Documentation complete
- Examples working
- Benchmarks documented

Notes: Helper method: `for_scientific_domain()` on DeepSeekProvider
```

---

## Checkpoint & Release Tasks

### Task: Week 2 Checkpoint
```
Title: week2-checkpoint: Verify progress on regional + voice + video
Effort: 0.5 days
Priority: HIGH
Assignee: Developer 1 (Lead)
Status: Pending (Week 2)

Description:
Mid-implementation checkpoint. Verify Phase 2, 3, 4 progress. Assess API
blocker status.

Acceptance Criteria:
- Mistral EU working ✅
- Deepgram v3 in progress or complete
- Runware video in progress or complete
- API decisions made (LightOn, LatamGPT, Grok)
- 634+ tests still passing ✅

Go/No-Go Decisions:
- LightOn: Continue or defer?
- LatamGPT: Full or skeleton?
- Grok: Continue or fallback?
- Time variance vs plan?

Action: Continue Week 3 as planned OR adjust based on blockers
```

### Task: Week 3 Checkpoint
```
Title: week3-checkpoint: Verify regional phase 2 + domain-specific progress
Effort: 0.5 days
Priority: HIGH
Assignee: Developer 1 (Lead)
Status: Pending (Week 3)

Description:
End-of-week 3 checkpoint. Feature freeze decision point.

Acceptance Criteria:
- Regional providers complete (3-4 of 4)
- Domain-specific models in progress (2+ of 4)
- Voice enhancements complete (2-3 of 3)
- Video generation complete (1 of 1)
- All tests passing ✅

Go/No-Go Decision:
- Ready for feature freeze in Week 4?
- Any critical issues?
- Performance acceptable?

Action: Proceed to Week 4 polish and release prep
```

### Task: Week 4 - Final Integration & Testing
```
Title: week4-integration: Run comprehensive integration tests
Effort: 2 days
Priority: HIGH
Assignee: All developers
Status: Pending (Week 4)

Description:
End-to-end testing of all new providers. Performance optimization.
Release readiness verification.

Acceptance Criteria:
- All 18 new providers integrated together
- Zero test failures (700+ tests)
- Build clean (no warnings)
- Performance benchmarks meet targets
- Documentation complete
- Examples working

Blockers: None (should be clean at this point)
```

### Task: Week 4-5 - Release Preparation
```
Title: release-v0.2.0: Prepare v0.2.0 release
Effort: 2 days
Priority: HIGH
Assignee: Developer 1 (Lead)
Status: Pending (Week 4)

Description:
Release preparation for v0.2.0 with 8-18 new providers.

Files to Modify:
- CHANGELOG.md
- PROVIDERS.md
- README.md (benchmark comparisons)
- docs/ (release notes)

Deliverables:
- Changelog entry (all changes)
- Migration guide (breaking changes)
- Performance benchmarks vs LiteLLM
- Blog post draft (implementation journey)
- Announcement text (Twitter, Reddit, HN)

Acceptance Criteria:
- All documentation updated
- Examples tested
- Benchmarks published
- Team reviewed and approved

Action: Ship v0.2.0
```

---

## Success Metrics (Exit Criteria)

### Required (Cannot Ship Without)
- ✅ 634+ tests passing (100% pass rate)
- ✅ Extended thinking: 4/4 providers
- ✅ Regional: 2/4 minimum (Mistral EU + Maritaca guaranteed)
- ✅ Voice: 2/3 minimum (Deepgram v3 + ElevenLabs guaranteed)
- ✅ Video: 1/1 (Runware)
- ✅ Domain: 2/4 minimum (Med-PaLM 2 + Scientific guaranteed)
- ✅ Build clean (no warnings)
- ✅ Zero breaking changes (or justified pre-1.0)

### Nice-to-Have (Defer if Needed)
- ⚠️ Grok Real-Time Voice (API may not be ready)
- ⚠️ LightOn France (API access uncertain)
- ⚠️ LatamGPT (API launch timing)
- ⚠️ ChatLAW (API access uncertain)

### Target Outcomes
- **Providers:** 52 → 60-70 new (best case 18, minimum 8)
- **Extended Thinking:** 2 → 4 (100%)
- **Real-Time Voice:** 1 → 2-3
- **Video:** 0 → 1 aggregator
- **Domain-Specific:** 0 → 2-4
- **Regional:** 6 → 7+ regions
- **Parity:** 153% → 175% vs LiteLLM
- **Tests:** 634 → 700+

---

## Task Status Legend

- **CRITICAL** - Must complete before proceeding
- **HIGH** - Important, affects multiple phases
- **MEDIUM** - Nice-to-have, can be deferred
- **Contingent** - Depends on external API access
- **Blocked** - Waiting for resolution
- **In Progress** - Currently being worked on
- **Done** - Completed and merged

---

**Created:** January 3, 2026
**Last Updated:** January 3, 2026
**For:** GitHub Projects board setup
**Reference:** `/home/yfedoseev/projects/llmkit/docs/plan_20260103.md`
