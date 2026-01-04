# Pricing Fixes - Quick Reference Guide

**File:** `/home/yfedoseev/projects/llmkit/src/models.rs`
**Lines:** 554-663 (MODEL_DATA section)
**Total Changes:** 6 pricing entries
**Expected Impact:** Improves accuracy from 77% to 96%

---

## CRITICAL FIXES (Do Immediately)

### 1. DeepSeek R1 (Together AI) - Line 662

**Current:**
```
together_ai/deepseek-ai/DeepSeek-R1|deepseek-r1-together|DeepSeek R1 (Together)|C|0.55,2.19|...
```

**Update to:**
```
together_ai/deepseek-ai/DeepSeek-R1|deepseek-r1-together|DeepSeek R1 (Together)|C|3.0,7.0|...
```

**Details:**
- **Issue:** Using DeepSeek direct API pricing instead of Together AI markup
- **Current:** $0.55 / $2.19 per 1M tokens
- **Correct:** $3.00 / $7.00 per 1M tokens
- **Difference:** -$2.45 input (-82%), -$4.81 output (-69%)
- **Severity:** CRITICAL - Users see costs 5-7x lower than reality

---

## HIGH PRIORITY FIXES (Next 2 Weeks)

### 2. Gemini 3 Flash - Line 581

**Current:**
```
google/gemini-3-flash|gemini-3-flash|Gemini 3 Flash|C|0.1,0.4|...
```

**Update to:**
```
google/gemini-3-flash|gemini-3-flash|Gemini 3 Flash|C|0.5,3.0|...
```

**Details:**
- **Issue:** Copy-pasted from Gemini 2.0 Flash without verification
- **Current:** $0.10 / $0.40 per 1M tokens
- **Correct:** $0.50 / $3.00 per 1M tokens
- **Difference:** -$0.40 input (-80%), -$2.60 output (-87%)
- **Severity:** HIGH - Costs estimated 5-8x too low
- **Source:** [Google AI Pricing](https://ai.google.dev/gemini-api/docs/pricing)

---

### 3. Mistral Small 3.1 - Line 611

**Current:**
```
mistral/mistral-small-3.1|mistral-small-3.1|Mistral Small 3.1|C|0.05,0.15|...
```

**Update to:**
```
mistral/mistral-small-3.1|mistral-small-3.1|Mistral Small 3.1|C|0.1,0.3|...
```

**Details:**
- **Issue:** Exact 50% of official price (suspicious estimate)
- **Current:** $0.05 / $0.15 per 1M tokens
- **Correct:** $0.10 / $0.30 per 1M tokens
- **Difference:** -$0.05 input (-50%), -$0.15 output (-50%)
- **Severity:** MODERATE - Costs underestimated by exactly 50%
- **Source:** [Mistral AI Pricing](https://mistral.ai/pricing/)

---

### 4. Mistral Medium 3.1 (Output only) - Line 610

**Current:**
```
mistral/mistral-medium-3.1|mistral-medium-3.1|Mistral Medium 3.1|C|0.4,1.2|...
```

**Update to:**
```
mistral/mistral-medium-3.1|mistral-medium-3.1|Mistral Medium 3.1|C|0.4,2.0|...
```

**Details:**
- **Issue:** Output pricing incorrect; input pricing is correct
- **Current:** $0.40 / $1.20 per 1M tokens
- **Correct:** $0.40 / $2.00 per 1M tokens
- **Difference:** $0.00 input (OK), -$0.80 output (-40%)
- **Severity:** MODERATE - Output costs underestimated by 40%
- **Source:** [Mistral AI Pricing](https://mistral.ai/pricing/)

---

## MINOR FIXES (Should be Corrected)

### 5. Gemini 3 Pro (Output only) - Line 580

**Current:**
```
google/gemini-3-pro|gemini-3-pro|Gemini 3 Pro|C|2.0,10.0|...
```

**Update to:**
```
google/gemini-3-pro|gemini-3-pro|Gemini 3 Pro|C|2.0,12.0|...
```

**Details:**
- **Issue:** Output pricing low (input is correct)
- **Current:** $2.00 / $10.00 per 1M tokens
- **Correct:** $2.00 / $12.00 per 1M tokens
- **Difference:** $0.00 input (OK), -$2.00 output (-17%)
- **Severity:** MINOR - Output costs underestimated by 17%
- **Note:** Still in preview; may have context-dependent pricing
- **Source:** [Google AI Pricing](https://ai.google.dev/gemini-api/docs/pricing)

---

### 6. Gemini 2.5 Flash (Output only) - Line 583

**Current:**
```
google/gemini-2.5-flash|gemini-2.5-flash|Gemini 2.5 Flash|C|0.075,0.30|...
```

**Update to:**
```
google/gemini-2.5-flash|gemini-2.5-flash|Gemini 2.5 Flash|C|0.075,0.40|...
```

**Details:**
- **Issue:** Output pricing low; input is correct
- **Current:** $0.075 / $0.30 per 1M tokens
- **Correct:** $0.075 / $0.40 per 1M tokens
- **Difference:** $0.00 input (OK), -$0.10 output (-25%)
- **Severity:** MINOR - Output costs underestimated by 25%
- **Source:** [Google AI Pricing](https://ai.google.dev/gemini-api/docs/pricing)

---

## Implementation Checklist

### Pre-changes
- [ ] Review audit report: `PRICING_AUDIT_SUMMARY.md`
- [ ] Backup current version: `git stash`
- [ ] Verify you're on correct branch

### Making Changes
- [ ] Update line 580 (Gemini 3 Pro output: 10.0 -> 12.0)
- [ ] Update line 581 (Gemini 3 Flash: 0.1,0.4 -> 0.5,3.0)
- [ ] Update line 583 (Gemini 2.5 Flash output: 0.30 -> 0.40)
- [ ] Update line 610 (Mistral Medium 3.1 output: 1.2 -> 2.0)
- [ ] Update line 611 (Mistral Small 3.1: 0.05,0.15 -> 0.1,0.3)
- [ ] Update line 662 (DeepSeek R1 Together: 0.55,2.19 -> 3.0,7.0)

### Verification
- [ ] Run `cargo test` - all tests pass
- [ ] Run `cargo build` - builds without errors
- [ ] Run `cargo clippy` - no new warnings
- [ ] Review changes: `git diff src/models.rs`
- [ ] Verify line numbers are correct

### Commit
- [ ] Stage changes: `git add src/models.rs`
- [ ] Create commit with template message below
- [ ] Push to branch

---

## Commit Message Template

```
Fix pricing discrepancies in MODEL_DATA

Updated 6 pricing entries to match official provider APIs as of Jan 2026:

CRITICAL:
- DeepSeek R1 (Together AI): 0.55,2.19 -> 3.0,7.0 (5-7x too cheap)

HIGH PRIORITY:
- Gemini 3 Flash: 0.1,0.4 -> 0.5,3.0 (5-8x too cheap)
- Mistral Small 3.1: 0.05,0.15 -> 0.1,0.3 (50% too cheap)
- Mistral Medium 3.1 output: 1.2 -> 2.0 (40% too cheap)

MINOR:
- Gemini 3 Pro output: 10.0 -> 12.0 (17% too cheap)
- Gemini 2.5 Flash output: 0.30 -> 0.40 (25% too cheap)

Pricing verification against official APIs (Jan 2026):
- Before fix: 20/26 models correct (77% accuracy)
- After fix: Expected 25/26 models correct (96% accuracy)

Refs:
- pricing_audit_report.csv (detailed comparison table)
- PRICING_AUDIT_SUMMARY.md (full audit analysis)
```

---

## Testing Commands

```bash
# Navigate to project
cd /home/yfedoseev/projects/llmkit

# Run all tests
cargo test

# Run only model tests
cargo test models::tests

# Build to check for errors
cargo build

# Run clippy for warnings
cargo clippy

# See the diff before committing
git diff src/models.rs
```

---

## Key Facts

| Metric | Value |
|--------|-------|
| Total Models Audited | 26 |
| Currently Correct | 20 (76.9%) |
| After Fixes | 25+ (96%+) |
| Critical Issues | 1 |
| High Priority Issues | 3 |
| Minor Issues | 2 |
| Files to Update | 1 (`src/models.rs`) |
| Lines to Change | 6 |

---

## Source Data

All pricing verified from official provider documentation:

- [Anthropic Pricing](https://platform.claude.com/docs/en/about-claude/pricing)
- [OpenAI Pricing](https://openai.com/api/pricing/)
- [Google Gemini Pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [Mistral AI Pricing](https://mistral.ai/pricing/)
- [DeepSeek API Pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [Cohere Pricing](https://www.cohere.com/pricing)
- [Groq Pricing](https://groq.com/pricing/)
- [Together AI Pricing](https://www.together.ai/pricing)

---

**Audit Date:** January 3, 2026
**Status:** Ready for implementation
