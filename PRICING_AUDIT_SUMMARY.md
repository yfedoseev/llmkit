# LLMKit Pricing Audit Report
**Date:** January 3, 2026
**Location:** `/home/yfedoseev/projects/modelsuite/src/models.rs`
**Status:** 76.9% Accurate (20/26 models match official pricing)

---

## Executive Summary

This comprehensive audit compared all LLM pricing in the LLMKit registry against official provider APIs. Results show generally good accuracy with significant issues requiring immediate correction:

- **Total Models Audited:** 26
- **Correct Pricing:** 20 models (76.9%)
- **Issues Found:** 6 models (23.1%)
- **Critical Issues:** 1 (DeepSeek R1 on Together AI - 69-82% underpriced)
- **Moderate Issues:** 3 (Gemini 3 Flash, Mistral Medium/Small)
- **Minor Issues:** 2 (Gemini 3 Pro, Gemini 2.5 Flash)

---

## Critical Issues (IMMEDIATE ACTION REQUIRED)

### 1. DeepSeek R1 (Together AI) - CRITICAL PRICE MISMATCH
**Location:** Line 662 in `/home/yfedoseev/projects/modelsuite/src/models.rs`

```
Current:  together_ai/deepseek-ai/DeepSeek-R1|...|0.55,2.19|...
Should be: together_ai/deepseek-ai/DeepSeek-R1|...|3.00,7.00|...
```

**Issue Details:**
- Our Price: $0.55 input / $2.19 output per 1M tokens
- Official Price: $3.00 input / $7.00 output per 1M tokens
- Difference: -$2.45 input (-82%), -$4.81 output (-69%)
- Impact: Users see costs 5-7x lower than reality

**Root Cause:** Using DeepSeek's direct API pricing instead of Together AI's separate pricing tier with markup.

**Action:** Update to $3.00/$7.00 immediately before release

---

## High Priority Issues (UPDATE WITHIN 2 WEEKS)

### 2. Gemini 3 Flash - SIGNIFICANTLY UNDERPRICED
**Location:** Line 581 in `/home/yfedoseev/projects/modelsuite/src/models.rs`

```
Current:  google/gemini-3-flash|...|0.1,0.4|...
Should be: google/gemini-3-flash|...|0.5,3.0|...
```

**Issue Details:**
- Our Price: $0.10 input / $0.40 output
- Official Price: $0.50 input / $3.00 output
- Difference: -$0.40 input (-80%), -$2.60 output (-87%)
- Impact: Gemini 3 Flash costs are estimated 5-8x too low

**Root Cause:** Copied from Gemini 2.0 Flash pricing without verification. Suspicious that prices are identical despite being different models.

**Source:** [Google AI Pricing (2026)](https://ai.google.dev/gemini-api/docs/pricing)

---

### 3. Mistral Medium 3.1 - OUTPUT PRICE TOO LOW
**Location:** Line 610 in `/home/yfedoseev/projects/modelsuite/src/models.rs`

```
Current:  mistral/mistral-medium-3.1|...|0.4,1.2|...
Should be: mistral/mistral-medium-3.1|...|0.4,2.0|...
```

**Issue Details:**
- Our Price: $0.40 input / $1.20 output
- Official Price: $0.40 input / $2.00 output
- Difference: $0.00 input (correct), -$0.80 output (-40%)
- Impact: Output token costs underestimated by 40%

**Source:** [Mistral AI Pricing](https://mistral.ai/pricing/)

---

### 4. Mistral Small 3.1 - BOTH PRICES TOO LOW (50%)
**Location:** Line 611 in `/home/yfedoseev/projects/modelsuite/src/models.rs`

```
Current:  mistral/mistral-small-3.1|...|0.05,0.15|...
Should be: mistral/mistral-small-3.1|...|0.1,0.3|...
```

**Issue Details:**
- Our Price: $0.05 input / $0.15 output
- Official Price: $0.10 input / $0.30 output
- Difference: -$0.05 input (-50%), -$0.15 output (-50%)
- Impact: Total costs underestimated by exactly 50% (suspicious round number)

**Root Cause:** Likely using outdated Mistral Small 2.x pricing or estimates

**Source:** [Mistral AI Pricing](https://mistral.ai/pricing/)

---

## Minor Issues (SHOULD BE CORRECTED)

### 5. Gemini 3 Pro - OUTPUT PRICE SLIGHTLY LOW
**Location:** Line 580 in `/home/yfedoseev/projects/modelsuite/src/models.rs`

```
Current:  google/gemini-3-pro|...|2.0,10.0|...
Should be: google/gemini-3-pro|...|2.0,12.0|...
```

- Our Price: $2.00 input / $10.00 output
- Official Price: $2.00 input / $12.00 output
- Difference: $0.00 input (correct), -$2.00 output (-17%)
- Impact: Output costs underestimated by 17%

**Note:** Gemini 3 Pro is still in preview. This may have context-dependent pricing variations (≤200k vs >200k tokens).

**Source:** [Google AI Pricing](https://ai.google.dev/gemini-api/docs/pricing)

---

### 6. Gemini 2.5 Flash - OUTPUT PRICE SLIGHTLY LOW
**Location:** Line 583 in `/home/yfedoseev/projects/modelsuite/src/models.rs`

```
Current:  google/gemini-2.5-flash|...|0.075,0.30|...
Should be: google/gemini-2.5-flash|...|0.075,0.4|...
```

- Our Price: $0.075 input / $0.30 output
- Official Price: $0.075 input / $0.40 output
- Difference: $0.00 input (correct), -$0.10 output (-25%)
- Impact: Output costs underestimated by 25%

**Source:** [Google AI Pricing](https://ai.google.dev/gemini-api/docs/pricing)

---

## Models with CORRECT Pricing

### Perfect Matches (100% Accurate)

**Anthropic (3/3):**
- ✓ Claude Opus 4.5: $5.00 / $25.00
- ✓ Claude Sonnet 4.5: $3.00 / $15.00
- ✓ Claude Haiku 4.5: $1.00 / $5.00

**OpenAI (5/5):**
- ✓ GPT-4o: $2.50 / $10.00
- ✓ GPT-4o Mini: $0.15 / $0.60
- ✓ o1: $15.00 / $60.00
- ✓ o1-mini: $1.10 / $4.40
- ✓ o3: $10.00 / $40.00
- ✓ o3-mini: $1.10 / $4.40

**DeepSeek (2/2):**
- ✓ DeepSeek V3: $0.14 / $0.28
- ✓ DeepSeek R1: $0.55 / $2.19

**Cohere (2/2):**
- ✓ Command R+: $2.50 / $10.00
- ✓ Command R: $0.15 / $0.60

**Groq (1/1):**
- ✓ Llama 3.3 70B: $0.59 / $0.79

**Google (3/6):**
- ✓ Gemini 2.5 Pro: $1.25 / $10.00
- ✓ Gemini 2.0 Flash: $0.10 / $0.40
- ✓ Gemini 1.5 Pro: $1.25 / $5.00

**Mistral (1/3):**
- ✓ Mistral Large 3: $0.50 / $1.50

**Together AI (2/3):**
- ✓ Llama 3.3 70B: $0.88 / $0.88
- ✓ DeepSeek V3: $1.25 / $1.25

---

## Root Cause Analysis

### Pattern 1: Copy-Paste Without Verification
- **Evidence:** Gemini 3 Flash has identical pricing to Gemini 2.0 Flash ($0.10/$0.40)
- **Issue:** New models inherit old pricing without checking official sources
- **Prevention:** Require official source link for each price

### Pattern 2: Round Number Estimates
- **Evidence:** Mistral Small 3.1 is exactly 50% of official price (too convenient)
- **Issue:** Estimated prices used instead of verified official prices
- **Prevention:** Add verification step before adding new models

### Pattern 3: Provider-Specific Pricing Ignored
- **Evidence:** Together AI DeepSeek R1 using DeepSeek direct pricing
- **Issue:** Resellers have different pricing than direct APIs
- **Prevention:** Document which provider channel is being used

---

## Recommendations

### Immediate (This Week)
1. Update DeepSeek R1 (Together) to $3.00/$7.00 (CRITICAL)
2. Update Gemini 3 Flash to $0.50/$3.00
3. Update Mistral Small 3.1 to $0.10/$0.30

### Short-term (Next 2 Weeks)
4. Update Mistral Medium 3.1 output to $2.00
5. Update Gemini 3 Pro output to $12.00
6. Update Gemini 2.5 Flash output to $0.40

### Medium-term (Next Month)
7. Add pricing source documentation (e.g., comments with URLs)
8. Add last-verified date for each price
9. Document context-dependent pricing (e.g., Gemini >200k tokens)
10. Create pricing verification checklist

### Long-term
11. Set up automated pricing tests against official APIs
12. Add CI/CD checks for pricing freshness
13. Create pricing changelog in documentation
14. Subscribe to provider announcements for price changes

---

## Accuracy by Provider

| Provider | Accuracy | Models | Issues | Impact |
|----------|----------|--------|--------|--------|
| Anthropic | 100% (3/3) | All correct | None | Critical baseline ✓ |
| OpenAI | 100% (5/5) | All correct | None | Critical baseline ✓ |
| DeepSeek Direct | 100% (2/2) | All correct | None | Perfect ✓ |
| Cohere | 100% (2/2) | All correct | None | Perfect ✓ |
| Groq | 100% (1/1) | All correct | None | Perfect ✓ |
| Together AI | 67% (2/3) | 1 critical issue | DeepSeek R1 | URGENT |
| Google | 50% (3/6) | 3 issues | Gemini 3/2.5 | MODERATE |
| Mistral | 33% (1/3) | 2 issues | Small/Medium | MODERATE |

**Overall: 77% Accuracy**

---

## Data Sources

All pricing verified from official provider documentation:

1. **Anthropic:** https://platform.claude.com/docs/en/about-claude/pricing
2. **OpenAI:** https://openai.com/api/pricing/
3. **Google Gemini:** https://ai.google.dev/gemini-api/docs/pricing
4. **Mistral AI:** https://mistral.ai/pricing/
5. **DeepSeek:** https://api-docs.deepseek.com/quick_start/pricing
6. **Cohere:** https://www.cohere.com/pricing
7. **Groq:** https://groq.com/pricing/
8. **Together AI:** https://www.together.ai/pricing
9. **AWS Bedrock:** https://aws.amazon.com/bedrock/pricing/

---

## Files Generated

- `/home/yfedoseev/projects/modelsuite/pricing_audit_report.csv` - Detailed CSV report
- `/home/yfedoseev/projects/modelsuite/PRICING_AUDIT_SUMMARY.md` - This document

---

## Next Steps

1. Review critical issues with team
2. Update prices in `/home/yfedoseev/projects/modelsuite/src/models.rs` (lines 554-663)
3. Run tests to verify no regressions
4. Document pricing sources in code comments
5. Set up monthly verification schedule

---

**Audit Confidence:** HIGH
**Data Freshness:** Current as of January 3, 2026
**Verification Status:** All major providers checked against official 2026 pricing
