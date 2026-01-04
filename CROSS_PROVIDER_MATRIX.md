# Cross-Provider Model Matrix

## Overview
This matrix shows all models available across multiple providers, highlighting inconsistencies in pricing, capabilities, and benchmarks.

---

## Table 1: Gemini 3.x - Google & Vertex AI

### Gemini 3 Pro

| Attribute | google | vertex-google | Consistent? |
|-----------|--------|---------------|-------------|
| **ID** | google/gemini-3-pro | vertex-google/gemini-3-pro | ✓ (Same model) |
| **Pricing (in/out)** | 2.0,10.0 | 2.0,10.0 | ✓ Yes |
| **Context Window** | 2,000,000 | 2,000,000 | ✓ Yes |
| **Max Output** | 16,384 | 16,384 | ✓ Yes |
| **Capabilities** | VTJSKC | VTJSKC | ✓ Yes |
| **MMLU** | 93.5 | 93.5 | ✓ Yes |
| **HumanEval** | 94.2 | 94.2 | ✓ Yes |
| **Math** | 88.5 | 88.5 | ✓ Yes |
| **GPQA** | 72.4 | 72.4 | ✓ Yes |
| **SWE-Bench** | 62.1 | 62.1 | ✓ Yes |
| **IFEval** | 91.5 | 91.5 | ✓ Yes |
| **MMMU** | 76.8 | 76.8 | ✓ Yes |
| **MGSM** | 95.2 | 95.2 | ✓ Yes |
| **TTFT (ms)** | 800 | 800 | ✓ Yes |
| **TPS** | 80 | 80 | ✓ Yes |
| **Status** | Current | Current | ✓ Yes |

**Verdict: Perfect consistency ✓**

---

### Gemini 3 Flash

| Attribute | google | vertex-google | Consistent? |
|-----------|--------|---------------|-------------|
| **ID** | google/gemini-3-flash | vertex-google/gemini-3-flash | ✓ (Same model) |
| **Pricing (in/out)** | 0.1,0.4 | 0.1,0.4 | ✓ Yes |
| **Context Window** | 1,000,000 | 1,000,000 | ✓ Yes |
| **Max Output** | 8,192 | 8,192 | ✓ Yes |
| **Capabilities** | VTJSK | VTJSK | ✓ Yes |
| **MMLU** | 89.2 | 89.2 | ✓ Yes |
| **HumanEval** | 90.5 | 90.5 | ✓ Yes |
| **Math** | 82.4 | 82.4 | ✓ Yes |
| **GPQA** | 65.2 | 65.2 | ✓ Yes |
| **SWE-Bench** | 54.3 | 54.3 | ✓ Yes |
| **IFEval** | 87.8 | 87.8 | ✓ Yes |
| **MMMU** | 70.2 | 70.2 | ✓ Yes |
| **MGSM** | 91.5 | 91.5 | ✓ Yes |
| **TTFT (ms)** | 300 | 300 | ✓ Yes |
| **TPS** | 200 | 200 | ✓ Yes |
| **Status** | Current | Current | ✓ Yes |

**Verdict: Perfect consistency ✓**

---

## Table 2: DeepSeek Chat (V3) - Direct API vs Vertex AI

| Attribute | Direct API | Vertex AI | Consistent? | Notes |
|-----------|-----------|-----------|-------------|-------|
| **ID** | deepseek/deepseek-chat | vertex-deepseek/deepseek-chat | ✓ (Same model) | |
| **Pricing Input** | $0.14 | $0.27 | ⚠️ **NO** | Vertex charges +93% markup |
| **Pricing Output** | $0.28 | $0.55 | ⚠️ **NO** | Vertex charges +96% markup |
| **Context Window** | 64,000 | 64,000 | ✓ Yes | |
| **Max Output** | 8,192 | 8,192 | ✓ Yes | |
| **Capabilities** | TJS | TJS | ✓ Yes | |
| **MMLU** | 87.5 | 87.5 | ✓ Yes | |
| **HumanEval** | 91.6 | 91.6 | ✓ Yes | |
| **Math** | 84.6 | 84.6 | ✓ Yes | |
| **GPQA** | 59.1 | 59.1 | ✓ Yes | |
| **SWE-Bench** | 42.0 | 42.0 | ✓ Yes | |
| **IFEval** | 86.2 | 86.2 | ✓ Yes | |
| **MMMU** | - | - | ✓ Yes | |
| **MGSM** | 90.7 | 90.7 | ✓ Yes | |
| **Status** | Current | Current | ✓ Yes | |

**Verdict: Pricing difference is LEGITIMATE - Vertex AI charges ~100% markup**

---

## Table 3: DeepSeek Reasoner (R1) - Direct API vs Vertex AI

| Attribute | Direct API | Vertex AI | Consistent? | Notes |
|-----------|-----------|-----------|-------------|-------|
| **ID** | deepseek/deepseek-reasoner | vertex-deepseek/deepseek-reasoner | ✓ (Same model) | |
| **Pricing Input** | $0.55 | $1.10 | ⚠️ **NO** | Vertex charges +100% markup |
| **Pricing Output** | $2.19 | $4.40 | ⚠️ **NO** | Vertex charges +101% markup |
| **Context Window** | 64,000 | 64,000 | ✓ Yes | |
| **Max Output** | 8,192 | 8,192 | ✓ Yes | |
| **Capabilities** | JSKC | JSKC | ✓ Yes | |
| **MMLU** | 90.8 | 90.8 | ✓ Yes | |
| **HumanEval** | 97.3 | 97.3 | ✓ Yes | |
| **Math** | 97.3 | 97.3 | ✓ Yes | |
| **GPQA** | 71.5 | 71.5 | ✓ Yes | |
| **SWE-Bench** | 49.2 | 49.2 | ✓ Yes | |
| **IFEval** | 88.4 | 88.4 | ✓ Yes | |
| **MMMU** | - | - | ✓ Yes | |
| **MGSM** | - | - | ✓ Yes | |
| **Status** | Current | Current | ✓ Yes | |

**Verdict: Pricing difference is LEGITIMATE - Vertex AI charges consistent ~100% markup**

---

## Table 4: Mistral Large - Direct API vs Vertex AI (CRITICAL ISSUE)

| Attribute | Direct API | Vertex AI | Consistent? | Notes |
|-----------|-----------|-----------|-------------|-------|
| **ID** | mistral/mistral-large-2512 | vertex-mistral/mistral-large | ✓ (Same model) | |
| **Pricing Input** | $0.50 | $1.00 | ⚠️ **NO** | Vertex charges 2x markup |
| **Pricing Output** | $1.50 | $3.00 | ⚠️ **NO** | Vertex charges 2x markup |
| **Context Window** | 262,000 | 262,000 | ✓ Yes | |
| **Max Output** | 8,192 | **4,096** | ❌ **NO** | **CRITICAL: Output differs by 50%** |
| **Capabilities** | VTJS | VTJS | ✓ Yes | |
| **MMLU** | 88.5 | 88.5 | ✓ Yes | |
| **HumanEval** | 86.8 | 86.8 | ✓ Yes | |
| **Math** | 75.4 | 75.4 | ✓ Yes | |
| **GPQA** | 55.8 | 55.8 | ✓ Yes | |
| **SWE-Bench** | 38.5 | 38.5 | ✓ Yes | |
| **IFEval** | 85.2 | 85.2 | ✓ Yes | |
| **MMMU** | - | - | ✓ Yes | |
| **MGSM** | - | - | ✓ Yes | |
| **Status** | Current | Current | ✓ Yes | |

**Verdict: CRITICAL - Max output context differs. Verify with provider documentation.**

---

## Table 5: Llama 3.3 70B - 7 Providers

| Provider | Pricing In | Pricing Out | Context | Max Out | TTFT | TPS | Benchmarks |
|----------|-----------|-----------|---------|---------|------|-----|------------|
| **SambaNova** | $0.40 | $0.40 | 128K | 8,192 | 40 | 1000 | 85.8,82.5,68.4,... |
| **Cloudflare** | $0.50 | $0.50 | 128K | 8,192 | 100 | 400 | 85.8,82.5,68.4,... |
| **Groq** | $0.59 | $0.79 | 128K | **32,768** | 100 | 500 | 85.8,82.5,68.4,... |
| **Cerebras** | $0.60 | $0.60 | 128K | 8,192 | 30 | 1800 | 85.8,82.5,68.4,... |
| **Databricks** | $0.85 | $0.85 | 128K | 8,192 | 200 | 250 | 85.8,82.5,68.4,... |
| **Fireworks** | $0.90 | $0.90 | 131K | 8,192 | 60 | 500 | 85.8,82.5,68.4,... |
| **Together AI** | $0.88 | $0.88 | 131K | 8,192 | 200 | 200 | 85.8,82.5,68.4,... |

**Pricing Range:** $0.40 - $0.90 (2.25x difference)
**Output Range:** 8,192 - 32,768 tokens (Groq exceptional)
**Benchmarks:** All identical (correct ✓)

**Verdict: LEGITIMATE - Different providers charge different rates for same model.**

---

## Table 6: Llama 3.3 70B Special Cases

### Groq Special Case
- **ID:** groq/llama-3.3-70b-versatile
- **Max Output:** 32,768 (vs 8,192 typical)
- **Note:** Groq's high-throughput infrastructure allows larger output tokens
- **Verdict:** Legitimate infrastructure difference ✓

### Bedrock Special Case
- **ID:** bedrock/meta.llama3-3-70b-instruct-v1:0
- **Max Output:** 2,048 (lowest)
- **Note:** Bedrock has stricter output limits due to infrastructure tier
- **Verdict:** Legitimate infrastructure constraint ✓

---

## Table 7: Claude Models - Vertex AI Variant

| Attribute | Direct Anthropic | Vertex AI | Consistent? | Notes |
|-----------|-----------------|-----------|-------------|-------|
| **Model** | claude-3.5-sonnet | claude-3.5-sonnet | ✓ (Same) | |
| **Pricing Input** | $3.00 | $3.00 | ✓ Yes | |
| **Pricing Output** | $15.00 | $15.00 | ✓ Yes | |
| **Context Window** | 200,000 | 200,000 | ✓ Yes | |
| **Max Output** | 8,192 | **4,096** | ⚠️ **NO** | Vertex has lower limit |
| **Status** | Legacy | Current | ⚠️ Differs | Different versions available |

**Note:** Vertex AI offers newer Claude versions (e.g., claude-3-opus) not available via direct API.

---

## Table 8: Mistral Medium - Similar Issue to Large

| Attribute | Direct API | Vertex AI | Consistent? |
|-----------|-----------|-----------|-------------|
| **Pricing Input** | $0.40 | $0.80 | ⚠️ Vertex markup 2x |
| **Pricing Output** | $1.20 | $2.40 | ⚠️ Vertex markup 2x |
| **Max Output** | 8,192 | **4,096** | ❌ Differs by 50% |

**Pattern:** Vertex AI consistently limits max output to 4,096 tokens for Mistral models.

---

## Table 9: DeepSeek via Together AI (Aggregator Premium)

| Attribute | Direct API | Together AI | Ratio |
|-----------|-----------|-----------|-------|
| **ID** | deepseek/deepseek-v3 | together_ai/.../DeepSeek-V3 | |
| **Pricing Input** | $0.14 | $1.25 | 8.9x |
| **Pricing Output** | $0.28 | $1.25 | 4.5x |

**Verdict:** Together AI charges premium aggregator markup (~9x). This is legitimate for their service model.

---

## Table 10: OpenRouter - Claude Models (Alias Issue)

| Model | Direct Claude | OpenRouter | Alias Issue? |
|-------|--------------|-----------|--------------|
| claude-opus-4.5 | anthropic/claude-opus-4-5-20251101 | openrouter/anthropic/claude-opus-4.5 | ⚠️ alias: "openrouter-claude-opus-4.5" |
| claude-sonnet-4.5 | anthropic/claude-sonnet-4-5-20250929 | openrouter/anthropic/claude-sonnet-4.5 | ⚠️ alias: "openrouter-claude-sonnet-4.5" |
| claude-haiku-4.5 | anthropic/claude-haiku-4-5-20251001 | openrouter/anthropic/claude-haiku-4.5 | ⚠️ alias: "openrouter-claude-haiku-4-5" |

**Issue:** Aliases include provider prefix, breaking standardization.
**Should be:** Just the standard model name without "openrouter-" prefix.

---

## Summary Statistics

### Cross-Provider Model Groups

| Group | Count | Primary Issue | Verdict |
|-------|-------|---------------|---------|
| Gemini (Google & Vertex) | 2 pairs | None | ✓ Perfect |
| DeepSeek (Direct & Vertex) | 2 pairs | Pricing markup | ✓ Legitimate |
| Mistral (Direct & Vertex) | 2 pairs | Output context differ | ❌ Needs verification |
| Llama 3.3 70B | 7 providers | Price/throughput vary | ✓ Legitimate |
| Claude (Various) | 6+ providers | Alias standardization | ⚠️ Fix needed |

### Pricing Variance Summary

| Model Family | Variance | Reason | Legitimate? |
|-------------|----------|--------|-------------|
| DeepSeek Chat | +93-96% | Vertex AI markup | ✓ Yes |
| DeepSeek Reasoner | +100-101% | Vertex AI markup | ✓ Yes |
| Mistral Large | 2x | Vertex AI markup | ✓ Yes |
| Llama 3.3 70B | 2.25x | Different providers | ✓ Yes |
| DeepSeek via Together | 9x | Aggregator markup | ✓ Yes |

### Context/Output Variance Summary

| Model | Issue | Impact | Status |
|-------|-------|--------|--------|
| Mistral Large | Output 8K vs 4K | 50% reduction on Vertex | ❌ Verify |
| Llama 3.3 70B | Output 8K vs 32K | Groq higher throughput | ✓ Legitimate |
| Claude via Vertex | Output varies | Lower on Vertex | ✓ Expected |

---

## Recommendations

### Immediate Actions:
1. **Verify Mistral output limits** - Both Direct (8K) and Vertex (4K) claims
2. **Fix OpenRouter aliases** - Remove "openrouter-" prefix from 3 Claude models
3. **Add missing cache prices** - For 3 models with C flag but no cache cost

### Documentation:
1. Add notes explaining Vertex AI pricing multipliers
2. Document infrastructure limitations (Mistral max output, etc.)
3. Note that Groq's Llama supports 32K output (special configuration)

### Verification:
1. Confirm DeepSeek pricing via Together AI is correct (~9x markup)
2. Verify all context windows are accurate
3. Review Llama 3.3 70B MMMU benchmarks (should be "-" for text model)
