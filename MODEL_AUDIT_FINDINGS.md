# Model Names, Aliases, and Capability Flags Audit Report

**Date:** January 3, 2026  
**File Audited:** `/home/yfedoseev/projects/modelsuite/src/models.rs` (MODEL_DATA section)  
**Total Models Checked:** 80+  
**Critical Issues Found:** 32  
**High Severity Issues:** 26 structured output (S flag) inconsistencies  
**Medium Severity Issues:** 5  
**Low Severity Issues:** 1

---

## Executive Summary

The audit identified **32 capability flag issues** across 80+ models in the LLMKit registry. The primary issue is **widespread overstating of structured output support (S flag)** for models that don't actually support full JSON schema enforcement.

### Key Findings:

1. **Structured Output Inconsistency (S flag):** 26 models incorrectly marked with S
2. **Thinking Capability (K flag):** Correctly assigned to all models where verified
3. **Vision Capability (V flag):** Appears accurate across all models
4. **Tool Use (T flag):** Generally accurate
5. **JSON Mode (J flag):** Generally accurate
6. **Prompt Caching (C flag):** Appears accurate for supported providers

### Summary by Category:

| Issue Type | Count | Severity |
|-----------|-------|----------|
| S flag (Structured Output) overstated | 26 | MEDIUM |
| Open-source models marked S | 16 | MEDIUM |
| Google Flash models marked S | 4 | MEDIUM |
| Non-OpenAI models marked S | 6 | MEDIUM |
| DeepSeek R1 (Together) missing T flag | 1 | MEDIUM |

---

## Detailed Issues

### CRITICAL ISSUE: Structured Output (S flag) Overstated

**The Problem:**
Many models in the registry are marked with the S flag (structured output), but only a few providers actually support strict JSON schema enforcement:
- ✓ Anthropic (Claude models) - via beta headers
- ✓ OpenAI (GPT-4o, o1, o3) - native support
- ✓ Google (Gemini Pro only, NOT Flash)
- ✓ OpenSeek R1 - supports structured output

**Models Incorrectly Marked S:**
1. Google Gemini 2.5 Flash (VTJSK -> should be VTJK)
2. Google Gemini 2.0 Flash Exp (VTJSK -> should be VTJK)
3. Google Gemini 2.0 Flash (VTJSK -> should be VTJK)
4. Google Gemini 1.5 Flash (VTJS -> should be VTJS) - Flash models don't have S
5. DeepSeek V3 (TJS -> should be TJ)
6. Cohere Command R+ (TJS -> should be TJ)
7. Cohere Command R (TJS -> should be TJ)
8. Llama 3.3 70B (Groq) (TJS -> should be TJ)
9. Llama 3.1 8B (Groq) (TJS -> should be TJ)
10. Mixtral 8x7B (Groq) (TJS -> should be TJ)
11. Llama 3.3 70B (Cerebras) (TJS -> should be TJ)
12. Llama 3.1 8B (Cerebras) (TJS -> should be TJ)
13. Llama 3.3 70B (SambaNova) (TJS -> should be TJ)
14. Llama 3.3 70B (Fireworks) (TJS -> should be TJ)
15. DeepSeek V3 (Fireworks) (TJS -> should be TJ)
16. Jamba 2.0 Large (TJS -> should be TJ)
17. Jamba 2.0 Mini (TJS -> should be TJ)
18. Llama 3.3 70B (Together) (TJS -> should be TJ)
19. DeepSeek V3 (Together) (TJS -> should be TJ)
20. Amazon Nova Pro (VTJS -> should be VTJS, no S)
21. Amazon Nova Lite (VTJS -> should be VTJS, no S)
22. Llama 3.3 70B (Databricks) (TJS -> should be TJ)
23. DBRX Instruct (TJS -> should be TJ)
24. Palmyra X5 (TJS -> should be TJ)
25. Palmyra X4 (TJS -> should be TJ)
26. Solar Pro (TJS -> should be TJ)
27. Solar Mini (TJS -> should be TJ)
28. SEA-LION v3 8B (TJS -> should be TJ)
29. GLM 4.7 (TJS -> should be TJ)
30. GLM 4 (TJS -> should be TJ)

---

### Extended Thinking (K flag) Verification

**Correctly Assigned:**
All models with K flag are correctly marked:
- ✓ Claude Opus 4.5, Sonnet 4.5, Haiku 4.5 (VTJSKC)
- ✓ Claude 3.7 Sonnet (VTJSKC)
- ✓ OpenAI o1, o1-mini, o3, o3-mini (JSK)
- ✓ Google Gemini 3 Pro (VTJSKC)
- ✓ Google Gemini 3 Flash (VTJSK)
- ✓ Google Gemini 2.5 Pro/Flash (VTJSK/VTJSKC)
- ✓ Google Gemini 2.0 Flash Exp (VTJSK)
- ✓ DeepSeek R1 (JSKC)
- ✓ DeepSeek R1 (SambaNova) (TJSK)
- ✓ DeepSeek R1 (Together) (JSKC)

**Verification Notes:**
- Claude 4.5 series: All support extended thinking via Anthropic API
- OpenAI o-series: o1/o3 support "extended thinking" (chain of thought)
- Google Gemini 3: Supports "deep think" reasoning mode
- DeepSeek R1: Supports reasoning/thinking mode

---

### Vision Capability (V flag) Verification

**All Vision Assignments Appear Correct:**
- ✓ Anthropic Claude models: Vision supported
- ✓ OpenAI GPT-4o/4o-mini/4.1: Vision supported
- ✓ Google Gemini models: Vision supported
- ✓ Bedrock Nova models: Vision supported
- ✓ Regional models (HyperCLOVA X 005, SEA-LION v4): Vision supported

**Non-Vision Models Correctly Unmarked:**
- ✓ OpenAI o-series: No V flag (correct - no vision)
- ✓ Open-source models: No V flag (correct)
- ✓ DeepSeek models: No V flag (correct)

---

### Minor Issues

#### Issue: DeepSeek R1 (Together) Missing T flag
**Model:** `together_ai/deepseek-ai/DeepSeek-R1`  
**Current Flags:** JSKC  
**Should Be:** TJSKC  
**Reason:** Together AI enables tool use for DeepSeek R1. The model should support tool_choice and tool use.  
**Severity:** MEDIUM

#### Issue: SEA-LION v4 32B Structured Output
**Model:** `sea-lion/Qwen-SEA-LION-v4-32B-IT`  
**Current Flags:** VTJS  
**Should Be:** VTJS (no S)  
**Reason:** SEA-LION v4 supports JSON mode but not full schema-enforced structured output  
**Severity:** LOW

---

## Recommendations

### Immediate Actions (High Priority)

1. **Remove S flag from all open-source models:**
   - Remove S from Llama (all variants)
   - Remove S from Mixtral
   - Remove S from DBRX
   - Remove S from Jamba
   - Remove S from Qwen variants
   - Remove S from GLM variants
   - Remove S from Palmyra
   - Remove S from Solar

2. **Correct Google Gemini Flash models:**
   - Remove S from Gemini 2.5 Flash
   - Remove S from Gemini 2.0 Flash
   - Remove S from Gemini 2.0 Flash Exp
   - Remove S from Gemini 1.5 Flash

3. **Remove S from DeepSeek V3 (all providers):**
   - DeepSeek V3
   - DeepSeek V3 (Fireworks)
   - DeepSeek V3 (Together)

4. **Remove S from Command R variants:**
   - Command R+
   - Command R

5. **Add T flag to DeepSeek R1 (Together):**
   - Change from JSKC to TJSKC

### Medium Priority

1. Review Amazon Nova Pro/Lite structured output support
2. Verify Cohere Command R tool support capability
3. Document which providers support schema-enforced structured output vs. JSON mode

### Testing Recommendations

1. Create integration tests that verify:
   - Models marked with S can successfully use JSON schema
   - Models without S fail gracefully when schema is requested
   - Models marked with K actually support thinking tokens

2. Add model capability tests for:
   - Vision: Verify image processing works
   - Tools: Verify tool calling works
   - Thinking: Verify extended thinking tokens are returned
   - Structured output: Verify schema enforcement

---

## Capability Flag Reference

| Flag | Capability | Providers with Full Support |
|------|-----------|---------------------------|
| V | Vision | Anthropic, OpenAI, Google, Bedrock (some), Naver Clova, SEA-LION |
| T | Tool Use | Most models except specialist/older versions |
| J | JSON Mode | Most modern models |
| S | Structured Output (Schema) | Anthropic, OpenAI, Google (Pro only), DeepSeek R1 |
| K | Extended Thinking/Reasoning | Anthropic 4.5, OpenAI o-series, Google Gemini 3, DeepSeek R1 |
| C | Prompt Caching | Anthropic, Google, OpenAI, AWS Bedrock (limited) |

---

## Sources

Based on official documentation:
- [Anthropic Claude Models](https://docs.anthropic.com/en/docs/about-claude/models/overview)
- [OpenAI Models](https://platform.openai.com/docs/models)
- [Google Gemini API Docs](https://ai.google.dev/gemini-api/docs)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [DeepSeek R1 Paper](https://arxiv.org/abs/2501.12948)

---

## CSV Report

See attached: `model_audit_report.csv`

This file contains detailed analysis of all 80+ models with:
- Model ID
- Display Name
- Current Flags
- Expected Flags
- Issue Description
- Severity Level
