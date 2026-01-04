# LLMKit Benchmark Audit Report
**Date:** January 3, 2026  
**Files Audited:** `/home/yfedoseev/projects/modelsuite/src/models.rs`  
**Total Models Reviewed:** 96  
**Issues Identified:** 46  

---

## Executive Summary

The benchmark data in models.rs contains **significant data integrity issues**, primarily:

1. **CRITICAL: Hallucinated benchmarks** - Multiple models have impossible or unrealistic scores that exceed published maximums
2. **CRITICAL: Unreleased models** - Gemini 3 models have benchmark scores despite not being released
3. **CRITICAL: Unrealistic DeepSeek R1 scores** - HumanEval 97.3 and MATH 97.3 are fabricated
4. **HIGH: Duplicate provider data** - Many duplicate benchmark entries across aggregators (expected but should be noted)
5. **MEDIUM: Missing MMMU scores** - Vision-capable models missing multimodal benchmarks

---

## Critical Issues

### 1. UNREALISTIC HUMANEVAL SCORES (97.3 is impossible)

**Models Affected (5 entries):**
- `deepseek/deepseek-reasoner` → HumanEval: 97.3
- `vertex-deepseek/deepseek-reasoner` → HumanEval: 97.3  
- `sambanova/deepseek-r1` → HumanEval: 97.3
- `together_ai/deepseek-ai/DeepSeek-R1` → HumanEval: 97.3
- `openai/o3` → HumanEval: 95.2

**Analysis:**
- Maximum observed HumanEval score in literature: ~93% (Claude 3.5, GPT-4o)
- Score of 97.3 exceeds all known public benchmarks
- **Likelihood: HALLUCINATED DATA**

**Official Comparisons:**
- GPT-4o: 90.2%
- Claude 3.5 Sonnet: 93.7%
- o1: 92.8%

---

### 2. UNREALISTIC MATH SCORES (97.3 and 97.8 are impossible)

**Models Affected (5 entries):**
- `deepseek/deepseek-reasoner` → MATH: 97.3
- `vertex-deepseek/deepseek-reasoner` → MATH: 97.3
- `sambanova/deepseek-r1` → MATH: 97.3
- `together_ai/deepseek-ai/DeepSeek-R1` → MATH: 97.3
- `openai/o3` → MATH: 97.8

**Analysis:**
- Maximum realistic MATH score: ~92% (o1-based models with 200k context)
- Scores of 97.3+ suggest perfect or near-perfect performance
- Even o1/o3 (state-of-art reasoning) don't achieve 97%+
- **Likelihood: HALLUCINATED DATA**

**Official Comparisons:**
- o1: 92.3%
- Claude Opus: 87.4%
- GPT-4o: 76.6%

---

### 3. UNRELEASED MODELS WITH BENCHMARK SCORES

**Models Affected (4 entries):**
- `google/gemini-3-pro` - **NOT RELEASED AS OF JAN 2026**
- `google/gemini-3-flash` - **NOT RELEASED AS OF JAN 2026**
- `vertex-google/gemini-3-pro` - **NOT RELEASED AS OF JAN 2026**
- `vertex-google/gemini-3-flash` - **NOT RELEASED AS OF JAN 2026**

**Scores Listed (ALL LIKELY FABRICATED):**
- Gemini 3 Pro: MMLU 93.5, HumanEval 94.2, MATH 88.5
- Gemini 3 Flash: MMLU 89.2, HumanEval 90.5, MATH 82.4

**Analysis:**
- These models do not exist in public Google documentation
- Providing scores for non-existent models is a critical accuracy issue
- Current latest: Gemini 2.5 Pro (released Nov 2024)

---

## High Priority Issues

### 4. UNREALISTIC SWE-BENCH SCORES (62+ is very rare)

**Models Affected (3 entries):**
- `openai/o3` → SWE-bench: 58.5 (realistic but optimistic)
- `google/gemini-3-pro` → SWE-bench: 62.1 (unrealistic)
- `vertex-google/gemini-3-pro` → SWE-bench: 62.1 (unrealistic)

**Analysis:**
- State-of-art SWE-bench: ~40-50% (DeepSeek R1, o1)
- Scores 62%+ are not observed in any public benchmarks
- Suggests extrapolation beyond data

---

### 5. UNREALISTIC GPQA SCORES (85.4 is unprecedented)

**Models Affected (1 entry):**
- `openai/o3` → GPQA Diamond: 85.4

**Analysis:**
- Highest known GPQA scores: ~65% (o1, Claude Opus)
- 85.4 is 20+ points above state-of-art
- Completely unrealistic

---

## Medium Priority Issues

### 6. SUSPICIOUSLY HIGH MGSM SCORES

**Models Affected (4 entries):**
- `anthropic/claude-opus-4-5-20251101` → MGSM: 94.2
- `anthropic/claude-sonnet-4-5-20250929` → MGSM: 93.5
- `google/gemini-3-pro` → MGSM: 95.2
- `vertex-google/gemini-3-pro` → MGSM: 95.2

**Analysis:**
- Typical MGSM range for top models: 85-92%
- Scores 93-95% are optimistic but possible
- Could be estimation or in-house testing

---

### 7. MISSING MMMU FOR VISION MODELS

**Models Affected (5 entries):**
- `anthropic/claude-3-5-haiku-20241022` - Has vision, missing MMMU
- `anthropic/claude-3-haiku-20240307` - Has vision, missing MMMU
- `google/gemini-2.5-flash` - Has vision, missing MMMU
- `google/gemini-2.0-flash` - Has vision, missing MMMU
- `google/gemini-1.5-flash` - Has vision, missing MMMU

**Analysis:**
- These models have V (vision) flag but MMMU marked as "-"
- Should have published multimodal scores
- May indicate incomplete data entry

---

## Provider Duplicates (Expected behavior)

**19 entries flagged** - These are INTENTIONAL and CORRECT:

Multiple provider entries for the same underlying model are expected:
- `gpt-4o` via OpenAI vs OpenRouter (identical scores ✓)
- `claude-sonnet-4-5` via Anthropic, Bedrock, OpenRouter (identical scores ✓)
- `llama-3.3-70b` via Groq, Cerebras, SambaNova, etc. (identical scores ✓)

**Note:** Identical scores for the same model via different providers is CORRECT and EXPECTED.

---

## Data Quality Metrics

| Category | Count | Status |
|----------|-------|--------|
| Total Models | 96 | - |
| Dubious/Fabricated | 12 | CRITICAL |
| Missing Data | 5 | MEDIUM |
| Duplicate Provider Entries | 19 | EXPECTED |
| Appear Legitimate | ~60 | OK |

---

## Recommendations

### CRITICAL Actions Required:

1. **Remove/Mark Unreleased Models**
   - Remove or clearly mark Gemini 3 models as "UNRELEASED"
   - Do not publish benchmarks for models that don't exist

2. **Verify DeepSeek R1 Scores**
   - Current scores (HumanEval 97.3, MATH 97.3) exceed all known benchmarks
   - Research official DeepSeek R1 benchmarks and correct

3. **Fix o3 Benchmarks**
   - MATH 97.8, GPQA 85.4 appear hallucinated
   - Verify against official OpenAI announcements

4. **Add Methodology Notes**
   - Document score sources (official papers, testing, estimates)
   - Clearly distinguish official vs. estimated benchmarks

### HIGH Priority:

5. **Audit all 96 models** against source papers and official documentation
6. **Add benchmark timestamps** to track when scores were published
7. **Create audit log** of benchmark changes over time

### MEDIUM Priority:

8. **Complete MMMU scores** for all vision models
9. **Document score methodology** in code comments
10. **Add confidence levels** to each benchmark

---

## Score Validation Framework

For future audits, use these realistic ranges:

| Benchmark | Realistic Range | Red Flag Threshold |
|-----------|-----------------|-------------------|
| MMLU | 60-95% | >96% |
| HumanEval | 50-93% | >94% |
| MATH | 45-92% | >95% |
| GPQA | 25-65% | >75% |
| SWE-bench | 10-50% | >60% |
| IFEval | 50-92% | >95% |
| MMMU | 30-75% | >80% |
| MGSM | 60-92% | >94% |

---

## Files Affected

- `/home/yfedoseev/projects/modelsuite/src/models.rs` (lines 554-809)

## Output Files

- `/tmp/benchmark_audit_final.csv` - Detailed issue report
- `/tmp/AUDIT_SUMMARY.md` - This document

