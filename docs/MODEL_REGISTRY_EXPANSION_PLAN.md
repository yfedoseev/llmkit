# Model Registry Expansion Plan: 120 to 1,200+ Models

**Created:** January 4, 2026
**Version:** 1.0
**Status:** Ready for Execution
**Target Completion:** End of February 2026 (6 weeks)

---

## Executive Summary

This document provides a comprehensive implementation plan for expanding the ModelSuite model registry from **120 models to 1,200+ models** (10x growth), achieving 100% coverage across all major LLM providers.

### Current State
- **Registry Size:** 120 models (~1,200 lines in `models.rs`)
- **Providers Covered:** 27 providers
- **Format:** Pipe-delimited DSL embedded in Rust source

### Target State
- **Registry Size:** 1,200+ models (~10,000+ lines if monolithic)
- **Providers Covered:** 50+ providers
- **Format:** Modular data files with build-time aggregation
- **Key Addition:** Data quality flags, freshness tracking, update automation

### Gap Analysis
| Provider Category | Current | Target | Gap |
|-------------------|---------|--------|-----|
| OpenRouter Models | 6 | 353 | +347 |
| AWS Bedrock Models | 5 | 85 | +80 |
| Together AI Models | 2 | 200 | +198 |
| Groq Models | 3 | 15 | +12 |
| Vision/Multimodal | 15 | 150 | +135 |
| HuggingFace | 0 | 100+ | +100 |
| Regional (Qwen, Baidu, etc.) | 12 | 100+ | +88 |
| Specialized (Cerebras, OctoAI) | 8 | 60 | +52 |
| **Total** | **120** | **1,200+** | **+1,080** |

---

## 1. Architecture Design

### 1.1 Data Flow Architecture

```
                                 +-------------------+
                                 |  Provider APIs    |
                                 |  (OpenRouter,     |
                                 |   Bedrock, etc.)  |
                                 +--------+----------+
                                          |
                                          v
+-------------------+           +-------------------+
|  Manual Curation  |           |  Data Fetchers    |
|  (CSV/JSON files) +---------->+  (build scripts)  |
|  for quality data |           |                   |
+-------------------+           +--------+----------+
                                          |
                                          v
                               +-------------------+
                               |  Data Merger &    |
                               |  Validator        |
                               |  (Rust build.rs)  |
                               +--------+----------+
                                          |
                                          v
                               +-------------------+
                               |  models_data.rs   |
                               |  (generated)      |
                               +-------------------+
                                          |
                                          v
                               +-------------------+
                               |  models.rs        |
                               |  (parsing logic)  |
                               +-------------------+
```

### 1.2 Recommended File Structure

```
modelsuite/
├── src/
│   ├── models.rs                    # Core parsing logic + API (unchanged)
│   └── models_data.rs               # Generated at build time
├── data/
│   └── models/
│       ├── README.md                # Data format documentation
│       ├── schema.json              # JSON Schema for validation
│       ├── core/                    # Hand-curated high-quality data
│       │   ├── anthropic.csv
│       │   ├── openai.csv
│       │   ├── google.csv
│       │   └── mistral.csv
│       ├── aggregators/             # Data from API fetchers
│       │   ├── openrouter.csv
│       │   ├── together.csv
│       │   └── bedrock.csv
│       ├── regional/                # Regional providers
│       │   ├── alibaba.csv
│       │   ├── baidu.csv
│       │   ├── zhipu.csv
│       │   └── regional_asia.csv
│       ├── specialized/             # Niche/specialized
│       │   ├── cerebras.csv
│       │   ├── sambanova.csv
│       │   └── inference_platforms.csv
│       └── community/               # HuggingFace, dynamic
│           ├── huggingface_top100.csv
│           └── community_verified.csv
├── scripts/
│   ├── fetch_openrouter.py          # OpenRouter API fetcher
│   ├── fetch_bedrock.py             # Bedrock models fetcher
│   ├── fetch_together.py            # Together AI fetcher
│   ├── validate_models.py           # Validation script
│   ├── merge_models.py              # Merge all sources
│   └── generate_rust.py             # Generate models_data.rs
└── build.rs                          # Build-time integration
```

### 1.3 SOLID Principles Compliance

| Principle | Implementation |
|-----------|----------------|
| **Single Responsibility** | Separate files: parsing logic (`models.rs`), data (`models_data.rs`), fetchers (Python scripts) |
| **Open/Closed** | New providers added by creating new CSV files, no core code changes |
| **Liskov Substitution** | All model data conforms to same schema regardless of source |
| **Interface Segregation** | Different fetcher scripts for different providers; optional fields allowed |
| **Dependency Inversion** | Core logic depends on abstract schema, not concrete providers |

---

## 2. Data Schema Design

### 2.1 Enhanced Pipe-Delimited Format

Extend the current format to support data quality tracking:

```
# Current Format (10 fields):
id|alias|name|status|pricing|context|caps|benchmarks|description|classify

# Enhanced Format (13 fields):
id|alias|name|status|pricing|context|caps|benchmarks|description|classify|source|quality|updated
```

**New Fields:**
| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `source` | enum | Data origin | `official`, `openrouter`, `community`, `estimated` |
| `quality` | enum | Data confidence | `verified`, `partial`, `estimated`, `unknown` |
| `updated` | date | Last verification | `2026-01-04` |

### 2.2 CSV Format for Data Files

```csv
# data/models/core/anthropic.csv
id,alias,name,status,input_per_1m,output_per_1m,cache_per_1m,max_context,max_output,caps,mmlu,humaneval,math,gpqa,swe_bench,ifeval,mmmu,mgsm,ttft_ms,tps,description,can_classify,source,quality,updated
anthropic/claude-opus-4-5-20251101,claude-opus-4-5,Claude Opus 4.5,C,5.0,25.0,0.5,200000,32000,VTJSKC,92.3,95.8,87.4,68.7,55.2,92.1,71.5,91.5,1200,60,Premium model with maximum intelligence,N,official,verified,2026-01-04
```

### 2.3 Handling Incomplete Data

Models with incomplete data use special markers:

| Scenario | Handling |
|----------|----------|
| Missing pricing | Use `-1` for unknown, default to free tier estimate |
| Missing benchmarks | Use `-` (hyphen) as currently done |
| Missing context limits | Default to 128000/8192 with `estimated` quality |
| Deprecated models | Status `D`, kept for backward compatibility |
| Regional pricing variants | Multiple entries with region suffix |

### 2.4 Duplicate Model Handling

When same model available on multiple providers:

1. **Primary Entry:** Direct provider (e.g., `anthropic/claude-3-5-sonnet`)
2. **Secondary Entries:** Via aggregators with provider prefix:
   - `openrouter/anthropic/claude-3.5-sonnet`
   - `bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0`
   - `together_ai/anthropic/claude-3.5-sonnet`

**Registry Behavior:**
- All entries stored
- Pricing reflects actual provider pricing (may differ)
- `get_model_info()` returns first match; use full ID for specific provider

---

## 3. Phase Breakdown

### Phase 1: Critical Path (Week 1-2) - +443 Models
**Priority:** CRITICAL
**Effort:** 10 developer-days

#### Task 1.1: OpenRouter Integration (+353 models)
**Effort:** 5 days
**Source:** OpenRouter API (`https://openrouter.ai/api/v1/models`)

**Steps:**
1. Create `scripts/fetch_openrouter.py`:
   ```python
   import requests
   import csv

   def fetch_openrouter_models():
       resp = requests.get("https://openrouter.ai/api/v1/models")
       models = resp.json()["data"]

       for model in models:
           yield {
               "id": f"openrouter/{model['id']}",
               "alias": model['id'].split('/')[-1],
               "name": model['name'],
               "pricing": {
                   "input": model['pricing']['prompt'],
                   "output": model['pricing']['completion']
               },
               "context_length": model['context_length'],
               "capabilities": extract_caps(model),
               # ... map remaining fields
           }
   ```

2. Map OpenRouter fields to ModelSuite schema:
   | OpenRouter Field | ModelSuite Field |
   |-----------------|------------------|
   | `id` | `id` (prefixed with `openrouter/`) |
   | `name` | `name` |
   | `pricing.prompt` | `input_per_1m` (convert from per-token) |
   | `pricing.completion` | `output_per_1m` |
   | `context_length` | `max_context` |
   | `top_provider.max_completion_tokens` | `max_output` |
   | `architecture.modality` | capabilities flags |

3. Generate `data/models/aggregators/openrouter.csv`

4. Validation:
   - [ ] All 353 models fetched
   - [ ] Pricing converted correctly (OpenRouter uses per-token)
   - [ ] Capabilities extracted (vision, tools, etc.)
   - [ ] Duplicates with existing entries identified

**Acceptance Criteria:**
- [ ] `scripts/fetch_openrouter.py` runs without errors
- [ ] `data/models/aggregators/openrouter.csv` generated
- [ ] 353 unique models in output
- [ ] Schema validation passes
- [ ] No duplicate IDs with core models

---

#### Task 1.2: AWS Bedrock Integration (+80 models)
**Effort:** 3 days
**Source:** AWS Bedrock Documentation + boto3 API

**Steps:**
1. Create `scripts/fetch_bedrock.py`:
   ```python
   import boto3

   def fetch_bedrock_models(region='us-east-1'):
       client = boto3.client('bedrock', region_name=region)
       paginator = client.get_paginator('list_foundation_models')

       for page in paginator.paginate():
           for model in page['modelSummaries']:
               yield transform_bedrock_model(model)
   ```

2. Map Bedrock fields:
   | Bedrock Field | ModelSuite Field |
   |--------------|------------------|
   | `modelId` | `id` (prefixed with `bedrock/`) |
   | `modelName` | `name` |
   | `inputModalities` | capabilities (vision if 'IMAGE' in list) |
   | `outputModalities` | capabilities |
   | `inferenceTypesSupported` | streaming flag |
   | `customizationsSupported` | additional capabilities |

3. Pricing sourcing:
   - Use `aws pricing get-products` API
   - Or scrape from https://aws.amazon.com/bedrock/pricing/

4. Generate `data/models/aggregators/bedrock.csv`

**Acceptance Criteria:**
- [ ] All 80+ Bedrock models captured
- [ ] Nova, Titan, Claude, Llama families complete
- [ ] Pricing from official AWS sources
- [ ] Region-specific variants noted

---

#### Task 1.3: Latest Model Updates (+10 models)
**Effort:** 2 days
**Source:** Official provider documentation

**Models to Add/Update:**
| Model | Provider | Notes |
|-------|----------|-------|
| Claude 4.5 Opus | Anthropic | Already in registry, verify specs |
| Claude 4.5 Sonnet | Anthropic | Already in registry, verify specs |
| Claude 4.5 Haiku | Anthropic | Already in registry, verify specs |
| Gemini 3 Pro | Google | Add if not present |
| Gemini 3 Flash | Google | Add if not present |
| Llama 4 Scout | Meta/Together | New architecture, 17B active |
| Llama 4 Maverick | Meta/Together | New architecture, 17B active |
| o3-mini | OpenAI | Already present, verify |
| o3 | OpenAI | Already present, verify |
| GPT-4.1 | OpenAI | Already present, verify |

**Acceptance Criteria:**
- [ ] All latest models have verified specs
- [ ] Benchmarks from official sources
- [ ] Pricing current as of January 2026

---

### Phase 2: High Priority (Week 3-4) - +330 Models
**Priority:** HIGH
**Effort:** 10 developer-days

#### Task 2.1: Together AI Integration (+200 models)
**Effort:** 4 days
**Source:** Together AI API (`https://api.together.xyz/v1/models`)

**Steps:**
1. Create `scripts/fetch_together.py`
2. Map API response to schema
3. Handle Together's unique pricing tiers:
   - Turbo tier (fastest)
   - Standard tier
   - Serverless tier

**Key Models:**
- Llama family (all sizes)
- Mistral family
- DeepSeek variants
- Qwen variants
- Specialized models (Code, Math, etc.)

**Acceptance Criteria:**
- [ ] 200+ models fetched
- [ ] All pricing tiers captured
- [ ] Model families properly grouped

---

#### Task 2.2: Groq Expansion (+12 models)
**Effort:** 1 day
**Source:** Groq API + documentation

**Current:** 3 models (Llama 3.3 70B, Llama 3.1 8B, Mixtral)
**Target:** 15 models

**Additional Models:**
- Llama 3.2 variants (1B, 3B, 11B, 90B)
- Gemma 2 variants
- Whisper (audio)
- Tool-use optimized models

---

#### Task 2.3: Vision/Multimodal Gap (+120 models)
**Effort:** 5 days
**Source:** Multiple providers

**Models by Category:**

| Category | Models to Add | Source |
|----------|---------------|--------|
| Pixtral | Pixtral 12B, 124B | Mistral direct |
| Qwen VL | Qwen2-VL series (2B, 7B, 72B) | Alibaba, Together |
| Llama Vision | Llama 3.2 Vision (11B, 90B) | Together, Groq |
| LLaVA | LLaVA 1.5, 1.6, NeXT | HuggingFace, Replicate |
| InternVL | InternVL2 series | Together |
| Idefics | Idefics2, Idefics3 | HuggingFace |
| Fuyu | Fuyu-8B | Replicate |
| CogVLM | CogVLM2 | Together |

**Acceptance Criteria:**
- [ ] All major vision models covered
- [ ] Vision capability flag (`V`) correctly set
- [ ] Image input limits documented

---

### Phase 3: Medium Priority (Week 5-6) - +300 Models
**Priority:** MEDIUM
**Effort:** 10 developer-days

#### Task 3.1: HuggingFace Top Models (+100 models)
**Effort:** 4 days
**Source:** HuggingFace Hub API

**Selection Criteria:**
- Top 100 by downloads in past month
- Inference Endpoints available
- Commercial use license

**Script:**
```python
from huggingface_hub import HfApi

api = HfApi()
models = api.list_models(
    sort="downloads",
    direction=-1,
    limit=100,
    filter=["text-generation-inference"],
)
```

**Challenges:**
- Variable pricing (self-hosted vs Inference Endpoints)
- Incomplete capability documentation
- Rapid model additions

**Solution:**
- Mark as `source=community`, `quality=partial`
- Include disclaimer in docs

---

#### Task 3.2: Regional Provider Expansion (+100 models)
**Effort:** 4 days

**Providers:**
| Provider | Region | Models to Add |
|----------|--------|---------------|
| Alibaba/Qwen | China | Qwen2, Qwen2.5, QwQ, Qwen-VL (+30) |
| Baidu/ERNIE | China | ERNIE 4.0, 4.5, 3.5 variants (+20) |
| Zhipu/GLM | China | GLM-4, GLM-4V, ChatGLM series (+20) |
| Moonshot/Kimi | China | Kimi K1, K2 series (+10) |
| Stepfun | China | Step-1, Step-2 series (+10) |
| Rakuten | Japan | Rakuten AI models (+5) |
| Sarvam | India | Sarvam series (+5) |

---

#### Task 3.3: Specialized Inference Platforms (+100 models)
**Effort:** 2 days

**Providers:**
| Provider | Models to Add |
|----------|---------------|
| Cerebras | Extended Llama variants (+10) |
| OctoAI | Mixtral, Llama, SDXL models (+20) |
| Novita | Budget inference models (+30) |
| Deepinfra | Cost-optimized variants (+20) |
| Hyperbolic | Research models (+10) |
| Lambda Labs | GPU cloud models (+10) |

---

## 4. Data Collection Strategy

### 4.1 API Sources

| Provider | API Endpoint | Auth Required | Rate Limit |
|----------|--------------|---------------|------------|
| OpenRouter | `GET /api/v1/models` | No | 100/min |
| Bedrock | `list_foundation_models` | AWS creds | Unlimited |
| Together | `GET /v1/models` | API key | 60/min |
| Groq | `GET /openai/v1/models` | API key | 100/min |
| Replicate | `GET /v1/models` | API key | 100/min |
| HuggingFace | Hub API | Optional | 100/min |

### 4.2 Documentation Sources

| Provider | Documentation URL | Data Available |
|----------|-------------------|----------------|
| Anthropic | docs.anthropic.com/en/docs/models | Pricing, limits, benchmarks |
| OpenAI | platform.openai.com/docs/models | Full specs |
| Google | cloud.google.com/vertex-ai/docs | Full specs |
| Mistral | docs.mistral.ai | Full specs |
| DeepSeek | platform.deepseek.com/docs | Full specs |

### 4.3 Benchmark Sources

| Benchmark | Source | Update Frequency |
|-----------|--------|------------------|
| MMLU | Papers With Code | Monthly |
| HumanEval | OpenAI/GitHub | On release |
| MATH | Official papers | On release |
| GPQA | Anthropic | Quarterly |
| SWE-bench | Princeton | Monthly |

---

## 5. Validation Strategy

### 5.1 Data Validation Checklist

```python
# scripts/validate_models.py

def validate_model(model: dict) -> List[str]:
    errors = []

    # Required fields
    if not model.get('id'):
        errors.append("Missing id")
    if not model.get('name'):
        errors.append("Missing name")

    # ID format validation
    if '/' not in model['id']:
        errors.append("ID must include provider prefix")

    # Pricing sanity checks
    if model.get('input_per_1m', 0) < 0:
        errors.append("Invalid input pricing")
    if model.get('output_per_1m', 0) < model.get('input_per_1m', 0) * 0.5:
        errors.append("Output pricing suspiciously low")

    # Context limit sanity
    if model.get('max_context', 0) < 1024:
        errors.append("Context limit too small")
    if model.get('max_context', 0) > 10_000_000:
        errors.append("Context limit suspiciously large")

    # Benchmark range validation
    for bench in ['mmlu', 'humaneval', 'math', 'gpqa']:
        val = model.get(bench)
        if val and (val < 0 or val > 100):
            errors.append(f"Benchmark {bench} out of range")

    return errors
```

### 5.2 Duplicate Detection

```python
def detect_duplicates(models: List[dict]) -> List[Tuple[dict, dict]]:
    duplicates = []

    # Check by normalized name
    by_name = {}
    for model in models:
        normalized = normalize_model_name(model['name'])
        if normalized in by_name:
            duplicates.append((by_name[normalized], model))
        else:
            by_name[normalized] = model

    # Check by raw ID
    by_raw_id = {}
    for model in models:
        raw_id = model['id'].split('/')[-1]
        if raw_id in by_raw_id:
            duplicates.append((by_raw_id[raw_id], model))
        else:
            by_raw_id[raw_id] = model

    return duplicates
```

### 5.3 Quality Assurance Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Schema Compliance | 100% | Automated validation |
| Pricing Accuracy | 95%+ | Spot-check 50 random models |
| Capability Accuracy | 90%+ | Test 20 models per provider |
| Benchmark Currency | <30 days | Check publication dates |
| Duplicate Rate | <5% | Automated detection |

---

## 6. File Structure Details

### 6.1 Generated Code Structure

```rust
// src/models_data.rs (auto-generated)

//! Auto-generated model data - DO NOT EDIT
//! Generated: 2026-01-04T12:00:00Z
//! Sources: core/4, aggregators/3, regional/4, specialized/3, community/2

pub const MODEL_DATA: &str = r#"
# Core Providers (verified, official pricing)
# Source: data/models/core/*.csv
anthropic/claude-opus-4-5-20251101|claude-opus-4-5|Claude Opus 4.5|C|5.0,25.0,0.5|200000,32000|VTJSKC|92.3,95.8,87.4,68.7,55.2,92.1,71.5,91.5,1200,60|Premium model|N|official|verified|2026-01-04
# ... more core models ...

# Aggregator Providers (API-sourced, updated weekly)
# Source: data/models/aggregators/*.csv
openrouter/openai/gpt-4o|openrouter-gpt-4o|GPT-4o (OpenRouter)|C|2.5,10.0|128000,16384|VTJS|88.7,90.2,76.6,53.6,38.4,86.5,69.1,90.6,500,90|Via OpenRouter|Y|openrouter|verified|2026-01-03
# ... more aggregator models ...

# Regional Providers (curated, partial benchmarks)
# Source: data/models/regional/*.csv
alibaba/qwen-max|qwen-max|Qwen Max|C|1.26,6.30|32000,2048|TJ|82.5,80.2,70.5,52.1,28.5,80.5,-,85.5,600,120|Flagship reasoning|Y|official|partial|2026-01-02
# ... more regional models ...
"#;

/// Number of models in the registry
pub const MODEL_COUNT: usize = 1247;

/// Registry version (for cache invalidation)
pub const REGISTRY_VERSION: &str = "2026-01-04-v1";

/// Sources included in this build
pub const DATA_SOURCES: &[(&str, usize)] = &[
    ("core", 50),
    ("aggregators", 800),
    ("regional", 200),
    ("specialized", 150),
    ("community", 47),
];
```

### 6.2 Build Script Integration

```rust
// build.rs

use std::fs;
use std::path::Path;

fn main() {
    // Trigger rebuild if data files change
    println!("cargo:rerun-if-changed=data/models/");

    // Check if data generation is needed
    let data_dir = Path::new("data/models");
    if data_dir.exists() {
        generate_models_data();
    }
}

fn generate_models_data() {
    // Read all CSV files from data/models/
    // Merge and validate
    // Generate src/models_data.rs

    // For CI/CD, call the Python script
    let output = std::process::Command::new("python3")
        .args(["scripts/generate_rust.py"])
        .output()
        .expect("Failed to run generator");

    if !output.status.success() {
        panic!("Model generation failed: {:?}", output.stderr);
    }
}
```

---

## 7. Build and Test Plan

### 7.1 Continuous Integration

```yaml
# .github/workflows/model-registry.yml

name: Model Registry Validation

on:
  push:
    paths:
      - 'data/models/**'
      - 'scripts/fetch_*.py'
      - 'scripts/validate_models.py'
  schedule:
    - cron: '0 6 * * 1'  # Weekly Monday 6 AM

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Install dependencies
        run: pip install requests pydantic boto3

      - name: Validate existing data
        run: python scripts/validate_models.py

      - name: Schema validation
        run: python -m jsonschema -i data/models/schema.json data/models/**/*.csv

  fetch-and-update:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule'
    steps:
      - name: Fetch OpenRouter
        run: python scripts/fetch_openrouter.py

      - name: Fetch Together
        run: python scripts/fetch_together.py

      - name: Merge and generate
        run: python scripts/generate_rust.py

      - name: Create PR if changes
        uses: peter-evans/create-pull-request@v6
        with:
          title: "chore: Update model registry (automated)"
          branch: auto-update-models
```

### 7.2 Local Testing Commands

```bash
# Validate all model data
python scripts/validate_models.py

# Fetch latest from one provider
python scripts/fetch_openrouter.py --output data/models/aggregators/openrouter.csv

# Generate Rust source
python scripts/generate_rust.py

# Build and test
cargo build --all-features
cargo test --all-features

# Check registry stats
cargo run --example registry_stats
```

### 7.3 Performance Testing

```rust
// tests/registry_performance.rs

#[test]
fn test_registry_init_time() {
    let start = std::time::Instant::now();
    let _registry = get_all_models();
    let elapsed = start.elapsed();

    // Parsing 1,200 models should take <50ms
    assert!(elapsed.as_millis() < 50, "Registry init too slow: {:?}", elapsed);
}

#[test]
fn test_model_lookup_time() {
    // Warm up
    let _ = get_model_info("claude-3-5-sonnet");

    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = get_model_info("claude-3-5-sonnet");
    }
    let elapsed = start.elapsed();

    // 1000 lookups should take <1ms (HashMap O(1))
    assert!(elapsed.as_micros() < 1000, "Lookup too slow: {:?}", elapsed);
}

#[test]
fn test_memory_usage() {
    let models = get_all_models();
    let estimated_size = std::mem::size_of_val(models) * models.len();

    // ~1200 models should use <10MB
    assert!(estimated_size < 10 * 1024 * 1024);
}
```

---

## 8. Success Criteria

### 8.1 Quantitative Metrics

| Metric | Target | Verification |
|--------|--------|--------------|
| Total Models | 1,200+ | `get_all_models().len()` |
| OpenRouter Models | 350+ | Filter by provider |
| Bedrock Models | 80+ | Filter by provider |
| Together Models | 200+ | Filter by provider |
| Vision Models | 100+ | Filter by capability |
| Thinking Models | 15+ | Filter by capability |
| Build Time | <5 seconds | CI timing |
| Registry Init | <50ms | Benchmark test |
| Lookup Time | <1us | Benchmark test |
| Memory Usage | <10MB | Memory profiling |

### 8.2 Quality Metrics

| Metric | Target | Verification |
|--------|--------|--------------|
| Schema Compliance | 100% | Automated validation |
| Pricing Accuracy | 95%+ | Manual spot-check |
| Capability Accuracy | 90%+ | Integration tests |
| No Build Warnings | 0 | `cargo clippy` |
| Test Coverage | 100% new code | `cargo llvm-cov` |

### 8.3 Documentation Metrics

| Deliverable | Status |
|-------------|--------|
| Data format README | Required |
| Provider coverage table | Required |
| Update frequency docs | Required |
| Contribution guide | Required |
| API documentation | Update existing |

---

## 9. Risk Analysis

### 9.1 High Risk Items

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API rate limits | Medium | Medium | Implement caching, use API keys |
| Pricing changes | High | Low | Weekly automated updates |
| Provider API changes | Medium | Medium | Fallback to documentation |
| Build time increase | Low | Medium | Incremental generation |
| Memory bloat | Low | High | Lazy loading option |

### 9.2 Mitigation Strategies

**API Rate Limits:**
- Use GitHub Actions secrets for API keys
- Implement exponential backoff
- Cache responses for 24 hours

**Pricing Changes:**
- Mark all aggregator data with `updated` date
- Document "as of" date in generated code
- Weekly automated refresh

**Build Time:**
- Pre-generate `models_data.rs` in CI
- Include generated file in source (optional)
- Lazy parsing with `LazyLock`

---

## 10. Technical Debt Considerations

### 10.1 Current Debt in models.rs

```
[DEBT:architecture:LOW] Single file contains both data and logic
- Resolution: This plan addresses with separate data files

[DEBT:maintenance:MEDIUM] Embedded string data not version controlled separately
- Resolution: Move to CSV files in data/ directory

[DEBT:testing:LOW] No validation of model data correctness
- Resolution: Add validation scripts and CI checks
```

### 10.2 Debt Introduced by This Plan

```
[DEBT:performance:LOW] Generated code may increase binary size
- Mitigation: Compress string literals, lazy loading

[DEBT:maintenance:MEDIUM] External dependencies on provider APIs
- Mitigation: Fallback to cached data on API failure

[DEBT:complexity:LOW] Build script complexity increases
- Mitigation: Clear documentation, simple Python scripts
```

---

## 11. Timeline

### Week 1 (Jan 6-10)
| Day | Task | Owner |
|-----|------|-------|
| Mon | Infrastructure setup (file structure, scripts) | Lead |
| Tue | OpenRouter fetcher + initial data | Dev 1 |
| Wed | OpenRouter validation + merge | Dev 1 |
| Thu | Bedrock fetcher + initial data | Dev 2 |
| Fri | Bedrock validation + latest models update | Dev 2 |

### Week 2 (Jan 13-17)
| Day | Task | Owner |
|-----|------|-------|
| Mon | OpenRouter/Bedrock merge + build integration | Lead |
| Tue | CI/CD setup + automated validation | Lead |
| Wed | Phase 1 testing + documentation | All |
| Thu | Together AI fetcher | Dev 1 |
| Fri | Groq expansion + Together validation | Dev 1 |

### Week 3 (Jan 20-24)
| Day | Task | Owner |
|-----|------|-------|
| Mon | Vision models (Pixtral, Qwen VL) | Dev 1 |
| Tue | Vision models (LLaVA, InternVL) | Dev 1 |
| Wed | Vision models validation | Dev 1 |
| Thu | Phase 2 merge + testing | All |
| Fri | Phase 2 documentation | All |

### Week 4 (Jan 27-31)
| Day | Task | Owner |
|-----|------|-------|
| Mon | HuggingFace top 100 | Dev 2 |
| Tue | HuggingFace validation | Dev 2 |
| Wed | Alibaba/Qwen regional | Dev 1 |
| Thu | Baidu/Zhipu regional | Dev 1 |
| Fri | Regional validation + merge | All |

### Week 5 (Feb 3-7)
| Day | Task | Owner |
|-----|------|-------|
| Mon | Specialized providers (Cerebras, OctoAI) | Dev 2 |
| Tue | Specialized providers (Novita, Deepinfra) | Dev 2 |
| Wed | Phase 3 merge + testing | All |
| Thu | Performance optimization | Lead |
| Fri | Documentation updates | All |

### Week 6 (Feb 10-14)
| Day | Task | Owner |
|-----|------|-------|
| Mon | Final validation + bug fixes | All |
| Tue | Integration testing | All |
| Wed | Documentation finalization | All |
| Thu | Release preparation | Lead |
| Fri | Release v0.2.0 with expanded registry | Lead |

---

## 12. Summary

### Deliverables

1. **1,200+ models** in registry (10x current)
2. **Modular data architecture** in `data/models/`
3. **Automated fetchers** for major providers
4. **Validation pipeline** with CI/CD integration
5. **Weekly update automation** via GitHub Actions
6. **Comprehensive documentation** for data format and contribution

### Key Decisions

1. **Use CSV files** for human-readable data management
2. **Generate Rust code** at build time (not runtime parsing)
3. **Keep pipe-delimited format** in generated code for backward compatibility
4. **Add data quality fields** (source, quality, updated)
5. **Maintain duplicates** across providers with full IDs

### Next Steps

1. Review and approve this plan
2. Set up file structure as specified
3. Begin Phase 1 implementation (OpenRouter fetcher)
4. Establish weekly update cadence

---

**Document Version:** 1.0
**Last Updated:** January 4, 2026
**Author:** Claude Code AI
**Status:** READY FOR REVIEW
