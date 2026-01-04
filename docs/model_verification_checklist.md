# Model Verification Checklist

Use this checklist when adding new models to the LLMKit registry. This prevents data quality issues and ensures consistency.

## Pre-Addition Review

- [ ] Model is officially released and publicly accessible
- [ ] Provider is active and maintaining API support
- [ ] Official documentation is available
- [ ] No conflicting models already in registry

## Provider & Pricing Verification

- [ ] Input pricing from official provider documentation
- [ ] Output pricing from official provider documentation
- [ ] Cache pricing included (if model has "C" capability)
- [ ] Pricing is in USD per 1M tokens
- [ ] Pricing checked within last 30 days
- [ ] Pricing matches other implementations of same model

## Context & Limits

- [ ] Context window from official documentation
- [ ] Max output limit from official documentation
- [ ] Max output limit is reasonable (typically 4K-32K)
- [ ] Context limits consistent across provider instances

## Capability Flags

Verify each capability against official documentation:

- [ ] **V (Vision)**: Model accepts image/video inputs
  - Only set if model documentation explicitly supports images
  - Research papers alone don't count

- [ ] **T (Tools)**: Model supports function calling
  - Check provider's function_calling support
  - Some models claim JSON output but not true tool use

- [ ] **J (JSON)**: Model supports JSON mode output
  - Set if model can output valid JSON
  - Doesn't require schema enforcement

- [ ] **S (Structured)**: Model enforces JSON schema
  - Only set if model validates against provided schema
  - NOT just "can output JSON"
  - Test in OpenAI Structured Outputs or equivalent

- [ ] **K (Extended Thinking)**: Model supports reasoning/thinking
  - Check for extended thinking/chain-of-thought support
  - Verify with provider documentation

- [ ] **C (Caching)**: Model/provider supports prompt caching
  - Check if provider supports cached prompts
  - Requires cache pricing (3rd price value)

## Benchmark Scores

### Realistic Ranges
- [ ] MMLU: 50-95% (most models 75-92%)
- [ ] HumanEval: 40-95% (most models 70-88%)
- [ ] MATH: 30-96% (most models 60-88%)
- [ ] GPQA: 30-75% (most models 45-70%)
- [ ] SWE-bench: 10-50% (most models 20-45%)
- [ ] IFEval: 50-95% (most models 75-92%)
- [ ] MMMU: 40-80% (most models 55-75%)
- [ ] MGSM: 60-92% (realistic max ~92%)

### Score Requirements
- [ ] Use "-" for unknown/unavailable scores
- [ ] Don't estimate or guess scores
- [ ] Use official benchmark results only
- [ ] If multiple sources exist, use most recent
- [ ] For multi-language benchmarks, use weighted average

### Performance Metrics (optional but helpful)
- [ ] TTFT (Time to First Token) in milliseconds
- [ ] TPS (Tokens Per Second) throughput
- [ ] Both are infrastructure-dependent

## Display & Naming

- [ ] Model ID follows `provider/model-name` format
- [ ] Aliases are short and memorable (if applicable)
- [ ] Display name is official model name or recognized alias
- [ ] Status is correct:
  - `C` = Current (actively maintained)
  - `L` = Legacy (older version, still supported)
  - `D` = Deprecated (no longer recommended)

## Data Format

- [ ] Model ID is unique (no duplicates)
- [ ] Alias is unique (no duplicates with other models)
- [ ] All 10 pipe-separated fields present
- [ ] Pricing has correct number of values (2 or 3)
- [ ] Context/output format: "context,output"
- [ ] No trailing spaces or special characters
- [ ] Special characters in descriptions properly escaped

## Description Quality

- [ ] Accurate and factual
- [ ] Concise (under 80 characters)
- [ ] Mentions key characteristics
- [ ] Notes any caveats or limitations
- [ ] Doesn't include pricing (that's in the data)
- [ ] Uses sentence case

## Testing Checklist

### Local Testing
- [ ] Run `cargo test --lib` - all tests pass
- [ ] Run `cargo fmt` - code formatting correct
- [ ] Run `cargo clippy --lib` - no warnings
- [ ] Build succeeds without errors: `cargo build --lib`

### Model Registry Tests
- [ ] Model appears in `get_model_info("id")` lookups
- [ ] Alias lookups work correctly
- [ ] Provider detection works
- [ ] Capability flags are readable in code

### API Testing (if possible)
- [ ] Can instantiate provider with model
- [ ] API call succeeds (even if API key not available)
- [ ] Pricing calculation works
- [ ] Provider routing works correctly

## Cross-Provider Verification

If model exists on multiple providers:

- [ ] Benchmarks are identical (or documented why they differ)
- [ ] Pricing differences are justified and documented
- [ ] Capability flags are consistent
- [ ] Context/output limits are consistent (or justified)
- [ ] Status is consistent across providers

## Documentation & Comments

- [ ] Add provider comment with relevant notes
- [ ] Document any special requirements
- [ ] Note infrastructure limitations
- [ ] Reference external documentation if unusual

## Git Commit

- [ ] Commit message describes what was added
- [ ] Include model name and provider
- [ ] Mention any special characteristics
- [ ] Keep to single model per commit if possible

Example:
```
feat: add Claude 3.5 Sonnet via OpenRouter

- Full capability support (VTJSKC)
- Cache pricing: 10% of input
- 200K context window
- Consistent pricing with direct API
```

## Common Mistakes to Avoid

❌ **Benchmark Hallucinations**
- Never make up benchmark scores
- Never extrapolate from one benchmark to another
- Use only official published scores

❌ **Capability Overstating**
- Don't set S flag for JSON-capable models
- Don't set K flag without explicit documentation
- Don't set V flag without official image support

❌ **Pricing Inconsistency**
- Don't forget cache pricing (3rd value) when C flag is set
- Don't use prices from wrong date/version
- Don't assume prices are same across regions

❌ **Duplicate Aliases**
- Check existing aliases before creating new ones
- Use `grep` to verify uniqueness
- Use provider prefix if needed for disambiguation

## Before Committing

Final checklist:
- [ ] All fields are filled in correctly
- [ ] No duplicate IDs or aliases
- [ ] All tests pass
- [ ] Benchmark scores are realistic
- [ ] Pricing is verified and current
- [ ] Capability flags are all documented
- [ ] Description is accurate and concise

## After Merging

- [ ] Monitor for user issues with new model
- [ ] Check if provider makes changes (pricing, capabilities)
- [ ] Update if model status changes (deprecated, etc.)
- [ ] Verify during quarterly audits

## Quarterly Audit Checklist

Every 3 months, verify all models:

- [ ] Pricing is still current (check provider website)
- [ ] Models still exist and are supported
- [ ] Capabilities haven't changed
- [ ] Benchmark scores are still accurate
- [ ] No new models missing from registry
- [ ] No deprecated models that should be marked "D"

## Questions?

Refer to:
1. `/home/yfedoseev/projects/llmkit/docs/audits/2026-01-03/` - Audit results
2. `src/models.rs` - Existing models as examples
3. Provider documentation - Official sources

## Version History

- **January 3, 2026** - Initial checklist created based on comprehensive audit
- Updated alongside model registry audit fixes
