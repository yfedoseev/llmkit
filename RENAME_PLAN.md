# Rename Plan: ModelSuite → LLMKit

## Overview

Rename the project from `modelsuite` to `llmkit` across all packages, documentation, and infrastructure.

### Package Names by Registry

| Registry | Current | New |
|----------|---------|-----|
| crates.io | `modelsuite` | `llmkit` |
| PyPI | `modelsuite` | `llmkit` |
| npm | `modelsuite` | `@llmkit/llmkit` |

### Casing Convention

| Context | Current | New |
|---------|---------|-----|
| Package/crate name | `modelsuite` | `llmkit` |
| Struct/Class names | `ModelSuite*` | `LLMKit*` |
| Error types | `ModelSuiteError` | `LLMKitError` |
| Environment vars | `MODELSUITE_*` | `LLMKIT_*` |
| Documentation | ModelSuite | LLMKit |

### High-Impact Files (by reference count)

| File | Refs | Key Changes |
|------|------|-------------|
| `modelsuite-python/src/client.rs` | 119 | `PyModelSuiteClient` → `PyLLMKitClient` |
| `modelsuite-python/src/async_client.rs` | 107 | `PyAsyncModelSuiteClient` → `PyAsyncLLMKitClient` |
| `modelsuite-node/src/client.rs` | 94 | `JsModelSuiteClient` → `JsLLMKitClient` |
| `README.md` | 41 | Full rebrand |
| `modelsuite-python/src/errors.rs` | 33 | All `ModelSuiteError` types |
| `src/client.rs` | 26 | `ModelSuiteClient`, `ModelSuiteClientBuilder` |
| `tests/integration_tests.rs` | 25 | Import updates |
| `modelsuite-node/src/errors.rs` | 20 | Error conversion |

---

## Phase 1: Code Changes (Pre-publish)

### 1.1 Rust Core (`src/`)

- [ ] `Cargo.toml` - Change package name from `modelsuite` to `llmkit`
- [ ] `src/lib.rs` - Update module docs and crate-level documentation
- [ ] `src/client.rs` - Rename `ModelSuiteClient` → `LLMKitClient`
- [ ] `src/client.rs` - Rename `ModelSuiteClientBuilder` → `LLMKitClientBuilder`
- [ ] `src/error.rs` - Rename `ModelSuiteError` → `LLMKitError`
- [ ] `src/config.rs` - Update any `MODELSUITE_*` env var references
- [ ] All source files - Update doc comments mentioning "ModelSuite"

### 1.2 Python Bindings (`modelsuite-python/`) - 119+ references

- [ ] Rename directory `modelsuite-python/` → `llmkit-python/`
- [ ] `Cargo.toml` - Change package name to `llmkit-python`
- [ ] `pyproject.toml` - Change name to `llmkit`
- [ ] `src/lib.rs` - Update `#[pymodule]` name from `modelsuite` to `llmkit`

**Classes to rename in `src/client.rs` (119 refs):**
- [ ] `PyModelSuiteClient` → `PyLLMKitClient`
- [ ] `PyModelSuiteClientBuilder` → `PyLLMKitClientBuilder`
- [ ] All `use modelsuite::` imports → `use llmkit::`

**Classes to rename in `src/async_client.rs` (107 refs):**
- [ ] `PyAsyncModelSuiteClient` → `PyAsyncLLMKitClient`
- [ ] All `use modelsuite::` imports → `use llmkit::`

**Error types in `src/errors.rs` (33 refs):**
- [ ] `ModelSuiteError` base exception → `LLMKitError`
- [ ] `ModelSuiteAuthenticationError` → `LLMKitAuthenticationError`
- [ ] `ModelSuiteRateLimitError` → `LLMKitRateLimitError`
- [ ] `ModelSuiteInvalidRequestError` → `LLMKitInvalidRequestError`
- [ ] `ModelSuiteNetworkError` → `LLMKitNetworkError`
- [ ] `ModelSuiteTimeoutError` → `LLMKitTimeoutError`
- [ ] `ModelSuiteStreamError` → `LLMKitStreamError`

**Doc comments to update:**
- [ ] `src/audio/mod.rs` - "ModelSuite audio functionality"
- [ ] `src/video/mod.rs` - "ModelSuite video functionality"
- [ ] `src/embedding.rs` - "ModelSuite embedding functionality"
- [ ] `src/types/mod.rs` - "Type definitions for ModelSuite"
- [ ] `src/types/enums.rs` - "Python enum types for ModelSuite"

### 1.3 Node.js Bindings (`modelsuite-node/`) - 94+ references

- [ ] Rename directory `modelsuite-node/` → `llmkit-node/`
- [ ] `Cargo.toml` - Change package name to `llmkit-node`
- [ ] `package.json` - Change name to `@llmkit/llmkit`
- [ ] `index.d.ts` - Update TypeScript definitions

**Classes to rename in `src/client.rs` (94 refs):**
- [ ] `JsModelSuiteClient` → `JsLLMKitClient`
- [ ] `JsModelSuiteClientBuilder` → `JsLLMKitClientBuilder`
- [ ] All `use modelsuite::` imports → `use llmkit::`

**Error handling in `src/errors.rs` (20 refs):**
- [ ] `ModelSuiteError` enum matching → `LLMKitError`
- [ ] Error code constants (keep as-is, they're generic)
- [ ] Doc comment: "Converts ModelSuite errors" → "Converts LLMKit errors"

**Doc comments to update:**
- [ ] `src/audio.rs` - "ModelSuite audio functionality"
- [ ] `src/types/mod.rs` - "Type bindings for ModelSuite"
- [ ] `src/types/enums.rs` - "JavaScript enum types for ModelSuite"
- [ ] `src/types/embedding.rs` - "ModelSuite embedding functionality"

### 1.4 Workspace Configuration

- [ ] Root `Cargo.toml` - Update workspace members paths
- [ ] Update any workspace-level metadata

### 1.5 Documentation

- [ ] `README.md` - Full rewrite with LLMKit branding
- [ ] `docs/` - Update all documentation files
- [ ] Code examples in `examples/` directory
- [ ] API documentation comments

### 1.6 Tests

- [ ] `tests/` - Update test file references
- [ ] Integration tests - Update import paths
- [ ] Mock tests - Update class names

---

## Phase 2: Search & Replace Checklist

### Global replacements (case-sensitive):

```
modelsuite      → llmkit
ModelSuite      → LLMKit
MODELSUITE      → LLMKIT
model_suite     → llm_kit (if any)
model-suite     → llm-kit (if any)
```

### Files to check:

```bash
# Find all occurrences
rg -l "modelsuite|ModelSuite|MODELSUITE" --type rust --type python --type ts --type json --type toml --type md

# Count occurrences
rg -c "modelsuite|ModelSuite|MODELSUITE" --type rust --type python --type ts --type json --type toml --type md
```

---

## Phase 3: Git & GitHub Changes

### 3.1 Pre-rename commits

- [ ] Commit all code changes with message: `refactor: rename ModelSuite to LLMKit`
- [ ] Tag the last `modelsuite` version: `git tag modelsuite-final`
- [ ] Push all changes to current repo

### 3.2 GitHub Repository Rename

1. Go to GitHub repo Settings → General
2. Change repository name from `modelsuite` to `llmkit`
3. GitHub will automatically redirect old URLs

### 3.3 Update Git Remotes (after GitHub rename)

```bash
git remote set-url origin git@github.com:yfedoseev/llmkit.git
```

---

## Phase 4: Registry Publishing

### 4.1 crates.io

```bash
cd llmkit
cargo publish
```

### 4.2 PyPI

```bash
cd llmkit-python
maturin publish
```

### 4.3 npm

```bash
cd llmkit-node
npm publish --access public
```

---

## Phase 5: Local Directory Rename (LAST STEP)

```bash
# Only after all other changes are committed and pushed
mv ~/projects/modelsuite ~/projects/llmkit

# Update any IDE/editor workspace settings
# Update any shell aliases or scripts referencing the old path
```

---

## Verification Checklist

After rename, verify:

- [ ] `cargo build` succeeds
- [ ] `cargo test` passes (208+ tests)
- [ ] Python bindings compile: `cd llmkit-python && maturin develop`
- [ ] Node.js bindings compile: `cd llmkit-node && npm run build`
- [ ] All examples run correctly
- [ ] Documentation renders correctly
- [ ] No remaining references to "modelsuite" (case-insensitive search)

```bash
# Final verification - should return no results
rg -i "modelsuite" --type rust --type python --type ts --type json --type toml --type md
```

---

## Rollback Plan

If issues arise:

1. Git: `git revert` or `git reset` to pre-rename commit
2. GitHub: Rename repo back to `modelsuite` in settings
3. Directory: `mv ~/projects/llmkit ~/projects/modelsuite`

---

## Timeline Estimate

| Phase | Task | Estimate |
|-------|------|----------|
| 1 | Code changes | 2-3 hours |
| 2 | Search & verify | 30 min |
| 3 | Git/GitHub | 15 min |
| 4 | Publishing | 30 min |
| 5 | Directory rename | 5 min |

**Total: ~4 hours**

---

## Notes

- The npm scope `@llmkit` needs to be created first on npmjs.com
- Consider adding `modelsuite` as a keyword in package metadata for discoverability
- Old package names can have deprecation notices pointing to new names
