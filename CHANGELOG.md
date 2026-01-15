# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2025-01-15

### Added

#### Rust Documentation & Quality Assurance
- **Comprehensive Rust documentation infrastructure**:
  - Documented 5 critical public functions with detailed examples and argument descriptions
  - Added dedicated CI/CD "Documentation" job with `RUSTDOCFLAGS="-D warnings"`
  - Added docs.rs badge to README for public documentation access
  - Pragmatic documentation strategy: public API fully documented, internal registries exempted

#### Python Enhancements (PEP 735 & Modern Tooling)
- **PEP 735 Dependency Groups support**:
  - Created `pdm.toml` with structured dependency groups (dev, docs)
  - Configured for modern Python package management
- **Expanded Python version support**:
  - Lowered `abi3` from `py39` to `py38` (supports Python 3.8+)
  - Updated classifiers to include Python 3.8
  - Updated pyproject.toml `requires-python` to `>=3.8`

- **Modern Python tooling configuration**:
  - Added `[tool.ty]` configuration for type checking with strict mode
  - Updated MyPy target from py39 to py38
  - Updated Ruff target from py39 to py38
  - Updated Black to support py38-py312 targets
  - All tools configured for Python 3.8 compatibility

#### Version Updates
- Rust core bumped to 0.1.3
- Python bindings bumped to 0.1.3 (with abi3-py38 for broad compatibility)
- Node.js bindings bumped to 0.1.3

### Technical Details

- **Rust Documentation Pattern**: Public types (Message, ContentBlock, CompletionRequest, etc.) are thoroughly documented; internal model registry exempted with allow(missing_docs)
- **CI/CD Enforcement**: Rustdoc warnings treated as errors, catching documentation issues early
- **Python ABI3 Compatibility**: abi3-py38 enables installation on Python 3.8-3.12 without version-specific wheels
- **Dependency Management**: PEP 735 provides better organization of optional and development dependencies

## [0.1.2] - 2025-01-13

### Added

#### Infrastructure & Developer Experience
- **Pre-commit hooks configuration** - Automated code quality checks for all three languages:
  - Rust: `cargo fmt`, `cargo clippy`, `cargo check`
  - Python: `black`, `ruff`, `mypy` with strict type checking
  - TypeScript/JavaScript: Biome unified formatter and linter
  - General: Trailing whitespace, line endings, YAML/TOML/JSON validation, spell checking
- **Biome configuration** - Unified TypeScript/JavaScript formatting and linting (single quotes, 2-space indent, 100-char line width)
- **Enhanced CONTRIBUTING.md** - Comprehensive guide for setting up pre-commit hooks, troubleshooting, and code quality standards

#### Documentation
- Pre-commit setup instructions for contributors
- Per-language code quality command examples
- Troubleshooting guide for common pre-commit issues
- Updated PR checklist to include code quality verification

### Fixed

#### Test Assertions (46 panic improvements across 30 provider files)
- **Core providers (5 files, 10 panics fixed)**:
  - `ollama.rs`: 2 panics - Text content and tool use assertions
  - `anthropic.rs`: 2 panics - Simple and structured system content validation
  - `openai.rs`: 3 panics - JsonObject and JsonSchema response format validation
  - `groq.rs`: 1 panic - Tool use content validation
  - `ai21.rs`: 2 panics - Text and tool use content blocks

- **Major providers (18 files, 31 panics fixed)**:
  - `cohere.rs`: 2 panics - Text and tool use content blocks
  - `huggingface.rs`: 2 panics - Text and tool use content blocks
  - `mistral.rs`: 2 panics - Tool use and text content
  - `replicate.rs`: 2 panics - Text content assertions
  - Single panic fixes in: `aleph_alpha.rs`, `nlp_cloud.rs`, `yandex.rs`, `clova.rs`, `writer.rs`, `maritaca.rs`, `watsonx.rs`, `cerebras.rs`, `cloudflare.rs`, `sambanova.rs`, `databricks.rs`, `fireworks.rs`, `openrouter.rs`, `azure.rs`

- **Advanced providers (2 files, 5 panics fixed)**:
  - `deepseek.rs`: 5 panics - Text content blocks and thinking content blocks
  - `openai_compatible.rs`: 2 panics - Text and tool use content blocks

- **Special APIs & Utilities (2 files, 5 panics fixed)**:
  - `runpod.rs`: 4 panics - Text content block assertions
  - `baseten.rs`: 4 panics - Text content block assertions
  - `openai_realtime.rs`: 3 panics - SessionCreated, Error, RateLimitUpdated validation
  - `streaming_multiplexer.rs`: 2 panics - Text delta and chunk reception

**Improvement**: All test panics now display actual values received when assertions fail, providing better debugging information instead of generic error messages. Pattern applied consistently: `panic!("Expected X, got {:?}", other)`

### Changed

- Version bumped to 0.1.2 across Rust, Python, and TypeScript packages
- All test assertions now follow consistent panic message patterns for improved debuggability
- Enhanced code quality standards with automated enforcement via pre-commit

### Technical Details

- **Panic pattern standardization**: Converted `if let ... else panic!("message")` to `match` statements with debug output
- **Pre-commit stages**: All checks configured for `pre-commit` stage (runs before commit)
- **Language-specific scoping**: TypeScript checks only run on `llmkit-node/` and `examples/nodejs/`
- **Biome configuration**:
  - Formatter: 2-space indent, 100-char line width, single quotes
  - Linter: Recommended rules enabled with correctness and style emphasis

## [0.1.1] - 2025-01-12

### Added

- Initial stable release with 100+ LLM provider support
- Rust core implementation with trait-based architecture
- Python bindings via Maturin (PyO3)
- TypeScript/Node.js bindings via NAPI-RS
- Support for multiple AI capabilities:
  - Text completion and streaming
  - Tool/function calling
  - Vision/image input
  - Audio synthesis and processing
  - Image generation
  - Video generation
  - Embeddings
  - Specialized APIs (OpenAI Realtime, etc.)
- Comprehensive feature set:
  - Request multiplexing
  - Circuit breaker pattern
  - Failover handling
  - Health checks
  - Metering and observability
  - Rate limiting and retry logic
  - Smart provider routing
  - Multi-tenancy support
- Complete documentation and examples for all three languages

## [0.1.0] - 2025-01-10

### Added

- Initial development release
- Core architecture and provider framework
- Basic functionality for major providers (OpenAI, Anthropic, etc.)
- Foundation for Python and TypeScript bindings
