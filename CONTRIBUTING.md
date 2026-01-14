# Contributing to LLMKit

Thank you for your interest in contributing to LLMKit! This document provides guidelines and instructions for contributing.

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates.

When creating a bug report, include:

- **Clear title** describing the issue
- **Steps to reproduce** the behavior
- **Expected behavior** vs what actually happened
- **Environment details**:
  - LLMKit version
  - Language (Rust/Python/Node.js) and version
  - Operating system
  - Provider being used (if applicable)
- **Code samples** or minimal reproduction
- **Error messages** (full stack trace if available)

### Suggesting Features

Feature requests are welcome! Please include:

- **Clear description** of the feature
- **Use case** - why is this feature needed?
- **Proposed API** (if applicable)
- **Alternative solutions** you've considered

### Pull Requests

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** following our style guidelines
4. **Add tests** for new functionality
5. **Run tests** to ensure nothing is broken:
   ```bash
   cargo test
   cd llmkit-python && pytest
   cd llmkit-node && npm test
   ```
6. **Commit** with a clear message (see commit guidelines below)
7. **Push** and create a Pull Request

## Development Setup

### Prerequisites

- Rust 1.75+
- Python 3.9+ (with `uv` recommended)
- Node.js 18+ (with npm)
- API keys for providers you want to test

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yfedoseev/llmkit.git
cd llmkit

# Build Rust core
cargo build

# Build Python bindings
cd llmkit-python
maturin develop
cd ..

# Build Node.js bindings
cd llmkit-node
npm install
npm run build
cd ..
```

### Running Tests

```bash
# Rust tests
cargo test

# Python tests
cd llmkit-python
pytest

# Node.js tests
cd llmkit-node
npm test
```

### Running Examples

```bash
# Rust examples
cargo run --example 01_simple_completion

# Python examples
cd examples/python
uv run python 01_simple_completion.py

# Node.js examples
cd examples/nodejs
npx ts-node 01-simple-completion.ts
```

### Setting Up Pre-Commit Hooks

We use [pre-commit](https://pre-commit.com/) to automatically check code quality before commits:

```bash
# Install pre-commit framework
pip install pre-commit

# Install the git hook scripts
pre-commit install

# Run checks manually on all files
pre-commit run --all-files

# Run checks on specific language
pre-commit run rust-fmt --all-files      # Rust formatting
pre-commit run clippy --all-files        # Rust linting
pre-commit run black --all-files         # Python formatting
pre-commit run ruff --all-files          # Python linting
pre-commit run biome-ci --all-files      # TypeScript/JavaScript
```

After setup, code quality checks will run automatically before each commit. If checks fail, the commit is blocked and you must fix the issues.

## Style Guidelines

### Rust

- Follow standard Rust formatting (`cargo fmt`) - **enforced by pre-commit**
- Pass clippy checks (`cargo clippy`) - **enforced by pre-commit**
- Ensure code compiles with `cargo check --all` - **enforced by pre-commit**
- Use meaningful variable and function names
- Add doc comments for public APIs

### Python

- Follow PEP 8 (enforced by Black)
- Use type hints for all functions
- Format with Black and Ruff - **enforced by pre-commit**
- Type check with MyPy strict mode - **enforced by pre-commit**
- All checks run automatically before commit

### TypeScript/JavaScript

- Follow the existing code style
- Use TypeScript types (avoid `any`) with strict mode
- Format and lint with Biome - **enforced by pre-commit**
- Use single quotes for strings (Biome default)
- All checks run automatically before commit

### Git Commit Messages

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(providers): add support for new LLM provider
fix(streaming): handle connection timeout gracefully
docs(readme): update installation instructions
```

## Adding a New Provider

1. Create provider module in `src/providers/`
2. Implement required traits (`Provider`, `CompletionProvider`, etc.)
3. Add provider to `Provider` enum in `src/providers/mod.rs`
4. Add feature flag in `Cargo.toml`
5. Add environment variable detection in `ClientBuilder`
6. Write tests
7. Add documentation
8. Add example usage

See existing providers like `src/providers/openai.rs` as reference.

## Pre-Commit Troubleshooting

### Pre-commit hooks take a long time

This is normal, especially for:
- First-time setup (dependencies download)
- `cargo check` (builds the project)
- MyPy type checking (can be slower on large codebases)

Subsequent runs are faster due to caching.

### "pre-commit: command not found"

Install it with: `pip install pre-commit`

Then run: `pre-commit install`

### Skipping hooks

If you absolutely need to skip checks (not recommended):

```bash
git commit --no-verify
```

Note: CI/CD will still run checks on pull requests, so this will likely fail in review.

### Clippy returns warnings as errors

This is intentional to maintain code quality. Fix the warnings or discuss with maintainers.

### Specific checks fail repeatedly

Run individual checks to debug:

```bash
# Rust
cargo fmt --check
cargo clippy --all-targets --all-features

# Python
cd llmkit-python
black --check .
ruff check .
mypy llmkit --strict

# TypeScript
cd llmkit-node
npx @biomejs/biome check .
```

## Pull Request Checklist

- [ ] Pre-commit hooks installed and passing locally (`pre-commit run --all-files`)
- [ ] Code follows the project's style guidelines (enforced by pre-commit)
- [ ] Tests added/updated for changes
- [ ] All tests pass locally (`cargo test`, `pytest`, `npm test`)
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
- [ ] PR description explains the changes and motivation

## Questions?

If you have questions, feel free to:

- Open a [GitHub Discussion](https://github.com/yfedoseev/llmkit/discussions)
- Open an issue with the `question` label

## License

By contributing, you agree that your contributions will be licensed under the same [MIT OR Apache-2.0](LICENSE) license as the project.
