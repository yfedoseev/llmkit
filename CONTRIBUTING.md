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

## Style Guidelines

### Rust

- Follow standard Rust formatting (`cargo fmt`)
- Pass clippy checks (`cargo clippy`)
- Use meaningful variable and function names
- Add doc comments for public APIs

### Python

- Follow PEP 8
- Use type hints
- Format with `black` or `ruff`

### TypeScript

- Follow the existing code style
- Use TypeScript types (avoid `any`)
- Format with `prettier`

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

## Pull Request Checklist

- [ ] Code follows the project's style guidelines
- [ ] Tests added/updated for changes
- [ ] All tests pass locally
- [ ] Documentation updated (if applicable)
- [ ] Commit messages follow conventions
- [ ] PR description explains the changes

## Questions?

If you have questions, feel free to:

- Open a [GitHub Discussion](https://github.com/yfedoseev/llmkit/discussions)
- Open an issue with the `question` label

## License

By contributing, you agree that your contributions will be licensed under the same [MIT OR Apache-2.0](LICENSE) license as the project.
