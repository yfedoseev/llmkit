# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Currently supported versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

## Reporting a Vulnerability

We take the security of LLMKit seriously. If you believe you have found a security vulnerability, please report it to us as described below.

### Where to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please use GitHub's private vulnerability reporting:
- Go to https://github.com/yfedoseev/llmkit/security/advisories/new

### What to Include

Please include the following information in your report:

* Type of issue (e.g. credential exposure, injection, etc.)
* Full paths of source file(s) related to the manifestation of the issue
* The location of the affected source code (tag/branch/commit or direct URL)
* Any special configuration required to reproduce the issue
* Step-by-step instructions to reproduce the issue
* Proof-of-concept or exploit code (if possible)
* Impact of the issue, including how an attacker might exploit it

### What to Expect

* We will acknowledge your email within 48 hours
* We will send a more detailed response within 7 days indicating the next steps
* We will keep you informed about progress towards a fix
* We may ask for additional information or guidance
* Once fixed, we will publicly disclose the vulnerability (crediting you if desired)

## Security Considerations

LLMKit handles API keys and communicates with external LLM providers. This library:

* **Secure credential handling**: API keys are not logged or exposed in error messages
* **HTTPS enforced**: All provider communications use HTTPS
* **No unsafe code**: Core library avoids unsafe Rust code
* **Input validation**: Request parameters are validated before sending
* **Dependency auditing**: Regular security audits via `cargo audit`

### Best Practices

When using LLMKit:

1. **Environment variables**: Store API keys in environment variables, not in code
2. **Key rotation**: Rotate API keys regularly
3. **Least privilege**: Use API keys with minimal required permissions
4. **Monitor usage**: Track API usage for anomalies
5. **Update regularly**: Keep LLMKit updated with latest security patches
6. **Secure logging**: Ensure your application doesn't log sensitive request/response data

### Known Risks

1. **API key exposure**: If API keys are hardcoded or logged, they could be exposed
   - Mitigation: Use environment variables, never log keys

2. **Prompt injection**: User-provided content in prompts could manipulate LLM behavior
   - Mitigation: Validate and sanitize user inputs, use system prompts carefully

3. **Response handling**: LLM responses should be treated as untrusted
   - Mitigation: Validate and sanitize LLM outputs before use

## Disclosure Policy

When we receive a security bug report, we will:

1. Confirm the problem and determine affected versions
2. Audit code to find similar problems
3. Prepare fixes for all supported versions
4. Release patches as soon as possible

We ask security researchers to:

* Give us reasonable time to respond before public disclosure
* Make a good faith effort to avoid privacy violations and service disruption
* Not access or modify other users' data

## Comments on this Policy

If you have suggestions on how this process could be improved, please submit a pull request.
