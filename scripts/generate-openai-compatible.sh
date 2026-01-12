#!/bin/bash
# Generate OpenAI-compatible provider configuration boilerplate
#
# Usage: ./scripts/generate-openai-compatible.sh "Provider Name" "provider_id" "api_url"
#
# Example:
#   ./scripts/generate-openai-compatible.sh "xAI" "xai" "https://api.x.ai/v1"
#   ./scripts/generate-openai-compatible.sh "Meta Llama" "meta_llama" "https://api.meta.ai/v1"

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <provider_name> <provider_id> <api_url>"
    echo ""
    echo "Example:"
    echo "  $0 \"xAI\" \"xai\" \"https://api.x.ai/v1\""
    echo "  $0 \"Meta Llama\" \"meta_llama\" \"https://api.meta.ai/v1\""
    exit 1
fi

PROVIDER_NAME="$1"
PROVIDER_ID="$2"
API_URL="$3"

# Convert provider ID to uppercase for environment variable
PROVIDER_ID_UPPER=$(echo "$PROVIDER_ID" | tr '[:lower:]' '[:upper:]' | tr '-' '_')

echo "Generating boilerplate for: $PROVIDER_NAME ($PROVIDER_ID)"
echo "API URL: $API_URL"
echo ""

# Create the configuration structure
cat << EOF
# Configuration for $PROVIDER_NAME

## 1. Add to Cargo.toml features:
[features]
# In the appropriate tier section, add:
${PROVIDER_ID} = []

## 2. Add to all-providers feature:
all-providers = [
    # ... existing providers ...
    "${PROVIDER_ID}",
]

## 3. Add to src/client.rs builder method:
#[cfg(feature = "${PROVIDER_ID}")]
pub fn with_${PROVIDER_ID}(mut self, api_key: String) -> Self {
    self.providers.insert(
        "${PROVIDER_ID}",
        Provider::OpenAiCompatible(
            OpenAiCompatibleConfig {
                name: "${PROVIDER_NAME}",
                base_url: "${API_URL}".to_string(),
                api_key,
                default_model: "default-model".to_string(),
            }
        ),
    );
    self
}

#[cfg(feature = "${PROVIDER_ID}")]
pub fn with_${PROVIDER_ID}_from_env(mut self) -> Result<Self> {
    let api_key = env::var("${PROVIDER_ID_UPPER}_API_KEY")
        .context("${PROVIDER_ID_UPPER}_API_KEY not set")?;
    Ok(self.with_${PROVIDER_ID}(api_key))
}

## 4. Add model entries to src/models.rs:
# Add models from $PROVIDER_NAME to the MODEL_DATA:
# Format: id|alias|name|status|pricing|context|caps|benchmarks|description|classify

## 5. Create tests:
# Add to tests/mock_tests.rs in openai_compatible_tests module:
#[tokio::test]
async fn test_${PROVIDER_ID}_success() {
    openai_compatible_tests::test_openai_compatible_success("${PROVIDER_NAME}", "/v1/chat/completions").await;
}

#[tokio::test]
async fn test_${PROVIDER_ID}_streaming() {
    openai_compatible_tests::test_openai_compatible_streaming("${PROVIDER_NAME}").await;
}

#[tokio::test]
async fn test_${PROVIDER_ID}_rate_limit() {
    openai_compatible_tests::test_openai_compatible_rate_limit("${PROVIDER_NAME}").await;
}

## 6. Add to tests/integration_tests.rs (if feature-gated with API key):
#[tokio::test]
#[ignore]
async fn test_${PROVIDER_ID}_integration() {
    if !has_env("${PROVIDER_ID_UPPER}_API_KEY") {
        println!("Skipping ${PROVIDER_NAME} test (${PROVIDER_ID_UPPER}_API_KEY not set)");
        return;
    }

    let client = LLMKitClient::builder()
        .with_${PROVIDER_ID}_from_env()
        .expect("Failed to initialize ${PROVIDER_NAME}")
        .build();

    let request = CompletionRequest::new(
        "${PROVIDER_ID}",
        vec![ContentBlock::Text("Test message".to_string())],
    );

    let response = client.complete(&request).await;
    assert!(response.is_ok(), "Failed to get response from ${PROVIDER_NAME}");
}

## Implementation Checklist:
- [ ] Add feature flag to Cargo.toml
- [ ] Add provider config to openai_compatible.rs (if needed)
- [ ] Add builder methods to src/client.rs
- [ ] Add model entries to src/models.rs
- [ ] Create unit tests
- [ ] Create mock integration tests
- [ ] Create integration tests (feature-gated)
- [ ] Add documentation

EOF

echo ""
echo "✓ Configuration template generated"
echo "✓ Follow the checklist above to complete the implementation"
