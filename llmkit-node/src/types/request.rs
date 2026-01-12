//! CompletionRequest, TokenCountRequest, and Batch types for JavaScript bindings

use llmkit::types::{
    BatchError, BatchJob, BatchRequest, BatchRequestCounts, BatchResult, BatchStatus,
    CompletionRequest, TokenCountRequest, TokenCountResult,
};
use napi_derive::napi;

use super::enums::JsThinkingEffort;
use super::message::{JsMessage, JsStructuredOutput, JsThinkingConfig};
use crate::tools::JsToolDefinition;

/// Request to complete a conversation.
///
/// Use the static factory method `create()` or the builder pattern.
///
/// @example
/// ```typescript
/// // Factory with model and messages
/// const request = CompletionRequest.create("claude-sonnet-4-20250514", [Message.user("Hello")])
///   .withSystem("You are helpful")
///   .withMaxTokens(1024)
///
/// // Or use builder pattern
/// const request = CompletionRequest.create("gpt-4o", [Message.user("Hi")])
///   .withTemperature(0.7)
///   .withStreaming()
/// ```
#[napi]
#[derive(Clone)]
pub struct JsCompletionRequest {
    pub(crate) inner: CompletionRequest,
}

#[napi]
impl JsCompletionRequest {
    /// Create a new completion request with model and messages.
    #[napi(factory)]
    pub fn create(model: String, messages: Vec<&JsMessage>) -> Self {
        Self {
            inner: CompletionRequest::new(
                model,
                messages.into_iter().map(|m| m.inner.clone()).collect(),
            ),
        }
    }

    // ==================== Builder Methods ====================
    // All builder methods return a new JsCompletionRequest for method chaining

    /// Builder method: Set the system prompt.
    #[napi]
    pub fn with_system(&self, system: String) -> Self {
        Self {
            inner: self.inner.clone().with_system(system),
        }
    }

    /// Builder method: Set max tokens.
    #[napi]
    pub fn with_max_tokens(&self, max_tokens: u32) -> Self {
        Self {
            inner: self.inner.clone().with_max_tokens(max_tokens),
        }
    }

    /// Builder method: Set temperature.
    #[napi]
    pub fn with_temperature(&self, temperature: f64) -> Self {
        Self {
            inner: self.inner.clone().with_temperature(temperature as f32),
        }
    }

    /// Builder method: Set top_p (nucleus sampling).
    #[napi]
    pub fn with_top_p(&self, top_p: f64) -> Self {
        Self {
            inner: self.inner.clone().with_top_p(top_p as f32),
        }
    }

    /// Builder method: Set tools.
    #[napi]
    pub fn with_tools(&self, tools: Vec<&JsToolDefinition>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_tools(tools.into_iter().map(|t| t.inner.clone()).collect()),
        }
    }

    /// Builder method: Set stop sequences.
    #[napi]
    pub fn with_stop_sequences(&self, stop_sequences: Vec<String>) -> Self {
        Self {
            inner: self.inner.clone().with_stop_sequences(stop_sequences),
        }
    }

    /// Builder method: Enable streaming.
    #[napi]
    pub fn with_streaming(&self) -> Self {
        Self {
            inner: self.inner.clone().with_streaming(),
        }
    }

    /// Builder method: Enable extended thinking with token budget.
    #[napi]
    pub fn with_thinking(&self, budget_tokens: u32) -> Self {
        Self {
            inner: self.inner.clone().with_thinking(budget_tokens),
        }
    }

    /// Builder method: Set thinking configuration.
    #[napi]
    pub fn with_thinking_config(&self, config: &JsThinkingConfig) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_thinking_config(config.inner.clone()),
        }
    }

    /// Builder method: Disable thinking/reasoning.
    ///
    /// Useful for getting faster, cheaper responses from reasoning models
    /// like Qwen3, DeepSeek-R1, or when using OpenRouter's reasoning control.
    #[napi]
    pub fn without_thinking(&self) -> Self {
        Self {
            inner: self.inner.clone().without_thinking(),
        }
    }

    /// Builder method: Set thinking effort level.
    ///
    /// Controls how much reasoning effort the model uses.
    /// Supported by OpenRouter and similar providers.
    #[napi]
    pub fn with_thinking_effort(&self, effort: JsThinkingEffort) -> Self {
        Self {
            inner: self.inner.clone().with_thinking_effort(effort.into()),
        }
    }

    /// Builder method: Set JSON schema for structured output.
    #[napi]
    pub fn with_json_schema(&self, name: String, schema: serde_json::Value) -> Self {
        Self {
            inner: self.inner.clone().with_json_schema(name, schema),
        }
    }

    /// Builder method: Set response format.
    #[napi]
    pub fn with_response_format(&self, format: &JsStructuredOutput) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_response_format(format.inner.clone()),
        }
    }

    /// Builder method: Enable JSON object output.
    #[napi]
    pub fn with_json_output(&self) -> Self {
        Self {
            inner: self.inner.clone().with_json_output(),
        }
    }

    /// Builder method: Set predicted output for speculative decoding.
    #[napi]
    pub fn with_prediction(&self, predicted_content: String) -> Self {
        Self {
            inner: self.inner.clone().with_prediction(predicted_content),
        }
    }

    /// Builder method: Enable system prompt caching (ephemeral).
    #[napi]
    pub fn with_system_caching(&self) -> Self {
        Self {
            inner: self.inner.clone().with_system_caching(),
        }
    }

    /// Builder method: Enable system prompt caching (extended, 1-hour TTL).
    #[napi]
    pub fn with_system_caching_extended(&self) -> Self {
        Self {
            inner: self.inner.clone().with_system_caching_extended(),
        }
    }

    /// Builder method: Enable extended output (128k tokens, Anthropic beta).
    #[napi]
    pub fn with_extended_output(&self) -> Self {
        Self {
            inner: self.inner.clone().with_extended_output(),
        }
    }

    /// Builder method: Enable interleaved thinking (Anthropic beta).
    #[napi]
    pub fn with_interleaved_thinking(&self) -> Self {
        Self {
            inner: self.inner.clone().with_interleaved_thinking(),
        }
    }

    /// Builder method: Set extra provider-specific options.
    #[napi]
    pub fn with_extra(&self, extra: serde_json::Value) -> Self {
        Self {
            inner: self.inner.clone().with_extra(extra),
        }
    }

    // ==================== Properties ====================

    /// The model identifier.
    #[napi(getter)]
    pub fn model(&self) -> String {
        self.inner.model.clone()
    }

    /// The conversation messages.
    #[napi(getter)]
    pub fn messages(&self) -> Vec<JsMessage> {
        self.inner
            .messages
            .iter()
            .cloned()
            .map(JsMessage::from)
            .collect()
    }

    /// The system prompt (if set).
    #[napi(getter)]
    pub fn system(&self) -> Option<String> {
        self.inner.system.clone()
    }

    /// Maximum tokens to generate.
    #[napi(getter)]
    pub fn max_tokens(&self) -> Option<u32> {
        self.inner.max_tokens
    }

    /// Sampling temperature.
    #[napi(getter)]
    pub fn temperature(&self) -> Option<f64> {
        self.inner.temperature.map(|t| t as f64)
    }

    /// Whether streaming is enabled.
    #[napi(getter)]
    pub fn stream(&self) -> bool {
        self.inner.stream
    }

    // ==================== Helper Methods ====================

    /// Check if the request has caching enabled.
    #[napi]
    pub fn has_caching(&self) -> bool {
        self.inner.has_caching()
    }

    /// Check if the request has thinking enabled.
    #[napi]
    pub fn has_thinking(&self) -> bool {
        self.inner.has_thinking()
    }

    /// Check if the request has structured output.
    #[napi]
    pub fn has_structured_output(&self) -> bool {
        self.inner.has_structured_output()
    }
}

impl From<CompletionRequest> for JsCompletionRequest {
    fn from(request: CompletionRequest) -> Self {
        Self { inner: request }
    }
}

impl From<JsCompletionRequest> for CompletionRequest {
    fn from(js_request: JsCompletionRequest) -> Self {
        js_request.inner
    }
}

// ============================================================================
// Token Counting Types
// ============================================================================

/// Request to count tokens for a model.
///
/// This allows estimation of token counts before making a completion request,
/// useful for cost estimation and context window management.
///
/// @example
/// ```typescript
/// // Create from model and messages
/// const request = TokenCountRequest.create(
///   "claude-sonnet-4-20250514",
///   [Message.user("Hello, how are you?")]
/// ).withSystem("You are a helpful assistant")
///
/// // Count tokens
/// const result = await client.countTokens(request)
/// console.log(`Input tokens: ${result.inputTokens}`)
///
/// // Or create from existing completion request
/// const request = TokenCountRequest.fromCompletionRequest(completionRequest)
/// ```
#[napi]
#[derive(Clone)]
pub struct JsTokenCountRequest {
    pub(crate) inner: TokenCountRequest,
}

#[napi]
impl JsTokenCountRequest {
    /// Create a new token count request with model and messages.
    #[napi(factory)]
    pub fn create(model: String, messages: Vec<&JsMessage>) -> Self {
        Self {
            inner: TokenCountRequest::new(
                model,
                messages.into_iter().map(|m| m.inner.clone()).collect(),
            ),
        }
    }

    /// Create a token count request from an existing completion request.
    #[napi(factory)]
    pub fn from_completion_request(request: &JsCompletionRequest) -> Self {
        Self {
            inner: TokenCountRequest::from_completion_request(&request.inner),
        }
    }

    /// Builder method: Set the system prompt.
    #[napi]
    pub fn with_system(&self, system: String) -> Self {
        Self {
            inner: self.inner.clone().with_system(system),
        }
    }

    /// Builder method: Set the tools.
    #[napi]
    pub fn with_tools(&self, tools: Vec<&JsToolDefinition>) -> Self {
        Self {
            inner: self
                .inner
                .clone()
                .with_tools(tools.into_iter().map(|t| t.inner.clone()).collect()),
        }
    }

    /// The model identifier.
    #[napi(getter)]
    pub fn model(&self) -> String {
        self.inner.model.clone()
    }

    /// The conversation messages.
    #[napi(getter)]
    pub fn messages(&self) -> Vec<JsMessage> {
        self.inner
            .messages
            .iter()
            .cloned()
            .map(JsMessage::from)
            .collect()
    }

    /// The system prompt (if set).
    #[napi(getter)]
    pub fn system(&self) -> Option<String> {
        self.inner.system.clone()
    }
}

impl From<TokenCountRequest> for JsTokenCountRequest {
    fn from(request: TokenCountRequest) -> Self {
        Self { inner: request }
    }
}

impl From<JsTokenCountRequest> for TokenCountRequest {
    fn from(js_request: JsTokenCountRequest) -> Self {
        js_request.inner
    }
}

// ============================================================================
// Batch Processing Types
// ============================================================================

use super::enums::JsBatchStatus;

/// A request within a batch.
///
/// @example
/// ```typescript
/// const batchRequests = [
///   BatchRequest.create("request-1", completionRequest1),
///   BatchRequest.create("request-2", completionRequest2),
/// ]
/// const batchJob = await client.createBatch(batchRequests)
/// ```
#[napi]
#[derive(Clone)]
pub struct JsBatchRequest {
    pub(crate) inner: BatchRequest,
}

#[napi]
impl JsBatchRequest {
    /// Create a new batch request with a custom ID and completion request.
    #[napi(factory)]
    pub fn create(custom_id: String, request: &JsCompletionRequest) -> Self {
        Self {
            inner: BatchRequest::new(custom_id, request.inner.clone()),
        }
    }

    /// The custom ID for this request.
    #[napi(getter)]
    pub fn custom_id(&self) -> String {
        self.inner.custom_id.clone()
    }
}

impl From<BatchRequest> for JsBatchRequest {
    fn from(request: BatchRequest) -> Self {
        Self { inner: request }
    }
}

impl From<JsBatchRequest> for BatchRequest {
    fn from(js_request: JsBatchRequest) -> Self {
        js_request.inner
    }
}

/// Request counts for a batch job.
#[napi]
#[derive(Clone)]
pub struct JsBatchRequestCounts {
    inner: BatchRequestCounts,
}

#[napi]
impl JsBatchRequestCounts {
    /// Total number of requests in the batch.
    #[napi(getter)]
    pub fn total(&self) -> u32 {
        self.inner.total
    }

    /// Number of successfully completed requests.
    #[napi(getter)]
    pub fn succeeded(&self) -> u32 {
        self.inner.succeeded
    }

    /// Number of failed requests.
    #[napi(getter)]
    pub fn failed(&self) -> u32 {
        self.inner.failed
    }

    /// Number of pending requests.
    #[napi(getter)]
    pub fn pending(&self) -> u32 {
        self.inner.pending
    }
}

impl From<BatchRequestCounts> for JsBatchRequestCounts {
    fn from(counts: BatchRequestCounts) -> Self {
        Self { inner: counts }
    }
}

/// A batch processing job.
///
/// Contains information about the status and progress of a batch.
#[napi]
#[derive(Clone)]
pub struct JsBatchJob {
    inner: BatchJob,
}

#[napi]
impl JsBatchJob {
    /// The batch ID.
    #[napi(getter)]
    pub fn id(&self) -> String {
        self.inner.id.clone()
    }

    /// The current status of the batch.
    #[napi(getter)]
    pub fn status(&self) -> JsBatchStatus {
        JsBatchStatus::from(self.inner.status)
    }

    /// Request counts.
    #[napi(getter)]
    pub fn request_counts(&self) -> JsBatchRequestCounts {
        JsBatchRequestCounts::from(self.inner.request_counts.clone())
    }

    /// When the batch was created (ISO 8601 timestamp).
    #[napi(getter)]
    pub fn created_at(&self) -> Option<String> {
        self.inner.created_at.clone()
    }

    /// When the batch started processing (ISO 8601 timestamp).
    #[napi(getter)]
    pub fn started_at(&self) -> Option<String> {
        self.inner.started_at.clone()
    }

    /// When the batch finished processing (ISO 8601 timestamp).
    #[napi(getter)]
    pub fn ended_at(&self) -> Option<String> {
        self.inner.ended_at.clone()
    }

    /// When the batch will expire (ISO 8601 timestamp).
    #[napi(getter)]
    pub fn expires_at(&self) -> Option<String> {
        self.inner.expires_at.clone()
    }

    /// Error message if the batch failed.
    #[napi(getter)]
    pub fn error(&self) -> Option<String> {
        self.inner.error.clone()
    }

    /// Check if the batch is complete (completed, failed, expired, or cancelled).
    #[napi]
    pub fn is_complete(&self) -> bool {
        matches!(
            self.inner.status,
            BatchStatus::Completed
                | BatchStatus::Failed
                | BatchStatus::Expired
                | BatchStatus::Cancelled
        )
    }

    /// Check if the batch is still in progress.
    #[napi]
    pub fn is_in_progress(&self) -> bool {
        matches!(
            self.inner.status,
            BatchStatus::Validating | BatchStatus::InProgress | BatchStatus::Finalizing
        )
    }
}

impl From<BatchJob> for JsBatchJob {
    fn from(job: BatchJob) -> Self {
        Self { inner: job }
    }
}

/// Error information for a failed batch request.
#[napi]
#[derive(Clone)]
pub struct JsBatchError {
    inner: BatchError,
}

#[napi]
impl JsBatchError {
    /// Error type.
    #[napi(getter)]
    pub fn error_type(&self) -> String {
        self.inner.error_type.clone()
    }

    /// Error message.
    #[napi(getter)]
    pub fn message(&self) -> String {
        self.inner.message.clone()
    }
}

impl From<BatchError> for JsBatchError {
    fn from(error: BatchError) -> Self {
        Self { inner: error }
    }
}

/// Result of a single request within a batch.
#[napi]
#[derive(Clone)]
pub struct JsBatchResult {
    inner: BatchResult,
}

#[napi]
impl JsBatchResult {
    /// The custom ID of the original request.
    #[napi(getter)]
    pub fn custom_id(&self) -> String {
        self.inner.custom_id.clone()
    }

    /// The completion response (if successful).
    #[napi(getter)]
    pub fn response(&self) -> Option<crate::types::response::JsCompletionResponse> {
        self.inner
            .response
            .clone()
            .map(crate::types::response::JsCompletionResponse::from)
    }

    /// The error (if failed).
    #[napi(getter)]
    pub fn error(&self) -> Option<JsBatchError> {
        self.inner.error.clone().map(JsBatchError::from)
    }

    /// Check if this result is successful.
    #[napi]
    pub fn is_success(&self) -> bool {
        self.inner.is_success()
    }

    /// Check if this result is an error.
    #[napi]
    pub fn is_error(&self) -> bool {
        self.inner.is_error()
    }
}

impl From<BatchResult> for JsBatchResult {
    fn from(result: BatchResult) -> Self {
        Self { inner: result }
    }
}

/// Result of a token counting request.
///
/// @example
/// ```typescript
/// const result = await client.countTokens(request)
/// console.log(`Input tokens: ${result.inputTokens}`)
/// ```
#[napi]
#[derive(Clone, Copy)]
pub struct JsTokenCountResult {
    inner: TokenCountResult,
}

#[napi]
impl JsTokenCountResult {
    /// Total number of input tokens.
    #[napi(getter)]
    pub fn input_tokens(&self) -> u32 {
        self.inner.input_tokens
    }
}

impl From<TokenCountResult> for JsTokenCountResult {
    fn from(result: TokenCountResult) -> Self {
        Self { inner: result }
    }
}
