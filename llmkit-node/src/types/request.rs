//! CompletionRequest type for JavaScript bindings

use llmkit::types::CompletionRequest;
use napi_derive::napi;

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
