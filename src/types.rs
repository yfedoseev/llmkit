//! Core types for the ModelSuite unified LLM API.
//!
//! This module defines the unified message format that works across all LLM providers.
//!
//! # Advanced Features
//!
//! This module supports advanced LLM features:
//! - **Prompt Caching**: Reduce costs by caching static content (`CacheControl`)
//! - **Extended Thinking**: Enable deep reasoning with budget control (`ThinkingConfig`)
//! - **Structured Outputs**: Guarantee JSON schema compliance (`StructuredOutput`)
//! - **Predicted Outputs**: Speed up generation for known content (`PredictionConfig`)
//! - **Document Support**: Process PDFs and other documents (`DocumentSource`)

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Message role in a conversation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    /// System message providing context or instructions
    System,
    /// User message
    User,
    /// Assistant (LLM) message
    Assistant,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
        }
    }
}

// ============================================================================
// Advanced Features Types
// ============================================================================

/// Cache control for prompt caching (Anthropic, Google).
///
/// Prompt caching allows you to cache static content (system prompts, tool definitions,
/// large context) to reduce costs by up to 90% on cache reads.
///
/// # Example
/// ```ignore
/// let request = CompletionRequest::new(model, messages)
///     .with_cache_control(CacheControl::ephemeral());
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CacheControl {
    /// Ephemeral cache with 5-minute TTL (default)
    #[default]
    Ephemeral,
    /// Extended cache with 1-hour TTL (Anthropic beta)
    Extended,
}

impl CacheControl {
    /// Create ephemeral cache control (5-minute TTL).
    pub fn ephemeral() -> Self {
        CacheControl::Ephemeral
    }

    /// Create extended cache control (1-hour TTL).
    pub fn extended() -> Self {
        CacheControl::Extended
    }
}

/// Cache breakpoint marking content to be cached.
///
/// Place this after static content (system prompt, tools, context) to mark
/// the caching boundary. Up to 4 breakpoints can be set per request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CacheBreakpoint {
    /// The cache control type
    pub cache_control: CacheControl,
}

impl CacheBreakpoint {
    /// Create a new cache breakpoint with ephemeral caching.
    pub fn ephemeral() -> Self {
        Self {
            cache_control: CacheControl::Ephemeral,
        }
    }

    /// Create a new cache breakpoint with extended (1-hour) caching.
    pub fn extended() -> Self {
        Self {
            cache_control: CacheControl::Extended,
        }
    }
}

/// Configuration for extended thinking / reasoning mode.
///
/// Extended thinking allows models to "think" more deeply about problems,
/// producing better results for complex reasoning tasks.
///
/// # Example
/// ```ignore
/// let request = CompletionRequest::new(model, messages)
///     .with_thinking(ThinkingConfig::enabled(10000));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Whether thinking is enabled
    #[serde(rename = "type")]
    pub thinking_type: ThinkingType,

    /// Maximum tokens for thinking (minimum 1024)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub budget_tokens: Option<u32>,
}

/// Type of thinking mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingType {
    /// Thinking is enabled
    Enabled,
    /// Thinking is disabled
    Disabled,
}

impl ThinkingConfig {
    /// Enable extended thinking with a token budget.
    ///
    /// # Arguments
    /// * `budget_tokens` - Maximum tokens for thinking (minimum 1024)
    pub fn enabled(budget_tokens: u32) -> Self {
        Self {
            thinking_type: ThinkingType::Enabled,
            budget_tokens: Some(budget_tokens.max(1024)),
        }
    }

    /// Disable extended thinking.
    pub fn disabled() -> Self {
        Self {
            thinking_type: ThinkingType::Disabled,
            budget_tokens: None,
        }
    }

    /// Check if thinking is enabled.
    pub fn is_enabled(&self) -> bool {
        matches!(self.thinking_type, ThinkingType::Enabled)
    }
}

/// Configuration for structured output / JSON schema enforcement.
///
/// Guarantees that model output adheres to a specific JSON schema.
/// Supported by OpenAI (100% reliability) and partially by others.
///
/// # Example
/// ```ignore
/// let schema = serde_json::json!({
///     "type": "object",
///     "properties": {
///         "name": {"type": "string"},
///         "age": {"type": "integer"}
///     },
///     "required": ["name", "age"]
/// });
/// let request = CompletionRequest::new(model, messages)
///     .with_structured_output(StructuredOutput::json_schema("person", schema));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuredOutput {
    /// Output format type
    #[serde(rename = "type")]
    pub format_type: StructuredOutputType,

    /// JSON schema definition (for json_schema type)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub json_schema: Option<JsonSchemaDefinition>,
}

/// Type of structured output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredOutputType {
    /// Plain text (no enforcement)
    Text,
    /// JSON object (basic validation)
    JsonObject,
    /// JSON with schema enforcement (strict)
    JsonSchema,
}

/// JSON schema definition for structured outputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JsonSchemaDefinition {
    /// Name of the schema
    pub name: String,
    /// Optional description
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The JSON schema
    pub schema: Value,
    /// Whether to enforce strict adherence (default: true)
    #[serde(default = "default_strict")]
    pub strict: bool,
}

fn default_strict() -> bool {
    true
}

impl StructuredOutput {
    /// Create a JSON schema structured output.
    pub fn json_schema(name: impl Into<String>, schema: Value) -> Self {
        Self {
            format_type: StructuredOutputType::JsonSchema,
            json_schema: Some(JsonSchemaDefinition {
                name: name.into(),
                description: None,
                schema,
                strict: true,
            }),
        }
    }

    /// Create a JSON schema structured output with description.
    pub fn json_schema_with_description(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: Value,
    ) -> Self {
        Self {
            format_type: StructuredOutputType::JsonSchema,
            json_schema: Some(JsonSchemaDefinition {
                name: name.into(),
                description: Some(description.into()),
                schema,
                strict: true,
            }),
        }
    }

    /// Create a basic JSON object output (no schema enforcement).
    pub fn json_object() -> Self {
        Self {
            format_type: StructuredOutputType::JsonObject,
            json_schema: None,
        }
    }

    /// Create plain text output (default).
    pub fn text() -> Self {
        Self {
            format_type: StructuredOutputType::Text,
            json_schema: None,
        }
    }
}

/// Configuration for predicted outputs (speculative decoding).
///
/// Speeds up generation when much of the output is already known,
/// useful for code editing, document updates, etc.
///
/// # Example
/// ```ignore
/// let request = CompletionRequest::new(model, messages)
///     .with_prediction(PredictionConfig::content(existing_code));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionConfig {
    /// Type of prediction
    #[serde(rename = "type")]
    pub prediction_type: PredictionType,

    /// The predicted content
    pub content: String,
}

/// Type of prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionType {
    /// Content prediction
    Content,
}

impl PredictionConfig {
    /// Create a content prediction.
    ///
    /// Use this when you expect the output to be similar to existing content,
    /// such as when editing code or updating documents.
    pub fn content(predicted_content: impl Into<String>) -> Self {
        Self {
            prediction_type: PredictionType::Content,
            content: predicted_content.into(),
        }
    }
}

/// Source for document content (PDF, etc.).
///
/// Allows processing of documents like PDFs directly in the API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DocumentSource {
    /// Base64-encoded document data
    Base64 {
        /// MIME type (e.g., "application/pdf")
        media_type: String,
        /// Base64-encoded document data
        data: String,
    },
    /// Document from URL
    Url {
        /// URL of the document
        url: String,
    },
    /// Document from file ID (for Files API)
    File {
        /// File ID from Files API
        file_id: String,
    },
}

impl DocumentSource {
    /// Create a document source from base64 data.
    pub fn base64(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        DocumentSource::Base64 {
            media_type: media_type.into(),
            data: data.into(),
        }
    }

    /// Create a PDF document source from base64 data.
    pub fn pdf_base64(data: impl Into<String>) -> Self {
        DocumentSource::Base64 {
            media_type: "application/pdf".to_string(),
            data: data.into(),
        }
    }

    /// Create a document source from URL.
    pub fn url(url: impl Into<String>) -> Self {
        DocumentSource::Url { url: url.into() }
    }

    /// Create a document source from file ID.
    pub fn file(file_id: impl Into<String>) -> Self {
        DocumentSource::File {
            file_id: file_id.into(),
        }
    }
}

/// Beta features to enable via headers.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum BetaFeature {
    /// Extended 128K output tokens
    Output128k,
    /// Extended 1-hour cache TTL
    ExtendedCacheTtl,
    /// Interleaved thinking (Claude 4)
    InterleavedThinking,
    /// Files API
    FilesApi,
    /// PDF support
    PdfSupport,
    /// Custom beta feature string
    Custom(String),
}

impl BetaFeature {
    /// Get the beta header value for Anthropic API.
    pub fn anthropic_header(&self) -> &str {
        match self {
            BetaFeature::Output128k => "output-128k-2025-02-19",
            BetaFeature::ExtendedCacheTtl => "extended-cache-ttl-2025-04-11",
            BetaFeature::InterleavedThinking => "interleaved-thinking-2025-05-14",
            BetaFeature::FilesApi => "files-api-2025-04-14",
            BetaFeature::PdfSupport => "pdfs-2024-09-25",
            BetaFeature::Custom(s) => s,
        }
    }
}

/// A block of content within a message.
///
/// Messages can contain multiple content blocks, allowing for mixed content
/// like text with images, or tool calls with text responses.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text content
    Text { text: String },

    /// Image content (base64 encoded)
    Image {
        /// MIME type (e.g., "image/png", "image/jpeg")
        media_type: String,
        /// Base64-encoded image data
        data: String,
    },

    /// Image from URL
    ImageUrl {
        /// URL of the image
        url: String,
    },

    /// Tool/function use request from the assistant
    ToolUse {
        /// Unique identifier for this tool use
        id: String,
        /// Name of the tool to call
        name: String,
        /// Input arguments as JSON
        input: Value,
    },

    /// Result of a tool execution
    ToolResult {
        /// ID of the tool_use this is a response to
        tool_use_id: String,
        /// The result content
        content: String,
        /// Whether the tool execution resulted in an error
        #[serde(default)]
        is_error: bool,
    },

    /// Thinking/reasoning block (for models that support it)
    Thinking {
        /// The thinking content
        thinking: String,
    },

    /// Document content (PDF, etc.)
    Document {
        /// Document source
        source: DocumentSource,
        /// Optional cache control for this document
        #[serde(skip_serializing_if = "Option::is_none")]
        cache_control: Option<CacheBreakpoint>,
    },

    /// Text with cache control (for prompt caching)
    TextWithCache {
        /// The text content
        text: String,
        /// Cache control settings
        cache_control: CacheBreakpoint,
    },
}

impl ContentBlock {
    /// Create a text content block.
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text { text: text.into() }
    }

    /// Create an image content block from base64 data.
    pub fn image(media_type: impl Into<String>, data: impl Into<String>) -> Self {
        ContentBlock::Image {
            media_type: media_type.into(),
            data: data.into(),
        }
    }

    /// Create an image content block from URL.
    pub fn image_url(url: impl Into<String>) -> Self {
        ContentBlock::ImageUrl { url: url.into() }
    }

    /// Create a tool use content block.
    pub fn tool_use(id: impl Into<String>, name: impl Into<String>, input: Value) -> Self {
        ContentBlock::ToolUse {
            id: id.into(),
            name: name.into(),
            input,
        }
    }

    /// Create a tool result content block.
    pub fn tool_result(
        tool_use_id: impl Into<String>,
        content: impl Into<String>,
        is_error: bool,
    ) -> Self {
        ContentBlock::ToolResult {
            tool_use_id: tool_use_id.into(),
            content: content.into(),
            is_error,
        }
    }

    /// Create a thinking content block.
    pub fn thinking(thinking: impl Into<String>) -> Self {
        ContentBlock::Thinking {
            thinking: thinking.into(),
        }
    }

    /// Create a document content block.
    pub fn document(source: DocumentSource) -> Self {
        ContentBlock::Document {
            source,
            cache_control: None,
        }
    }

    /// Create a document content block with cache control.
    pub fn document_cached(source: DocumentSource, cache_control: CacheBreakpoint) -> Self {
        ContentBlock::Document {
            source,
            cache_control: Some(cache_control),
        }
    }

    /// Create a PDF document content block from base64 data.
    pub fn pdf(data: impl Into<String>) -> Self {
        ContentBlock::Document {
            source: DocumentSource::pdf_base64(data),
            cache_control: None,
        }
    }

    /// Create a text content block with cache control (for prompt caching).
    pub fn text_cached(text: impl Into<String>, cache_control: CacheBreakpoint) -> Self {
        ContentBlock::TextWithCache {
            text: text.into(),
            cache_control,
        }
    }

    /// Create a text content block with ephemeral caching.
    pub fn text_cached_ephemeral(text: impl Into<String>) -> Self {
        ContentBlock::TextWithCache {
            text: text.into(),
            cache_control: CacheBreakpoint::ephemeral(),
        }
    }

    /// Check if this is a text block (including cached text).
    pub fn is_text(&self) -> bool {
        matches!(
            self,
            ContentBlock::Text { .. } | ContentBlock::TextWithCache { .. }
        )
    }

    /// Check if this is a document block.
    pub fn is_document(&self) -> bool {
        matches!(self, ContentBlock::Document { .. })
    }

    /// Check if this is a tool use block.
    pub fn is_tool_use(&self) -> bool {
        matches!(self, ContentBlock::ToolUse { .. })
    }

    /// Check if this is a tool result block.
    pub fn is_tool_result(&self) -> bool {
        matches!(self, ContentBlock::ToolResult { .. })
    }

    /// Extract text content if this is a text block.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            ContentBlock::Text { text } => Some(text),
            ContentBlock::TextWithCache { text, .. } => Some(text),
            _ => None,
        }
    }

    /// Extract tool use details if this is a tool use block.
    pub fn as_tool_use(&self) -> Option<(&str, &str, &Value)> {
        match self {
            ContentBlock::ToolUse { id, name, input } => Some((id, name, input)),
            _ => None,
        }
    }
}

/// A message in a conversation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// The role of the message sender
    pub role: Role,
    /// The content blocks of the message
    pub content: Vec<ContentBlock>,
}

impl Message {
    /// Create a new message with the given role and content blocks.
    pub fn new(role: Role, content: Vec<ContentBlock>) -> Self {
        Self { role, content }
    }

    /// Create a system message with text content.
    pub fn system(text: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: vec![ContentBlock::text(text)],
        }
    }

    /// Create a user message with text content.
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: vec![ContentBlock::text(text)],
        }
    }

    /// Create an assistant message with text content.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: vec![ContentBlock::text(text)],
        }
    }

    /// Create a user message with multiple content blocks.
    pub fn user_with_content(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content,
        }
    }

    /// Create an assistant message with multiple content blocks (e.g., tool calls).
    pub fn assistant_with_content(content: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::Assistant,
            content,
        }
    }

    /// Create a tool result message.
    pub fn tool_results(results: Vec<ContentBlock>) -> Self {
        Self {
            role: Role::User,
            content: results,
        }
    }

    /// Get all text content from the message concatenated.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| block.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Extract all tool use blocks from the message.
    pub fn tool_uses(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|block| block.is_tool_use())
            .collect()
    }

    /// Check if the message contains any tool use blocks.
    pub fn has_tool_use(&self) -> bool {
        self.content.iter().any(|block| block.is_tool_use())
    }
}

/// Definition of a tool that can be called by the LLM.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolDefinition {
    /// Name of the tool
    pub name: String,
    /// Description of what the tool does
    pub description: String,
    /// JSON Schema for the tool's input parameters
    pub input_schema: Value,
}

impl ToolDefinition {
    /// Create a new tool definition.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            input_schema,
        }
    }
}

/// Request to complete a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionRequest {
    /// Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o")
    pub model: String,

    /// Conversation messages
    pub messages: Vec<Message>,

    /// System prompt (separate from messages for providers that support it)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Available tools for the model to use
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,

    /// Maximum tokens to generate
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,

    /// Sampling temperature (0.0 to 2.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,

    /// Top-p sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,

    /// Stop sequences
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_sequences: Option<Vec<String>>,

    /// Whether to stream the response
    #[serde(default)]
    pub stream: bool,

    // ========== Advanced Features ==========
    /// Extended thinking configuration (Anthropic Claude 3.7+)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking: Option<ThinkingConfig>,

    /// Structured output configuration (OpenAI, Google)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response_format: Option<StructuredOutput>,

    /// Predicted output for speculative decoding (OpenAI)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prediction: Option<PredictionConfig>,

    /// Cache control for the system prompt (Anthropic)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system_cache_control: Option<CacheBreakpoint>,

    /// Beta features to enable (Anthropic)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub beta_features: Option<Vec<BetaFeature>>,

    /// Provider-specific options
    #[serde(skip_serializing_if = "Option::is_none")]
    pub extra: Option<Value>,
}

impl CompletionRequest {
    /// Create a new completion request with required fields.
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            system: None,
            tools: None,
            max_tokens: None,
            temperature: None,
            top_p: None,
            stop_sequences: None,
            stream: false,
            thinking: None,
            response_format: None,
            prediction: None,
            system_cache_control: None,
            beta_features: None,
            extra: None,
        }
    }

    /// Builder method: Set the system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Builder method: Set available tools.
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Builder method: Set max tokens.
    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Builder method: Set temperature.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Builder method: Set top-p.
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Builder method: Set stop sequences.
    pub fn with_stop_sequences(mut self, stop_sequences: Vec<String>) -> Self {
        self.stop_sequences = Some(stop_sequences);
        self
    }

    /// Builder method: Enable streaming.
    pub fn with_streaming(mut self) -> Self {
        self.stream = true;
        self
    }

    // ========== Advanced Features Builder Methods ==========

    /// Builder method: Enable extended thinking with a token budget.
    ///
    /// Extended thinking allows Claude to reason more deeply about complex problems.
    /// Available on Claude 3.7+ models.
    ///
    /// # Example
    /// ```ignore
    /// let request = CompletionRequest::new(model, messages)
    ///     .with_thinking(10000);  // 10k token budget for thinking
    /// ```
    pub fn with_thinking(mut self, budget_tokens: u32) -> Self {
        self.thinking = Some(ThinkingConfig::enabled(budget_tokens));
        self
    }

    /// Builder method: Set extended thinking configuration.
    pub fn with_thinking_config(mut self, config: ThinkingConfig) -> Self {
        self.thinking = Some(config);
        self
    }

    /// Builder method: Set structured output with JSON schema.
    ///
    /// Guarantees the model output adheres to the specified JSON schema.
    /// Supported by OpenAI with 100% reliability.
    ///
    /// # Example
    /// ```ignore
    /// let schema = serde_json::json!({
    ///     "type": "object",
    ///     "properties": {"name": {"type": "string"}},
    ///     "required": ["name"]
    /// });
    /// let request = CompletionRequest::new(model, messages)
    ///     .with_json_schema("response", schema);
    /// ```
    pub fn with_json_schema(mut self, name: impl Into<String>, schema: Value) -> Self {
        self.response_format = Some(StructuredOutput::json_schema(name, schema));
        self
    }

    /// Builder method: Set structured output configuration.
    pub fn with_response_format(mut self, format: StructuredOutput) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Builder method: Enable JSON object output (basic, no schema).
    pub fn with_json_output(mut self) -> Self {
        self.response_format = Some(StructuredOutput::json_object());
        self
    }

    /// Builder method: Set predicted output for speculative decoding.
    ///
    /// Speeds up generation when much of the output is already known.
    /// Useful for code editing, document updates, etc.
    ///
    /// # Example
    /// ```ignore
    /// let request = CompletionRequest::new(model, messages)
    ///     .with_prediction(existing_code);
    /// ```
    pub fn with_prediction(mut self, predicted_content: impl Into<String>) -> Self {
        self.prediction = Some(PredictionConfig::content(predicted_content));
        self
    }

    /// Builder method: Enable prompt caching for the system prompt.
    ///
    /// Caches the system prompt for 5 minutes to reduce costs on subsequent calls.
    /// Available on Anthropic Claude models.
    pub fn with_system_caching(mut self) -> Self {
        self.system_cache_control = Some(CacheBreakpoint::ephemeral());
        self
    }

    /// Builder method: Enable extended (1-hour) prompt caching for the system prompt.
    ///
    /// Requires the `extended-cache-ttl` beta feature.
    pub fn with_system_caching_extended(mut self) -> Self {
        self.system_cache_control = Some(CacheBreakpoint::extended());
        // Automatically add the beta feature
        let mut features = self.beta_features.unwrap_or_default();
        if !features.contains(&BetaFeature::ExtendedCacheTtl) {
            features.push(BetaFeature::ExtendedCacheTtl);
        }
        self.beta_features = Some(features);
        self
    }

    /// Builder method: Add a beta feature.
    pub fn with_beta_feature(mut self, feature: BetaFeature) -> Self {
        let mut features = self.beta_features.unwrap_or_default();
        if !features.contains(&feature) {
            features.push(feature);
        }
        self.beta_features = Some(features);
        self
    }

    /// Builder method: Enable 128K output tokens (Anthropic beta).
    pub fn with_extended_output(self) -> Self {
        self.with_beta_feature(BetaFeature::Output128k)
    }

    /// Builder method: Enable interleaved thinking (Claude 4 only).
    pub fn with_interleaved_thinking(self) -> Self {
        self.with_beta_feature(BetaFeature::InterleavedThinking)
    }

    /// Builder method: Set provider-specific extra options.
    pub fn with_extra(mut self, extra: Value) -> Self {
        self.extra = Some(extra);
        self
    }

    // ========== Helper Methods ==========

    /// Check if prompt caching is enabled.
    pub fn has_caching(&self) -> bool {
        self.system_cache_control.is_some()
            || self.messages.iter().any(|m| {
                m.content.iter().any(|c| {
                    matches!(
                        c,
                        ContentBlock::TextWithCache { .. }
                            | ContentBlock::Document {
                                cache_control: Some(_),
                                ..
                            }
                    )
                })
            })
    }

    /// Check if extended thinking is enabled.
    pub fn has_thinking(&self) -> bool {
        self.thinking.as_ref().is_some_and(|t| t.is_enabled())
    }

    /// Check if structured output is enabled.
    pub fn has_structured_output(&self) -> bool {
        self.response_format.is_some()
    }

    /// Get the required beta headers for Anthropic.
    pub fn anthropic_beta_headers(&self) -> Vec<&str> {
        let mut headers = Vec::new();

        if let Some(ref features) = self.beta_features {
            for feature in features {
                headers.push(feature.anthropic_header());
            }
        }

        // Auto-add headers based on features used
        if self.thinking.is_some() && !headers.iter().any(|h| h.contains("thinking")) {
            // Extended thinking doesn't require a specific beta header for basic use
        }

        if let Some(CacheBreakpoint {
            cache_control: CacheControl::Extended,
        }) = &self.system_cache_control
        {
            if !headers.iter().any(|h| h.contains("cache-ttl")) {
                headers.push(BetaFeature::ExtendedCacheTtl.anthropic_header());
            }
        }

        headers
    }
}

/// Reason the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    /// Natural end of response
    EndTurn,
    /// Hit max tokens limit
    MaxTokens,
    /// Model wants to use a tool
    ToolUse,
    /// Hit a stop sequence
    StopSequence,
    /// Response was filtered by content moderation
    ContentFilter,
}

/// Token usage information.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    /// Number of tokens in the prompt
    pub input_tokens: u32,
    /// Number of tokens in the completion
    pub output_tokens: u32,
    /// Cache creation tokens (if applicable)
    #[serde(default)]
    pub cache_creation_input_tokens: u32,
    /// Cache read tokens (if applicable)
    #[serde(default)]
    pub cache_read_input_tokens: u32,
}

impl Usage {
    /// Total tokens used (input + output).
    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.output_tokens
    }
}

// ============================================================================
// Token Counting Types
// ============================================================================

/// Request to count tokens in content.
///
/// This allows estimation of token counts before making a completion request,
/// useful for cost estimation and context window management.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenCountRequest {
    /// Model to use for tokenization
    pub model: String,

    /// Messages to count tokens for
    pub messages: Vec<Message>,

    /// System prompt to include in count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,

    /// Tools to include in count
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolDefinition>>,
}

impl TokenCountRequest {
    /// Create a new token count request.
    pub fn new(model: impl Into<String>, messages: Vec<Message>) -> Self {
        Self {
            model: model.into(),
            messages,
            system: None,
            tools: None,
        }
    }

    /// Builder: Set the system prompt.
    pub fn with_system(mut self, system: impl Into<String>) -> Self {
        self.system = Some(system.into());
        self
    }

    /// Builder: Set the tools.
    pub fn with_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = Some(tools);
        self
    }

    /// Create from a CompletionRequest.
    pub fn from_completion_request(request: &CompletionRequest) -> Self {
        Self {
            model: request.model.clone(),
            messages: request.messages.clone(),
            system: request.system.clone(),
            tools: request.tools.clone(),
        }
    }
}

/// Result of a token counting request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenCountResult {
    /// Total number of input tokens
    pub input_tokens: u32,
}

impl TokenCountResult {
    /// Create a new token count result.
    pub fn new(input_tokens: u32) -> Self {
        Self { input_tokens }
    }
}

// ============================================================================
// Batch Processing Types
// ============================================================================

/// A single request within a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    /// Custom ID for this request (used to match results)
    pub custom_id: String,

    /// The completion request
    pub request: CompletionRequest,
}

impl BatchRequest {
    /// Create a new batch request.
    pub fn new(custom_id: impl Into<String>, request: CompletionRequest) -> Self {
        Self {
            custom_id: custom_id.into(),
            request,
        }
    }
}

/// Status of a batch job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BatchStatus {
    /// Batch is being validated
    Validating,
    /// Batch is queued for processing
    InProgress,
    /// Batch is being finalized
    Finalizing,
    /// Batch completed successfully
    Completed,
    /// Batch failed
    Failed,
    /// Batch expired before completion
    Expired,
    /// Batch was cancelled
    Cancelled,
}

impl BatchStatus {
    /// Check if the batch is still processing.
    pub fn is_processing(&self) -> bool {
        matches!(
            self,
            BatchStatus::Validating | BatchStatus::InProgress | BatchStatus::Finalizing
        )
    }

    /// Check if the batch is done (successfully or not).
    pub fn is_done(&self) -> bool {
        matches!(
            self,
            BatchStatus::Completed
                | BatchStatus::Failed
                | BatchStatus::Expired
                | BatchStatus::Cancelled
        )
    }

    /// Check if the batch completed successfully.
    pub fn is_success(&self) -> bool {
        matches!(self, BatchStatus::Completed)
    }
}

/// Information about a batch job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchJob {
    /// Unique batch ID
    pub id: String,

    /// Current status
    pub status: BatchStatus,

    /// When the batch was created
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<String>,

    /// When the batch started processing
    #[serde(skip_serializing_if = "Option::is_none")]
    pub started_at: Option<String>,

    /// When the batch finished
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<String>,

    /// When the batch expires
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,

    /// Total requests in the batch
    #[serde(default)]
    pub request_counts: BatchRequestCounts,

    /// Error message if the batch failed
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Request counts for a batch job.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BatchRequestCounts {
    /// Total number of requests
    pub total: u32,
    /// Successfully completed requests
    pub succeeded: u32,
    /// Failed requests
    pub failed: u32,
    /// Pending requests
    pub pending: u32,
}

/// Result of a single request in a batch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    /// Custom ID matching the request
    pub custom_id: String,

    /// The completion response (if successful)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<CompletionResponse>,

    /// Error message (if failed)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<BatchError>,
}

/// Error from a batch request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchError {
    /// Error type
    #[serde(rename = "type")]
    pub error_type: String,
    /// Error message
    pub message: String,
}

impl BatchResult {
    /// Check if this result was successful.
    pub fn is_success(&self) -> bool {
        self.response.is_some()
    }

    /// Check if this result was an error.
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }
}

/// Response from a completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    /// Unique response ID
    pub id: String,

    /// Model that generated the response
    pub model: String,

    /// Content blocks in the response
    pub content: Vec<ContentBlock>,

    /// Reason the model stopped
    pub stop_reason: StopReason,

    /// Token usage
    pub usage: Usage,
}

impl CompletionResponse {
    /// Get all text content from the response concatenated.
    pub fn text_content(&self) -> String {
        self.content
            .iter()
            .filter_map(|block| block.as_text())
            .collect::<Vec<_>>()
            .join("")
    }

    /// Extract all tool use blocks from the response.
    pub fn tool_uses(&self) -> Vec<&ContentBlock> {
        self.content
            .iter()
            .filter(|block| block.is_tool_use())
            .collect()
    }

    /// Check if the response contains tool use.
    pub fn has_tool_use(&self) -> bool {
        self.stop_reason == StopReason::ToolUse || self.content.iter().any(|b| b.is_tool_use())
    }
}

/// Delta content for streaming responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentDelta {
    /// Text being streamed
    Text { text: String },

    /// Tool use being streamed
    ToolUse {
        id: Option<String>,
        name: Option<String>,
        /// Partial JSON input
        input_json_delta: Option<String>,
    },

    /// Thinking content being streamed
    Thinking { thinking: String },
}

/// A chunk from a streaming response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// The type of stream event
    pub event_type: StreamEventType,

    /// Index of the content block being updated (for multi-block responses)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<usize>,

    /// The delta content (if applicable)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delta: Option<ContentDelta>,

    /// Stop reason (only on message_stop)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop_reason: Option<StopReason>,

    /// Usage information (may be partial or final)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// Type of streaming event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamEventType {
    /// Message started
    MessageStart,
    /// Content block started
    ContentBlockStart,
    /// Content block delta
    ContentBlockDelta,
    /// Content block finished
    ContentBlockStop,
    /// Message delta (usually contains stop reason)
    MessageDelta,
    /// Message finished
    MessageStop,
    /// Ping (keepalive)
    Ping,
    /// Error occurred
    Error,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_creation() {
        let msg = Message::user("Hello, world!");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.text_content(), "Hello, world!");
    }

    #[test]
    fn test_content_block_helpers() {
        let text = ContentBlock::text("test");
        assert!(text.is_text());
        assert_eq!(text.as_text(), Some("test"));

        let tool = ContentBlock::tool_use("id1", "bash", serde_json::json!({"command": "ls"}));
        assert!(tool.is_tool_use());
    }

    #[test]
    fn test_completion_request_builder() {
        let req = CompletionRequest::new("claude-sonnet-4-20250514", vec![Message::user("Hi")])
            .with_max_tokens(1024)
            .with_temperature(0.7)
            .with_streaming();

        assert_eq!(req.model, "claude-sonnet-4-20250514");
        assert_eq!(req.max_tokens, Some(1024));
        assert_eq!(req.temperature, Some(0.7));
        assert!(req.stream);
    }

    #[test]
    fn test_batch_status() {
        assert!(BatchStatus::Validating.is_processing());
        assert!(BatchStatus::InProgress.is_processing());
        assert!(BatchStatus::Finalizing.is_processing());

        assert!(!BatchStatus::Completed.is_processing());
        assert!(!BatchStatus::Failed.is_processing());

        assert!(BatchStatus::Completed.is_done());
        assert!(BatchStatus::Failed.is_done());
        assert!(BatchStatus::Expired.is_done());
        assert!(BatchStatus::Cancelled.is_done());

        assert!(!BatchStatus::InProgress.is_done());

        assert!(BatchStatus::Completed.is_success());
        assert!(!BatchStatus::Failed.is_success());
    }

    #[test]
    fn test_batch_request_creation() {
        let request =
            CompletionRequest::new("claude-sonnet-4-20250514", vec![Message::user("Hello")]);
        let batch_req = BatchRequest::new("req-001", request);

        assert_eq!(batch_req.custom_id, "req-001");
        assert_eq!(batch_req.request.model, "claude-sonnet-4-20250514");
    }
}
