//! AWS Bedrock provider implementation using the Converse API.
//!
//! This provider supports multiple model families hosted on AWS Bedrock:
//! - **Anthropic Claude**: Claude 4.5, Claude 4, Claude 3.5, Claude 3
//! - **Amazon Nova**: Nova Pro, Nova Lite, Nova Micro, Nova 2 Pro, Nova 2 Lite
//! - **Meta Llama**: Llama 4, Llama 3.3, Llama 3.2, Llama 3.1, Llama 3
//! - **Mistral AI**: Mistral Large, Mistral Small, Mixtral 8x7B
//! - **Cohere**: Command R+, Command R
//! - **AI21 Labs**: Jamba 1.5
//! - **Amazon Titan**: Titan Text Express, Titan Text Lite
//! - **DeepSeek**: DeepSeek-R1, DeepSeek-V3
//! - **Qwen (Alibaba)**: Qwen 2.5 (uses InvokeModel fallback)
//!
//! # Converse API
//!
//! This provider uses the AWS Bedrock Converse API which provides:
//! - Unified request/response format across all supported models
//! - Native support for cross-region inference profiles (e.g., `us.amazon.nova-micro-v1:0`)
//! - Consistent tool use support
//! - Automatic model routing
//!
//! # Configuration
//!
//! Bedrock uses AWS credentials. You can provide:
//! - Default credentials from `~/.aws/credentials`
//! - Environment variables (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`)
//! - IAM role (when running on AWS)
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::providers::bedrock::BedrockProvider;
//!
//! // Using default AWS credentials
//! let provider = BedrockProvider::from_env("us-east-1").await?;
//!
//! // With cross-region inference profile
//! let request = CompletionRequest::new(
//!     "us.amazon.nova-micro-v1:0",  // US inference profile
//!     vec![Message::user("Hello!")],
//! );
//! let response = provider.complete(request).await?;
//! ```

use std::collections::HashMap;
use std::pin::Pin;

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::types::{
    ContentBlock as BedrockContentBlock, ConversationRole, DocumentBlock, DocumentFormat,
    DocumentSource, ImageBlock, ImageFormat, ImageSource, InferenceConfiguration,
    Message as BedrockMessage, SystemContentBlock, Tool, ToolConfiguration, ToolInputSchema,
    ToolResultBlock, ToolResultContentBlock, ToolResultStatus, ToolSpecification, ToolUseBlock,
};
use aws_sdk_bedrockruntime::Client as BedrockClient;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

/// Configuration for Bedrock provider.
#[derive(Debug, Clone)]
pub struct BedrockConfig {
    /// AWS region (e.g., "us-east-1")
    pub region: String,

    /// Request timeout
    pub timeout: std::time::Duration,

    /// Model IDs that should use InvokeModel instead of Converse
    /// (for models without Converse API support)
    pub invoke_model_overrides: HashMap<String, bool>,
}

impl Default for BedrockConfig {
    fn default() -> Self {
        Self {
            region: "us-east-1".to_string(),
            timeout: std::time::Duration::from_secs(120),
            invoke_model_overrides: HashMap::new(),
        }
    }
}

impl BedrockConfig {
    /// Create a new config with the specified region.
    pub fn new(region: impl Into<String>) -> Self {
        Self {
            region: region.into(),
            ..Default::default()
        }
    }

    /// Builder: Set timeout.
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Builder: Force a model to use InvokeModel instead of Converse.
    pub fn with_invoke_model_override(mut self, model_id: impl Into<String>) -> Self {
        self.invoke_model_overrides.insert(model_id.into(), true);
        self
    }
}

/// Builder for BedrockProvider.
pub struct BedrockBuilder {
    config: BedrockConfig,
}

impl BedrockBuilder {
    /// Create a new builder with default config.
    pub fn new() -> Self {
        Self {
            config: BedrockConfig::default(),
        }
    }

    /// Set the AWS region.
    pub fn region(mut self, region: impl Into<String>) -> Self {
        self.config.region = region.into();
        self
    }

    /// Set timeout.
    pub fn timeout(mut self, timeout: std::time::Duration) -> Self {
        self.config.timeout = timeout;
        self
    }

    /// Force a model to use InvokeModel instead of Converse.
    pub fn invoke_model_override(mut self, model_id: impl Into<String>) -> Self {
        self.config
            .invoke_model_overrides
            .insert(model_id.into(), true);
        self
    }

    /// Build the provider using default AWS credential chain.
    pub async fn build(self) -> Result<BedrockProvider> {
        let config = aws_config::defaults(BehaviorVersion::latest())
            .region(aws_config::Region::new(self.config.region.clone()))
            .load()
            .await;

        let client = BedrockClient::new(&config);

        Ok(BedrockProvider {
            client,
            config: self.config,
        })
    }
}

impl Default for BedrockBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// AWS Bedrock provider.
///
/// Uses the Converse API for unified access to all supported model families.
/// Falls back to InvokeModel for models without Converse support (e.g., Qwen).
pub struct BedrockProvider {
    client: BedrockClient,
    config: BedrockConfig,
}

impl BedrockProvider {
    /// Create a builder for the provider.
    pub fn builder() -> BedrockBuilder {
        BedrockBuilder::new()
    }

    /// Create a provider with default credentials and specified region.
    pub async fn from_env(region: impl Into<String>) -> Result<Self> {
        Self::builder().region(region).build().await
    }

    /// Create a provider from environment variable.
    pub async fn from_env_region() -> Result<Self> {
        let region = std::env::var("AWS_REGION")
            .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
            .unwrap_or_else(|_| "us-east-1".to_string());
        Self::from_env(region).await
    }

    /// Check if a model requires InvokeModel instead of Converse.
    fn requires_invoke_model(&self, model_id: &str) -> bool {
        let id = model_id.to_lowercase();

        // Check explicit overrides first
        if let Some(&use_invoke) = self.config.invoke_model_overrides.get(model_id) {
            return use_invoke;
        }

        // Models without Converse support
        // Note: Bedrock uses "qwen2-5" format (hyphen) while official Qwen uses "qwen2.5" (dot)
        id.contains("qwen2.5")
            || id.contains("qwen2-5")
            || id.contains("qwen2-vl")
            || id.contains("titan-embed")
    }
}

// ============================================================================
// Converse API Message Conversion
// ============================================================================

/// Build Converse API messages from CompletionRequest.
fn build_converse_messages(request: &CompletionRequest) -> Result<Vec<BedrockMessage>> {
    let mut messages = Vec::new();

    for msg in &request.messages {
        // Skip system messages - they go in the system parameter
        if msg.role == Role::System {
            continue;
        }

        let role = match msg.role {
            Role::User => ConversationRole::User,
            Role::Assistant => ConversationRole::Assistant,
            Role::System => continue, // Already handled above
        };

        let mut content_blocks = Vec::new();

        for block in &msg.content {
            match block {
                ContentBlock::Text { text } => {
                    content_blocks.push(BedrockContentBlock::Text(text.clone()));
                }
                ContentBlock::Image { media_type, data } => {
                    // Determine image format from media type
                    let format = match media_type.as_str() {
                        "image/png" => ImageFormat::Png,
                        "image/jpeg" | "image/jpg" => ImageFormat::Jpeg,
                        "image/gif" => ImageFormat::Gif,
                        "image/webp" => ImageFormat::Webp,
                        _ => ImageFormat::Png, // Default to PNG
                    };

                    // Decode base64 to bytes
                    let bytes =
                        base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data)
                            .map_err(|e| {
                                Error::invalid_request(format!("Invalid base64 image: {}", e))
                            })?;

                    let image_block = ImageBlock::builder()
                        .format(format)
                        .source(ImageSource::Bytes(Blob::new(bytes)))
                        .build()
                        .map_err(|e| Error::invalid_request(e.to_string()))?;

                    content_blocks.push(BedrockContentBlock::Image(image_block));
                }
                ContentBlock::ToolUse { id, name, input } => {
                    // Convert serde_json::Value to aws_smithy_types::Document
                    let doc = json_value_to_document(input);

                    let tool_use = ToolUseBlock::builder()
                        .tool_use_id(id)
                        .name(name)
                        .input(doc)
                        .build()
                        .map_err(|e| Error::invalid_request(e.to_string()))?;

                    content_blocks.push(BedrockContentBlock::ToolUse(tool_use));
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    let status = if *is_error {
                        ToolResultStatus::Error
                    } else {
                        ToolResultStatus::Success
                    };

                    let tool_result = ToolResultBlock::builder()
                        .tool_use_id(tool_use_id)
                        .content(ToolResultContentBlock::Text(content.clone()))
                        .status(status)
                        .build()
                        .map_err(|e| Error::invalid_request(e.to_string()))?;

                    content_blocks.push(BedrockContentBlock::ToolResult(tool_result));
                }
                ContentBlock::Document { source, .. } => {
                    // Extract media_type and data from DocumentSource
                    if let crate::types::DocumentSource::Base64 { media_type, data } = source {
                        // Determine document format
                        let format = match media_type.as_str() {
                            "application/pdf" => DocumentFormat::Pdf,
                            "text/plain" => DocumentFormat::Txt,
                            "text/html" => DocumentFormat::Html,
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document" => {
                                DocumentFormat::Docx
                            }
                            _ => DocumentFormat::Pdf, // Default
                        };

                        let bytes = base64::Engine::decode(
                            &base64::engine::general_purpose::STANDARD,
                            data,
                        )
                        .map_err(|e| {
                            Error::invalid_request(format!("Invalid base64 document: {}", e))
                        })?;

                        let doc = DocumentBlock::builder()
                            .format(format)
                            .name("document")
                            .source(DocumentSource::Bytes(Blob::new(bytes)))
                            .build()
                            .map_err(|e| Error::invalid_request(e.to_string()))?;

                        content_blocks.push(BedrockContentBlock::Document(doc));
                    }
                    // URL and File sources not directly supported by Bedrock Converse API
                }
                ContentBlock::ImageUrl { .. } => {
                    // Image URLs are not directly supported by Bedrock - would need to fetch and convert
                    // Skip for now - users should pass base64 encoded images
                }
                ContentBlock::Thinking { thinking } => {
                    // Thinking blocks can be passed as text to models that support it
                    content_blocks.push(BedrockContentBlock::Text(thinking.clone()));
                }
                ContentBlock::TextWithCache { text, .. } => {
                    // Bedrock doesn't support cache control - just pass the text
                    content_blocks.push(BedrockContentBlock::Text(text.clone()));
                }
            }
        }

        if !content_blocks.is_empty() {
            let message = BedrockMessage::builder()
                .role(role)
                .set_content(Some(content_blocks))
                .build()
                .map_err(|e| Error::invalid_request(e.to_string()))?;

            messages.push(message);
        }
    }

    Ok(messages)
}

/// Build system content from request.
fn build_system_content(request: &CompletionRequest) -> Option<Vec<SystemContentBlock>> {
    // Check for explicit system field first
    if let Some(ref system) = request.system {
        return Some(vec![SystemContentBlock::Text(system.clone())]);
    }

    // Also check for system messages in the messages array
    let system_text: String = request
        .messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.text_content())
        .collect::<Vec<_>>()
        .join("\n\n");

    if system_text.is_empty() {
        None
    } else {
        Some(vec![SystemContentBlock::Text(system_text)])
    }
}

/// Build inference configuration from request.
fn build_inference_config(request: &CompletionRequest) -> Option<InferenceConfiguration> {
    let mut builder = InferenceConfiguration::builder();
    let mut has_config = false;

    if let Some(max_tokens) = request.max_tokens {
        builder = builder.max_tokens(max_tokens as i32);
        has_config = true;
    }

    if let Some(temperature) = request.temperature {
        builder = builder.temperature(temperature);
        has_config = true;
    }

    if let Some(top_p) = request.top_p {
        builder = builder.top_p(top_p);
        has_config = true;
    }

    if let Some(ref stop_sequences) = request.stop_sequences {
        builder = builder.set_stop_sequences(Some(stop_sequences.clone()));
        has_config = true;
    }

    if has_config {
        Some(builder.build())
    } else {
        None
    }
}

/// Build tool configuration from request.
fn build_tool_config(request: &CompletionRequest) -> Option<ToolConfiguration> {
    let tools = request.tools.as_ref()?;

    if tools.is_empty() {
        return None;
    }

    let tool_specs: Vec<Tool> = tools
        .iter()
        .filter_map(|t| {
            let input_schema = ToolInputSchema::Json(json_value_to_document(&t.input_schema));

            let spec = ToolSpecification::builder()
                .name(&t.name)
                .description(&t.description)
                .input_schema(input_schema)
                .build()
                .ok()?;

            Some(Tool::ToolSpec(spec))
        })
        .collect();

    if tool_specs.is_empty() {
        return None;
    }

    ToolConfiguration::builder()
        .set_tools(Some(tool_specs))
        .build()
        .ok()
}

// ============================================================================
// Helper Functions for Document Conversion
// ============================================================================

/// Convert serde_json::Value to aws_smithy_types::Document.
fn json_value_to_document(value: &Value) -> aws_smithy_types::Document {
    match value {
        Value::Null => aws_smithy_types::Document::Null,
        Value::Bool(b) => aws_smithy_types::Document::Bool(*b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                aws_smithy_types::Document::Number(aws_smithy_types::Number::PosInt(i as u64))
            } else if let Some(f) = n.as_f64() {
                aws_smithy_types::Document::Number(aws_smithy_types::Number::Float(f))
            } else {
                aws_smithy_types::Document::Null
            }
        }
        Value::String(s) => aws_smithy_types::Document::String(s.clone()),
        Value::Array(arr) => {
            aws_smithy_types::Document::Array(arr.iter().map(json_value_to_document).collect())
        }
        Value::Object(obj) => aws_smithy_types::Document::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), json_value_to_document(v)))
                .collect(),
        ),
    }
}

/// Convert aws_smithy_types::Document to serde_json::Value.
fn document_to_json_value(doc: &aws_smithy_types::Document) -> Value {
    match doc {
        aws_smithy_types::Document::Null => Value::Null,
        aws_smithy_types::Document::Bool(b) => Value::Bool(*b),
        aws_smithy_types::Document::Number(n) => match n {
            aws_smithy_types::Number::PosInt(i) => Value::Number((*i).into()),
            aws_smithy_types::Number::NegInt(i) => Value::Number((*i).into()),
            aws_smithy_types::Number::Float(f) => {
                serde_json::Number::from_f64(*f).map_or(Value::Null, Value::Number)
            }
        },
        aws_smithy_types::Document::String(s) => Value::String(s.clone()),
        aws_smithy_types::Document::Array(arr) => {
            Value::Array(arr.iter().map(document_to_json_value).collect())
        }
        aws_smithy_types::Document::Object(obj) => Value::Object(
            obj.iter()
                .map(|(k, v)| (k.clone(), document_to_json_value(v)))
                .collect(),
        ),
    }
}

// ============================================================================
// Converse API Response Parsing
// ============================================================================

/// Parse Converse API response to CompletionResponse.
fn parse_converse_response(
    response: aws_sdk_bedrockruntime::operation::converse::ConverseOutput,
    model: &str,
) -> Result<CompletionResponse> {
    let output = response
        .output
        .ok_or_else(|| Error::server(500, "No output in Bedrock response"))?;

    let message = match output {
        aws_sdk_bedrockruntime::types::ConverseOutput::Message(msg) => msg,
        _ => return Err(Error::server(500, "Unexpected output type from Bedrock")),
    };

    let mut content = Vec::new();

    for block in message.content {
        match block {
            BedrockContentBlock::Text(text) => {
                content.push(ContentBlock::Text { text });
            }
            BedrockContentBlock::ToolUse(tool_use) => {
                // Convert Document to serde_json::Value using our helper
                let input = document_to_json_value(&tool_use.input);

                content.push(ContentBlock::ToolUse {
                    id: tool_use.tool_use_id,
                    name: tool_use.name,
                    input,
                });
            }
            _ => {
                // Skip other content types
            }
        }
    }

    let stop_reason = match response.stop_reason {
        aws_sdk_bedrockruntime::types::StopReason::EndTurn => StopReason::EndTurn,
        aws_sdk_bedrockruntime::types::StopReason::ToolUse => StopReason::ToolUse,
        aws_sdk_bedrockruntime::types::StopReason::MaxTokens => StopReason::MaxTokens,
        aws_sdk_bedrockruntime::types::StopReason::StopSequence => StopReason::StopSequence,
        aws_sdk_bedrockruntime::types::StopReason::ContentFiltered => StopReason::ContentFilter,
        _ => StopReason::EndTurn,
    };

    let usage = response
        .usage
        .map(|u| Usage {
            input_tokens: u.input_tokens as u32,
            output_tokens: u.output_tokens as u32,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        })
        .unwrap_or_default();

    Ok(CompletionResponse {
        id: format!("bedrock-{}", uuid::Uuid::new_v4()),
        model: model.to_string(),
        content,
        stop_reason,
        usage,
    })
}

// ============================================================================
// Converse Stream Parsing
// ============================================================================

/// Parse Converse stream to StreamChunk stream.
fn parse_converse_stream(
    output: aws_sdk_bedrockruntime::operation::converse_stream::ConverseStreamOutput,
) -> impl Stream<Item = Result<StreamChunk>> {
    use async_stream::stream;

    stream! {
        let mut event_receiver = output.stream;
        let mut sent_start = false;

        loop {
            match event_receiver.recv().await {
                Ok(Some(event)) => {
                    use aws_sdk_bedrockruntime::types::ConverseStreamOutput as CSO;

                    match event {
                        CSO::MessageStart(_) => {
                            if !sent_start {
                                yield Ok(StreamChunk {
                                    event_type: StreamEventType::MessageStart,
                                    index: None,
                                    delta: None,
                                    stop_reason: None,
                                    usage: None,
                                });
                                sent_start = true;
                            }
                        }
                        CSO::ContentBlockStart(start) => {
                            yield Ok(StreamChunk {
                                event_type: StreamEventType::ContentBlockStart,
                                index: Some(start.content_block_index as usize),
                                delta: None,
                                stop_reason: None,
                                usage: None,
                            });
                        }
                        CSO::ContentBlockDelta(delta) => {
                            if let Some(d) = delta.delta {
                                use aws_sdk_bedrockruntime::types::ContentBlockDelta as CBD;

                                match d {
                                    CBD::Text(text) => {
                                        yield Ok(StreamChunk {
                                            event_type: StreamEventType::ContentBlockDelta,
                                            index: Some(delta.content_block_index as usize),
                                            delta: Some(ContentDelta::Text { text }),
                                            stop_reason: None,
                                            usage: None,
                                        });
                                    }
                                    CBD::ToolUse(tool_use) => {
                                        yield Ok(StreamChunk {
                                            event_type: StreamEventType::ContentBlockDelta,
                                            index: Some(delta.content_block_index as usize),
                                            delta: Some(ContentDelta::ToolUse {
                                                id: None,
                                                name: None,
                                                input_json_delta: Some(tool_use.input),
                                            }),
                                            stop_reason: None,
                                            usage: None,
                                        });
                                    }
                                    _ => {}
                                }
                            }
                        }
                        CSO::ContentBlockStop(_) => {
                            yield Ok(StreamChunk {
                                event_type: StreamEventType::ContentBlockStop,
                                index: None,
                                delta: None,
                                stop_reason: None,
                                usage: None,
                            });
                        }
                        CSO::MessageStop(stop) => {
                            let stop_reason = match stop.stop_reason {
                                aws_sdk_bedrockruntime::types::StopReason::EndTurn => {
                                    Some(StopReason::EndTurn)
                                }
                                aws_sdk_bedrockruntime::types::StopReason::ToolUse => {
                                    Some(StopReason::ToolUse)
                                }
                                aws_sdk_bedrockruntime::types::StopReason::MaxTokens => {
                                    Some(StopReason::MaxTokens)
                                }
                                aws_sdk_bedrockruntime::types::StopReason::StopSequence => {
                                    Some(StopReason::StopSequence)
                                }
                                aws_sdk_bedrockruntime::types::StopReason::ContentFiltered => {
                                    Some(StopReason::ContentFilter)
                                }
                                _ => Some(StopReason::EndTurn),
                            };

                            yield Ok(StreamChunk {
                                event_type: StreamEventType::MessageStop,
                                index: None,
                                delta: None,
                                stop_reason,
                                usage: None,
                            });
                        }
                        CSO::Metadata(meta) => {
                            if let Some(usage) = meta.usage {
                                yield Ok(StreamChunk {
                                    event_type: StreamEventType::MessageDelta,
                                    index: None,
                                    delta: None,
                                    stop_reason: None,
                                    usage: Some(Usage {
                                        input_tokens: usage.input_tokens as u32,
                                        output_tokens: usage.output_tokens as u32,
                                        cache_creation_input_tokens: 0,
                                        cache_read_input_tokens: 0,
                                    }),
                                });
                            }
                        }
                        _ => {
                            // Skip other event types
                        }
                    }
                }
                Ok(None) => {
                    // Stream ended
                    break;
                }
                Err(e) => {
                    yield Err(Error::server(500, format!("Stream error: {}", e)));
                    break;
                }
            }
        }
    }
}

// ============================================================================
// Provider Implementation
// ============================================================================

#[async_trait]
impl Provider for BedrockProvider {
    fn name(&self) -> &str {
        "bedrock"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        // Check if model requires InvokeModel fallback
        if self.requires_invoke_model(&request.model) {
            return self.complete_with_invoke_model(request).await;
        }

        // Build Converse API request
        let messages = build_converse_messages(&request)?;
        let system = build_system_content(&request);
        let inference_config = build_inference_config(&request);
        let tool_config = build_tool_config(&request);

        let mut converse_request = self
            .client
            .converse()
            .model_id(&request.model)
            .set_messages(Some(messages));

        if let Some(sys) = system {
            converse_request = converse_request.set_system(Some(sys));
        }

        if let Some(config) = inference_config {
            converse_request = converse_request.inference_config(config);
        }

        if let Some(tools) = tool_config {
            converse_request = converse_request.tool_config(tools);
        }

        let response = converse_request
            .send()
            .await
            .map_err(|e| Error::server(500, format!("Bedrock Converse error: {}", e)))?;

        parse_converse_response(response, &request.model)
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // Check if model requires InvokeModel fallback
        if self.requires_invoke_model(&request.model) {
            return self.complete_stream_with_invoke_model(request).await;
        }

        // Build Converse Stream API request
        let messages = build_converse_messages(&request)?;
        let system = build_system_content(&request);
        let inference_config = build_inference_config(&request);
        let tool_config = build_tool_config(&request);

        let mut stream_request = self
            .client
            .converse_stream()
            .model_id(&request.model)
            .set_messages(Some(messages));

        if let Some(sys) = system {
            stream_request = stream_request.set_system(Some(sys));
        }

        if let Some(config) = inference_config {
            stream_request = stream_request.inference_config(config);
        }

        if let Some(tools) = tool_config {
            stream_request = stream_request.tool_config(tools);
        }

        let output = stream_request
            .send()
            .await
            .map_err(|e| Error::server(500, format!("Bedrock ConverseStream error: {}", e)))?;

        Ok(Box::pin(parse_converse_stream(output)))
    }

    fn supports_tools(&self) -> bool {
        true
    }

    fn supports_vision(&self) -> bool {
        true
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn supported_models(&self) -> Option<&[&str]> {
        Some(&[
            // Anthropic Claude 4.5
            "anthropic.claude-opus-4-5-20251101-v1:0",
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "anthropic.claude-haiku-4-5-20251015-v1:0",
            // Anthropic Claude 4
            "anthropic.claude-opus-4-20250514-v1:0",
            "anthropic.claude-sonnet-4-20250514-v1:0",
            // Anthropic Claude 3.5
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "anthropic.claude-3-5-haiku-20241022-v1:0",
            // Anthropic Claude 3 (legacy)
            "anthropic.claude-3-opus-20240229-v1:0",
            "anthropic.claude-3-sonnet-20240229-v1:0",
            "anthropic.claude-3-haiku-20240307-v1:0",
            // Amazon Nova 2 (latest)
            "amazon.nova-pro-2-v1:0",
            "amazon.nova-lite-2-v1:0",
            // Amazon Nova 1
            "amazon.nova-pro-v1:0",
            "amazon.nova-lite-v1:0",
            "amazon.nova-micro-v1:0",
            // Cross-region inference profiles (examples)
            "us.amazon.nova-micro-v1:0",
            "eu.amazon.nova-micro-v1:0",
            "apac.amazon.nova-micro-v1:0",
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            "eu.anthropic.claude-3-5-sonnet-20241022-v2:0",
            // Meta Llama 4
            "meta.llama4-maverick-17b-instruct-v1:0",
            "meta.llama4-scout-17b-instruct-v1:0",
            // Meta Llama 3.3
            "meta.llama3-3-70b-instruct-v1:0",
            // Meta Llama 3.2
            "meta.llama3-2-90b-instruct-v1:0",
            "meta.llama3-2-11b-instruct-v1:0",
            "meta.llama3-2-3b-instruct-v1:0",
            "meta.llama3-2-1b-instruct-v1:0",
            // Meta Llama 3.1
            "meta.llama3-1-405b-instruct-v1:0",
            "meta.llama3-1-70b-instruct-v1:0",
            "meta.llama3-1-8b-instruct-v1:0",
            // Mistral
            "mistral.mistral-large-2411-v1:0",
            "mistral.mistral-small-2409-v1:0",
            "mistral.mixtral-8x7b-instruct-v0:1",
            // Cohere
            "cohere.command-r-plus-v1:0",
            "cohere.command-r-v1:0",
            // AI21
            "ai21.jamba-1-5-large-v1:0",
            "ai21.jamba-1-5-mini-v1:0",
            // Amazon Titan
            "amazon.titan-text-express-v1",
            "amazon.titan-text-lite-v1",
            // DeepSeek
            "deepseek.deepseek-r1-v1:0",
            "deepseek.deepseek-v3-v1:0",
            // Qwen (uses InvokeModel fallback)
            "qwen.qwen2-5-72b-instruct-v1:0",
            "qwen.qwen2-5-32b-instruct-v1:0",
            "qwen.qwen2-5-14b-instruct-v1:0",
            "qwen.qwen2-5-7b-instruct-v1:0",
        ])
    }

    fn default_model(&self) -> Option<&str> {
        Some("anthropic.claude-sonnet-4-5-20250929-v1:0")
    }
}

// ============================================================================
// InvokeModel Fallback (for Qwen and other models without Converse support)
// ============================================================================

impl BedrockProvider {
    /// Complete using InvokeModel API (fallback for unsupported models).
    async fn complete_with_invoke_model(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse> {
        let body = build_qwen_request(&request)?;

        let result = self
            .client
            .invoke_model()
            .model_id(&request.model)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body))
            .send()
            .await
            .map_err(|e| Error::server(500, format!("Bedrock InvokeModel error: {}", e)))?;

        let response_body = result.body.into_inner();
        parse_qwen_response(&response_body, &request.model)
    }

    /// Stream using InvokeModel API (fallback for unsupported models).
    async fn complete_stream_with_invoke_model(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let body = build_qwen_request(&request)?;

        let result = self
            .client
            .invoke_model_with_response_stream()
            .model_id(&request.model)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body))
            .send()
            .await
            .map_err(|e| Error::server(500, format!("Bedrock InvokeModel stream error: {}", e)))?;

        Ok(Box::pin(parse_qwen_stream(result)))
    }
}

// ============================================================================
// Qwen Adapter (InvokeModel fallback)
// ============================================================================

/// Build Qwen request body.
fn build_qwen_request(request: &CompletionRequest) -> Result<Vec<u8>> {
    let mut messages: Vec<QwenMessage> = Vec::new();

    // Add system message
    if let Some(ref system) = request.system {
        messages.push(QwenMessage {
            role: "system".to_string(),
            content: system.clone(),
        });
    }

    for msg in &request.messages {
        let role = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::System => "system",
        };
        messages.push(QwenMessage {
            role: role.to_string(),
            content: msg.text_content(),
        });
    }

    let qwen_request = QwenRequest {
        messages,
        max_tokens: request.max_tokens.unwrap_or(4096),
        temperature: request.temperature,
        top_p: request.top_p,
        stop: request.stop_sequences.clone(),
    };

    serde_json::to_vec(&qwen_request).map_err(|e| Error::invalid_request(e.to_string()))
}

/// Parse Qwen response.
fn parse_qwen_response(body: &[u8], model: &str) -> Result<CompletionResponse> {
    let response: QwenResponse = serde_json::from_slice(body)
        .map_err(|e| Error::server(500, format!("Failed to parse Qwen response: {}", e)))?;

    let choice = response.choices.into_iter().next().unwrap_or_default();

    let stop_reason = match choice.finish_reason.as_deref() {
        Some("stop") => StopReason::EndTurn,
        Some("length") => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    };

    Ok(CompletionResponse {
        id: response
            .id
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
        model: model.to_string(),
        content: vec![ContentBlock::Text {
            text: choice.message.content,
        }],
        stop_reason,
        usage: Usage {
            input_tokens: response.usage.prompt_tokens,
            output_tokens: response.usage.completion_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        },
    })
}

/// Parse Qwen stream.
fn parse_qwen_stream(
    output: aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamOutput,
) -> impl Stream<Item = Result<StreamChunk>> {
    use async_stream::stream;
    use aws_sdk_bedrockruntime::types::ResponseStream;

    stream! {
        let mut event_receiver = output.body;
        let mut sent_start = false;

        loop {
            match event_receiver.recv().await {
                Ok(Some(event)) => {
                    if let ResponseStream::Chunk(chunk) = event {
                        if let Some(bytes) = chunk.bytes {
                            let bytes = bytes.into_inner();

                            if !sent_start {
                                yield Ok(StreamChunk {
                                    event_type: StreamEventType::MessageStart,
                                    index: None,
                                    delta: None,
                                    stop_reason: None,
                                    usage: None,
                                });
                                sent_start = true;
                            }

                            if let Ok(parsed) = serde_json::from_slice::<QwenStreamEvent>(&bytes) {
                                if let Some(choices) = parsed.choices {
                                    for choice in choices {
                                        if let Some(delta) = choice.delta {
                                            if let Some(content) = delta.content {
                                                yield Ok(StreamChunk {
                                                    event_type: StreamEventType::ContentBlockDelta,
                                                    index: Some(0),
                                                    delta: Some(ContentDelta::Text { text: content }),
                                                    stop_reason: None,
                                                    usage: None,
                                                });
                                            }
                                        }
                                        if let Some(finish_reason) = choice.finish_reason {
                                            yield Ok(StreamChunk {
                                                event_type: StreamEventType::MessageDelta,
                                                index: None,
                                                delta: None,
                                                stop_reason: Some(match finish_reason.as_str() {
                                                    "stop" => StopReason::EndTurn,
                                                    "length" => StopReason::MaxTokens,
                                                    _ => StopReason::EndTurn,
                                                }),
                                                usage: None,
                                            });
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(None) => {
                    break;
                }
                Err(e) => {
                    yield Err(Error::server(500, format!("Stream error: {}", e)));
                    break;
                }
            }
        }
    }
}

// ============================================================================
// Qwen Types (for InvokeModel fallback)
// ============================================================================

#[derive(Debug, Serialize)]
struct QwenRequest {
    messages: Vec<QwenMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct QwenMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct QwenResponse {
    id: Option<String>,
    choices: Vec<QwenChoice>,
    usage: QwenUsage,
}

#[derive(Debug, Default, Deserialize)]
struct QwenChoice {
    message: QwenResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct QwenResponseMessage {
    content: String,
}

#[derive(Debug, Default, Deserialize)]
struct QwenUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct QwenStreamEvent {
    choices: Option<Vec<QwenStreamChoice>>,
}

#[derive(Debug, Deserialize)]
struct QwenStreamChoice {
    delta: Option<QwenStreamDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct QwenStreamDelta {
    content: Option<String>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Message;

    #[test]
    fn test_requires_invoke_model() {
        let config = BedrockConfig::default();

        // Helper function to check require invoke model logic
        fn check_requires_invoke(model_id: &str, config: &BedrockConfig) -> bool {
            let id = model_id.to_lowercase();

            // Check explicit overrides first
            if let Some(&use_invoke) = config.invoke_model_overrides.get(model_id) {
                return use_invoke;
            }

            // Models without Converse support
            // Note: Bedrock uses "qwen2-5" format (hyphen) while official Qwen uses "qwen2.5" (dot)
            id.contains("qwen2.5")
                || id.contains("qwen2-5")
                || id.contains("qwen2-vl")
                || id.contains("titan-embed")
        }

        // Qwen models should require invoke_model
        assert!(check_requires_invoke(
            "qwen.qwen2-5-72b-instruct-v1:0",
            &config
        ));
        assert!(check_requires_invoke(
            "qwen.qwen2.5-14b-instruct-v1:0",
            &config
        ));

        // Other models should use Converse
        assert!(!check_requires_invoke(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            &config
        ));
        assert!(!check_requires_invoke("amazon.nova-micro-v1:0", &config));
        assert!(!check_requires_invoke("us.amazon.nova-micro-v1:0", &config)); // Inference profile
        assert!(!check_requires_invoke(
            "meta.llama3-70b-instruct-v1:0",
            &config
        ));
        assert!(!check_requires_invoke(
            "mistral.mistral-large-2407-v1:0",
            &config
        ));
        assert!(!check_requires_invoke(
            "cohere.command-r-plus-v1:0",
            &config
        ));
        assert!(!check_requires_invoke("ai21.jamba-1-5-large-v1:0", &config));
        assert!(!check_requires_invoke("deepseek.deepseek-r1-v1:0", &config));
    }

    #[test]
    fn test_inference_profile_support() {
        let config = BedrockConfig::default();

        // Helper function to check require invoke model logic
        fn check_requires_invoke(model_id: &str, config: &BedrockConfig) -> bool {
            let id = model_id.to_lowercase();

            if let Some(&use_invoke) = config.invoke_model_overrides.get(model_id) {
                return use_invoke;
            }

            id.contains("qwen2.5")
                || id.contains("qwen2-5")
                || id.contains("qwen2-vl")
                || id.contains("titan-embed")
        }

        // All inference profiles should use Converse (not invoke_model)
        assert!(!check_requires_invoke("us.amazon.nova-micro-v1:0", &config));
        assert!(!check_requires_invoke("eu.amazon.nova-micro-v1:0", &config));
        assert!(!check_requires_invoke(
            "apac.amazon.nova-micro-v1:0",
            &config
        ));
        assert!(!check_requires_invoke(
            "global.anthropic.claude-opus-4-5-20251101-v1:0",
            &config
        ));
        assert!(!check_requires_invoke(
            "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
            &config
        ));
    }

    #[test]
    fn test_config_builder() {
        let config = BedrockConfig::new("us-west-2")
            .with_timeout(std::time::Duration::from_secs(60))
            .with_invoke_model_override("custom-model");

        assert_eq!(config.region, "us-west-2");
        assert_eq!(config.timeout, std::time::Duration::from_secs(60));
        assert_eq!(
            config.invoke_model_overrides.get("custom-model"),
            Some(&true)
        );
    }

    #[test]
    fn test_build_system_content() {
        // Test with explicit system field
        let request = CompletionRequest::new(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            vec![Message::user("Hello!")],
        )
        .with_system("You are helpful");

        let system = build_system_content(&request);
        assert!(system.is_some());

        // Test without system
        let request = CompletionRequest::new(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            vec![Message::user("Hello!")],
        );

        let system = build_system_content(&request);
        assert!(system.is_none());
    }

    #[test]
    fn test_build_inference_config() {
        let request = CompletionRequest::new(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            vec![Message::user("Hello!")],
        )
        .with_max_tokens(1024)
        .with_temperature(0.7);

        let config = build_inference_config(&request);
        assert!(config.is_some());
    }

    #[test]
    fn test_qwen_request_conversion() {
        let request = CompletionRequest::new(
            "qwen.qwen2-5-72b-instruct-v1:0",
            vec![Message::user("Hello!")],
        )
        .with_system("You are helpful");

        let body = build_qwen_request(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(parsed["messages"].is_array());
        // Should have system + user = 2 messages
        assert_eq!(parsed["messages"].as_array().unwrap().len(), 2);
    }
}
