//! AWS Bedrock provider implementation.
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
//! - **Qwen (Alibaba)**: Qwen 2.5
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
//! use llmkit::providers::bedrock::BedrockProvider;
//!
//! // Using default AWS credentials
//! let provider = BedrockProvider::from_env("us-east-1").await?;
//!
//! // Or with explicit region
//! let provider = BedrockProvider::builder()
//!     .region("us-west-2")
//!     .build()
//!     .await?;
//! ```

use std::collections::HashMap;
use std::pin::Pin;

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_bedrockruntime::primitives::Blob;
use aws_sdk_bedrockruntime::types::ResponseStream;
use aws_sdk_bedrockruntime::Client as BedrockClient;
use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Message, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

/// AWS Bedrock model family.
///
/// Each family has a different request/response format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    /// Anthropic Claude models (Claude 4.5, 4, 3.5, 3)
    Anthropic,
    /// Amazon Nova models (Nova Pro, Lite, Micro, Nova 2)
    Nova,
    /// Meta Llama models (Llama 4, 3.3, 3.2, 3.1, 3)
    Llama,
    /// Mistral AI models
    Mistral,
    /// Cohere Command models
    Cohere,
    /// AI21 Labs models
    AI21,
    /// Amazon Titan models
    Titan,
    /// DeepSeek models (DeepSeek-R1, DeepSeek-V3)
    DeepSeek,
    /// Qwen/Alibaba models
    Qwen,
}

impl ModelFamily {
    /// Detect model family from model ID.
    pub fn from_model_id(model_id: &str) -> Option<Self> {
        let id = model_id.to_lowercase();

        if id.contains("anthropic") || id.contains("claude") {
            Some(ModelFamily::Anthropic)
        } else if id.contains("nova") {
            // Nova must be checked before generic "amazon" check
            Some(ModelFamily::Nova)
        } else if id.contains("meta") || id.contains("llama") {
            Some(ModelFamily::Llama)
        } else if id.contains("mistral") || id.contains("mixtral") {
            Some(ModelFamily::Mistral)
        } else if id.contains("cohere") || id.contains("command") {
            Some(ModelFamily::Cohere)
        } else if id.contains("ai21") || id.contains("jamba") || id.contains("jurassic") {
            Some(ModelFamily::AI21)
        } else if id.contains("titan") {
            Some(ModelFamily::Titan)
        } else if id.contains("deepseek") {
            Some(ModelFamily::DeepSeek)
        } else if id.contains("qwen") {
            Some(ModelFamily::Qwen)
        } else {
            None
        }
    }
}

/// Configuration for Bedrock provider.
#[derive(Debug, Clone)]
pub struct BedrockConfig {
    /// AWS region (e.g., "us-east-1")
    pub region: String,

    /// Request timeout
    pub timeout: std::time::Duration,

    /// Model ID to family overrides (for custom inference profiles)
    pub model_overrides: HashMap<String, ModelFamily>,
}

impl Default for BedrockConfig {
    fn default() -> Self {
        Self {
            region: "us-east-1".to_string(),
            timeout: std::time::Duration::from_secs(120),
            model_overrides: HashMap::new(),
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

    /// Builder: Add a model override.
    pub fn with_model_override(mut self, model_id: impl Into<String>, family: ModelFamily) -> Self {
        self.model_overrides.insert(model_id.into(), family);
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

    /// Add a model family override.
    pub fn model_override(mut self, model_id: impl Into<String>, family: ModelFamily) -> Self {
        self.config.model_overrides.insert(model_id.into(), family);
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
/// Provides access to multiple model families through AWS Bedrock's unified API.
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

    /// Detect the model family for a model ID.
    fn detect_family(&self, model_id: &str) -> Result<ModelFamily> {
        // Check overrides first
        if let Some(family) = self.config.model_overrides.get(model_id) {
            return Ok(*family);
        }

        ModelFamily::from_model_id(model_id).ok_or_else(|| {
            Error::ModelNotFound(format!(
                "Unknown model family for: {}. Use model_override to specify.",
                model_id
            ))
        })
    }

    /// Get the adapter for a model family.
    fn get_adapter(&self, family: ModelFamily) -> Box<dyn ModelAdapter> {
        match family {
            ModelFamily::Anthropic => Box::new(AnthropicAdapter),
            ModelFamily::Nova => Box::new(NovaAdapter),
            ModelFamily::Llama => Box::new(LlamaAdapter),
            ModelFamily::Mistral => Box::new(MistralAdapter),
            ModelFamily::Cohere => Box::new(CohereAdapter),
            ModelFamily::AI21 => Box::new(AI21Adapter),
            ModelFamily::Titan => Box::new(TitanAdapter),
            ModelFamily::DeepSeek => Box::new(DeepSeekAdapter),
            ModelFamily::Qwen => Box::new(QwenAdapter),
        }
    }
}

#[async_trait]
impl Provider for BedrockProvider {
    fn name(&self) -> &str {
        "bedrock"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let family = self.detect_family(&request.model)?;
        let adapter = self.get_adapter(family);

        let body = adapter.convert_request(&request)?;

        let result = self
            .client
            .invoke_model()
            .model_id(&request.model)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body))
            .send()
            .await
            .map_err(|e| Error::server(500, e.to_string()))?;

        let response_body = result.body.into_inner();
        adapter.parse_response(&response_body, &request.model)
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let family = self.detect_family(&request.model)?;
        let adapter = self.get_adapter(family);

        let body = adapter.convert_request(&request)?;

        let result = self
            .client
            .invoke_model_with_response_stream()
            .model_id(&request.model)
            .content_type("application/json")
            .accept("application/json")
            .body(Blob::new(body))
            .send()
            .await
            .map_err(|e| Error::server(500, e.to_string()))?;

        let stream = parse_bedrock_stream(result, family);
        Ok(Box::pin(stream))
    }

    fn supports_tools(&self) -> bool {
        // Only Anthropic Claude models support tools on Bedrock
        true
    }

    fn supports_vision(&self) -> bool {
        // Only Claude 3+ models support vision
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
            // Qwen (Alibaba)
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
// Model Adapters
// ============================================================================

/// Adapter trait for different model families.
///
/// Each model family on Bedrock has its own request/response format.
trait ModelAdapter: Send + Sync {
    /// Convert a unified request to the model-specific format.
    fn convert_request(&self, request: &CompletionRequest) -> Result<Vec<u8>>;

    /// Parse model-specific response to unified format.
    fn parse_response(&self, body: &[u8], model: &str) -> Result<CompletionResponse>;

    /// Parse a stream event to unified format.
    fn parse_stream_event(&self, event: &[u8]) -> Option<StreamChunk>;
}

// ============================================================================
// Anthropic Claude Adapter
// ============================================================================

struct AnthropicAdapter;

impl ModelAdapter for AnthropicAdapter {
    fn convert_request(&self, request: &CompletionRequest) -> Result<Vec<u8>> {
        let mut messages: Vec<BedrockClaudeMessage> = Vec::new();

        // Convert messages
        for msg in &request.messages {
            messages.extend(convert_claude_message(msg));
        }

        // Convert tools
        let tools: Option<Vec<BedrockClaudeTool>> = request.tools.as_ref().map(|tools| {
            tools
                .iter()
                .map(|t| BedrockClaudeTool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.input_schema.clone(),
                })
                .collect()
        });

        let bedrock_request = BedrockClaudeRequest {
            anthropic_version: "bedrock-2023-05-31".to_string(),
            messages,
            system: request.system.clone(),
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            top_p: request.top_p,
            stop_sequences: request.stop_sequences.clone(),
            tools,
        };

        serde_json::to_vec(&bedrock_request).map_err(|e| Error::invalid_request(e.to_string()))
    }

    fn parse_response(&self, body: &[u8], model: &str) -> Result<CompletionResponse> {
        let response: BedrockClaudeResponse = serde_json::from_slice(body)
            .map_err(|e| Error::server(500, format!("Failed to parse response: {}", e)))?;

        let mut content = Vec::new();

        for block in response.content {
            match block {
                BedrockClaudeContentBlock::Text { text } => {
                    content.push(ContentBlock::Text { text });
                }
                BedrockClaudeContentBlock::ToolUse { id, name, input } => {
                    content.push(ContentBlock::ToolUse { id, name, input });
                }
            }
        }

        let stop_reason = match response.stop_reason.as_deref() {
            Some("end_turn") => StopReason::EndTurn,
            Some("max_tokens") => StopReason::MaxTokens,
            Some("tool_use") => StopReason::ToolUse,
            Some("stop_sequence") => StopReason::StopSequence,
            _ => StopReason::EndTurn,
        };

        Ok(CompletionResponse {
            id: response.id,
            model: model.to_string(),
            content,
            stop_reason,
            usage: Usage {
                input_tokens: response.usage.input_tokens,
                output_tokens: response.usage.output_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        })
    }

    fn parse_stream_event(&self, event: &[u8]) -> Option<StreamChunk> {
        let parsed: BedrockClaudeStreamEvent = serde_json::from_slice(event).ok()?;

        match parsed.event_type.as_str() {
            "message_start" => Some(StreamChunk {
                event_type: StreamEventType::MessageStart,
                index: None,
                delta: None,
                stop_reason: None,
                usage: None,
            }),
            "content_block_delta" => {
                if let Some(delta) = parsed.delta {
                    match delta {
                        BedrockClaudeDelta::TextDelta { text } => Some(StreamChunk {
                            event_type: StreamEventType::ContentBlockDelta,
                            index: parsed.index,
                            delta: Some(ContentDelta::TextDelta { text }),
                            stop_reason: None,
                            usage: None,
                        }),
                        BedrockClaudeDelta::InputJsonDelta { partial_json } => Some(StreamChunk {
                            event_type: StreamEventType::ContentBlockDelta,
                            index: parsed.index,
                            delta: Some(ContentDelta::ToolUseDelta {
                                id: None,
                                name: None,
                                input_json_delta: Some(partial_json),
                            }),
                            stop_reason: None,
                            usage: None,
                        }),
                    }
                } else {
                    None
                }
            }
            "message_delta" => {
                let stop_reason = parsed.delta.and_then(|d| {
                    if let BedrockClaudeDelta::TextDelta { .. } = d {
                        None
                    } else {
                        None
                    }
                });
                Some(StreamChunk {
                    event_type: StreamEventType::MessageDelta,
                    index: None,
                    delta: None,
                    stop_reason,
                    usage: parsed.usage.map(|u| Usage {
                        input_tokens: u.input_tokens,
                        output_tokens: u.output_tokens,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                    }),
                })
            }
            "message_stop" => Some(StreamChunk {
                event_type: StreamEventType::MessageStop,
                index: None,
                delta: None,
                stop_reason: Some(StopReason::EndTurn),
                usage: None,
            }),
            _ => None,
        }
    }
}

fn convert_claude_message(message: &Message) -> Vec<BedrockClaudeMessage> {
    let mut result = Vec::new();

    match message.role {
        Role::System => {
            // System messages are handled separately in Bedrock Claude
        }
        Role::User => {
            let content: Vec<BedrockClaudeContent> = message
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => {
                        Some(BedrockClaudeContent::Text { text: text.clone() })
                    }
                    ContentBlock::Image { media_type, data } => Some(BedrockClaudeContent::Image {
                        source: BedrockImageSource {
                            source_type: "base64".to_string(),
                            media_type: media_type.clone(),
                            data: data.clone(),
                        },
                    }),
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => Some(BedrockClaudeContent::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: content.clone(),
                        is_error: Some(*is_error),
                    }),
                    _ => None,
                })
                .collect();

            if !content.is_empty() {
                result.push(BedrockClaudeMessage {
                    role: "user".to_string(),
                    content,
                });
            }
        }
        Role::Assistant => {
            let content: Vec<BedrockClaudeContent> = message
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => {
                        Some(BedrockClaudeContent::Text { text: text.clone() })
                    }
                    ContentBlock::ToolUse { id, name, input } => {
                        Some(BedrockClaudeContent::ToolUse {
                            id: id.clone(),
                            name: name.clone(),
                            input: input.clone(),
                        })
                    }
                    _ => None,
                })
                .collect();

            if !content.is_empty() {
                result.push(BedrockClaudeMessage {
                    role: "assistant".to_string(),
                    content,
                });
            }
        }
    }

    result
}

// ============================================================================
// Llama Adapter
// ============================================================================

struct LlamaAdapter;

impl ModelAdapter for LlamaAdapter {
    fn convert_request(&self, request: &CompletionRequest) -> Result<Vec<u8>> {
        // Llama uses a simple prompt format
        let mut prompt = String::new();

        // Add system prompt
        if let Some(ref system) = request.system {
            prompt.push_str(&format!(
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|>",
                system
            ));
        } else {
            prompt.push_str("<|begin_of_text|>");
        }

        // Add messages
        for msg in &request.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "system",
            };
            let text = msg.text_content();
            prompt.push_str(&format!(
                "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                role, text
            ));
        }

        // Add generation prompt
        prompt.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");

        let llama_request = BedrockLlamaRequest {
            prompt,
            max_gen_len: request.max_tokens.unwrap_or(2048),
            temperature: request.temperature.unwrap_or(0.7),
            top_p: request.top_p.unwrap_or(0.9),
        };

        serde_json::to_vec(&llama_request).map_err(|e| Error::invalid_request(e.to_string()))
    }

    fn parse_response(&self, body: &[u8], model: &str) -> Result<CompletionResponse> {
        let response: BedrockLlamaResponse = serde_json::from_slice(body)
            .map_err(|e| Error::server(500, format!("Failed to parse response: {}", e)))?;

        let stop_reason = match response.stop_reason.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        Ok(CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: model.to_string(),
            content: vec![ContentBlock::Text {
                text: response.generation,
            }],
            stop_reason,
            usage: Usage {
                input_tokens: response.prompt_token_count.unwrap_or(0),
                output_tokens: response.generation_token_count.unwrap_or(0),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        })
    }

    fn parse_stream_event(&self, event: &[u8]) -> Option<StreamChunk> {
        let parsed: BedrockLlamaStreamEvent = serde_json::from_slice(event).ok()?;

        if let Some(generation) = parsed.generation {
            Some(StreamChunk {
                event_type: StreamEventType::ContentBlockDelta,
                index: Some(0),
                delta: Some(ContentDelta::TextDelta { text: generation }),
                stop_reason: None,
                usage: None,
            })
        } else if parsed.stop_reason.is_some() {
            Some(StreamChunk {
                event_type: StreamEventType::MessageStop,
                index: None,
                delta: None,
                stop_reason: Some(StopReason::EndTurn),
                usage: None,
            })
        } else {
            None
        }
    }
}

// ============================================================================
// Mistral Adapter
// ============================================================================

struct MistralAdapter;

impl ModelAdapter for MistralAdapter {
    fn convert_request(&self, request: &CompletionRequest) -> Result<Vec<u8>> {
        // Mistral uses chat format similar to OpenAI
        let mut messages: Vec<BedrockMistralMessage> = Vec::new();

        // Add system as first user message (Mistral doesn't have system role in Bedrock)
        if let Some(ref system) = request.system {
            messages.push(BedrockMistralMessage {
                role: "user".to_string(),
                content: format!("[SYSTEM]: {}", system),
            });
            messages.push(BedrockMistralMessage {
                role: "assistant".to_string(),
                content: "Understood.".to_string(),
            });
        }

        for msg in &request.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => "user",
            };
            messages.push(BedrockMistralMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        let mistral_request = BedrockMistralRequest {
            messages,
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            top_p: request.top_p,
        };

        serde_json::to_vec(&mistral_request).map_err(|e| Error::invalid_request(e.to_string()))
    }

    fn parse_response(&self, body: &[u8], model: &str) -> Result<CompletionResponse> {
        let response: BedrockMistralResponse = serde_json::from_slice(body)
            .map_err(|e| Error::server(500, format!("Failed to parse response: {}", e)))?;

        let output = response.outputs.into_iter().next().unwrap_or_default();

        let stop_reason = match output.stop_reason.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        Ok(CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: model.to_string(),
            content: vec![ContentBlock::Text { text: output.text }],
            stop_reason,
            usage: Usage::default(), // Mistral Bedrock doesn't return usage
        })
    }

    fn parse_stream_event(&self, event: &[u8]) -> Option<StreamChunk> {
        let parsed: BedrockMistralStreamEvent = serde_json::from_slice(event).ok()?;

        if let Some(outputs) = parsed.outputs {
            let text = outputs
                .into_iter()
                .map(|o| o.text)
                .collect::<Vec<_>>()
                .join("");

            if !text.is_empty() {
                return Some(StreamChunk {
                    event_type: StreamEventType::ContentBlockDelta,
                    index: Some(0),
                    delta: Some(ContentDelta::TextDelta { text }),
                    stop_reason: None,
                    usage: None,
                });
            }
        }

        None
    }
}

// ============================================================================
// Cohere Adapter
// ============================================================================

struct CohereAdapter;

impl ModelAdapter for CohereAdapter {
    fn convert_request(&self, request: &CompletionRequest) -> Result<Vec<u8>> {
        let mut chat_history: Vec<BedrockCohereMessage> = Vec::new();
        let mut message = String::new();

        // Convert history
        for msg in &request.messages[..request.messages.len().saturating_sub(1)] {
            let role = match msg.role {
                Role::User => "USER",
                Role::Assistant => "CHATBOT",
                Role::System => "SYSTEM",
            };
            chat_history.push(BedrockCohereMessage {
                role: role.to_string(),
                message: msg.text_content(),
            });
        }

        // Last message is the current query
        if let Some(last) = request.messages.last() {
            message = last.text_content();
        }

        let cohere_request = BedrockCohereRequest {
            message,
            chat_history: if chat_history.is_empty() {
                None
            } else {
                Some(chat_history)
            },
            preamble: request.system.clone(),
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            p: request.top_p,
        };

        serde_json::to_vec(&cohere_request).map_err(|e| Error::invalid_request(e.to_string()))
    }

    fn parse_response(&self, body: &[u8], model: &str) -> Result<CompletionResponse> {
        let response: BedrockCohereResponse = serde_json::from_slice(body)
            .map_err(|e| Error::server(500, format!("Failed to parse response: {}", e)))?;

        let stop_reason = match response.finish_reason.as_deref() {
            Some("COMPLETE") => StopReason::EndTurn,
            Some("MAX_TOKENS") => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        Ok(CompletionResponse {
            id: response
                .generation_id
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            model: model.to_string(),
            content: vec![ContentBlock::Text {
                text: response.text,
            }],
            stop_reason,
            usage: Usage::default(), // Cohere doesn't return usage in Bedrock
        })
    }

    fn parse_stream_event(&self, event: &[u8]) -> Option<StreamChunk> {
        let parsed: BedrockCohereStreamEvent = serde_json::from_slice(event).ok()?;

        if let Some(text) = parsed.text {
            Some(StreamChunk {
                event_type: StreamEventType::ContentBlockDelta,
                index: Some(0),
                delta: Some(ContentDelta::TextDelta { text }),
                stop_reason: None,
                usage: None,
            })
        } else if parsed.is_finished.unwrap_or(false) {
            Some(StreamChunk {
                event_type: StreamEventType::MessageStop,
                index: None,
                delta: None,
                stop_reason: Some(StopReason::EndTurn),
                usage: None,
            })
        } else {
            None
        }
    }
}

// ============================================================================
// AI21 Adapter
// ============================================================================

struct AI21Adapter;

impl ModelAdapter for AI21Adapter {
    fn convert_request(&self, request: &CompletionRequest) -> Result<Vec<u8>> {
        let mut messages: Vec<BedrockAI21Message> = Vec::new();

        // Add system message
        if let Some(ref system) = request.system {
            messages.push(BedrockAI21Message {
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
            messages.push(BedrockAI21Message {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        let ai21_request = BedrockAI21Request {
            messages,
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            top_p: request.top_p,
        };

        serde_json::to_vec(&ai21_request).map_err(|e| Error::invalid_request(e.to_string()))
    }

    fn parse_response(&self, body: &[u8], model: &str) -> Result<CompletionResponse> {
        let response: BedrockAI21Response = serde_json::from_slice(body)
            .map_err(|e| Error::server(500, format!("Failed to parse response: {}", e)))?;

        let choice = response.choices.into_iter().next().unwrap_or_default();

        let stop_reason = match choice.finish_reason.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        Ok(CompletionResponse {
            id: response.id,
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

    fn parse_stream_event(&self, event: &[u8]) -> Option<StreamChunk> {
        let parsed: BedrockAI21StreamEvent = serde_json::from_slice(event).ok()?;

        if let Some(choices) = parsed.choices {
            let text: String = choices
                .into_iter()
                .filter_map(|c| c.delta.content)
                .collect();

            if !text.is_empty() {
                return Some(StreamChunk {
                    event_type: StreamEventType::ContentBlockDelta,
                    index: Some(0),
                    delta: Some(ContentDelta::TextDelta { text }),
                    stop_reason: None,
                    usage: None,
                });
            }
        }

        None
    }
}

// ============================================================================
// Titan Adapter
// ============================================================================

struct TitanAdapter;

impl ModelAdapter for TitanAdapter {
    fn convert_request(&self, request: &CompletionRequest) -> Result<Vec<u8>> {
        // Titan uses a simple text format
        let mut text = String::new();

        if let Some(ref system) = request.system {
            text.push_str(&format!("System: {}\n\n", system));
        }

        for msg in &request.messages {
            let role = match msg.role {
                Role::User => "User",
                Role::Assistant => "Bot",
                Role::System => "System",
            };
            text.push_str(&format!("{}: {}\n", role, msg.text_content()));
        }

        text.push_str("Bot:");

        let titan_request = BedrockTitanRequest {
            input_text: text,
            text_generation_config: TitanTextConfig {
                max_token_count: request.max_tokens.unwrap_or(4096),
                temperature: request.temperature.unwrap_or(0.7),
                top_p: request.top_p.unwrap_or(0.9),
                stop_sequences: request.stop_sequences.clone().unwrap_or_default(),
            },
        };

        serde_json::to_vec(&titan_request).map_err(|e| Error::invalid_request(e.to_string()))
    }

    fn parse_response(&self, body: &[u8], model: &str) -> Result<CompletionResponse> {
        let response: BedrockTitanResponse = serde_json::from_slice(body)
            .map_err(|e| Error::server(500, format!("Failed to parse response: {}", e)))?;

        let result = response.results.into_iter().next().unwrap_or_default();

        let stop_reason = match result.completion_reason.as_deref() {
            Some("FINISH") => StopReason::EndTurn,
            Some("LENGTH") => StopReason::MaxTokens,
            Some("CONTENT_FILTERED") => StopReason::ContentFilter,
            _ => StopReason::EndTurn,
        };

        Ok(CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: model.to_string(),
            content: vec![ContentBlock::Text {
                text: result.output_text,
            }],
            stop_reason,
            usage: Usage {
                input_tokens: response.input_text_token_count.unwrap_or(0),
                output_tokens: result.token_count.unwrap_or(0),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        })
    }

    fn parse_stream_event(&self, event: &[u8]) -> Option<StreamChunk> {
        let parsed: BedrockTitanStreamEvent = serde_json::from_slice(event).ok()?;

        if let Some(text) = parsed.output_text {
            Some(StreamChunk {
                event_type: StreamEventType::ContentBlockDelta,
                index: Some(0),
                delta: Some(ContentDelta::TextDelta { text }),
                stop_reason: None,
                usage: None,
            })
        } else if parsed.completion_reason.is_some() {
            Some(StreamChunk {
                event_type: StreamEventType::MessageStop,
                index: None,
                delta: None,
                stop_reason: Some(StopReason::EndTurn),
                usage: None,
            })
        } else {
            None
        }
    }
}

// ============================================================================
// Nova Adapter (Amazon's foundation models)
// ============================================================================

struct NovaAdapter;

impl ModelAdapter for NovaAdapter {
    fn convert_request(&self, request: &CompletionRequest) -> Result<Vec<u8>> {
        // Nova uses Converse API format (similar to Claude/OpenAI style)
        let mut messages: Vec<NovaMessage> = Vec::new();

        for msg in &request.messages {
            let role = match msg.role {
                Role::User => "user",
                Role::Assistant => "assistant",
                Role::System => continue, // System handled separately
            };

            let content: Vec<NovaContent> = msg
                .content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(NovaContent::Text { text: text.clone() }),
                    ContentBlock::Image { media_type, data } => Some(NovaContent::Image {
                        image: NovaImageSource {
                            format: media_type
                                .split('/')
                                .next_back()
                                .unwrap_or("png")
                                .to_string(),
                            source: NovaImageBytes {
                                bytes: data.clone(),
                            },
                        },
                    }),
                    ContentBlock::ToolUse { id, name, input } => Some(NovaContent::ToolUse {
                        tool_use_id: id.clone(),
                        name: name.clone(),
                        input: input.clone(),
                    }),
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                        is_error,
                    } => Some(NovaContent::ToolResult {
                        tool_use_id: tool_use_id.clone(),
                        content: vec![NovaToolResultContent::Text {
                            text: content.clone(),
                        }],
                        status: if *is_error {
                            "error".to_string()
                        } else {
                            "success".to_string()
                        },
                    }),
                    _ => None,
                })
                .collect();

            if !content.is_empty() {
                messages.push(NovaMessage {
                    role: role.to_string(),
                    content,
                });
            }
        }

        // Convert tools
        let tool_config = request.tools.as_ref().map(|tools| NovaToolConfig {
            tools: tools
                .iter()
                .map(|t| NovaTool {
                    tool_spec: NovaToolSpec {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        input_schema: NovaInputSchema {
                            json: t.input_schema.clone(),
                        },
                    },
                })
                .collect(),
        });

        let nova_request = BedrockNovaRequest {
            messages,
            system: request
                .system
                .as_ref()
                .map(|s| vec![NovaSystemContent { text: s.clone() }]),
            inference_config: NovaInferenceConfig {
                max_tokens: request.max_tokens.unwrap_or(4096),
                temperature: request.temperature,
                top_p: request.top_p,
                stop_sequences: request.stop_sequences.clone(),
            },
            tool_config,
        };

        serde_json::to_vec(&nova_request).map_err(|e| Error::invalid_request(e.to_string()))
    }

    fn parse_response(&self, body: &[u8], model: &str) -> Result<CompletionResponse> {
        let response: BedrockNovaResponse = serde_json::from_slice(body)
            .map_err(|e| Error::server(500, format!("Failed to parse response: {}", e)))?;

        let mut content = Vec::new();

        if let Some(output) = response.output {
            if let Some(message) = output.message {
                for block in message.content {
                    match block {
                        NovaResponseContent::Text { text } => {
                            content.push(ContentBlock::Text { text });
                        }
                        NovaResponseContent::ToolUse {
                            tool_use_id,
                            name,
                            input,
                        } => {
                            content.push(ContentBlock::ToolUse {
                                id: tool_use_id,
                                name,
                                input,
                            });
                        }
                    }
                }
            }
        }

        let stop_reason = match response.stop_reason.as_deref() {
            Some("end_turn") => StopReason::EndTurn,
            Some("max_tokens") => StopReason::MaxTokens,
            Some("tool_use") => StopReason::ToolUse,
            Some("stop_sequence") => StopReason::StopSequence,
            Some("content_filtered") => StopReason::ContentFilter,
            _ => StopReason::EndTurn,
        };

        Ok(CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: model.to_string(),
            content,
            stop_reason,
            usage: Usage {
                input_tokens: response.usage.as_ref().map(|u| u.input_tokens).unwrap_or(0),
                output_tokens: response
                    .usage
                    .as_ref()
                    .map(|u| u.output_tokens)
                    .unwrap_or(0),
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        })
    }

    fn parse_stream_event(&self, event: &[u8]) -> Option<StreamChunk> {
        let parsed: BedrockNovaStreamEvent = serde_json::from_slice(event).ok()?;

        if let Some(content_block_delta) = parsed.content_block_delta {
            if let Some(delta) = content_block_delta.delta {
                match delta {
                    NovaStreamDelta::Text { text } => {
                        return Some(StreamChunk {
                            event_type: StreamEventType::ContentBlockDelta,
                            index: Some(content_block_delta.content_block_index.unwrap_or(0)),
                            delta: Some(ContentDelta::TextDelta { text }),
                            stop_reason: None,
                            usage: None,
                        });
                    }
                    NovaStreamDelta::ToolUse { input } => {
                        return Some(StreamChunk {
                            event_type: StreamEventType::ContentBlockDelta,
                            index: Some(content_block_delta.content_block_index.unwrap_or(0)),
                            delta: Some(ContentDelta::ToolUseDelta {
                                id: None,
                                name: None,
                                input_json_delta: Some(input),
                            }),
                            stop_reason: None,
                            usage: None,
                        });
                    }
                }
            }
        }

        if let Some(message_stop) = parsed.message_stop {
            return Some(StreamChunk {
                event_type: StreamEventType::MessageStop,
                index: None,
                delta: None,
                stop_reason: match message_stop.stop_reason.as_deref() {
                    Some("end_turn") => Some(StopReason::EndTurn),
                    Some("max_tokens") => Some(StopReason::MaxTokens),
                    Some("tool_use") => Some(StopReason::ToolUse),
                    _ => Some(StopReason::EndTurn),
                },
                usage: None,
            });
        }

        if let Some(metadata) = parsed.metadata {
            if let Some(usage) = metadata.usage {
                return Some(StreamChunk {
                    event_type: StreamEventType::MessageDelta,
                    index: None,
                    delta: None,
                    stop_reason: None,
                    usage: Some(Usage {
                        input_tokens: usage.input_tokens,
                        output_tokens: usage.output_tokens,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                    }),
                });
            }
        }

        None
    }
}

// ============================================================================
// DeepSeek Adapter
// ============================================================================

struct DeepSeekAdapter;

impl ModelAdapter for DeepSeekAdapter {
    fn convert_request(&self, request: &CompletionRequest) -> Result<Vec<u8>> {
        // DeepSeek uses OpenAI-compatible format on Bedrock
        let mut messages: Vec<DeepSeekMessage> = Vec::new();

        // Add system message
        if let Some(ref system) = request.system {
            messages.push(DeepSeekMessage {
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
            messages.push(DeepSeekMessage {
                role: role.to_string(),
                content: msg.text_content(),
            });
        }

        let deepseek_request = BedrockDeepSeekRequest {
            messages,
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop_sequences.clone(),
        };

        serde_json::to_vec(&deepseek_request).map_err(|e| Error::invalid_request(e.to_string()))
    }

    fn parse_response(&self, body: &[u8], model: &str) -> Result<CompletionResponse> {
        let response: BedrockDeepSeekResponse = serde_json::from_slice(body)
            .map_err(|e| Error::server(500, format!("Failed to parse response: {}", e)))?;

        let choice = response.choices.into_iter().next().unwrap_or_default();

        let stop_reason = match choice.finish_reason.as_deref() {
            Some("stop") => StopReason::EndTurn,
            Some("length") => StopReason::MaxTokens,
            _ => StopReason::EndTurn,
        };

        // DeepSeek-R1 may include thinking in a special format
        let text = choice.message.content;

        Ok(CompletionResponse {
            id: response
                .id
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string()),
            model: model.to_string(),
            content: vec![ContentBlock::Text { text }],
            stop_reason,
            usage: Usage {
                input_tokens: response.usage.prompt_tokens,
                output_tokens: response.usage.completion_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        })
    }

    fn parse_stream_event(&self, event: &[u8]) -> Option<StreamChunk> {
        let parsed: BedrockDeepSeekStreamEvent = serde_json::from_slice(event).ok()?;

        if let Some(choices) = parsed.choices {
            for choice in choices {
                if let Some(delta) = choice.delta {
                    if let Some(content) = delta.content {
                        return Some(StreamChunk {
                            event_type: StreamEventType::ContentBlockDelta,
                            index: Some(0),
                            delta: Some(ContentDelta::TextDelta { text: content }),
                            stop_reason: None,
                            usage: None,
                        });
                    }
                }
                if let Some(finish_reason) = choice.finish_reason {
                    return Some(StreamChunk {
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

        None
    }
}

// ============================================================================
// Qwen Adapter (Alibaba)
// ============================================================================

struct QwenAdapter;

impl ModelAdapter for QwenAdapter {
    fn convert_request(&self, request: &CompletionRequest) -> Result<Vec<u8>> {
        // Qwen uses OpenAI-compatible format on Bedrock
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

        let qwen_request = BedrockQwenRequest {
            messages,
            max_tokens: request.max_tokens.unwrap_or(4096),
            temperature: request.temperature,
            top_p: request.top_p,
            stop: request.stop_sequences.clone(),
        };

        serde_json::to_vec(&qwen_request).map_err(|e| Error::invalid_request(e.to_string()))
    }

    fn parse_response(&self, body: &[u8], model: &str) -> Result<CompletionResponse> {
        let response: BedrockQwenResponse = serde_json::from_slice(body)
            .map_err(|e| Error::server(500, format!("Failed to parse response: {}", e)))?;

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

    fn parse_stream_event(&self, event: &[u8]) -> Option<StreamChunk> {
        let parsed: BedrockQwenStreamEvent = serde_json::from_slice(event).ok()?;

        if let Some(choices) = parsed.choices {
            for choice in choices {
                if let Some(delta) = choice.delta {
                    if let Some(content) = delta.content {
                        return Some(StreamChunk {
                            event_type: StreamEventType::ContentBlockDelta,
                            index: Some(0),
                            delta: Some(ContentDelta::TextDelta { text: content }),
                            stop_reason: None,
                            usage: None,
                        });
                    }
                }
                if let Some(finish_reason) = choice.finish_reason {
                    return Some(StreamChunk {
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

        None
    }
}

// ============================================================================
// Stream Parser
// ============================================================================

fn parse_bedrock_stream(
    output: aws_sdk_bedrockruntime::operation::invoke_model_with_response_stream::InvokeModelWithResponseStreamOutput,
    family: ModelFamily,
) -> impl Stream<Item = Result<StreamChunk>> {
    use async_stream::stream;

    stream! {
        let adapter: Box<dyn ModelAdapter> = match family {
            ModelFamily::Anthropic => Box::new(AnthropicAdapter),
            ModelFamily::Nova => Box::new(NovaAdapter),
            ModelFamily::Llama => Box::new(LlamaAdapter),
            ModelFamily::Mistral => Box::new(MistralAdapter),
            ModelFamily::Cohere => Box::new(CohereAdapter),
            ModelFamily::AI21 => Box::new(AI21Adapter),
            ModelFamily::Titan => Box::new(TitanAdapter),
            ModelFamily::DeepSeek => Box::new(DeepSeekAdapter),
            ModelFamily::Qwen => Box::new(QwenAdapter),
        };

        let mut event_receiver = output.body;
        let mut sent_start = false;

        loop {
            match event_receiver.recv().await {
                Ok(Some(event)) => {
                    match event {
                        ResponseStream::Chunk(chunk) => {
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

                                if let Some(chunk) = adapter.parse_stream_event(&bytes) {
                                    yield Ok(chunk);
                                }
                            }
                        }
                        _ => {
                            // Other event types we don't handle
                        }
                    }
                }
                Ok(None) => {
                    // Stream ended
                    break;
                }
                Err(e) => {
                    yield Ok(StreamChunk {
                        event_type: StreamEventType::Error,
                        index: None,
                        delta: None,
                        stop_reason: None,
                        usage: None,
                    });
                    yield Err(Error::server(500, e.to_string()));
                    break;
                }
            }
        }
    }
}

// ============================================================================
// Bedrock API Types
// ============================================================================

// Claude types
#[derive(Debug, Serialize)]
struct BedrockClaudeRequest {
    anthropic_version: String,
    messages: Vec<BedrockClaudeMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<BedrockClaudeTool>>,
}

#[derive(Debug, Serialize)]
struct BedrockClaudeMessage {
    role: String,
    content: Vec<BedrockClaudeContent>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum BedrockClaudeContent {
    Text {
        text: String,
    },
    Image {
        source: BedrockImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Serialize)]
struct BedrockImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct BedrockClaudeTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Debug, Deserialize)]
struct BedrockClaudeResponse {
    id: String,
    content: Vec<BedrockClaudeContentBlock>,
    stop_reason: Option<String>,
    usage: BedrockClaudeUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum BedrockClaudeContentBlock {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Deserialize)]
struct BedrockClaudeUsage {
    input_tokens: u32,
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct BedrockClaudeStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    index: Option<usize>,
    delta: Option<BedrockClaudeDelta>,
    usage: Option<BedrockClaudeUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum BedrockClaudeDelta {
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
}

// Llama types
#[derive(Debug, Serialize)]
struct BedrockLlamaRequest {
    prompt: String,
    max_gen_len: u32,
    temperature: f32,
    top_p: f32,
}

#[derive(Debug, Deserialize)]
struct BedrockLlamaResponse {
    generation: String,
    stop_reason: Option<String>,
    prompt_token_count: Option<u32>,
    generation_token_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct BedrockLlamaStreamEvent {
    generation: Option<String>,
    stop_reason: Option<String>,
}

// Mistral types
#[derive(Debug, Serialize)]
struct BedrockMistralRequest {
    messages: Vec<BedrockMistralMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Debug, Serialize)]
struct BedrockMistralMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct BedrockMistralResponse {
    outputs: Vec<BedrockMistralOutput>,
}

#[derive(Debug, Default, Deserialize)]
struct BedrockMistralOutput {
    text: String,
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BedrockMistralStreamEvent {
    outputs: Option<Vec<BedrockMistralOutput>>,
}

// Cohere types
#[derive(Debug, Serialize)]
struct BedrockCohereRequest {
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    chat_history: Option<Vec<BedrockCohereMessage>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    preamble: Option<String>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    p: Option<f32>,
}

#[derive(Debug, Serialize)]
struct BedrockCohereMessage {
    role: String,
    message: String,
}

#[derive(Debug, Deserialize)]
struct BedrockCohereResponse {
    text: String,
    generation_id: Option<String>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BedrockCohereStreamEvent {
    text: Option<String>,
    is_finished: Option<bool>,
}

// AI21 types
#[derive(Debug, Serialize)]
struct BedrockAI21Request {
    messages: Vec<BedrockAI21Message>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Debug, Serialize)]
struct BedrockAI21Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct BedrockAI21Response {
    id: String,
    choices: Vec<BedrockAI21Choice>,
    usage: BedrockAI21Usage,
}

#[derive(Debug, Default, Deserialize)]
struct BedrockAI21Choice {
    message: BedrockAI21ChoiceMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct BedrockAI21ChoiceMessage {
    content: String,
}

#[derive(Debug, Default, Deserialize)]
struct BedrockAI21Usage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct BedrockAI21StreamEvent {
    choices: Option<Vec<BedrockAI21StreamChoice>>,
}

#[derive(Debug, Deserialize)]
struct BedrockAI21StreamChoice {
    delta: BedrockAI21Delta,
}

#[derive(Debug, Deserialize)]
struct BedrockAI21Delta {
    content: Option<String>,
}

// Titan types
#[derive(Debug, Serialize)]
struct BedrockTitanRequest {
    #[serde(rename = "inputText")]
    input_text: String,
    #[serde(rename = "textGenerationConfig")]
    text_generation_config: TitanTextConfig,
}

#[derive(Debug, Serialize)]
struct TitanTextConfig {
    #[serde(rename = "maxTokenCount")]
    max_token_count: u32,
    temperature: f32,
    #[serde(rename = "topP")]
    top_p: f32,
    #[serde(rename = "stopSequences")]
    stop_sequences: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct BedrockTitanResponse {
    #[serde(rename = "inputTextTokenCount")]
    input_text_token_count: Option<u32>,
    results: Vec<BedrockTitanResult>,
}

#[derive(Debug, Default, Deserialize)]
struct BedrockTitanResult {
    #[serde(rename = "outputText")]
    output_text: String,
    #[serde(rename = "tokenCount")]
    token_count: Option<u32>,
    #[serde(rename = "completionReason")]
    completion_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BedrockTitanStreamEvent {
    #[serde(rename = "outputText")]
    output_text: Option<String>,
    #[serde(rename = "completionReason")]
    completion_reason: Option<String>,
}

// Nova types (Amazon's foundation models)
#[derive(Debug, Serialize)]
struct BedrockNovaRequest {
    messages: Vec<NovaMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<Vec<NovaSystemContent>>,
    #[serde(rename = "inferenceConfig")]
    inference_config: NovaInferenceConfig,
    #[serde(rename = "toolConfig", skip_serializing_if = "Option::is_none")]
    tool_config: Option<NovaToolConfig>,
}

#[derive(Debug, Serialize)]
struct NovaMessage {
    role: String,
    content: Vec<NovaContent>,
}

#[derive(Debug, Serialize)]
struct NovaSystemContent {
    text: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum NovaContent {
    Text {
        text: String,
    },
    Image {
        image: NovaImageSource,
    },
    #[serde(rename = "toolUse")]
    ToolUse {
        #[serde(rename = "toolUseId")]
        tool_use_id: String,
        name: String,
        input: Value,
    },
    #[serde(rename = "toolResult")]
    ToolResult {
        #[serde(rename = "toolUseId")]
        tool_use_id: String,
        content: Vec<NovaToolResultContent>,
        status: String,
    },
}

#[derive(Debug, Serialize)]
struct NovaImageSource {
    format: String,
    source: NovaImageBytes,
}

#[derive(Debug, Serialize)]
struct NovaImageBytes {
    bytes: String,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum NovaToolResultContent {
    Text { text: String },
}

#[derive(Debug, Serialize)]
struct NovaInferenceConfig {
    #[serde(rename = "maxTokens")]
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(rename = "topP", skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(rename = "stopSequences", skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct NovaToolConfig {
    tools: Vec<NovaTool>,
}

#[derive(Debug, Serialize)]
struct NovaTool {
    #[serde(rename = "toolSpec")]
    tool_spec: NovaToolSpec,
}

#[derive(Debug, Serialize)]
struct NovaToolSpec {
    name: String,
    description: String,
    #[serde(rename = "inputSchema")]
    input_schema: NovaInputSchema,
}

#[derive(Debug, Serialize)]
struct NovaInputSchema {
    json: Value,
}

#[derive(Debug, Deserialize)]
struct BedrockNovaResponse {
    output: Option<NovaOutput>,
    #[serde(rename = "stopReason")]
    stop_reason: Option<String>,
    usage: Option<NovaUsage>,
}

#[derive(Debug, Deserialize)]
struct NovaOutput {
    message: Option<NovaResponseMessage>,
}

#[derive(Debug, Deserialize)]
struct NovaResponseMessage {
    content: Vec<NovaResponseContent>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum NovaResponseContent {
    Text {
        text: String,
    },
    #[serde(rename = "toolUse")]
    ToolUse {
        #[serde(rename = "toolUseId")]
        tool_use_id: String,
        name: String,
        input: Value,
    },
}

#[derive(Debug, Deserialize)]
struct NovaUsage {
    #[serde(rename = "inputTokens")]
    input_tokens: u32,
    #[serde(rename = "outputTokens")]
    output_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct BedrockNovaStreamEvent {
    #[serde(rename = "contentBlockDelta")]
    content_block_delta: Option<NovaContentBlockDelta>,
    #[serde(rename = "messageStop")]
    message_stop: Option<NovaMessageStop>,
    metadata: Option<NovaMetadata>,
}

#[derive(Debug, Deserialize)]
struct NovaContentBlockDelta {
    #[serde(rename = "contentBlockIndex")]
    content_block_index: Option<usize>,
    delta: Option<NovaStreamDelta>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
enum NovaStreamDelta {
    Text {
        text: String,
    },
    #[serde(rename = "toolUse")]
    ToolUse {
        input: String,
    },
}

#[derive(Debug, Deserialize)]
struct NovaMessageStop {
    #[serde(rename = "stopReason")]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct NovaMetadata {
    usage: Option<NovaUsage>,
}

// DeepSeek types
#[derive(Debug, Serialize)]
struct BedrockDeepSeekRequest {
    messages: Vec<DeepSeekMessage>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
}

#[derive(Debug, Serialize)]
struct DeepSeekMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct BedrockDeepSeekResponse {
    id: Option<String>,
    choices: Vec<DeepSeekChoice>,
    usage: DeepSeekUsage,
}

#[derive(Debug, Default, Deserialize)]
struct DeepSeekChoice {
    message: DeepSeekResponseMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct DeepSeekResponseMessage {
    content: String,
}

#[derive(Debug, Default, Deserialize)]
struct DeepSeekUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct BedrockDeepSeekStreamEvent {
    choices: Option<Vec<DeepSeekStreamChoice>>,
}

#[derive(Debug, Deserialize)]
struct DeepSeekStreamChoice {
    delta: Option<DeepSeekStreamDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct DeepSeekStreamDelta {
    content: Option<String>,
}

// Qwen types (Alibaba)
#[derive(Debug, Serialize)]
struct BedrockQwenRequest {
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
struct BedrockQwenResponse {
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
struct BedrockQwenStreamEvent {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_family_detection() {
        // Claude models
        assert_eq!(
            ModelFamily::from_model_id("anthropic.claude-3-5-sonnet-20241022-v2:0"),
            Some(ModelFamily::Anthropic)
        );
        assert_eq!(
            ModelFamily::from_model_id("anthropic.claude-opus-4-5-20251101-v1:0"),
            Some(ModelFamily::Anthropic)
        );

        // Nova models
        assert_eq!(
            ModelFamily::from_model_id("amazon.nova-pro-v1:0"),
            Some(ModelFamily::Nova)
        );
        assert_eq!(
            ModelFamily::from_model_id("amazon.nova-lite-2-v1:0"),
            Some(ModelFamily::Nova)
        );

        // Llama models
        assert_eq!(
            ModelFamily::from_model_id("meta.llama3-70b-instruct-v1:0"),
            Some(ModelFamily::Llama)
        );
        assert_eq!(
            ModelFamily::from_model_id("meta.llama4-maverick-17b-instruct-v1:0"),
            Some(ModelFamily::Llama)
        );

        // Mistral models
        assert_eq!(
            ModelFamily::from_model_id("mistral.mistral-large-2407-v1:0"),
            Some(ModelFamily::Mistral)
        );

        // Cohere models
        assert_eq!(
            ModelFamily::from_model_id("cohere.command-r-plus-v1:0"),
            Some(ModelFamily::Cohere)
        );

        // AI21 models
        assert_eq!(
            ModelFamily::from_model_id("ai21.jamba-1-5-large-v1:0"),
            Some(ModelFamily::AI21)
        );

        // Titan models
        assert_eq!(
            ModelFamily::from_model_id("amazon.titan-text-express-v1"),
            Some(ModelFamily::Titan)
        );

        // DeepSeek models
        assert_eq!(
            ModelFamily::from_model_id("deepseek.deepseek-r1-v1:0"),
            Some(ModelFamily::DeepSeek)
        );

        // Qwen models
        assert_eq!(
            ModelFamily::from_model_id("qwen.qwen2-5-72b-instruct-v1:0"),
            Some(ModelFamily::Qwen)
        );

        // Unknown model
        assert_eq!(ModelFamily::from_model_id("unknown-model"), None);
    }

    #[test]
    fn test_config_builder() {
        let config = BedrockConfig::new("us-west-2")
            .with_timeout(std::time::Duration::from_secs(60))
            .with_model_override("custom-model", ModelFamily::Anthropic);

        assert_eq!(config.region, "us-west-2");
        assert_eq!(config.timeout, std::time::Duration::from_secs(60));
        assert_eq!(
            config.model_overrides.get("custom-model"),
            Some(&ModelFamily::Anthropic)
        );
    }

    #[test]
    fn test_claude_request_conversion() {
        let adapter = AnthropicAdapter;
        let request = CompletionRequest::new(
            "anthropic.claude-3-5-sonnet-20241022-v2:0",
            vec![Message::user("Hello!")],
        )
        .with_system("You are helpful");

        let body = adapter.convert_request(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert_eq!(parsed["anthropic_version"], "bedrock-2023-05-31");
        assert_eq!(parsed["system"], "You are helpful");
        assert!(parsed["messages"].is_array());
    }

    #[test]
    fn test_llama_request_conversion() {
        let adapter = LlamaAdapter;
        let request = CompletionRequest::new(
            "meta.llama3-70b-instruct-v1:0",
            vec![Message::user("Hello!")],
        )
        .with_system("You are helpful");

        let body = adapter.convert_request(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(parsed["prompt"].as_str().unwrap().contains("system"));
        assert!(parsed["prompt"].as_str().unwrap().contains("Hello!"));
    }

    #[test]
    fn test_mistral_request_conversion() {
        let adapter = MistralAdapter;
        let request = CompletionRequest::new(
            "mistral.mistral-large-2407-v1:0",
            vec![Message::user("Hello!")],
        )
        .with_system("You are helpful")
        .with_max_tokens(1024);

        let body = adapter.convert_request(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(parsed["messages"].is_array());
        assert_eq!(parsed["max_tokens"], 1024);
    }

    #[test]
    fn test_cohere_request_conversion() {
        let adapter = CohereAdapter;
        let request = CompletionRequest::new(
            "cohere.command-r-plus-v1:0",
            vec![
                Message::user("Hello"),
                Message::assistant("Hi!"),
                Message::user("How are you?"),
            ],
        )
        .with_system("Be helpful");

        let body = adapter.convert_request(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        // Last message should be the current query
        assert_eq!(parsed["message"], "How are you?");
        // System should be preamble
        assert_eq!(parsed["preamble"], "Be helpful");
        // Chat history should have 2 messages
        assert_eq!(parsed["chat_history"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_titan_request_conversion() {
        let adapter = TitanAdapter;
        let request = CompletionRequest::new(
            "amazon.titan-text-express-v1",
            vec![Message::user("Hello!")],
        )
        .with_system("You are helpful")
        .with_max_tokens(1024)
        .with_temperature(0.7);

        let body = adapter.convert_request(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(parsed["inputText"].as_str().unwrap().contains("Hello!"));
        assert_eq!(parsed["textGenerationConfig"]["maxTokenCount"], 1024);
        assert_eq!(parsed["textGenerationConfig"]["temperature"], 0.7);
    }

    #[test]
    fn test_nova_request_conversion() {
        let adapter = NovaAdapter;
        let request = CompletionRequest::new("amazon.nova-pro-v1:0", vec![Message::user("Hello!")])
            .with_system("You are helpful")
            .with_max_tokens(1024);

        let body = adapter.convert_request(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(parsed["messages"].is_array());
        assert!(parsed["system"].is_array());
        assert_eq!(parsed["inferenceConfig"]["maxTokens"], 1024);
    }

    #[test]
    fn test_deepseek_request_conversion() {
        let adapter = DeepSeekAdapter;
        let request =
            CompletionRequest::new("deepseek.deepseek-r1-v1:0", vec![Message::user("Hello!")])
                .with_system("You are helpful")
                .with_max_tokens(2048);

        let body = adapter.convert_request(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(parsed["messages"].is_array());
        assert_eq!(parsed["max_tokens"], 2048);
    }

    #[test]
    fn test_qwen_request_conversion() {
        let adapter = QwenAdapter;
        let request = CompletionRequest::new(
            "qwen.qwen2-5-72b-instruct-v1:0",
            vec![Message::user("Hello!")],
        )
        .with_system("You are helpful");

        let body = adapter.convert_request(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(parsed["messages"].is_array());
        // Should have system + user = 2 messages
        assert_eq!(parsed["messages"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_claude_response_parsing() {
        let adapter = AnthropicAdapter;
        let response_json = r#"{
            "id": "msg_123",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 20}
        }"#;

        let result = adapter
            .parse_response(
                response_json.as_bytes(),
                "anthropic.claude-3-5-sonnet-20241022-v2:0",
            )
            .unwrap();

        assert_eq!(result.id, "msg_123");
        assert_eq!(result.content.len(), 1);
        assert!(matches!(result.stop_reason, StopReason::EndTurn));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 20);
    }

    #[test]
    fn test_llama_response_parsing() {
        let adapter = LlamaAdapter;
        let response_json = r#"{
            "generation": "Hello there!",
            "stop_reason": "stop",
            "prompt_token_count": 10,
            "generation_token_count": 20
        }"#;

        let result = adapter
            .parse_response(response_json.as_bytes(), "meta.llama3-70b-instruct-v1:0")
            .unwrap();

        assert_eq!(result.content.len(), 1);
        if let ContentBlock::Text { text } = &result.content[0] {
            assert_eq!(text, "Hello there!");
        }
        assert!(matches!(result.stop_reason, StopReason::EndTurn));
        assert_eq!(result.usage.input_tokens, 10);
        assert_eq!(result.usage.output_tokens, 20);
    }

    #[test]
    fn test_model_family_case_insensitive() {
        // Test case insensitivity
        assert_eq!(
            ModelFamily::from_model_id("ANTHROPIC.CLAUDE-3-5-SONNET"),
            Some(ModelFamily::Anthropic)
        );
        assert_eq!(
            ModelFamily::from_model_id("Meta.LLAMA3-70b"),
            Some(ModelFamily::Llama)
        );
        assert_eq!(
            ModelFamily::from_model_id("DEEPSEEK.r1"),
            Some(ModelFamily::DeepSeek)
        );
    }

    #[test]
    fn test_ai21_request_conversion() {
        let adapter = AI21Adapter;
        let request =
            CompletionRequest::new("ai21.jamba-1-5-large-v1:0", vec![Message::user("Hello!")])
                .with_system("You are helpful")
                .with_max_tokens(1024);

        let body = adapter.convert_request(&request).unwrap();
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        assert!(parsed["messages"].is_array());
        assert_eq!(parsed["max_tokens"], 1024);
    }
}
