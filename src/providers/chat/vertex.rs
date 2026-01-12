//! Google Vertex AI provider implementation.
//!
//! This module provides access to Google's Gemini models via Vertex AI,
//! Google Cloud's enterprise ML platform.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::VertexProvider;
//!
//! // From environment variables with automatic credential discovery
//! let provider = VertexProvider::from_env().await?;
//!
//! // With explicit service account file
//! let provider = VertexProvider::from_service_account_file(
//!     "/path/to/service-account.json",
//!     "my-project-id",
//!     "us-central1",
//! ).await?;
//! ```
//!
//! # Authentication
//!
//! Vertex AI uses Google Cloud Application Default Credentials (ADC).
//! The provider automatically discovers credentials in this priority order:
//!
//! 1. `GOOGLE_APPLICATION_CREDENTIALS` environment variable pointing to a service account JSON file
//! 2. `~/.config/gcloud/application_default_credentials.json` (from `gcloud auth application-default login`)
//! 3. GCP Metadata Server (automatic on Compute Engine, Cloud Run, GKE)
//! 4. gcloud CLI (fallback to `gcloud auth print-access-token`)
//!
//! Tokens are automatically refreshed before expiry - no manual token management required.
//!
//! # Environment Variables
//!
//! - `GOOGLE_CLOUD_PROJECT` or `VERTEX_PROJECT` - Your GCP project ID
//! - `GOOGLE_CLOUD_LOCATION` or `VERTEX_LOCATION` - Region (default: us-central1)
//! - `GOOGLE_APPLICATION_CREDENTIALS` - (Optional) Path to service account JSON file
//!
//! # Supported Models
//!
//! - `gemini-2.0-flash-exp` - Latest experimental
//! - `gemini-1.5-pro` - Best for complex tasks
//! - `gemini-1.5-flash` - Fast and efficient

use std::path::Path;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use gcp_auth::TokenProvider;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, ContentDelta, Message, Role, StopReason,
    StreamChunk, StreamEventType, Usage,
};

/// OAuth scopes required for Vertex AI API access.
const VERTEX_SCOPES: &[&str] = &["https://www.googleapis.com/auth/cloud-platform"];

/// Vertex AI provider configuration.
#[derive(Clone)]
pub struct VertexConfig {
    /// GCP project ID.
    pub project_id: String,
    /// GCP location/region (e.g., "us-central1").
    pub location: String,
    /// Token provider for automatic credential management.
    token_provider: Arc<dyn TokenProvider>,
    /// Publisher name ("google", "anthropic", "deepseek", "meta", "mistralai", "ai21labs").
    pub publisher: String,
    /// Request timeout.
    pub timeout: std::time::Duration,
    /// Default model to use (if None, uses provider default).
    pub default_model: Option<String>,
}

impl std::fmt::Debug for VertexConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VertexConfig")
            .field("project_id", &self.project_id)
            .field("location", &self.location)
            .field("publisher", &self.publisher)
            .field("timeout", &self.timeout)
            .field("default_model", &self.default_model)
            .field("token_provider", &"<TokenProvider>")
            .finish()
    }
}

impl VertexConfig {
    /// Create configuration from environment variables with automatic credential discovery.
    ///
    /// This is the recommended way to create a Vertex AI configuration.
    /// It automatically discovers credentials from:
    ///
    /// 1. `GOOGLE_APPLICATION_CREDENTIALS` - Service account JSON file path
    /// 2. `~/.config/gcloud/application_default_credentials.json` - ADC from gcloud CLI
    /// 3. GCP Metadata Server - For workloads running on GCP
    /// 4. gcloud CLI - Fallback to `gcloud auth print-access-token`
    ///
    /// # Environment Variables
    ///
    /// - `GOOGLE_CLOUD_PROJECT` or `VERTEX_PROJECT` - Required: Your GCP project ID
    /// - `GOOGLE_CLOUD_LOCATION` or `VERTEX_LOCATION` - Optional: Region (default: us-central1)
    ///
    /// # Example
    ///
    /// ```ignore
    /// // First run: gcloud auth application-default login
    /// // Or set GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    ///
    /// let config = VertexConfig::from_env().await?;
    /// ```
    pub async fn from_env() -> Result<Self> {
        let project_id = std::env::var("GOOGLE_CLOUD_PROJECT")
            .or_else(|_| std::env::var("VERTEX_PROJECT"))
            .map_err(|_| {
                Error::config(
                    "GOOGLE_CLOUD_PROJECT or VERTEX_PROJECT environment variable not set. \
                     Set one of these to your GCP project ID.",
                )
            })?;

        let location = std::env::var("GOOGLE_CLOUD_LOCATION")
            .or_else(|_| std::env::var("VERTEX_LOCATION"))
            .unwrap_or_else(|_| "us-central1".to_string());

        // Use gcp_auth for automatic credential discovery and token refresh
        let token_provider = gcp_auth::provider().await.map_err(|e| {
            Error::config(format!(
                "Failed to initialize GCP authentication: {}. \
                 Run 'gcloud auth application-default login' or set \
                 GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json",
                e
            ))
        })?;

        Ok(Self {
            project_id,
            location,
            token_provider,
            publisher: "google".to_string(),
            timeout: std::time::Duration::from_secs(300),
            default_model: None,
        })
    }

    /// Create configuration from a service account JSON file.
    ///
    /// Use this when you want to explicitly specify credentials rather than
    /// relying on automatic discovery.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the service account JSON file
    /// * `project_id` - Your GCP project ID
    /// * `location` - GCP region (e.g., "us-central1")
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = VertexConfig::from_service_account_file(
    ///     "/path/to/service-account.json",
    ///     "my-project-id",
    ///     "us-central1",
    /// ).await?;
    /// ```
    pub async fn from_service_account_file(
        path: impl AsRef<Path>,
        project_id: impl Into<String>,
        location: impl Into<String>,
    ) -> Result<Self> {
        let service_account =
            gcp_auth::CustomServiceAccount::from_file(path.as_ref()).map_err(|e| {
                Error::config(format!(
                    "Failed to load service account from {:?}: {}",
                    path.as_ref(),
                    e
                ))
            })?;

        Ok(Self {
            project_id: project_id.into(),
            location: location.into(),
            token_provider: Arc::new(service_account),
            publisher: "google".to_string(),
            timeout: std::time::Duration::from_secs(300),
            default_model: None,
        })
    }

    /// Create configuration from a service account JSON string.
    ///
    /// Use this when credentials are provided as a string (e.g., from a secret manager).
    ///
    /// # Arguments
    ///
    /// * `json` - Service account JSON as a string
    /// * `project_id` - Your GCP project ID
    /// * `location` - GCP region (e.g., "us-central1")
    ///
    /// # Example
    ///
    /// ```ignore
    /// let json = std::fs::read_to_string("/path/to/service-account.json")?;
    /// let config = VertexConfig::from_service_account_json(&json, "my-project", "us-central1").await?;
    /// ```
    pub async fn from_service_account_json(
        json: &str,
        project_id: impl Into<String>,
        location: impl Into<String>,
    ) -> Result<Self> {
        let service_account = gcp_auth::CustomServiceAccount::from_json(json)
            .map_err(|e| Error::config(format!("Failed to parse service account JSON: {}", e)))?;

        Ok(Self {
            project_id: project_id.into(),
            location: location.into(),
            token_provider: Arc::new(service_account),
            publisher: "google".to_string(),
            timeout: std::time::Duration::from_secs(300),
            default_model: None,
        })
    }

    /// Set the request timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: std::time::Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the publisher for Vertex AI partner models.
    pub fn set_publisher(&mut self, publisher: impl Into<String>) -> &mut Self {
        self.publisher = publisher.into();
        self
    }

    /// Set the default model.
    #[must_use]
    pub fn with_default_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Create config from environment with a specific publisher for partner models.
    ///
    /// This is useful for accessing partner models like Claude (anthropic),
    /// Llama (meta), Mistral (mistralai), etc.
    pub async fn from_env_with_publisher(publisher: impl Into<String>) -> Result<Self> {
        let mut config = Self::from_env().await?;
        config.publisher = publisher.into();
        Ok(config)
    }
}

/// Google Vertex AI provider.
///
/// Provides access to Gemini models through Google Cloud's Vertex AI platform
/// with automatic credential discovery and token refresh.
pub struct VertexProvider {
    config: VertexConfig,
    client: Client,
}

impl VertexProvider {
    /// Create a new Vertex AI provider from environment variables.
    ///
    /// This is the recommended way to create a provider. It automatically
    /// discovers GCP credentials and refreshes tokens as needed.
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Ensure you have credentials set up:
    /// // - Run: gcloud auth application-default login
    /// // - Or set: GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
    /// //
    /// // And set your project:
    /// // - GOOGLE_CLOUD_PROJECT=my-project-id
    ///
    /// let provider = VertexProvider::from_env().await?;
    /// let response = provider.complete(request).await?;
    /// ```
    pub async fn from_env() -> Result<Self> {
        let config = VertexConfig::from_env().await?;
        Self::with_config(config)
    }

    /// Create a new Vertex AI provider from a service account file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the service account JSON file
    /// * `project_id` - Your GCP project ID
    /// * `location` - GCP region (e.g., "us-central1")
    pub async fn from_service_account_file(
        path: impl AsRef<Path>,
        project_id: impl Into<String>,
        location: impl Into<String>,
    ) -> Result<Self> {
        let config = VertexConfig::from_service_account_file(path, project_id, location).await?;
        Self::with_config(config)
    }

    /// Create a new Vertex AI provider from a service account JSON string.
    pub async fn from_service_account_json(
        json: &str,
        project_id: impl Into<String>,
        location: impl Into<String>,
    ) -> Result<Self> {
        let config = VertexConfig::from_service_account_json(json, project_id, location).await?;
        Self::with_config(config)
    }

    /// Create a new Vertex AI provider with custom configuration.
    pub fn with_config(config: VertexConfig) -> Result<Self> {
        let client = Client::builder().timeout(config.timeout).build()?;

        Ok(Self { config, client })
    }

    /// Create a new Vertex AI provider configured for medical domain applications.
    ///
    /// This helper configures the provider with Med-PaLM 2, Google's specialized model
    /// for medical use cases.
    ///
    /// # HIPAA Compliance Note
    ///
    /// When using this provider with Protected Health Information (PHI):
    /// - Ensure Vertex AI is configured with HIPAA-eligible resources
    /// - Enable data residency controls for your region
    /// - Review Google Cloud's BAA (Business Associate Agreement) terms
    pub async fn for_medical_domain(
        project_id: impl Into<String>,
        location: impl Into<String>,
    ) -> Result<Self> {
        let project_id = project_id.into();
        let location = location.into();

        // Use automatic credential discovery
        let token_provider = gcp_auth::provider().await.map_err(|e| {
            Error::config(format!("Failed to initialize GCP authentication: {}", e))
        })?;

        let config = VertexConfig {
            project_id,
            location,
            token_provider,
            publisher: "google".to_string(),
            timeout: std::time::Duration::from_secs(300),
            default_model: Some("medpalm-2".to_string()),
        };

        Self::with_config(config)
    }

    /// Get a fresh access token for API requests.
    ///
    /// The gcp_auth library handles token caching and automatic refresh,
    /// so this method is efficient to call for every request.
    async fn get_token(&self) -> Result<String> {
        let token = self
            .config
            .token_provider
            .token(VERTEX_SCOPES)
            .await
            .map_err(|e| {
                Error::auth(format!(
                    "Failed to get GCP access token: {}. \
                     Ensure your credentials are valid and have Vertex AI permissions.",
                    e
                ))
            })?;

        Ok(token.as_str().to_string())
    }

    fn api_url(&self, model: &str, streaming: bool) -> String {
        let method = if streaming {
            "streamGenerateContent"
        } else {
            "generateContent"
        };

        format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/{}/models/{}:{}",
            self.config.location,
            self.config.project_id,
            self.config.location,
            self.config.publisher,
            model,
            method
        )
    }

    /// Convert our unified request to Vertex AI's format.
    fn convert_request(&self, request: &CompletionRequest) -> VertexRequest {
        use crate::types::StructuredOutputType;

        let mut contents = Vec::new();

        // Convert messages
        for msg in &request.messages {
            contents.extend(self.convert_message(msg));
        }

        // Handle structured output
        let (response_mime_type, response_schema) = if let Some(ref rf) = request.response_format {
            match rf.format_type {
                StructuredOutputType::JsonSchema => {
                    if let Some(ref schema_def) = rf.json_schema {
                        (
                            Some("application/json".to_string()),
                            Some(schema_def.schema.clone()),
                        )
                    } else {
                        (Some("application/json".to_string()), None)
                    }
                }
                StructuredOutputType::JsonObject => (Some("application/json".to_string()), None),
                StructuredOutputType::Text => (None, None),
            }
        } else {
            (None, None)
        };

        // Build generation config
        let generation_config = Some(VertexGenerationConfig {
            temperature: request.temperature,
            top_p: request.top_p,
            max_output_tokens: request.max_tokens,
            stop_sequences: request.stop_sequences.clone(),
            response_mime_type,
            response_schema,
        });

        // Build system instruction if present
        let system_instruction = request.system.as_ref().map(|s| VertexContent {
            role: None,
            parts: vec![VertexPart::Text { text: s.clone() }],
        });

        // Convert tools
        let tools = request.tools.as_ref().map(|tools| {
            vec![VertexTool {
                function_declarations: tools
                    .iter()
                    .map(|t| VertexFunctionDeclaration {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: Some(t.input_schema.clone()),
                    })
                    .collect(),
            }]
        });

        // Convert thinking configuration to Vertex format.
        let thinking = request.thinking.as_ref().and_then(|t| {
            use crate::types::ThinkingType;
            match t.thinking_type {
                ThinkingType::Disabled => None,
                ThinkingType::Enabled => Some(VertexThinking {
                    enabled: true,
                    budget_tokens: t.budget_tokens,
                }),
            }
        });

        VertexRequest {
            contents,
            generation_config,
            system_instruction,
            tools,
            thinking,
        }
    }

    fn convert_message(&self, message: &Message) -> Vec<VertexContent> {
        let role = match message.role {
            Role::User => "user",
            Role::Assistant => "model",
            Role::System => "user",
        };

        let mut parts = Vec::new();

        for block in &message.content {
            match block {
                ContentBlock::Text { text } => {
                    parts.push(VertexPart::Text { text: text.clone() });
                }
                ContentBlock::Image { media_type, data } => {
                    parts.push(VertexPart::InlineData {
                        inline_data: VertexBlob {
                            mime_type: media_type.clone(),
                            data: data.clone(),
                        },
                    });
                }
                ContentBlock::ImageUrl { url } => {
                    parts.push(VertexPart::FileData {
                        file_data: VertexFileData {
                            mime_type: None,
                            file_uri: url.clone(),
                        },
                    });
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    parts.push(VertexPart::FunctionCall {
                        function_call: VertexFunctionCall {
                            name: name.clone(),
                            args: input.clone(),
                        },
                    });
                }
                ContentBlock::ToolResult {
                    tool_use_id,
                    content,
                    is_error,
                } => {
                    parts.push(VertexPart::FunctionResponse {
                        function_response: VertexFunctionResponse {
                            name: tool_use_id.clone(),
                            response: serde_json::json!({
                                "result": content,
                                "is_error": is_error
                            }),
                        },
                    });
                }
                ContentBlock::Thinking { .. } => {}
                _ => {
                    // Skip other blocks (Document, TextWithCache, etc.)
                }
            }
        }

        if parts.is_empty() {
            return vec![];
        }

        vec![VertexContent {
            role: Some(role.to_string()),
            parts,
        }]
    }

    fn convert_response(&self, response: VertexResponse) -> CompletionResponse {
        let mut content = Vec::new();
        let mut stop_reason = StopReason::EndTurn;

        if let Some(candidate) = response.candidates.into_iter().next() {
            // Extract stop reason first
            if let Some(ref reason) = candidate.finish_reason {
                stop_reason = match reason.as_str() {
                    "STOP" => StopReason::EndTurn,
                    "MAX_TOKENS" => StopReason::MaxTokens,
                    "SAFETY" => StopReason::ContentFilter,
                    _ => StopReason::EndTurn,
                };
            }

            if let Some(vertex_content) = candidate.content {
                for part in vertex_content.parts {
                    match part {
                        VertexPart::Text { text } => {
                            content.push(ContentBlock::Text { text });
                        }
                        VertexPart::FunctionCall { function_call } => {
                            content.push(ContentBlock::ToolUse {
                                id: uuid::Uuid::new_v4().to_string(),
                                name: function_call.name,
                                input: function_call.args,
                            });
                        }
                        _ => {}
                    }
                }
            }
        }

        let (input_tokens, output_tokens) = response
            .usage_metadata
            .map(|u| (u.prompt_token_count, u.candidates_token_count))
            .unwrap_or((0, 0));

        CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: "vertex".to_string(),
            content,
            stop_reason,
            usage: Usage {
                input_tokens,
                output_tokens,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        }
    }

    async fn handle_error_response(&self, response: reqwest::Response) -> Error {
        let status = response.status().as_u16();

        match response.json::<VertexErrorResponse>().await {
            Ok(err) => {
                let message = &err.error.message;
                let code = err.error.code;

                match code {
                    401 | 403 => Error::auth(format!(
                        "{}. Ensure your credentials have Vertex AI permissions \
                         (roles/aiplatform.user or similar).",
                        message
                    )),
                    429 => Error::rate_limited(message, None),
                    400 => {
                        if message.contains("not found") {
                            Error::ModelNotFound(message.clone())
                        } else {
                            Error::invalid_request(message)
                        }
                    }
                    _ => Error::server(status, message),
                }
            }
            Err(_) => Error::server(status, "Unknown error"),
        }
    }
}

#[async_trait]
impl Provider for VertexProvider {
    fn name(&self) -> &str {
        "vertex"
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = &request.model;
        let api_request = self.convert_request(&request);

        // Get fresh token for this request
        let token = self.get_token().await?;

        let response = self
            .client
            .post(self.api_url(model, false))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let vertex_response: VertexResponse = response.json().await?;
        Ok(self.convert_response(vertex_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let model = &request.model;
        let api_request = self.convert_request(&request);

        // Get fresh token for this request
        let token = self.get_token().await?;

        let response = self
            .client
            .post(self.api_url(model, true))
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .json(&api_request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(self.handle_error_response(response).await);
        }

        let stream = parse_vertex_stream(response);
        Ok(Box::pin(stream))
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
            "gemini-2.0-flash-exp",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
            "gemini-1.5-flash-002",
            "gemini-1.5-pro-002",
        ])
    }

    fn default_model(&self) -> Option<&str> {
        self.config
            .default_model
            .as_deref()
            .or(Some("gemini-1.5-flash"))
    }
}

/// Parse Vertex AI streaming response.
fn parse_vertex_stream(response: reqwest::Response) -> impl Stream<Item = Result<StreamChunk>> {
    use async_stream::try_stream;
    use futures::StreamExt;

    try_stream! {
        let mut event_stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut sent_start = false;
        let mut in_array = false;

        while let Some(chunk) = event_stream.next().await {
            let chunk = chunk?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            loop {
                let trimmed = buffer.trim_start().to_string();
                if trimmed.is_empty() {
                    break;
                }

                if let Some(rest) = trimmed.strip_prefix('[') {
                    in_array = true;
                    buffer = rest.to_string();
                    continue;
                }

                if let Some(rest) = trimmed.strip_prefix(',') {
                    buffer = rest.to_string();
                    continue;
                }

                if let Some(rest) = trimmed.strip_prefix(']') {
                    buffer = rest.to_string();
                    yield StreamChunk {
                        event_type: StreamEventType::MessageStop,
                        index: None,
                        delta: None,
                        stop_reason: None,
                        usage: None,
                    };
                    break;
                }

                if !in_array && !trimmed.starts_with('{') {
                    break;
                }

                // Parse JSON object
                let mut depth = 0;
                let mut end_pos = None;
                let mut in_string = false;
                let mut escape_next = false;

                for (i, c) in trimmed.char_indices() {
                    if escape_next {
                        escape_next = false;
                        continue;
                    }

                    match c {
                        '\\' if in_string => escape_next = true,
                        '"' => in_string = !in_string,
                        '{' if !in_string => depth += 1,
                        '}' if !in_string => {
                            depth -= 1;
                            if depth == 0 {
                                end_pos = Some(i + 1);
                                break;
                            }
                        }
                        _ => {}
                    }
                }

                if let Some(end) = end_pos {
                    let json_str = &trimmed[..end];
                    buffer = trimmed[end..].to_string();

                    if let Ok(parsed) = serde_json::from_str::<VertexStreamResponse>(json_str) {
                        if !sent_start {
                            yield StreamChunk {
                                event_type: StreamEventType::MessageStart,
                                index: None,
                                delta: None,
                                stop_reason: None,
                                usage: None,
                            };
                            sent_start = true;
                        }

                        for candidate in &parsed.candidates {
                            if let Some(ref content) = candidate.content {
                                for part in &content.parts {
                                    if let VertexPart::Text { text } = part {
                                        yield StreamChunk {
                                            event_type: StreamEventType::ContentBlockDelta,
                                            index: Some(0),
                                            delta: Some(ContentDelta::Text { text: text.clone() }),
                                            stop_reason: None,
                                            usage: None,
                                        };
                                    }
                                }
                            }

                            if let Some(ref reason) = candidate.finish_reason {
                                let stop_reason = match reason.as_str() {
                                    "STOP" => StopReason::EndTurn,
                                    "MAX_TOKENS" => StopReason::MaxTokens,
                                    "SAFETY" => StopReason::ContentFilter,
                                    _ => StopReason::EndTurn,
                                };

                                yield StreamChunk {
                                    event_type: StreamEventType::MessageDelta,
                                    index: None,
                                    delta: None,
                                    stop_reason: Some(stop_reason),
                                    usage: None,
                                };
                            }
                        }

                        if let Some(ref usage) = parsed.usage_metadata {
                            yield StreamChunk {
                                event_type: StreamEventType::MessageDelta,
                                index: None,
                                delta: None,
                                stop_reason: None,
                                usage: Some(Usage {
                                    input_tokens: usage.prompt_token_count,
                                    output_tokens: usage.candidates_token_count,
                                    cache_creation_input_tokens: 0,
                                    cache_read_input_tokens: 0,
                                }),
                            };
                        }
                    }
                } else {
                    break;
                }
            }
        }
    }
}

// ========== Vertex AI API Types ==========

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexRequest {
    contents: Vec<VertexContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    generation_config: Option<VertexGenerationConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system_instruction: Option<VertexContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<VertexTool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<VertexThinking>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexThinking {
    enabled: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    budget_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    role: Option<String>,
    parts: Vec<VertexPart>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(untagged)]
enum VertexPart {
    Text {
        text: String,
    },
    InlineData {
        #[serde(rename = "inlineData")]
        inline_data: VertexBlob,
    },
    FileData {
        #[serde(rename = "fileData")]
        file_data: VertexFileData,
    },
    FunctionCall {
        #[serde(rename = "functionCall")]
        function_call: VertexFunctionCall,
    },
    FunctionResponse {
        #[serde(rename = "functionResponse")]
        function_response: VertexFunctionResponse,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexBlob {
    mime_type: String,
    data: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexFileData {
    #[serde(skip_serializing_if = "Option::is_none")]
    mime_type: Option<String>,
    file_uri: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexFunctionCall {
    name: String,
    args: Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct VertexFunctionResponse {
    name: String,
    response: Value,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexGenerationConfig {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_mime_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_schema: Option<Value>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct VertexTool {
    function_declarations: Vec<VertexFunctionDeclaration>,
}

#[derive(Debug, Serialize)]
struct VertexFunctionDeclaration {
    name: String,
    description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    parameters: Option<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexResponse {
    candidates: Vec<VertexCandidate>,
    #[serde(default)]
    usage_metadata: Option<VertexUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexStreamResponse {
    candidates: Vec<VertexCandidate>,
    #[serde(default)]
    usage_metadata: Option<VertexUsageMetadata>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexCandidate {
    content: Option<VertexContent>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct VertexUsageMetadata {
    prompt_token_count: u32,
    candidates_token_count: u32,
}

#[derive(Debug, Deserialize)]
struct VertexErrorResponse {
    error: VertexError,
}

#[derive(Debug, Deserialize)]
struct VertexError {
    code: u16,
    message: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: Most tests require a mock TokenProvider since we can't easily
    // create real GCP credentials in unit tests. Integration tests should
    // be used to verify actual GCP connectivity.

    #[test]
    fn test_api_url_generation() {
        // Test URL generation without needing a real provider
        let location = "us-central1";
        let project_id = "my-project";
        let publisher = "google";
        let model = "gemini-1.5-flash";

        let url = format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/{}/models/{}:{}",
            location, project_id, location, publisher, model, "generateContent"
        );

        assert!(url.contains("my-project"));
        assert!(url.contains("us-central1"));
        assert!(url.contains("gemini-1.5-flash"));
        assert!(url.contains("generateContent"));

        let stream_url = format!(
            "https://{}-aiplatform.googleapis.com/v1/projects/{}/locations/{}/publishers/{}/models/{}:{}",
            location, project_id, location, publisher, model, "streamGenerateContent"
        );
        assert!(stream_url.contains("streamGenerateContent"));
    }

    #[test]
    fn test_vertex_request_serialization() {
        let request = VertexRequest {
            contents: vec![VertexContent {
                role: Some("user".to_string()),
                parts: vec![VertexPart::Text {
                    text: "Hello".to_string(),
                }],
            }],
            generation_config: Some(VertexGenerationConfig {
                temperature: Some(0.7),
                top_p: Some(0.9),
                max_output_tokens: Some(1024),
                stop_sequences: None,
                response_mime_type: None,
                response_schema: None,
            }),
            system_instruction: None,
            tools: None,
            thinking: None,
        };

        let json = serde_json::to_string(&request).expect("Should serialize");
        assert!(json.contains("\"temperature\":0.7"));
        assert!(json.contains("\"maxOutputTokens\":1024"));
    }

    #[test]
    fn test_vertex_response_deserialization() {
        let json = r#"{
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello! How can I help?"}]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 20
            }
        }"#;

        let response: VertexResponse = serde_json::from_str(json).expect("Should deserialize");
        assert_eq!(response.candidates.len(), 1);
        assert!(response.usage_metadata.is_some());
        let usage = response.usage_metadata.unwrap();
        assert_eq!(usage.prompt_token_count, 10);
        assert_eq!(usage.candidates_token_count, 20);
    }

    #[test]
    fn test_stop_reason_mapping() {
        // Test the stop reason string mapping
        let reasons = vec![
            ("STOP", StopReason::EndTurn),
            ("MAX_TOKENS", StopReason::MaxTokens),
            ("SAFETY", StopReason::ContentFilter),
            ("UNKNOWN", StopReason::EndTurn), // Default fallback
        ];

        for (vertex_reason, expected) in reasons {
            let stop_reason = match vertex_reason {
                "STOP" => StopReason::EndTurn,
                "MAX_TOKENS" => StopReason::MaxTokens,
                "SAFETY" => StopReason::ContentFilter,
                _ => StopReason::EndTurn,
            };
            assert!(
                matches!(stop_reason, expected_reason if std::mem::discriminant(&stop_reason) == std::mem::discriminant(&expected))
            );
        }
    }

    #[test]
    fn test_thinking_serialization() {
        let thinking = VertexThinking {
            enabled: true,
            budget_tokens: Some(10000),
        };

        let json = serde_json::to_string(&thinking).expect("Should serialize");
        assert!(json.contains("\"enabled\":true"));
        assert!(json.contains("\"budgetTokens\":10000"));
    }

    #[test]
    fn test_function_call_serialization() {
        let fc = VertexFunctionCall {
            name: "get_weather".to_string(),
            args: serde_json::json!({"location": "NYC"}),
        };

        let json = serde_json::to_string(&fc).expect("Should serialize");
        assert!(json.contains("get_weather"));
        assert!(json.contains("NYC"));
    }

    #[test]
    fn test_vertex_scopes() {
        assert!(VERTEX_SCOPES.contains(&"https://www.googleapis.com/auth/cloud-platform"));
    }
}
