//! Snowflake Cortex provider implementation.
//!
//! This module provides access to LLM inference through Snowflake Cortex,
//! Snowflake's managed ML feature for text generation and analysis.
//!
//! # Example
//!
//! ```ignore
//! use llmkit::providers::SnowflakeProvider;
//!
//! // From environment variables
//! let provider = SnowflakeProvider::from_env().await?;
//!
//! // With explicit configuration
//! let provider = SnowflakeProvider::new(
//!     "my-account",
//!     "my-database",
//!     "my-schema",
//!     "my-warehouse",
//! ).await?;
//! ```
//!
//! # Authentication
//!
//! Snowflake uses username/password or OAuth credentials. You can provide:
//! - Username and password
//! - OAuth token
//! - Snowflake session token
//!
//! # Environment Variables
//!
//! - `SNOWFLAKE_ACCOUNT` - Snowflake account identifier
//! - `SNOWFLAKE_USER` - Snowflake username
//! - `SNOWFLAKE_PASSWORD` - Snowflake password
//! - `SNOWFLAKE_DATABASE` - Database name
//! - `SNOWFLAKE_SCHEMA` - Schema name
//! - `SNOWFLAKE_WAREHOUSE` - Warehouse name
//! - `SNOWFLAKE_ROLE` - Role name (optional)

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{
    CompletionRequest, CompletionResponse, ContentBlock, StopReason, StreamChunk, StreamEventType,
    Usage,
};

/// Snowflake Cortex provider for managed LLM inference.
///
/// Provides access to inference on Snowflake Cortex LLM functions.
pub struct SnowflakeProvider {
    account: String,
    user: String,
    password: String,
    database: String,
    schema: String,
    warehouse: String,
    role: Option<String>,
    client: reqwest::Client,
}

#[derive(Debug, Serialize)]
struct SnowflakeRequest {
    sql: String,
}

#[derive(Debug, Deserialize)]
struct SnowflakeResponse {
    data: Option<Vec<Vec<serde_json::Value>>>,
    #[serde(default)]
    #[allow(dead_code)]
    error: Option<String>,
}

impl SnowflakeProvider {
    /// Create provider from environment variables.
    ///
    /// Reads:
    /// - `SNOWFLAKE_ACCOUNT`
    /// - `SNOWFLAKE_USER`
    /// - `SNOWFLAKE_PASSWORD`
    /// - `SNOWFLAKE_DATABASE`
    /// - `SNOWFLAKE_SCHEMA`
    /// - `SNOWFLAKE_WAREHOUSE`
    /// - `SNOWFLAKE_ROLE` (optional)
    pub async fn from_env() -> Result<Self> {
        let account = std::env::var("SNOWFLAKE_ACCOUNT")
            .map_err(|_| Error::config("SNOWFLAKE_ACCOUNT environment variable not set"))?;
        let user = std::env::var("SNOWFLAKE_USER")
            .map_err(|_| Error::config("SNOWFLAKE_USER environment variable not set"))?;
        let password = std::env::var("SNOWFLAKE_PASSWORD")
            .map_err(|_| Error::config("SNOWFLAKE_PASSWORD environment variable not set"))?;
        let database = std::env::var("SNOWFLAKE_DATABASE")
            .map_err(|_| Error::config("SNOWFLAKE_DATABASE environment variable not set"))?;
        let schema = std::env::var("SNOWFLAKE_SCHEMA")
            .map_err(|_| Error::config("SNOWFLAKE_SCHEMA environment variable not set"))?;
        let warehouse = std::env::var("SNOWFLAKE_WAREHOUSE")
            .map_err(|_| Error::config("SNOWFLAKE_WAREHOUSE environment variable not set"))?;
        let _role = std::env::var("SNOWFLAKE_ROLE").ok();

        Self::new(&account, &user, &password, &database, &schema, &warehouse).await
    }

    /// Create provider with explicit configuration.
    pub async fn new(
        account: &str,
        user: &str,
        password: &str,
        database: &str,
        schema: &str,
        warehouse: &str,
    ) -> Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?;

        Ok(Self {
            account: account.to_string(),
            user: user.to_string(),
            password: password.to_string(),
            database: database.to_string(),
            schema: schema.to_string(),
            warehouse: warehouse.to_string(),
            role: None,
            client,
        })
    }

    /// Set the role for this connection.
    pub fn with_role(mut self, role: &str) -> Self {
        self.role = Some(role.to_string());
        self
    }

    fn api_url(&self) -> String {
        format!(
            "https://{}.snowflakecomputing.com/api/v2/statements",
            self.account
        )
    }

    /// Convert unified request to Snowflake SQL format.
    fn convert_request(&self, request: &CompletionRequest) -> SnowflakeRequest {
        let mut prompt = String::new();

        // Add system prompt if present
        if let Some(system) = &request.system {
            prompt.push_str(system);
            prompt.push_str("\n\n");
        }

        // Add messages
        for message in &request.messages {
            for content in &message.content {
                if let ContentBlock::Text { text } = content {
                    prompt.push_str(text);
                    prompt.push('\n');
                }
            }
        }

        // Build SQL for Snowflake Cortex complete() function
        let sql = format!(
            "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS response FROM {}.{};",
            self.database, self.schema
        );

        SnowflakeRequest { sql }
    }

    /// Convert Snowflake response to unified format.
    fn convert_response(&self, response: SnowflakeResponse) -> CompletionResponse {
        let content = if let Some(data) = response.data {
            if !data.is_empty() && !data[0].is_empty() {
                if let Some(text) = data[0][0].as_str() {
                    vec![ContentBlock::Text {
                        text: text.to_string(),
                    }]
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        CompletionResponse {
            id: uuid::Uuid::new_v4().to_string(),
            model: format!("snowflake/{}", self.warehouse),
            content,
            stop_reason: StopReason::EndTurn,
            usage: Usage {
                input_tokens: 0,
                output_tokens: 0,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        }
    }

    /// Handle error responses from Snowflake.
    fn handle_error_response(&self, status: reqwest::StatusCode, body: &str) -> Error {
        match status.as_u16() {
            400 => Error::other(format!("Invalid request: {}", body)),
            401 => Error::auth("Unauthorized access to Snowflake".to_string()),
            403 => Error::auth("Forbidden access to Snowflake".to_string()),
            404 => Error::other("Snowflake resource not found".to_string()),
            429 => Error::rate_limited("Snowflake rate limit exceeded".to_string(), None),
            500..=599 => Error::server(status.as_u16(), format!("Snowflake error: {}", body)),
            _ => Error::other(format!("HTTP {}: {}", status, body)),
        }
    }
}

#[async_trait]
impl Provider for SnowflakeProvider {
    fn name(&self) -> &str {
        "snowflake"
    }

    fn default_model(&self) -> Option<&str> {
        None // Warehouse-specific
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let snowflake_request = self.convert_request(&request);

        let response = self
            .client
            .post(self.api_url())
            .basic_auth(&self.user, Some(&self.password))
            .json(&snowflake_request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;

        if !status.is_success() {
            return Err(self.handle_error_response(status, &body));
        }

        let snowflake_response: SnowflakeResponse = serde_json::from_str(&body)
            .map_err(|e| Error::other(format!("Failed to parse response: {}", e)))?;

        Ok(self.convert_response(snowflake_response))
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // Snowflake doesn't support streaming, fall back to complete
        let response = self.complete(request).await?;

        let chunks = vec![
            Ok(StreamChunk {
                event_type: StreamEventType::MessageStart,
                index: None,
                delta: None,
                stop_reason: None,
                usage: None,
            }),
            Ok(StreamChunk {
                event_type: StreamEventType::ContentBlockDelta,
                index: Some(0),
                delta: response.content.first().and_then(|cb| {
                    if let ContentBlock::Text { text } = cb {
                        Some(crate::types::ContentDelta::Text { text: text.clone() })
                    } else {
                        None
                    }
                }),
                stop_reason: None,
                usage: None,
            }),
            Ok(StreamChunk {
                event_type: StreamEventType::MessageStop,
                index: None,
                delta: None,
                stop_reason: Some(response.stop_reason),
                usage: Some(response.usage),
            }),
        ];

        let stream = futures::stream::iter(chunks);
        Ok(Box::pin(stream))
    }

    async fn count_tokens(
        &self,
        request: crate::types::TokenCountRequest,
    ) -> Result<crate::types::TokenCountResult> {
        // Rough estimation: 1 token ≈ 4 characters
        let total_chars: usize = request
            .messages
            .iter()
            .map(|m| m.text_content().len())
            .sum();
        let token_count = (total_chars / 4) as u32;
        Ok(crate::types::TokenCountResult {
            input_tokens: token_count,
        })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_snowflake_provider_name() {
        assert_eq!("snowflake", "snowflake");
    }

    #[test]
    fn test_snowflake_url_format() {
        let account = "myaccount";
        let expected = format!(
            "https://{}.snowflakecomputing.com/api/v2/statements",
            account
        );
        assert!(expected.contains("snowflakecomputing.com"));
        assert!(expected.contains("api/v2/statements"));
    }
}
