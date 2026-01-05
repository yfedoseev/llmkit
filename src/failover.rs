//! Failover configuration and logic for provider routing.
//!
//! This module provides automatic failover between providers when errors occur,
//! with support for model mapping between different providers.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::{FailoverConfig, FallbackProvider, FailoverTrigger};
//!
//! let config = FailoverConfig::new("openai")
//!     .add_fallback(FallbackProvider::new("anthropic")
//!         .with_model_mapping("gpt-4o", "claude-sonnet-4-20250514"))
//!     .trigger_on(FailoverTrigger::RateLimit)
//!     .trigger_on(FailoverTrigger::ServerError);
//! ```

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{
    BatchJob, BatchRequest, BatchResult, CompletionRequest, CompletionResponse, StreamChunk,
    TokenCountRequest, TokenCountResult,
};

/// Triggers that cause failover to occur.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailoverTrigger {
    /// Failover on rate limit errors
    RateLimit,
    /// Failover on server errors (5xx)
    ServerError,
    /// Failover on timeout
    Timeout,
    /// Failover on any network error
    NetworkError,
    /// Failover on authentication errors (useful for key rotation)
    AuthError,
    /// Failover on any error
    AnyError,
}

impl FailoverTrigger {
    /// Check if an error matches this trigger.
    pub fn matches(&self, error: &Error) -> bool {
        match self {
            FailoverTrigger::RateLimit => matches!(error, Error::RateLimited { .. }),
            FailoverTrigger::ServerError => {
                matches!(error, Error::Server { status, .. } if *status >= 500)
            }
            FailoverTrigger::Timeout => matches!(error, Error::Timeout),
            FailoverTrigger::NetworkError => matches!(error, Error::Network(_)),
            FailoverTrigger::AuthError => matches!(error, Error::Authentication(_)),
            FailoverTrigger::AnyError => true,
        }
    }
}

/// Configuration for a fallback provider.
#[derive(Clone)]
pub struct FallbackProvider {
    /// Provider name
    pub name: String,
    /// The actual provider
    pub provider: Arc<dyn Provider>,
    /// Model mapping from primary to fallback (primary_model -> fallback_model)
    pub model_mapping: HashMap<String, String>,
}

impl FallbackProvider {
    /// Create a new fallback provider.
    pub fn new(name: impl Into<String>, provider: impl Provider + 'static) -> Self {
        Self {
            name: name.into(),
            provider: Arc::new(provider),
            model_mapping: HashMap::new(),
        }
    }

    /// Create from an Arc'd provider.
    pub fn from_arc(name: impl Into<String>, provider: Arc<dyn Provider>) -> Self {
        Self {
            name: name.into(),
            provider,
            model_mapping: HashMap::new(),
        }
    }

    /// Add a model mapping.
    pub fn with_model_mapping(
        mut self,
        primary_model: impl Into<String>,
        fallback_model: impl Into<String>,
    ) -> Self {
        self.model_mapping
            .insert(primary_model.into(), fallback_model.into());
        self
    }

    /// Add multiple model mappings.
    pub fn with_model_mappings(mut self, mappings: HashMap<String, String>) -> Self {
        self.model_mapping.extend(mappings);
        self
    }

    /// Map a model name to the fallback model.
    pub fn map_model(&self, model: &str) -> String {
        self.model_mapping
            .get(model)
            .cloned()
            .unwrap_or_else(|| model.to_string())
    }
}

/// Configuration for failover behavior.
#[derive(Clone)]
pub struct FailoverConfig {
    /// Primary provider name
    pub primary_name: String,
    /// Primary provider
    pub primary: Arc<dyn Provider>,
    /// Fallback providers in order of preference
    pub fallbacks: Vec<FallbackProvider>,
    /// Triggers that cause failover
    pub triggers: Vec<FailoverTrigger>,
    /// Maximum number of fallback attempts (0 = try all fallbacks)
    pub max_attempts: usize,
    /// Whether to retry the primary on transient errors before failing over
    pub retry_primary_first: bool,
    /// Number of retries on primary before failing over
    pub primary_retries: u32,
}

impl FailoverConfig {
    /// Create a new failover config with a primary provider.
    pub fn new(name: impl Into<String>, primary: impl Provider + 'static) -> Self {
        Self {
            primary_name: name.into(),
            primary: Arc::new(primary),
            fallbacks: Vec::new(),
            triggers: vec![FailoverTrigger::RateLimit, FailoverTrigger::ServerError],
            max_attempts: 0,
            retry_primary_first: false,
            primary_retries: 0,
        }
    }

    /// Create from an Arc'd provider.
    pub fn from_arc(name: impl Into<String>, primary: Arc<dyn Provider>) -> Self {
        Self {
            primary_name: name.into(),
            primary,
            fallbacks: Vec::new(),
            triggers: vec![FailoverTrigger::RateLimit, FailoverTrigger::ServerError],
            max_attempts: 0,
            retry_primary_first: false,
            primary_retries: 0,
        }
    }

    /// Add a fallback provider.
    pub fn add_fallback(mut self, fallback: FallbackProvider) -> Self {
        self.fallbacks.push(fallback);
        self
    }

    /// Set the failover triggers.
    pub fn with_triggers(mut self, triggers: Vec<FailoverTrigger>) -> Self {
        self.triggers = triggers;
        self
    }

    /// Add a failover trigger.
    pub fn trigger_on(mut self, trigger: FailoverTrigger) -> Self {
        if !self.triggers.contains(&trigger) {
            self.triggers.push(trigger);
        }
        self
    }

    /// Set maximum number of failover attempts.
    pub fn with_max_attempts(mut self, max: usize) -> Self {
        self.max_attempts = max;
        self
    }

    /// Enable retrying primary before failover.
    pub fn retry_primary(mut self, retries: u32) -> Self {
        self.retry_primary_first = true;
        self.primary_retries = retries;
        self
    }

    /// Check if an error should trigger failover.
    pub fn should_failover(&self, error: &Error) -> bool {
        self.triggers.iter().any(|t| t.matches(error))
    }
}

/// Provider wrapper that implements failover logic.
pub struct FailoverProvider {
    config: FailoverConfig,
}

impl FailoverProvider {
    /// Create a new failover provider.
    pub fn new(config: FailoverConfig) -> Self {
        Self { config }
    }

    /// Get the primary provider name.
    pub fn primary_name(&self) -> &str {
        &self.config.primary_name
    }

    /// Get the fallback provider names.
    pub fn fallback_names(&self) -> Vec<&str> {
        self.config
            .fallbacks
            .iter()
            .map(|f| f.name.as_str())
            .collect()
    }

    /// Execute with failover logic.
    async fn execute_with_failover<F, T>(&self, request: CompletionRequest, execute: F) -> Result<T>
    where
        F: Fn(
            Arc<dyn Provider>,
            CompletionRequest,
        ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>,
    {
        let original_model = request.model.clone();

        // Try primary first
        let mut last_error = None;

        // Optionally retry primary
        let primary_attempts = if self.config.retry_primary_first {
            self.config.primary_retries + 1
        } else {
            1
        };

        for attempt in 0..primary_attempts {
            match execute(self.config.primary.clone(), request.clone()).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    tracing::warn!(
                        provider = %self.config.primary_name,
                        attempt = attempt + 1,
                        error = %e,
                        "Primary provider failed"
                    );

                    if !self.config.should_failover(&e) {
                        return Err(e);
                    }

                    last_error = Some(e);
                }
            }
        }

        // Try fallbacks
        let max_attempts = if self.config.max_attempts == 0 {
            self.config.fallbacks.len()
        } else {
            self.config.max_attempts.min(self.config.fallbacks.len())
        };

        for fallback in self.config.fallbacks.iter().take(max_attempts) {
            let mapped_model = fallback.map_model(&original_model);
            let mut fallback_request = request.clone();
            fallback_request.model = mapped_model;

            match execute(fallback.provider.clone(), fallback_request).await {
                Ok(result) => {
                    tracing::info!(
                        provider = %fallback.name,
                        original_model = %original_model,
                        "Failover successful"
                    );
                    return Ok(result);
                }
                Err(e) => {
                    tracing::warn!(
                        provider = %fallback.name,
                        error = %e,
                        "Fallback provider failed"
                    );
                    last_error = Some(e);
                }
            }
        }

        // All providers failed
        Err(last_error.unwrap_or_else(|| Error::other("All providers failed")))
    }
}

#[async_trait]
impl Provider for FailoverProvider {
    fn name(&self) -> &str {
        // Return a composite name
        &self.config.primary_name
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        self.execute_with_failover(request, |provider, req| {
            Box::pin(async move { provider.complete(req).await })
        })
        .await
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let original_model = request.model.clone();

        // Try primary first
        match self.config.primary.complete_stream(request.clone()).await {
            Ok(stream) => return Ok(stream),
            Err(e) => {
                if !self.config.should_failover(&e) {
                    return Err(e);
                }
                tracing::warn!(
                    provider = %self.config.primary_name,
                    error = %e,
                    "Primary provider streaming failed, trying fallbacks"
                );
            }
        }

        // Try fallbacks
        for fallback in &self.config.fallbacks {
            let mapped_model = fallback.map_model(&original_model);
            let mut fallback_request = request.clone();
            fallback_request.model = mapped_model;

            match fallback.provider.complete_stream(fallback_request).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    tracing::warn!(
                        provider = %fallback.name,
                        error = %e,
                        "Fallback provider streaming failed"
                    );
                }
            }
        }

        Err(Error::other("All providers failed for streaming"))
    }

    fn supports_tools(&self) -> bool {
        self.config.primary.supports_tools()
    }

    fn supports_vision(&self) -> bool {
        self.config.primary.supports_vision()
    }

    fn supports_streaming(&self) -> bool {
        self.config.primary.supports_streaming()
    }

    fn supports_token_counting(&self) -> bool {
        self.config.primary.supports_token_counting()
    }

    async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCountResult> {
        // Token counting doesn't need failover - use primary
        self.config.primary.count_tokens(request).await
    }

    fn supports_batch(&self) -> bool {
        self.config.primary.supports_batch()
    }

    async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchJob> {
        // Batch operations use primary only
        self.config.primary.create_batch(requests).await
    }

    async fn get_batch(&self, batch_id: &str) -> Result<BatchJob> {
        self.config.primary.get_batch(batch_id).await
    }

    async fn get_batch_results(&self, batch_id: &str) -> Result<Vec<BatchResult>> {
        self.config.primary.get_batch_results(batch_id).await
    }

    async fn cancel_batch(&self, batch_id: &str) -> Result<BatchJob> {
        self.config.primary.cancel_batch(batch_id).await
    }

    async fn list_batches(&self, limit: Option<u32>) -> Result<Vec<BatchJob>> {
        self.config.primary.list_batches(limit).await
    }
}

/// Quick helper for common failover configurations.
pub mod presets {
    use super::*;

    /// Create a simple failover config with common triggers.
    pub fn simple_failover(
        primary_name: impl Into<String>,
        primary: impl Provider + 'static,
        fallback_name: impl Into<String>,
        fallback: impl Provider + 'static,
    ) -> FailoverConfig {
        FailoverConfig::new(primary_name, primary)
            .add_fallback(FallbackProvider::new(fallback_name, fallback))
            .trigger_on(FailoverTrigger::RateLimit)
            .trigger_on(FailoverTrigger::ServerError)
            .trigger_on(FailoverTrigger::Timeout)
    }

    /// OpenAI to Anthropic failover with model mappings.
    pub fn openai_to_anthropic(
        openai: impl Provider + 'static,
        anthropic: impl Provider + 'static,
    ) -> FailoverConfig {
        FailoverConfig::new("openai", openai).add_fallback(
            FallbackProvider::new("anthropic", anthropic)
                .with_model_mapping("gpt-4o", "claude-sonnet-4-20250514")
                .with_model_mapping("gpt-4o-mini", "claude-3-5-haiku-20241022")
                .with_model_mapping("gpt-4-turbo", "claude-sonnet-4-20250514")
                .with_model_mapping("o1", "claude-sonnet-4-20250514"),
        )
    }

    /// Anthropic to OpenAI failover with model mappings.
    pub fn anthropic_to_openai(
        anthropic: impl Provider + 'static,
        openai: impl Provider + 'static,
    ) -> FailoverConfig {
        FailoverConfig::new("anthropic", anthropic).add_fallback(
            FallbackProvider::new("openai", openai)
                .with_model_mapping("claude-sonnet-4-20250514", "gpt-4o")
                .with_model_mapping("claude-3-5-haiku-20241022", "gpt-4o-mini")
                .with_model_mapping("claude-opus-4-20250514", "gpt-4o"),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_failover_trigger_matches() {
        let rate_limit = Error::rate_limited("too many requests", None);
        let server_error = Error::server(503, "overloaded");
        let timeout = Error::Timeout;
        let auth_error = Error::auth("invalid key");

        assert!(FailoverTrigger::RateLimit.matches(&rate_limit));
        assert!(!FailoverTrigger::RateLimit.matches(&server_error));

        assert!(FailoverTrigger::ServerError.matches(&server_error));
        assert!(!FailoverTrigger::ServerError.matches(&rate_limit));

        assert!(FailoverTrigger::Timeout.matches(&timeout));

        assert!(FailoverTrigger::AuthError.matches(&auth_error));

        assert!(FailoverTrigger::AnyError.matches(&rate_limit));
        assert!(FailoverTrigger::AnyError.matches(&auth_error));
    }

    #[test]
    fn test_fallback_model_mapping() {
        struct DummyProvider;

        #[async_trait]
        impl Provider for DummyProvider {
            fn name(&self) -> &str {
                "dummy"
            }

            async fn complete(&self, _: CompletionRequest) -> Result<CompletionResponse> {
                unimplemented!()
            }

            async fn complete_stream(
                &self,
                _: CompletionRequest,
            ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
                unimplemented!()
            }
        }

        let fallback = FallbackProvider::new("anthropic", DummyProvider)
            .with_model_mapping("gpt-4o", "claude-sonnet-4-20250514")
            .with_model_mapping("gpt-4o-mini", "claude-3-5-haiku-20241022");

        assert_eq!(fallback.map_model("gpt-4o"), "claude-sonnet-4-20250514");
        assert_eq!(
            fallback.map_model("gpt-4o-mini"),
            "claude-3-5-haiku-20241022"
        );
        assert_eq!(fallback.map_model("unknown"), "unknown"); // No mapping, pass through
    }

    #[test]
    fn test_failover_config_builder() {
        struct DummyProvider;

        #[async_trait]
        impl Provider for DummyProvider {
            fn name(&self) -> &str {
                "dummy"
            }

            async fn complete(&self, _: CompletionRequest) -> Result<CompletionResponse> {
                unimplemented!()
            }

            async fn complete_stream(
                &self,
                _: CompletionRequest,
            ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
                unimplemented!()
            }
        }

        let config = FailoverConfig::new("primary", DummyProvider)
            .add_fallback(FallbackProvider::new("fallback1", DummyProvider))
            .add_fallback(FallbackProvider::new("fallback2", DummyProvider))
            .trigger_on(FailoverTrigger::RateLimit)
            .trigger_on(FailoverTrigger::Timeout)
            .with_max_attempts(1)
            .retry_primary(2);

        assert_eq!(config.primary_name, "primary");
        assert_eq!(config.fallbacks.len(), 2);
        assert_eq!(config.triggers.len(), 3); // Default has RateLimit, ServerError + added Timeout
        assert_eq!(config.max_attempts, 1);
        assert!(config.retry_primary_first);
        assert_eq!(config.primary_retries, 2);
    }
}
