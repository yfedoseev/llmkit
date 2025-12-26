//! Retry logic for LLM provider operations.
//!
//! This module provides a wrapper that adds automatic retry with exponential backoff
//! for transient failures.

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;
use tokio::time::sleep;

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{CompletionRequest, CompletionResponse, StreamChunk};

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: u32,

    /// Initial delay before first retry.
    pub initial_delay: Duration,

    /// Maximum delay between retries.
    pub max_delay: Duration,

    /// Multiplier for exponential backoff (e.g., 2.0 doubles the delay each retry).
    pub backoff_multiplier: f64,

    /// Whether to add random jitter to delays.
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 10,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(300), // 5 minutes
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }
}

impl RetryConfig {
    /// Create a new retry config with specified max retries.
    pub fn new(max_retries: u32) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    /// Production-ready config with aggressive retry.
    ///
    /// 10 retries with exponential backoff:
    /// 1s → 2s → 4s → 8s → 16s → 32s → 64s → 128s → 256s → 300s (capped)
    ///
    /// Total max wait time: ~13 minutes across all retries.
    pub fn production() -> Self {
        Self::default()
    }

    /// Disabled retry - operations fail immediately on first error.
    ///
    /// Use for testing or when retry is handled at a higher level.
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            ..Default::default()
        }
    }

    /// Conservative config for latency-sensitive operations.
    ///
    /// 3 retries: 1s → 2s → 4s (max 30s)
    pub fn conservative() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_secs(1),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            jitter: true,
        }
    }

    /// Builder: Set initial delay.
    pub fn with_initial_delay(mut self, delay: Duration) -> Self {
        self.initial_delay = delay;
        self
    }

    /// Builder: Set max delay.
    pub fn with_max_delay(mut self, delay: Duration) -> Self {
        self.max_delay = delay;
        self
    }

    /// Builder: Set backoff multiplier.
    pub fn with_backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.backoff_multiplier = multiplier;
        self
    }

    /// Builder: Enable or disable jitter.
    pub fn with_jitter(mut self, jitter: bool) -> Self {
        self.jitter = jitter;
        self
    }

    /// Calculate delay for a given attempt number (0-indexed).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        let base_delay =
            self.initial_delay.as_millis() as f64 * self.backoff_multiplier.powi(attempt as i32);

        let capped_delay = base_delay.min(self.max_delay.as_millis() as f64);

        let final_delay = if self.jitter {
            // Add random jitter of +/- 25%
            let jitter_factor = 0.75 + (rand_simple() * 0.5);
            capped_delay * jitter_factor
        } else {
            capped_delay
        };

        Duration::from_millis(final_delay as u64)
    }
}

/// Simple random number generator for jitter (0.0 to 1.0).
/// Uses a basic approach that doesn't require external crates.
fn rand_simple() -> f64 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write_u64(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64,
    );

    (hasher.finish() % 1000) as f64 / 1000.0
}

/// A provider wrapper that adds automatic retry logic.
pub struct RetryingProvider<P: Provider> {
    inner: P,
    config: RetryConfig,
}

impl<P: Provider> RetryingProvider<P> {
    /// Create a new retrying provider with default config.
    pub fn new(provider: P) -> Self {
        Self {
            inner: provider,
            config: RetryConfig::default(),
        }
    }

    /// Create a new retrying provider with custom config.
    pub fn with_config(provider: P, config: RetryConfig) -> Self {
        Self {
            inner: provider,
            config,
        }
    }

    /// Get a reference to the inner provider.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Get the retry configuration.
    pub fn config(&self) -> &RetryConfig {
        &self.config
    }

    /// Execute an operation with retry logic.
    async fn execute_with_retry<T, F, Fut>(&self, operation_name: &str, mut f: F) -> Result<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = Result<T>>,
    {
        let mut last_error: Option<Error> = None;

        for attempt in 0..=self.config.max_retries {
            match f().await {
                Ok(result) => {
                    if attempt > 0 {
                        tracing::info!(
                            provider = %self.inner.name(),
                            operation = %operation_name,
                            attempt = attempt + 1,
                            "Operation succeeded after retry"
                        );
                    }
                    return Ok(result);
                }
                Err(e) => {
                    if !e.is_retryable() {
                        tracing::debug!(
                            provider = %self.inner.name(),
                            operation = %operation_name,
                            error = %e,
                            "Non-retryable error, failing immediately"
                        );
                        return Err(e);
                    }

                    if attempt < self.config.max_retries {
                        // Calculate delay, respecting retry-after header if present
                        let delay = e
                            .retry_after()
                            .unwrap_or_else(|| self.config.delay_for_attempt(attempt));

                        tracing::warn!(
                            provider = %self.inner.name(),
                            operation = %operation_name,
                            attempt = attempt + 1,
                            max_retries = self.config.max_retries,
                            delay_ms = delay.as_millis(),
                            error = %e,
                            "Retryable error, will retry after delay"
                        );

                        sleep(delay).await;
                    }

                    last_error = Some(e);
                }
            }
        }

        tracing::error!(
            provider = %self.inner.name(),
            operation = %operation_name,
            max_retries = self.config.max_retries,
            "All retry attempts exhausted"
        );

        Err(last_error.unwrap_or_else(|| Error::other("Unknown retry failure")))
    }
}

#[async_trait]
impl<P: Provider> Provider for RetryingProvider<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        // Clone the request for potential retries
        let request = Arc::new(request);

        self.execute_with_retry("complete", || {
            let request = (*request).clone();
            async move { self.inner.complete(request).await }
        })
        .await
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // For streaming, we retry the initial connection but not mid-stream errors
        let request = Arc::new(request);

        self.execute_with_retry("complete_stream", || {
            let request = (*request).clone();
            async move { self.inner.complete_stream(request).await }
        })
        .await
    }

    fn supports_tools(&self) -> bool {
        self.inner.supports_tools()
    }

    fn supports_vision(&self) -> bool {
        self.inner.supports_vision()
    }

    fn supports_streaming(&self) -> bool {
        self.inner.supports_streaming()
    }
}

/// Extension trait to easily wrap providers with retry logic.
pub trait ProviderExt: Provider + Sized {
    /// Wrap this provider with default retry logic.
    fn with_retry(self) -> RetryingProvider<Self> {
        RetryingProvider::new(self)
    }

    /// Wrap this provider with custom retry configuration.
    fn with_retry_config(self, config: RetryConfig) -> RetryingProvider<Self> {
        RetryingProvider::with_config(self, config)
    }
}

impl<P: Provider> ProviderExt for P {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 10);
        assert_eq!(config.initial_delay, Duration::from_secs(1));
        assert_eq!(config.max_delay, Duration::from_secs(300));
        assert_eq!(config.backoff_multiplier, 2.0);
        assert!(config.jitter);
    }

    #[test]
    fn test_retry_config_production() {
        let config = RetryConfig::production();
        assert_eq!(config.max_retries, 10);
        assert_eq!(config.max_delay, Duration::from_secs(300));
    }

    #[test]
    fn test_retry_config_none() {
        let config = RetryConfig::none();
        assert_eq!(config.max_retries, 0);
    }

    #[test]
    fn test_retry_config_conservative() {
        let config = RetryConfig::conservative();
        assert_eq!(config.max_retries, 3);
        assert_eq!(config.max_delay, Duration::from_secs(30));
    }

    #[test]
    fn test_retry_config_builder() {
        let config = RetryConfig::new(5)
            .with_initial_delay(Duration::from_millis(500))
            .with_max_delay(Duration::from_secs(30))
            .with_backoff_multiplier(1.5)
            .with_jitter(false);

        assert_eq!(config.max_retries, 5);
        assert_eq!(config.initial_delay, Duration::from_millis(500));
        assert_eq!(config.max_delay, Duration::from_secs(30));
        assert_eq!(config.backoff_multiplier, 1.5);
        assert!(!config.jitter);
    }

    #[test]
    fn test_delay_calculation_no_jitter() {
        let config = RetryConfig::new(5)
            .with_initial_delay(Duration::from_millis(100))
            .with_backoff_multiplier(2.0)
            .with_jitter(false);

        assert_eq!(config.delay_for_attempt(0), Duration::from_millis(100));
        assert_eq!(config.delay_for_attempt(1), Duration::from_millis(200));
        assert_eq!(config.delay_for_attempt(2), Duration::from_millis(400));
        assert_eq!(config.delay_for_attempt(3), Duration::from_millis(800));
    }

    #[test]
    fn test_delay_respects_max() {
        let config = RetryConfig::new(5)
            .with_initial_delay(Duration::from_secs(10))
            .with_max_delay(Duration::from_secs(30))
            .with_backoff_multiplier(2.0)
            .with_jitter(false);

        // 10 * 2^0 = 10s
        assert_eq!(config.delay_for_attempt(0), Duration::from_secs(10));
        // 10 * 2^1 = 20s
        assert_eq!(config.delay_for_attempt(1), Duration::from_secs(20));
        // 10 * 2^2 = 40s, capped at 30s
        assert_eq!(config.delay_for_attempt(2), Duration::from_secs(30));
        // Still capped at 30s
        assert_eq!(config.delay_for_attempt(3), Duration::from_secs(30));
    }

    #[test]
    fn test_delay_with_jitter_varies() {
        let config = RetryConfig::new(5)
            .with_initial_delay(Duration::from_millis(1000))
            .with_jitter(true);

        // With jitter, the delay should be between 75% and 125% of base
        let delay = config.delay_for_attempt(0);
        assert!(delay >= Duration::from_millis(750));
        assert!(delay <= Duration::from_millis(1250));
    }
}
