//! Retry configuration for JavaScript bindings

use modelsuite::retry::RetryConfig;
use napi_derive::napi;
use std::time::Duration;

/// Configuration for retry behavior on transient failures.
///
/// @example
/// ```typescript
/// import { ModelSuiteClient, RetryConfig } from 'modelsuite'
///
/// // Use production defaults (10 retries, exponential backoff)
/// const client = ModelSuiteClient.fromEnv()
///
/// // Use conservative config (3 retries, faster)
/// const client = new ModelSuiteClient({ retryConfig: RetryConfig.conservative() })
///
/// // Disable retry entirely
/// const client = new ModelSuiteClient({ retryConfig: RetryConfig.none() })
///
/// // Custom config
/// const client = new ModelSuiteClient({
///   retryConfig: new RetryConfig({
///     maxRetries: 5,
///     initialDelayMs: 500,
///     maxDelayMs: 10000,
///   })
/// })
/// ```
#[napi]
pub struct JsRetryConfig {
    pub(crate) inner: RetryConfig,
}

/// Options for creating a RetryConfig.
#[napi(object)]
pub struct RetryConfigOptions {
    /// Maximum number of retry attempts (default: 10)
    pub max_retries: Option<u32>,
    /// Initial delay before first retry in milliseconds (default: 1000)
    pub initial_delay_ms: Option<u32>,
    /// Maximum delay between retries in milliseconds (default: 300000)
    pub max_delay_ms: Option<u32>,
    /// Multiplier for exponential backoff (default: 2.0)
    pub backoff_multiplier: Option<f64>,
    /// Whether to add random jitter to delays (default: true)
    pub jitter: Option<bool>,
}

#[napi]
impl JsRetryConfig {
    /// Create a custom retry configuration.
    ///
    /// @param options - Configuration options
    ///
    /// @example
    /// ```typescript
    /// const config = new RetryConfig({
    ///   maxRetries: 5,
    ///   initialDelayMs: 500,
    ///   maxDelayMs: 10000,
    ///   backoffMultiplier: 2.0,
    ///   jitter: true,
    /// })
    /// ```
    #[napi(constructor)]
    pub fn new(options: Option<RetryConfigOptions>) -> Self {
        let opts = options.unwrap_or(RetryConfigOptions {
            max_retries: None,
            initial_delay_ms: None,
            max_delay_ms: None,
            backoff_multiplier: None,
            jitter: None,
        });

        Self {
            inner: RetryConfig {
                max_retries: opts.max_retries.unwrap_or(10),
                initial_delay: Duration::from_millis(opts.initial_delay_ms.unwrap_or(1000) as u64),
                max_delay: Duration::from_millis(opts.max_delay_ms.unwrap_or(300000) as u64),
                backoff_multiplier: opts.backoff_multiplier.unwrap_or(2.0),
                jitter: opts.jitter.unwrap_or(true),
            },
        }
    }

    /// Production-ready config with aggressive retry.
    ///
    /// 10 retries with exponential backoff:
    /// 1s -> 2s -> 4s -> 8s -> 16s -> 32s -> 64s -> 128s -> 256s -> 300s (capped)
    ///
    /// Total max wait time: ~13 minutes across all retries.
    #[napi(factory)]
    pub fn production() -> Self {
        Self {
            inner: RetryConfig::production(),
        }
    }

    /// Conservative config for latency-sensitive operations.
    ///
    /// 3 retries: 1s -> 2s -> 4s (max 30s)
    #[napi(factory)]
    pub fn conservative() -> Self {
        Self {
            inner: RetryConfig::conservative(),
        }
    }

    /// Disabled retry - operations fail immediately on first error.
    ///
    /// Use for testing or when retry is handled at a higher level.
    #[napi(factory)]
    pub fn none() -> Self {
        Self {
            inner: RetryConfig::none(),
        }
    }

    /// Maximum number of retry attempts.
    #[napi(getter)]
    pub fn max_retries(&self) -> u32 {
        self.inner.max_retries
    }

    /// Initial delay before first retry in milliseconds.
    #[napi(getter)]
    pub fn initial_delay_ms(&self) -> u32 {
        self.inner.initial_delay.as_millis() as u32
    }

    /// Maximum delay between retries in milliseconds.
    #[napi(getter)]
    pub fn max_delay_ms(&self) -> u32 {
        self.inner.max_delay.as_millis() as u32
    }

    /// Multiplier for exponential backoff.
    #[napi(getter)]
    pub fn backoff_multiplier(&self) -> f64 {
        self.inner.backoff_multiplier
    }

    /// Whether random jitter is added to delays.
    #[napi(getter)]
    pub fn jitter(&self) -> bool {
        self.inner.jitter
    }
}

impl From<RetryConfig> for JsRetryConfig {
    fn from(config: RetryConfig) -> Self {
        Self { inner: config }
    }
}

impl From<JsRetryConfig> for RetryConfig {
    fn from(config: JsRetryConfig) -> Self {
        config.inner
    }
}
