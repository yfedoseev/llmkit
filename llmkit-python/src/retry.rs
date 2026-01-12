//! Retry configuration for Python bindings

use llmkit::retry::RetryConfig;
use pyo3::prelude::*;
use std::time::Duration;

/// Configuration for retry behavior on transient failures.
///
/// Example:
/// ```python
/// from llmkit import RetryConfig, LLMKitClient
///
/// # Use production defaults (10 retries, exponential backoff)
/// client = LLMKitClient.from_env()
///
/// # Use conservative config (3 retries, faster)
/// client = LLMKitClient(retry_config=RetryConfig.conservative())
///
/// # Disable retry entirely
/// client = LLMKitClient(retry_config=False)
///
/// # Custom config
/// client = LLMKitClient(retry_config=RetryConfig(
///     max_retries=5,
///     initial_delay_ms=500,
///     max_delay_ms=10000,
/// ))
/// ```
#[pyclass(name = "RetryConfig")]
#[derive(Clone)]
pub struct PyRetryConfig {
    pub(crate) inner: RetryConfig,
}

#[pymethods]
impl PyRetryConfig {
    /// Create a custom retry configuration.
    ///
    /// Args:
    ///     max_retries: Maximum number of retry attempts (default: 10)
    ///     initial_delay_ms: Initial delay before first retry in milliseconds (default: 1000)
    ///     max_delay_ms: Maximum delay between retries in milliseconds (default: 300000)
    ///     backoff_multiplier: Multiplier for exponential backoff (default: 2.0)
    ///     jitter: Whether to add random jitter to delays (default: True)
    ///
    /// Returns:
    ///     RetryConfig: Custom retry configuration
    #[new]
    #[pyo3(signature = (max_retries=10, initial_delay_ms=1000, max_delay_ms=300000, backoff_multiplier=2.0, jitter=true))]
    fn new(
        max_retries: u32,
        initial_delay_ms: u64,
        max_delay_ms: u64,
        backoff_multiplier: f64,
        jitter: bool,
    ) -> Self {
        Self {
            inner: RetryConfig {
                max_retries,
                initial_delay: Duration::from_millis(initial_delay_ms),
                max_delay: Duration::from_millis(max_delay_ms),
                backoff_multiplier,
                jitter,
            },
        }
    }

    /// Production-ready config with aggressive retry.
    ///
    /// 10 retries with exponential backoff:
    /// 1s → 2s → 4s → 8s → 16s → 32s → 64s → 128s → 256s → 300s (capped)
    ///
    /// Total max wait time: ~13 minutes across all retries.
    ///
    /// Returns:
    ///     RetryConfig: Production retry configuration
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: RetryConfig::default(),
        }
    }

    /// Alias for default() - production-ready config.
    ///
    /// Returns:
    ///     RetryConfig: Production retry configuration
    #[staticmethod]
    fn production() -> Self {
        Self {
            inner: RetryConfig::production(),
        }
    }

    /// Conservative config for latency-sensitive operations.
    ///
    /// 3 retries: 1s → 2s → 4s (max 30s)
    ///
    /// Returns:
    ///     RetryConfig: Conservative retry configuration
    #[staticmethod]
    fn conservative() -> Self {
        Self {
            inner: RetryConfig::conservative(),
        }
    }

    /// Disabled retry - operations fail immediately on first error.
    ///
    /// Use for testing or when retry is handled at a higher level.
    ///
    /// Returns:
    ///     RetryConfig: Disabled retry configuration (0 retries)
    #[staticmethod]
    fn none() -> Self {
        Self {
            inner: RetryConfig::none(),
        }
    }

    /// Maximum number of retry attempts.
    #[getter]
    fn max_retries(&self) -> u32 {
        self.inner.max_retries
    }

    /// Initial delay before first retry in milliseconds.
    #[getter]
    fn initial_delay_ms(&self) -> u64 {
        self.inner.initial_delay.as_millis() as u64
    }

    /// Maximum delay between retries in milliseconds.
    #[getter]
    fn max_delay_ms(&self) -> u64 {
        self.inner.max_delay.as_millis() as u64
    }

    /// Multiplier for exponential backoff.
    #[getter]
    fn backoff_multiplier(&self) -> f64 {
        self.inner.backoff_multiplier
    }

    /// Whether random jitter is added to delays.
    #[getter]
    fn jitter(&self) -> bool {
        self.inner.jitter
    }

    fn __repr__(&self) -> String {
        format!(
            "RetryConfig(max_retries={}, initial_delay_ms={}, max_delay_ms={}, backoff_multiplier={}, jitter={})",
            self.inner.max_retries,
            self.inner.initial_delay.as_millis(),
            self.inner.max_delay.as_millis(),
            self.inner.backoff_multiplier,
            if self.inner.jitter { "True" } else { "False" }
        )
    }
}

impl From<RetryConfig> for PyRetryConfig {
    fn from(config: RetryConfig) -> Self {
        Self { inner: config }
    }
}

impl From<PyRetryConfig> for RetryConfig {
    fn from(config: PyRetryConfig) -> Self {
        config.inner
    }
}
