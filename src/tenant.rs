//! Tenant context and multi-tenancy support for LLM requests.
//!
//! This module provides tenant isolation, rate limiting, and cost controls
//! for multi-tenant LLM applications.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::{TenantConfig, TenantProvider, RateLimitConfig, CostLimitConfig};
//!
//! let config = TenantConfig::new("acme-corp")
//!     .with_allowed_models(vec!["gpt-4o", "claude-sonnet-4-20250514"])
//!     .with_rate_limit(RateLimitConfig {
//!         requests_per_minute: Some(60),
//!         tokens_per_minute: Some(100_000),
//!         ..Default::default()
//!     })
//!     .with_cost_limit(CostLimitConfig {
//!         daily_limit_usd: Some(100.0),
//!         monthly_limit_usd: Some(1000.0),
//!         ..Default::default()
//!     });
//!
//! let provider = TenantProvider::new(inner_provider, config);
//! ```

use std::collections::{HashMap, HashSet};
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::Stream;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{
    BatchJob, BatchRequest, BatchResult, CompletionRequest, CompletionResponse, StreamChunk,
    TokenCountRequest, TokenCountResult,
};

/// Unique identifier for a tenant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TenantId(String);

impl TenantId {
    /// Create a new tenant ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the ID as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for TenantId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<String> for TenantId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl From<&str> for TenantId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

/// Rate limiting configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Maximum requests per minute
    pub requests_per_minute: Option<u32>,
    /// Maximum requests per hour
    pub requests_per_hour: Option<u32>,
    /// Maximum requests per day
    pub requests_per_day: Option<u32>,
    /// Maximum tokens per minute (input + output)
    pub tokens_per_minute: Option<u64>,
    /// Maximum tokens per hour
    pub tokens_per_hour: Option<u64>,
    /// Maximum tokens per day
    pub tokens_per_day: Option<u64>,
    /// Maximum concurrent requests
    pub max_concurrent: Option<u32>,
}

impl RateLimitConfig {
    /// Create a basic rate limit config.
    pub fn basic(requests_per_minute: u32, tokens_per_minute: u64) -> Self {
        Self {
            requests_per_minute: Some(requests_per_minute),
            tokens_per_minute: Some(tokens_per_minute),
            ..Default::default()
        }
    }

    /// Set the max concurrent requests.
    pub fn with_max_concurrent(mut self, max: u32) -> Self {
        self.max_concurrent = Some(max);
        self
    }
}

/// Cost limiting configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CostLimitConfig {
    /// Maximum cost per request in USD
    pub per_request_limit_usd: Option<f64>,
    /// Maximum daily cost in USD
    pub daily_limit_usd: Option<f64>,
    /// Maximum weekly cost in USD
    pub weekly_limit_usd: Option<f64>,
    /// Maximum monthly cost in USD
    pub monthly_limit_usd: Option<f64>,
    /// Alert threshold as percentage of limit (0.0 to 1.0)
    pub alert_threshold: Option<f64>,
}

impl CostLimitConfig {
    /// Create a basic cost limit config.
    pub fn basic(daily_limit: f64, monthly_limit: f64) -> Self {
        Self {
            daily_limit_usd: Some(daily_limit),
            monthly_limit_usd: Some(monthly_limit),
            ..Default::default()
        }
    }

    /// Set alert threshold.
    pub fn with_alert_threshold(mut self, threshold: f64) -> Self {
        self.alert_threshold = Some(threshold.clamp(0.0, 1.0));
        self
    }
}

/// Tenant configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantConfig {
    /// Unique tenant identifier
    pub id: TenantId,
    /// Display name
    pub name: Option<String>,
    /// Allowed models (empty = all allowed)
    pub allowed_models: HashSet<String>,
    /// Blocked models
    pub blocked_models: HashSet<String>,
    /// Rate limit configuration
    pub rate_limit: Option<RateLimitConfig>,
    /// Cost limit configuration
    pub cost_limit: Option<CostLimitConfig>,
    /// Whether the tenant is active
    pub active: bool,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl TenantConfig {
    /// Create a new tenant config.
    pub fn new(id: impl Into<TenantId>) -> Self {
        Self {
            id: id.into(),
            name: None,
            allowed_models: HashSet::new(),
            blocked_models: HashSet::new(),
            rate_limit: None,
            cost_limit: None,
            active: true,
            metadata: HashMap::new(),
        }
    }

    /// Set the display name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set allowed models.
    pub fn with_allowed_models<I, S>(mut self, models: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_models = models.into_iter().map(Into::into).collect();
        self
    }

    /// Add an allowed model.
    pub fn allow_model(mut self, model: impl Into<String>) -> Self {
        self.allowed_models.insert(model.into());
        self
    }

    /// Block a model.
    pub fn block_model(mut self, model: impl Into<String>) -> Self {
        self.blocked_models.insert(model.into());
        self
    }

    /// Set rate limit configuration.
    pub fn with_rate_limit(mut self, config: RateLimitConfig) -> Self {
        self.rate_limit = Some(config);
        self
    }

    /// Set cost limit configuration.
    pub fn with_cost_limit(mut self, config: CostLimitConfig) -> Self {
        self.cost_limit = Some(config);
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Check if a model is allowed for this tenant.
    pub fn is_model_allowed(&self, model: &str) -> bool {
        // If blocked, not allowed
        if self.blocked_models.contains(model) {
            return false;
        }

        // If allowed list is empty, all non-blocked models are allowed
        if self.allowed_models.is_empty() {
            return true;
        }

        // Otherwise, must be in allowed list
        self.allowed_models.contains(model)
    }
}

/// Rate limiter state.
#[derive(Debug)]
struct RateLimiterState {
    /// Request counts per window
    requests_minute: AtomicU32,
    requests_hour: AtomicU32,
    requests_day: AtomicU32,
    /// Token counts per window
    tokens_minute: AtomicU64,
    tokens_hour: AtomicU64,
    tokens_day: AtomicU64,
    /// Current concurrent requests
    concurrent: AtomicU32,
    /// Window start times
    minute_start: RwLock<Instant>,
    hour_start: RwLock<Instant>,
    day_start: RwLock<Instant>,
}

impl Default for RateLimiterState {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            requests_minute: AtomicU32::new(0),
            requests_hour: AtomicU32::new(0),
            requests_day: AtomicU32::new(0),
            tokens_minute: AtomicU64::new(0),
            tokens_hour: AtomicU64::new(0),
            tokens_day: AtomicU64::new(0),
            concurrent: AtomicU32::new(0),
            minute_start: RwLock::new(now),
            hour_start: RwLock::new(now),
            day_start: RwLock::new(now),
        }
    }
}

impl RateLimiterState {
    fn reset_if_needed(&self) {
        let now = Instant::now();

        // Check minute window
        {
            let mut minute_start = self.minute_start.write();
            if now.duration_since(*minute_start) >= Duration::from_secs(60) {
                *minute_start = now;
                self.requests_minute.store(0, Ordering::Relaxed);
                self.tokens_minute.store(0, Ordering::Relaxed);
            }
        }

        // Check hour window
        {
            let mut hour_start = self.hour_start.write();
            if now.duration_since(*hour_start) >= Duration::from_secs(3600) {
                *hour_start = now;
                self.requests_hour.store(0, Ordering::Relaxed);
                self.tokens_hour.store(0, Ordering::Relaxed);
            }
        }

        // Check day window
        {
            let mut day_start = self.day_start.write();
            if now.duration_since(*day_start) >= Duration::from_secs(86400) {
                *day_start = now;
                self.requests_day.store(0, Ordering::Relaxed);
                self.tokens_day.store(0, Ordering::Relaxed);
            }
        }
    }
}

/// Cost tracker state.
#[derive(Debug, Default)]
struct CostTrackerState {
    /// Cost in microdollars
    daily_cost: AtomicU64,
    weekly_cost: AtomicU64,
    monthly_cost: AtomicU64,
    /// Window start times (Unix timestamp ms)
    day_start_ms: AtomicU64,
    week_start_ms: AtomicU64,
    month_start_ms: AtomicU64,
}

impl CostTrackerState {
    fn new() -> Self {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            daily_cost: AtomicU64::new(0),
            weekly_cost: AtomicU64::new(0),
            monthly_cost: AtomicU64::new(0),
            day_start_ms: AtomicU64::new(now_ms),
            week_start_ms: AtomicU64::new(now_ms),
            month_start_ms: AtomicU64::new(now_ms),
        }
    }

    fn reset_if_needed(&self) {
        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        let day_ms = 86400 * 1000;
        let week_ms = 7 * day_ms;
        let month_ms = 30 * day_ms;

        let day_start = self.day_start_ms.load(Ordering::Relaxed);
        if now_ms - day_start >= day_ms {
            self.day_start_ms.store(now_ms, Ordering::Relaxed);
            self.daily_cost.store(0, Ordering::Relaxed);
        }

        let week_start = self.week_start_ms.load(Ordering::Relaxed);
        if now_ms - week_start >= week_ms {
            self.week_start_ms.store(now_ms, Ordering::Relaxed);
            self.weekly_cost.store(0, Ordering::Relaxed);
        }

        let month_start = self.month_start_ms.load(Ordering::Relaxed);
        if now_ms - month_start >= month_ms {
            self.month_start_ms.store(now_ms, Ordering::Relaxed);
            self.monthly_cost.store(0, Ordering::Relaxed);
        }
    }

    fn add_cost(&self, cost_usd: f64) {
        let microdollars = (cost_usd * 1_000_000.0) as u64;
        self.daily_cost.fetch_add(microdollars, Ordering::Relaxed);
        self.weekly_cost.fetch_add(microdollars, Ordering::Relaxed);
        self.monthly_cost.fetch_add(microdollars, Ordering::Relaxed);
    }

    fn daily_cost_usd(&self) -> f64 {
        self.daily_cost.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    fn weekly_cost_usd(&self) -> f64 {
        self.weekly_cost.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    fn monthly_cost_usd(&self) -> f64 {
        self.monthly_cost.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }
}

/// Error when rate limit is exceeded.
#[derive(Debug, Clone)]
pub struct RateLimitExceeded {
    /// Type of limit exceeded
    pub limit_type: RateLimitType,
    /// Current value
    pub current: u64,
    /// Maximum allowed
    pub limit: u64,
    /// Time until reset (if known)
    pub retry_after: Option<Duration>,
}

/// Type of rate limit that was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateLimitType {
    RequestsPerMinute,
    RequestsPerHour,
    RequestsPerDay,
    TokensPerMinute,
    TokensPerHour,
    TokensPerDay,
    Concurrent,
}

impl std::fmt::Display for RateLimitType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RequestsPerMinute => write!(f, "requests per minute"),
            Self::RequestsPerHour => write!(f, "requests per hour"),
            Self::RequestsPerDay => write!(f, "requests per day"),
            Self::TokensPerMinute => write!(f, "tokens per minute"),
            Self::TokensPerHour => write!(f, "tokens per hour"),
            Self::TokensPerDay => write!(f, "tokens per day"),
            Self::Concurrent => write!(f, "concurrent requests"),
        }
    }
}

/// Error when cost limit is exceeded.
#[derive(Debug, Clone)]
pub struct CostLimitExceeded {
    /// Type of limit exceeded
    pub limit_type: CostLimitType,
    /// Current cost in USD
    pub current_usd: f64,
    /// Maximum allowed in USD
    pub limit_usd: f64,
}

/// Type of cost limit that was exceeded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostLimitType {
    Daily,
    Weekly,
    Monthly,
}

impl std::fmt::Display for CostLimitType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Daily => write!(f, "daily"),
            Self::Weekly => write!(f, "weekly"),
            Self::Monthly => write!(f, "monthly"),
        }
    }
}

/// Tenant error types.
#[derive(Debug)]
pub enum TenantError {
    /// Tenant is not active
    Inactive,
    /// Model is not allowed
    ModelNotAllowed(String),
    /// Rate limit exceeded
    RateLimitExceeded(RateLimitExceeded),
    /// Cost limit exceeded
    CostLimitExceeded(CostLimitExceeded),
}

impl std::fmt::Display for TenantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Inactive => write!(f, "Tenant is inactive"),
            Self::ModelNotAllowed(model) => write!(f, "Model '{}' is not allowed", model),
            Self::RateLimitExceeded(info) => {
                write!(
                    f,
                    "Rate limit exceeded: {} ({}/{})",
                    info.limit_type, info.current, info.limit
                )
            }
            Self::CostLimitExceeded(info) => {
                write!(
                    f,
                    "Cost limit exceeded: {} (${:.2}/${:.2})",
                    info.limit_type, info.current_usd, info.limit_usd
                )
            }
        }
    }
}

impl std::error::Error for TenantError {}

/// Provider wrapper that enforces tenant restrictions.
pub struct TenantProvider<P: Provider> {
    inner: P,
    config: TenantConfig,
    rate_state: RateLimiterState,
    cost_state: CostTrackerState,
}

impl<P: Provider> TenantProvider<P> {
    /// Create a new tenant provider.
    pub fn new(inner: P, config: TenantConfig) -> Self {
        Self {
            inner,
            config,
            rate_state: RateLimiterState::default(),
            cost_state: CostTrackerState::new(),
        }
    }

    /// Get the tenant ID.
    pub fn tenant_id(&self) -> &TenantId {
        &self.config.id
    }

    /// Get the tenant config.
    pub fn config(&self) -> &TenantConfig {
        &self.config
    }

    /// Check if a request is allowed.
    fn check_request(&self, model: &str) -> std::result::Result<(), TenantError> {
        // Check if tenant is active
        if !self.config.active {
            return Err(TenantError::Inactive);
        }

        // Check if model is allowed
        if !self.config.is_model_allowed(model) {
            return Err(TenantError::ModelNotAllowed(model.to_string()));
        }

        // Reset rate limiter windows if needed
        self.rate_state.reset_if_needed();

        // Check rate limits
        if let Some(ref limits) = self.config.rate_limit {
            // Check concurrent
            if let Some(max_concurrent) = limits.max_concurrent {
                let current = self.rate_state.concurrent.load(Ordering::Relaxed);
                if current >= max_concurrent {
                    return Err(TenantError::RateLimitExceeded(RateLimitExceeded {
                        limit_type: RateLimitType::Concurrent,
                        current: current as u64,
                        limit: max_concurrent as u64,
                        retry_after: None,
                    }));
                }
            }

            // Check requests per minute
            if let Some(rpm) = limits.requests_per_minute {
                let current = self.rate_state.requests_minute.load(Ordering::Relaxed);
                if current >= rpm {
                    return Err(TenantError::RateLimitExceeded(RateLimitExceeded {
                        limit_type: RateLimitType::RequestsPerMinute,
                        current: current as u64,
                        limit: rpm as u64,
                        retry_after: Some(Duration::from_secs(60)),
                    }));
                }
            }

            // Check requests per hour
            if let Some(rph) = limits.requests_per_hour {
                let current = self.rate_state.requests_hour.load(Ordering::Relaxed);
                if current >= rph {
                    return Err(TenantError::RateLimitExceeded(RateLimitExceeded {
                        limit_type: RateLimitType::RequestsPerHour,
                        current: current as u64,
                        limit: rph as u64,
                        retry_after: Some(Duration::from_secs(3600)),
                    }));
                }
            }

            // Check requests per day
            if let Some(rpd) = limits.requests_per_day {
                let current = self.rate_state.requests_day.load(Ordering::Relaxed);
                if current >= rpd {
                    return Err(TenantError::RateLimitExceeded(RateLimitExceeded {
                        limit_type: RateLimitType::RequestsPerDay,
                        current: current as u64,
                        limit: rpd as u64,
                        retry_after: Some(Duration::from_secs(86400)),
                    }));
                }
            }
        }

        // Reset cost windows if needed
        self.cost_state.reset_if_needed();

        // Check cost limits
        if let Some(ref limits) = self.config.cost_limit {
            if let Some(daily) = limits.daily_limit_usd {
                let current = self.cost_state.daily_cost_usd();
                if current >= daily {
                    return Err(TenantError::CostLimitExceeded(CostLimitExceeded {
                        limit_type: CostLimitType::Daily,
                        current_usd: current,
                        limit_usd: daily,
                    }));
                }
            }

            if let Some(weekly) = limits.weekly_limit_usd {
                let current = self.cost_state.weekly_cost_usd();
                if current >= weekly {
                    return Err(TenantError::CostLimitExceeded(CostLimitExceeded {
                        limit_type: CostLimitType::Weekly,
                        current_usd: current,
                        limit_usd: weekly,
                    }));
                }
            }

            if let Some(monthly) = limits.monthly_limit_usd {
                let current = self.cost_state.monthly_cost_usd();
                if current >= monthly {
                    return Err(TenantError::CostLimitExceeded(CostLimitExceeded {
                        limit_type: CostLimitType::Monthly,
                        current_usd: current,
                        limit_usd: monthly,
                    }));
                }
            }
        }

        Ok(())
    }

    /// Record a completed request.
    fn record_request(&self, tokens: u64, cost_usd: f64) {
        // Increment request counters
        self.rate_state
            .requests_minute
            .fetch_add(1, Ordering::Relaxed);
        self.rate_state
            .requests_hour
            .fetch_add(1, Ordering::Relaxed);
        self.rate_state.requests_day.fetch_add(1, Ordering::Relaxed);

        // Increment token counters
        self.rate_state
            .tokens_minute
            .fetch_add(tokens, Ordering::Relaxed);
        self.rate_state
            .tokens_hour
            .fetch_add(tokens, Ordering::Relaxed);
        self.rate_state
            .tokens_day
            .fetch_add(tokens, Ordering::Relaxed);

        // Record cost
        self.cost_state.add_cost(cost_usd);
    }

    /// Start a request (increment concurrent counter).
    fn start_request(&self) {
        self.rate_state.concurrent.fetch_add(1, Ordering::Relaxed);
    }

    /// End a request (decrement concurrent counter).
    fn end_request(&self) {
        self.rate_state.concurrent.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current usage stats.
    pub fn usage_stats(&self) -> TenantUsageStats {
        self.rate_state.reset_if_needed();
        self.cost_state.reset_if_needed();

        TenantUsageStats {
            requests_minute: self.rate_state.requests_minute.load(Ordering::Relaxed),
            requests_hour: self.rate_state.requests_hour.load(Ordering::Relaxed),
            requests_day: self.rate_state.requests_day.load(Ordering::Relaxed),
            tokens_minute: self.rate_state.tokens_minute.load(Ordering::Relaxed),
            tokens_hour: self.rate_state.tokens_hour.load(Ordering::Relaxed),
            tokens_day: self.rate_state.tokens_day.load(Ordering::Relaxed),
            concurrent: self.rate_state.concurrent.load(Ordering::Relaxed),
            daily_cost_usd: self.cost_state.daily_cost_usd(),
            weekly_cost_usd: self.cost_state.weekly_cost_usd(),
            monthly_cost_usd: self.cost_state.monthly_cost_usd(),
        }
    }
}

/// Current usage statistics for a tenant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TenantUsageStats {
    /// Requests in current minute
    pub requests_minute: u32,
    /// Requests in current hour
    pub requests_hour: u32,
    /// Requests in current day
    pub requests_day: u32,
    /// Tokens in current minute
    pub tokens_minute: u64,
    /// Tokens in current hour
    pub tokens_hour: u64,
    /// Tokens in current day
    pub tokens_day: u64,
    /// Current concurrent requests
    pub concurrent: u32,
    /// Cost today in USD
    pub daily_cost_usd: f64,
    /// Cost this week in USD
    pub weekly_cost_usd: f64,
    /// Cost this month in USD
    pub monthly_cost_usd: f64,
}

#[async_trait]
impl<P: Provider> Provider for TenantProvider<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        // Check if request is allowed
        self.check_request(&request.model)
            .map_err(|e| Error::other(e.to_string()))?;

        self.start_request();
        let result = self.inner.complete(request).await;
        self.end_request();

        if let Ok(ref response) = result {
            let tokens = (response.usage.input_tokens + response.usage.output_tokens) as u64;
            // Estimate cost - in production, use actual pricing
            let cost_usd = tokens as f64 * 0.000001; // Placeholder
            self.record_request(tokens, cost_usd);
        }

        result
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // Check if request is allowed
        self.check_request(&request.model)
            .map_err(|e| Error::other(e.to_string()))?;

        self.start_request();
        // Note: For proper tracking, we'd need to wrap the stream
        // to track completion and tokens
        self.inner.complete_stream(request).await
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

    fn supports_token_counting(&self) -> bool {
        self.inner.supports_token_counting()
    }

    async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCountResult> {
        self.inner.count_tokens(request).await
    }

    fn supports_batch(&self) -> bool {
        self.inner.supports_batch()
    }

    async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchJob> {
        self.inner.create_batch(requests).await
    }

    async fn get_batch(&self, batch_id: &str) -> Result<BatchJob> {
        self.inner.get_batch(batch_id).await
    }

    async fn get_batch_results(&self, batch_id: &str) -> Result<Vec<BatchResult>> {
        self.inner.get_batch_results(batch_id).await
    }

    async fn cancel_batch(&self, batch_id: &str) -> Result<BatchJob> {
        self.inner.cancel_batch(batch_id).await
    }

    async fn list_batches(&self, limit: Option<u32>) -> Result<Vec<BatchJob>> {
        self.inner.list_batches(limit).await
    }
}

/// Manager for multiple tenants.
pub struct TenantManager {
    tenants: RwLock<HashMap<TenantId, TenantConfig>>,
}

impl Default for TenantManager {
    fn default() -> Self {
        Self::new()
    }
}

impl TenantManager {
    /// Create a new tenant manager.
    pub fn new() -> Self {
        Self {
            tenants: RwLock::new(HashMap::new()),
        }
    }

    /// Register a tenant.
    pub fn register(&self, config: TenantConfig) {
        self.tenants.write().insert(config.id.clone(), config);
    }

    /// Get a tenant config.
    pub fn get(&self, id: &TenantId) -> Option<TenantConfig> {
        self.tenants.read().get(id).cloned()
    }

    /// Remove a tenant.
    pub fn remove(&self, id: &TenantId) -> Option<TenantConfig> {
        self.tenants.write().remove(id)
    }

    /// List all tenant IDs.
    pub fn list(&self) -> Vec<TenantId> {
        self.tenants.read().keys().cloned().collect()
    }

    /// Check if a tenant exists.
    pub fn exists(&self, id: &TenantId) -> bool {
        self.tenants.read().contains_key(id)
    }

    /// Update a tenant config.
    pub fn update(&self, config: TenantConfig) -> bool {
        let mut tenants = self.tenants.write();
        if tenants.contains_key(&config.id) {
            tenants.insert(config.id.clone(), config);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tenant_id() {
        let id = TenantId::new("test-tenant");
        assert_eq!(id.as_str(), "test-tenant");
        assert_eq!(id.to_string(), "test-tenant");
    }

    #[test]
    fn test_tenant_config_allowed_models() {
        let config = TenantConfig::new("test")
            .with_allowed_models(vec!["gpt-4o", "claude-sonnet-4-20250514"]);

        assert!(config.is_model_allowed("gpt-4o"));
        assert!(config.is_model_allowed("claude-sonnet-4-20250514"));
        assert!(!config.is_model_allowed("gpt-3.5-turbo"));
    }

    #[test]
    fn test_tenant_config_blocked_models() {
        let config = TenantConfig::new("test").block_model("gpt-3.5-turbo");

        assert!(config.is_model_allowed("gpt-4o"));
        assert!(!config.is_model_allowed("gpt-3.5-turbo"));
    }

    #[test]
    fn test_rate_limit_config() {
        let config = RateLimitConfig::basic(60, 100_000).with_max_concurrent(10);

        assert_eq!(config.requests_per_minute, Some(60));
        assert_eq!(config.tokens_per_minute, Some(100_000));
        assert_eq!(config.max_concurrent, Some(10));
    }

    #[test]
    fn test_cost_limit_config() {
        let config = CostLimitConfig::basic(100.0, 1000.0).with_alert_threshold(0.8);

        assert_eq!(config.daily_limit_usd, Some(100.0));
        assert_eq!(config.monthly_limit_usd, Some(1000.0));
        assert_eq!(config.alert_threshold, Some(0.8));
    }

    #[test]
    fn test_tenant_manager() {
        let manager = TenantManager::new();

        let config = TenantConfig::new("acme");
        manager.register(config);

        assert!(manager.exists(&TenantId::new("acme")));
        assert!(!manager.exists(&TenantId::new("other")));

        let ids = manager.list();
        assert_eq!(ids.len(), 1);

        let removed = manager.remove(&TenantId::new("acme"));
        assert!(removed.is_some());
        assert!(!manager.exists(&TenantId::new("acme")));
    }
}
