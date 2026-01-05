//! Cost tracking and usage metering for LLM requests.
//!
//! This module provides usage tracking, cost calculation, and metering infrastructure
//! for monitoring LLM API usage across providers and tenants.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::{MeteringProvider, InMemoryMeteringSink};
//!
//! let sink = Arc::new(InMemoryMeteringSink::new());
//! let provider = MeteringProvider::new(anthropic_provider, sink.clone())
//!     .with_tenant("acme-corp");
//!
//! // Make requests...
//!
//! // Query usage
//! let usage = sink.query(UsageFilter {
//!     tenant_id: Some("acme-corp".into()),
//!     since: Some(Utc::now() - Duration::days(7)),
//!     ..Default::default()
//! }).await?;
//! ```

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use futures::Stream;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::Result;
use crate::models::get_model_info;
use crate::provider::Provider;
use crate::types::{
    BatchJob, BatchRequest, BatchResult, CompletionRequest, CompletionResponse, StreamChunk,
    TokenCountRequest, TokenCountResult,
};

/// Usage record for a single request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageRecord {
    /// Unique request identifier
    pub request_id: String,
    /// Model used
    pub model: String,
    /// Provider name
    pub provider: String,
    /// Input tokens
    pub input_tokens: u32,
    /// Output tokens
    pub output_tokens: u32,
    /// Cached tokens (read)
    pub cached_tokens: u32,
    /// Estimated cost in USD
    pub cost_usd: f64,
    /// Unix timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Latency in milliseconds
    pub latency_ms: u64,
    /// Tenant identifier
    pub tenant_id: Option<String>,
    /// Additional metadata
    pub metadata: HashMap<String, Value>,
}

impl UsageRecord {
    /// Create a new usage record.
    pub fn new(
        request_id: impl Into<String>,
        model: impl Into<String>,
        provider: impl Into<String>,
    ) -> Self {
        let timestamp_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;

        Self {
            request_id: request_id.into(),
            model: model.into(),
            provider: provider.into(),
            input_tokens: 0,
            output_tokens: 0,
            cached_tokens: 0,
            cost_usd: 0.0,
            timestamp_ms,
            latency_ms: 0,
            tenant_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Set token counts and calculate cost.
    pub fn with_tokens(mut self, input: u32, output: u32, cached: u32) -> Self {
        self.input_tokens = input;
        self.output_tokens = output;
        self.cached_tokens = cached;

        // Try to calculate cost from model registry
        if let Some(info) = get_model_info(&self.model) {
            let p = &info.pricing;
            let input_cost = p.input_per_1m * (input as f64 / 1_000_000.0);
            let output_cost = p.output_per_1m * (output as f64 / 1_000_000.0);
            self.cost_usd = input_cost + output_cost;
        }

        self
    }

    /// Set the cost explicitly.
    pub fn with_cost(mut self, cost_usd: f64) -> Self {
        self.cost_usd = cost_usd;
        self
    }

    /// Set the latency.
    pub fn with_latency(mut self, latency_ms: u64) -> Self {
        self.latency_ms = latency_ms;
        self
    }

    /// Set the tenant.
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Add metadata.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }
}

/// Filter for querying usage records.
#[derive(Debug, Clone, Default)]
pub struct UsageFilter {
    /// Filter by tenant ID
    pub tenant_id: Option<String>,
    /// Filter by model
    pub model: Option<String>,
    /// Filter by provider
    pub provider: Option<String>,
    /// Records since this timestamp (Unix ms)
    pub since_ms: Option<u64>,
    /// Records until this timestamp (Unix ms)
    pub until_ms: Option<u64>,
    /// Maximum number of records to return
    pub limit: Option<usize>,
}

impl UsageFilter {
    /// Create a filter for a specific tenant.
    pub fn for_tenant(tenant_id: impl Into<String>) -> Self {
        Self {
            tenant_id: Some(tenant_id.into()),
            ..Default::default()
        }
    }

    /// Filter by model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// Filter by provider.
    pub fn with_provider(mut self, provider: impl Into<String>) -> Self {
        self.provider = Some(provider.into());
        self
    }

    /// Filter records since a duration ago.
    pub fn since_duration(mut self, duration: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        self.since_ms = Some(now.saturating_sub(duration.as_millis() as u64));
        self
    }

    /// Limit the number of records.
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    /// Check if a record matches this filter.
    pub fn matches(&self, record: &UsageRecord) -> bool {
        if let Some(ref tenant) = self.tenant_id {
            if record.tenant_id.as_ref() != Some(tenant) {
                return false;
            }
        }

        if let Some(ref model) = self.model {
            if &record.model != model {
                return false;
            }
        }

        if let Some(ref provider) = self.provider {
            if &record.provider != provider {
                return false;
            }
        }

        if let Some(since) = self.since_ms {
            if record.timestamp_ms < since {
                return false;
            }
        }

        if let Some(until) = self.until_ms {
            if record.timestamp_ms > until {
                return false;
            }
        }

        true
    }
}

/// Aggregated usage statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UsageStats {
    /// Total number of requests
    pub total_requests: u64,
    /// Total input tokens
    pub total_input_tokens: u64,
    /// Total output tokens
    pub total_output_tokens: u64,
    /// Total cached tokens
    pub total_cached_tokens: u64,
    /// Total cost in USD
    pub total_cost_usd: f64,
    /// Average latency in milliseconds
    pub avg_latency_ms: f64,
    /// Breakdown by model
    pub by_model: HashMap<String, ModelStats>,
    /// Breakdown by tenant
    pub by_tenant: HashMap<String, TenantStats>,
}

/// Statistics for a specific model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelStats {
    /// Number of requests
    pub requests: u64,
    /// Total tokens (input + output)
    pub total_tokens: u64,
    /// Total cost
    pub cost_usd: f64,
}

/// Statistics for a specific tenant.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TenantStats {
    /// Number of requests
    pub requests: u64,
    /// Total tokens
    pub total_tokens: u64,
    /// Total cost
    pub cost_usd: f64,
}

/// Trait for metering sinks that persist usage data.
#[async_trait]
pub trait MeteringSink: Send + Sync {
    /// Record a usage entry.
    async fn record(&self, usage: UsageRecord) -> Result<()>;

    /// Query usage records.
    async fn query(&self, filter: UsageFilter) -> Result<Vec<UsageRecord>>;

    /// Get aggregated statistics.
    async fn stats(&self, filter: UsageFilter) -> Result<UsageStats>;

    /// Clear all records (for testing).
    async fn clear(&self) -> Result<()>;
}

/// In-memory metering sink for development and testing.
pub struct InMemoryMeteringSink {
    records: RwLock<Vec<UsageRecord>>,
    request_count: AtomicU64,
}

impl Default for InMemoryMeteringSink {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryMeteringSink {
    /// Create a new in-memory sink.
    pub fn new() -> Self {
        Self {
            records: RwLock::new(Vec::new()),
            request_count: AtomicU64::new(0),
        }
    }

    /// Get the number of recorded requests.
    pub fn count(&self) -> u64 {
        self.request_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl MeteringSink for InMemoryMeteringSink {
    async fn record(&self, usage: UsageRecord) -> Result<()> {
        self.records.write().push(usage);
        self.request_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    async fn query(&self, filter: UsageFilter) -> Result<Vec<UsageRecord>> {
        let records = self.records.read();
        let mut results: Vec<UsageRecord> = records
            .iter()
            .filter(|r| filter.matches(r))
            .cloned()
            .collect();

        // Sort by timestamp descending (most recent first)
        results.sort_by(|a, b| b.timestamp_ms.cmp(&a.timestamp_ms));

        if let Some(limit) = filter.limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    async fn stats(&self, filter: UsageFilter) -> Result<UsageStats> {
        let records = self.records.read();
        let mut stats = UsageStats::default();
        let mut total_latency = 0u64;

        for record in records.iter().filter(|r| filter.matches(r)) {
            stats.total_requests += 1;
            stats.total_input_tokens += record.input_tokens as u64;
            stats.total_output_tokens += record.output_tokens as u64;
            stats.total_cached_tokens += record.cached_tokens as u64;
            stats.total_cost_usd += record.cost_usd;
            total_latency += record.latency_ms;

            // Model stats
            let model_stats = stats.by_model.entry(record.model.clone()).or_default();
            model_stats.requests += 1;
            model_stats.total_tokens += (record.input_tokens + record.output_tokens) as u64;
            model_stats.cost_usd += record.cost_usd;

            // Tenant stats
            if let Some(ref tenant) = record.tenant_id {
                let tenant_stats = stats.by_tenant.entry(tenant.clone()).or_default();
                tenant_stats.requests += 1;
                tenant_stats.total_tokens += (record.input_tokens + record.output_tokens) as u64;
                tenant_stats.cost_usd += record.cost_usd;
            }
        }

        if stats.total_requests > 0 {
            stats.avg_latency_ms = total_latency as f64 / stats.total_requests as f64;
        }

        Ok(stats)
    }

    async fn clear(&self) -> Result<()> {
        self.records.write().clear();
        self.request_count.store(0, Ordering::Relaxed);
        Ok(())
    }
}

/// Provider wrapper that records usage metrics.
pub struct MeteringProvider<P: Provider> {
    inner: P,
    sink: Arc<dyn MeteringSink>,
    tenant_id: Option<String>,
    metadata: HashMap<String, Value>,
}

impl<P: Provider> MeteringProvider<P> {
    /// Create a new metering provider.
    pub fn new(inner: P, sink: Arc<dyn MeteringSink>) -> Self {
        Self {
            inner,
            sink,
            tenant_id: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the tenant ID for all requests.
    pub fn with_tenant(mut self, tenant_id: impl Into<String>) -> Self {
        self.tenant_id = Some(tenant_id.into());
        self
    }

    /// Add metadata that will be included in all usage records.
    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<Value>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Record usage from a completion response.
    async fn record_usage(&self, model: &str, response: &CompletionResponse, latency_ms: u64) {
        let mut record = UsageRecord::new(&response.id, model, self.inner.name())
            .with_tokens(
                response.usage.input_tokens,
                response.usage.output_tokens,
                response.usage.cache_read_input_tokens,
            )
            .with_latency(latency_ms);

        if let Some(ref tenant) = self.tenant_id {
            record = record.with_tenant(tenant.clone());
        }

        for (key, value) in &self.metadata {
            record.metadata.insert(key.clone(), value.clone());
        }

        if let Err(e) = self.sink.record(record).await {
            tracing::warn!(error = %e, "Failed to record usage metrics");
        }
    }
}

#[async_trait]
impl<P: Provider> Provider for MeteringProvider<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let model = request.model.clone();
        let start = Instant::now();

        let response = self.inner.complete(request).await?;

        let latency_ms = start.elapsed().as_millis() as u64;
        self.record_usage(&model, &response, latency_ms).await;

        Ok(response)
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // For streaming, we record initial metrics but can't track exact token counts
        // until the stream completes (which would require wrapping the stream)
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

/// Simple cost tracker that just accumulates totals.
#[derive(Debug, Default)]
pub struct CostTracker {
    total_cost: AtomicU64, // Stored as microdollars for precision
    total_requests: AtomicU64,
    total_input_tokens: AtomicU64,
    total_output_tokens: AtomicU64,
}

impl CostTracker {
    /// Create a new cost tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a request's cost and tokens.
    pub fn record(&self, cost_usd: f64, input_tokens: u32, output_tokens: u32) {
        let microdollars = (cost_usd * 1_000_000.0) as u64;
        self.total_cost.fetch_add(microdollars, Ordering::Relaxed);
        self.total_requests.fetch_add(1, Ordering::Relaxed);
        self.total_input_tokens
            .fetch_add(input_tokens as u64, Ordering::Relaxed);
        self.total_output_tokens
            .fetch_add(output_tokens as u64, Ordering::Relaxed);
    }

    /// Get total cost in USD.
    pub fn total_cost_usd(&self) -> f64 {
        self.total_cost.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Get total request count.
    pub fn total_requests(&self) -> u64 {
        self.total_requests.load(Ordering::Relaxed)
    }

    /// Get total input tokens.
    pub fn total_input_tokens(&self) -> u64 {
        self.total_input_tokens.load(Ordering::Relaxed)
    }

    /// Get total output tokens.
    pub fn total_output_tokens(&self) -> u64 {
        self.total_output_tokens.load(Ordering::Relaxed)
    }

    /// Reset all counters.
    pub fn reset(&self) {
        self.total_cost.store(0, Ordering::Relaxed);
        self.total_requests.store(0, Ordering::Relaxed);
        self.total_input_tokens.store(0, Ordering::Relaxed);
        self.total_output_tokens.store(0, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_record_creation() {
        let record = UsageRecord::new("req-123", "gpt-4o", "openai")
            .with_tokens(100, 50, 0)
            .with_latency(1500)
            .with_tenant("acme")
            .with_metadata("key", "value");

        assert_eq!(record.request_id, "req-123");
        assert_eq!(record.model, "gpt-4o");
        assert_eq!(record.provider, "openai");
        assert_eq!(record.input_tokens, 100);
        assert_eq!(record.output_tokens, 50);
        assert_eq!(record.latency_ms, 1500);
        assert_eq!(record.tenant_id, Some("acme".to_string()));
        assert!(record.metadata.contains_key("key"));
    }

    #[test]
    fn test_usage_filter_matches() {
        let record = UsageRecord::new("req-123", "gpt-4o", "openai").with_tenant("acme");

        let filter = UsageFilter::for_tenant("acme");
        assert!(filter.matches(&record));

        let filter = UsageFilter::for_tenant("other");
        assert!(!filter.matches(&record));

        let filter = UsageFilter::default().with_model("gpt-4o");
        assert!(filter.matches(&record));

        let filter = UsageFilter::default().with_model("claude");
        assert!(!filter.matches(&record));
    }

    #[tokio::test]
    async fn test_in_memory_sink() {
        let sink = InMemoryMeteringSink::new();

        // Record some usage
        sink.record(UsageRecord::new("req-1", "gpt-4o", "openai").with_tenant("acme"))
            .await
            .unwrap();
        sink.record(UsageRecord::new("req-2", "claude", "anthropic").with_tenant("acme"))
            .await
            .unwrap();
        sink.record(UsageRecord::new("req-3", "gpt-4o", "openai").with_tenant("other"))
            .await
            .unwrap();

        assert_eq!(sink.count(), 3);

        // Query all
        let all = sink.query(UsageFilter::default()).await.unwrap();
        assert_eq!(all.len(), 3);

        // Query by tenant
        let acme = sink.query(UsageFilter::for_tenant("acme")).await.unwrap();
        assert_eq!(acme.len(), 2);

        // Query by model
        let gpt4o = sink
            .query(UsageFilter::default().with_model("gpt-4o"))
            .await
            .unwrap();
        assert_eq!(gpt4o.len(), 2);

        // Get stats
        let stats = sink.stats(UsageFilter::default()).await.unwrap();
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.by_model.len(), 2);
        assert_eq!(stats.by_tenant.len(), 2);
    }

    #[test]
    fn test_cost_tracker() {
        let tracker = CostTracker::new();

        tracker.record(0.01, 100, 50);
        tracker.record(0.02, 200, 100);

        assert!((tracker.total_cost_usd() - 0.03).abs() < 0.0001);
        assert_eq!(tracker.total_requests(), 2);
        assert_eq!(tracker.total_input_tokens(), 300);
        assert_eq!(tracker.total_output_tokens(), 150);

        tracker.reset();
        assert_eq!(tracker.total_requests(), 0);
    }
}
