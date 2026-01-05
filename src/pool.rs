//! Provider pool for load balancing and failover.
//!
//! This module provides load balancing across multiple deployments of the same provider,
//! with automatic health checking and failover support.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::{ProviderPool, RoutingStrategy};
//!
//! let pool = ProviderPool::builder()
//!     .add_deployment("openai-1", provider1, 1, 100)
//!     .add_deployment("openai-2", provider2, 1, 100)
//!     .strategy(RoutingStrategy::LeastLatency)
//!     .build();
//! ```

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::Stream;
use parking_lot::RwLock;
use tokio::sync::Semaphore;

use crate::error::{Error, Result};
use crate::provider::Provider;
use crate::types::{
    BatchJob, BatchRequest, BatchResult, CompletionRequest, CompletionResponse, StreamChunk,
    TokenCountRequest, TokenCountResult,
};

/// Configuration for a single deployment (provider instance).
#[derive(Clone)]
pub struct DeploymentConfig {
    /// Unique name for this deployment
    pub name: String,
    /// The underlying provider
    pub provider: Arc<dyn Provider>,
    /// Weight for weighted routing (higher = more traffic)
    pub weight: u32,
    /// Priority for failover (lower = higher priority, used first)
    pub priority: u32,
    /// Maximum concurrent requests (None = unlimited)
    pub max_concurrent: Option<u32>,
    /// Optional region identifier
    pub region: Option<String>,
}

/// Health status of a deployment.
#[derive(Debug, Clone)]
pub struct DeploymentHealth {
    /// Whether the deployment is currently healthy
    pub healthy: bool,
    /// Time of last health check
    pub last_check: Option<Instant>,
    /// Number of consecutive failures
    pub consecutive_failures: u32,
    /// Number of consecutive successes (for recovery)
    pub consecutive_successes: u32,
    /// P50 latency in milliseconds
    pub latency_p50_ms: u64,
    /// P99 latency in milliseconds
    pub latency_p99_ms: u64,
    /// Current number of in-flight requests
    pub requests_in_flight: u32,
    /// Total requests handled
    pub total_requests: u64,
    /// Total errors
    pub total_errors: u64,
}

impl Default for DeploymentHealth {
    fn default() -> Self {
        Self {
            healthy: true,
            last_check: None,
            consecutive_failures: 0,
            consecutive_successes: 0,
            latency_p50_ms: 0,
            latency_p99_ms: 0,
            requests_in_flight: 0,
            total_requests: 0,
            total_errors: 0,
        }
    }
}

/// Routing strategy for the provider pool.
#[derive(Debug, Clone, Default)]
pub enum RoutingStrategy {
    /// Simple round-robin distribution
    #[default]
    RoundRobin,
    /// Route to the deployment with lowest latency
    LeastLatency,
    /// Route to the deployment with fewest in-flight requests
    LeastConnections,
    /// Weighted random distribution based on weights
    Weighted,
    /// Random distribution
    Random,
    /// Use primary (lowest priority), failover on error
    Primary,
}

/// Configuration for health checking.
#[derive(Debug, Clone)]
pub struct HealthCheckConfig {
    /// Interval between health checks
    pub interval: Duration,
    /// Timeout for health check requests
    pub timeout: Duration,
    /// Number of consecutive failures before marking unhealthy
    pub failure_threshold: u32,
    /// Number of consecutive successes before marking healthy again
    pub recovery_threshold: u32,
    /// Whether to enable active health checking
    pub enabled: bool,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(10),
            failure_threshold: 3,
            recovery_threshold: 2,
            enabled: true,
        }
    }
}

/// Internal state for tracking routing.
struct PoolState {
    /// Round-robin counter
    rr_counter: AtomicU32,
    /// Health status per deployment
    health: RwLock<HashMap<String, DeploymentHealth>>,
    /// Concurrency limiters per deployment
    semaphores: HashMap<String, Arc<Semaphore>>,
    /// Latency samples for calculating percentiles (ring buffer)
    latency_samples: RwLock<HashMap<String, Vec<u64>>>,
}

/// Provider pool for load balancing across multiple deployments.
pub struct ProviderPool {
    /// Deployments in the pool
    deployments: Vec<DeploymentConfig>,
    /// Routing strategy
    strategy: RoutingStrategy,
    /// Health check configuration
    health_config: HealthCheckConfig,
    /// Internal state
    state: PoolState,
    /// Pool name (for logging)
    name: String,
}

impl ProviderPool {
    /// Create a new pool builder.
    pub fn builder() -> ProviderPoolBuilder {
        ProviderPoolBuilder::new()
    }

    /// Get all deployments.
    pub fn deployments(&self) -> &[DeploymentConfig] {
        &self.deployments
    }

    /// Get health status for a deployment.
    pub fn health(&self, deployment_name: &str) -> Option<DeploymentHealth> {
        self.state.health.read().get(deployment_name).cloned()
    }

    /// Get health status for all deployments.
    pub fn all_health(&self) -> HashMap<String, DeploymentHealth> {
        self.state.health.read().clone()
    }

    /// Get the number of healthy deployments.
    pub fn healthy_count(&self) -> usize {
        self.state
            .health
            .read()
            .values()
            .filter(|h| h.healthy)
            .count()
    }

    /// Select a deployment based on the routing strategy.
    fn select_deployment(&self) -> Result<&DeploymentConfig> {
        let healthy_deployments: Vec<_> = self
            .deployments
            .iter()
            .filter(|d| {
                self.state
                    .health
                    .read()
                    .get(&d.name)
                    .map(|h| h.healthy)
                    .unwrap_or(true)
            })
            .collect();

        if healthy_deployments.is_empty() {
            return Err(Error::other("No healthy deployments available"));
        }

        let selected = match &self.strategy {
            RoutingStrategy::RoundRobin => {
                let idx = self.state.rr_counter.fetch_add(1, Ordering::Relaxed) as usize;
                healthy_deployments[idx % healthy_deployments.len()]
            }
            RoutingStrategy::LeastLatency => {
                let health = self.state.health.read();
                healthy_deployments
                    .iter()
                    .min_by_key(|d| {
                        health
                            .get(&d.name)
                            .map(|h| h.latency_p50_ms)
                            .unwrap_or(u64::MAX)
                    })
                    .copied()
                    .unwrap_or(healthy_deployments[0])
            }
            RoutingStrategy::LeastConnections => {
                let health = self.state.health.read();
                healthy_deployments
                    .iter()
                    .min_by_key(|d| {
                        health
                            .get(&d.name)
                            .map(|h| h.requests_in_flight)
                            .unwrap_or(0)
                    })
                    .copied()
                    .unwrap_or(healthy_deployments[0])
            }
            RoutingStrategy::Weighted => {
                let total_weight: u32 = healthy_deployments.iter().map(|d| d.weight).sum();
                if total_weight == 0 {
                    healthy_deployments[0]
                } else {
                    let random = rand_u32() % total_weight;
                    let mut cumulative = 0;
                    healthy_deployments
                        .iter()
                        .find(|d| {
                            cumulative += d.weight;
                            cumulative > random
                        })
                        .copied()
                        .unwrap_or(healthy_deployments[0])
                }
            }
            RoutingStrategy::Random => {
                let idx = rand_u32() as usize % healthy_deployments.len();
                healthy_deployments[idx]
            }
            RoutingStrategy::Primary => {
                // Sort by priority (lowest first) and pick the first healthy one
                let mut sorted = healthy_deployments.clone();
                sorted.sort_by_key(|d| d.priority);
                sorted[0]
            }
        };

        Ok(selected)
    }

    /// Record a successful request.
    fn record_success(&self, deployment_name: &str, latency_ms: u64) {
        let mut health = self.state.health.write();
        if let Some(h) = health.get_mut(deployment_name) {
            h.total_requests += 1;
            h.consecutive_successes += 1;
            h.consecutive_failures = 0;

            // Recovery check
            if !h.healthy && h.consecutive_successes >= self.health_config.recovery_threshold {
                h.healthy = true;
                tracing::info!(deployment = deployment_name, "Deployment recovered");
            }
        }

        // Update latency samples
        let mut samples = self.state.latency_samples.write();
        let entry = samples.entry(deployment_name.to_string()).or_default();
        if entry.len() >= 100 {
            entry.remove(0);
        }
        entry.push(latency_ms);

        // Update percentiles
        if let Some(h) = health.get_mut(deployment_name) {
            if let Some(samples) = samples.get(deployment_name) {
                let mut sorted = samples.clone();
                sorted.sort_unstable();
                if !sorted.is_empty() {
                    h.latency_p50_ms = sorted[sorted.len() / 2];
                    h.latency_p99_ms = sorted[(sorted.len() * 99) / 100];
                }
            }
        }
    }

    /// Record a failed request.
    fn record_failure(&self, deployment_name: &str, _error: &Error) {
        let mut health = self.state.health.write();
        if let Some(h) = health.get_mut(deployment_name) {
            h.total_requests += 1;
            h.total_errors += 1;
            h.consecutive_failures += 1;
            h.consecutive_successes = 0;

            // Mark unhealthy if threshold exceeded
            if h.healthy && h.consecutive_failures >= self.health_config.failure_threshold {
                h.healthy = false;
                tracing::warn!(
                    deployment = deployment_name,
                    failures = h.consecutive_failures,
                    "Deployment marked unhealthy"
                );
            }
        }
    }

    /// Increment in-flight counter.
    fn inc_in_flight(&self, deployment_name: &str) {
        let mut health = self.state.health.write();
        if let Some(h) = health.get_mut(deployment_name) {
            h.requests_in_flight += 1;
        }
    }

    /// Decrement in-flight counter.
    fn dec_in_flight(&self, deployment_name: &str) {
        let mut health = self.state.health.write();
        if let Some(h) = health.get_mut(deployment_name) {
            h.requests_in_flight = h.requests_in_flight.saturating_sub(1);
        }
    }
}

// Simple random number generator (avoid extra dependency)
fn rand_u32() -> u32 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let seed = COUNTER.fetch_add(1, Ordering::Relaxed);
    let time = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    ((seed.wrapping_mul(6364136223846793005).wrapping_add(time)) >> 32) as u32
}

#[async_trait]
impl Provider for ProviderPool {
    fn name(&self) -> &str {
        &self.name
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let deployment = self.select_deployment()?;
        let deployment_name = deployment.name.clone();

        // Acquire semaphore permit if max_concurrent is set
        let _permit = if let Some(sem) = self.state.semaphores.get(&deployment_name) {
            Some(
                sem.acquire()
                    .await
                    .map_err(|_| Error::other("Semaphore closed"))?,
            )
        } else {
            None
        };

        self.inc_in_flight(&deployment_name);
        let start = Instant::now();

        let result = deployment.provider.complete(request).await;

        let latency_ms = start.elapsed().as_millis() as u64;
        self.dec_in_flight(&deployment_name);

        match &result {
            Ok(_) => self.record_success(&deployment_name, latency_ms),
            Err(e) => self.record_failure(&deployment_name, e),
        }

        result
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        let deployment = self.select_deployment()?;
        let deployment_name = deployment.name.clone();

        self.inc_in_flight(&deployment_name);
        let start = Instant::now();

        let result = deployment.provider.complete_stream(request).await;

        let latency_ms = start.elapsed().as_millis() as u64;
        self.dec_in_flight(&deployment_name);

        match &result {
            Ok(_) => self.record_success(&deployment_name, latency_ms),
            Err(e) => self.record_failure(&deployment_name, e),
        }

        result
    }

    fn supports_tools(&self) -> bool {
        self.deployments
            .first()
            .map(|d| d.provider.supports_tools())
            .unwrap_or(false)
    }

    fn supports_vision(&self) -> bool {
        self.deployments
            .first()
            .map(|d| d.provider.supports_vision())
            .unwrap_or(false)
    }

    fn supports_streaming(&self) -> bool {
        self.deployments
            .first()
            .map(|d| d.provider.supports_streaming())
            .unwrap_or(true)
    }

    fn supports_token_counting(&self) -> bool {
        self.deployments
            .first()
            .map(|d| d.provider.supports_token_counting())
            .unwrap_or(false)
    }

    async fn count_tokens(&self, request: TokenCountRequest) -> Result<TokenCountResult> {
        let deployment = self.select_deployment()?;
        deployment.provider.count_tokens(request).await
    }

    fn supports_batch(&self) -> bool {
        self.deployments
            .first()
            .map(|d| d.provider.supports_batch())
            .unwrap_or(false)
    }

    async fn create_batch(&self, requests: Vec<BatchRequest>) -> Result<BatchJob> {
        let deployment = self.select_deployment()?;
        deployment.provider.create_batch(requests).await
    }

    async fn get_batch(&self, batch_id: &str) -> Result<BatchJob> {
        // Try all deployments since we don't know which one created the batch
        for deployment in &self.deployments {
            if let Ok(job) = deployment.provider.get_batch(batch_id).await {
                return Ok(job);
            }
        }
        Err(Error::other(format!("Batch not found: {}", batch_id)))
    }

    async fn get_batch_results(&self, batch_id: &str) -> Result<Vec<BatchResult>> {
        for deployment in &self.deployments {
            if let Ok(results) = deployment.provider.get_batch_results(batch_id).await {
                return Ok(results);
            }
        }
        Err(Error::other(format!("Batch not found: {}", batch_id)))
    }

    async fn cancel_batch(&self, batch_id: &str) -> Result<BatchJob> {
        for deployment in &self.deployments {
            if let Ok(job) = deployment.provider.cancel_batch(batch_id).await {
                return Ok(job);
            }
        }
        Err(Error::other(format!("Batch not found: {}", batch_id)))
    }

    async fn list_batches(&self, limit: Option<u32>) -> Result<Vec<BatchJob>> {
        // Aggregate batches from all deployments
        let mut all_batches = Vec::new();
        for deployment in &self.deployments {
            if let Ok(batches) = deployment.provider.list_batches(limit).await {
                all_batches.extend(batches);
            }
        }
        Ok(all_batches)
    }
}

/// Builder for creating a provider pool.
pub struct ProviderPoolBuilder {
    deployments: Vec<DeploymentConfig>,
    strategy: RoutingStrategy,
    health_config: HealthCheckConfig,
    name: String,
}

impl Default for ProviderPoolBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderPoolBuilder {
    /// Create a new pool builder.
    pub fn new() -> Self {
        Self {
            deployments: Vec::new(),
            strategy: RoutingStrategy::default(),
            health_config: HealthCheckConfig::default(),
            name: "pool".to_string(),
        }
    }

    /// Set the pool name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Add a deployment to the pool.
    pub fn add_deployment(
        mut self,
        name: impl Into<String>,
        provider: impl Provider + 'static,
        weight: u32,
        priority: u32,
    ) -> Self {
        self.deployments.push(DeploymentConfig {
            name: name.into(),
            provider: Arc::new(provider),
            weight,
            priority,
            max_concurrent: None,
            region: None,
        });
        self
    }

    /// Add a deployment with full configuration.
    pub fn add_deployment_config(mut self, config: DeploymentConfig) -> Self {
        self.deployments.push(config);
        self
    }

    /// Set the routing strategy.
    pub fn strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the health check configuration.
    pub fn health_check(mut self, config: HealthCheckConfig) -> Self {
        self.health_config = config;
        self
    }

    /// Disable health checking.
    pub fn disable_health_check(mut self) -> Self {
        self.health_config.enabled = false;
        self
    }

    /// Build the provider pool.
    pub fn build(self) -> Result<ProviderPool> {
        if self.deployments.is_empty() {
            return Err(Error::config("Pool must have at least one deployment"));
        }

        // Initialize health status and semaphores
        let mut health_map = HashMap::new();
        let mut semaphores = HashMap::new();

        for deployment in &self.deployments {
            health_map.insert(deployment.name.clone(), DeploymentHealth::default());

            if let Some(max_concurrent) = deployment.max_concurrent {
                semaphores.insert(
                    deployment.name.clone(),
                    Arc::new(Semaphore::new(max_concurrent as usize)),
                );
            }
        }

        Ok(ProviderPool {
            deployments: self.deployments,
            strategy: self.strategy,
            health_config: self.health_config,
            state: PoolState {
                rr_counter: AtomicU32::new(0),
                health: RwLock::new(health_map),
                semaphores,
                latency_samples: RwLock::new(HashMap::new()),
            },
            name: self.name,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mock provider for testing
    struct MockProvider {
        name: String,
    }

    #[async_trait]
    impl Provider for MockProvider {
        fn name(&self) -> &str {
            &self.name
        }

        async fn complete(&self, _request: CompletionRequest) -> Result<CompletionResponse> {
            Ok(CompletionResponse {
                id: "test".to_string(),
                model: "test".to_string(),
                content: vec![],
                stop_reason: crate::types::StopReason::EndTurn,
                usage: crate::types::Usage {
                    input_tokens: 0,
                    output_tokens: 0,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            })
        }

        async fn complete_stream(
            &self,
            _request: CompletionRequest,
        ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
            Err(Error::not_supported("streaming"))
        }
    }

    #[test]
    fn test_pool_builder() {
        let pool = ProviderPool::builder()
            .name("test-pool")
            .add_deployment(
                "d1",
                MockProvider {
                    name: "mock1".into(),
                },
                1,
                100,
            )
            .add_deployment(
                "d2",
                MockProvider {
                    name: "mock2".into(),
                },
                2,
                200,
            )
            .strategy(RoutingStrategy::Weighted)
            .build()
            .unwrap();

        assert_eq!(pool.deployments.len(), 2);
        assert_eq!(pool.healthy_count(), 2);
    }

    #[test]
    fn test_empty_pool_error() {
        let result = ProviderPool::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_health_tracking() {
        let pool = ProviderPool::builder()
            .add_deployment(
                "d1",
                MockProvider {
                    name: "mock".into(),
                },
                1,
                1,
            )
            .build()
            .unwrap();

        // Initially healthy
        let health = pool.health("d1").unwrap();
        assert!(health.healthy);
        assert_eq!(health.consecutive_failures, 0);

        // Record failures
        for _ in 0..3 {
            pool.record_failure("d1", &Error::Timeout);
        }

        let health = pool.health("d1").unwrap();
        assert!(!health.healthy);
        assert_eq!(health.consecutive_failures, 3);

        // Record successes for recovery
        for _ in 0..2 {
            pool.record_success("d1", 100);
        }

        let health = pool.health("d1").unwrap();
        assert!(health.healthy);
    }
}
