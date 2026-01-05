//! Response caching infrastructure for LLM API calls.
//!
//! This module provides a flexible caching system to reduce API costs
//! and improve response times by caching identical requests.
//!
//! # Example
//!
//! ```ignore
//! use modelsuite::{CacheConfig, CachingProvider, InMemoryCache, OpenAIProvider};
//!
//! // Create a caching provider
//! let inner = OpenAIProvider::from_env()?;
//! let cache = InMemoryCache::new(CacheConfig::default());
//! let provider = CachingProvider::new(inner, cache);
//!
//! // First request hits the API
//! let response1 = provider.complete(request.clone()).await?;
//!
//! // Second identical request hits the cache
//! let response2 = provider.complete(request).await?;
//! ```
//!
//! # Cache Key Computation
//!
//! Cache keys are computed from:
//! - Model name
//! - Messages content
//! - Tools (if any)
//! - System prompt
//!
//! By default, non-deterministic parameters (temperature, top_p) are excluded
//! from the cache key to allow caching regardless of sampling settings.

use std::collections::HashSet;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use async_trait::async_trait;
use dashmap::DashMap;
use futures::Stream;
use sha2::{Digest, Sha256};

use crate::error::Result;
use crate::provider::Provider;
use crate::types::{CompletionRequest, CompletionResponse, StreamChunk};

/// Configuration for the caching system.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Whether caching is enabled.
    pub enabled: bool,
    /// Time-to-live for cached entries.
    pub ttl: Duration,
    /// Maximum number of entries in the cache.
    pub max_entries: usize,
    /// Whether to cache streaming responses (after collection).
    pub cache_streaming: bool,
    /// Fields to exclude from cache key (for deterministic caching).
    pub exclude_fields: HashSet<String>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            ttl: Duration::from_secs(3600), // 1 hour
            max_entries: 10_000,
            cache_streaming: false,
            exclude_fields: HashSet::from_iter([
                "temperature".to_string(),
                "top_p".to_string(),
                "top_k".to_string(),
                "seed".to_string(),
            ]),
        }
    }
}

impl CacheConfig {
    /// Create a new cache configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the TTL for cached entries.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Set the maximum number of entries.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Enable or disable caching.
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Enable or disable streaming cache.
    pub fn with_cache_streaming(mut self, cache_streaming: bool) -> Self {
        self.cache_streaming = cache_streaming;
        self
    }
}

/// A cached response with metadata.
#[derive(Debug, Clone)]
pub struct CachedResponse {
    /// The cached response.
    pub response: CompletionResponse,
    /// When this entry was created.
    pub created_at: SystemTime,
    /// Number of times this entry was hit.
    pub hit_count: Arc<AtomicU64>,
}

impl CachedResponse {
    /// Create a new cached response.
    pub fn new(response: CompletionResponse) -> Self {
        Self {
            response,
            created_at: SystemTime::now(),
            hit_count: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Check if this entry has expired.
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at
            .elapsed()
            .map(|elapsed| elapsed > ttl)
            .unwrap_or(true)
    }

    /// Increment the hit count and return the new value.
    pub fn record_hit(&self) -> u64 {
        self.hit_count.fetch_add(1, Ordering::Relaxed) + 1
    }
}

/// Statistics about cache performance.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: u64,
    /// Number of cache misses.
    pub misses: u64,
    /// Current number of entries.
    pub entries: usize,
    /// Approximate size in bytes.
    pub size_bytes: usize,
}

impl CacheStats {
    /// Get the hit rate (0.0 to 1.0).
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// Trait for cache backends.
#[async_trait]
pub trait CacheBackend: Send + Sync {
    /// Get a cached response by key.
    async fn get(&self, key: &str) -> Option<CachedResponse>;

    /// Store a response in the cache.
    async fn set(&self, key: &str, response: CachedResponse);

    /// Invalidate a specific key.
    async fn invalidate(&self, key: &str);

    /// Clear all cache entries.
    async fn clear(&self);

    /// Get cache statistics.
    fn stats(&self) -> CacheStats;
}

/// In-memory cache backend using DashMap.
pub struct InMemoryCache {
    entries: DashMap<String, CachedResponse>,
    config: CacheConfig,
    hits: AtomicU64,
    misses: AtomicU64,
}

impl InMemoryCache {
    /// Create a new in-memory cache with the given configuration.
    pub fn new(config: CacheConfig) -> Arc<Self> {
        Arc::new(Self {
            entries: DashMap::new(),
            config,
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
        })
    }

    /// Create a new in-memory cache with default configuration.
    pub fn default_cache() -> Arc<Self> {
        Self::new(CacheConfig::default())
    }

    /// Evict expired entries.
    pub fn evict_expired(&self) {
        let ttl = self.config.ttl;
        self.entries.retain(|_, v| !v.is_expired(ttl));
    }

    /// Evict entries to meet the max size.
    fn evict_if_needed(&self) {
        if self.entries.len() >= self.config.max_entries {
            // Simple LRU-like eviction: remove oldest entries
            // In a production system, you'd want a proper LRU implementation
            let mut oldest_keys: Vec<(String, SystemTime)> = self
                .entries
                .iter()
                .map(|e| (e.key().clone(), e.value().created_at))
                .collect();

            oldest_keys.sort_by(|a, b| a.1.cmp(&b.1));

            // Remove 10% of oldest entries
            let to_remove = self.config.max_entries / 10;
            for (key, _) in oldest_keys.into_iter().take(to_remove) {
                self.entries.remove(&key);
            }
        }
    }
}

#[async_trait]
impl CacheBackend for InMemoryCache {
    async fn get(&self, key: &str) -> Option<CachedResponse> {
        if let Some(entry) = self.entries.get(key) {
            if entry.is_expired(self.config.ttl) {
                self.entries.remove(key);
                self.misses.fetch_add(1, Ordering::Relaxed);
                None
            } else {
                entry.record_hit();
                self.hits.fetch_add(1, Ordering::Relaxed);
                Some(entry.clone())
            }
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    async fn set(&self, key: &str, response: CachedResponse) {
        self.evict_if_needed();
        self.entries.insert(key.to_string(), response);
    }

    async fn invalidate(&self, key: &str) {
        self.entries.remove(key);
    }

    async fn clear(&self) {
        self.entries.clear();
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            entries: self.entries.len(),
            size_bytes: 0, // Would need serialization to compute accurately
        }
    }
}

/// A provider wrapper that caches responses.
pub struct CachingProvider<P> {
    /// The inner provider.
    inner: P,
    /// The cache backend.
    cache: Arc<dyn CacheBackend>,
    /// Cache configuration.
    config: CacheConfig,
}

impl<P> CachingProvider<P> {
    /// Create a new caching provider.
    pub fn new(inner: P, cache: Arc<dyn CacheBackend>) -> Self {
        Self {
            inner,
            cache,
            config: CacheConfig::default(),
        }
    }

    /// Create a new caching provider with custom configuration.
    pub fn with_config(inner: P, cache: Arc<dyn CacheBackend>, config: CacheConfig) -> Self {
        Self {
            inner,
            cache,
            config,
        }
    }

    /// Get the inner provider.
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Clear the cache.
    pub async fn clear_cache(&self) {
        self.cache.clear().await;
    }

    /// Compute a cache key for a request.
    fn compute_cache_key(&self, request: &CompletionRequest) -> String {
        // Create a normalized representation for hashing
        let mut hasher = Sha256::new();

        // Include model
        hasher.update(request.model.as_bytes());
        hasher.update(b"|");

        // Include system prompt
        if let Some(ref system) = request.system {
            hasher.update(system.as_bytes());
        }
        hasher.update(b"|");

        // Include messages
        for msg in &request.messages {
            hasher.update(format!("{:?}", msg.role).as_bytes());
            hasher.update(b":");
            for block in &msg.content {
                hasher.update(format!("{:?}", block).as_bytes());
            }
            hasher.update(b";");
        }
        hasher.update(b"|");

        // Include tools
        if let Some(ref tools) = request.tools {
            for tool in tools {
                hasher.update(tool.name.as_bytes());
                hasher.update(b":");
                hasher.update(tool.description.as_bytes());
                hasher.update(b";");
            }
        }
        hasher.update(b"|");

        // Include response format
        if let Some(ref format) = request.response_format {
            hasher.update(format!("{:?}", format.format_type).as_bytes());
        }

        format!("cache:{}", hex::encode(hasher.finalize()))
    }
}

#[async_trait]
impl<P: Provider> Provider for CachingProvider<P> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        if !self.config.enabled {
            return self.inner.complete(request).await;
        }

        let cache_key = self.compute_cache_key(&request);

        // Check cache
        if let Some(cached) = self.cache.get(&cache_key).await {
            tracing::debug!(key = %cache_key, "Cache hit");
            return Ok(cached.response);
        }

        // Call inner provider
        tracing::debug!(key = %cache_key, "Cache miss");
        let response = self.inner.complete(request).await?;

        // Store in cache
        let cached = CachedResponse::new(response.clone());
        self.cache.set(&cache_key, cached).await;

        Ok(response)
    }

    async fn complete_stream(
        &self,
        request: CompletionRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk>> + Send>>> {
        // For streaming, we don't cache by default
        // (would need to collect the stream and replay it)
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

    fn supported_models(&self) -> Option<&[&str]> {
        self.inner.supported_models()
    }

    fn default_model(&self) -> Option<&str> {
        self.inner.default_model()
    }
}

/// Cache key builder for custom cache key computation.
#[derive(Default)]
pub struct CacheKeyBuilder {
    parts: Vec<String>,
}

impl CacheKeyBuilder {
    /// Create a new cache key builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a part to the cache key.
    pub fn with_part(mut self, part: impl Into<String>) -> Self {
        self.parts.push(part.into());
        self
    }

    /// Build the cache key.
    pub fn build(self) -> String {
        let mut hasher = Sha256::new();
        for part in self.parts {
            hasher.update(part.as_bytes());
            hasher.update(b"|");
        }
        format!("cache:{}", hex::encode(hasher.finalize()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_config_default() {
        let config = CacheConfig::default();
        assert!(config.enabled);
        assert_eq!(config.ttl, Duration::from_secs(3600));
        assert_eq!(config.max_entries, 10_000);
        assert!(!config.cache_streaming);
    }

    #[test]
    fn test_cache_config_builder() {
        let config = CacheConfig::new()
            .with_ttl(Duration::from_secs(600))
            .with_max_entries(1000)
            .with_enabled(false);

        assert!(!config.enabled);
        assert_eq!(config.ttl, Duration::from_secs(600));
        assert_eq!(config.max_entries, 1000);
    }

    #[test]
    fn test_cached_response_expiry() {
        let response = CompletionResponse {
            id: "test".to_string(),
            model: "test".to_string(),
            content: vec![],
            stop_reason: crate::types::StopReason::EndTurn,
            usage: crate::types::Usage::default(),
        };

        let cached = CachedResponse::new(response);

        // Should not be expired with long TTL
        assert!(!cached.is_expired(Duration::from_secs(3600)));

        // Should be expired with zero TTL
        assert!(cached.is_expired(Duration::from_secs(0)));
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            hits: 80,
            misses: 20,
            entries: 100,
            size_bytes: 0,
        };

        assert!((stats.hit_rate() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_cache_stats_hit_rate_zero() {
        let stats = CacheStats::default();
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[tokio::test]
    async fn test_in_memory_cache() {
        let cache = InMemoryCache::new(CacheConfig::default());

        let response = CompletionResponse {
            id: "test".to_string(),
            model: "test".to_string(),
            content: vec![],
            stop_reason: crate::types::StopReason::EndTurn,
            usage: crate::types::Usage::default(),
        };

        // Initially empty
        assert!(cache.get("key1").await.is_none());

        // After set
        cache.set("key1", CachedResponse::new(response)).await;
        assert!(cache.get("key1").await.is_some());

        // Stats
        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.entries, 1);

        // Invalidate
        cache.invalidate("key1").await;
        assert!(cache.get("key1").await.is_none());

        // Clear
        cache
            .set(
                "key2",
                CachedResponse::new(CompletionResponse {
                    id: "test2".to_string(),
                    model: "test".to_string(),
                    content: vec![],
                    stop_reason: crate::types::StopReason::EndTurn,
                    usage: crate::types::Usage::default(),
                }),
            )
            .await;
        cache.clear().await;
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn test_cache_key_builder() {
        let key = CacheKeyBuilder::new()
            .with_part("model")
            .with_part("prompt")
            .build();

        assert!(key.starts_with("cache:"));
        assert_eq!(key.len(), 6 + 64); // "cache:" + 64 hex chars
    }
}
