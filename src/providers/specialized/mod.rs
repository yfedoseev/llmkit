//! Specialized providers with unique APIs.
//!
//! This module contains providers that don't fit standard modalities,
//! including realtime APIs, serverless platforms, and reasoning-focused models.

#[cfg(feature = "openai-realtime")]
pub mod openai_realtime;

// Re-exports

#[cfg(feature = "openai-realtime")]
pub use openai_realtime::{RealtimeProvider, RealtimeSession, ServerEvent, SessionConfig};
