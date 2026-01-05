//! JavaScript enum types for ModelSuite
//!
//! These enums are exposed as TypeScript enums with numeric values.

use modelsuite::types::{
    BatchStatus, CacheControl, Role, StopReason, StreamEventType, ThinkingType,
};
use napi_derive::napi;

/// Message role in a conversation.
#[napi]
pub enum JsRole {
    /// System message providing context or instructions
    System,
    /// User message
    User,
    /// Assistant (LLM) message
    Assistant,
}

impl From<Role> for JsRole {
    fn from(role: Role) -> Self {
        match role {
            Role::System => JsRole::System,
            Role::User => JsRole::User,
            Role::Assistant => JsRole::Assistant,
        }
    }
}

impl From<JsRole> for Role {
    fn from(role: JsRole) -> Self {
        match role {
            JsRole::System => Role::System,
            JsRole::User => Role::User,
            JsRole::Assistant => Role::Assistant,
        }
    }
}

/// Reason the model stopped generating.
#[napi]
pub enum JsStopReason {
    /// Natural end of response
    EndTurn,
    /// Hit max tokens limit
    MaxTokens,
    /// Model wants to use a tool
    ToolUse,
    /// Hit a stop sequence
    StopSequence,
    /// Response was filtered by content moderation
    ContentFilter,
}

impl From<StopReason> for JsStopReason {
    fn from(reason: StopReason) -> Self {
        match reason {
            StopReason::EndTurn => JsStopReason::EndTurn,
            StopReason::MaxTokens => JsStopReason::MaxTokens,
            StopReason::ToolUse => JsStopReason::ToolUse,
            StopReason::StopSequence => JsStopReason::StopSequence,
            StopReason::ContentFilter => JsStopReason::ContentFilter,
        }
    }
}

impl From<JsStopReason> for StopReason {
    fn from(reason: JsStopReason) -> Self {
        match reason {
            JsStopReason::EndTurn => StopReason::EndTurn,
            JsStopReason::MaxTokens => StopReason::MaxTokens,
            JsStopReason::ToolUse => StopReason::ToolUse,
            JsStopReason::StopSequence => StopReason::StopSequence,
            JsStopReason::ContentFilter => StopReason::ContentFilter,
        }
    }
}

/// Streaming event type.
#[napi]
pub enum JsStreamEventType {
    /// Message started
    MessageStart,
    /// Content block started
    ContentBlockStart,
    /// Content block delta (partial content)
    ContentBlockDelta,
    /// Content block stopped
    ContentBlockStop,
    /// Message delta
    MessageDelta,
    /// Message stopped
    MessageStop,
    /// Ping event
    Ping,
    /// Error event
    Error,
}

impl From<StreamEventType> for JsStreamEventType {
    fn from(event_type: StreamEventType) -> Self {
        match event_type {
            StreamEventType::MessageStart => JsStreamEventType::MessageStart,
            StreamEventType::ContentBlockStart => JsStreamEventType::ContentBlockStart,
            StreamEventType::ContentBlockDelta => JsStreamEventType::ContentBlockDelta,
            StreamEventType::ContentBlockStop => JsStreamEventType::ContentBlockStop,
            StreamEventType::MessageDelta => JsStreamEventType::MessageDelta,
            StreamEventType::MessageStop => JsStreamEventType::MessageStop,
            StreamEventType::Ping => JsStreamEventType::Ping,
            StreamEventType::Error => JsStreamEventType::Error,
        }
    }
}

/// Cache control type for prompt caching.
#[napi]
pub enum JsCacheControl {
    /// 5-minute TTL cache
    Ephemeral,
    /// 1-hour TTL cache (Anthropic beta)
    Extended,
}

impl From<CacheControl> for JsCacheControl {
    fn from(cache_control: CacheControl) -> Self {
        match cache_control {
            CacheControl::Ephemeral => JsCacheControl::Ephemeral,
            CacheControl::Extended => JsCacheControl::Extended,
        }
    }
}

impl From<JsCacheControl> for CacheControl {
    fn from(cache_control: JsCacheControl) -> Self {
        match cache_control {
            JsCacheControl::Ephemeral => CacheControl::Ephemeral,
            JsCacheControl::Extended => CacheControl::Extended,
        }
    }
}

/// Thinking mode type.
#[napi]
pub enum JsThinkingType {
    /// Extended thinking is enabled
    Enabled,
    /// Extended thinking is disabled
    Disabled,
}

impl From<ThinkingType> for JsThinkingType {
    fn from(thinking_type: ThinkingType) -> Self {
        match thinking_type {
            ThinkingType::Enabled => JsThinkingType::Enabled,
            ThinkingType::Disabled => JsThinkingType::Disabled,
        }
    }
}

impl From<JsThinkingType> for ThinkingType {
    fn from(thinking_type: JsThinkingType) -> Self {
        match thinking_type {
            JsThinkingType::Enabled => ThinkingType::Enabled,
            JsThinkingType::Disabled => ThinkingType::Disabled,
        }
    }
}

/// Batch job status.
#[napi]
pub enum JsBatchStatus {
    /// Batch is being validated
    Validating,
    /// Batch is in progress
    InProgress,
    /// Batch is finalizing
    Finalizing,
    /// Batch completed successfully
    Completed,
    /// Batch failed
    Failed,
    /// Batch expired
    Expired,
    /// Batch was cancelled
    Cancelled,
}

impl From<BatchStatus> for JsBatchStatus {
    fn from(status: BatchStatus) -> Self {
        match status {
            BatchStatus::Validating => JsBatchStatus::Validating,
            BatchStatus::InProgress => JsBatchStatus::InProgress,
            BatchStatus::Finalizing => JsBatchStatus::Finalizing,
            BatchStatus::Completed => JsBatchStatus::Completed,
            BatchStatus::Failed => JsBatchStatus::Failed,
            BatchStatus::Expired => JsBatchStatus::Expired,
            BatchStatus::Cancelled => JsBatchStatus::Cancelled,
        }
    }
}

impl From<JsBatchStatus> for BatchStatus {
    fn from(status: JsBatchStatus) -> Self {
        match status {
            JsBatchStatus::Validating => BatchStatus::Validating,
            JsBatchStatus::InProgress => BatchStatus::InProgress,
            JsBatchStatus::Finalizing => BatchStatus::Finalizing,
            JsBatchStatus::Completed => BatchStatus::Completed,
            JsBatchStatus::Failed => BatchStatus::Failed,
            JsBatchStatus::Expired => BatchStatus::Expired,
            JsBatchStatus::Cancelled => BatchStatus::Cancelled,
        }
    }
}
