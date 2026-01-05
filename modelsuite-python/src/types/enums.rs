//! Python enum types for ModelSuite
//!
//! These are simple enums exposed as Python IntEnum-like classes.

use modelsuite::types::{
    BatchStatus, CacheControl, Role, StopReason, StreamEventType, ThinkingType,
};
use pyo3::prelude::*;

/// Message role in a conversation.
///
/// - `System`: System message providing context or instructions
/// - `User`: User message
/// - `Assistant`: Assistant (LLM) message
#[pyclass(name = "Role", eq, eq_int, hash, frozen)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyRole {
    /// System message providing context or instructions
    System = 0,
    /// User message
    User = 1,
    /// Assistant (LLM) message
    Assistant = 2,
}

#[pymethods]
impl PyRole {
    /// Convert to string representation.
    fn __str__(&self) -> &'static str {
        match self {
            PyRole::System => "system",
            PyRole::User => "user",
            PyRole::Assistant => "assistant",
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Role.{}",
            match self {
                PyRole::System => "System",
                PyRole::User => "User",
                PyRole::Assistant => "Assistant",
            }
        )
    }
}

impl From<Role> for PyRole {
    fn from(role: Role) -> Self {
        match role {
            Role::System => PyRole::System,
            Role::User => PyRole::User,
            Role::Assistant => PyRole::Assistant,
        }
    }
}

impl From<PyRole> for Role {
    fn from(role: PyRole) -> Self {
        match role {
            PyRole::System => Role::System,
            PyRole::User => Role::User,
            PyRole::Assistant => Role::Assistant,
        }
    }
}

/// Reason the model stopped generating.
///
/// - `EndTurn`: Natural end of response
/// - `MaxTokens`: Hit max tokens limit
/// - `ToolUse`: Model wants to use a tool
/// - `StopSequence`: Hit a stop sequence
/// - `ContentFilter`: Response was filtered by content moderation
#[pyclass(name = "StopReason", eq, eq_int, hash, frozen)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyStopReason {
    /// Natural end of response
    EndTurn = 0,
    /// Hit max tokens limit
    MaxTokens = 1,
    /// Model wants to use a tool
    ToolUse = 2,
    /// Hit a stop sequence
    StopSequence = 3,
    /// Response was filtered by content moderation
    ContentFilter = 4,
}

#[pymethods]
impl PyStopReason {
    fn __str__(&self) -> &'static str {
        match self {
            PyStopReason::EndTurn => "end_turn",
            PyStopReason::MaxTokens => "max_tokens",
            PyStopReason::ToolUse => "tool_use",
            PyStopReason::StopSequence => "stop_sequence",
            PyStopReason::ContentFilter => "content_filter",
        }
    }

    fn __repr__(&self) -> String {
        format!("StopReason.{:?}", self)
    }

    /// Check if this stop reason indicates tool use.
    #[getter]
    fn is_tool_use(&self) -> bool {
        matches!(self, PyStopReason::ToolUse)
    }

    /// Check if this stop reason indicates natural completion.
    #[getter]
    fn is_complete(&self) -> bool {
        matches!(self, PyStopReason::EndTurn)
    }
}

impl From<StopReason> for PyStopReason {
    fn from(reason: StopReason) -> Self {
        match reason {
            StopReason::EndTurn => PyStopReason::EndTurn,
            StopReason::MaxTokens => PyStopReason::MaxTokens,
            StopReason::ToolUse => PyStopReason::ToolUse,
            StopReason::StopSequence => PyStopReason::StopSequence,
            StopReason::ContentFilter => PyStopReason::ContentFilter,
        }
    }
}

impl From<PyStopReason> for StopReason {
    fn from(reason: PyStopReason) -> Self {
        match reason {
            PyStopReason::EndTurn => StopReason::EndTurn,
            PyStopReason::MaxTokens => StopReason::MaxTokens,
            PyStopReason::ToolUse => StopReason::ToolUse,
            PyStopReason::StopSequence => StopReason::StopSequence,
            PyStopReason::ContentFilter => StopReason::ContentFilter,
        }
    }
}

/// Streaming event type.
#[pyclass(name = "StreamEventType", eq, eq_int, hash, frozen)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyStreamEventType {
    /// Message started
    MessageStart = 0,
    /// Content block started
    ContentBlockStart = 1,
    /// Content block delta (partial content)
    ContentBlockDelta = 2,
    /// Content block stopped
    ContentBlockStop = 3,
    /// Message delta
    MessageDelta = 4,
    /// Message stopped
    MessageStop = 5,
    /// Ping event
    Ping = 6,
    /// Error event
    Error = 7,
}

#[pymethods]
impl PyStreamEventType {
    fn __str__(&self) -> &'static str {
        match self {
            PyStreamEventType::MessageStart => "message_start",
            PyStreamEventType::ContentBlockStart => "content_block_start",
            PyStreamEventType::ContentBlockDelta => "content_block_delta",
            PyStreamEventType::ContentBlockStop => "content_block_stop",
            PyStreamEventType::MessageDelta => "message_delta",
            PyStreamEventType::MessageStop => "message_stop",
            PyStreamEventType::Ping => "ping",
            PyStreamEventType::Error => "error",
        }
    }

    fn __repr__(&self) -> String {
        format!("StreamEventType.{:?}", self)
    }

    /// Check if this is a terminal event.
    #[getter]
    fn is_done(&self) -> bool {
        matches!(self, PyStreamEventType::MessageStop)
    }
}

impl From<StreamEventType> for PyStreamEventType {
    fn from(event_type: StreamEventType) -> Self {
        match event_type {
            StreamEventType::MessageStart => PyStreamEventType::MessageStart,
            StreamEventType::ContentBlockStart => PyStreamEventType::ContentBlockStart,
            StreamEventType::ContentBlockDelta => PyStreamEventType::ContentBlockDelta,
            StreamEventType::ContentBlockStop => PyStreamEventType::ContentBlockStop,
            StreamEventType::MessageDelta => PyStreamEventType::MessageDelta,
            StreamEventType::MessageStop => PyStreamEventType::MessageStop,
            StreamEventType::Ping => PyStreamEventType::Ping,
            StreamEventType::Error => PyStreamEventType::Error,
        }
    }
}

/// Cache control type for prompt caching.
///
/// - `Ephemeral`: 5-minute TTL cache
/// - `Extended`: 1-hour TTL cache (Anthropic beta)
#[pyclass(name = "CacheControl", eq, eq_int, hash, frozen)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyCacheControl {
    /// 5-minute TTL cache
    Ephemeral = 0,
    /// 1-hour TTL cache (Anthropic beta)
    Extended = 1,
}

#[pymethods]
impl PyCacheControl {
    fn __str__(&self) -> &'static str {
        match self {
            PyCacheControl::Ephemeral => "ephemeral",
            PyCacheControl::Extended => "extended",
        }
    }

    fn __repr__(&self) -> String {
        format!("CacheControl.{:?}", self)
    }
}

impl From<CacheControl> for PyCacheControl {
    fn from(cache_control: CacheControl) -> Self {
        match cache_control {
            CacheControl::Ephemeral => PyCacheControl::Ephemeral,
            CacheControl::Extended => PyCacheControl::Extended,
        }
    }
}

impl From<PyCacheControl> for CacheControl {
    fn from(cache_control: PyCacheControl) -> Self {
        match cache_control {
            PyCacheControl::Ephemeral => CacheControl::Ephemeral,
            PyCacheControl::Extended => CacheControl::Extended,
        }
    }
}

/// Thinking mode type.
///
/// - `Enabled`: Extended thinking is enabled with a token budget
/// - `Disabled`: Extended thinking is disabled
#[pyclass(name = "ThinkingType", eq, eq_int, hash, frozen)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyThinkingType {
    /// Extended thinking is enabled
    Enabled = 0,
    /// Extended thinking is disabled
    Disabled = 1,
}

#[pymethods]
impl PyThinkingType {
    fn __str__(&self) -> &'static str {
        match self {
            PyThinkingType::Enabled => "enabled",
            PyThinkingType::Disabled => "disabled",
        }
    }

    fn __repr__(&self) -> String {
        format!("ThinkingType.{:?}", self)
    }
}

impl From<ThinkingType> for PyThinkingType {
    fn from(thinking_type: ThinkingType) -> Self {
        match thinking_type {
            ThinkingType::Enabled => PyThinkingType::Enabled,
            ThinkingType::Disabled => PyThinkingType::Disabled,
        }
    }
}

impl From<PyThinkingType> for ThinkingType {
    fn from(thinking_type: PyThinkingType) -> Self {
        match thinking_type {
            PyThinkingType::Enabled => ThinkingType::Enabled,
            PyThinkingType::Disabled => ThinkingType::Disabled,
        }
    }
}

/// Batch job status.
#[pyclass(name = "BatchStatus", eq, eq_int, hash, frozen)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PyBatchStatus {
    /// Batch is being validated
    Validating = 0,
    /// Batch is in progress
    InProgress = 1,
    /// Batch is finalizing
    Finalizing = 2,
    /// Batch completed successfully
    Completed = 3,
    /// Batch failed
    Failed = 4,
    /// Batch expired
    Expired = 5,
    /// Batch was cancelled
    Cancelled = 6,
}

#[pymethods]
impl PyBatchStatus {
    fn __str__(&self) -> &'static str {
        match self {
            PyBatchStatus::Validating => "validating",
            PyBatchStatus::InProgress => "in_progress",
            PyBatchStatus::Finalizing => "finalizing",
            PyBatchStatus::Completed => "completed",
            PyBatchStatus::Failed => "failed",
            PyBatchStatus::Expired => "expired",
            PyBatchStatus::Cancelled => "cancelled",
        }
    }

    fn __repr__(&self) -> String {
        format!("BatchStatus.{:?}", self)
    }

    /// Check if the batch is still processing.
    #[getter]
    fn is_processing(&self) -> bool {
        matches!(
            self,
            PyBatchStatus::Validating | PyBatchStatus::InProgress | PyBatchStatus::Finalizing
        )
    }

    /// Check if the batch is done (completed, failed, expired, or cancelled).
    #[getter]
    fn is_done(&self) -> bool {
        matches!(
            self,
            PyBatchStatus::Completed
                | PyBatchStatus::Failed
                | PyBatchStatus::Expired
                | PyBatchStatus::Cancelled
        )
    }

    /// Check if the batch completed successfully.
    #[getter]
    fn is_success(&self) -> bool {
        matches!(self, PyBatchStatus::Completed)
    }
}

impl From<BatchStatus> for PyBatchStatus {
    fn from(status: BatchStatus) -> Self {
        match status {
            BatchStatus::Validating => PyBatchStatus::Validating,
            BatchStatus::InProgress => PyBatchStatus::InProgress,
            BatchStatus::Finalizing => PyBatchStatus::Finalizing,
            BatchStatus::Completed => PyBatchStatus::Completed,
            BatchStatus::Failed => PyBatchStatus::Failed,
            BatchStatus::Expired => PyBatchStatus::Expired,
            BatchStatus::Cancelled => PyBatchStatus::Cancelled,
        }
    }
}

impl From<PyBatchStatus> for BatchStatus {
    fn from(status: PyBatchStatus) -> Self {
        match status {
            PyBatchStatus::Validating => BatchStatus::Validating,
            PyBatchStatus::InProgress => BatchStatus::InProgress,
            PyBatchStatus::Finalizing => BatchStatus::Finalizing,
            PyBatchStatus::Completed => BatchStatus::Completed,
            PyBatchStatus::Failed => BatchStatus::Failed,
            PyBatchStatus::Expired => BatchStatus::Expired,
            PyBatchStatus::Cancelled => BatchStatus::Cancelled,
        }
    }
}
