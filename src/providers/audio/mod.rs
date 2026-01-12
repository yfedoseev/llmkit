//! Audio processing providers (TTS and STT).
//!
//! This module contains providers for speech-to-text transcription
//! and text-to-speech synthesis.

#[cfg(feature = "deepgram")]
pub mod deepgram;

#[cfg(feature = "elevenlabs")]
pub mod elevenlabs;

#[cfg(feature = "assemblyai")]
pub mod assemblyai;

// Contingent providers (pending API access)
pub mod grok_realtime;

// Re-exports

#[cfg(feature = "deepgram")]
pub use deepgram::{DeepgramConfig, DeepgramProvider, DeepgramVersion};

#[cfg(feature = "elevenlabs")]
pub use elevenlabs::{ElevenLabsProvider, LatencyMode, StreamingOptions};

#[cfg(feature = "assemblyai")]
pub use assemblyai::{AssemblyAIProvider, AudioLanguage, TranscriptionConfig};

// Contingent provider re-exports (pending API access)
pub use grok_realtime::GrokRealtimeProvider;
