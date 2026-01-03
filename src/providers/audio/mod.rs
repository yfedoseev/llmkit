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

// Re-exports

#[cfg(feature = "deepgram")]
pub use deepgram::DeepgramProvider;

#[cfg(feature = "elevenlabs")]
pub use elevenlabs::ElevenLabsProvider;

#[cfg(feature = "assemblyai")]
pub use assemblyai::{AssemblyAIProvider, AudioLanguage, TranscriptionConfig};
