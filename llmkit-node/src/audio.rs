//! Audio API bindings for Node.js/TypeScript.
//!
//! Provides access to LLMKit audio functionality for speech-to-text transcription
//! and text-to-speech synthesis from various providers (Deepgram, AssemblyAI, ElevenLabs).

use napi_derive::napi;

// ============================================================================
// DEEPGRAM AUDIO (Speech-to-Text)
// ============================================================================

/// Deepgram API version for selecting model features and endpoints.
#[napi]
pub enum JsDeepgramVersion {
    /// API v1 (2023-12-01) - legacy support
    V1,
    /// API v3 (2025-01-01) - latest with Nova-3 models
    V3,
}

/// Options for Deepgram transcription.
#[napi]
pub struct JsTranscribeOptions {
    pub model: Option<String>,
    pub smart_format: bool,
    pub diarize: bool,
    pub language: Option<String>,
    pub punctuate: bool,
}

#[napi]
impl JsTranscribeOptions {
    /// Create a new TranscribeOptions with defaults.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            model: None,
            smart_format: false,
            diarize: false,
            language: None,
            punctuate: false,
        }
    }

    /// Set the Deepgram model to use.
    #[napi]
    pub fn with_model(&self, model: String) -> Self {
        Self {
            model: Some(model),
            ..(*self).clone()
        }
    }

    /// Enable smart formatting for better punctuation and capitalization.
    #[napi]
    pub fn with_smart_format(&self, enabled: bool) -> Self {
        Self {
            smart_format: enabled,
            ..(*self).clone()
        }
    }

    /// Enable speaker diarization to identify different speakers.
    #[napi]
    pub fn with_diarize(&self, enabled: bool) -> Self {
        Self {
            diarize: enabled,
            ..(*self).clone()
        }
    }

    /// Set the language of the audio.
    #[napi]
    pub fn with_language(&self, language: String) -> Self {
        Self {
            language: Some(language),
            ..(*self).clone()
        }
    }

    /// Enable automatic punctuation addition.
    #[napi]
    pub fn with_punctuate(&self, enabled: bool) -> Self {
        Self {
            punctuate: enabled,
            ..(*self).clone()
        }
    }
}

impl Default for JsTranscribeOptions {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for JsTranscribeOptions {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            smart_format: self.smart_format,
            diarize: self.diarize,
            language: self.language.clone(),
            punctuate: self.punctuate,
        }
    }
}

/// A single word from transcription with timing and confidence.
#[napi(object)]
#[derive(Clone)]
pub struct JsWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
    pub speaker: Option<u32>,
}

/// Response from a transcription request.
#[napi(object)]
#[derive(Clone)]
pub struct JsTranscribeResponse {
    pub transcript: String,
    pub confidence: Option<f64>,
    pub words: Vec<JsWord>,
    pub duration: Option<f64>,
    pub metadata: Option<String>,
}

// ============================================================================
// ELEVENLABS AUDIO (Text-to-Speech)
// ============================================================================

/// Latency mode for ElevenLabs synthesis.
#[napi]
#[derive(Clone, Copy)]
pub enum JsLatencyMode {
    /// Lowest possible latency (fastest)
    LowestLatency,
    /// Low latency
    LowLatency,
    /// Balanced (default)
    Balanced,
    /// High quality
    HighQuality,
    /// Highest quality (slowest)
    HighestQuality,
}

/// Voice settings for ElevenLabs synthesis.
#[napi(object)]
#[derive(Clone)]
pub struct JsVoiceSettings {
    pub stability: f64,
    pub similarity_boost: f64,
    pub style: Option<f64>,
    pub use_speaker_boost: bool,
}

/// Options for ElevenLabs text-to-speech synthesis.
#[napi(object)]
pub struct JsSynthesizeOptions {
    pub model_id: Option<String>,
    pub voice_settings: Option<JsVoiceSettings>,
    pub latency_mode: JsLatencyMode,
    pub output_format: Option<String>,
}

impl Clone for JsSynthesizeOptions {
    fn clone(&self) -> Self {
        Self {
            model_id: self.model_id.clone(),
            voice_settings: self.voice_settings.clone(),
            latency_mode: self.latency_mode,
            output_format: self.output_format.clone(),
        }
    }
}

/// Information about an available voice.
#[napi(object)]
#[derive(Clone)]
pub struct JsVoice {
    pub voice_id: String,
    pub name: String,
    pub category: Option<String>,
    pub description: Option<String>,
}

/// Response from text-to-speech synthesis.
#[napi(object)]
#[derive(Clone)]
pub struct JsSynthesizeResponse {
    pub audio_bytes: Vec<u8>,
    pub format: String,
    pub duration: Option<f64>,
}

// ============================================================================
// ASSEMBLYAI AUDIO (Speech-to-Text)
// ============================================================================

/// Language for AssemblyAI transcription.
#[napi]
#[derive(Clone, Copy)]
pub enum JsAudioLanguage {
    English,
    Spanish,
    French,
    German,
    ChineseSimplified,
    ChineseTraditional,
    Japanese,
}

/// Configuration for AssemblyAI transcription.
#[napi]
pub struct JsTranscriptionConfig {
    pub language: Option<JsAudioLanguage>,
    pub enable_diarization: bool,
    pub enable_entity_detection: bool,
    pub enable_sentiment_analysis: bool,
}

#[napi]
impl JsTranscriptionConfig {
    /// Create new transcription configuration with defaults.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            language: None,
            enable_diarization: false,
            enable_entity_detection: false,
            enable_sentiment_analysis: false,
        }
    }

    /// Set the language for transcription.
    #[napi]
    pub fn with_language(&self, language: JsAudioLanguage) -> Self {
        Self {
            language: Some(language),
            ..(*self).clone()
        }
    }

    /// Enable speaker diarization.
    #[napi]
    pub fn with_diarization(&self, enabled: bool) -> Self {
        Self {
            enable_diarization: enabled,
            ..(*self).clone()
        }
    }

    /// Enable entity detection.
    #[napi]
    pub fn with_entity_detection(&self, enabled: bool) -> Self {
        Self {
            enable_entity_detection: enabled,
            ..(*self).clone()
        }
    }

    /// Enable sentiment analysis.
    #[napi]
    pub fn with_sentiment_analysis(&self, enabled: bool) -> Self {
        Self {
            enable_sentiment_analysis: enabled,
            ..(*self).clone()
        }
    }
}

impl Default for JsTranscriptionConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for JsTranscriptionConfig {
    fn clone(&self) -> Self {
        Self {
            language: self.language,
            enable_diarization: self.enable_diarization,
            enable_entity_detection: self.enable_entity_detection,
            enable_sentiment_analysis: self.enable_sentiment_analysis,
        }
    }
}

// ============================================================================
// REQUEST WRAPPER TYPES (for client method signatures)
// ============================================================================

/// Request for audio transcription.
#[napi]
pub struct JsTranscriptionRequest {
    pub audio_bytes: Vec<u8>,
    pub model: Option<String>,
    pub language: Option<String>,
}

#[napi]
impl JsTranscriptionRequest {
    /// Create a new transcription request.
    #[napi(constructor)]
    pub fn new(audio_bytes: Vec<u8>) -> Self {
        Self {
            audio_bytes,
            model: None,
            language: None,
        }
    }

    #[napi]
    pub fn with_model(&self, model: String) -> Self {
        Self {
            model: Some(model),
            ..(*self).clone()
        }
    }

    #[napi]
    pub fn with_language(&self, language: String) -> Self {
        Self {
            language: Some(language),
            ..(*self).clone()
        }
    }
}

impl Clone for JsTranscriptionRequest {
    fn clone(&self) -> Self {
        Self {
            audio_bytes: self.audio_bytes.clone(),
            model: self.model.clone(),
            language: self.language.clone(),
        }
    }
}

/// Request for text-to-speech synthesis.
#[napi]
pub struct JsSynthesisRequest {
    pub text: String,
    pub voice_id: Option<String>,
    pub model: Option<String>,
}

#[napi]
impl JsSynthesisRequest {
    /// Create a new synthesis request.
    #[napi(constructor)]
    pub fn new(text: String) -> Self {
        Self {
            text,
            voice_id: None,
            model: None,
        }
    }

    #[napi]
    pub fn with_voice(&self, voice_id: String) -> Self {
        Self {
            voice_id: Some(voice_id),
            ..(*self).clone()
        }
    }

    #[napi]
    pub fn with_model(&self, model: String) -> Self {
        Self {
            model: Some(model),
            ..(*self).clone()
        }
    }
}

impl Clone for JsSynthesisRequest {
    fn clone(&self) -> Self {
        Self {
            text: self.text.clone(),
            voice_id: self.voice_id.clone(),
            model: self.model.clone(),
        }
    }
}
