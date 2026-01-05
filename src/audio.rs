//! Audio APIs for text-to-speech (TTS) and speech-to-text (STT).
//!
//! This module provides unified interfaces for audio synthesis and transcription
//! across various providers including OpenAI, ElevenLabs, Deepgram, and others.
//!
//! # Text-to-Speech Example
//!
//! ```ignore
//! use modelsuite::{SpeechProvider, SpeechRequest, AudioFormat};
//!
//! // Create provider
//! let provider = OpenAIProvider::from_env()?;
//!
//! // Generate speech
//! let request = SpeechRequest::new("tts-1", "Hello, world!", "alloy");
//!
//! let response = provider.speech(request).await?;
//! std::fs::write("output.mp3", &response.audio)?;
//! ```
//!
//! # Speech-to-Text Example
//!
//! ```ignore
//! use modelsuite::{TranscriptionProvider, TranscriptionRequest, AudioInput};
//!
//! // Create provider
//! let provider = OpenAIProvider::from_env()?;
//!
//! // Transcribe audio
//! let request = TranscriptionRequest::new("whisper-1", AudioInput::file("audio.mp3"));
//!
//! let response = provider.transcribe(request).await?;
//! println!("Transcription: {}", response.text);
//! ```

use std::path::PathBuf;
use std::pin::Pin;

use async_trait::async_trait;
use bytes::Bytes;
use futures::Stream;
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

// ============================================================================
// Text-to-Speech (TTS)
// ============================================================================

/// Request for generating speech from text.
#[derive(Debug, Clone)]
pub struct SpeechRequest {
    /// The text to convert to speech.
    pub input: String,
    /// The model to use (e.g., "tts-1", "tts-1-hd").
    pub model: String,
    /// The voice to use (e.g., "alloy", "echo", "fable").
    pub voice: String,
    /// Audio format for the output.
    pub response_format: Option<AudioFormat>,
    /// Speed of speech (0.25 to 4.0, default 1.0).
    pub speed: Option<f32>,
}

impl SpeechRequest {
    /// Create a new speech request.
    pub fn new(
        model: impl Into<String>,
        input: impl Into<String>,
        voice: impl Into<String>,
    ) -> Self {
        Self {
            input: input.into(),
            model: model.into(),
            voice: voice.into(),
            response_format: None,
            speed: None,
        }
    }

    /// Set the audio format.
    pub fn with_format(mut self, format: AudioFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Set the speech speed (0.25 to 4.0).
    pub fn with_speed(mut self, speed: f32) -> Self {
        self.speed = Some(speed.clamp(0.25, 4.0));
        self
    }
}

/// Audio output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AudioFormat {
    /// MP3 audio (default).
    #[default]
    Mp3,
    /// Opus audio (for WebRTC).
    Opus,
    /// AAC audio.
    Aac,
    /// FLAC lossless audio.
    Flac,
    /// WAV audio.
    Wav,
    /// Raw PCM audio.
    Pcm,
}

impl AudioFormat {
    /// Get the file extension for this format.
    pub fn extension(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "mp3",
            AudioFormat::Opus => "opus",
            AudioFormat::Aac => "aac",
            AudioFormat::Flac => "flac",
            AudioFormat::Wav => "wav",
            AudioFormat::Pcm => "pcm",
        }
    }

    /// Get the MIME type for this format.
    pub fn mime_type(&self) -> &'static str {
        match self {
            AudioFormat::Mp3 => "audio/mpeg",
            AudioFormat::Opus => "audio/opus",
            AudioFormat::Aac => "audio/aac",
            AudioFormat::Flac => "audio/flac",
            AudioFormat::Wav => "audio/wav",
            AudioFormat::Pcm => "audio/L16",
        }
    }
}

/// Response from a speech generation request.
#[derive(Debug, Clone)]
pub struct SpeechResponse {
    /// The generated audio data.
    pub audio: Vec<u8>,
    /// The format of the audio.
    pub format: AudioFormat,
    /// Duration of the audio in seconds (if known).
    pub duration_seconds: Option<f32>,
}

impl SpeechResponse {
    /// Create a new speech response.
    pub fn new(audio: Vec<u8>, format: AudioFormat) -> Self {
        Self {
            audio,
            format,
            duration_seconds: None,
        }
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration_seconds = Some(duration);
        self
    }

    /// Save the audio to a file.
    pub fn save(&self, path: impl Into<PathBuf>) -> std::io::Result<()> {
        std::fs::write(path.into(), &self.audio)
    }
}

/// Information about a voice.
#[derive(Debug, Clone)]
pub struct VoiceInfo {
    /// Voice ID.
    pub id: String,
    /// Voice name.
    pub name: String,
    /// Voice description.
    pub description: Option<String>,
    /// Voice gender (if applicable).
    pub gender: Option<String>,
    /// Language/locale.
    pub locale: Option<String>,
}

/// Trait for providers that support text-to-speech.
#[async_trait]
pub trait SpeechProvider: Send + Sync {
    /// Get the provider name.
    fn name(&self) -> &str;

    /// Generate speech from text.
    async fn speech(&self, request: SpeechRequest) -> Result<SpeechResponse>;

    /// Generate speech as a stream (for real-time playback).
    async fn speech_stream(
        &self,
        request: SpeechRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Bytes>> + Send>>> {
        // Default implementation: generate full audio and yield as single chunk
        let response = self.speech(request).await?;
        let bytes = Bytes::from(response.audio);
        let stream = futures::stream::once(async move { Ok(bytes) });
        Ok(Box::pin(stream))
    }

    /// Get available voices for this provider.
    fn available_voices(&self) -> &[VoiceInfo] {
        &[]
    }

    /// Get supported audio formats.
    fn supported_formats(&self) -> &[AudioFormat] {
        &[AudioFormat::Mp3]
    }

    /// Get the default model for this provider.
    fn default_speech_model(&self) -> Option<&str> {
        None
    }
}

// ============================================================================
// Speech-to-Text (STT) / Transcription
// ============================================================================

/// Request for transcribing audio to text.
#[derive(Debug, Clone)]
pub struct TranscriptionRequest {
    /// The audio to transcribe.
    pub audio: AudioInput,
    /// The model to use (e.g., "whisper-1").
    pub model: String,
    /// Language of the audio (ISO-639-1 code).
    pub language: Option<String>,
    /// Prompt to guide transcription style.
    pub prompt: Option<String>,
    /// Response format.
    pub response_format: Option<TranscriptFormat>,
    /// Timestamp granularities to include.
    pub timestamp_granularities: Option<Vec<TimestampGranularity>>,
}

impl TranscriptionRequest {
    /// Create a new transcription request.
    pub fn new(model: impl Into<String>, audio: AudioInput) -> Self {
        Self {
            audio,
            model: model.into(),
            language: None,
            prompt: None,
            response_format: None,
            timestamp_granularities: None,
        }
    }

    /// Set the language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set a prompt to guide transcription.
    pub fn with_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Set the response format.
    pub fn with_format(mut self, format: TranscriptFormat) -> Self {
        self.response_format = Some(format);
        self
    }

    /// Enable word-level timestamps.
    pub fn with_word_timestamps(mut self) -> Self {
        self.timestamp_granularities = Some(vec![TimestampGranularity::Word]);
        self
    }

    /// Enable segment-level timestamps.
    pub fn with_segment_timestamps(mut self) -> Self {
        self.timestamp_granularities = Some(vec![TimestampGranularity::Segment]);
        self
    }
}

/// Input audio source.
#[derive(Debug, Clone)]
pub enum AudioInput {
    /// Path to a local audio file.
    File(PathBuf),
    /// Audio data in memory.
    Bytes {
        data: Vec<u8>,
        filename: String,
        media_type: String,
    },
    /// URL to an audio file.
    Url(String),
}

impl AudioInput {
    /// Create an input from a file path.
    pub fn file(path: impl Into<PathBuf>) -> Self {
        AudioInput::File(path.into())
    }

    /// Create an input from bytes.
    pub fn bytes(
        data: Vec<u8>,
        filename: impl Into<String>,
        media_type: impl Into<String>,
    ) -> Self {
        AudioInput::Bytes {
            data,
            filename: filename.into(),
            media_type: media_type.into(),
        }
    }

    /// Create an input from a URL.
    pub fn url(url: impl Into<String>) -> Self {
        AudioInput::Url(url.into())
    }
}

/// Response format for transcription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptFormat {
    /// Plain text.
    #[default]
    Text,
    /// JSON with metadata.
    Json,
    /// Verbose JSON with timing info.
    VerboseJson,
    /// SRT subtitles.
    Srt,
    /// VTT subtitles.
    Vtt,
}

/// Timestamp granularity options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TimestampGranularity {
    /// Word-level timestamps.
    Word,
    /// Segment-level timestamps.
    Segment,
}

/// Response from a transcription request.
#[derive(Debug, Clone)]
pub struct TranscriptionResponse {
    /// The transcribed text.
    pub text: String,
    /// Detected language.
    pub language: Option<String>,
    /// Duration of the audio in seconds.
    pub duration: Option<f32>,
    /// Transcript segments with timing.
    pub segments: Option<Vec<TranscriptSegment>>,
    /// Word-level timing information.
    pub words: Option<Vec<TranscriptWord>>,
}

impl TranscriptionResponse {
    /// Create a new transcription response.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            language: None,
            duration: None,
            segments: None,
            words: None,
        }
    }

    /// Set the language.
    pub fn with_language(mut self, language: impl Into<String>) -> Self {
        self.language = Some(language.into());
        self
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set segments.
    pub fn with_segments(mut self, segments: Vec<TranscriptSegment>) -> Self {
        self.segments = Some(segments);
        self
    }

    /// Set words.
    pub fn with_words(mut self, words: Vec<TranscriptWord>) -> Self {
        self.words = Some(words);
        self
    }
}

/// A segment of the transcript with timing.
#[derive(Debug, Clone)]
pub struct TranscriptSegment {
    /// Segment index.
    pub id: usize,
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
    /// Segment text.
    pub text: String,
}

/// A word with timing information.
#[derive(Debug, Clone)]
pub struct TranscriptWord {
    /// The word.
    pub word: String,
    /// Start time in seconds.
    pub start: f32,
    /// End time in seconds.
    pub end: f32,
}

/// Trait for providers that support speech-to-text transcription.
#[async_trait]
pub trait TranscriptionProvider: Send + Sync {
    /// Get the provider name.
    fn name(&self) -> &str;

    /// Transcribe audio to text.
    async fn transcribe(&self, request: TranscriptionRequest) -> Result<TranscriptionResponse>;

    /// Translate audio to English text (for non-English audio).
    async fn translate(&self, _request: TranscriptionRequest) -> Result<TranscriptionResponse> {
        Err(Error::not_supported("Audio translation"))
    }

    /// Get supported audio formats for input.
    fn supported_input_formats(&self) -> &[&str] {
        &["mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"]
    }

    /// Get maximum file size in bytes.
    fn max_file_size(&self) -> usize {
        25 * 1024 * 1024 // 25 MB default
    }

    /// Get the default model for this provider.
    fn default_transcription_model(&self) -> Option<&str> {
        None
    }
}

/// Information about an audio model.
#[derive(Debug, Clone)]
pub struct AudioModelInfo {
    /// Model ID/name.
    pub id: &'static str,
    /// Provider that offers this model.
    pub provider: &'static str,
    /// Model type (TTS or STT).
    pub model_type: AudioModelType,
    /// Supported languages.
    pub languages: &'static [&'static str],
    /// Price per minute (USD).
    pub price_per_minute: f64,
}

/// Type of audio model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioModelType {
    /// Text-to-speech.
    Tts,
    /// Speech-to-text.
    Stt,
}

/// Registry of known audio models.
pub static AUDIO_MODELS: &[AudioModelInfo] = &[
    // OpenAI TTS
    AudioModelInfo {
        id: "tts-1",
        provider: "openai",
        model_type: AudioModelType::Tts,
        languages: &["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
        price_per_minute: 0.015,
    },
    AudioModelInfo {
        id: "tts-1-hd",
        provider: "openai",
        model_type: AudioModelType::Tts,
        languages: &["en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko"],
        price_per_minute: 0.030,
    },
    // OpenAI STT
    AudioModelInfo {
        id: "whisper-1",
        provider: "openai",
        model_type: AudioModelType::Stt,
        languages: &[
            "en", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ko", "ar", "hi",
        ],
        price_per_minute: 0.006,
    },
];

/// Get audio model info by ID.
pub fn get_audio_model_info(model_id: &str) -> Option<&'static AudioModelInfo> {
    AUDIO_MODELS.iter().find(|m| m.id == model_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speech_request_builder() {
        let request = SpeechRequest::new("tts-1", "Hello", "alloy")
            .with_format(AudioFormat::Mp3)
            .with_speed(1.5);

        assert_eq!(request.model, "tts-1");
        assert_eq!(request.input, "Hello");
        assert_eq!(request.voice, "alloy");
        assert_eq!(request.response_format, Some(AudioFormat::Mp3));
        assert_eq!(request.speed, Some(1.5));
    }

    #[test]
    fn test_speed_clamping() {
        let request = SpeechRequest::new("tts-1", "test", "alloy").with_speed(10.0);
        assert_eq!(request.speed, Some(4.0));

        let request = SpeechRequest::new("tts-1", "test", "alloy").with_speed(0.1);
        assert_eq!(request.speed, Some(0.25));
    }

    #[test]
    fn test_audio_format() {
        assert_eq!(AudioFormat::Mp3.extension(), "mp3");
        assert_eq!(AudioFormat::Mp3.mime_type(), "audio/mpeg");
        assert_eq!(AudioFormat::Opus.extension(), "opus");
    }

    #[test]
    fn test_transcription_request_builder() {
        let request = TranscriptionRequest::new("whisper-1", AudioInput::file("test.mp3"))
            .with_language("en")
            .with_word_timestamps();

        assert_eq!(request.model, "whisper-1");
        assert_eq!(request.language, Some("en".to_string()));
        assert!(request.timestamp_granularities.is_some());
    }

    #[test]
    fn test_audio_input() {
        let file_input = AudioInput::file("audio.mp3");
        assert!(matches!(file_input, AudioInput::File(_)));

        let url_input = AudioInput::url("https://example.com/audio.mp3");
        assert!(matches!(url_input, AudioInput::Url(_)));
    }

    #[test]
    fn test_audio_model_registry() {
        let model = get_audio_model_info("whisper-1");
        assert!(model.is_some());
        let model = model.unwrap();
        assert_eq!(model.provider, "openai");
        assert_eq!(model.model_type, AudioModelType::Stt);
    }
}
