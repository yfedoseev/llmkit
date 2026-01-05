//! Audio API bindings for Python.
//!
//! Provides access to ModelSuite audio functionality for speech-to-text transcription
//! and text-to-speech synthesis from various providers (Deepgram, AssemblyAI, ElevenLabs).

use pyo3::prelude::*;

// ============================================================================
// DEEPGRAM AUDIO (Speech-to-Text)
// ============================================================================

/// Deepgram API version for selecting model features and endpoints.
#[pyclass(name = "DeepgramVersion", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum PyDeepgramVersion {
    /// API v1 (2023-12-01) - legacy support
    #[default]
    V1 = 0,
    /// API v3 (2025-01-01) - latest with Nova-3 models
    V3 = 1,
}

/// Options for Deepgram transcription.
#[pyclass(name = "TranscribeOptions")]
#[derive(Clone)]
pub struct PyTranscribeOptions {
    pub model: Option<String>,
    pub smart_format: bool,
    pub diarize: bool,
    pub language: Option<String>,
    pub punctuate: bool,
}

#[pymethods]
impl PyTranscribeOptions {
    /// Create a new TranscribeOptions with defaults.
    ///
    /// Example:
    ///     >>> opts = TranscribeOptions()
    #[new]
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
    ///
    /// Args:
    ///     model: Model name (e.g., "nova-3" or "nova-2").
    ///
    /// Returns:
    ///     Self for method chaining.
    pub fn with_model(&self, model: String) -> Self {
        let mut opts = self.clone();
        opts.model = Some(model);
        opts
    }

    /// Enable smart formatting for better punctuation and capitalization.
    pub fn with_smart_format(&self, enabled: bool) -> Self {
        let mut opts = self.clone();
        opts.smart_format = enabled;
        opts
    }

    /// Enable speaker diarization to identify different speakers.
    pub fn with_diarize(&self, enabled: bool) -> Self {
        let mut opts = self.clone();
        opts.diarize = enabled;
        opts
    }

    /// Set the language of the audio.
    ///
    /// Args:
    ///     language: Language code (e.g., "en", "es", "fr").
    pub fn with_language(&self, language: String) -> Self {
        let mut opts = self.clone();
        opts.language = Some(language);
        opts
    }

    /// Enable automatic punctuation addition.
    pub fn with_punctuate(&self, enabled: bool) -> Self {
        let mut opts = self.clone();
        opts.punctuate = enabled;
        opts
    }

    #[getter]
    pub fn model(&self) -> Option<String> {
        self.model.clone()
    }

    #[getter]
    pub fn smart_format(&self) -> bool {
        self.smart_format
    }

    #[getter]
    pub fn diarize(&self) -> bool {
        self.diarize
    }

    #[getter]
    pub fn language(&self) -> Option<String> {
        self.language.clone()
    }

    #[getter]
    pub fn punctuate(&self) -> bool {
        self.punctuate
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscribeOptions(model={:?}, smart_format={}, diarize={}, language={:?}, punctuate={})",
            self.model, self.smart_format, self.diarize, self.language, self.punctuate
        )
    }
}

/// A single word from transcription with timing and confidence.
#[pyclass(name = "Word")]
#[derive(Clone)]
pub struct PyWord {
    pub word: String,
    pub start: f64,
    pub end: f64,
    pub confidence: f64,
    pub speaker: Option<u32>,
}

#[pymethods]
impl PyWord {
    #[getter]
    pub fn word(&self) -> &str {
        &self.word
    }

    #[getter]
    pub fn start(&self) -> f64 {
        self.start
    }

    #[getter]
    pub fn end(&self) -> f64 {
        self.end
    }

    #[getter]
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    #[getter]
    pub fn speaker(&self) -> Option<u32> {
        self.speaker
    }

    /// Duration of this word in seconds.
    #[getter]
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }

    fn __repr__(&self) -> String {
        format!(
            "Word(word='{}', start={:.2}, end={:.2}, confidence={:.2})",
            self.word, self.start, self.end, self.confidence
        )
    }
}

/// Response from a transcription request.
#[pyclass(name = "TranscribeResponse")]
#[derive(Clone)]
pub struct PyTranscribeResponse {
    pub transcript: String,
    pub confidence: Option<f64>,
    pub words: Vec<PyWord>,
    pub duration: Option<f64>,
    pub metadata: Option<String>,
}

#[pymethods]
impl PyTranscribeResponse {
    /// The complete transcribed text.
    #[getter]
    pub fn transcript(&self) -> &str {
        &self.transcript
    }

    /// Overall confidence score for the transcription.
    #[getter]
    pub fn confidence(&self) -> Option<f64> {
        self.confidence
    }

    /// Word-level details including timing and confidence.
    #[getter]
    pub fn words(&self) -> Vec<PyWord> {
        self.words.clone()
    }

    /// Duration of the audio in seconds.
    #[getter]
    pub fn duration(&self) -> Option<f64> {
        self.duration
    }

    /// Number of words in the transcription.
    #[getter]
    pub fn word_count(&self) -> usize {
        self.words.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscribeResponse(transcript='{}...', confidence={:?}, word_count={})",
            &self.transcript[..std::cmp::min(50, self.transcript.len())],
            self.confidence,
            self.words.len()
        )
    }
}

// ============================================================================
// ELEVENLABS AUDIO (Text-to-Speech)
// ============================================================================

/// Latency mode for ElevenLabs synthesis.
#[pyclass(name = "LatencyMode", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum PyLatencyMode {
    /// Lowest possible latency (fastest)
    LowestLatency = 0,
    /// Low latency
    LowLatency = 1,
    /// Balanced (default)
    #[default]
    Balanced = 2,
    /// High quality
    HighQuality = 3,
    /// Highest quality (slowest)
    HighestQuality = 4,
}

/// Voice settings for ElevenLabs synthesis.
#[pyclass(name = "VoiceSettings")]
#[derive(Clone)]
pub struct PyVoiceSettings {
    pub stability: f32,
    pub similarity_boost: f32,
    pub style: Option<f32>,
    pub use_speaker_boost: bool,
}

#[pymethods]
impl PyVoiceSettings {
    /// Create new voice settings with defaults.
    ///
    /// Args:
    ///     stability: Voice stability (0.0-1.0). Default: 0.5
    ///     similarity_boost: Similarity to voice (0.0-1.0). Default: 0.75
    #[new]
    #[pyo3(signature = (stability = 0.5, similarity_boost = 0.75))]
    pub fn new(stability: f32, similarity_boost: f32) -> Self {
        Self {
            stability,
            similarity_boost,
            style: None,
            use_speaker_boost: false,
        }
    }

    /// Set the style parameter (0.0-1.0) for stylization of speech.
    pub fn with_style(&self, style: f32) -> Self {
        let mut opts = self.clone();
        opts.style = Some(style);
        opts
    }

    /// Enable speaker boost for more consistent voice characteristics.
    pub fn with_speaker_boost(&self, enabled: bool) -> Self {
        let mut opts = self.clone();
        opts.use_speaker_boost = enabled;
        opts
    }

    #[getter]
    pub fn stability(&self) -> f32 {
        self.stability
    }

    #[getter]
    pub fn similarity_boost(&self) -> f32 {
        self.similarity_boost
    }

    #[getter]
    pub fn style(&self) -> Option<f32> {
        self.style
    }

    #[getter]
    pub fn use_speaker_boost(&self) -> bool {
        self.use_speaker_boost
    }

    fn __repr__(&self) -> String {
        format!(
            "VoiceSettings(stability={:.2}, similarity_boost={:.2}, style={:?}, use_speaker_boost={})",
            self.stability, self.similarity_boost, self.style, self.use_speaker_boost
        )
    }
}

/// Options for ElevenLabs text-to-speech synthesis.
#[pyclass(name = "SynthesizeOptions")]
#[derive(Clone)]
pub struct PySynthesizeOptions {
    pub model_id: Option<String>,
    pub voice_settings: Option<PyVoiceSettings>,
    pub latency_mode: PyLatencyMode,
    pub output_format: Option<String>,
}

#[pymethods]
impl PySynthesizeOptions {
    /// Create new synthesis options with defaults.
    #[new]
    pub fn new() -> Self {
        Self {
            model_id: None,
            voice_settings: Some(PyVoiceSettings::new(0.5, 0.75)),
            latency_mode: PyLatencyMode::Balanced,
            output_format: Some("mp3_44100_64".to_string()),
        }
    }

    /// Set the ElevenLabs model to use.
    pub fn with_model(&self, model_id: String) -> Self {
        let mut opts = self.clone();
        opts.model_id = Some(model_id);
        opts
    }

    /// Set voice settings for the synthesis.
    pub fn with_voice_settings(&self, settings: PyVoiceSettings) -> Self {
        let mut opts = self.clone();
        opts.voice_settings = Some(settings);
        opts
    }

    /// Set the latency mode for the synthesis.
    pub fn with_latency_mode(&self, mode: PyLatencyMode) -> Self {
        let mut opts = self.clone();
        opts.latency_mode = mode;
        opts
    }

    /// Set the output audio format.
    pub fn with_output_format(&self, format: String) -> Self {
        let mut opts = self.clone();
        opts.output_format = Some(format);
        opts
    }

    #[getter]
    pub fn model_id(&self) -> Option<String> {
        self.model_id.clone()
    }

    #[getter]
    pub fn voice_settings(&self) -> Option<PyVoiceSettings> {
        self.voice_settings.clone()
    }

    #[getter]
    pub fn latency_mode(&self) -> u32 {
        self.latency_mode as u32
    }

    #[getter]
    pub fn output_format(&self) -> Option<String> {
        self.output_format.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SynthesizeOptions(model={:?}, latency_mode={}, output_format={:?})",
            self.model_id, self.latency_mode as u32, self.output_format
        )
    }
}

/// Information about an available voice.
#[pyclass(name = "Voice")]
#[derive(Clone)]
pub struct PyVoice {
    pub voice_id: String,
    pub name: String,
    pub category: Option<String>,
    pub description: Option<String>,
}

#[pymethods]
impl PyVoice {
    /// Unique identifier for this voice.
    #[getter]
    pub fn voice_id(&self) -> &str {
        &self.voice_id
    }

    /// Human-readable name of the voice.
    #[getter]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Category of the voice (e.g., "premade").
    #[getter]
    pub fn category(&self) -> Option<String> {
        self.category.clone()
    }

    /// Description of the voice characteristics.
    #[getter]
    pub fn description(&self) -> Option<String> {
        self.description.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "Voice(id='{}', name='{}', category={:?})",
            self.voice_id, self.name, self.category
        )
    }
}

/// Response from text-to-speech synthesis.
#[pyclass(name = "SynthesizeResponse")]
#[derive(Clone)]
pub struct PySynthesizeResponse {
    pub audio_bytes: Vec<u8>,
    pub format: String,
    pub duration: Option<f64>,
}

#[pymethods]
impl PySynthesizeResponse {
    /// The synthesized audio data as bytes.
    #[getter]
    pub fn audio_bytes(&self) -> Vec<u8> {
        self.audio_bytes.clone()
    }

    /// The audio format (e.g., "mp3").
    #[getter]
    pub fn format(&self) -> &str {
        &self.format
    }

    /// Estimated duration of the audio in seconds (if available).
    #[getter]
    pub fn duration(&self) -> Option<f64> {
        self.duration
    }

    /// Size of the audio in bytes.
    #[getter]
    pub fn size(&self) -> usize {
        self.audio_bytes.len()
    }

    fn __repr__(&self) -> String {
        format!(
            "SynthesizeResponse(format='{}', size={} bytes, duration={:?})",
            self.format,
            self.audio_bytes.len(),
            self.duration
        )
    }
}

// ============================================================================
// ASSEMBLYAI AUDIO (Speech-to-Text)
// ============================================================================

/// Language for AssemblyAI transcription.
#[pyclass(name = "AudioLanguage", eq, eq_int)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PyAudioLanguage {
    English = 0,
    Spanish = 1,
    French = 2,
    German = 3,
    ChineseSimplified = 4,
    ChineseTraditional = 5,
    Japanese = 6,
}

/// Configuration for AssemblyAI transcription.
#[pyclass(name = "TranscriptionConfig")]
#[derive(Clone)]
pub struct PyTranscriptionConfig {
    pub language: Option<PyAudioLanguage>,
    pub enable_diarization: bool,
    pub enable_entity_detection: bool,
    pub enable_sentiment_analysis: bool,
}

#[pymethods]
impl PyTranscriptionConfig {
    /// Create new transcription configuration with defaults.
    #[new]
    pub fn new() -> Self {
        Self {
            language: None,
            enable_diarization: false,
            enable_entity_detection: false,
            enable_sentiment_analysis: false,
        }
    }

    /// Set the language for transcription.
    pub fn with_language(&self, language: PyAudioLanguage) -> Self {
        let mut config = self.clone();
        config.language = Some(language);
        config
    }

    /// Enable speaker diarization.
    pub fn with_diarization(&self, enabled: bool) -> Self {
        let mut config = self.clone();
        config.enable_diarization = enabled;
        config
    }

    /// Enable entity detection.
    pub fn with_entity_detection(&self, enabled: bool) -> Self {
        let mut config = self.clone();
        config.enable_entity_detection = enabled;
        config
    }

    /// Enable sentiment analysis.
    pub fn with_sentiment_analysis(&self, enabled: bool) -> Self {
        let mut config = self.clone();
        config.enable_sentiment_analysis = enabled;
        config
    }

    #[getter]
    pub fn language(&self) -> Option<u32> {
        self.language.map(|l| l as u32)
    }

    #[getter]
    pub fn enable_diarization(&self) -> bool {
        self.enable_diarization
    }

    #[getter]
    pub fn enable_entity_detection(&self) -> bool {
        self.enable_entity_detection
    }

    #[getter]
    pub fn enable_sentiment_analysis(&self) -> bool {
        self.enable_sentiment_analysis
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscriptionConfig(language={:?}, diarization={}, entity_detection={}, sentiment={})",
            self.language,
            self.enable_diarization,
            self.enable_entity_detection,
            self.enable_sentiment_analysis
        )
    }
}

// ============================================================================
// REQUEST WRAPPER TYPES (for client method signatures)
// ============================================================================

/// Request for audio transcription.
#[pyclass(name = "TranscriptionRequest")]
#[derive(Clone)]
pub struct PyTranscriptionRequest {
    pub audio_bytes: Vec<u8>,
    pub model: Option<String>,
    pub language: Option<String>,
}

#[pymethods]
impl PyTranscriptionRequest {
    /// Create a new transcription request.
    ///
    /// Args:
    ///     audio_bytes: The audio file as bytes (required)
    ///     model: The transcription model to use (e.g., "nova-3" for Deepgram)
    ///     language: Language code if required by the provider
    #[new]
    pub fn new(audio_bytes: Vec<u8>) -> Self {
        Self {
            audio_bytes,
            model: None,
            language: None,
        }
    }

    pub fn with_model(&self, model: String) -> Self {
        let mut req = self.clone();
        req.model = Some(model);
        req
    }

    pub fn with_language(&self, language: String) -> Self {
        let mut req = self.clone();
        req.language = Some(language);
        req
    }

    #[getter]
    pub fn audio_bytes(&self) -> Vec<u8> {
        self.audio_bytes.clone()
    }

    #[getter]
    pub fn model(&self) -> Option<String> {
        self.model.clone()
    }

    #[getter]
    pub fn language(&self) -> Option<String> {
        self.language.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "TranscriptionRequest(audio_bytes={} bytes, model={:?}, language={:?})",
            self.audio_bytes.len(),
            self.model,
            self.language
        )
    }
}

/// Request for text-to-speech synthesis.
#[pyclass(name = "SynthesisRequest")]
#[derive(Clone)]
pub struct PySynthesisRequest {
    pub text: String,
    pub voice_id: Option<String>,
    pub model: Option<String>,
}

#[pymethods]
impl PySynthesisRequest {
    /// Create a new synthesis request.
    ///
    /// Args:
    ///     text: The text to convert to speech (required)
    ///     voice_id: The voice identifier to use (e.g., "pNInY14gQrG92XwBIHVr" for ElevenLabs)
    ///     model: The synthesis model to use
    #[new]
    pub fn new(text: String) -> Self {
        Self {
            text,
            voice_id: None,
            model: None,
        }
    }

    pub fn with_voice(&self, voice_id: String) -> Self {
        let mut req = self.clone();
        req.voice_id = Some(voice_id);
        req
    }

    pub fn with_model(&self, model: String) -> Self {
        let mut req = self.clone();
        req.model = Some(model);
        req
    }

    #[getter]
    pub fn text(&self) -> &str {
        &self.text
    }

    #[getter]
    pub fn voice_id(&self) -> Option<String> {
        self.voice_id.clone()
    }

    #[getter]
    pub fn model(&self) -> Option<String> {
        self.model.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "SynthesisRequest(text='{}...', voice_id={:?}, model={:?})",
            &self.text[..std::cmp::min(50, self.text.len())],
            self.voice_id,
            self.model
        )
    }
}
