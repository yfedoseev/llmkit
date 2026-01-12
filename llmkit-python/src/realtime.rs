//! OpenAI Realtime API bindings for Python.
//!
//! Provides access to the OpenAI Realtime API for real-time bidirectional communication.
//! Supports text and audio streaming with voice activity detection.

use llmkit::providers::specialized::openai_realtime::{
    RealtimeProvider, RealtimeSession, SessionConfig, VadConfig,
};
use pyo3::prelude::*;
use std::sync::Arc;
use tokio::sync::Mutex;

// ============================================================================
// VAD CONFIG
// ============================================================================

/// Voice Activity Detection configuration.
///
/// Controls how the Realtime API detects when the user has stopped speaking.
#[pyclass(name = "VadConfig")]
#[derive(Clone)]
pub struct PyVadConfig {
    /// Silence duration in milliseconds to trigger end-of-turn (default: 500)
    #[pyo3(get, set)]
    pub silence_duration_ms: u32,
    /// Threshold for voice detection (0.0 to 1.0)
    #[pyo3(get, set)]
    pub threshold: Option<f32>,
}

#[pymethods]
impl PyVadConfig {
    /// Create a new VAD configuration.
    ///
    /// Args:
    ///     silence_duration_ms: Silence duration in ms to trigger end-of-turn (default: 500)
    ///     threshold: Voice detection threshold 0.0-1.0 (optional)
    #[new]
    #[pyo3(signature = (silence_duration_ms=500, threshold=None))]
    pub fn new(silence_duration_ms: u32, threshold: Option<f32>) -> Self {
        Self {
            silence_duration_ms,
            threshold,
        }
    }

    /// Create default VAD configuration.
    #[staticmethod]
    pub fn default_config() -> Self {
        Self {
            silence_duration_ms: 500,
            threshold: None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "VadConfig(silence_duration_ms={}, threshold={:?})",
            self.silence_duration_ms, self.threshold
        )
    }
}

impl From<PyVadConfig> for VadConfig {
    fn from(py_config: PyVadConfig) -> Self {
        VadConfig {
            silence_duration_ms: py_config.silence_duration_ms,
            threshold: py_config.threshold,
        }
    }
}

impl From<VadConfig> for PyVadConfig {
    fn from(config: VadConfig) -> Self {
        PyVadConfig {
            silence_duration_ms: config.silence_duration_ms,
            threshold: config.threshold,
        }
    }
}

// ============================================================================
// SESSION CONFIG
// ============================================================================

/// Configuration for a realtime session.
///
/// Configures the behavior of the OpenAI Realtime API session including
/// modality, voice, audio format, and VAD settings.
#[pyclass(name = "SessionConfig")]
#[derive(Clone)]
pub struct PySessionConfig {
    /// Voice model to use (e.g., "gpt-4o-realtime-preview")
    #[pyo3(get, set)]
    pub model: Option<String>,
    /// Modality: "text-and-audio", "text-only", or "audio-only"
    #[pyo3(get, set)]
    pub modalities: Vec<String>,
    /// Instructions for the model
    #[pyo3(get, set)]
    pub instructions: Option<String>,
    /// Voice to use for audio output: "alloy", "echo", "shimmer"
    #[pyo3(get, set)]
    pub voice: String,
    /// Input audio encoding: "pcm16" or "g711_ulaw"
    #[pyo3(get, set)]
    pub input_audio_format: String,
    /// Output audio encoding: "pcm16" or "g711_ulaw"
    #[pyo3(get, set)]
    pub output_audio_format: String,
    /// Voice activity detection (VAD) configuration
    #[pyo3(get, set)]
    pub voice_activity_detection: Option<PyVadConfig>,
    /// Maximum output tokens
    #[pyo3(get, set)]
    pub max_response_output_tokens: Option<u32>,
    /// Tool choice: "auto", "required", or "none"
    #[pyo3(get, set)]
    pub tool_choice: Option<String>,
    /// Temperature for sampling
    #[pyo3(get, set)]
    pub temperature: Option<f32>,
}

#[pymethods]
impl PySessionConfig {
    /// Create a new session configuration with defaults.
    ///
    /// Default configuration:
    /// - modalities: ["text-and-audio"]
    /// - voice: "alloy"
    /// - input_audio_format: "pcm16"
    /// - output_audio_format: "pcm16"
    /// - VAD enabled with 500ms silence threshold
    /// - max_response_output_tokens: 4096
    #[new]
    #[pyo3(signature = ())]
    pub fn new() -> Self {
        Self {
            model: None,
            modalities: vec!["text-and-audio".to_string()],
            instructions: None,
            voice: "alloy".to_string(),
            input_audio_format: "pcm16".to_string(),
            output_audio_format: "pcm16".to_string(),
            voice_activity_detection: Some(PyVadConfig::default_config()),
            max_response_output_tokens: Some(4096),
            tool_choice: None,
            temperature: None,
        }
    }

    /// Set the model.
    pub fn with_model(&self, model: String) -> Self {
        let mut config = self.clone();
        config.model = Some(model);
        config
    }

    /// Set the modalities.
    pub fn with_modalities(&self, modalities: Vec<String>) -> Self {
        let mut config = self.clone();
        config.modalities = modalities;
        config
    }

    /// Set the instructions.
    pub fn with_instructions(&self, instructions: String) -> Self {
        let mut config = self.clone();
        config.instructions = Some(instructions);
        config
    }

    /// Set the voice.
    pub fn with_voice(&self, voice: String) -> Self {
        let mut config = self.clone();
        config.voice = voice;
        config
    }

    /// Set the input audio format.
    pub fn with_input_audio_format(&self, format: String) -> Self {
        let mut config = self.clone();
        config.input_audio_format = format;
        config
    }

    /// Set the output audio format.
    pub fn with_output_audio_format(&self, format: String) -> Self {
        let mut config = self.clone();
        config.output_audio_format = format;
        config
    }

    /// Set the VAD configuration.
    pub fn with_vad(&self, vad: PyVadConfig) -> Self {
        let mut config = self.clone();
        config.voice_activity_detection = Some(vad);
        config
    }

    /// Disable VAD.
    pub fn without_vad(&self) -> Self {
        let mut config = self.clone();
        config.voice_activity_detection = None;
        config
    }

    /// Set the maximum output tokens.
    pub fn with_max_tokens(&self, max_tokens: u32) -> Self {
        let mut config = self.clone();
        config.max_response_output_tokens = Some(max_tokens);
        config
    }

    /// Set the tool choice.
    pub fn with_tool_choice(&self, tool_choice: String) -> Self {
        let mut config = self.clone();
        config.tool_choice = Some(tool_choice);
        config
    }

    /// Set the temperature.
    pub fn with_temperature(&self, temperature: f32) -> Self {
        let mut config = self.clone();
        config.temperature = Some(temperature);
        config
    }

    /// Create a text-only configuration.
    #[staticmethod]
    pub fn text_only() -> Self {
        let mut config = Self::new();
        config.modalities = vec!["text-only".to_string()];
        config.voice_activity_detection = None;
        config
    }

    /// Create an audio-only configuration.
    #[staticmethod]
    pub fn audio_only() -> Self {
        let mut config = Self::new();
        config.modalities = vec!["audio-only".to_string()];
        config
    }

    fn __repr__(&self) -> String {
        format!(
            "SessionConfig(modalities={:?}, voice='{}', input_format='{}', output_format='{}')",
            self.modalities, self.voice, self.input_audio_format, self.output_audio_format
        )
    }
}

impl From<PySessionConfig> for SessionConfig {
    fn from(py_config: PySessionConfig) -> Self {
        SessionConfig {
            model: py_config.model,
            modalities: py_config.modalities,
            instructions: py_config.instructions,
            voice: py_config.voice,
            input_audio_format: py_config.input_audio_format,
            output_audio_format: py_config.output_audio_format,
            voice_activity_detection: py_config.voice_activity_detection.map(|v| v.into()),
            max_response_output_tokens: py_config.max_response_output_tokens,
            tools: None,
            tool_choice: py_config.tool_choice,
            temperature: py_config.temperature,
        }
    }
}

impl From<SessionConfig> for PySessionConfig {
    fn from(config: SessionConfig) -> Self {
        PySessionConfig {
            model: config.model,
            modalities: config.modalities,
            instructions: config.instructions,
            voice: config.voice,
            input_audio_format: config.input_audio_format,
            output_audio_format: config.output_audio_format,
            voice_activity_detection: config.voice_activity_detection.map(|v| v.into()),
            max_response_output_tokens: config.max_response_output_tokens,
            tool_choice: config.tool_choice,
            temperature: config.temperature,
        }
    }
}

// ============================================================================
// REALTIME SESSION
// ============================================================================

/// Active realtime session for bidirectional communication.
///
/// Provides methods to send text and audio to the OpenAI Realtime API.
///
/// Example:
/// ```python
/// from llmkit import RealtimeProvider, SessionConfig
///
/// provider = RealtimeProvider.from_env()
/// session = provider.create_session(SessionConfig())
///
/// # Send text
/// session.send_text("Hello, how are you?")
///
/// # Send audio (PCM16 format)
/// session.send_audio(audio_bytes)
/// session.commit_audio()
/// ```
#[pyclass(name = "RealtimeSession")]
pub struct PyRealtimeSession {
    inner: Arc<Mutex<RealtimeSession>>,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyRealtimeSession {
    /// Send a text message.
    ///
    /// Args:
    ///     text: The text message to send
    ///
    /// Raises:
    ///     RuntimeError: If sending fails
    fn send_text(&self, py: Python<'_>, text: String) -> PyResult<()> {
        let inner = self.inner.clone();
        let runtime = self.runtime.clone();
        py.detach(|| {
            runtime.block_on(async move {
                let session = inner.lock().await;
                session
                    .send_text(&text)
                    .await
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            })
        })
    }

    /// Send audio data.
    ///
    /// The audio should be in the format specified by input_audio_format
    /// in the session config (default: PCM16, 24kHz, mono).
    ///
    /// Args:
    ///     audio_data: Raw audio bytes
    ///
    /// Raises:
    ///     RuntimeError: If sending fails
    fn send_audio(&self, py: Python<'_>, audio_data: Vec<u8>) -> PyResult<()> {
        let inner = self.inner.clone();
        let runtime = self.runtime.clone();
        py.detach(|| {
            runtime.block_on(async move {
                let session = inner.lock().await;
                session
                    .send_audio(audio_data)
                    .await
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            })
        })
    }

    /// Commit the audio buffer and trigger response generation.
    ///
    /// Call this after sending audio to indicate the end of input
    /// and trigger the model to generate a response.
    ///
    /// Raises:
    ///     RuntimeError: If committing fails
    fn commit_audio(&self, py: Python<'_>) -> PyResult<()> {
        let inner = self.inner.clone();
        let runtime = self.runtime.clone();
        py.detach(|| {
            runtime.block_on(async move {
                let session = inner.lock().await;
                session
                    .commit_audio()
                    .await
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            })
        })
    }

    /// Get the current session configuration.
    ///
    /// Returns:
    ///     SessionConfig: The current configuration
    fn get_config(&self, py: Python<'_>) -> PyResult<PySessionConfig> {
        let inner = self.inner.clone();
        let runtime = self.runtime.clone();
        py.detach(|| {
            runtime.block_on(async move {
                let session = inner.lock().await;
                Ok(PySessionConfig::from(session.get_config().await))
            })
        })
    }

    /// Update the session configuration.
    ///
    /// Args:
    ///     config: The new configuration
    ///
    /// Raises:
    ///     RuntimeError: If updating fails
    fn update_config(&self, py: Python<'_>, config: PySessionConfig) -> PyResult<()> {
        let inner = self.inner.clone();
        let runtime = self.runtime.clone();
        py.detach(|| {
            runtime.block_on(async move {
                let session = inner.lock().await;
                session
                    .update_config(config.into())
                    .await
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            })
        })
    }

    fn __repr__(&self) -> String {
        "RealtimeSession(connected)".to_string()
    }
}

// ============================================================================
// REALTIME PROVIDER
// ============================================================================

/// OpenAI Realtime API provider.
///
/// Creates and manages realtime sessions for bidirectional voice and text communication.
///
/// Example:
/// ```python
/// from llmkit import RealtimeProvider, SessionConfig
///
/// # Create from environment variable (OPENAI_API_KEY)
/// provider = RealtimeProvider.from_env()
///
/// # Or with explicit API key
/// provider = RealtimeProvider("your-api-key")
///
/// # Create a session
/// config = SessionConfig().with_voice("shimmer").with_instructions("You are helpful.")
/// session = provider.create_session(config)
/// ```
#[pyclass(name = "RealtimeProvider")]
pub struct PyRealtimeProvider {
    api_key: String,
    model: String,
    runtime: Arc<tokio::runtime::Runtime>,
}

#[pymethods]
impl PyRealtimeProvider {
    /// Create a new Realtime provider with the given API key.
    ///
    /// Args:
    ///     api_key: OpenAI API key
    ///     model: Model to use (default: "gpt-4o-realtime-preview")
    #[new]
    #[pyo3(signature = (api_key, model=None))]
    pub fn new(api_key: String, model: Option<String>) -> PyResult<Self> {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );

        let model = model.unwrap_or_else(|| "gpt-4o-realtime-preview".to_string());

        Ok(Self {
            api_key,
            model,
            runtime,
        })
    }

    /// Create from environment variable `OPENAI_API_KEY`.
    ///
    /// Returns:
    ///     RealtimeProvider: A new provider instance
    ///
    /// Raises:
    ///     RuntimeError: If OPENAI_API_KEY is not set
    #[staticmethod]
    pub fn from_env() -> PyResult<Self> {
        let runtime = Arc::new(
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?,
        );

        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("OPENAI_API_KEY environment variable not set")
        })?;

        Ok(Self {
            api_key,
            model: "gpt-4o-realtime-preview".to_string(),
            runtime,
        })
    }

    /// Create a new realtime session.
    ///
    /// Establishes a WebSocket connection to the OpenAI Realtime API.
    ///
    /// Args:
    ///     config: Session configuration
    ///
    /// Returns:
    ///     RealtimeSession: An active session for bidirectional communication
    ///
    /// Raises:
    ///     RuntimeError: If connection fails
    fn create_session(
        &self,
        py: Python<'_>,
        config: PySessionConfig,
    ) -> PyResult<PyRealtimeSession> {
        let api_key = self.api_key.clone();
        let model = self.model.clone();
        let runtime = self.runtime.clone();

        let session = py.detach(|| {
            runtime.block_on(async move {
                let provider = RealtimeProvider::new(&api_key, &model);
                provider
                    .create_session(config.into())
                    .await
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            })
        })?;

        Ok(PyRealtimeSession {
            inner: Arc::new(Mutex::new(session)),
            runtime: self.runtime.clone(),
        })
    }

    /// Get the model name.
    #[getter]
    fn model(&self) -> &str {
        &self.model
    }

    fn __repr__(&self) -> String {
        format!("RealtimeProvider(model='{}')", self.model)
    }
}
