//! OpenAI Realtime API bindings for Node.js/TypeScript.
//!
//! Provides access to the OpenAI Realtime API for real-time bidirectional communication.
//! Supports text and audio streaming with voice activity detection.

use llmkit::providers::specialized::openai_realtime::{
    RealtimeProvider, RealtimeSession, SessionConfig, VadConfig,
};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;
use tokio::sync::Mutex;

// ============================================================================
// VAD CONFIG
// ============================================================================

/// Voice Activity Detection configuration.
///
/// Controls how the Realtime API detects when the user has stopped speaking.
#[napi]
pub struct JsVadConfig {
    /// Silence duration in milliseconds to trigger end-of-turn (default: 500)
    pub silence_duration_ms: u32,
    /// Threshold for voice detection (0.0 to 1.0)
    pub threshold: Option<f64>,
}

#[napi]
impl JsVadConfig {
    /// Create a new VAD configuration.
    #[napi(constructor)]
    pub fn new(silence_duration_ms: Option<u32>, threshold: Option<f64>) -> Self {
        Self {
            silence_duration_ms: silence_duration_ms.unwrap_or(500),
            threshold,
        }
    }

    /// Create default VAD configuration.
    #[napi(js_name = "default", factory)]
    pub fn default_config() -> Self {
        Self {
            silence_duration_ms: 500,
            threshold: None,
        }
    }
}

impl Clone for JsVadConfig {
    fn clone(&self) -> Self {
        Self {
            silence_duration_ms: self.silence_duration_ms,
            threshold: self.threshold,
        }
    }
}

// Manual FromNapiValue implementation - JsVadConfig is only created by Rust
impl FromNapiValue for JsVadConfig {
    unsafe fn from_napi_value(
        env: napi::sys::napi_env,
        val: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        // Extract from JS object
        let obj = Object::from_napi_value(env, val)?;
        let silence_duration_ms: u32 = obj.get::<_, u32>("silenceDurationMs")?.unwrap_or(500);
        let threshold: Option<f64> = obj.get::<_, f64>("threshold")?;
        Ok(Self {
            silence_duration_ms,
            threshold,
        })
    }
}

impl From<JsVadConfig> for VadConfig {
    fn from(js_config: JsVadConfig) -> Self {
        VadConfig {
            silence_duration_ms: js_config.silence_duration_ms,
            threshold: js_config.threshold.map(|t| t as f32),
        }
    }
}

impl From<VadConfig> for JsVadConfig {
    fn from(config: VadConfig) -> Self {
        JsVadConfig {
            silence_duration_ms: config.silence_duration_ms,
            threshold: config.threshold.map(|t| t as f64),
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
#[napi]
pub struct JsSessionConfig {
    /// Voice model to use (e.g., "gpt-4o-realtime-preview")
    pub model: Option<String>,
    /// Modality: "text-and-audio", "text-only", or "audio-only"
    pub modalities: Vec<String>,
    /// Instructions for the model
    pub instructions: Option<String>,
    /// Voice to use for audio output: "alloy", "echo", "shimmer"
    pub voice: String,
    /// Input audio encoding: "pcm16" or "g711_ulaw"
    pub input_audio_format: String,
    /// Output audio encoding: "pcm16" or "g711_ulaw"
    pub output_audio_format: String,
    /// Voice activity detection (VAD) configuration
    pub voice_activity_detection: Option<JsVadConfig>,
    /// Maximum output tokens
    pub max_response_output_tokens: Option<u32>,
    /// Tool choice: "auto", "required", or "none"
    pub tool_choice: Option<String>,
    /// Temperature for sampling
    pub temperature: Option<f64>,
}

#[napi]
impl JsSessionConfig {
    /// Create a new session configuration with defaults.
    #[napi(constructor)]
    pub fn new() -> Self {
        Self {
            model: None,
            modalities: vec!["text-and-audio".to_string()],
            instructions: None,
            voice: "alloy".to_string(),
            input_audio_format: "pcm16".to_string(),
            output_audio_format: "pcm16".to_string(),
            voice_activity_detection: Some(JsVadConfig::default_config()),
            max_response_output_tokens: Some(4096),
            tool_choice: None,
            temperature: None,
        }
    }

    /// Set the model.
    #[napi(js_name = "withModel")]
    pub fn with_model(&self, model: String) -> Self {
        let mut config = self.clone();
        config.model = Some(model);
        config
    }

    /// Set the modalities.
    #[napi(js_name = "withModalities")]
    pub fn with_modalities(&self, modalities: Vec<String>) -> Self {
        let mut config = self.clone();
        config.modalities = modalities;
        config
    }

    /// Set the instructions.
    #[napi(js_name = "withInstructions")]
    pub fn with_instructions(&self, instructions: String) -> Self {
        let mut config = self.clone();
        config.instructions = Some(instructions);
        config
    }

    /// Set the voice.
    #[napi(js_name = "withVoice")]
    pub fn with_voice(&self, voice: String) -> Self {
        let mut config = self.clone();
        config.voice = voice;
        config
    }

    /// Set the input audio format.
    #[napi(js_name = "withInputAudioFormat")]
    pub fn with_input_audio_format(&self, format: String) -> Self {
        let mut config = self.clone();
        config.input_audio_format = format;
        config
    }

    /// Set the output audio format.
    #[napi(js_name = "withOutputAudioFormat")]
    pub fn with_output_audio_format(&self, format: String) -> Self {
        let mut config = self.clone();
        config.output_audio_format = format;
        config
    }

    /// Set the VAD configuration.
    #[napi(js_name = "withVad")]
    pub fn with_vad(&self, vad: &JsVadConfig) -> Self {
        let mut config = self.clone();
        config.voice_activity_detection = Some(vad.clone());
        config
    }

    /// Disable VAD.
    #[napi(js_name = "withoutVad")]
    pub fn without_vad(&self) -> Self {
        let mut config = self.clone();
        config.voice_activity_detection = None;
        config
    }

    /// Set the maximum output tokens.
    #[napi(js_name = "withMaxTokens")]
    pub fn with_max_tokens(&self, max_tokens: u32) -> Self {
        let mut config = self.clone();
        config.max_response_output_tokens = Some(max_tokens);
        config
    }

    /// Set the tool choice.
    #[napi(js_name = "withToolChoice")]
    pub fn with_tool_choice(&self, tool_choice: String) -> Self {
        let mut config = self.clone();
        config.tool_choice = Some(tool_choice);
        config
    }

    /// Set the temperature.
    #[napi(js_name = "withTemperature")]
    pub fn with_temperature(&self, temperature: f64) -> Self {
        let mut config = self.clone();
        config.temperature = Some(temperature);
        config
    }

    /// Create a text-only configuration.
    #[napi(js_name = "textOnly", factory)]
    pub fn text_only() -> Self {
        let mut config = Self::new();
        config.modalities = vec!["text-only".to_string()];
        config.voice_activity_detection = None;
        config
    }

    /// Create an audio-only configuration.
    #[napi(js_name = "audioOnly", factory)]
    pub fn audio_only() -> Self {
        let mut config = Self::new();
        config.modalities = vec!["audio-only".to_string()];
        config
    }
}

impl Clone for JsSessionConfig {
    fn clone(&self) -> Self {
        Self {
            model: self.model.clone(),
            modalities: self.modalities.clone(),
            instructions: self.instructions.clone(),
            voice: self.voice.clone(),
            input_audio_format: self.input_audio_format.clone(),
            output_audio_format: self.output_audio_format.clone(),
            voice_activity_detection: self.voice_activity_detection.clone(),
            max_response_output_tokens: self.max_response_output_tokens,
            tool_choice: self.tool_choice.clone(),
            temperature: self.temperature,
        }
    }
}

impl From<JsSessionConfig> for SessionConfig {
    fn from(js_config: JsSessionConfig) -> Self {
        SessionConfig {
            model: js_config.model,
            modalities: js_config.modalities,
            instructions: js_config.instructions,
            voice: js_config.voice,
            input_audio_format: js_config.input_audio_format,
            output_audio_format: js_config.output_audio_format,
            voice_activity_detection: js_config.voice_activity_detection.map(|v| v.into()),
            max_response_output_tokens: js_config.max_response_output_tokens,
            tools: None,
            tool_choice: js_config.tool_choice,
            temperature: js_config.temperature.map(|t| t as f32),
        }
    }
}

impl From<SessionConfig> for JsSessionConfig {
    fn from(config: SessionConfig) -> Self {
        JsSessionConfig {
            model: config.model,
            modalities: config.modalities,
            instructions: config.instructions,
            voice: config.voice,
            input_audio_format: config.input_audio_format,
            output_audio_format: config.output_audio_format,
            voice_activity_detection: config.voice_activity_detection.map(|v| v.into()),
            max_response_output_tokens: config.max_response_output_tokens,
            tool_choice: config.tool_choice,
            temperature: config.temperature.map(|t| t as f64),
        }
    }
}

// ============================================================================
// REALTIME SESSION
// ============================================================================

/// Active realtime session for bidirectional communication.
///
/// Provides methods to send text and audio to the OpenAI Realtime API.
#[napi]
pub struct JsRealtimeSession {
    inner: Arc<Mutex<RealtimeSession>>,
}

#[napi]
impl JsRealtimeSession {
    /// Send a text message.
    #[napi(js_name = "sendText")]
    pub async fn send_text(&self, text: String) -> napi::Result<()> {
        let session = self.inner.lock().await;
        session
            .send_text(&text)
            .await
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))
    }

    /// Send audio data.
    ///
    /// The audio should be in the format specified by input_audio_format
    /// in the session config (default: PCM16, 24kHz, mono).
    #[napi(js_name = "sendAudio")]
    pub async fn send_audio(&self, audio_data: Buffer) -> napi::Result<()> {
        let session = self.inner.lock().await;
        session
            .send_audio(audio_data.to_vec())
            .await
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))
    }

    /// Commit the audio buffer and trigger response generation.
    ///
    /// Call this after sending audio to indicate the end of input
    /// and trigger the model to generate a response.
    #[napi(js_name = "commitAudio")]
    pub async fn commit_audio(&self) -> napi::Result<()> {
        let session = self.inner.lock().await;
        session
            .commit_audio()
            .await
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))
    }

    /// Get the current session configuration.
    #[napi(js_name = "getConfig")]
    pub async fn get_config(&self) -> JsSessionConfig {
        let session = self.inner.lock().await;
        JsSessionConfig::from(session.get_config().await)
    }

    /// Update the session configuration.
    #[napi(js_name = "updateConfig")]
    pub async fn update_config(&self, config: &JsSessionConfig) -> napi::Result<()> {
        let session = self.inner.lock().await;
        session
            .update_config(config.clone().into())
            .await
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))
    }
}

// ============================================================================
// REALTIME PROVIDER
// ============================================================================

/// OpenAI Realtime API provider.
///
/// Creates and manages realtime sessions for bidirectional voice and text communication.
#[napi]
pub struct JsRealtimeProvider {
    api_key: String,
    model: String,
}

#[napi]
impl JsRealtimeProvider {
    /// Create a new Realtime provider with the given API key.
    #[napi(constructor)]
    pub fn new(api_key: String, model: Option<String>) -> Self {
        let model = model.unwrap_or_else(|| "gpt-4o-realtime-preview".to_string());
        Self { api_key, model }
    }

    /// Create from environment variable `OPENAI_API_KEY`.
    #[napi(js_name = "fromEnv", factory)]
    pub fn from_env() -> napi::Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            napi::Error::new(
                napi::Status::GenericFailure,
                "OPENAI_API_KEY environment variable not set",
            )
        })?;

        Ok(Self {
            api_key,
            model: "gpt-4o-realtime-preview".to_string(),
        })
    }

    /// Create a new realtime session.
    ///
    /// Establishes a WebSocket connection to the OpenAI Realtime API.
    #[napi(js_name = "createSession")]
    pub async fn create_session(
        &self,
        config: &JsSessionConfig,
    ) -> napi::Result<JsRealtimeSession> {
        let provider = RealtimeProvider::new(&self.api_key, &self.model);
        let session = provider
            .create_session(config.clone().into())
            .await
            .map_err(|e| napi::Error::new(napi::Status::GenericFailure, e.to_string()))?;

        Ok(JsRealtimeSession {
            inner: Arc::new(Mutex::new(session)),
        })
    }

    /// Get the model name.
    #[napi(getter)]
    pub fn model(&self) -> &str {
        &self.model
    }
}
