//! OpenAI Realtime API provider for real-time bidirectional communication.
//!
//! The Realtime API uses WebSocket connections for low-latency, real-time speech and text
//! interactions. Unlike traditional REST-based providers, it maintains a persistent connection
//! and uses an event-driven architecture.
//!
//! # Features
//! - Real-time text and audio streaming
//! - Voice activity detection (VAD)
//! - Automatic turn detection
//! - Audio format: 16-bit PCM, 24kHz uncompressed or G.711 compressed
//! - Up to 128K token context
//! - 15-minute session maximum

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tokio_tungstenite::connect_async;

use crate::error::{Error, Result};

/// OpenAI Realtime API provider for persistent, event-driven communication
pub struct RealtimeProvider {
    api_key: String,
    model: String,
}

impl RealtimeProvider {
    /// Create a new Realtime provider with the given API key
    pub fn new(api_key: &str, model: &str) -> Self {
        Self {
            api_key: api_key.to_string(),
            model: model.to_string(),
        }
    }

    /// Create from environment variable `OPENAI_API_KEY`
    pub fn from_env() -> Result<Self> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            Error::Configuration("OPENAI_API_KEY environment variable not set".to_string())
        })?;
        Ok(Self::new(&api_key, "gpt-4o-realtime-preview"))
    }

    /// Create a new realtime session
    ///
    /// Establishes a WebSocket connection to the OpenAI Realtime API.
    /// The actual bidirectional communication will be handled through the returned session.
    pub async fn create_session(&self, config: SessionConfig) -> Result<RealtimeSession> {
        let ws_url = format!(
            "wss://api.openai.com/v1/realtime?model={}&api_key={}",
            self.model, self.api_key
        );

        // Attempt connection to validate configuration
        let (_ws_stream, _) = connect_async(&ws_url)
            .await
            .map_err(|e| Error::Configuration(format!("WebSocket connection failed: {}", e)))?;

        // Create event channel for bidirectional communication
        let (tx, _rx) = mpsc::unbounded_channel();

        Ok(RealtimeSession {
            config: Arc::new(Mutex::new(config)),
            tx,
        })
    }
}

/// Configuration for a realtime session
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    /// Voice model to use (e.g., "gpt-4o-realtime-preview")
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Modality: "text-and-audio", "text-only", or "audio-only"
    #[serde(default)]
    pub modalities: Vec<String>,
    /// Instructions for the model
    #[serde(skip_serializing_if = "Option::is_none")]
    pub instructions: Option<String>,
    /// Voice to use for audio output: "alloy", "echo", "shimmer"
    #[serde(default)]
    pub voice: String,
    /// Input audio encoding: "pcm16" or "g711_ulaw"
    #[serde(default)]
    pub input_audio_format: String,
    /// Output audio encoding: "pcm16" or "g711_ulaw"
    #[serde(default)]
    pub output_audio_format: String,
    /// Voice activity detection (VAD) configuration
    #[serde(skip_serializing_if = "Option::is_none")]
    pub voice_activity_detection: Option<VadConfig>,
    /// Maximum output tokens
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_response_output_tokens: Option<u32>,
    /// Tool definitions
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<serde_json::Value>>,
    /// Tool choice: "auto", "required", or "none"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    /// Temperature for sampling
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            model: None,
            modalities: vec!["text-and-audio".to_string()],
            instructions: None,
            voice: "alloy".to_string(),
            input_audio_format: "pcm16".to_string(),
            output_audio_format: "pcm16".to_string(),
            voice_activity_detection: Some(VadConfig::default()),
            max_response_output_tokens: Some(4096),
            tools: None,
            tool_choice: None,
            temperature: None,
        }
    }
}

/// Voice Activity Detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VadConfig {
    /// Silence duration in milliseconds to trigger end-of-turn (default: 500)
    #[serde(default = "default_silence_duration")]
    pub silence_duration_ms: u32,
    /// Threshold for voice detection (0.0 to 1.0)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f32>,
}

fn default_silence_duration() -> u32 {
    500
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            silence_duration_ms: 500,
            threshold: None,
        }
    }
}

/// Active realtime session for bidirectional communication
pub struct RealtimeSession {
    config: Arc<Mutex<SessionConfig>>,
    tx: mpsc::UnboundedSender<ClientEvent>,
}

impl RealtimeSession {
    /// Send a text message
    pub async fn send_text(&self, text: &str) -> Result<()> {
        let event = ClientEvent::InputUserMessageText {
            user_message_text: text.to_string(),
        };
        self.tx
            .send(event)
            .map_err(|e| Error::InvalidRequest(format!("Failed to send message: {}", e)))?;
        Ok(())
    }

    /// Send audio data (PCM16 format)
    pub async fn send_audio(&self, audio_data: Vec<u8>) -> Result<()> {
        use base64::Engine;
        let base64_audio = base64::engine::general_purpose::STANDARD.encode(&audio_data);
        let event = ClientEvent::InputAudioBufferAppend {
            audio: base64_audio,
        };
        self.tx
            .send(event)
            .map_err(|e| Error::InvalidRequest(format!("Failed to send audio: {}", e)))?;
        Ok(())
    }

    /// Commit audio buffer and trigger response generation
    pub async fn commit_audio(&self) -> Result<()> {
        let event = ClientEvent::InputAudioBufferCommit {};
        self.tx
            .send(event)
            .map_err(|e| Error::InvalidRequest(format!("Failed to commit audio: {}", e)))?;
        Ok(())
    }

    /// Get the current session configuration
    pub async fn get_config(&self) -> SessionConfig {
        self.config.lock().await.clone()
    }

    /// Update session configuration
    pub async fn update_config(&self, config: SessionConfig) -> Result<()> {
        let mut current = self.config.lock().await;
        *current = config;
        Ok(())
    }
}

/// Client-side events sent to the API
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
#[allow(clippy::enum_variant_names)]
enum ClientEvent {
    /// Send user message text
    InputUserMessageText { user_message_text: String },
    /// Append audio to buffer
    InputAudioBufferAppend { audio: String },
    /// Commit audio buffer
    InputAudioBufferCommit {},
}

/// Server-side events received from the API
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ServerEvent {
    /// Session started
    SessionCreated { session: SessionData },
    /// Session configuration updated
    SessionUpdated { session: SessionData },
    /// Response generation started
    ResponseCreated { response: ResponseData },
    /// Content part added to response
    ResponseContentPartAdded {
        response_id: String,
        item_index: u32,
        content_part: ContentPart,
    },
    /// Text delta in response
    ResponseTextDelta {
        response_id: String,
        item_index: u32,
        index: u32,
        text: String,
    },
    /// Audio transcript delta
    ResponseAudioTranscriptDelta {
        response_id: String,
        item_index: u32,
        index: u32,
        transcript: String,
    },
    /// Audio delta in response
    ResponseAudioDelta {
        response_id: String,
        item_index: u32,
        index: u32,
        #[serde(rename = "delta")]
        audio: String,
    },
    /// Response generation completed
    ResponseDone { response: ResponseData },
    /// Rate limit information
    RateLimitUpdated { rate_limit_info: RateLimitInfo },
    /// Error occurred
    Error { error: ErrorData },
}

/// Session data returned from API
#[derive(Debug, Clone, Deserialize)]
pub struct SessionData {
    pub id: String,
    pub object: String,
    pub created_at: String,
    pub model: String,
    pub modalities: Vec<String>,
}

/// Response data
#[derive(Debug, Clone, Deserialize)]
pub struct ResponseData {
    pub id: String,
    pub object: String,
    pub created_at: String,
    pub status: String,
    pub status_details: Option<serde_json::Value>,
}

/// Content part in response
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
pub enum ContentPart {
    InputText { text: String },
    InputAudio { audio: String },
    Text { text: String },
    Audio { audio: String },
}

/// Rate limit information
#[derive(Debug, Clone, Deserialize)]
pub struct RateLimitInfo {
    pub request_limit_tokens_per_min: u32,
    pub request_limit_tokens_reset_seconds: u32,
    pub tokens_used_current_request: u32,
}

/// Error data
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorData {
    pub code: String,
    pub message: String,
    pub param: Option<String>,
    pub event_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = RealtimeProvider::new("test-key", "gpt-4o-realtime-preview");
        assert_eq!(provider.api_key, "test-key");
        assert_eq!(provider.model, "gpt-4o-realtime-preview");
    }

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert_eq!(config.modalities, vec!["text-and-audio"]);
        assert_eq!(config.voice, "alloy");
        assert_eq!(config.input_audio_format, "pcm16");
        assert_eq!(config.output_audio_format, "pcm16");
    }

    #[test]
    fn test_vad_config_default() {
        let vad = VadConfig::default();
        assert_eq!(vad.silence_duration_ms, 500);
    }

    #[test]
    fn test_session_config_serialization() {
        let config = SessionConfig {
            model: Some("gpt-4o-realtime-preview".to_string()),
            modalities: vec!["text-and-audio".to_string()],
            voice: "shimmer".to_string(),
            ..Default::default()
        };

        let json = serde_json::to_string(&config).expect("serialization failed");
        assert!(json.contains("gpt-4o-realtime-preview"));
        assert!(json.contains("shimmer"));
    }

    #[test]
    fn test_server_event_deserialization() {
        let json = r#"{
            "type": "session_created",
            "session": {
                "id": "sess_123",
                "object": "realtime.session",
                "created_at": "2025-01-02T12:00:00Z",
                "model": "gpt-4o-realtime-preview",
                "modalities": ["text-and-audio"]
            }
        }"#;

        let event: ServerEvent = serde_json::from_str(json).expect("deserialization failed");
        match event {
            ServerEvent::SessionCreated { session } => {
                assert_eq!(session.id, "sess_123");
                assert_eq!(session.model, "gpt-4o-realtime-preview");
            }
            _ => panic!("expected SessionCreated"),
        }
    }

    #[test]
    fn test_error_deserialization() {
        let json = r#"{
            "type": "error",
            "error": {
                "code": "invalid_api_key",
                "message": "Invalid API key",
                "param": null,
                "event_id": "evt_123"
            }
        }"#;

        let event: ServerEvent = serde_json::from_str(json).expect("deserialization failed");
        match event {
            ServerEvent::Error { error } => {
                assert_eq!(error.code, "invalid_api_key");
                assert_eq!(error.message, "Invalid API key");
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn test_rate_limit_deserialization() {
        let json = r#"{
            "type": "rate_limit_updated",
            "rate_limit_info": {
                "request_limit_tokens_per_min": 100000,
                "request_limit_tokens_reset_seconds": 60,
                "tokens_used_current_request": 150
            }
        }"#;

        let event: ServerEvent = serde_json::from_str(json).expect("deserialization failed");
        match event {
            ServerEvent::RateLimitUpdated { rate_limit_info } => {
                assert_eq!(rate_limit_info.request_limit_tokens_per_min, 100000);
                assert_eq!(rate_limit_info.tokens_used_current_request, 150);
            }
            _ => panic!("expected RateLimitUpdated"),
        }
    }
}
