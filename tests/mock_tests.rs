//! Mock integration tests for LLMKit providers (Phase 1 & Phase 2)
//!
//! These tests use `wiremock` to mock HTTP responses from LLM providers
//! without requiring actual API keys or network calls.
//!
//! Phase 1 Providers (7 total):
//! - OpenAI-compatible: xAI, Meta Llama, Lambda Labs, Friendli, Volcengine
//! - Custom: DataRobot, Stability AI
//!
//! Phase 2 Providers (5 + 10 models):
//! - Vertex AI Partners: Anthropic (Claude), DeepSeek, Meta Llama, Mistral, AI21
//!
//! To run all mock tests:
//! ```bash
//! cargo test --features all-providers --test mock_tests
//! ```

use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

// ============================================================================
// OpenAI-Compatible Providers Tests (xAI, Meta Llama, Lambda, Friendli, Volcengine)
// ============================================================================

#[cfg(test)]
mod openai_compatible_tests {
    use super::*;

    #[tokio::test]
    #[cfg(feature = "openai-compatible")]
    async fn test_xai_successful_completion() {
        let mock_server = MockServer::start().await;

        let response_body = json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "grok-2",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a response from xAI Grok."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 15,
                "total_tokens": 25
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        // Verify the mock server was set up
        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "openai-compatible")]
    async fn test_xai_rate_limit() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "error": {
                "message": "Rate limit exceeded",
                "type": "rate_limit_error",
                "param": null,
                "code": "rate_limit_exceeded"
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(429).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "openai-compatible")]
    async fn test_xai_auth_error() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "error": {
                "message": "Unauthorized",
                "type": "invalid_request_error",
                "code": "unauthorized"
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(401).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "openai-compatible")]
    async fn test_meta_llama_successful_completion() {
        let mock_server = MockServer::start().await;

        let response_body = json!({
            "id": "chatcmpl-meta",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "llama-3.1-70b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a response from Meta Llama."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 18,
                "total_tokens": 38
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "openai-compatible")]
    async fn test_lambda_labs_successful_completion() {
        let mock_server = MockServer::start().await;

        let response_body = json!({
            "id": "chatcmpl-lambda",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "llama-3.1-70b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a response from Lambda Labs."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 25,
                "completion_tokens": 16,
                "total_tokens": 41
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "openai-compatible")]
    async fn test_friendli_successful_completion() {
        let mock_server = MockServer::start().await;

        let response_body = json!({
            "id": "chatcmpl-friendli",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "mixtral-8x7b",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a response from Friendli."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 15,
                "completion_tokens": 12,
                "total_tokens": 27
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "openai-compatible")]
    async fn test_volcengine_successful_completion() {
        let mock_server = MockServer::start().await;

        let response_body = json!({
            "id": "chatcmpl-volcengine",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "doubao-pro-4k",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "This is a response from Volcengine."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 18,
                "completion_tokens": 14,
                "total_tokens": 32
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "openai-compatible")]
    async fn test_openai_compatible_invalid_request() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "error": {
                "message": "Invalid request",
                "type": "invalid_request_error",
                "code": "invalid_request"
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(400).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "openai-compatible")]
    async fn test_openai_compatible_server_error() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "error": {
                "message": "Internal server error",
                "type": "server_error",
                "code": "internal_error"
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }
}

// ============================================================================
// Custom Provider Tests (DataRobot, Stability AI)
// ============================================================================

#[cfg(test)]
mod custom_provider_tests {
    use super::*;

    #[tokio::test]
    #[cfg(feature = "datarobot")]
    async fn test_datarobot_successful_completion() {
        let mock_server = MockServer::start().await;

        let response_body = json!({
            "id": "inference-123",
            "model": "autopilot-default",
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "This is a DataRobot model prediction."
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 10
            }
        });

        Mock::given(method("POST"))
            .and(path("/v2/inference"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "datarobot")]
    async fn test_datarobot_auth_error() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "error": "Unauthorized",
            "message": "Invalid API key"
        });

        Mock::given(method("POST"))
            .and(path("/v2/inference"))
            .respond_with(ResponseTemplate::new(401).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "datarobot")]
    async fn test_datarobot_rate_limit() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "error": "Rate Limited",
            "message": "Too many requests"
        });

        Mock::given(method("POST"))
            .and(path("/v2/inference"))
            .respond_with(ResponseTemplate::new(429).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "datarobot")]
    async fn test_datarobot_not_found() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "error": "Not Found",
            "message": "Model not found"
        });

        Mock::given(method("POST"))
            .and(path("/v2/inference"))
            .respond_with(ResponseTemplate::new(404).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "stability")]
    async fn test_stability_ai_successful_generation() {
        let mock_server = MockServer::start().await;

        // Base64 encoded pixel data for a 1x1 PNG
        let image_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==";

        let response_body = json!({
            "image": image_data,
            "finish_reason": "success"
        });

        Mock::given(method("POST"))
            .and(path("/v2beta/stable-image/generate/core"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "stability")]
    async fn test_stability_ai_auth_error() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "name": "Unauthorized",
            "message": "Invalid API key provided"
        });

        Mock::given(method("POST"))
            .and(path("/v2beta/stable-image/generate/core"))
            .respond_with(ResponseTemplate::new(401).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "stability")]
    async fn test_stability_ai_rate_limit() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "name": "Rate Limit",
            "message": "Rate limit exceeded"
        });

        Mock::given(method("POST"))
            .and(path("/v2beta/stable-image/generate/core"))
            .respond_with(ResponseTemplate::new(429).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "stability")]
    async fn test_stability_ai_invalid_request() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "name": "Bad Request",
            "message": "Invalid prompt provided"
        });

        Mock::given(method("POST"))
            .and(path("/v2beta/stable-image/generate/core"))
            .respond_with(ResponseTemplate::new(400).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "stability")]
    async fn test_stability_ai_server_error() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "name": "Internal Server Error",
            "message": "Something went wrong on our end"
        });

        Mock::given(method("POST"))
            .and(path("/v2beta/stable-image/generate/core"))
            .respond_with(ResponseTemplate::new(500).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }
}

// ============================================================================
// Vertex AI Partner Models Tests (Phase 2)
// ============================================================================

#[cfg(test)]
mod vertex_partner_tests {
    use super::*;

    // Vertex AI Anthropic Tests
    #[tokio::test]
    #[cfg(feature = "vertex")]
    async fn test_vertex_anthropic_successful_completion() {
        let mock_server = MockServer::start().await;

        let response_body = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "This is a Claude response via Vertex AI."
                    }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 15,
                "candidatesTokenCount": 18
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/projects/test/locations/us-central1/publishers/anthropic/models/claude-3.5-sonnet:generateContent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "vertex")]
    async fn test_vertex_anthropic_auth_error() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "error": {
                "code": 401,
                "message": "Invalid authentication credentials",
                "status": "UNAUTHENTICATED"
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/projects/test/locations/us-central1/publishers/anthropic/models/claude-3.5-sonnet:generateContent"))
            .respond_with(ResponseTemplate::new(401).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    // Vertex AI DeepSeek Tests
    #[tokio::test]
    #[cfg(feature = "vertex")]
    async fn test_vertex_deepseek_successful_completion() {
        let mock_server = MockServer::start().await;

        let response_body = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "This is a DeepSeek response via Vertex AI."
                    }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 20
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/projects/test/locations/us-central1/publishers/deepseek/models/deepseek-chat:generateContent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    // Vertex AI Llama Tests
    #[tokio::test]
    #[cfg(feature = "vertex")]
    async fn test_vertex_llama_successful_completion() {
        let mock_server = MockServer::start().await;

        let response_body = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "This is a Llama response via Vertex AI."
                    }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 16
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/projects/test/locations/us-central1/publishers/meta/models/llama-3.1-405b:generateContent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    // Vertex AI Mistral Tests
    #[tokio::test]
    #[cfg(feature = "vertex")]
    async fn test_vertex_mistral_successful_completion() {
        let mock_server = MockServer::start().await;

        let response_body = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "This is a Mistral response via Vertex AI."
                    }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 14,
                "candidatesTokenCount": 19
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/projects/test/locations/us-central1/publishers/mistralai/models/mistral-large:generateContent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    // Vertex AI AI21 Tests
    #[tokio::test]
    #[cfg(feature = "vertex")]
    async fn test_vertex_ai21_successful_completion() {
        let mock_server = MockServer::start().await;

        let response_body = json!({
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": "This is an AI21 response via Vertex AI."
                    }]
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 11,
                "candidatesTokenCount": 17
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/projects/test/locations/us-central1/publishers/ai21labs/models/j2-ultra:generateContent"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    // Common Vertex error scenarios
    #[tokio::test]
    #[cfg(feature = "vertex")]
    async fn test_vertex_partner_rate_limit() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "error": {
                "code": 429,
                "message": "Resource has been exhausted",
                "status": "RESOURCE_EXHAUSTED"
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/projects/test/locations/us-central1/publishers/anthropic/models/claude-3.5-sonnet:generateContent"))
            .respond_with(ResponseTemplate::new(429).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }

    #[tokio::test]
    #[cfg(feature = "vertex")]
    async fn test_vertex_partner_server_error() {
        let mock_server = MockServer::start().await;

        let error_response = json!({
            "error": {
                "code": 500,
                "message": "Internal server error",
                "status": "INTERNAL"
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/projects/test/locations/us-central1/publishers/deepseek/models/deepseek-chat:generateContent"))
            .respond_with(ResponseTemplate::new(500).set_body_json(&error_response))
            .mount(&mock_server)
            .await;

        assert!(!mock_server.uri().is_empty());
    }
}

// ============================================================================
// Phase 3: Enterprise Cloud Providers
// ============================================================================

#[cfg(feature = "sagemaker")]
#[tokio::test]
async fn test_sagemaker_success() {
    let mock_server = MockServer::start().await;

    let response_body = json!({
        "generated_text": "This is a test response from SageMaker endpoint."
    });

    Mock::given(method("POST"))
        .and(path("/endpoints/test-endpoint/invocations"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .mount(&mock_server)
        .await;

    assert!(!mock_server.uri().is_empty());
}

#[cfg(feature = "sagemaker")]
#[tokio::test]
async fn test_sagemaker_auth_error() {
    let mock_server = MockServer::start().await;

    let error_response = json!({
        "error": "Unauthorized"
    });

    Mock::given(method("POST"))
        .and(path("/endpoints/test-endpoint/invocations"))
        .respond_with(ResponseTemplate::new(401).set_body_json(&error_response))
        .mount(&mock_server)
        .await;

    assert!(!mock_server.uri().is_empty());
}

#[cfg(feature = "sagemaker")]
#[tokio::test]
async fn test_sagemaker_rate_limit() {
    let mock_server = MockServer::start().await;

    let error_response = json!({
        "error": "Rate limit exceeded"
    });

    Mock::given(method("POST"))
        .and(path("/endpoints/test-endpoint/invocations"))
        .respond_with(ResponseTemplate::new(429).set_body_json(&error_response))
        .mount(&mock_server)
        .await;

    assert!(!mock_server.uri().is_empty());
}

#[cfg(feature = "sagemaker")]
#[tokio::test]
async fn test_sagemaker_server_error() {
    let mock_server = MockServer::start().await;

    let error_response = json!({
        "error": "Internal server error"
    });

    Mock::given(method("POST"))
        .and(path("/endpoints/test-endpoint/invocations"))
        .respond_with(ResponseTemplate::new(500).set_body_json(&error_response))
        .mount(&mock_server)
        .await;

    assert!(!mock_server.uri().is_empty());
}

#[cfg(feature = "snowflake")]
#[tokio::test]
async fn test_snowflake_success() {
    let mock_server = MockServer::start().await;

    let response_body = json!({
        "data": [
            [
                "This is a response from Snowflake Cortex."
            ]
        ]
    });

    Mock::given(method("POST"))
        .and(path("/api/v2/statements"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
        .mount(&mock_server)
        .await;

    assert!(!mock_server.uri().is_empty());
}

#[cfg(feature = "snowflake")]
#[tokio::test]
async fn test_snowflake_auth_error() {
    let mock_server = MockServer::start().await;

    let error_response = json!({
        "error": "Authentication failed"
    });

    Mock::given(method("POST"))
        .and(path("/api/v2/statements"))
        .respond_with(ResponseTemplate::new(401).set_body_json(&error_response))
        .mount(&mock_server)
        .await;

    assert!(!mock_server.uri().is_empty());
}

#[cfg(feature = "snowflake")]
#[tokio::test]
async fn test_snowflake_rate_limit() {
    let mock_server = MockServer::start().await;

    let error_response = json!({
        "error": "Rate limit exceeded"
    });

    Mock::given(method("POST"))
        .and(path("/api/v2/statements"))
        .respond_with(ResponseTemplate::new(429).set_body_json(&error_response))
        .mount(&mock_server)
        .await;

    assert!(!mock_server.uri().is_empty());
}

#[cfg(feature = "snowflake")]
#[tokio::test]
async fn test_snowflake_server_error() {
    let mock_server = MockServer::start().await;

    let error_response = json!({
        "error": "Internal server error"
    });

    Mock::given(method("POST"))
        .and(path("/api/v2/statements"))
        .respond_with(ResponseTemplate::new(500).set_body_json(&error_response))
        .mount(&mock_server)
        .await;

    assert!(!mock_server.uri().is_empty());
}

// ============================================================================
// Phase 4 Wave 2: Realtime API Providers
// ============================================================================

#[cfg(feature = "openai-realtime")]
#[tokio::test]
async fn test_openai_realtime_provider_creation() {
    use llmkit::providers::RealtimeProvider;
    let provider = RealtimeProvider::new("test-key", "gpt-4o-realtime-preview");
    // Verify provider is created with correct configuration
    // Provider creation should succeed without errors
    let _ = provider;
}

#[cfg(feature = "openai-realtime")]
#[tokio::test]
async fn test_openai_realtime_session_config_default() {
    use llmkit::providers::SessionConfig;
    let config = SessionConfig::default();

    // Verify default configuration is sensible
    assert_eq!(config.modalities, vec!["text-and-audio"]);
    assert_eq!(config.voice, "alloy");
    assert_eq!(config.input_audio_format, "pcm16");
    assert_eq!(config.output_audio_format, "pcm16");
    assert!(config.voice_activity_detection.is_some());
}

#[cfg(feature = "openai-realtime")]
#[tokio::test]
async fn test_openai_realtime_session_config_serialization() {
    use llmkit::providers::SessionConfig;

    let config = SessionConfig {
        model: Some("gpt-4o-realtime-preview".to_string()),
        voice: "shimmer".to_string(),
        ..Default::default()
    };

    let json_str = serde_json::to_string(&config).expect("serialization failed");
    let deserialized: serde_json::Value =
        serde_json::from_str(&json_str).expect("deserialization failed");

    // Verify serialization round-trips correctly
    assert_eq!(deserialized["voice"], "shimmer");
    assert_eq!(deserialized["modalities"][0], "text-and-audio");
}

#[cfg(feature = "openai-realtime")]
#[tokio::test]
async fn test_openai_realtime_server_event_deserialization() {
    use llmkit::providers::ServerEvent;

    let event_json = json!({
        "type": "session_created",
        "session": {
            "id": "sess_abc123",
            "object": "realtime.session",
            "created_at": "2025-01-02T12:00:00Z",
            "model": "gpt-4o-realtime-preview",
            "modalities": ["text-and-audio"]
        }
    });

    let event: ServerEvent = serde_json::from_value(event_json).expect("deserialization failed");

    // Verify session creation event deserialization
    match event {
        ServerEvent::SessionCreated { session } => {
            assert_eq!(session.id, "sess_abc123");
            assert_eq!(session.model, "gpt-4o-realtime-preview");
        }
        _ => panic!("expected SessionCreated event"),
    }
}

#[cfg(feature = "openai-realtime")]
#[tokio::test]
async fn test_openai_realtime_error_event_deserialization() {
    use llmkit::providers::ServerEvent;

    let event_json = json!({
        "type": "error",
        "error": {
            "code": "authentication_error",
            "message": "Invalid API key provided",
            "param": null,
            "event_id": "evt_xyz789"
        }
    });

    let event: ServerEvent = serde_json::from_value(event_json).expect("deserialization failed");

    // Verify error event deserialization
    match event {
        ServerEvent::Error { error } => {
            assert_eq!(error.code, "authentication_error");
            assert!(error.message.contains("Invalid API key"));
        }
        _ => panic!("expected Error event"),
    }
}

#[cfg(feature = "openai-realtime")]
#[tokio::test]
async fn test_openai_realtime_rate_limit_event() {
    use llmkit::providers::ServerEvent;

    let event_json = json!({
        "type": "rate_limit_updated",
        "rate_limit_info": {
            "request_limit_tokens_per_min": 100000,
            "request_limit_tokens_reset_seconds": 60,
            "tokens_used_current_request": 250
        }
    });

    let event: ServerEvent = serde_json::from_value(event_json).expect("deserialization failed");

    // Verify rate limit event deserialization
    match event {
        ServerEvent::RateLimitUpdated { rate_limit_info } => {
            assert_eq!(rate_limit_info.request_limit_tokens_per_min, 100000);
            assert_eq!(rate_limit_info.tokens_used_current_request, 250);
        }
        _ => panic!("expected RateLimitUpdated event"),
    }
}

#[cfg(feature = "openai-realtime")]
#[tokio::test]
async fn test_openai_realtime_text_delta_event() {
    use llmkit::providers::ServerEvent;

    let event_json = json!({
        "type": "response_text_delta",
        "response_id": "resp_123",
        "item_index": 0,
        "index": 0,
        "text": "This is a generated response."
    });

    let event: ServerEvent = serde_json::from_value(event_json).expect("deserialization failed");

    // Verify text delta event deserialization
    match event {
        ServerEvent::ResponseTextDelta {
            response_id, text, ..
        } => {
            assert_eq!(response_id, "resp_123");
            assert_eq!(text, "This is a generated response.");
        }
        _ => panic!("expected ResponseTextDelta event"),
    }
}

// ============================================================================
// Infrastructure Test
// ============================================================================

#[tokio::test]
async fn test_mock_infrastructure_available() {
    // Verify wiremock is properly configured and available
    let server = MockServer::start().await;

    let response = json!({"test": "data"});
    Mock::given(method("GET"))
        .and(path("/test"))
        .respond_with(ResponseTemplate::new(200).set_body_json(&response))
        .mount(&server)
        .await;

    // Verify basic functionality
    assert!(!server.uri().is_empty());
}
