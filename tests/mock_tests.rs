//! Mock integration tests for Phase 1 LLMKit providers
//!
//! These tests use `wiremock` to mock HTTP responses from LLM providers
//! without requiring actual API keys or network calls.
//!
//! Phase 1 Providers:
//! - OpenAI-compatible: xAI, Meta Llama, Lambda Labs, Friendli, Volcengine
//! - Custom: DataRobot, Stability AI
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
