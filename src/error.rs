//! Error types for the ModelSuite library.

use thiserror::Error;

/// Main error type for ModelSuite operations.
#[derive(Debug, Error)]
pub enum Error {
    /// Provider not found or not configured
    #[error("Provider not found: {0}")]
    ProviderNotFound(String),

    /// Provider configuration error
    #[error("Provider configuration error: {0}")]
    Configuration(String),

    /// Authentication error (invalid or missing API key)
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Rate limit exceeded
    #[error("Rate limit exceeded: {message}. Retry after: {retry_after:?}")]
    RateLimited {
        message: String,
        retry_after: Option<std::time::Duration>,
    },

    /// Request validation error
    #[error("Invalid request: {0}")]
    InvalidRequest(String),

    /// Model not found or not supported
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    /// Content moderation triggered
    #[error("Content filtered: {0}")]
    ContentFiltered(String),

    /// Context length exceeded
    #[error("Context length exceeded: {0}")]
    ContextLengthExceeded(String),

    /// Network/HTTP error
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    /// JSON serialization/deserialization error
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    /// Streaming error
    #[error("Stream error: {0}")]
    Stream(String),

    /// Timeout error
    #[error("Request timed out")]
    Timeout,

    /// Server error from the provider
    #[error("Server error ({status}): {message}")]
    Server { status: u16, message: String },

    /// Feature not supported by the provider
    #[error("Feature not supported: {0}")]
    NotSupported(String),

    /// Generic/unknown error
    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Create a configuration error.
    pub fn config(message: impl Into<String>) -> Self {
        Error::Configuration(message.into())
    }

    /// Create an authentication error.
    pub fn auth(message: impl Into<String>) -> Self {
        Error::Authentication(message.into())
    }

    /// Create a rate limit error.
    pub fn rate_limited(
        message: impl Into<String>,
        retry_after: Option<std::time::Duration>,
    ) -> Self {
        Error::RateLimited {
            message: message.into(),
            retry_after,
        }
    }

    /// Create an invalid request error.
    pub fn invalid_request(message: impl Into<String>) -> Self {
        Error::InvalidRequest(message.into())
    }

    /// Create a server error.
    pub fn server(status: u16, message: impl Into<String>) -> Self {
        Error::Server {
            status,
            message: message.into(),
        }
    }

    /// Create a not supported error.
    pub fn not_supported(feature: impl Into<String>) -> Self {
        Error::NotSupported(feature.into())
    }

    /// Create a stream error.
    pub fn stream(message: impl Into<String>) -> Self {
        Error::Stream(message.into())
    }

    /// Create an other/generic error.
    pub fn other(message: impl Into<String>) -> Self {
        Error::Other(message.into())
    }

    /// Check if this error is retryable.
    pub fn is_retryable(&self) -> bool {
        match self {
            Error::RateLimited { .. } | Error::Timeout | Error::Network(_) => true,
            Error::Server { status, .. } => *status >= 500,
            _ => false,
        }
    }

    /// Get retry-after duration if applicable.
    pub fn retry_after(&self) -> Option<std::time::Duration> {
        match self {
            Error::RateLimited { retry_after, .. } => *retry_after,
            _ => None,
        }
    }
}

/// Result type alias for ModelSuite operations.
pub type Result<T> = std::result::Result<T, Error>;

/// API error response structure (common format).
#[derive(Debug, serde::Deserialize)]
pub struct ApiErrorResponse {
    #[serde(alias = "error")]
    pub error: ApiErrorDetail,
}

/// API error detail.
#[derive(Debug, serde::Deserialize)]
pub struct ApiErrorDetail {
    #[serde(alias = "type", alias = "code")]
    pub error_type: Option<String>,
    pub message: String,
}

impl From<ApiErrorResponse> for Error {
    fn from(resp: ApiErrorResponse) -> Self {
        let error_type = resp.error.error_type.as_deref().unwrap_or("unknown");
        let message = &resp.error.message;

        match error_type {
            "authentication_error" | "invalid_api_key" => Error::auth(message),
            "rate_limit_error" | "rate_limit_exceeded" => Error::rate_limited(message, None),
            "invalid_request_error" | "invalid_request" => Error::invalid_request(message),
            "model_not_found" | "model_not_found_error" => Error::ModelNotFound(message.clone()),
            "content_filter" | "content_policy_violation" => {
                Error::ContentFiltered(message.clone())
            }
            "context_length_exceeded" => Error::ContextLengthExceeded(message.clone()),
            "overloaded_error" | "server_error" => Error::server(500, message),
            _ => Error::other(format!("{}: {}", error_type, message)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_retryable() {
        assert!(Error::rate_limited("too many requests", None).is_retryable());
        assert!(Error::Timeout.is_retryable());
        assert!(Error::server(503, "overloaded").is_retryable());

        assert!(!Error::auth("invalid key").is_retryable());
        assert!(!Error::invalid_request("bad param").is_retryable());
    }

    #[test]
    fn test_error_display() {
        let err = Error::rate_limited("too fast", Some(std::time::Duration::from_secs(30)));
        assert!(err.to_string().contains("Rate limit"));
    }
}
