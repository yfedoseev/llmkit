//! Python exception hierarchy for LLMKit errors

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

// Base exception
create_exception!(
    llmkit,
    LLMKitError,
    PyException,
    "Base exception for LLMKit errors."
);

// Specific exceptions
create_exception!(
    llmkit,
    ProviderNotFoundError,
    LLMKitError,
    "Provider not found or not configured."
);

create_exception!(
    llmkit,
    ConfigurationError,
    LLMKitError,
    "Provider configuration error."
);

create_exception!(
    llmkit,
    AuthenticationError,
    LLMKitError,
    "Authentication error (invalid or missing API key)."
);

create_exception!(
    llmkit,
    RateLimitError,
    LLMKitError,
    "Rate limit exceeded. Check retry_after_seconds attribute."
);

create_exception!(
    llmkit,
    InvalidRequestError,
    LLMKitError,
    "Request validation error."
);

create_exception!(
    llmkit,
    ModelNotFoundError,
    LLMKitError,
    "Model not found or not supported."
);

create_exception!(
    llmkit,
    ContentFilteredError,
    LLMKitError,
    "Content moderation triggered."
);

create_exception!(
    llmkit,
    ContextLengthError,
    LLMKitError,
    "Context length exceeded."
);

create_exception!(llmkit, NetworkError, LLMKitError, "Network/HTTP error.");

create_exception!(llmkit, StreamError, LLMKitError, "Streaming error.");

create_exception!(llmkit, TimeoutError, LLMKitError, "Request timed out.");

create_exception!(
    llmkit,
    ServerError,
    LLMKitError,
    "Server error from the provider. Check status attribute."
);

create_exception!(
    llmkit,
    NotSupportedError,
    LLMKitError,
    "Feature not supported by the provider."
);

/// Convert LLMKit Rust errors to Python exceptions.
pub fn convert_error(error: llmkit::error::Error) -> PyErr {
    use llmkit::error::Error;

    match error {
        Error::ProviderNotFound(msg) => ProviderNotFoundError::new_err(msg),
        Error::Configuration(msg) => ConfigurationError::new_err(msg),
        Error::Authentication(msg) => AuthenticationError::new_err(msg),
        Error::RateLimited {
            message,
            retry_after,
        } => {
            let msg = if let Some(duration) = retry_after {
                format!("{} (retry after {:.1}s)", message, duration.as_secs_f64())
            } else {
                message
            };
            RateLimitError::new_err(msg)
        }
        Error::InvalidRequest(msg) => InvalidRequestError::new_err(msg),
        Error::ModelNotFound(msg) => ModelNotFoundError::new_err(msg),
        Error::ContentFiltered(msg) => ContentFilteredError::new_err(msg),
        Error::ContextLengthExceeded(msg) => ContextLengthError::new_err(msg),
        Error::Network(e) => NetworkError::new_err(e.to_string()),
        Error::Json(e) => pyo3::exceptions::PyValueError::new_err(format!("JSON error: {}", e)),
        Error::Stream(msg) => StreamError::new_err(msg),
        Error::Timeout => TimeoutError::new_err("Request timed out"),
        Error::Server { status, message } => {
            ServerError::new_err(format!("Server error ({}): {}", status, message))
        }
        Error::NotSupported(msg) => NotSupportedError::new_err(msg),
        Error::Other(msg) => LLMKitError::new_err(msg),
    }
}
