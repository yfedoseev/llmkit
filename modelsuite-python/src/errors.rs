//! Python exception hierarchy for ModelSuite errors

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

// Base exception
create_exception!(
    modelsuite,
    ModelSuiteError,
    PyException,
    "Base exception for ModelSuite errors."
);

// Specific exceptions
create_exception!(
    modelsuite,
    ProviderNotFoundError,
    ModelSuiteError,
    "Provider not found or not configured."
);

create_exception!(
    modelsuite,
    ConfigurationError,
    ModelSuiteError,
    "Provider configuration error."
);

create_exception!(
    modelsuite,
    AuthenticationError,
    ModelSuiteError,
    "Authentication error (invalid or missing API key)."
);

create_exception!(
    modelsuite,
    RateLimitError,
    ModelSuiteError,
    "Rate limit exceeded. Check retry_after_seconds attribute."
);

create_exception!(
    modelsuite,
    InvalidRequestError,
    ModelSuiteError,
    "Request validation error."
);

create_exception!(
    modelsuite,
    ModelNotFoundError,
    ModelSuiteError,
    "Model not found or not supported."
);

create_exception!(
    modelsuite,
    ContentFilteredError,
    ModelSuiteError,
    "Content moderation triggered."
);

create_exception!(
    modelsuite,
    ContextLengthError,
    ModelSuiteError,
    "Context length exceeded."
);

create_exception!(
    modelsuite,
    NetworkError,
    ModelSuiteError,
    "Network/HTTP error."
);

create_exception!(modelsuite, StreamError, ModelSuiteError, "Streaming error.");

create_exception!(
    modelsuite,
    TimeoutError,
    ModelSuiteError,
    "Request timed out."
);

create_exception!(
    modelsuite,
    ServerError,
    ModelSuiteError,
    "Server error from the provider. Check status attribute."
);

create_exception!(
    modelsuite,
    NotSupportedError,
    ModelSuiteError,
    "Feature not supported by the provider."
);

/// Convert ModelSuite Rust errors to Python exceptions.
pub fn convert_error(error: modelsuite::error::Error) -> PyErr {
    use modelsuite::error::Error;

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
        Error::Other(msg) => ModelSuiteError::new_err(msg),
    }
}
