//! Error handling for JavaScript bindings
//!
//! Converts ModelSuite errors to JavaScript errors with descriptive messages.

use napi::bindgen_prelude::*;

/// Error codes for ModelSuite errors.
/// These can be used to programmatically identify error types.
pub mod error_codes {
    pub const PROVIDER_NOT_FOUND: &str = "PROVIDER_NOT_FOUND";
    pub const CONFIGURATION: &str = "CONFIGURATION";
    pub const AUTHENTICATION: &str = "AUTHENTICATION";
    pub const RATE_LIMITED: &str = "RATE_LIMITED";
    pub const INVALID_REQUEST: &str = "INVALID_REQUEST";
    pub const MODEL_NOT_FOUND: &str = "MODEL_NOT_FOUND";
    pub const CONTENT_FILTERED: &str = "CONTENT_FILTERED";
    pub const CONTEXT_LENGTH: &str = "CONTEXT_LENGTH";
    pub const NETWORK: &str = "NETWORK";
    pub const JSON: &str = "JSON";
    pub const STREAM: &str = "STREAM";
    pub const TIMEOUT: &str = "TIMEOUT";
    pub const SERVER: &str = "SERVER";
    pub const NOT_SUPPORTED: &str = "NOT_SUPPORTED";
    pub const UNKNOWN: &str = "UNKNOWN";
}

/// Convert ModelSuite Rust errors to JavaScript errors.
pub fn convert_error(error: modelsuite::error::Error) -> Error {
    use modelsuite::error::Error as ModelSuiteError;

    match error {
        ModelSuiteError::ProviderNotFound(msg) => Error::new(
            Status::GenericFailure,
            format!("[{}] {}", error_codes::PROVIDER_NOT_FOUND, msg),
        ),
        ModelSuiteError::Configuration(msg) => Error::new(
            Status::GenericFailure,
            format!("[{}] {}", error_codes::CONFIGURATION, msg),
        ),
        ModelSuiteError::Authentication(msg) => Error::new(
            Status::GenericFailure,
            format!("[{}] {}", error_codes::AUTHENTICATION, msg),
        ),
        ModelSuiteError::RateLimited {
            message,
            retry_after,
        } => {
            let msg = if let Some(duration) = retry_after {
                format!(
                    "[{}] {} (retry after {:.1}s)",
                    error_codes::RATE_LIMITED,
                    message,
                    duration.as_secs_f64()
                )
            } else {
                format!("[{}] {}", error_codes::RATE_LIMITED, message)
            };
            Error::new(Status::GenericFailure, msg)
        }
        ModelSuiteError::InvalidRequest(msg) => Error::new(
            Status::GenericFailure,
            format!("[{}] {}", error_codes::INVALID_REQUEST, msg),
        ),
        ModelSuiteError::ModelNotFound(msg) => Error::new(
            Status::GenericFailure,
            format!("[{}] {}", error_codes::MODEL_NOT_FOUND, msg),
        ),
        ModelSuiteError::ContentFiltered(msg) => Error::new(
            Status::GenericFailure,
            format!("[{}] {}", error_codes::CONTENT_FILTERED, msg),
        ),
        ModelSuiteError::ContextLengthExceeded(msg) => Error::new(
            Status::GenericFailure,
            format!("[{}] {}", error_codes::CONTEXT_LENGTH, msg),
        ),
        ModelSuiteError::Network(e) => Error::new(
            Status::GenericFailure,
            format!("[{}] {}", error_codes::NETWORK, e),
        ),
        ModelSuiteError::Json(e) => Error::new(
            Status::GenericFailure,
            format!("[{}] {}", error_codes::JSON, e),
        ),
        ModelSuiteError::Stream(msg) => Error::new(
            Status::GenericFailure,
            format!("[{}] {}", error_codes::STREAM, msg),
        ),
        ModelSuiteError::Timeout => Error::new(
            Status::GenericFailure,
            format!("[{}] Request timed out", error_codes::TIMEOUT),
        ),
        ModelSuiteError::Server { status, message } => Error::new(
            Status::GenericFailure,
            format!(
                "[{}] Server error ({}): {}",
                error_codes::SERVER,
                status,
                message
            ),
        ),
        ModelSuiteError::NotSupported(msg) => Error::new(
            Status::GenericFailure,
            format!("[{}] {}", error_codes::NOT_SUPPORTED, msg),
        ),
        ModelSuiteError::Other(msg) => Error::new(
            Status::GenericFailure,
            format!("[{}] {}", error_codes::UNKNOWN, msg),
        ),
    }
}

/// Helper to check if an error message contains a specific error code.
pub fn is_error_code(error_message: &str, code: &str) -> bool {
    error_message.starts_with(&format!("[{}]", code))
}
