//! Image generation and manipulation providers.
//!
//! This module contains providers specialized in image generation,
//! editing, and manipulation tasks.

#[cfg(feature = "stability")]
pub mod stability;

#[cfg(feature = "fal")]
pub mod fal;

#[cfg(feature = "recraft")]
pub mod recraft;

#[cfg(feature = "runwayml")]
pub mod runwayml;

// Re-exports

#[cfg(feature = "stability")]
pub use stability::StabilityProvider;

#[cfg(feature = "fal")]
pub use fal::FalProvider;

#[cfg(feature = "recraft")]
pub use recraft::RecraftProvider;

#[cfg(feature = "runwayml")]
pub use runwayml::RunwayMLProvider;
