//! ModelSuite JavaScript/TypeScript bindings
//!
//! This crate provides Node.js bindings for the ModelSuite unified LLM client library.
//!
//! # Example (JavaScript/TypeScript)
//!
//! ```typescript
//! import { ModelSuiteClient, Message, CompletionRequest } from 'modelsuite'
//!
//! const client = ModelSuiteClient.fromEnv()
//!
//! const response = await client.complete(
//!   new CompletionRequest({
//!     model: 'claude-sonnet-4-20250514',
//!     messages: [Message.user('Hello!')],
//!     maxTokens: 1024
//!   })
//! )
//!
//! console.log(response.textContent())
//! ```

#![deny(clippy::all)]

mod audio;
mod client;
mod errors;
mod image;
mod models;
mod retry;
mod specialized;
mod stream_internal;
mod tools;
mod types;
mod video;

pub use audio::*;
pub use client::*;
pub use errors::*;
pub use image::*;
pub use models::*;
pub use retry::*;
pub use specialized::*;
pub use tools::*;
pub use types::*;
pub use video::*;
