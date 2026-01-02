//! LLMKit JavaScript/TypeScript bindings
//!
//! This crate provides Node.js bindings for the LLMKit unified LLM client library.
//!
//! # Example (JavaScript/TypeScript)
//!
//! ```typescript
//! import { LLMKitClient, Message, CompletionRequest } from 'llmkit'
//!
//! const client = LLMKitClient.fromEnv()
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

mod client;
mod errors;
mod models;
mod stream_internal;
mod tools;
mod types;

pub use client::*;
pub use errors::*;
pub use models::*;
pub use tools::*;
pub use types::*;
