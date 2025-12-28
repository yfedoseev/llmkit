//! LLMKit client for JavaScript

use std::sync::Arc;

use futures::StreamExt;
use llmkit::LLMKitClient;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::sync::Mutex;

use crate::errors::convert_error;
use crate::types::request::JsCompletionRequest;
use crate::types::response::JsCompletionResponse;
use crate::types::stream::JsStreamChunk;

/// Options for creating an LLMKitClient.
#[napi(object)]
pub struct LLMKitClientOptions {
    /// Anthropic API key
    pub anthropic_api_key: Option<String>,
    /// OpenAI API key
    pub openai_api_key: Option<String>,
    /// Groq API key
    pub groq_api_key: Option<String>,
    /// Mistral API key
    pub mistral_api_key: Option<String>,
    /// Default provider name
    pub default_provider: Option<String>,
}

/// LLMKit client for JavaScript/TypeScript.
///
/// @example
/// ```typescript
/// import { LLMKitClient, Message, CompletionRequest } from 'llmkit'
///
/// // Create client from environment variables
/// const client = LLMKitClient.fromEnv()
///
/// // Make a completion request
/// const response = await client.complete(
///   CompletionRequest.create("claude-sonnet-4-20250514", [Message.user("Hello!")])
/// )
/// console.log(response.textContent())
///
/// // Streaming with callback
/// client.completeStream(request.withStreaming(), (error, chunk, done) => {
///   if (error) throw new Error(error)
///   if (done) return
///   if (chunk?.text) process.stdout.write(chunk.text)
/// })
/// ```
#[napi]
pub struct JsLLMKitClient {
    inner: Arc<LLMKitClient>,
}

#[napi]
impl JsLLMKitClient {
    /// Create a new LLMKit client with explicit API keys.
    #[napi(constructor)]
    pub fn new(options: Option<LLMKitClientOptions>) -> Result<Self> {
        let mut builder = LLMKitClient::builder();

        if let Some(opts) = options {
            // Add providers with explicit keys
            if let Some(key) = opts.anthropic_api_key {
                builder = builder
                    .with_anthropic(key)
                    .map_err(|e| Error::from_reason(e.to_string()))?;
            }
            if let Some(key) = opts.openai_api_key {
                builder = builder
                    .with_openai(key)
                    .map_err(|e| Error::from_reason(e.to_string()))?;
            }
            if let Some(key) = opts.groq_api_key {
                builder = builder
                    .with_groq(key)
                    .map_err(|e| Error::from_reason(e.to_string()))?;
            }
            if let Some(key) = opts.mistral_api_key {
                builder = builder
                    .with_mistral(key)
                    .map_err(|e| Error::from_reason(e.to_string()))?;
            }

            // Set default provider
            if let Some(provider) = opts.default_provider {
                builder = builder.with_default(provider);
            }
        }

        // Enable default retry
        builder = builder.with_default_retry();

        let client = builder
            .build()
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(client),
        })
    }

    /// Create client from environment variables.
    ///
    /// Automatically detects and configures providers from environment variables:
    /// - ANTHROPIC_API_KEY
    /// - OPENAI_API_KEY
    /// - GROQ_API_KEY
    /// - MISTRAL_API_KEY
    /// - etc.
    #[napi(factory)]
    pub fn from_env() -> Result<Self> {
        let client = LLMKitClient::builder()
            .with_anthropic_from_env()
            .with_openai_from_env()
            .with_groq_from_env()
            .with_mistral_from_env()
            .with_default_retry()
            .build()
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(client),
        })
    }

    /// Make a completion request.
    ///
    /// Returns a Promise that resolves to CompletionResponse.
    #[napi]
    pub async fn complete(&self, request: &JsCompletionRequest) -> Result<JsCompletionResponse> {
        let req = request.inner.clone();

        self.inner
            .complete(req)
            .await
            .map(JsCompletionResponse::from)
            .map_err(convert_error)
    }

    /// Make a streaming completion request with callback.
    ///
    /// The callback receives the stream chunk directly, or null on error/done.
    /// Check chunk.isDone to determine when streaming is complete.
    #[napi]
    pub fn complete_stream(
        &self,
        request: &JsCompletionRequest,
        #[napi(ts_arg_type = "(chunk: StreamChunk | null, error: string | null) => void")]
        callback: ThreadsafeFunction<(Option<JsStreamChunk>, Option<String>), ErrorStrategy::Fatal>,
    ) -> Result<()> {
        let inner = self.inner.clone();
        let mut req = request.inner.clone();

        // Ensure streaming is enabled
        if !req.stream {
            req = req.with_streaming();
        }

        // Spawn the streaming task
        tokio::spawn(async move {
            match inner.complete_stream(req).await {
                Ok(stream) => {
                    let stream = Arc::new(Mutex::new(stream));

                    loop {
                        let chunk_result = {
                            let mut guard = stream.lock().await;
                            guard.next().await
                        };

                        match chunk_result {
                            Some(Ok(chunk)) => {
                                let js_chunk = JsStreamChunk::from(chunk);
                                callback.call(
                                    (Some(js_chunk), None),
                                    ThreadsafeFunctionCallMode::NonBlocking,
                                );
                            }
                            Some(Err(e)) => {
                                callback.call(
                                    (None, Some(e.to_string())),
                                    ThreadsafeFunctionCallMode::NonBlocking,
                                );
                                break;
                            }
                            None => {
                                // Stream ended - callback with null chunk to signal done
                                callback
                                    .call((None, None), ThreadsafeFunctionCallMode::NonBlocking);
                                break;
                            }
                        }
                    }
                }
                Err(e) => {
                    callback.call(
                        (None, Some(e.to_string())),
                        ThreadsafeFunctionCallMode::NonBlocking,
                    );
                }
            }
        });

        Ok(())
    }

    /// Make a completion request with a specific provider.
    #[napi]
    pub async fn complete_with_provider(
        &self,
        provider_name: String,
        request: &JsCompletionRequest,
    ) -> Result<JsCompletionResponse> {
        let req = request.inner.clone();

        self.inner
            .complete_with_provider(&provider_name, req)
            .await
            .map(JsCompletionResponse::from)
            .map_err(convert_error)
    }

    /// List all registered providers.
    #[napi]
    pub fn providers(&self) -> Vec<String> {
        self.inner
            .providers()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Get the default provider name.
    #[napi(getter)]
    pub fn default_provider(&self) -> Option<String> {
        self.inner.default_provider().map(|p| p.name().to_string())
    }
}
