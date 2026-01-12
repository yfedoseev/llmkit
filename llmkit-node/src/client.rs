//! LLMKit client for JavaScript

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use llmkit::providers::chat::azure::AzureConfig;
use llmkit::LLMKitClient;
use napi::bindgen_prelude::*;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};
use napi_derive::napi;
use tokio::sync::Mutex;

use crate::audio::{
    JsSynthesisRequest, JsSynthesizeResponse, JsTranscribeResponse, JsTranscriptionRequest, JsWord,
};
use crate::errors::convert_error;
use crate::image::{
    JsGeneratedImage, JsImageGenerationRequest, JsImageGenerationResponse, JsImageQuality,
    JsImageSize,
};
use crate::retry::JsRetryConfig;
use crate::specialized::{
    JsClassificationRequest, JsClassificationResponse, JsClassificationResult, JsModerationRequest,
    JsModerationResponse, JsModerationScores, JsRankedDocument, JsRankingRequest,
    JsRankingResponse, JsRerankedResult, JsRerankingRequest, JsRerankingResponse,
};
use crate::stream_internal::StreamHandler;
use crate::types::embedding::{JsEmbeddingRequest, JsEmbeddingResponse};
use crate::types::request::{
    JsBatchJob, JsBatchRequest, JsBatchResult, JsCompletionRequest, JsTokenCountRequest,
    JsTokenCountResult,
};
use crate::types::response::JsCompletionResponse;
use crate::types::stream::{JsAsyncStreamIterator, JsStreamChunk};
use crate::video::{JsVideoGenerationRequest, JsVideoGenerationResponse};

/// Configuration for a single provider.
#[napi(object)]
#[derive(Clone)]
pub struct ProviderConfig {
    /// API key for the provider
    pub api_key: Option<String>,
    /// Secret key for providers that require it (e.g., Baidu)
    pub secret_key: Option<String>,
    /// Custom base URL (optional)
    pub base_url: Option<String>,
    /// Azure OpenAI endpoint
    pub endpoint: Option<String>,
    /// Azure OpenAI deployment name
    pub deployment: Option<String>,
    /// AWS region for Bedrock, or location for Vertex
    pub region: Option<String>,
    /// Google Cloud project ID for Vertex
    pub project: Option<String>,
    /// Location for Vertex AI
    pub location: Option<String>,
    /// Cloudflare account ID
    pub account_id: Option<String>,
    /// Cloudflare API token
    pub api_token: Option<String>,
    /// Vertex AI access token (deprecated: use ADC via GOOGLE_APPLICATION_CREDENTIALS instead)
    #[allow(dead_code)]
    pub access_token: Option<String>,
    /// Databricks host URL
    pub host: Option<String>,
    /// Databricks token
    pub token: Option<String>,
    /// RunPod endpoint ID
    pub endpoint_id: Option<String>,
    /// Model ID (for openai_compatible)
    pub model_id: Option<String>,
}

/// Options for creating an LLMKitClient.
#[napi(object)]
pub struct LLMKitClientOptions {
    /// Provider configurations (key is provider name, value is config)
    /// Supported providers: anthropic, openai, azure, bedrock, vertex, google,
    /// groq, mistral, cohere, ai21, deepseek, together, fireworks, perplexity,
    /// cerebras, sambanova, openrouter, ollama, huggingface, replicate,
    /// cloudflare, watsonx, databricks, baseten, runpod, anyscale, deepinfra,
    /// novita, hyperbolic, lm_studio, vllm, tgi, llamafile
    pub providers: Option<HashMap<String, ProviderConfig>>,
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
/// // Create client with explicit provider config
/// const client = new LLMKitClient({
///   providers: {
///     anthropic: { apiKey: "sk-..." },
///     openai: { apiKey: "sk-..." },
///     azure: { apiKey: "...", endpoint: "https://...", deployment: "gpt-4" },
///     bedrock: { region: "us-east-1" },
///   }
/// })
///
/// // Make a completion request
/// const response = await client.complete(
///   CompletionRequest.create("claude-sonnet-4-20250514", [Message.user("Hello!")])
/// )
/// console.log(response.textContent())
///
/// // Streaming with callback
/// client.completeStream(request.withStreaming(), (chunk, error) => {
///   if (error) throw new Error(error)
///   if (!chunk) return // done
///   if (chunk.text) process.stdout.write(chunk.text)
/// })
/// ```
#[napi]
pub struct JsLLMKitClient {
    inner: Arc<LLMKitClient>,
}

#[napi]
impl JsLLMKitClient {
    /// Create a new LLMKit client with provider configurations.
    ///
    /// @param options - Configuration options including providers dict
    /// @param retryConfig - Optional retry configuration. If not provided, uses default
    ///   (10 retries with exponential backoff). Pass `RetryConfig.none()` to disable retry.
    ///
    /// @example
    /// ```typescript
    /// const client = new LLMKitClient({
    ///   providers: {
    ///     anthropic: { apiKey: "sk-..." },
    ///     azure: { apiKey: "...", endpoint: "https://...", deployment: "gpt-4" },
    ///   }
    /// })
    ///
    /// // With custom retry
    /// const client = new LLMKitClient({}, RetryConfig.conservative())
    ///
    /// // Disable retry
    /// const client = new LLMKitClient({}, RetryConfig.none())
    /// ```
    #[napi(constructor)]
    pub fn new(
        options: Option<LLMKitClientOptions>,
        retry_config: Option<&JsRetryConfig>,
    ) -> Result<Self> {
        // Install ring as the default crypto provider for rustls
        #[cfg(feature = "vertex")]
        let _ = rustls::crypto::ring::default_provider().install_default();

        // Create a temporary runtime for initialization (for Bedrock which is async)
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::from_reason(e.to_string()))?;

        let mut builder = LLMKitClient::builder();

        if let Some(opts) = options {
            // Add providers from config
            if let Some(providers) = opts.providers {
                for (name, config) in providers {
                    builder = Self::add_provider_to_builder(builder, &name, config, &runtime)?;
                }
            }

            // Set default provider
            if let Some(provider) = opts.default_provider {
                builder = builder.with_default(provider);
            }
        }

        // Apply retry config
        builder = Self::apply_retry_config_ref(builder, retry_config);

        // Build client (async to initialize Vertex/Bedrock credentials)
        let client = runtime
            .block_on(builder.build())
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(Self {
            inner: Arc::new(client),
        })
    }

    /// Create client from environment variables.
    ///
    /// Automatically detects and configures all available providers from environment variables.
    ///
    /// @param retryConfig - Optional retry configuration. If not provided, uses default
    ///   (10 retries with exponential backoff). Pass `RetryConfig.none()` to disable retry.
    ///
    /// Supported environment variables:
    /// - ANTHROPIC_API_KEY: Anthropic (Claude)
    /// - OPENAI_API_KEY: OpenAI (GPT)
    /// - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT: Azure OpenAI
    /// - OPENROUTER_API_KEY: OpenRouter
    /// - AWS_REGION or AWS_DEFAULT_REGION: AWS Bedrock (uses default credential chain)
    /// - GOOGLE_API_KEY: Google AI (Gemini)
    /// - GOOGLE_CLOUD_PROJECT, VERTEX_LOCATION, VERTEX_ACCESS_TOKEN: Google Vertex AI
    /// - GROQ_API_KEY: Groq
    /// - MISTRAL_API_KEY: Mistral
    /// - COHERE_API_KEY or CO_API_KEY: Cohere
    /// - AI21_API_KEY: AI21 Labs
    /// - DEEPSEEK_API_KEY: DeepSeek
    /// - XAI_API_KEY: xAI (Grok)
    /// - TOGETHER_API_KEY: Together AI
    /// - FIREWORKS_API_KEY: Fireworks AI
    /// - PERPLEXITY_API_KEY: Perplexity
    /// - CEREBRAS_API_KEY: Cerebras
    /// - SAMBANOVA_API_KEY: SambaNova
    /// - NVIDIA_NIM_API_KEY: NVIDIA NIM
    /// - DATAROBOT_API_KEY: DataRobot
    /// - HUGGINGFACE_API_KEY or HF_TOKEN: HuggingFace
    /// - REPLICATE_API_TOKEN: Replicate
    /// - CLOUDFLARE_API_TOKEN, CLOUDFLARE_ACCOUNT_ID: Cloudflare Workers AI
    /// - WATSONX_API_KEY, WATSONX_PROJECT_ID: IBM watsonx.ai
    /// - DATABRICKS_TOKEN, DATABRICKS_HOST: Databricks
    /// - BASETEN_API_KEY: Baseten
    /// - RUNPOD_API_KEY, RUNPOD_ENDPOINT_ID: RunPod
    /// - ANYSCALE_API_KEY: Anyscale
    /// - DEEPINFRA_API_KEY: DeepInfra
    /// - NOVITA_API_KEY: Novita AI
    /// - HYPERBOLIC_API_KEY: Hyperbolic
    /// - LAMBDA_API_KEY: Lambda
    /// - FRIENDLI_API_KEY: Friendli
    /// - BAIDU_API_KEY: Baidu (ERNIE)
    /// - ALIBABA_API_KEY: Alibaba (Qwen)
    /// - VOLCENGINE_API_KEY: Volcengine
    /// - MARITACA_API_KEY: Maritaca
    /// - LIGHTON_API_KEY: LightOn
    /// - VOYAGE_API_KEY: Voyage AI
    /// - JINA_API_KEY: Jina AI
    /// - STABILITY_API_KEY: Stability AI
    /// - OLLAMA_BASE_URL: Ollama (local, defaults to http://localhost:11434)
    ///
    /// @example
    /// ```typescript
    /// // Default retry
    /// const client = LLMKitClient.fromEnv()
    ///
    /// // Custom retry
    /// const client = LLMKitClient.fromEnv(RetryConfig.conservative())
    ///
    /// // Disable retry
    /// const client = LLMKitClient.fromEnv(RetryConfig.none())
    /// ```
    #[napi(factory)]
    pub fn from_env(retry_config: Option<&JsRetryConfig>) -> Result<Self> {
        // Install ring as the default crypto provider for rustls
        #[cfg(feature = "vertex")]
        let _ = rustls::crypto::ring::default_provider().install_default();

        // Create a temporary runtime for initialization (for Bedrock which is async)
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::from_reason(e.to_string()))?;

        // Build client with all providers from environment
        let builder = LLMKitClient::builder()
            // Core providers
            .with_anthropic_from_env()
            .with_openai_from_env()
            .with_azure_from_env()
            .with_openrouter_from_env()
            // Google/Cloud providers (Vertex and Bedrock initialized in async build())
            .with_google_from_env()
            .with_vertex_from_env()
            .with_bedrock_from_env()
            // Fast inference providers
            .with_groq_from_env()
            .with_mistral_from_env()
            .with_cerebras_from_env()
            .with_sambanova_from_env()
            .with_fireworks_from_env()
            .with_deepseek_from_env()
            .with_xai_from_env()
            // Enterprise providers
            .with_cohere_from_env()
            .with_ai21_from_env()
            .with_nvidia_nim_from_env()
            .with_datarobot_from_env()
            // OpenAI-compatible hosted providers
            .with_together_from_env()
            .with_perplexity_from_env()
            .with_anyscale_from_env()
            .with_deepinfra_from_env()
            .with_novita_from_env()
            .with_hyperbolic_from_env()
            .with_lambda_from_env()
            .with_friendli_from_env()
            // Inference platforms
            .with_huggingface_from_env()
            .with_replicate_from_env()
            .with_baseten_from_env()
            .with_runpod_from_env()
            // Cloud providers
            .with_cloudflare_from_env()
            .with_watsonx_from_env()
            .with_databricks_from_env()
            // Asian providers
            .with_baidu_from_env()
            .with_alibaba_from_env()
            .with_volcengine_from_env()
            // Regional providers
            .with_maritaca_from_env()
            .with_lighton_from_env()
            // Embedding/multimodal providers
            .with_voyage_from_env()
            .with_jina_from_env()
            .with_stability_from_env();

        // Apply retry config
        let builder = Self::apply_retry_config_ref(builder, retry_config);

        // Build client (async to initialize Vertex/Bedrock credentials)
        let client = runtime
            .block_on(builder.build())
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

    /// Make a streaming completion request with async iterator.
    ///
    /// Returns an async iterator that you can call `next()` on to get stream chunks:
    ///
    /// @example
    /// ```typescript
    /// const stream = await client.stream(request);
    /// let chunk;
    /// while ((chunk = await stream.next()) !== null) {
    ///   if (chunk.text) process.stdout.write(chunk.text);
    ///   if (chunk.isDone) break;
    /// }
    /// ```
    #[napi]
    pub async fn stream(&self, request: &JsCompletionRequest) -> Result<JsAsyncStreamIterator> {
        let mut req = request.inner.clone();

        // Ensure streaming is enabled
        if !req.stream {
            req = req.with_streaming();
        }

        let stream = self
            .inner
            .complete_stream(req)
            .await
            .map_err(convert_error)?;

        let handler = StreamHandler::new(stream);
        Ok(JsAsyncStreamIterator::from_handler(handler))
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

    /// Count tokens for a request.
    ///
    /// This allows estimation of token counts before making a completion request,
    /// useful for cost estimation and context window management.
    ///
    /// Note: Not all providers support token counting. Currently only Anthropic
    /// provides native token counting support.
    #[napi]
    pub async fn count_tokens(&self, request: &JsTokenCountRequest) -> Result<JsTokenCountResult> {
        let req = request.inner.clone();

        self.inner
            .count_tokens(req)
            .await
            .map(JsTokenCountResult::from)
            .map_err(convert_error)
    }

    /// Count tokens for a request with a specific provider.
    ///
    /// @param providerName - Name of the provider to use
    /// @param request - TokenCountRequest with model, messages, optional system and tools
    /// @returns TokenCountResult containing input_tokens count
    #[napi]
    pub async fn count_tokens_with_provider(
        &self,
        provider_name: String,
        request: &JsTokenCountRequest,
    ) -> Result<JsTokenCountResult> {
        let req = request.inner.clone();

        self.inner
            .count_tokens_with_provider(&provider_name, req)
            .await
            .map(JsTokenCountResult::from)
            .map_err(convert_error)
    }

    // ==================== Batch Processing ====================

    /// Create a batch processing job.
    ///
    /// Submits multiple completion requests to be processed asynchronously.
    /// Returns a BatchJob that can be used to track progress and retrieve results.
    ///
    /// @example
    /// ```typescript
    /// const requests = [
    ///   BatchRequest.create("req-1", CompletionRequest.create("claude-sonnet-4-20250514", [Message.user("Hello")])),
    ///   BatchRequest.create("req-2", CompletionRequest.create("claude-sonnet-4-20250514", [Message.user("World")])),
    /// ]
    /// const batchJob = await client.createBatch(requests)
    /// console.log(`Batch created: ${batchJob.id}`)
    /// ```
    #[napi]
    pub async fn create_batch(&self, requests: Vec<&JsBatchRequest>) -> Result<JsBatchJob> {
        let batch_requests: Vec<_> = requests.iter().map(|r| r.inner.clone()).collect();

        self.inner
            .create_batch(batch_requests)
            .await
            .map(JsBatchJob::from)
            .map_err(convert_error)
    }

    /// Get the status of a batch job.
    ///
    /// @param providerName - The provider that created the batch
    /// @param batchId - The batch ID
    ///
    /// @example
    /// ```typescript
    /// const job = await client.getBatch("anthropic", batchJob.id)
    /// console.log(`Status: ${job.status}`)
    /// ```
    #[napi]
    pub async fn get_batch(&self, provider_name: String, batch_id: String) -> Result<JsBatchJob> {
        self.inner
            .get_batch(&provider_name, &batch_id)
            .await
            .map(JsBatchJob::from)
            .map_err(convert_error)
    }

    /// Get the results of a completed batch.
    ///
    /// @param providerName - The provider that created the batch
    /// @param batchId - The batch ID
    ///
    /// @example
    /// ```typescript
    /// const results = await client.getBatchResults("anthropic", batchJob.id)
    /// for (const result of results) {
    ///   if (result.isSuccess()) {
    ///     console.log(`${result.customId}: ${result.response?.textContent()}`)
    ///   } else {
    ///     console.error(`${result.customId}: ${result.error?.message}`)
    ///   }
    /// }
    /// ```
    #[napi]
    pub async fn get_batch_results(
        &self,
        provider_name: String,
        batch_id: String,
    ) -> Result<Vec<JsBatchResult>> {
        self.inner
            .get_batch_results(&provider_name, &batch_id)
            .await
            .map(|results| results.into_iter().map(JsBatchResult::from).collect())
            .map_err(convert_error)
    }

    /// Cancel a batch job.
    ///
    /// @param providerName - The provider that created the batch
    /// @param batchId - The batch ID
    ///
    /// @example
    /// ```typescript
    /// const job = await client.cancelBatch("anthropic", batchJob.id)
    /// console.log(`Batch cancelled: ${job.status}`)
    /// ```
    #[napi]
    pub async fn cancel_batch(
        &self,
        provider_name: String,
        batch_id: String,
    ) -> Result<JsBatchJob> {
        self.inner
            .cancel_batch(&provider_name, &batch_id)
            .await
            .map(JsBatchJob::from)
            .map_err(convert_error)
    }

    /// List batch jobs for a provider.
    ///
    /// @param providerName - The provider to list batches for
    /// @param limit - Maximum number of batches to return (optional)
    ///
    /// @example
    /// ```typescript
    /// const batches = await client.listBatches("anthropic", 10)
    /// for (const batch of batches) {
    ///   console.log(`${batch.id}: ${batch.status}`)
    /// }
    /// ```
    #[napi]
    pub async fn list_batches(
        &self,
        provider_name: String,
        limit: Option<u32>,
    ) -> Result<Vec<JsBatchJob>> {
        self.inner
            .list_batches(&provider_name, limit)
            .await
            .map(|jobs| jobs.into_iter().map(JsBatchJob::from).collect())
            .map_err(convert_error)
    }

    // ==================== Embeddings ====================

    /// Generate embeddings for text.
    ///
    /// Creates vector representations of text that can be used for semantic search,
    /// clustering, classification, and other NLP tasks.
    ///
    /// Note: Not all providers support embeddings. Currently OpenAI and Cohere
    /// support this feature.
    ///
    /// @example
    /// ```typescript
    /// const response = await client.embed(
    ///   new EmbeddingRequest("text-embedding-3-small", "Hello, world!")
    /// )
    /// console.log(`Dimensions: ${response.dimensionCount}`)
    /// console.log(`Values: ${response.values()?.slice(0, 5)}...`)
    /// ```
    #[napi]
    pub async fn embed(&self, request: &JsEmbeddingRequest) -> Result<JsEmbeddingResponse> {
        let req = request.inner.clone();

        self.inner
            .embed(req)
            .await
            .map(JsEmbeddingResponse::from)
            .map_err(convert_error)
    }

    /// Generate embeddings with a specific provider.
    ///
    /// @param providerName - Name of the embedding provider (e.g., "openai", "cohere")
    /// @param request - EmbeddingRequest with model and text(s) to embed
    ///
    /// @example
    /// ```typescript
    /// const response = await client.embedWithProvider(
    ///   "openai",
    ///   new EmbeddingRequest("text-embedding-3-small", "Hello")
    /// )
    /// ```
    #[napi]
    pub async fn embed_with_provider(
        &self,
        provider_name: String,
        request: &JsEmbeddingRequest,
    ) -> Result<JsEmbeddingResponse> {
        let req = request.inner.clone();

        self.inner
            .embed_with_provider(&provider_name, req)
            .await
            .map(JsEmbeddingResponse::from)
            .map_err(convert_error)
    }

    /// List all registered embedding providers.
    ///
    /// @returns Names of providers that support embeddings
    #[napi]
    pub fn embedding_providers(&self) -> Vec<String> {
        self.inner
            .embedding_providers()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Check if a provider supports embeddings.
    ///
    /// @param providerName - Name of the provider to check
    /// @returns True if the provider supports embeddings
    #[napi]
    pub fn supports_embeddings(&self, provider_name: String) -> bool {
        self.inner.supports_embeddings(&provider_name)
    }

    /// List all registered speech synthesis providers.
    ///
    /// @returns Names of providers that support text-to-speech
    #[napi]
    pub fn speech_providers(&self) -> Vec<String> {
        self.inner
            .speech_providers()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// List all registered transcription providers.
    ///
    /// @returns Names of providers that support speech-to-text
    #[napi]
    pub fn transcription_providers(&self) -> Vec<String> {
        self.inner
            .transcription_providers()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// List all registered image generation providers.
    ///
    /// @returns Names of providers that support image generation
    #[napi]
    pub fn image_providers(&self) -> Vec<String> {
        self.inner
            .image_providers()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// List all registered video generation providers.
    ///
    /// @returns Names of providers that support video generation
    #[napi]
    pub fn video_providers(&self) -> Vec<String> {
        self.inner
            .video_providers()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// List all registered ranking/reranking providers.
    ///
    /// @returns Names of providers that support document ranking
    #[napi]
    pub fn ranking_providers(&self) -> Vec<String> {
        self.inner
            .ranking_providers()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// List all registered moderation providers.
    ///
    /// @returns Names of providers that support content moderation
    #[napi]
    pub fn moderation_providers(&self) -> Vec<String> {
        self.inner
            .moderation_providers()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// List all registered classification providers.
    ///
    /// @returns Names of providers that support text classification
    #[napi]
    pub fn classification_providers(&self) -> Vec<String> {
        self.inner
            .classification_providers()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    // ==================== Audio APIs ====================

    /// Transcribe audio to text.
    ///
    /// Converts speech audio to text using various providers (Deepgram, AssemblyAI).
    ///
    /// @param request - The transcription request with audio bytes and options
    /// @returns The transcribed text with word-level details
    ///
    /// @example
    /// ```typescript
    /// import fs from 'fs'
    /// import { LLMKitClient, TranscriptionRequest } from 'llmkit'
    ///
    /// const client = LLMKitClient.fromEnv()
    /// const audioBytes = fs.readFileSync('speech.wav')
    ///
    /// const request = new TranscriptionRequest(audioBytes)
    /// request.with_model('nova-3')
    ///
    /// const response = await client.transcribeAudio(request)
    /// console.log(response.transcript)
    /// ```
    #[napi]
    pub async fn transcribe_audio(
        &self,
        request: &JsTranscriptionRequest,
    ) -> Result<JsTranscribeResponse> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "deepgram/nova-2".to_string());
        let audio_input =
            llmkit::AudioInput::bytes(request.audio_bytes.clone(), "audio.mp3", "audio/mpeg");
        let core_request = llmkit::TranscriptionRequest::new(model, audio_input);

        let response = self
            .inner
            .transcribe(core_request)
            .await
            .map_err(convert_error)?;

        let words = response
            .words
            .unwrap_or_default()
            .into_iter()
            .map(|w| JsWord {
                word: w.word,
                start: w.start as f64,
                end: w.end as f64,
                confidence: 1.0,
                speaker: None,
            })
            .collect();

        Ok(JsTranscribeResponse {
            transcript: response.text,
            confidence: None,
            words,
            duration: response.duration.map(|d| d as f64),
            metadata: None,
        })
    }

    /// Transcribe audio to text with a specific provider.
    ///
    /// @param providerName - Name of the provider to use
    /// @param request - TranscriptionRequest with audio data
    /// @returns TranscribeResponse with transcript text
    #[napi]
    pub async fn transcribe_audio_with_provider(
        &self,
        provider_name: String,
        request: &JsTranscriptionRequest,
    ) -> Result<JsTranscribeResponse> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "deepgram/nova-2".to_string());
        let audio_input =
            llmkit::AudioInput::bytes(request.audio_bytes.clone(), "audio.mp3", "audio/mpeg");
        let core_request = llmkit::TranscriptionRequest::new(model, audio_input);

        let response = self
            .inner
            .transcribe_with_provider(&provider_name, core_request)
            .await
            .map_err(convert_error)?;

        let words = response
            .words
            .unwrap_or_default()
            .into_iter()
            .map(|w| JsWord {
                word: w.word,
                start: w.start as f64,
                end: w.end as f64,
                confidence: 1.0,
                speaker: None,
            })
            .collect();

        Ok(JsTranscribeResponse {
            transcript: response.text,
            confidence: None,
            words,
            duration: response.duration.map(|d| d as f64),
            metadata: None,
        })
    }

    /// Synthesize text to speech.
    ///
    /// Converts text to speech audio using various providers (ElevenLabs, AssemblyAI).
    ///
    /// @param request - The synthesis request with text and voice options
    /// @returns The synthesized audio as bytes
    ///
    /// @example
    /// ```typescript
    /// import fs from 'fs'
    /// import { LLMKitClient, SynthesisRequest } from 'llmkit'
    ///
    /// const client = LLMKitClient.fromEnv()
    ///
    /// const request = new SynthesisRequest('Hello, world!')
    /// request.with_voice('pNInY14gQrG92XwBIHVr')
    ///
    /// const response = await client.synthesizeSpeech(request)
    /// fs.writeFileSync('speech.mp3', Buffer.from(response.audioBytes))
    /// ```
    #[napi]
    pub async fn synthesize_speech(
        &self,
        request: &JsSynthesisRequest,
    ) -> Result<JsSynthesizeResponse> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "elevenlabs/eleven_monolingual_v1".to_string());
        let voice = request
            .voice_id
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let core_request = llmkit::SpeechRequest::new(model, request.text.clone(), voice);

        let response = self
            .inner
            .speech(core_request)
            .await
            .map_err(convert_error)?;

        let format = match response.format {
            llmkit::AudioFormat::Mp3 => "mp3",
            llmkit::AudioFormat::Opus => "opus",
            llmkit::AudioFormat::Aac => "aac",
            llmkit::AudioFormat::Flac => "flac",
            llmkit::AudioFormat::Wav => "wav",
            llmkit::AudioFormat::Pcm => "pcm",
        };

        Ok(JsSynthesizeResponse {
            audio_bytes: response.audio,
            format: format.to_string(),
            duration: response.duration_seconds.map(|d| d as f64),
        })
    }

    /// Synthesize text to speech with a specific provider.
    ///
    /// @param providerName - Name of the provider to use
    /// @param request - SynthesisRequest with text and voice options
    /// @returns SynthesizeResponse with audio data
    #[napi]
    pub async fn synthesize_speech_with_provider(
        &self,
        provider_name: String,
        request: &JsSynthesisRequest,
    ) -> Result<JsSynthesizeResponse> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "elevenlabs/eleven_monolingual_v1".to_string());
        let voice = request
            .voice_id
            .clone()
            .unwrap_or_else(|| "default".to_string());
        let core_request = llmkit::SpeechRequest::new(model, request.text.clone(), voice);

        let response = self
            .inner
            .speech_with_provider(&provider_name, core_request)
            .await
            .map_err(convert_error)?;

        let format = match response.format {
            llmkit::AudioFormat::Mp3 => "mp3",
            llmkit::AudioFormat::Opus => "opus",
            llmkit::AudioFormat::Aac => "aac",
            llmkit::AudioFormat::Flac => "flac",
            llmkit::AudioFormat::Wav => "wav",
            llmkit::AudioFormat::Pcm => "pcm",
        };

        Ok(JsSynthesizeResponse {
            audio_bytes: response.audio,
            format: format.to_string(),
            duration: response.duration_seconds.map(|d| d as f64),
        })
    }

    // ==================== Video APIs ====================

    /// Generate video from a text prompt.
    ///
    /// Generates video content using various providers (Runware, DiffusionRouter).
    ///
    /// @param request - The video generation request with prompt and options
    /// @returns The generated video or task information
    ///
    /// @example
    /// ```typescript
    /// import { LLMKitClient, VideoGenerationRequest } from 'llmkit'
    ///
    /// const client = LLMKitClient.fromEnv()
    ///
    /// const request = new VideoGenerationRequest('A cat chasing a red ball')
    /// request.with_model('runway-gen-4.5')
    /// request.with_duration(10)
    ///
    /// const response = await client.generateVideo(request)
    /// console.log(`Video task ID: ${response.taskId}`)
    /// ```
    #[napi]
    pub async fn generate_video(
        &self,
        request: &JsVideoGenerationRequest,
    ) -> Result<JsVideoGenerationResponse> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "runwayml/gen-3".to_string());
        let mut core_request = llmkit::VideoGenerationRequest::new(model, request.prompt.clone());

        if let Some(duration) = request.duration {
            core_request = core_request.with_duration(duration);
        }

        let response = self
            .inner
            .generate_video(core_request)
            .await
            .map_err(convert_error)?;

        let (status, video_url, duration) = match response.status {
            llmkit::VideoJobStatus::Queued => ("queued".to_string(), None, None),
            llmkit::VideoJobStatus::Processing { progress, .. } => (
                format!("processing ({}%)", progress.unwrap_or(0)),
                None,
                None,
            ),
            llmkit::VideoJobStatus::Completed {
                video_url,
                duration_seconds,
                ..
            } => (
                "completed".to_string(),
                Some(video_url),
                duration_seconds.map(|d| d as f64),
            ),
            llmkit::VideoJobStatus::Failed { error, .. } => {
                (format!("failed: {}", error), None, None)
            }
            llmkit::VideoJobStatus::Cancelled => ("cancelled".to_string(), None, None),
        };

        Ok(JsVideoGenerationResponse {
            video_bytes: None,
            video_url,
            format: "mp4".to_string(),
            duration,
            width: None,
            height: None,
            task_id: Some(response.job_id),
            status: Some(status),
        })
    }

    /// Generate video from a prompt with a specific provider.
    ///
    /// @param providerName - Name of the provider to use
    /// @param request - VideoGenerationRequest with prompt and parameters
    /// @returns VideoGenerationResponse with video URL or job ID
    #[napi]
    pub async fn generate_video_with_provider(
        &self,
        provider_name: String,
        request: &JsVideoGenerationRequest,
    ) -> Result<JsVideoGenerationResponse> {
        let model = request
            .model
            .clone()
            .unwrap_or_else(|| "runwayml/gen-3".to_string());
        let mut core_request = llmkit::VideoGenerationRequest::new(model, request.prompt.clone());

        if let Some(duration) = request.duration {
            core_request = core_request.with_duration(duration);
        }

        let response = self
            .inner
            .generate_video_with_provider(&provider_name, core_request)
            .await
            .map_err(convert_error)?;

        let (status, video_url, duration) = match response.status {
            llmkit::VideoJobStatus::Queued => ("queued".to_string(), None, None),
            llmkit::VideoJobStatus::Processing { progress, .. } => (
                format!("processing ({}%)", progress.unwrap_or(0)),
                None,
                None,
            ),
            llmkit::VideoJobStatus::Completed {
                video_url,
                duration_seconds,
                ..
            } => (
                "completed".to_string(),
                Some(video_url),
                duration_seconds.map(|d| d as f64),
            ),
            llmkit::VideoJobStatus::Failed { error, .. } => {
                (format!("failed: {}", error), None, None)
            }
            llmkit::VideoJobStatus::Cancelled => ("cancelled".to_string(), None, None),
        };

        Ok(JsVideoGenerationResponse {
            video_bytes: None,
            video_url,
            format: "mp4".to_string(),
            duration,
            width: None,
            height: None,
            task_id: Some(response.job_id),
            status: Some(status),
        })
    }

    /// Get the status of a video generation job.
    ///
    /// @param providerName - Name of the video provider
    /// @param jobId - The job ID returned from generateVideo
    /// @returns VideoGenerationResponse with current status
    #[napi]
    pub async fn get_video_status(
        &self,
        provider_name: String,
        job_id: String,
    ) -> Result<JsVideoGenerationResponse> {
        let video_status = self
            .inner
            .get_video_status(&provider_name, &job_id)
            .await
            .map_err(convert_error)?;

        let (status, video_url, duration) = match video_status {
            llmkit::VideoJobStatus::Queued => ("queued".to_string(), None, None),
            llmkit::VideoJobStatus::Processing { progress, .. } => (
                format!("processing ({}%)", progress.unwrap_or(0)),
                None,
                None,
            ),
            llmkit::VideoJobStatus::Completed {
                video_url,
                duration_seconds,
                ..
            } => (
                "completed".to_string(),
                Some(video_url),
                duration_seconds.map(|d| d as f64),
            ),
            llmkit::VideoJobStatus::Failed { error, .. } => {
                (format!("failed: {}", error), None, None)
            }
            llmkit::VideoJobStatus::Cancelled => ("cancelled".to_string(), None, None),
        };

        Ok(JsVideoGenerationResponse {
            video_bytes: None,
            video_url,
            format: "mp4".to_string(),
            duration,
            width: None,
            height: None,
            task_id: Some(job_id),
            status: Some(status),
        })
    }

    /// Generate images from a text prompt.
    ///
    /// # Arguments
    ///
    /// * `request` - `ImageGenerationRequest` with prompt, model, and optional parameters
    ///
    /// # Returns
    ///
    /// `ImageGenerationResponse` containing generated image data
    ///
    /// # Example
    ///
    /// ```typescript
    /// import { LLMKitClient, ImageGenerationRequest, ImageSize, ImageQuality } from 'llmkit'
    ///
    /// const client = LLMKitClient.fromEnv()
    ///
    /// const request = new ImageGenerationRequest('fal-ai/flux/dev', 'A serene landscape')
    /// request.with_n(1)
    /// request.with_size(ImageSize.Square1024)
    /// request.with_quality(ImageQuality.Hd)
    ///
    /// const response = await client.generateImage(request)
    /// console.log(`Generated ${response.count} images`)
    /// ```
    #[napi]
    pub async fn generate_image(
        &self,
        request: &JsImageGenerationRequest,
    ) -> Result<JsImageGenerationResponse> {
        let mut core_request =
            llmkit::ImageGenerationRequest::new(request.model.clone(), request.prompt.clone());

        if let Some(n) = request.n {
            core_request = core_request.with_n(n);
        }
        if let Some(size) = request.size {
            let image_size = match size {
                JsImageSize::Square256 => llmkit::ImageSize::Square256,
                JsImageSize::Square512 => llmkit::ImageSize::Square512,
                JsImageSize::Square1024 => llmkit::ImageSize::Square1024,
                JsImageSize::Portrait1024x1792 => llmkit::ImageSize::Portrait1024x1792,
                JsImageSize::Landscape1792x1024 => llmkit::ImageSize::Landscape1792x1024,
            };
            core_request = core_request.with_size(image_size);
        }
        if let Some(quality) = request.quality {
            let image_quality = match quality {
                JsImageQuality::Hd => llmkit::ImageQuality::Hd,
                JsImageQuality::Standard => llmkit::ImageQuality::Standard,
            };
            core_request = core_request.with_quality(image_quality);
        }

        let response = self
            .inner
            .generate_image(core_request)
            .await
            .map_err(convert_error)?;

        let images = response
            .images
            .into_iter()
            .map(|img| JsGeneratedImage {
                url: img.url,
                b64_json: img.b64_json,
                revised_prompt: img.revised_prompt,
            })
            .collect();

        Ok(JsImageGenerationResponse {
            created: response.created as i64,
            images,
        })
    }

    /// Generate images from a text prompt with a specific provider.
    ///
    /// @param providerName - Name of the provider to use
    /// @param request - ImageGenerationRequest with prompt and parameters
    /// @returns ImageGenerationResponse with generated images
    #[napi]
    pub async fn generate_image_with_provider(
        &self,
        provider_name: String,
        request: &JsImageGenerationRequest,
    ) -> Result<JsImageGenerationResponse> {
        let mut core_request =
            llmkit::ImageGenerationRequest::new(request.model.clone(), request.prompt.clone());

        if let Some(n) = request.n {
            core_request = core_request.with_n(n);
        }
        if let Some(size) = request.size {
            let image_size = match size {
                JsImageSize::Square256 => llmkit::ImageSize::Square256,
                JsImageSize::Square512 => llmkit::ImageSize::Square512,
                JsImageSize::Square1024 => llmkit::ImageSize::Square1024,
                JsImageSize::Portrait1024x1792 => llmkit::ImageSize::Portrait1024x1792,
                JsImageSize::Landscape1792x1024 => llmkit::ImageSize::Landscape1792x1024,
            };
            core_request = core_request.with_size(image_size);
        }
        if let Some(quality) = request.quality {
            let image_quality = match quality {
                JsImageQuality::Hd => llmkit::ImageQuality::Hd,
                JsImageQuality::Standard => llmkit::ImageQuality::Standard,
            };
            core_request = core_request.with_quality(image_quality);
        }

        let response = self
            .inner
            .generate_image_with_provider(&provider_name, core_request)
            .await
            .map_err(convert_error)?;

        let images = response
            .images
            .into_iter()
            .map(|img| JsGeneratedImage {
                url: img.url,
                b64_json: img.b64_json,
                revised_prompt: img.revised_prompt,
            })
            .collect();

        Ok(JsImageGenerationResponse {
            created: response.created as i64,
            images,
        })
    }

    /// Rank documents by relevance to a query.
    #[napi]
    pub async fn rank_documents(&self, request: &JsRankingRequest) -> Result<JsRankingResponse> {
        let mut core_request = llmkit::RankingRequest::new(
            request.model.clone(),
            request.query.clone(),
            request.documents.clone(),
        );

        if let Some(top_k) = request.top_k {
            core_request = core_request.with_top_k(top_k as usize);
        }
        core_request = core_request.with_documents();

        let response = self.inner.rank(core_request).await.map_err(convert_error)?;

        let results = response
            .results
            .into_iter()
            .map(|r| JsRankedDocument {
                index: r.index as u32,
                document: r.document.unwrap_or_default(),
                score: r.score as f64,
            })
            .collect();

        Ok(JsRankingResponse { results })
    }

    /// Rank documents by relevance to a query with a specific provider.
    ///
    /// @param providerName - Name of the provider to use
    /// @param request - RankingRequest with query and documents
    /// @returns RankingResponse with ranked documents
    #[napi]
    pub async fn rank_documents_with_provider(
        &self,
        provider_name: String,
        request: &JsRankingRequest,
    ) -> Result<JsRankingResponse> {
        let mut core_request = llmkit::RankingRequest::new(
            request.model.clone(),
            request.query.clone(),
            request.documents.clone(),
        );

        if let Some(top_k) = request.top_k {
            core_request = core_request.with_top_k(top_k as usize);
        }
        core_request = core_request.with_documents();

        let response = self
            .inner
            .rank_with_provider(&provider_name, core_request)
            .await
            .map_err(convert_error)?;

        let results = response
            .results
            .into_iter()
            .map(|r| JsRankedDocument {
                index: r.index as u32,
                document: r.document.unwrap_or_default(),
                score: r.score as f64,
            })
            .collect();

        Ok(JsRankingResponse { results })
    }

    /// Rerank search results for semantic relevance.
    #[napi]
    pub async fn rerank_results(
        &self,
        request: &JsRerankingRequest,
    ) -> Result<JsRerankingResponse> {
        let mut core_request = llmkit::RankingRequest::new(
            request.model.clone(),
            request.query.clone(),
            request.documents.clone(),
        );

        if let Some(top_n) = request.top_n {
            core_request = core_request.with_top_k(top_n as usize);
        }
        core_request = core_request.with_documents();

        let response = self.inner.rank(core_request).await.map_err(convert_error)?;

        let results = response
            .results
            .into_iter()
            .map(|r| JsRerankedResult {
                index: r.index as u32,
                document: r.document.unwrap_or_default(),
                relevance_score: r.score as f64,
            })
            .collect();

        Ok(JsRerankingResponse { results })
    }

    /// Check content for policy violations.
    #[napi]
    pub async fn moderate_text(
        &self,
        request: &JsModerationRequest,
    ) -> Result<JsModerationResponse> {
        let core_request =
            llmkit::ModerationRequest::new(request.model.clone(), request.text.clone());

        let response = self
            .inner
            .moderate(core_request)
            .await
            .map_err(convert_error)?;

        let scores = JsModerationScores {
            hate: response.category_scores.hate as f64,
            hate_threatening: 0.0,
            harassment: response.category_scores.harassment as f64,
            harassment_threatening: 0.0,
            self_harm: response.category_scores.self_harm as f64,
            self_harm_intent: 0.0,
            self_harm_instructions: 0.0,
            sexual: response.category_scores.sexual as f64,
            sexual_minors: 0.0,
            violence: response.category_scores.violence as f64,
            violence_graphic: 0.0,
        };

        Ok(JsModerationResponse {
            flagged: response.flagged,
            scores,
        })
    }

    /// Check content for policy violations with a specific provider.
    ///
    /// @param providerName - Name of the provider to use
    /// @param request - ModerationRequest with text to check
    /// @returns ModerationResponse with flagged status and scores
    #[napi]
    pub async fn moderate_text_with_provider(
        &self,
        provider_name: String,
        request: &JsModerationRequest,
    ) -> Result<JsModerationResponse> {
        let core_request =
            llmkit::ModerationRequest::new(request.model.clone(), request.text.clone());

        let response = self
            .inner
            .moderate_with_provider(&provider_name, core_request)
            .await
            .map_err(convert_error)?;

        let scores = JsModerationScores {
            hate: response.category_scores.hate as f64,
            hate_threatening: 0.0,
            harassment: response.category_scores.harassment as f64,
            harassment_threatening: 0.0,
            self_harm: response.category_scores.self_harm as f64,
            self_harm_intent: 0.0,
            self_harm_instructions: 0.0,
            sexual: response.category_scores.sexual as f64,
            sexual_minors: 0.0,
            violence: response.category_scores.violence as f64,
            violence_graphic: 0.0,
        };

        Ok(JsModerationResponse {
            flagged: response.flagged,
            scores,
        })
    }

    /// Classify text into provided labels.
    #[napi]
    pub async fn classify_text(
        &self,
        request: &JsClassificationRequest,
    ) -> Result<JsClassificationResponse> {
        let core_request = llmkit::ClassificationRequest::new(
            request.model.clone(),
            request.text.clone(),
            request.labels.clone(),
        );

        let response = self
            .inner
            .classify(core_request)
            .await
            .map_err(convert_error)?;

        let results = response
            .predictions
            .into_iter()
            .map(|p| JsClassificationResult {
                label: p.label,
                confidence: p.score as f64,
            })
            .collect();

        Ok(JsClassificationResponse { results })
    }

    /// Classify text into provided labels with a specific provider.
    ///
    /// @param providerName - Name of the provider to use
    /// @param request - ClassificationRequest with text and labels
    /// @returns ClassificationResponse with classifications
    #[napi]
    pub async fn classify_text_with_provider(
        &self,
        provider_name: String,
        request: &JsClassificationRequest,
    ) -> Result<JsClassificationResponse> {
        let core_request = llmkit::ClassificationRequest::new(
            request.model.clone(),
            request.text.clone(),
            request.labels.clone(),
        );

        let response = self
            .inner
            .classify_with_provider(&provider_name, core_request)
            .await
            .map_err(convert_error)?;

        let results = response
            .predictions
            .into_iter()
            .map(|p| JsClassificationResult {
                label: p.label,
                confidence: p.score as f64,
            })
            .collect();

        Ok(JsClassificationResponse { results })
    }
}

// Helper methods (not exposed to JavaScript)
impl JsLLMKitClient {
    /// Apply retry configuration to the builder (reference version).
    ///
    /// - None: Use default retry (with_default_retry())
    /// - &JsRetryConfig: Use custom retry configuration
    fn apply_retry_config_ref(
        builder: llmkit::ClientBuilder,
        retry_config: Option<&JsRetryConfig>,
    ) -> llmkit::ClientBuilder {
        match retry_config {
            None => {
                // Default: use production retry config
                builder.with_default_retry()
            }
            Some(config) => {
                // Use the provided config (clone the inner since we have a reference)
                builder.with_retry(config.inner.clone())
            }
        }
    }

    /// Add a provider to the builder based on the provider name and configuration.
    fn add_provider_to_builder(
        builder: llmkit::ClientBuilder,
        provider_name: &str,
        config: ProviderConfig,
        _runtime: &tokio::runtime::Runtime,
    ) -> Result<llmkit::ClientBuilder> {
        let err = |e: llmkit::Error| Error::from_reason(e.to_string());

        match provider_name.to_lowercase().as_str() {
            // Core providers
            "anthropic" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_anthropic(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("anthropic requires 'apiKey'"))
                }
            }
            "openai" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_openai(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("openai requires 'apiKey'"))
                }
            }
            "azure" | "azure_openai" | "azureopenai" => {
                let api_key = config
                    .api_key
                    .ok_or_else(|| Error::from_reason("azure requires 'apiKey'"))?;
                let deployment = config
                    .deployment
                    .ok_or_else(|| Error::from_reason("azure requires 'deployment'"))?;

                // Either endpoint (full URL) or base_url is required
                let azure_config = if let Some(endpoint) = config.endpoint.clone() {
                    // Extract resource name from endpoint URL
                    let resource_name = endpoint
                        .trim_start_matches("https://")
                        .split('.')
                        .next()
                        .unwrap_or("azure")
                        .to_string();
                    AzureConfig::new(resource_name, deployment, api_key).with_base_url(endpoint)
                } else if let Some(base_url) = config.base_url.clone() {
                    let resource_name = base_url
                        .trim_start_matches("https://")
                        .split('.')
                        .next()
                        .unwrap_or("azure")
                        .to_string();
                    AzureConfig::new(resource_name, deployment, api_key).with_base_url(base_url)
                } else {
                    return Err(Error::from_reason("azure requires 'endpoint' or 'baseUrl'"));
                };

                Ok(builder.with_azure(azure_config).map_err(err)?)
            }
            "bedrock" => {
                let region = config.region.unwrap_or_else(|| "us-east-1".to_string());
                Ok(builder.with_bedrock_region(region))
            }
            // Google providers
            "google" | "gemini" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_google(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("google requires 'apiKey'"))
                }
            }
            "vertex" | "vertex_ai" | "vertexai" => {
                // Vertex now uses ADC (Application Default Credentials)
                // Service account file path can be set via GOOGLE_APPLICATION_CREDENTIALS
                // or use gcloud auth application-default login
                let project = config.project.ok_or_else(|| {
                    Error::from_reason(
                        "vertex requires 'project'. Also set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'",
                    )
                })?;
                let location = config
                    .location
                    .or(config.region)
                    .unwrap_or_else(|| "us-central1".to_string());

                // Set environment variables for Vertex config
                std::env::set_var("GOOGLE_CLOUD_PROJECT", &project);
                std::env::set_var("GOOGLE_CLOUD_LOCATION", &location);

                // Use from_env (ADC) - returns builder directly (async initialization deferred to build())
                Ok(builder.with_vertex_from_env())
            }
            // Fast inference providers
            "groq" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_groq(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("groq requires 'apiKey'"))
                }
            }
            "mistral" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_mistral(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("mistral requires 'apiKey'"))
                }
            }
            "cerebras" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_cerebras(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("cerebras requires 'apiKey'"))
                }
            }
            "sambanova" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_sambanova(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("sambanova requires 'apiKey'"))
                }
            }
            "fireworks" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_fireworks(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("fireworks requires 'apiKey'"))
                }
            }
            "deepseek" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_deepseek(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("deepseek requires 'apiKey'"))
                }
            }
            // Enterprise providers
            "cohere" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_cohere(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("cohere requires 'apiKey'"))
                }
            }
            "ai21" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_ai21(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("ai21 requires 'apiKey'"))
                }
            }
            // OpenAI-compatible hosted providers
            "together" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_together(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("together requires 'apiKey'"))
                }
            }
            "perplexity" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_perplexity(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("perplexity requires 'apiKey'"))
                }
            }
            "anyscale" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_anyscale(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("anyscale requires 'apiKey'"))
                }
            }
            "deepinfra" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_deepinfra(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("deepinfra requires 'apiKey'"))
                }
            }
            "novita" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_novita(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("novita requires 'apiKey'"))
                }
            }
            "hyperbolic" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_hyperbolic(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("hyperbolic requires 'apiKey'"))
                }
            }
            // Inference platforms
            "huggingface" | "hf" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_huggingface(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("huggingface requires 'apiKey'"))
                }
            }
            "replicate" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_replicate(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("replicate requires 'apiKey'"))
                }
            }
            "baseten" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_baseten(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("baseten requires 'apiKey'"))
                }
            }
            "runpod" => {
                let endpoint_id = config
                    .endpoint_id
                    .ok_or_else(|| Error::from_reason("runpod requires 'endpointId'"))?;
                let api_key = config
                    .api_key
                    .ok_or_else(|| Error::from_reason("runpod requires 'apiKey'"))?;
                Ok(builder.with_runpod(endpoint_id, api_key).map_err(err)?)
            }
            // Cloud providers
            "cloudflare" => {
                let account_id = config
                    .account_id
                    .ok_or_else(|| Error::from_reason("cloudflare requires 'accountId'"))?;
                let api_token = config.api_token.or(config.api_key).ok_or_else(|| {
                    Error::from_reason("cloudflare requires 'apiToken' or 'apiKey'")
                })?;
                Ok(builder
                    .with_cloudflare(account_id, api_token)
                    .map_err(err)?)
            }
            "watsonx" => {
                let api_key = config
                    .api_key
                    .ok_or_else(|| Error::from_reason("watsonx requires 'apiKey'"))?;
                let project_id = config
                    .project
                    .ok_or_else(|| Error::from_reason("watsonx requires 'project'"))?;
                Ok(builder.with_watsonx(api_key, project_id).map_err(err)?)
            }
            "databricks" => {
                let host = config
                    .host
                    .ok_or_else(|| Error::from_reason("databricks requires 'host'"))?;
                let token = config
                    .token
                    .or(config.api_key)
                    .ok_or_else(|| Error::from_reason("databricks requires 'token' or 'apiKey'"))?;
                Ok(builder.with_databricks(host, token).map_err(err)?)
            }
            // Local providers
            "ollama" => {
                if let Some(base_url) = config.base_url {
                    Ok(builder
                        .with_openai_compatible("ollama", base_url, None)
                        .map_err(err)?)
                } else {
                    // Default Ollama URL
                    Ok(builder
                        .with_openai_compatible("ollama", "http://localhost:11434/v1", None)
                        .map_err(err)?)
                }
            }
            // Router/gateway providers
            "openrouter" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_openrouter(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("openrouter requires 'apiKey'"))
                }
            }
            // Additional inference providers
            "xai" | "grok" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_xai(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("xai requires 'apiKey'"))
                }
            }
            "nvidia_nim" | "nvidia" | "nim" | "nvidiaNim" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_nvidia_nim(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("nvidia_nim requires 'apiKey'"))
                }
            }
            "lambda" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_lambda(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("lambda requires 'apiKey'"))
                }
            }
            "friendli" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_friendli(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("friendli requires 'apiKey'"))
                }
            }
            // Asian providers
            "baidu" | "ernie" => {
                let api_key = config.api_key.ok_or_else(|| {
                    Error::from_reason("baidu requires 'apiKey'")
                })?;
                let secret_key = config.secret_key.ok_or_else(|| {
                    Error::from_reason("baidu requires 'secretKey'")
                })?;
                Ok(builder.with_baidu(api_key, secret_key).map_err(err)?)
            }
            "alibaba" | "qwen" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_alibaba(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("alibaba requires 'apiKey'"))
                }
            }
            "volcengine" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_volcengine(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("volcengine requires 'apiKey'"))
                }
            }
            // Regional providers
            "maritaca" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_maritaca(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("maritaca requires 'apiKey'"))
                }
            }
            "lighton" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_lighton(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("lighton requires 'apiKey'"))
                }
            }
            // Embedding/multimodal providers
            "voyage" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_voyage(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("voyage requires 'apiKey'"))
                }
            }
            "jina" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_jina(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("jina requires 'apiKey'"))
                }
            }
            "stability" => {
                if let Some(key) = config.api_key {
                    Ok(builder.with_stability(key).map_err(err)?)
                } else {
                    Err(Error::from_reason("stability requires 'apiKey'"))
                }
            }
            // Local providers (lm_studio, vllm, tgi, llamafile) are not yet supported in Node.js bindings
            // They will be implemented in a future release
            "lm_studio" | "lmstudio" | "vllm" | "tgi" | "llamafile" => {
                Err(Error::from_reason(format!("Local provider '{}' is not yet supported. Please use a cloud provider instead.", provider_name)))
            }
            // Generic OpenAI-compatible
            "openai_compatible" | "openaicompatible" => {
                let name = config.model_id.unwrap_or_else(|| "custom".to_string());
                let base_url = config
                    .base_url
                    .ok_or_else(|| Error::from_reason("openaiCompatible requires 'baseUrl'"))?;
                Ok(builder
                    .with_openai_compatible(name, base_url, config.api_key)
                    .map_err(err)?)
            }
            _ => Err(Error::from_reason(format!(
                "Unknown provider: '{}'. Supported providers: anthropic, openai, azure, bedrock, \
                google, vertex, groq, mistral, cerebras, sambanova, fireworks, deepseek, cohere, \
                ai21, together, perplexity, anyscale, deepinfra, novita, hyperbolic, huggingface, \
                replicate, baseten, runpod, cloudflare, watsonx, databricks, ollama, openrouter, \
                xai, nvidiaNim, lambda, friendli, baidu, alibaba, volcengine, maritaca, lighton, \
                voyage, jina, stability, openaiCompatible",
                provider_name
            ))),
        }
    }
}

// ============================================================================
// CLIENT BUILDER - Fluent Builder Pattern
// ============================================================================

/// Fluent builder for LLMKitClient.
///
/// Provides a fluent builder pattern for configuring the client with specific providers.
/// Each provider can be added using `with*FromEnv()` or `with*(apiKey)` methods.
///
/// @example
/// ```typescript
/// import { ClientBuilder } from 'llmkit'
///
/// // Build client with specific providers
/// const client = await new ClientBuilder()
///     .withAnthropicFromEnv()
///     .withOpenAIFromEnv()
///     .withGroq("your-groq-api-key")
///     .withDefaultRetry()
///     .build()
/// ```
#[napi]
pub struct JsClientBuilder {
    builder: std::sync::Mutex<Option<llmkit::ClientBuilder>>,
}

#[napi]
impl JsClientBuilder {
    /// Create a new client builder.
    #[napi(constructor)]
    pub fn new() -> napi::Result<Self> {
        Ok(Self {
            builder: std::sync::Mutex::new(Some(LLMKitClient::builder())),
        })
    }

    // ========================================================================
    // CORE PROVIDERS
    // ========================================================================

    /// Add Anthropic provider from ANTHROPIC_API_KEY environment variable.
    #[napi(js_name = "withAnthropicFromEnv")]
    pub fn with_anthropic_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_anthropic_from_env());
        Ok(self)
    }

    /// Add Anthropic provider with explicit API key.
    #[napi(js_name = "withAnthropic")]
    pub fn with_anthropic(&self, api_key: String) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        let builder = builder
            .with_anthropic(api_key)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        *guard = Some(builder);
        Ok(self)
    }

    /// Add OpenAI provider from OPENAI_API_KEY environment variable.
    #[napi(js_name = "withOpenAIFromEnv")]
    pub fn with_openai_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_openai_from_env());
        Ok(self)
    }

    /// Add OpenAI provider with explicit API key.
    #[napi(js_name = "withOpenAI")]
    pub fn with_openai(&self, api_key: String) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        let builder = builder
            .with_openai(api_key)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        *guard = Some(builder);
        Ok(self)
    }

    /// Add Azure OpenAI provider from environment variables.
    #[napi(js_name = "withAzureFromEnv")]
    pub fn with_azure_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_azure_from_env());
        Ok(self)
    }

    // ========================================================================
    // CLOUD PROVIDERS
    // ========================================================================

    /// Add AWS Bedrock provider from environment.
    #[napi(js_name = "withBedrockFromEnv")]
    pub fn with_bedrock_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_bedrock_from_env());
        Ok(self)
    }

    /// Add Google Vertex AI provider from environment.
    #[napi(js_name = "withVertexFromEnv")]
    pub fn with_vertex_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_vertex_from_env());
        Ok(self)
    }

    /// Add Google AI (Gemini) provider from GOOGLE_API_KEY environment variable.
    #[napi(js_name = "withGoogleFromEnv")]
    pub fn with_google_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_google_from_env());
        Ok(self)
    }

    // ========================================================================
    // FAST INFERENCE PROVIDERS
    // ========================================================================

    /// Add Groq provider from GROQ_API_KEY environment variable.
    #[napi(js_name = "withGroqFromEnv")]
    pub fn with_groq_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_groq_from_env());
        Ok(self)
    }

    /// Add Groq provider with explicit API key.
    #[napi(js_name = "withGroq")]
    pub fn with_groq(&self, api_key: String) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        let builder = builder
            .with_groq(api_key)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        *guard = Some(builder);
        Ok(self)
    }

    /// Add Mistral provider from MISTRAL_API_KEY environment variable.
    #[napi(js_name = "withMistralFromEnv")]
    pub fn with_mistral_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_mistral_from_env());
        Ok(self)
    }

    /// Add Mistral provider with explicit API key.
    #[napi(js_name = "withMistral")]
    pub fn with_mistral(&self, api_key: String) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        let builder = builder
            .with_mistral(api_key)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        *guard = Some(builder);
        Ok(self)
    }

    /// Add DeepSeek provider from DEEPSEEK_API_KEY environment variable.
    #[napi(js_name = "withDeepSeekFromEnv")]
    pub fn with_deepseek_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_deepseek_from_env());
        Ok(self)
    }

    /// Add DeepSeek provider with explicit API key.
    #[napi(js_name = "withDeepSeek")]
    pub fn with_deepseek(&self, api_key: String) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        let builder = builder
            .with_deepseek(api_key)
            .map_err(|e| Error::from_reason(e.to_string()))?;
        *guard = Some(builder);
        Ok(self)
    }

    // ========================================================================
    // ENTERPRISE & HOSTED PROVIDERS
    // ========================================================================

    /// Add Cohere provider from COHERE_API_KEY environment variable.
    #[napi(js_name = "withCohereFromEnv")]
    pub fn with_cohere_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_cohere_from_env());
        Ok(self)
    }

    /// Add Together AI provider from TOGETHER_API_KEY environment variable.
    #[napi(js_name = "withTogetherFromEnv")]
    pub fn with_together_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_together_from_env());
        Ok(self)
    }

    /// Add Perplexity provider from PERPLEXITY_API_KEY environment variable.
    #[napi(js_name = "withPerplexityFromEnv")]
    pub fn with_perplexity_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_perplexity_from_env());
        Ok(self)
    }

    /// Add OpenRouter provider from OPENROUTER_API_KEY environment variable.
    #[napi(js_name = "withOpenRouterFromEnv")]
    pub fn with_openrouter_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_openrouter_from_env());
        Ok(self)
    }

    /// Add xAI (Grok) provider from XAI_API_KEY environment variable.
    #[napi(js_name = "withXAIFromEnv")]
    pub fn with_xai_from_env(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_xai_from_env());
        Ok(self)
    }

    // ========================================================================
    // RETRY CONFIGURATION
    // ========================================================================

    /// Use default retry configuration (10 retries with exponential backoff).
    #[napi(js_name = "withDefaultRetry")]
    pub fn with_default_retry(&self) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_default_retry());
        Ok(self)
    }

    /// Use custom retry configuration.
    #[napi(js_name = "withRetry")]
    pub fn with_retry(&self, config: &JsRetryConfig) -> napi::Result<&Self> {
        let mut guard = self
            .builder
            .lock()
            .map_err(|e| Error::from_reason(e.to_string()))?;
        let builder = guard
            .take()
            .ok_or_else(|| Error::from_reason("Builder already consumed"))?;
        *guard = Some(builder.with_retry(config.inner.clone()));
        Ok(self)
    }

    // ========================================================================
    // BUILD
    // ========================================================================

    /// Build the LLMKitClient.
    #[napi]
    pub async fn build(&self) -> napi::Result<JsLLMKitClient> {
        let builder = {
            let mut guard = self
                .builder
                .lock()
                .map_err(|e| Error::from_reason(e.to_string()))?;
            guard
                .take()
                .ok_or_else(|| Error::from_reason("Builder already consumed"))?
        };

        let client = builder
            .build()
            .await
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(JsLLMKitClient {
            inner: Arc::new(client),
        })
    }
}
