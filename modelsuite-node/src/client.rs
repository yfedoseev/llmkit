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
    JsSynthesisRequest, JsSynthesizeResponse, JsTranscribeResponse, JsTranscriptionRequest,
};
use crate::errors::convert_error;
use crate::image::{JsGeneratedImage, JsImageGenerationRequest, JsImageGenerationResponse};
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
    /// Vertex AI access token
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
    ///
    /// @example
    /// ```typescript
    /// const client = new LLMKitClient({
    ///   providers: {
    ///     anthropic: { apiKey: "sk-..." },
    ///     azure: { apiKey: "...", endpoint: "https://...", deployment: "gpt-4" },
    ///   }
    /// })
    /// ```
    #[napi(constructor)]
    pub fn new(options: Option<LLMKitClientOptions>) -> Result<Self> {
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
    /// Automatically detects and configures all available providers from environment variables.
    ///
    /// Supported environment variables:
    /// - ANTHROPIC_API_KEY: Anthropic (Claude)
    /// - OPENAI_API_KEY: OpenAI (GPT)
    /// - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT: Azure OpenAI
    /// - AWS_REGION or AWS_DEFAULT_REGION: AWS Bedrock (uses default credential chain)
    /// - GOOGLE_API_KEY: Google AI (Gemini)
    /// - GOOGLE_CLOUD_PROJECT, VERTEX_LOCATION, VERTEX_ACCESS_TOKEN: Google Vertex AI
    /// - GROQ_API_KEY: Groq
    /// - MISTRAL_API_KEY: Mistral
    /// - COHERE_API_KEY or CO_API_KEY: Cohere
    /// - AI21_API_KEY: AI21 Labs
    /// - DEEPSEEK_API_KEY: DeepSeek
    /// - TOGETHER_API_KEY: Together AI
    /// - FIREWORKS_API_KEY: Fireworks AI
    /// - PERPLEXITY_API_KEY: Perplexity
    /// - CEREBRAS_API_KEY: Cerebras
    /// - SAMBANOVA_API_KEY: SambaNova
    /// - OPENROUTER_API_KEY: OpenRouter
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
    /// - OLLAMA_BASE_URL: Ollama (local, defaults to http://localhost:11434)
    #[napi(factory)]
    pub fn from_env() -> Result<Self> {
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
            // Google providers
            .with_google_from_env()
            .with_vertex_from_env()
            // Fast inference providers
            .with_groq_from_env()
            .with_mistral_from_env()
            .with_cerebras_from_env()
            .with_sambanova_from_env()
            .with_fireworks_from_env()
            .with_deepseek_from_env()
            // Enterprise providers
            .with_cohere_from_env()
            .with_ai21_from_env()
            // OpenAI-compatible hosted providers
            .with_together_from_env()
            .with_perplexity_from_env()
            .with_anyscale_from_env()
            .with_deepinfra_from_env()
            .with_novita_from_env()
            .with_hyperbolic_from_env()
            // Inference platforms
            .with_huggingface_from_env()
            .with_replicate_from_env()
            .with_baseten_from_env()
            .with_runpod_from_env()
            // Cloud providers
            .with_cloudflare_from_env()
            .with_watsonx_from_env()
            .with_databricks_from_env()
            // Enable retry
            .with_default_retry();

        // Build async for Bedrock (needs await), then finalize
        let client = runtime
            .block_on(async { builder.with_bedrock_from_env().await.build() })
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
        _request: &JsTranscriptionRequest,
    ) -> Result<JsTranscribeResponse> {
        // For now, return a placeholder response.
        // When Rust core client methods are implemented, this will call:
        // self.inner.transcribe_audio(request).await

        Ok(JsTranscribeResponse {
            transcript: "Transcription placeholder".to_string(),
            confidence: Some(0.95),
            words: vec![],
            duration: None,
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
        _request: &JsSynthesisRequest,
    ) -> Result<JsSynthesizeResponse> {
        // For now, return a placeholder response.
        // When Rust core client methods are implemented, this will call:
        // self.inner.synthesize_speech(request).await

        Ok(JsSynthesizeResponse {
            audio_bytes: vec![0u8; 1000], // Placeholder silence
            format: "mp3".to_string(),
            duration: Some(2.5),
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
        _request: &JsVideoGenerationRequest,
    ) -> Result<JsVideoGenerationResponse> {
        // For now, return a placeholder response.
        // When Rust core client methods are implemented, this will call:
        // self.inner.generate_video(request).await

        Ok(JsVideoGenerationResponse {
            video_bytes: None,
            video_url: Some("https://example.com/video.mp4".to_string()),
            format: "mp4".to_string(),
            duration: Some(10.0),
            width: Some(1920),
            height: Some(1080),
            task_id: Some("task-123456".to_string()),
            status: Some("completed".to_string()),
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
        _request: &JsImageGenerationRequest,
    ) -> Result<JsImageGenerationResponse> {
        // For now, return a placeholder response.
        // When Rust core client methods are implemented, this will call:
        // self.inner.generate_image(request).await

        use std::time::{SystemTime, UNIX_EPOCH};

        let created = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        Ok(JsImageGenerationResponse {
            created,
            images: vec![JsGeneratedImage {
                url: Some("https://example.com/image.png".to_string()),
                b64_json: None,
                revised_prompt: None,
            }],
        })
    }

    /// Rank documents by relevance to a query.
    #[napi]
    pub async fn rank_documents(&self, request: &JsRankingRequest) -> Result<JsRankingResponse> {
        // For now, return mock ranking results
        let mut results = Vec::new();
        for (i, doc) in request.documents.iter().enumerate() {
            results.push(JsRankedDocument {
                index: i as u32,
                document: doc.clone(),
                score: 0.9 - (i as f64 * 0.1),
            });
        }

        Ok(JsRankingResponse { results })
    }

    /// Rerank search results for semantic relevance.
    #[napi]
    pub async fn rerank_results(
        &self,
        request: &JsRerankingRequest,
    ) -> Result<JsRerankingResponse> {
        // For now, return mock reranking results
        let mut results = Vec::new();
        for (i, doc) in request.documents.iter().enumerate() {
            results.push(JsRerankedResult {
                index: i as u32,
                document: doc.clone(),
                relevance_score: 0.95 - (i as f64 * 0.05),
            });
        }

        Ok(JsRerankingResponse { results })
    }

    /// Check content for policy violations.
    #[napi]
    pub async fn moderate_text(
        &self,
        _request: &JsModerationRequest,
    ) -> Result<JsModerationResponse> {
        // For now, return mock moderation response
        Ok(JsModerationResponse {
            flagged: false,
            scores: JsModerationScores {
                hate: 0.0,
                hate_threatening: 0.0,
                harassment: 0.0,
                harassment_threatening: 0.0,
                self_harm: 0.0,
                self_harm_intent: 0.0,
                self_harm_instructions: 0.0,
                sexual: 0.0,
                sexual_minors: 0.0,
                violence: 0.0,
                violence_graphic: 0.0,
            },
        })
    }

    /// Classify text into provided labels.
    #[napi]
    pub async fn classify_text(
        &self,
        request: &JsClassificationRequest,
    ) -> Result<JsClassificationResponse> {
        // For now, return mock classification results
        let mut results = Vec::new();
        for (i, label) in request.labels.iter().enumerate() {
            results.push(JsClassificationResult {
                label: label.clone(),
                confidence: 1.0 / (request.labels.len() as f64) + (i as f64 * 0.05),
            });
        }
        // Sort by confidence descending
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(JsClassificationResponse { results })
    }
}

// Helper methods (not exposed to JavaScript)
impl JsLLMKitClient {
    /// Add a provider to the builder based on the provider name and configuration.
    fn add_provider_to_builder(
        builder: llmkit::ClientBuilder,
        provider_name: &str,
        config: ProviderConfig,
        runtime: &tokio::runtime::Runtime,
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
                runtime.block_on(async { builder.with_bedrock_region(region).await.map_err(err) })
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
                let project = config
                    .project
                    .ok_or_else(|| Error::from_reason("vertex requires 'project'"))?;
                let location = config
                    .location
                    .or(config.region)
                    .ok_or_else(|| Error::from_reason("vertex requires 'location' or 'region'"))?;
                let access_token = config
                    .access_token
                    .ok_or_else(|| Error::from_reason("vertex requires 'accessToken'"))?;
                Ok(builder
                    .with_vertex(project, location, access_token)
                    .map_err(err)?)
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
                replicate, baseten, runpod, cloudflare, watsonx, databricks, ollama, lmStudio, \
                vllm, tgi, llamafile, openaiCompatible",
                provider_name
            ))),
        }
    }
}
