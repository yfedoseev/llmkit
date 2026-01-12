"""
LLMKit: Unified LLM API client for Python

Provides access to 70+ LLM providers through a single interface.

Example usage:

    # Synchronous client
    from llmkit import LLMKitClient, CompletionRequest, Message

    client = LLMKitClient.from_env()
    response = client.complete(CompletionRequest(
        model="claude-sonnet-4-20250514",
        messages=[Message.user("Hello!")],
    ))
    print(response.text_content())

    # Async client
    from llmkit import AsyncLLMKitClient

    async def main():
        client = AsyncLLMKitClient.from_env()
        response = await client.complete(request)
        print(response.text_content())
"""

from llmkit._llmkit import (
    AsyncLLMKitClient,
    AsyncStreamIterator,
    AuthenticationError,
    BatchError,
    BatchJob,
    # Batch processing
    BatchRequest,
    BatchRequestCounts,
    BatchResult,
    BatchStatus,
    CacheBreakpoint,
    CacheControl,
    # ClientBuilder for fluent API
    ClientBuilder,
    # Request/Response types
    CompletionRequest,
    CompletionResponse,
    ConfigurationError,
    # Message types
    ContentBlock,
    ContentDelta,
    ContentFilteredError,
    ContextLengthError,
    Embedding,
    EmbeddingInputType,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    # Embedding types
    EncodingFormat,
    InvalidRequestError,
    Message,
    ModelBenchmarks,
    ModelCapabilities,
    ModelInfo,
    ModelNotFoundError,
    ModelPricing,
    ModelStatus,
    # Clients
    LLMKitClient,
    # Exceptions
    LLMKitError,
    NetworkError,
    NotSupportedError,
    # Model Registry types
    Provider,
    ProviderNotFoundError,
    RateLimitError,
    # OpenAI Realtime API types
    RealtimeProvider,
    RealtimeSession,
    RegistryStats,
    # Retry configuration
    RetryConfig,
    # Enums
    Role,
    ServerError,
    SessionConfig,
    StopReason,
    # Streaming types
    StreamChunk,
    StreamError,
    StreamEventType,
    StreamIterator,
    StructuredOutput,
    ThinkingConfig,
    ThinkingEffort,
    ThinkingType,
    TimeoutError,
    # Token counting
    TokenCountRequest,
    TokenCountResult,
    ToolBuilder,
    # Tools
    ToolDefinition,
    Usage,
    VadConfig,
    # Audio transcription types
    AudioLanguage,
    DeepgramVersion,
    TranscribeOptions,
    TranscribeResponse,
    TranscriptionConfig,
    TranscriptionRequest,
    Word,
    # Audio synthesis types
    LatencyMode,
    SynthesisRequest,
    SynthesizeOptions,
    SynthesizeResponse,
    Voice,
    VoiceSettings,
    # Video generation types
    VideoGenerationOptions,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoModel,
    # Image generation types
    GeneratedImage,
    ImageFormat,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageQuality,
    ImageSize,
    ImageStyle,
    # Specialized API types - Ranking
    RankedDocument,
    RankingRequest,
    RankingResponse,
    RerankedResult,
    RerankingRequest,
    RerankingResponse,
    # Specialized API types - Moderation
    ModerationRequest,
    ModerationResponse,
    ModerationScores,
    # Specialized API types - Classification
    ClassificationRequest,
    ClassificationResponse,
    ClassificationResult,
    # Model Registry functions
    get_all_models,
    get_available_models,
    get_cheapest_model,
    get_classifier_models,
    get_current_models,
    get_model_info,
    get_models_by_provider,
    get_models_with_capability,
    get_registry_stats,
    list_providers,
    supports_structured_output,
)

__all__ = [
    # Enums
    "Role",
    "StopReason",
    "StreamEventType",
    "CacheControl",
    "ThinkingType",
    "ThinkingEffort",
    "BatchStatus",
    # Message types
    "ContentBlock",
    "Message",
    "CacheBreakpoint",
    "ThinkingConfig",
    "StructuredOutput",
    # Request/Response types
    "CompletionRequest",
    "CompletionResponse",
    "Usage",
    # Token counting
    "TokenCountRequest",
    "TokenCountResult",
    # Batch processing
    "BatchRequest",
    "BatchJob",
    "BatchRequestCounts",
    "BatchResult",
    "BatchError",
    # Embedding types
    "EncodingFormat",
    "EmbeddingInputType",
    "EmbeddingRequest",
    "Embedding",
    "EmbeddingUsage",
    "EmbeddingResponse",
    # Streaming types
    "StreamChunk",
    "ContentDelta",
    "StreamIterator",
    "AsyncStreamIterator",
    # Tools
    "ToolDefinition",
    "ToolBuilder",
    # OpenAI Realtime API types
    "VadConfig",
    "SessionConfig",
    "RealtimeSession",
    "RealtimeProvider",
    # Clients
    "LLMKitClient",
    "AsyncLLMKitClient",
    "ClientBuilder",
    # Retry configuration
    "RetryConfig",
    # Audio transcription types
    "AudioLanguage",
    "DeepgramVersion",
    "TranscribeOptions",
    "TranscribeResponse",
    "TranscriptionConfig",
    "TranscriptionRequest",
    "Word",
    # Audio synthesis types
    "LatencyMode",
    "SynthesisRequest",
    "SynthesizeOptions",
    "SynthesizeResponse",
    "Voice",
    "VoiceSettings",
    # Video generation types
    "VideoGenerationOptions",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
    "VideoModel",
    # Image generation types
    "GeneratedImage",
    "ImageFormat",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "ImageQuality",
    "ImageSize",
    "ImageStyle",
    # Specialized API types - Ranking
    "RankedDocument",
    "RankingRequest",
    "RankingResponse",
    "RerankedResult",
    "RerankingRequest",
    "RerankingResponse",
    # Specialized API types - Moderation
    "ModerationRequest",
    "ModerationResponse",
    "ModerationScores",
    # Specialized API types - Classification
    "ClassificationRequest",
    "ClassificationResponse",
    "ClassificationResult",
    # Exceptions
    "LLMKitError",
    "ProviderNotFoundError",
    "ConfigurationError",
    "AuthenticationError",
    "RateLimitError",
    "InvalidRequestError",
    "ModelNotFoundError",
    "ContentFilteredError",
    "ContextLengthError",
    "NetworkError",
    "StreamError",
    "TimeoutError",
    "ServerError",
    "NotSupportedError",
    # Model Registry types
    "Provider",
    "ModelStatus",
    "ModelPricing",
    "ModelCapabilities",
    "ModelBenchmarks",
    "RegistryStats",
    "ModelInfo",
    # Model Registry functions
    "get_model_info",
    "get_all_models",
    "get_models_by_provider",
    "get_current_models",
    "get_classifier_models",
    "get_available_models",
    "get_models_with_capability",
    "get_cheapest_model",
    "supports_structured_output",
    "get_registry_stats",
    "list_providers",
]

__version__ = "0.1.0"
