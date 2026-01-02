"""
LLMKit: Unified LLM API client for Python

Provides access to 30+ LLM providers through a single interface.

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
    # Enums
    Role,
    StopReason,
    StreamEventType,
    CacheControl,
    ThinkingType,
    BatchStatus,
    # Message types
    ContentBlock,
    Message,
    CacheBreakpoint,
    ThinkingConfig,
    StructuredOutput,
    # Request/Response types
    CompletionRequest,
    CompletionResponse,
    Usage,
    # Token counting
    TokenCountRequest,
    TokenCountResult,
    # Batch processing
    BatchRequest,
    BatchJob,
    BatchRequestCounts,
    BatchResult,
    BatchError,
    # Embedding types
    EncodingFormat,
    EmbeddingInputType,
    EmbeddingRequest,
    Embedding,
    EmbeddingUsage,
    EmbeddingResponse,
    # Streaming types
    StreamChunk,
    ContentDelta,
    StreamIterator,
    AsyncStreamIterator,
    # Tools
    ToolDefinition,
    ToolBuilder,
    # Clients
    LLMKitClient,
    AsyncLLMKitClient,
    # Exceptions
    LLMKitError,
    ProviderNotFoundError,
    ConfigurationError,
    AuthenticationError,
    RateLimitError,
    InvalidRequestError,
    ModelNotFoundError,
    ContentFilteredError,
    ContextLengthError,
    NetworkError,
    StreamError,
    TimeoutError,
    ServerError,
    NotSupportedError,
    # Model Registry types
    Provider,
    ModelStatus,
    ModelPricing,
    ModelCapabilities,
    ModelBenchmarks,
    RegistryStats,
    ModelInfo,
    # Model Registry functions
    get_model_info,
    get_all_models,
    get_models_by_provider,
    get_current_models,
    get_classifier_models,
    get_available_models,
    get_models_with_capability,
    get_cheapest_model,
    supports_structured_output,
    get_registry_stats,
    list_providers,
)

__all__ = [
    # Enums
    "Role",
    "StopReason",
    "StreamEventType",
    "CacheControl",
    "ThinkingType",
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
    # Clients
    "LLMKitClient",
    "AsyncLLMKitClient",
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
