"""
ModelSuite: Unified LLM API client for Python

Provides access to 30+ LLM providers through a single interface.

Example usage:

    # Synchronous client
    from modelsuite import ModelSuiteClient, CompletionRequest, Message

    client = ModelSuiteClient.from_env()
    response = client.complete(CompletionRequest(
        model="claude-sonnet-4-20250514",
        messages=[Message.user("Hello!")],
    ))
    print(response.text_content())

    # Async client
    from modelsuite import AsyncModelSuiteClient

    async def main():
        client = AsyncModelSuiteClient.from_env()
        response = await client.complete(request)
        print(response.text_content())
"""

from modelsuite._modelsuite import (
    AsyncModelSuiteClient,
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
    ModelSuiteClient,
    # Exceptions
    ModelSuiteError,
    NetworkError,
    NotSupportedError,
    # Model Registry types
    Provider,
    ProviderNotFoundError,
    RateLimitError,
    RegistryStats,
    # Retry configuration
    RetryConfig,
    # Enums
    Role,
    ServerError,
    StopReason,
    # Streaming types
    StreamChunk,
    StreamError,
    StreamEventType,
    StreamIterator,
    StructuredOutput,
    ThinkingConfig,
    ThinkingType,
    TimeoutError,
    # Token counting
    TokenCountRequest,
    TokenCountResult,
    ToolBuilder,
    # Tools
    ToolDefinition,
    Usage,
    get_all_models,
    get_available_models,
    get_cheapest_model,
    get_classifier_models,
    get_current_models,
    # Model Registry functions
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
    "ModelSuiteClient",
    "AsyncModelSuiteClient",
    # Retry configuration
    "RetryConfig",
    # Exceptions
    "ModelSuiteError",
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
