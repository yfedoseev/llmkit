"""Type stubs for llmkit - Unified LLM API client library."""

from collections.abc import AsyncIterator, Iterator
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

# ==================== Enums ====================

class Role(IntEnum):
    """Message role in a conversation."""

    System = 0
    User = 1
    Assistant = 2

class StopReason(IntEnum):
    """Reason the model stopped generating."""

    EndTurn = 0
    MaxTokens = 1
    ToolUse = 2
    StopSequence = 3
    ContentFilter = 4

    @property
    def is_tool_use(self) -> bool: ...
    @property
    def is_complete(self) -> bool: ...

class StreamEventType(IntEnum):
    """Streaming event type."""

    MessageStart = 0
    ContentBlockStart = 1
    ContentBlockDelta = 2
    ContentBlockStop = 3
    MessageDelta = 4
    MessageStop = 5
    Ping = 6
    Error = 7

    @property
    def is_done(self) -> bool: ...

class CacheControl(IntEnum):
    """Cache control type for prompt caching."""

    Ephemeral = 0
    Extended = 1

class ThinkingType(IntEnum):
    """Thinking mode type."""

    Enabled = 0
    Disabled = 1

class BatchStatus(IntEnum):
    """Batch job status."""

    Validating = 0
    InProgress = 1
    Finalizing = 2
    Completed = 3
    Failed = 4
    Expired = 5
    Cancelled = 6

    @property
    def is_processing(self) -> bool: ...
    @property
    def is_done(self) -> bool: ...
    @property
    def is_success(self) -> bool: ...

# ==================== Exceptions ====================

class LLMKitError(Exception):
    """Base exception for all LLMKit errors."""

    ...

class ProviderNotFoundError(LLMKitError):
    """Provider not found or not configured."""

    ...

class ConfigurationError(LLMKitError):
    """Configuration error."""

    ...

class AuthenticationError(LLMKitError):
    """Authentication failed."""

    ...

class RateLimitError(LLMKitError):
    """Rate limit exceeded."""

    retry_after_seconds: Optional[float]

class InvalidRequestError(LLMKitError):
    """Invalid request parameters."""

    ...

class ModelNotFoundError(LLMKitError):
    """Model not found."""

    ...

class ContentFilteredError(LLMKitError):
    """Content was filtered by moderation."""

    ...

class ContextLengthError(LLMKitError):
    """Context length exceeded."""

    ...

class NetworkError(LLMKitError):
    """Network error."""

    ...

class StreamError(LLMKitError):
    """Streaming error."""

    ...

class TimeoutError(LLMKitError):
    """Request timeout."""

    ...

class ServerError(LLMKitError):
    """Server error."""

    status: int

class NotSupportedError(LLMKitError):
    """Feature not supported by provider."""

    ...

# ==================== Content Types ====================

class ContentBlock:
    """A block of content within a message."""

    @staticmethod
    def text(text: str) -> ContentBlock:
        """Create a text content block."""
        ...

    @staticmethod
    def image(media_type: str, data: str) -> ContentBlock:
        """Create an image content block from base64 data."""
        ...

    @staticmethod
    def image_url(url: str) -> ContentBlock:
        """Create an image content block from URL."""
        ...

    @staticmethod
    def tool_use(id: str, name: str, input: Dict[str, Any]) -> ContentBlock:
        """Create a tool use content block."""
        ...

    @staticmethod
    def tool_result(tool_use_id: str, content: str, is_error: bool = False) -> ContentBlock:
        """Create a tool result content block."""
        ...

    @staticmethod
    def thinking(thinking: str) -> ContentBlock:
        """Create a thinking content block."""
        ...

    @staticmethod
    def pdf(data: str) -> ContentBlock:
        """Create a PDF document content block."""
        ...

    @staticmethod
    def text_cached(text: str) -> ContentBlock:
        """Create a text content block with ephemeral caching."""
        ...

    @property
    def is_text(self) -> bool: ...
    @property
    def is_tool_use(self) -> bool: ...
    @property
    def is_tool_result(self) -> bool: ...
    @property
    def is_document(self) -> bool: ...
    @property
    def is_thinking(self) -> bool: ...
    @property
    def is_image(self) -> bool: ...
    @property
    def text_value(self) -> Optional[str]: ...
    @property
    def thinking_content(self) -> Optional[str]: ...
    def as_tool_use(self) -> Optional[Tuple[str, str, Dict[str, Any]]]:
        """Get tool use details if this is a tool use block."""
        ...

    def as_tool_result(self) -> Optional[Tuple[str, str, bool]]:
        """Get tool result details if this is a tool result block."""
        ...

class Message:
    """A message in a conversation."""

    @staticmethod
    def system(content: str) -> Message:
        """Create a system message."""
        ...

    @staticmethod
    def user(content: str) -> Message:
        """Create a user message with text content."""
        ...

    @staticmethod
    def assistant(content: str) -> Message:
        """Create an assistant message with text content."""
        ...

    @staticmethod
    def user_with_content(content: List[ContentBlock]) -> Message:
        """Create a user message with multiple content blocks."""
        ...

    @staticmethod
    def assistant_with_content(content: List[ContentBlock]) -> Message:
        """Create an assistant message with multiple content blocks."""
        ...

    @staticmethod
    def tool_results(results: List[ContentBlock]) -> Message:
        """Create a user message with tool results."""
        ...

    @property
    def role(self) -> Role: ...
    @property
    def content(self) -> List[ContentBlock]: ...
    def text_content(self) -> str:
        """Get concatenated text content from all text blocks."""
        ...

    def has_tool_use(self) -> bool:
        """Check if message contains tool use blocks."""
        ...

    def tool_uses(self) -> List[ContentBlock]:
        """Get all tool use blocks from the message."""
        ...

# ==================== Configuration Types ====================

class CacheBreakpoint:
    """Cache breakpoint for prompt caching."""

    @staticmethod
    def ephemeral() -> CacheBreakpoint:
        """Create ephemeral cache breakpoint (5-minute TTL)."""
        ...

    @staticmethod
    def extended() -> CacheBreakpoint:
        """Create extended cache breakpoint (1-hour TTL)."""
        ...

    @property
    def cache_control(self) -> CacheControl: ...

class ThinkingConfig:
    """Configuration for extended thinking."""

    @staticmethod
    def enabled(budget_tokens: int) -> ThinkingConfig:
        """Enable thinking with a token budget."""
        ...

    @staticmethod
    def disabled() -> ThinkingConfig:
        """Disable thinking."""
        ...

    @property
    def thinking_type(self) -> ThinkingType: ...
    @property
    def budget_tokens(self) -> Optional[int]: ...

class StructuredOutput:
    """Configuration for structured output."""

    @staticmethod
    def json_schema(name: str, schema: Dict[str, Any]) -> StructuredOutput:
        """Create JSON schema structured output."""
        ...

    @staticmethod
    def json_object() -> StructuredOutput:
        """Create JSON object structured output."""
        ...

# ==================== Tool Types ====================

class ToolDefinition:
    """Definition of a tool that can be used by the model."""

    def __init__(self, name: str, description: str, input_schema: Dict[str, Any]) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def description(self) -> str: ...
    @property
    def input_schema(self) -> Dict[str, Any]: ...

class ToolBuilder:
    """Builder for creating tool definitions with a fluent API."""

    def __init__(self, name: str) -> None: ...
    def description(self, description: str) -> ToolBuilder:
        """Set the tool description."""
        ...

    def string_param(self, name: str, description: str, required: bool = True) -> ToolBuilder:
        """Add a string parameter."""
        ...

    def integer_param(self, name: str, description: str, required: bool = True) -> ToolBuilder:
        """Add an integer parameter."""
        ...

    def number_param(self, name: str, description: str, required: bool = True) -> ToolBuilder:
        """Add a number (float) parameter."""
        ...

    def boolean_param(self, name: str, description: str, required: bool = True) -> ToolBuilder:
        """Add a boolean parameter."""
        ...

    def array_param(
        self, name: str, description: str, item_type: str, required: bool = True
    ) -> ToolBuilder:
        """Add an array parameter."""
        ...

    def enum_param(
        self, name: str, description: str, values: List[str], required: bool = True
    ) -> ToolBuilder:
        """Add an enum parameter (string with allowed values)."""
        ...

    def custom_param(self, name: str, schema: Dict[str, Any], required: bool = True) -> ToolBuilder:
        """Add a custom parameter with a JSON schema."""
        ...

    def build(self) -> ToolDefinition:
        """Build the tool definition."""
        ...

# ==================== Request/Response Types ====================

class Usage:
    """Token usage information."""

    @property
    def input_tokens(self) -> int: ...
    @property
    def output_tokens(self) -> int: ...
    @property
    def cache_creation_input_tokens(self) -> Optional[int]: ...
    @property
    def cache_read_input_tokens(self) -> Optional[int]: ...
    def total_tokens(self) -> int:
        """Get total tokens (input + output)."""
        ...

class CompletionRequest:
    """Request to complete a conversation."""

    def __init__(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        tools: Optional[List[ToolDefinition]] = None,
        stop_sequences: Optional[List[str]] = None,
        stream: bool = False,
        thinking_budget: Optional[int] = None,
    ) -> None: ...

    # Builder methods
    def with_system(self, system: str) -> CompletionRequest: ...
    def with_max_tokens(self, max_tokens: int) -> CompletionRequest: ...
    def with_temperature(self, temperature: float) -> CompletionRequest: ...
    def with_top_p(self, top_p: float) -> CompletionRequest: ...
    def with_tools(self, tools: List[ToolDefinition]) -> CompletionRequest: ...
    def with_stop_sequences(self, stop_sequences: List[str]) -> CompletionRequest: ...
    def with_streaming(self) -> CompletionRequest: ...
    def with_thinking(self, budget_tokens: int) -> CompletionRequest: ...
    def with_thinking_config(self, config: ThinkingConfig) -> CompletionRequest: ...
    def with_json_schema(self, name: str, schema: Dict[str, Any]) -> CompletionRequest: ...
    def with_response_format(self, format: StructuredOutput) -> CompletionRequest: ...
    def with_json_output(self) -> CompletionRequest: ...
    def with_prediction(self, predicted_content: str) -> CompletionRequest: ...
    def with_system_caching(self) -> CompletionRequest: ...
    def with_system_caching_extended(self) -> CompletionRequest: ...
    def with_extended_output(self) -> CompletionRequest: ...
    def with_interleaved_thinking(self) -> CompletionRequest: ...
    def with_extra(self, extra: Dict[str, Any]) -> CompletionRequest: ...

    # Properties
    @property
    def model(self) -> str: ...
    @property
    def messages(self) -> List[Message]: ...
    @property
    def system(self) -> Optional[str]: ...
    @property
    def max_tokens(self) -> Optional[int]: ...
    @property
    def temperature(self) -> Optional[float]: ...
    @property
    def stream(self) -> bool: ...

    # Helper methods
    def has_caching(self) -> bool: ...
    def has_thinking(self) -> bool: ...
    def has_structured_output(self) -> bool: ...

class CompletionResponse:
    """Response from a completion request."""

    @property
    def id(self) -> str: ...
    @property
    def model(self) -> str: ...
    @property
    def content(self) -> List[ContentBlock]: ...
    @property
    def stop_reason(self) -> Optional[StopReason]: ...
    @property
    def usage(self) -> Optional[Usage]: ...
    def text_content(self) -> str:
        """Get concatenated text content from all text blocks."""
        ...

    def tool_uses(self) -> List[ContentBlock]:
        """Get all tool use blocks."""
        ...

    def has_tool_use(self) -> bool:
        """Check if response contains tool use blocks."""
        ...

    def thinking_content(self) -> Optional[str]:
        """Get thinking content if present."""
        ...

# ==================== Token Counting Types ====================

class TokenCountRequest:
    """Request to count tokens for a model.

    This allows estimation of token counts before making a completion request,
    useful for cost estimation and context window management.

    Example:
        request = TokenCountRequest(
            model="claude-sonnet-4-20250514",
            messages=[Message.user("Hello, how are you?")],
            system="You are a helpful assistant",
        )
        result = client.count_tokens(request)
        print(f"Input tokens: {result.input_tokens}")
    """

    def __init__(
        self,
        model: str,
        messages: List[Message],
        system: Optional[str] = None,
        tools: Optional[List[ToolDefinition]] = None,
    ) -> None:
        """Create a new token count request.

        Args:
            model: Model identifier (e.g., "claude-sonnet-4-20250514")
            messages: Conversation messages
            system: Optional system prompt
            tools: Optional list of tool definitions
        """
        ...

    @staticmethod
    def from_completion_request(request: CompletionRequest) -> TokenCountRequest:
        """Create a token count request from an existing completion request."""
        ...

    def with_system(self, system: str) -> TokenCountRequest:
        """Builder method: Set the system prompt."""
        ...

    def with_tools(self, tools: List[ToolDefinition]) -> TokenCountRequest:
        """Builder method: Set the tools."""
        ...

    @property
    def model(self) -> str: ...
    @property
    def messages(self) -> List[Message]: ...
    @property
    def system(self) -> Optional[str]: ...

class TokenCountResult:
    """Result of a token counting request.

    Example:
        result = client.count_tokens(request)
        print(f"Input tokens: {result.input_tokens}")
    """

    @property
    def input_tokens(self) -> int:
        """Total number of input tokens."""
        ...

# ==================== Batch Processing Types ====================

class BatchRequest:
    """A request within a batch.

    Example:
        batch_requests = [
            BatchRequest("request-1", completion_request1),
            BatchRequest("request-2", completion_request2),
        ]
        batch_job = client.create_batch(batch_requests)
    """

    def __init__(self, custom_id: str, request: CompletionRequest) -> None:
        """Create a new batch request.

        Args:
            custom_id: Unique identifier for this request within the batch
            request: The completion request to execute
        """
        ...

    @property
    def custom_id(self) -> str:
        """The custom ID for this request."""
        ...

class BatchRequestCounts:
    """Request counts for a batch job."""

    @property
    def total(self) -> int:
        """Total number of requests in the batch."""
        ...

    @property
    def succeeded(self) -> int:
        """Number of successfully completed requests."""
        ...

    @property
    def failed(self) -> int:
        """Number of failed requests."""
        ...

    @property
    def pending(self) -> int:
        """Number of pending requests."""
        ...

class BatchJob:
    """A batch processing job.

    Contains information about the status and progress of a batch.
    """

    @property
    def id(self) -> str:
        """The batch ID."""
        ...

    @property
    def status(self) -> BatchStatus:
        """The current status of the batch."""
        ...

    @property
    def request_counts(self) -> BatchRequestCounts:
        """Request counts."""
        ...

    @property
    def created_at(self) -> Optional[str]:
        """When the batch was created (ISO 8601 timestamp)."""
        ...

    @property
    def started_at(self) -> Optional[str]:
        """When the batch started processing (ISO 8601 timestamp)."""
        ...

    @property
    def ended_at(self) -> Optional[str]:
        """When the batch finished processing (ISO 8601 timestamp)."""
        ...

    @property
    def expires_at(self) -> Optional[str]:
        """When the batch will expire (ISO 8601 timestamp)."""
        ...

    @property
    def error(self) -> Optional[str]:
        """Error message if the batch failed."""
        ...

    def is_complete(self) -> bool:
        """Check if the batch is complete (completed, failed, expired, or cancelled)."""
        ...

    def is_in_progress(self) -> bool:
        """Check if the batch is still in progress."""
        ...

class BatchError:
    """Error information for a failed batch request."""

    @property
    def error_type(self) -> str:
        """Error type."""
        ...

    @property
    def message(self) -> str:
        """Error message."""
        ...

class BatchResult:
    """Result of a single request within a batch."""

    @property
    def custom_id(self) -> str:
        """The custom ID of the original request."""
        ...

    @property
    def response(self) -> Optional[CompletionResponse]:
        """The completion response (if successful)."""
        ...

    @property
    def error(self) -> Optional[BatchError]:
        """The error (if failed)."""
        ...

    def is_success(self) -> bool:
        """Check if this result is successful."""
        ...

    def is_error(self) -> bool:
        """Check if this result is an error."""
        ...

# ==================== Embedding Types ====================

class EncodingFormat(IntEnum):
    """Output encoding format for embeddings."""

    Float = 0  # Float32 array (default)
    Base64 = 1  # Base64-encoded binary

class EmbeddingInputType(IntEnum):
    """Input type hint for embedding optimization."""

    Query = 0  # The input is a search query
    Document = 1  # The input is a document to be indexed

class EmbeddingRequest:
    """Request for generating embeddings.

    Example:
        # Single text
        request = EmbeddingRequest("text-embedding-3-small", "Hello, world!")

        # Batch
        request = EmbeddingRequest.batch(
            "text-embedding-3-small",
            ["Hello", "World", "How are you?"]
        )
    """

    def __init__(self, model: str, text: str) -> None:
        """Create a new embedding request for a single text.

        Args:
            model: The embedding model to use (e.g., "text-embedding-3-small")
            text: The text to embed
        """
        ...

    @staticmethod
    def batch(model: str, texts: List[str]) -> EmbeddingRequest:
        """Create a new embedding request for multiple texts (batch).

        Args:
            model: The embedding model to use
            texts: List of texts to embed
        """
        ...

    def with_dimensions(self, dimensions: int) -> EmbeddingRequest:
        """Set the output dimensions (for models that support dimension reduction).

        Args:
            dimensions: The number of dimensions for the output embedding

        Returns:
            Self for method chaining
        """
        ...

    def with_encoding_format(self, format: EncodingFormat) -> EmbeddingRequest:
        """Set the encoding format.

        Args:
            format: The encoding format (Float or Base64)

        Returns:
            Self for method chaining
        """
        ...

    def with_input_type(self, input_type: EmbeddingInputType) -> EmbeddingRequest:
        """Set the input type hint for optimized embeddings.

        Args:
            input_type: The input type (Query or Document)

        Returns:
            Self for method chaining
        """
        ...

    @property
    def model(self) -> str: ...
    @property
    def text_count(self) -> int:
        """Get the number of texts to embed."""
        ...

    def texts(self) -> List[str]:
        """Get all input texts as a list."""
        ...

    @property
    def dimensions(self) -> Optional[int]: ...

class Embedding:
    """A single embedding vector."""

    @property
    def index(self) -> int:
        """The index of this embedding in the batch."""
        ...

    @property
    def values(self) -> List[float]:
        """The embedding vector values."""
        ...

    @property
    def dimensions(self) -> int:
        """Get the number of dimensions."""
        ...

    def cosine_similarity(self, other: Embedding) -> float:
        """Compute cosine similarity with another embedding.

        Args:
            other: Another embedding to compare with

        Returns:
            Cosine similarity score (-1 to 1)
        """
        ...

    def dot_product(self, other: Embedding) -> float:
        """Compute dot product with another embedding.

        Args:
            other: Another embedding to compute dot product with

        Returns:
            Dot product value
        """
        ...

    def euclidean_distance(self, other: Embedding) -> float:
        """Compute Euclidean distance to another embedding.

        Args:
            other: Another embedding to compute distance to

        Returns:
            Euclidean distance
        """
        ...

    def __len__(self) -> int: ...

class EmbeddingUsage:
    """Token usage for embedding requests."""

    @property
    def prompt_tokens(self) -> int:
        """Number of tokens in the input."""
        ...

    @property
    def total_tokens(self) -> int:
        """Total tokens processed."""
        ...

class EmbeddingResponse:
    """Response from an embedding request."""

    @property
    def model(self) -> str:
        """The model used for embedding."""
        ...

    @property
    def embeddings(self) -> List[Embedding]:
        """The generated embeddings."""
        ...

    @property
    def usage(self) -> EmbeddingUsage:
        """Token usage information."""
        ...

    def first(self) -> Optional[Embedding]:
        """Get the first embedding (convenience for single-text requests)."""
        ...

    def values(self) -> Optional[List[float]]:
        """Get embedding values as a flat list (for single-text requests)."""
        ...

    @property
    def dimensions(self) -> int:
        """Get the embedding dimensions."""
        ...

    def __len__(self) -> int: ...

# ==================== Streaming Types ====================

class ContentDelta:
    """Delta content for streaming responses."""

    @property
    def text(self) -> Optional[str]: ...
    @property
    def thinking(self) -> Optional[str]: ...
    @property
    def is_text(self) -> bool: ...
    @property
    def is_tool_use(self) -> bool: ...
    @property
    def is_thinking(self) -> bool: ...
    def as_tool_use_delta(self) -> Optional[Dict[str, Any]]:
        """Get tool use delta details."""
        ...

class StreamChunk:
    """A chunk from a streaming response."""

    @property
    def event_type(self) -> StreamEventType: ...
    @property
    def index(self) -> Optional[int]: ...
    @property
    def delta(self) -> Optional[ContentDelta]: ...
    @property
    def text(self) -> Optional[str]: ...
    @property
    def stop_reason(self) -> Optional[StopReason]: ...
    @property
    def usage(self) -> Optional[Usage]: ...
    @property
    def is_done(self) -> bool: ...

class StreamIterator(Iterator[StreamChunk]):
    """Synchronous stream iterator."""

    def __iter__(self) -> StreamIterator: ...
    def __next__(self) -> StreamChunk: ...

class AsyncStreamIterator(AsyncIterator[StreamChunk]):
    """Asynchronous stream iterator."""

    def __aiter__(self) -> AsyncStreamIterator: ...
    async def __anext__(self) -> StreamChunk: ...

# ==================== Provider Configuration ====================

class ProviderConfig:
    """Configuration for a provider.

    Different providers require different configuration options:

    - Most providers: api_key only
    - Azure: api_key, endpoint, deployment (and optionally api_version)
    - AWS Bedrock: region (uses AWS credentials from environment)
    - Google Vertex: project, location (region)
    - Cloudflare: account_id, api_token
    - IBM WatsonX: api_key, project
    - Databricks: host, token
    - HuggingFace: api_key, endpoint_id (for endpoints)
    - Ollama: base_url only (defaults to localhost)
    """

    api_key: Optional[str]
    base_url: Optional[str]
    endpoint: Optional[str]
    deployment: Optional[str]
    region: Optional[str]
    project: Optional[str]
    location: Optional[str]
    account_id: Optional[str]
    api_token: Optional[str]
    access_token: Optional[str]
    host: Optional[str]
    token: Optional[str]
    endpoint_id: Optional[str]
    model_id: Optional[str]
    api_version: Optional[str]

# Type alias for provider configuration dictionary
ProviderConfigDict = Dict[str, Any]
"""Provider configuration dictionary. Can contain any of the fields from ProviderConfig."""

# Supported provider names
SUPPORTED_PROVIDERS: List[str] = [
    "anthropic",
    "openai",
    "azure",
    "bedrock",
    "vertex",
    "google",
    "groq",
    "mistral",
    "cohere",
    "deepseek",
    "openrouter",
    "ollama",
    "ai21",
    "cerebras",
    "fireworks",
    "sambanova",
    "huggingface",
    "replicate",
    "cloudflare",
    "databricks",
    "watsonx",
    "together",
    "perplexity",
    "xai",
    "lepton",
    "novita",
    "hyperbolic",
    "nebiusai",
    "writer",
    "gigachat",
    "yandex",
    "maritaca",
    "moonshot",
]

# ==================== Clients ====================

class LLMKitClient:
    """Synchronous LLMKit client.

    Example:
        # Using from_env() - auto-detects providers from environment
        client = LLMKitClient.from_env()

        # Using explicit provider configuration
        client = LLMKitClient(
            providers={
                "anthropic": {"api_key": "sk-..."},
                "openai": {"api_key": "sk-..."},
                "azure": {
                    "api_key": "...",
                    "endpoint": "https://myresource.openai.azure.com",
                    "deployment": "gpt-4"
                },
                "bedrock": {"region": "us-east-1"},
            },
            default_provider="anthropic"
        )
    """

    def __init__(
        self,
        providers: Optional[Dict[str, ProviderConfigDict]] = None,
        default_provider: Optional[str] = None,
    ) -> None:
        """Create a new LLMKit client.

        Args:
            providers: Dictionary mapping provider names to their configuration.
                Each provider config can contain: api_key, base_url, endpoint,
                deployment, region, project, location, account_id, api_token,
                access_token, host, token, endpoint_id, model_id, api_version.
            default_provider: Default provider to use when model doesn't specify one.
        """
        ...

    @staticmethod
    def from_env() -> LLMKitClient:
        """Create client from environment variables.

        Auto-detects all configured providers from environment variables:
        - ANTHROPIC_API_KEY for Anthropic
        - OPENAI_API_KEY for OpenAI
        - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT for Azure
        - AWS_REGION or AWS_DEFAULT_REGION for Bedrock
        - GOOGLE_CLOUD_PROJECT and VERTEX_REGION for Vertex AI
        - GROQ_API_KEY for Groq
        - MISTRAL_API_KEY for Mistral
        - And many more...
        """
        ...

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Make a completion request."""
        ...

    def complete_stream(self, request: CompletionRequest) -> StreamIterator:
        """Make a streaming completion request."""
        ...

    def complete_with_provider(
        self, provider_name: str, request: CompletionRequest
    ) -> CompletionResponse:
        """Make a completion request with a specific provider."""
        ...

    def providers(self) -> List[str]:
        """List all registered providers."""
        ...

    def count_tokens(self, request: TokenCountRequest) -> TokenCountResult:
        """Count tokens for a request.

        This allows estimation of token counts before making a completion request,
        useful for cost estimation and context window management.

        Note: Not all providers support token counting. Currently only Anthropic
        provides native token counting support.

        Args:
            request: TokenCountRequest with model, messages, optional system and tools

        Returns:
            TokenCountResult containing input_tokens count

        Raises:
            NotSupportedError: If the provider doesn't support token counting
        """
        ...
    # ==================== Batch Processing ====================
    def create_batch(self, requests: List[BatchRequest]) -> BatchJob:
        """Create a batch processing job.

        Submits multiple completion requests to be processed asynchronously.
        Returns a BatchJob that can be used to track progress and retrieve results.

        Args:
            requests: List of BatchRequest objects

        Returns:
            BatchJob containing batch ID and initial status
        """
        ...

    def get_batch(self, provider_name: str, batch_id: str) -> BatchJob:
        """Get the status of a batch job.

        Args:
            provider_name: The provider that created the batch
            batch_id: The batch ID

        Returns:
            BatchJob with current status and progress
        """
        ...

    def get_batch_results(self, provider_name: str, batch_id: str) -> List[BatchResult]:
        """Get the results of a completed batch.

        Args:
            provider_name: The provider that created the batch
            batch_id: The batch ID

        Returns:
            List of BatchResult objects with responses or errors
        """
        ...

    def cancel_batch(self, provider_name: str, batch_id: str) -> BatchJob:
        """Cancel a batch job.

        Args:
            provider_name: The provider that created the batch
            batch_id: The batch ID

        Returns:
            BatchJob with updated status
        """
        ...

    def list_batches(self, provider_name: str, limit: Optional[int] = None) -> List[BatchJob]:
        """List batch jobs for a provider.

        Args:
            provider_name: The provider to list batches for
            limit: Maximum number of batches to return (optional)

        Returns:
            List of BatchJob objects
        """
        ...
    # ==================== Embeddings ====================
    def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for text.

        Creates vector representations of text that can be used for semantic search,
        clustering, classification, and other NLP tasks.

        Note: Not all providers support embeddings. Currently OpenAI and Cohere
        support this feature.

        Args:
            request: EmbeddingRequest with model and text(s) to embed

        Returns:
            EmbeddingResponse containing embedding vectors and usage info

        Raises:
            NotSupportedError: If no embedding provider is configured
        """
        ...

    def embed_with_provider(
        self, provider_name: str, request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """Generate embeddings with a specific provider.

        Args:
            provider_name: Name of the embedding provider (e.g., "openai", "cohere")
            request: EmbeddingRequest with model and text(s) to embed

        Returns:
            EmbeddingResponse containing embedding vectors and usage info

        Raises:
            ProviderNotFoundError: If the provider is not configured for embeddings
        """
        ...

    def embedding_providers(self) -> List[str]:
        """List all registered embedding providers.

        Returns:
            Names of providers that support embeddings
        """
        ...

    def supports_embeddings(self, provider_name: str) -> bool:
        """Check if a provider supports embeddings.

        Args:
            provider_name: Name of the provider to check

        Returns:
            True if the provider supports embeddings
        """
        ...

    @property
    def default_provider(self) -> Optional[str]: ...

class AsyncLLMKitClient:
    """Asynchronous LLMKit client.

    Example:
        # Using from_env() - auto-detects providers from environment
        client = AsyncLLMKitClient.from_env()

        # Using explicit provider configuration
        client = AsyncLLMKitClient(
            providers={
                "anthropic": {"api_key": "sk-..."},
                "openai": {"api_key": "sk-..."},
            }
        )

        response = await client.complete(request)
    """

    def __init__(
        self,
        providers: Optional[Dict[str, ProviderConfigDict]] = None,
        default_provider: Optional[str] = None,
    ) -> None:
        """Create a new async LLMKit client.

        Args:
            providers: Dictionary mapping provider names to their configuration.
            default_provider: Default provider to use when model doesn't specify one.
        """
        ...

    @staticmethod
    def from_env() -> AsyncLLMKitClient:
        """Create client from environment variables.

        Auto-detects all configured providers from environment variables.
        See LLMKitClient.from_env() for the full list of supported variables.
        """
        ...

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Make a completion request."""
        ...

    async def complete_stream(self, request: CompletionRequest) -> AsyncStreamIterator:
        """Make a streaming completion request."""
        ...

    async def complete_with_provider(
        self, provider_name: str, request: CompletionRequest
    ) -> CompletionResponse:
        """Make a completion request with a specific provider."""
        ...

    def providers(self) -> List[str]:
        """List all registered providers."""
        ...

    async def count_tokens(self, request: TokenCountRequest) -> TokenCountResult:
        """Count tokens for a request (async).

        This allows estimation of token counts before making a completion request,
        useful for cost estimation and context window management.

        Note: Not all providers support token counting. Currently only Anthropic
        provides native token counting support.

        Args:
            request: TokenCountRequest with model, messages, optional system and tools

        Returns:
            TokenCountResult containing input_tokens count

        Raises:
            NotSupportedError: If the provider doesn't support token counting
        """
        ...
    # ==================== Batch Processing ====================
    async def create_batch(self, requests: List[BatchRequest]) -> BatchJob:
        """Create a batch processing job (async).

        Submits multiple completion requests to be processed asynchronously.
        Returns a BatchJob that can be used to track progress and retrieve results.

        Args:
            requests: List of BatchRequest objects

        Returns:
            BatchJob containing batch ID and initial status
        """
        ...

    async def get_batch(self, provider_name: str, batch_id: str) -> BatchJob:
        """Get the status of a batch job (async).

        Args:
            provider_name: The provider that created the batch
            batch_id: The batch ID

        Returns:
            BatchJob with current status and progress
        """
        ...

    async def get_batch_results(self, provider_name: str, batch_id: str) -> List[BatchResult]:
        """Get the results of a completed batch (async).

        Args:
            provider_name: The provider that created the batch
            batch_id: The batch ID

        Returns:
            List of BatchResult objects with responses or errors
        """
        ...

    async def cancel_batch(self, provider_name: str, batch_id: str) -> BatchJob:
        """Cancel a batch job (async).

        Args:
            provider_name: The provider that created the batch
            batch_id: The batch ID

        Returns:
            BatchJob with updated status
        """
        ...

    async def list_batches(self, provider_name: str, limit: Optional[int] = None) -> List[BatchJob]:
        """List batch jobs for a provider (async).

        Args:
            provider_name: The provider to list batches for
            limit: Maximum number of batches to return (optional)

        Returns:
            List of BatchJob objects
        """
        ...
    # ==================== Embeddings ====================
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Generate embeddings for text (async).

        Creates vector representations of text that can be used for semantic search,
        clustering, classification, and other NLP tasks.

        Note: Not all providers support embeddings. Currently OpenAI and Cohere
        support this feature.

        Args:
            request: EmbeddingRequest with model and text(s) to embed

        Returns:
            EmbeddingResponse containing embedding vectors and usage info

        Raises:
            NotSupportedError: If no embedding provider is configured
        """
        ...

    async def embed_with_provider(
        self, provider_name: str, request: EmbeddingRequest
    ) -> EmbeddingResponse:
        """Generate embeddings with a specific provider (async).

        Args:
            provider_name: Name of the embedding provider (e.g., "openai", "cohere")
            request: EmbeddingRequest with model and text(s) to embed

        Returns:
            EmbeddingResponse containing embedding vectors and usage info

        Raises:
            ProviderNotFoundError: If the provider is not configured for embeddings
        """
        ...

    def embedding_providers(self) -> List[str]:
        """List all registered embedding providers.

        Returns:
            Names of providers that support embeddings
        """
        ...

    def supports_embeddings(self, provider_name: str) -> bool:
        """Check if a provider supports embeddings.

        Args:
            provider_name: Name of the provider to check

        Returns:
            True if the provider supports embeddings
        """
        ...

    @property
    def default_provider(self) -> Optional[str]: ...

# ==================== Model Registry Types ====================

class Provider(IntEnum):
    """LLM Provider identifier."""

    Anthropic = 0
    OpenAI = 1
    Google = 2
    Mistral = 3
    Groq = 4
    DeepSeek = 5
    Cohere = 6
    Bedrock = 7
    AzureOpenAI = 8
    VertexAI = 9
    TogetherAI = 10
    OpenRouter = 11
    Cerebras = 12
    SambaNova = 13
    Fireworks = 14
    AI21 = 15
    HuggingFace = 16
    Replicate = 17
    Cloudflare = 18
    Databricks = 19
    Writer = 20
    Maritaca = 21
    Clova = 22
    Yandex = 23
    GigaChat = 24
    Upstage = 25
    SeaLion = 26
    Local = 27
    Custom = 28

class ModelStatus(IntEnum):
    """Model availability status."""

    Current = 0  # Currently recommended model
    Legacy = 1  # Still available but superseded by newer version
    Deprecated = 2  # Scheduled for removal

class ModelPricing:
    """Model pricing (per 1M tokens in USD)."""

    @property
    def input_per_1m(self) -> float:
        """Input token price per 1M tokens."""
        ...

    @property
    def output_per_1m(self) -> float:
        """Output token price per 1M tokens."""
        ...

    @property
    def cached_input_per_1m(self) -> Optional[float]:
        """Cached input token price per 1M tokens (if supported)."""
        ...

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for given token counts."""
        ...

class ModelCapabilities:
    """Model capabilities."""

    @property
    def max_context(self) -> int:
        """Maximum input context size in tokens."""
        ...

    @property
    def max_output(self) -> int:
        """Maximum output tokens."""
        ...

    @property
    def vision(self) -> bool:
        """Supports vision/image input."""
        ...

    @property
    def tools(self) -> bool:
        """Supports tool/function calling."""
        ...

    @property
    def streaming(self) -> bool:
        """Supports streaming responses."""
        ...

    @property
    def json_mode(self) -> bool:
        """Supports JSON mode."""
        ...

    @property
    def structured_output(self) -> bool:
        """Supports structured output with JSON schema enforcement."""
        ...

    @property
    def thinking(self) -> bool:
        """Supports extended thinking/reasoning."""
        ...

    @property
    def caching(self) -> bool:
        """Supports prompt caching."""
        ...

class ModelBenchmarks:
    """Benchmark scores (0-100 scale, higher is better)."""

    @property
    def mmlu(self) -> Optional[float]:
        """MMLU - General knowledge."""
        ...

    @property
    def humaneval(self) -> Optional[float]:
        """HumanEval - Code generation."""
        ...

    @property
    def math(self) -> Optional[float]:
        """MATH - Mathematical reasoning."""
        ...

    @property
    def gpqa(self) -> Optional[float]:
        """GPQA Diamond - Graduate-level science."""
        ...

    @property
    def swe_bench(self) -> Optional[float]:
        """SWE-bench - Software engineering."""
        ...

    @property
    def ifeval(self) -> Optional[float]:
        """IFEval - Instruction following."""
        ...

    @property
    def mmmu(self) -> Optional[float]:
        """MMMU - Multimodal understanding."""
        ...

    @property
    def mgsm(self) -> Optional[float]:
        """MGSM - Multilingual math."""
        ...

    @property
    def ttft_ms(self) -> Optional[int]:
        """Time to first token (ms)."""
        ...

    @property
    def tokens_per_sec(self) -> Optional[int]:
        """Tokens per second."""
        ...

    def quality_score(self) -> float:
        """Calculate weighted quality score (0-100)."""
        ...

class RegistryStats:
    """Registry statistics."""

    @property
    def total_models(self) -> int:
        """Total number of models in the registry."""
        ...

    @property
    def current_models(self) -> int:
        """Number of current (non-deprecated) models."""
        ...

    @property
    def providers(self) -> int:
        """Number of providers."""
        ...

    @property
    def available_models(self) -> int:
        """Number of models available (API key configured)."""
        ...

class ModelInfo:
    """Complete model specification."""

    @property
    def id(self) -> str:
        """Unified model ID (e.g., "anthropic/claude-3-5-sonnet")."""
        ...

    @property
    def alias(self) -> Optional[str]:
        """Short alias (e.g., "claude-3-5-sonnet")."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name."""
        ...

    @property
    def provider(self) -> Provider:
        """Provider."""
        ...

    @property
    def status(self) -> ModelStatus:
        """Model status."""
        ...

    @property
    def pricing(self) -> ModelPricing:
        """Pricing information."""
        ...

    @property
    def capabilities(self) -> ModelCapabilities:
        """Model capabilities."""
        ...

    @property
    def benchmarks(self) -> ModelBenchmarks:
        """Benchmark scores."""
        ...

    @property
    def description(self) -> str:
        """Model description."""
        ...

    @property
    def can_classify(self) -> bool:
        """Whether the model can be used as a classifier."""
        ...

    def raw_id(self) -> str:
        """Get the raw model ID without provider prefix."""
        ...

    def quality_per_dollar(self) -> float:
        """Calculate quality per dollar (higher is better value)."""
        ...

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        ...

    def quality_score(self) -> float:
        """Calculate weighted quality score from benchmarks (0-100).

        Convenience method equivalent to `model.benchmarks.quality_score()`.
        """
        ...

# ==================== Model Registry Functions ====================

def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Get model info by ID, alias, or raw ID.

    Args:
        model_id: Model identifier (e.g., "claude-sonnet-4-5", "gpt-4o")

    Returns:
        ModelInfo if found, None otherwise.

    Example:
        >>> info = get_model_info("claude-sonnet-4-5")
        >>> if info:
        ...     print(f"{info.name}: ${info.pricing.input_per_1m}/1M tokens")
    """
    ...

def get_all_models() -> List[ModelInfo]:
    """Get all models in the registry.

    Returns:
        List of all ModelInfo objects.
    """
    ...

def get_models_by_provider(provider: Provider) -> List[ModelInfo]:
    """Get all models for a specific provider.

    Args:
        provider: Provider enum value.

    Returns:
        List of ModelInfo objects for the provider.
    """
    ...

def get_current_models() -> List[ModelInfo]:
    """Get all current (non-deprecated) models.

    Returns:
        List of current ModelInfo objects.
    """
    ...

def get_classifier_models() -> List[ModelInfo]:
    """Get models that can be used as classifiers.

    Returns:
        List of classifier-suitable ModelInfo objects.
    """
    ...

def get_available_models() -> List[ModelInfo]:
    """Get available models (provider API key is configured).

    Returns:
        List of available ModelInfo objects.
    """
    ...

def get_models_with_capability(
    vision: Optional[bool] = None,
    tools: Optional[bool] = None,
    thinking: Optional[bool] = None,
) -> List[ModelInfo]:
    """Get models with specific capabilities.

    Args:
        vision: Filter by vision support (None to ignore).
        tools: Filter by tool calling support (None to ignore).
        thinking: Filter by extended thinking support (None to ignore).

    Returns:
        List of matching ModelInfo objects.
    """
    ...

def get_cheapest_model(
    min_context: Optional[int] = None,
    needs_vision: bool = False,
    needs_tools: bool = False,
) -> Optional[ModelInfo]:
    """Get the cheapest model that meets requirements.

    Args:
        min_context: Minimum context window size (None for any).
        needs_vision: Whether vision support is required.
        needs_tools: Whether tool calling support is required.

    Returns:
        Cheapest ModelInfo if found, None otherwise.
    """
    ...

def supports_structured_output(model_id: str) -> bool:
    """Check if a model supports structured output (JSON schema enforcement).

    Args:
        model_id: Model identifier.

    Returns:
        True if the model supports structured output.
    """
    ...

def get_registry_stats() -> RegistryStats:
    """Get registry statistics.

    Returns:
        RegistryStats with counts of models and providers.
    """
    ...

def list_providers() -> List[Provider]:
    """List all providers with at least one model.

    Returns:
        List of Provider enum values.
    """
    ...
