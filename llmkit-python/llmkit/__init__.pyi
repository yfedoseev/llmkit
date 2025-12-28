"""Type stubs for llmkit - Unified LLM API client library."""

from typing import Optional, List, Dict, Any, Tuple, Iterator, AsyncIterator, Union
from enum import IntEnum

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

    def array_param(self, name: str, description: str, item_type: str, required: bool = True) -> ToolBuilder:
        """Add an array parameter."""
        ...

    def enum_param(self, name: str, description: str, values: List[str], required: bool = True) -> ToolBuilder:
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

# ==================== Clients ====================

class LLMKitClient:
    """Synchronous LLMKit client."""

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
        default_provider: Optional[str] = None,
    ) -> None: ...

    @staticmethod
    def from_env() -> LLMKitClient:
        """Create client from environment variables."""
        ...

    def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Make a completion request."""
        ...

    def complete_stream(self, request: CompletionRequest) -> StreamIterator:
        """Make a streaming completion request."""
        ...

    def complete_with_provider(self, provider_name: str, request: CompletionRequest) -> CompletionResponse:
        """Make a completion request with a specific provider."""
        ...

    def providers(self) -> List[str]:
        """List all registered providers."""
        ...

    @property
    def default_provider(self) -> Optional[str]: ...

class AsyncLLMKitClient:
    """Asynchronous LLMKit client."""

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        groq_api_key: Optional[str] = None,
        mistral_api_key: Optional[str] = None,
        default_provider: Optional[str] = None,
    ) -> None: ...

    @staticmethod
    def from_env() -> AsyncLLMKitClient:
        """Create client from environment variables."""
        ...

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """Make a completion request."""
        ...

    async def complete_stream(self, request: CompletionRequest) -> AsyncStreamIterator:
        """Make a streaming completion request."""
        ...

    async def complete_with_provider(self, provider_name: str, request: CompletionRequest) -> CompletionResponse:
        """Make a completion request with a specific provider."""
        ...

    def providers(self) -> List[str]:
        """List all registered providers."""
        ...

    @property
    def default_provider(self) -> Optional[str]: ...
