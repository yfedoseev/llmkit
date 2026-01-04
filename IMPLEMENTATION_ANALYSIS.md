# Implementation Analysis: Top Uncovered Providers for LLMKit

**Document**: Technical feasibility and implementation roadmap for integrating uncovered LLM providers

---

## Tier 1: High ROI, Feasible Implementations (Phase 4 Candidates)

### 1. Exa AI Search Integration

**Current Status**: Not in LLMKit or LiteLLM

**API Specification**:
```
Endpoint: https://api.exa.ai/search
Method: POST
Authentication: API Key (header: Authorization: Bearer {api_key})
Content-Type: application/json

Request Body:
{
  "query": "What are the latest developments in AI",
  "numResults": 10,
  "searchType": "auto", // Options: "auto", "neural", "fast", "deep"
  "includeText": true,
  "type": "search",
  "startPublishedDate": "2024-01-01",
  "startRecency": "day"
}

Response:
{
  "results": [
    {
      "title": "...",
      "url": "...",
      "publishedDate": "...",
      "author": "...",
      "text": "..."
    }
  ]
}
```

**LLMKit Integration Approach**:
```rust
pub struct ExaProvider {
    api_key: String,
    client: reqwest::Client,
}

impl ExaProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }

    pub async fn search(&self, query: &str) -> Result<Vec<SearchResult>> {
        // Implement search endpoint
    }
}
```

**Implementation Effort**: LOW (2-3 days)
- Simple REST endpoint
- No complex model handling
- Straightforward serialization

**Integration Points**:
- Add `exa` feature gate to `Cargo.toml`
- Create `/src/providers/exa.rs`
- Add to `mod.rs` with feature gate
- No changes to core `Provider` trait needed
- Could expose via new `SearchProvider` trait

**Pricing Consideration**:
- Free tier: 1000 searches/month sufficient for testing
- Per-search billing after free tier

**When to Integrate**: Phase 4, Wave 1

---

### 2. Brave Search API + MCP Support

**Current Status**: Not in LLMKit or LiteLLM

**API Specification**:
```
Endpoint: https://api.search.brave.com/res/v1/web/search
Method: GET
Authentication: API Key (header: Accept-Encoding)
Query Parameters:
  - q: search query
  - count: results count (max 20)
  - result_filter: "web", "news", "videos"
  - offset: pagination
  - summary: true (AI summary)

Response:
{
  "web": {
    "results": [
      {
        "title": "...",
        "url": "...",
        "description": "...",
        "page_age": "...",
        "profile": { "img": "...", "name": "..." }
      }
    ]
  },
  "summary": "..."
}
```

**MCP Support**:
- Brave offers Model Context Protocol (MCP) server
- Would allow standardized tool integration
- LLMKit could expose as optional MCP adapter

**LLMKit Integration Approach**:
```rust
pub struct BraveSearchProvider {
    api_key: String,
    client: reqwest::Client,
}

// Standard search
impl BraveSearchProvider {
    pub async fn search(&self, query: &str) -> Result<SearchResults> { }
    pub async fn search_with_summary(&self, query: &str) -> Result<SearchSummary> { }
}

// MCP adapter (optional)
pub struct BraveSearchMCP {
    provider: BraveSearchProvider,
}
```

**Implementation Effort**: LOW-MEDIUM (3-4 days)
- Simple REST endpoint (reuse Exa code structure)
- Optional MCP adapter adds complexity
- MCP support emerging as standard

**Integration Points**:
- Add `brave-search` feature gate
- Create `/src/providers/brave_search.rs`
- Optional: Create `/src/mcp_adapters/brave_search_mcp.rs`

**Competitive Advantage**:
- Privacy-first positioning
- MCP standardization alignment
- Currently used by Anthropic, Cursor, Cline

**When to Integrate**: Phase 4, Wave 1 (after Exa)

---

### 3. OpenAI Realtime API (Voice Streaming)

**Current Status**: OpenAI provider covers standard API, NOT Realtime

**API Specification**:
```
WebSocket Connection: wss://api.openai.com/v1/realtime
Authentication: Bearer {api_key} in URL
Subprotocol: realtime

Events (bidirectional):
- session.update
- session.created
- input_audio_buffer.append
- input_audio_buffer.commit
- input_audio_buffer.clear
- conversation.item.create
- response.create
- response.update
- response.done
- error

Configuration:
{
  "type": "session.update",
  "session": {
    "modalities": ["text", "audio"],
    "instructions": "You are a helpful assistant.",
    "voice": "alloy",
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "model": "gpt-4o-realtime-preview"
  }
}
```

**LLMKit Integration Approach**:
```rust
pub struct OpenAIRealtimeProvider {
    api_key: String,
    base_url: String,
    // WebSocket handling
    ws_client: Option<tokio_tungstenite::WebSocketStream>,
}

#[async_trait]
impl Provider for OpenAIRealtimeProvider {
    // Existing trait methods need WebSocket variants
    async fn complete_stream_realtime(
        &self,
        request: RealtimeRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<RealtimeChunk>> + Send>>> {
        // WebSocket streaming implementation
    }
}

pub struct RealtimeRequest {
    pub modalities: Vec<Modality>, // ["text", "audio"]
    pub instructions: String,
    pub voice: String, // "alloy", "echo", "shimmer"
    pub audio_format: AudioFormat,
    pub messages: Vec<RealtimeMessage>,
}

pub struct RealtimeChunk {
    pub event_type: String,
    pub delta: Option<String>, // For text
    pub audio_data: Option<Vec<u8>>, // For audio
}
```

**Implementation Effort**: MEDIUM (5-7 days)
- WebSocket protocol (new for LLMKit)
- Bidirectional streaming (different from SSE)
- Audio buffer management
- Error handling for connection

**Integration Points**:
- Extend OpenAI provider (don't duplicate)
- Add WebSocket support to core streaming types
- New `Provider` trait extensions for realtime (or new trait)
- Could use `tokio-tungstenite` or `ws-rs`

**Protocol Differences**:
- Standard API: HTTP/SSE streaming
- Realtime API: WebSocket bidirectional
- Need separate streaming implementation

**When to Integrate**: Phase 4, Wave 2 (more complex)

---

### 4. Chinese Regional Providers Bundle

**A. Moonshot/Kimi**

**API Specification**:
```
Endpoint: https://api.moonshot.ai/v1/chat/completions
Method: POST
Authentication: Bearer {api_key}
Content-Type: application/json

NOT OpenAI-compatible, but similar structure:
{
  "model": "moonshot-v1-8k", // or 32k, 128k
  "messages": [
    {
      "role": "user",
      "content": "..."
    }
  ],
  "temperature": 0.3,
  "max_tokens": 1024,
  "top_p": 1
}

Response:
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 100,
    "completion_tokens": 50,
    "total_tokens": 150
  }
}
```

**B. Baidu ERNIE (Qianfan)**

**API Specification**:
```
Endpoint: https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie_4_0_8k
Method: POST
Authentication: Access token (obtained via OAuth)
Content-Type: application/json

{
  "messages": [
    {
      "role": "user",
      "content": "..."
    }
  ],
  "temperature": 0.5,
  "top_p": 0.8,
  "max_output_tokens": 1024,
  "penalty_score": 1.0
}

Response: Similar to OpenAI structure
```

**C. Baichuan**

**API Specification**:
```
Endpoint: https://api.baichuan-ai.com/v1/chat/completions
Method: POST
Authentication: Bearer {api_key}
OpenAI-compatible!

Same as OpenAI request/response format
```

**LLMKit Integration Approach**:

```rust
// For OpenAI-compatible (Baichuan)
// Can use existing openai-compatible provider

// For non-compatible (Moonshot, ERNIE)
pub struct MoonshotProvider {
    api_key: String,
    client: reqwest::Client,
}

pub struct BaiduErnieProvider {
    api_key: String,
    secret_key: String, // Requires OAuth
    client: reqwest::Client,
}

// Common handling
#[async_trait]
impl Provider for MoonshotProvider {
    async fn complete(&self, request: CompletionRequest) -> Result<CompletionResponse> {
        let moonshot_request = convert_to_moonshot_format(&request);
        let response = self.client.post(endpoint).json(&moonshot_request).send().await?;
        convert_from_moonshot_format(&response)
    }
}
```

**Implementation Effort**: LOW (2-3 days per provider)
- API formats are similar (mostly JSON differences)
- Standard REST endpoints
- No streaming complexity
- Separate implementations needed due to format differences

**Integration Points**:
- Add `moonshot`, `baidu-ernie`, (skip `baichuan` - use openai-compatible)
- Create `/src/providers/moonshot.rs`
- Create `/src/providers/baidu_ernie.rs`
- Each ~200-300 lines of code

**Considerations**:
- Moonshot: Requires China-based API access
- ERNIE: Requires Baidu OAuth setup (more complex)
- Baichuan: Already covered via openai-compatible

**Market**:
- Moonshot/Kimi: Leading Chinese startup (valued at $2.5B)
- Baidu ERNIE: Largest market in China
- Combined capture growing China enterprise segment

**When to Integrate**: Phase 4, Wave 1-2 (if China market targeted)

---

## Tier 2: Medium Priority, More Complex

### NVIDIA NIM (Enterprise Self-Hosted)

**Current Status**: Not in LLMKit (complex infrastructure play)

**What NIM is**:
- Docker containers with optimized LLM inference
- OpenAI-compatible API + NVIDIA extensions
- Enterprise on-premise deployment

**Technical Challenge**:
- Not an API provider (it's infrastructure)
- Users deploy NIM themselves
- Can access via OpenAI-compatible provider (pointing to localhost:8000)

**LLMKit Integration Approach**:
```rust
// Could add as variant of openai-compatible
pub struct NimProvider {
    base_url: String, // e.g., "http://localhost:8000"
    // Uses standard OpenAI protocol
}

// Or detect NIM automatically
impl OpenAICompatibleProvider {
    pub fn from_nim(base_url: String) -> Self {
        // Detect and configure for NIM specifically
        // Handle NVIDIA-specific extensions if needed
    }
}
```

**Implementation Effort**: LOW (if using openai-compatible)
- Just document how to deploy NIM
- No code changes needed
- Base URL configuration

**When to Integrate**: Documentation/Examples only (Phase 4)

---

### Portkey AI (Enterprise Multi-Provider Gateway)

**Current Status**: Not in LLMKit (orchestration platform)

**What Portkey is**:
- Multi-provider routing/orchestration platform
- Manages 1600+ LLMs
- Dynamic routing, fallback, caching

**Technical Challenge**:
- Not an LLM provider, but orchestration layer
- Portkey exposes OpenAI-compatible API
- LLMKit could route through Portkey

**LLMKit Integration Approach**:
```rust
// Portkey is essentially an OpenAI-compatible provider
pub struct PortkeyProvider {
    base_url: String, // portkey.ai
    api_key: String, // Portkey key
    // Additional: Portkey-specific routing config
    config: PortkeyConfig,
}

pub struct PortkeyConfig {
    pub fallback_order: Vec<String>, // Provider priorities
    pub cache_enabled: bool,
    pub rate_limit_enabled: bool,
}
```

**Implementation Effort**: MEDIUM
- Basic: 50 lines (just use openai-compatible)
- Advanced: Add Portkey-specific routing configuration

**When to Integrate**: Phase 5 (lower priority)

---

### AssemblyAI LLM Gateway

**Current Status**: Deepgram covered, but not AssemblyAI LLM Gateway

**What it is**:
- Speech-to-text transcription
- Built-in LLM routing (OpenAI, Anthropic, Google, Mistral)
- Unified API

**Technical Challenge**:
- Two components: STT + LLM
- Could model as specialized workflow provider

**When to Integrate**: Phase 5 (specialized use case)

---

## Tier 3: Skip (Already Covered or Low Value)

### Why NOT to Add:

**vLLM / TGI / Ollama**:
- Already available via openai-compatible provider
- Users deploy themselves
- Adding native support adds complexity with little benefit

**Vector Databases**:
- Not LLM providers (Pinecone, Weaviate, Qdrant, Chroma)
- LLMKit correctly focuses on LLM generation
- These are infrastructure for RAG

**Search/Tool APIs**:
- Brave + Exa recommend: Include in Phase 4
- Skip: Metaphor (less differentiated), Tavily (commercial)

**Together AI / Anyscale**:
- Verify: Likely already covered via openai-compatible
- Not worth separate integration unless major gaps

**GitHub Copilot Chat API**:
- Not yet public
- Reverse-engineered wrappers unreliable
- Defer until official public API

**Phi Models / IBM Granite**:
- Can deploy via Azure (Bedrock) or self-hosted
- Direct integration lower priority

---

## Implementation Roadmap Recommendation

### Phase 4 (Q1 2026 - Recommended)

**Wave 1** (Parallel, 2 weeks):
1. Exa AI Search (LOW effort) ✅
2. Brave Search API (LOW effort) ✅

**Wave 2** (Parallel, 3 weeks):
3. OpenAI Realtime API (MEDIUM effort) ✅
4. Chinese Regional Providers (Moonshot + ERNIE) (LOW effort each) ✅

**Estimated Total**: 6-7 weeks for 4 meaningful provider additions

### Phase 5 (Q2 2026 - Optional)

- Portkey AI (orchestration)
- AssemblyAI LLM Gateway
- Advanced Ray Serve integration
- Documentation/examples for NVIDIA NIM

---

## Architecture Considerations

### New Provider Traits Needed

**For Search Providers**:
```rust
pub trait SearchProvider: Send + Sync {
    fn name(&self) -> &str;
    async fn search(&self, query: &str) -> Result<Vec<SearchResult>>;
    async fn search_with_summary(&self, query: &str) -> Result<SearchSummary>;
}

pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
    pub published_date: Option<DateTime<Utc>>,
}
```

**For Voice/Realtime**:
```rust
#[async_trait]
pub trait Provider {
    // Existing methods...

    // New optional methods for realtime
    async fn complete_realtime(
        &self,
        request: RealtimeRequest,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<RealtimeChunk>> + Send>>>;

    fn supports_realtime(&self) -> bool {
        false
    }
}
```

### Feature Organization

```toml
[features]
# Existing
default = ["anthropic", "openai"]

# New search providers
exa-search = []
brave-search = []

# New voice/realtime
openai-realtime = []  # Extends openai feature
anthropic-realtime = [] # If Anthropic adds realtime

# Regional providers
moonshot = []
baidu-ernie = []

# Bundle flags
search-providers = ["exa-search", "brave-search"]
realtime-providers = ["openai-realtime"]
chinese-providers = ["moonshot", "baidu-ernie"]
```

---

## Testing Strategy

### Unit Tests
- Mock HTTP responses
- Test request/response serialization
- Test error handling

### Integration Tests
- Real API calls (optional, with feature gate)
- Require valid API keys (environment variables)
- Rate limit aware

### Example Code
- Simple search query
- Voice streaming example
- Chinese provider example

---

## Documentation Updates

### New Pages Needed
- `/docs/providers/search-apis.md` (Exa, Brave)
- `/docs/providers/voice-realtime.md` (OpenAI Realtime)
- `/docs/providers/chinese-providers.md` (Moonshot, ERNIE)
- `/docs/deployment/nim.md` (NVIDIA NIM setup guide)
- `/docs/deployment/portkey.md` (Portkey routing)

### Existing Updates
- Update provider count (41 → 45+)
- Feature matrix with new capabilities

---

## Conclusion

**Recommended Phase 4 Implementation** (in order):

1. **Exa AI** - High ROI, enables RAG/agent use cases
2. **Brave Search** - Privacy positioning, MCP alignment
3. **OpenAI Realtime** - Emerging voice use case category
4. **Chinese Providers** - If targeting that market (Moonshot/ERNIE)

**Expected Timeline**: 6-8 weeks implementation
**Expected Users Benefitted**: Significant (especially agents/voice builders)
**Maintenance Burden**: Low (straightforward APIs)

This roadmap balances new capabilities with implementation complexity.
