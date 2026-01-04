# Comprehensive Research: LLM Providers NOT Currently Supported by LiteLLM or LLMKit

**Research Date**: January 2, 2026
**Current Status**:
- LiteLLM: ~100+ providers (with multiple models per provider)
- LLMKit: 41 implemented providers (Phase 3 complete)

---

## Executive Summary

After comprehensive research across all major LLM categories, this document identifies genuine gaps in LLMKit/LiteLLM coverage. These are providers that either:
1. Offer unique capabilities not available through existing integrations
2. Have significant market share or enterprise demand
3. Provide specialized/niche services worth integrating

**Key Finding**: Most gaps fall into 3 categories:
- **Specialized Platforms** (search APIs, voice, vector DBs with generation)
- **Chinese Regional Providers** (different from OpenAI-compatible wrappers)
- **Enterprise/MLOps Platforms** (serverless, deployment-specific)
- **Multi-Provider Aggregators** (alternative routing/orchestration)

---

## VERIFIED LLMKit Coverage (41 Providers)

Based on `/home/yfedoseev/projects/modelsuite/src/providers/` and `Cargo.toml`:

### Core (2)
- ✅ Anthropic
- ✅ OpenAI

### Major Cloud (5)
- ✅ Azure OpenAI
- ✅ AWS Bedrock
- ✅ Google (Gemini)
- ✅ Vertex AI
- ✅ Cloudflare Workers AI

### Specialized Fast Inference (6)
- ✅ Groq
- ✅ Mistral
- ✅ Cerebras
- ✅ SambaNova
- ✅ Fireworks
- ✅ DeepSeek

### Enterprise (4)
- ✅ Cohere
- ✅ AI21 Labs
- ✅ Databricks
- ✅ WatsonX (IBM)

### Local/Self-Hosted (1)
- ✅ Ollama
- (Note: vLLM, LM Studio, Llamafile, TGI covered via openai-compatible)

### OpenAI-Compatible Proxy (15+)
- ✅ OpenRouter
- ✅ OpenAI-compatible (generic)
- Covers: xAI/Grok, Meta Llama API, Lambda Labs, Friendli, Volcengine, etc.

### Inference Platforms (4)
- ✅ HuggingFace
- ✅ Replicate
- ✅ Baseten
- ✅ RunPod

### Specialized Services (4)
- ✅ Stability (image generation)
- ✅ Voyage (embeddings)
- ✅ Jina (embeddings/search)
- ✅ FAL (media/generation)

### Audio/Speech (2)
- ✅ Deepgram
- ✅ ElevenLabs

### Regional (4)
- ✅ YandexGPT
- ✅ GigaChat (Russia)
- ✅ Clova (Korea/Naver)
- ✅ Maritaca (Brazil)

### Enterprise (1)
- ✅ Writer

### AWS/Cloud (3)
- ✅ SageMaker
- ✅ DataRobot
- ✅ Snowflake

---

## UNCOVERED PROVIDERS: Tier 1 (High Priority)

### 1. **Portkey AI** - Multi-Provider Gateway/Abstraction

**Status**: NOT Covered by LiteLLM or LLMKit
**Category**: Enterprise AI Gateway

**What it does**:
- Unified abstraction layer for 1600+ LLMs across providers
- Dynamic routing, fallback, and load balancing
- Single standardized API for: OpenAI, Anthropic, Mistral, Google, xAI, custom models

**Key Differences**:
- Not just an API proxy, but orchestration layer with:
  - Token normalization across providers
  - Dynamic model routing based on cost/performance/latency
  - Centralized logging, rate limiting, guardrails
  - Support for custom/fine-tuned models

**API Type**: REST (OpenAI-compatible)
**Authentication**: API Key
**Unique Value**: Complete orchestration platform vs. simple proxy

**Why Add**: Enterprises using Portkey as infrastructure layer would benefit from native support

---

### 2. **NVIDIA NIM (Microservices)** - Enterprise LLM Deployment

**Status**: NOT Covered
**Category**: Enterprise/Self-Hosted LLM Platform

**What it does**:
- Official NVIDIA microservice containers for enterprise LLM deployment
- OpenAI-compatible API + NVIDIA custom extensions
- Optimized for TensorRT-LLM, vLLM, SGLang backends
- Can deploy any HuggingFace model or custom models

**Key Differences**:
- Enterprise-grade self-hosted solution
- Kubernetes-native deployment
- Pre-optimized containers for NVIDIA GPUs
- Supports up to 16 GPUs free tier for NVIDIA DevBox members

**API Type**: REST (OpenAI-compatible + NVIDIA extensions)
**Authentication**: API Key
**Unique Value**: Enterprise self-hosted alternative to managed services

**Why Add**: Growing enterprise demand for on-premise LLM solutions; NVIDIA's official offering

---

### 3. **AssemblyAI LLM Gateway** - Speech-to-Text + LLM Pipeline

**Status**: NOT Covered (Assembly itself is covered, but not LLM Gateway)
**Category**: Audio + LLM Integration

**What it does**:
- Streaming speech-to-text transcription
- Built-in LLM Gateway for OpenAI, Anthropic, Google, Mistral
- Unified API: Audio → Text → LLM
- Real-time processing for voice agents

**Key Differences**:
- Not a generic LLM provider
- Specialized for voice AI workflows
- Integrated STT + LLM abstraction
- Real-time streaming capabilities

**API Type**: REST/WebSocket
**Authentication**: API Key
**Unique Value**: First unified voice-to-insights API

**Why Add**: Growing voice AI use cases; unique end-to-end pipeline

---

### 4. **OpenAI Realtime API** - Voice Streaming (NOT Standard API)

**Status**: Partially Covered (main ChatGPT API is covered, but not Realtime)
**Category**: Voice/Streaming Specialization

**What it does**:
- Direct audio streaming (not speech-to-text pipelines)
- Real-time voice agent capabilities
- New gpt-realtime model (2025)
- Phone integration via SIP

**Key Differences**:
- Different from standard /v1/chat/completions
- WebSocket-based streaming protocol
- Audio input/output handling
- Remote MCP server support

**API Type**: WebSocket
**Authentication**: API Key
**Unique Value**: Native voice agents without cascading models

**Why Add**: High-growth use case in 2025; fundamentally different from text API

---

### 5. **Exa AI** - Neural Search for LLMs

**Status**: NOT Covered
**Category**: Search + LLM Integration

**What it does**:
- Semantic/neural web search optimized for LLM agents
- Multiple search modes (neural, auto, fast, deep)
- Returns parsed content, not just links
- LLM-specific relevance ranking

**Key Differences**:
- Not a general search engine
- Transformer-based semantic search
- Specific for RAG/agentic use cases
- Better than traditional APIs for LLM integration

**API Type**: REST
**Authentication**: API Key
**Pricing**: Free tier (1000 searches/month)
**Unique Value**: LLM-optimized search (not web search)

**Why Add**: Growing agent/RAG use cases; differentiated from search APIs

---

### 6. **Brave Search API** - Privacy-Focused Search for LLMs

**Status**: NOT Covered
**Category**: Search + MCP Support

**What it does**:
- Privacy-focused web search for LLMs
- Model Context Protocol (MCP) support
- AI Grounding with cited sources
- Now on AWS Marketplace

**Key Differences**:
- Privacy-first alternative to Google
- MCP integration (standardized LLM tools)
- Used by Anthropic, Cursor, Cline, Windsurf
- Available on AWS Marketplace

**API Type**: REST + MCP
**Authentication**: API Key
**Unique Value**: Privacy + MCP standard support

**Why Add**: Growing privacy demands; MCP becomes LLM standard

---

### 7. **Modal Labs** - Serverless LLM Deployment & Inference

**Status**: NOT Covered
**Category**: Serverless Infrastructure

**What it does**:
- Serverless GPU compute for LLM inference
- Deploy any model (vLLM, Ollama, custom)
- Python-native, low operational overhead
- OpenAI-compatible API mode available

**Key Differences**:
- Not an LLM provider, but infrastructure
- Pay-only-for-compute model
- First-class Python integration
- Suitable for ML teams

**API Type**: REST (OpenAI-compatible via vLLM)
**Authentication**: API Key
**Pricing**: $30/month free compute
**Unique Value**: Most developer-friendly serverless for LLMs

**Why Add**: MLOps/ML team demand; growing adoption among developers

---

### 8. **Clarifai** - Multimodal Platform with LLM Support

**Status**: NOT Covered
**Category**: Vision-LLM-Audio Integration

**What it does**:
- 500+ models (LLM, vision, audio, multimodal)
- Complete AI lifecycle (data prep → training → inference)
- Vision-Language Models (VLMs) support
- OpenAI-compatible outputs

**Key Differences**:
- Not just LLM provider
- Integrated multimodal platform
- Vision + LLM in one system
- Enterprise-focused (fine-tuning, custom models)

**API Type**: REST (OpenAI-compatible)
**Authentication**: API Key
**Unique Value**: Integrated LLM + Vision platform

**Why Add**: Growing multimodal demand; enterprise AI lifecycle platform

---

### 9. **Ray Serve LLM** (Anyscale) - Advanced Serving Infrastructure

**Status**: NOT Covered (Anyscale platform itself may be covered)
**Category**: Advanced Inference Infrastructure

**What it does**:
- Distributed LLM serving with custom routing
- Disaggregated serving (prefill/decode separation)
- Mixture-of-Experts (MoE) support
- OpenAI-compatible API

**Key Differences**:
- Not simple API proxy
- Advanced serving patterns (prefix caching, expert routing)
- High-throughput distributed inference
- Custom request routing

**API Type**: REST (OpenAI-compatible)
**Authentication**: API Key (via Anyscale)
**Unique Value**: Cutting-edge serving patterns for large models

**Why Add**: Advanced users, production workloads; 60% TTFT reduction

---

## UNCOVERED PROVIDERS: Tier 2 (Medium Priority)

### 10. **Phi Models** (Microsoft) - Efficient SLMs with API Access

**Status**: NOT Covered
**Category**: Specialized Small Language Models

**Models**: Phi-4, Phi-4-multimodal, Phi-4-mini (latest 2025)
**Available Via**:
- Azure AI Foundry
- HuggingFace (open weights)
- NVIDIA API Catalog
- Ollama (local)

**Key Differences**:
- Not available via standard OpenAI API
- Requires Azure integration or self-hosting
- Specialized for edge/on-device deployment
- Superior efficiency (14B params, GPT-4 level reasoning)

**Why Not Yet Integrated**:
- Can be deployed via Bedrock (Azure) or OpenAI-compatible
- But DIRECT Microsoft API support missing
- Users must route through Azure

**Consider Adding**: If supporting direct Azure Phi API becomes priority

---

### 11. **IBM Granite LLM** - Enterprise Open Models

**Status**: NOT Covered
**Category**: Enterprise Open-Source LLM

**Models**: Granite 4.0, Granite 3.0
**Available Via**:
- NVIDIA NIM (inference as service)
- IBM WatsonX (already in LLMKit)
- Self-hosted

**Key Differences**:
- WatsonX is covered, but direct Granite API support might be missing
- Granite 4.0 just released (Oct 2025)
- Focus on enterprise: safety, transparency, tool-calling
- Apache 2.0 licensed

**Why Not Yet Integrated**:
- Can be deployed via WatsonX
- But direct API support could be cleaner

**Consider Adding**: If WatsonX integration is insufficient for Granite-specific features

---

### 12. **Moonshot/Kimi** - Chinese LLM with Context Window

**Status**: NOT Covered
**Category**: Regional (China) - Text Generation

**Models**: Kimi K2 (1T parameters, MoE)
**API**: https://platform.moonshot.ai/
**Key Features**:
- 200K+ context window
- Competitively priced: $0.60/M input, $2.50/M output
- API available for integration

**Why Not Covered**:
- Not OpenAI-compatible
- Requires China-based authentication
- Regional provider, limited non-China adoption
- But growing in enterprise China use cases

**Consider Adding**: If targeting Chinese market

---

### 13. **Baichuan LLM** - Chinese Enterprise LLM

**Status**: NOT Covered
**Category**: Regional (China) - Text Generation

**Models**: Baichuan 4
**API**: https://platform.baichuan-ai.com/
**Key Features**:
- Open-source + commercial API
- Enterprise services (fine-tuning, private cloud)
- Fine-tuning, private cloud deployment available

**Why Not Covered**:
- Regional focus (China market)
- Not OpenAI-compatible
- Limited international adoption
- But strong in Chinese enterprise segment

**Consider Adding**: If targeting Chinese enterprise market

---

### 14. **Baidu Ernie Bot/ERNIE 4.5** - Chinese Market Leader

**Status**: NOT Covered
**Category**: Regional (China) - Text Generation

**Models**: ERNIE 4.5, ERNIE X1 (reasoning)
**API**: Qianfan Platform (https://qianfan.cloud.baidu.com/)
**Key Features**:
- Competitive pricing: $0.55/M input, $2.2/M output
- Free access via web, API for enterprises
- Reasoning model (ERNIE X1) available

**Why Not Covered**:
- Regional provider (China-specific)
- Not OpenAI-compatible
- Requires China IP for some features
- But dominates Chinese market

**Consider Adding**: If targeting China; significant market opportunity

---

### 15. **xAI Grok Direct API** - Differentiated from OpenAI-Compatible

**Status**: PARTIALLY Covered (via openai-compatible)
**Category**: Text Generation with Unique Tools

**Key Differences**:
- Standard API: OpenAI-compatible
- Advanced Features: Web search, X/Twitter search, code execution
- Responses API: Server-side tools (unique vs. function calling)
- Voice API: OpenAI Realtime compatible

**Why Partial Coverage**:
- Can be used via openai-compatible provider
- But direct API would expose unique tool integrations

**Consider Adding**: If supporting server-side tools becomes important

---

## UNCOVERED PROVIDERS: Tier 3 (Nice to Have)

### 16. **Alibaba Qwen/QianWen** - Chinese Market LLM

**Status**: NOT Covered
**Category**: Regional (China) - Text Generation

**Models**: Qwen 2.5-Max
**API**: DashScope or direct OpenAI-compatible
**Key Differences**:
- OpenAI-compatible API available
- But regional authentication required

**Why Not Critical**:
- Can be accessed via openai-compatible provider
- Primarily regional focus

---

### 17. **Hugging Face Inference Endpoints** - Dedicated Managed Endpoints

**Status**: Partially Covered (HF base API covered)
**Category**: Inference Platform

**Key Differences** from HF base:
- Dedicated (not serverless shared)
- Custom hardware selection
- Private networking
- Scaling controls

**Why Not Separate**:
- Can use HF provider with custom endpoints
- Complexity vs. benefit trade-off

---

### 18. **Hugging Face Text Generation Inference (TGI)** - Local/Managed

**Status**: Partially Covered (via openai-compatible)
**Category**: Local Deployment Infrastructure

**Key Features**:
- OpenAI Messages API compatible
- OpenAI Chat Completions compatible
- Can deploy any HF model

**Why Not Separate**:
- Can be deployed via openai-compatible provider
- Already covered via HF infrastructure

---

### 19. **vLLM** - Local/Managed Inference Engine

**Status**: Partially Covered (via openai-compatible)
**Category**: Local Deployment Infrastructure

**Key Features**:
- OpenAI-compatible API server
- PagedAttention optimization
- vLLM-Omni for multimodal

**Why Not Separate**:
- Can be deployed via openai-compatible provider
- Popular but users deploy on their own infrastructure

---

### 20. **Replicate** - Alternative Model Hosting

**Status**: Already Covered
**Note**: Already in LLMKit providers

---

### 21. **Together AI** - Model Hosting + OpenAI API

**Status**: Partially Covered (may be via openai-compatible)
**Category**: Model Hosting Platform

**Models**: Llama 4 (latest), Llama 3.3, Mistral, CodeStral
**Key Features**:
- OpenAI-compatible API
- Latest Llama access
- Pay-per-token

**Why Coverage Uncertain**:
- Can use openai-compatible provider
- But dedicated integration would be cleaner

**Recommendation**: Verify if openai-compatible covers, or add native integration

---

## SPECIALIZED SERVICES (Not LLM Providers, but Relevant)

### Search/Retrieval APIs (NOT LLM Providers)
These provide search for LLMs to use, not generation themselves:

- **Metaphor Search API** - Neural search for LLMs (embeddings-based)
- **Brave Search API** - Privacy-focused search (+ MCP support)
- **Exa AI** - Neural web search
- **Tavily** - Search API for agents
- **Firecrawl** - Web scraping for RAG

**Note**: These should NOT be in LLMKit core, but could be companion integrations

### Vector Databases (NOT LLM Providers)
- **Pinecone** - Vector search (not generation)
- **Weaviate** - Vector search + RAG integration
- **Qdrant** - Vector DB with hybrid search
- **Chroma** - Embedding database (local)
- **Milvus** - Vector DB

**Note**: These are infrastructure for RAG, not LLM providers. LLMKit correctly excludes them.

### Voice/Speech APIs (Not Pure LLM)
- **OpenAI Realtime API** - Voice streaming (different protocol from text API)
- **Groq Realtime** - If available
- **Deepgram STT** - Speech-to-text (already in LLMKit for Deepgram API)

---

## INFRASTRUCTURE/DEPLOYMENT (Not LLM Providers)

### Serverless/Deployment Platforms
- **Modal Labs** - Serverless GPU compute (deploy any model)
- **Railway** - General deployment (hosts inference servers)
- **Replit** - Developer platform with AI integrations
- **GitHub Copilot Chat API** - Not publicly available yet

**Note**: These are deployment platforms, not LLM providers. LLMKit correctly excludes them.

---

## SUMMARY TABLE: Priority Ranking

| Rank | Provider | Category | Why Add | ROI | Effort |
|------|----------|----------|---------|-----|--------|
| 1 | Portkey AI | Multi-provider gateway | Enterprise orchestration layer | HIGH | HIGH |
| 2 | NVIDIA NIM | Enterprise self-hosted | Growing on-premise demand | HIGH | MEDIUM |
| 3 | AssemblyAI LLM Gateway | Voice-to-LLM | Voice AI growth | MEDIUM | MEDIUM |
| 4 | OpenAI Realtime API | Voice streaming | Different protocol, growing use | MEDIUM | MEDIUM |
| 5 | Exa AI | LLM search | RAG/agentic growth | MEDIUM | LOW |
| 6 | Brave Search API | Privacy search + MCP | MCP standardization | MEDIUM | LOW |
| 7 | Modal Labs | Serverless inference | Developer/ML ops demand | MEDIUM | MEDIUM |
| 8 | Clarifai | Multimodal platform | Growing multimodal demand | MEDIUM | MEDIUM |
| 9 | Ray Serve LLM | Advanced serving | Production infrastructure | LOW | HIGH |
| 10 | Moonshot/Kimi | Chinese regional LLM | Market growth in China | MEDIUM | LOW |
| 11 | Baidu ERNIE | Chinese market leader | Largest Chinese LLM demand | MEDIUM | LOW |
| 12 | Baichuan | Chinese enterprise | Enterprise China segment | LOW | LOW |
| 13 | xAI Grok (direct) | Unique tools (web/X search) | Differentiated capabilities | MEDIUM | LOW |
| 14 | Alibaba Qwen | Chinese regional LLM | Growing market | LOW | LOW |
| 15 | Microsoft Phi Direct | Efficient SLMs | Edge computing demand | LOW | MEDIUM |
| 16 | IBM Granite Direct | Enterprise open model | Enterprise adoption | LOW | MEDIUM |

---

## RECOMMENDATIONS FOR LLMKit EXPANSION

### Phase 4 Candidates (High ROI, Feasible)

**1. Exa AI Search Integration** ⭐⭐⭐
- **Why**: Growing agentic/RAG use cases
- **Effort**: LOW (simple REST API)
- **Impact**: Differentiate LLMKit for agent builders
- **Precedent**: Similar to existing Voyage/Jina

**2. Brave Search API** ⭐⭐⭐
- **Why**: Privacy-focused, MCP support emerging standard
- **Effort**: LOW (REST API + optional MCP)
- **Impact**: Privacy-first positioning
- **Market**: Growing demand in Europe/privacy-conscious orgs

**3. OpenAI Realtime API** ⭐⭐
- **Why**: Voice use cases exploding
- **Effort**: MEDIUM (WebSocket protocol)
- **Impact**: Enable voice agent builders
- **Market**: New use case category in 2025

**4. Chinese Regional Providers** (Bundle) ⭐⭐
- **Why**: Growing market opportunity
- **Effort**: LOW (similar API patterns)
- **Impact**: Enterprise China support
- **Market**: Significant if targeting China

### Later Candidates (Medium ROI or High Effort)

**5. NVIDIA NIM** ⭐
- **Why**: Enterprise self-hosted demand
- **Effort**: MEDIUM (container orchestration)
- **Impact**: Enterprise on-prem market
- **Market**: Growing but specialized

**6. Portkey AI** ⭐
- **Why**: Enterprise orchestration
- **Effort**: HIGH (complex abstraction)
- **Impact**: Multi-provider routing
- **Market**: Enterprise infrastructure play

**7. AssemblyAI LLM Gateway** ⭐
- **Why**: Voice AI pipeline
- **Effort**: MEDIUM (WebSocket + integration)
- **Impact**: Voice-specific workflows
- **Market**: Emerging but growing

### Skip (Already Covered or Low Priority)

- Modal Labs: Infrastructure, not provider (users deploy themselves)
- Railway: Deployment platform, not provider
- Replit: Developer platform (no public LLM API)
- GitHub Copilot Chat: Not public API yet
- Vector DBs (Pinecone, Weaviate, Qdrant, Chroma): Not LLM providers
- vLLM/TGI: Already covered via openai-compatible
- Together AI: Likely covered via openai-compatible (verify)

---

## VERIFICATION NOTES

### Already Implemented in LLMKit
✅ All 41 providers verified in `/home/yfedoseev/projects/modelsuite/src/providers/`

### Covered via OpenAI-Compatible Provider
- xAI/Grok (basic API)
- Meta Llama API
- Together AI (likely)
- HuggingFace TGI
- vLLM
- Lambda Labs
- Ollama (local)

### Chinese Providers NOT Covered
- Moonshot/Kimi (no openai-compatible support confirmed)
- Baidu ERNIE (no openai-compatible by default)
- Baichuan (no openai-compatible by default)
- Alibaba Qwen (has openai-compatible option)

---

## DATA SOURCES

All research conducted January 2, 2026:

1. **Official Documentation**:
   - LiteLLM: https://docs.litellm.ai/docs/providers
   - LLMKit: GitHub repo `/src/providers/` directory
   - NVIDIA NIM: https://docs.nvidia.com/nim/
   - Portkey AI: https://portkey.ai/docs/
   - Brave Search: https://brave.com/search/api/
   - Exa AI: https://docs.exa.ai/
   - AssemblyAI: https://www.assemblyai.com/docs/
   - Modal: https://modal.com/docs/
   - OpenAI Realtime: https://platform.openai.com/docs/guides/realtime

2. **Chinese Provider APIs**:
   - Moonshot: https://platform.moonshot.ai/
   - Baidu ERNIE: https://qianfan.cloud.baidu.com/
   - Baichuan: https://platform.baichuan-ai.com/
   - Alibaba Qwen: https://dashscope.aliyuncs.com/

3. **Web Search Results**: Compiled from Google Search 2025 archives

---

## CONCLUSION

LLMKit has excellent coverage of major providers (41 implemented). Gaps are primarily:

1. **Specialized Services** (search APIs, voice streaming) - niche but growing
2. **Regional Providers** (China) - significant if targeting those markets
3. **Enterprise Infrastructure** (Portkey, NVIDIA NIM) - for advanced users
4. **Newer APIs** (Realtime, MCP-based) - emerging in 2025

**Recommendation**: Focus Phase 4 on:
1. **Exa AI** (search for RAG/agents)
2. **Brave Search API** (privacy + MCP)
3. **OpenAI Realtime API** (voice use cases)
4. **Chinese Providers Bundle** (if China market is priority)

These 4 additions would address the most significant gaps with reasonable implementation effort.
