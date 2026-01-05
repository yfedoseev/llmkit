# Official Model List Sources for All Providers

This document contains the official documentation URLs and API endpoints for fetching model lists from each provider. Use these to verify model completeness.

---

## Tier 1: Major LLM Providers

### OpenAI
- **Documentation**: https://platform.openai.com/docs/models
- **API Endpoint**: `GET https://api.openai.com/v1/models`
- **API Reference**: https://platform.openai.com/docs/api-reference/models/list
- **Key Models (2025)**:
  - GPT-5, GPT-5 mini
  - GPT-4.1, GPT-4.1 mini, GPT-4.1 nano
  - o3, o3-pro, o3-mini, o4-mini
  - GPT-4o (legacy), GPT-4o Audio
  - Whisper, DALL-E 3, TTS
- **Expected Count**: ~20-25 models

### Anthropic (Claude)
- **Documentation**: https://docs.anthropic.com/en/docs/about-claude/models/overview
- **API Endpoint**: `GET https://api.anthropic.com/v1/models`
- **API Reference**: https://platform.claude.com/docs/en/api/models/list
- **Key Models (2025)**:
  - Claude 4.5 Sonnet (claude-sonnet-4-5-*)
  - Claude 4.1 Opus (claude-opus-4-1-*)
  - Claude 4 Opus, Claude 4 Sonnet
  - Claude 3.7 Sonnet
  - Claude 3.5 Sonnet v2, Claude 3.5 Haiku
  - Claude 3 Opus, Haiku (3 Sonnet retired July 2025)
- **Expected Count**: ~10-12 models

### Google (Gemini)
- **Documentation**: https://ai.google.dev/gemini-api/docs/models
- **API Endpoint**: `GET https://generativelanguage.googleapis.com/v1beta/models`
- **Models Page**: https://ai.google.dev/api/models
- **Key Models (2025)**:
  - Gemini 3 Pro, Gemini 3 Flash (preview)
  - Gemini 2.5 Pro, Gemini 2.5 Flash, Gemini 2.5 Flash-Lite
  - Gemini 2.0 Flash, Gemini 2.0 Flash-Lite
  - Lyria RealTime (music generation)
  - Gemini Live API (audio)
  - Text Embeddings, Image Gen
- **Expected Count**: ~15-20 models

### Mistral AI
- **Documentation**: https://docs.mistral.ai/deployment/ai-studio/pricing
- **Pricing Page**: https://mistral.ai/pricing
- **Key Models (2025)**:
  - Mistral Large 3 (675B total, 41B active MoE)
  - Mistral Medium 3.1
  - Mistral Small 3.2
  - Ministral 14B, 8B, 3B
  - Magistral Small/Medium (reasoning, June 2025)
  - Devstral 2, Devstral Small 2 (coding, Dec 2025)
  - Pixtral (vision)
  - Codestral
- **Expected Count**: ~15-20 models

### Cohere
- **Documentation**: https://docs.cohere.com/docs/models
- **Deprecations**: https://docs.cohere.com/docs/deprecations
- **Key Models (2025)**:
  - Command A (03-2025) - strongest model
  - Command R+ 08-2024
  - Command R 08-2024
  - Command R7B
  - Command A Translate, Reasoning, Vision
  - Embed v3 (English, Multilingual)
  - Rerank v3
- **Expected Count**: ~12-15 models

### DeepSeek
- **Documentation**: https://api-docs.deepseek.com/quick_start/pricing
- **API Endpoint**: https://api-docs.deepseek.com/api/list-models
- **Key Models (2025)**:
  - DeepSeek-V3, V3.1, V3.2-Exp
  - DeepSeek-R1 (reasoning)
  - DeepSeek-Coder
- **Expected Count**: ~5-8 models

---

## Tier 2: Inference Platforms

### Groq
- **Documentation**: https://console.groq.com/docs/models
- **API Endpoint**: `GET https://api.groq.com/openai/v1/models`
- **Deprecations**: https://console.groq.com/docs/deprecations
- **Key Models (2025)**:
  - Llama 4 Maverick 17B, Scout 17B
  - Llama 3.3 70B Versatile
  - GPT-OSS 120B, 20B
  - Qwen 3 32B
  - Kimi K2 Instruct
  - Whisper Large v3, v3 Turbo
  - Orpheus TTS (replacing PlayAI)
- **Expected Count**: ~15-20 models

### Cerebras
- **Documentation**: https://inference-docs.cerebras.ai/introduction
- **Partnership**: Meta collaboration for Llama API
- **Key Models (2025)**:
  - Llama 4 Scout (2,600+ tokens/sec)
  - Llama 4 Maverick
  - Llama 3.3 70B
  - Llama 3.1 405B (969 tokens/sec)
- **Expected Count**: ~5-8 models

### Together AI
- **Documentation**: https://docs.together.ai/reference/models-1
- **Models Page**: https://www.together.ai/models
- **API Endpoint**: Python `client.models.list()`
- **Key Features**:
  - 200+ open-source models
  - DeepSeek V3/R1 variants
  - Llama, Qwen, Mistral families
  - TTS with 300+ voices
- **Expected Count**: ~200+ models

### Fireworks AI
- **Documentation**: https://docs.fireworks.ai/api-reference/list-models
- **Models Page**: https://fireworks.ai/models
- **API Endpoint**: `GET https://api.fireworks.ai/v1/accounts/{account_id}/models`
- **Key Features**:
  - Open-source LLMs and image models
  - Fine-tuning support
  - Blazing fast inference
- **Expected Count**: ~50-100 models

### OpenRouter
- **Documentation**: https://openrouter.ai/docs/api/api-reference/models/get-models
- **Models Page**: https://openrouter.ai/models
- **API Endpoint**: `GET https://openrouter.ai/api/v1/models`
- **Key Features**:
  - 400+ models from dozens of providers
  - GPT-5, Claude 4, Gemini 2.5 Pro access
  - Automatic fallbacks and routing
- **Expected Count**: ~400+ models

### Perplexity
- **Documentation**: https://docs.perplexity.ai/getting-started/models/models/sonar
- **Key Models (2025)**:
  - Sonar (based on Llama 3.3 70B)
  - Sonar Pro (2× more citations)
  - Sonar Reasoning
  - Sonar Reasoning Pro (DeepSeek-R1 1776)
  - Sonar Deep Research
  - Third-party: GPT-5, Claude Opus 4.1, Gemini 2.5 Pro, Grok 4
- **Expected Count**: ~10-15 models

---

## Tier 3: Cloud Providers

### AWS Bedrock
- **Documentation**: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
- **Key Providers on Bedrock**:
  - AI21 Labs (Jamba 1.5)
  - Amazon (Nova Pro/Lite/Micro/Premier/Canvas/Reel/Sonic, Titan)
  - Anthropic (Claude 3-4.5)
  - Cohere (Command R/R+, Embed, Rerank)
  - DeepSeek (R1, V3.1)
  - Google (Gemma 3)
  - Luma AI (Ray v2)
  - Meta (Llama 2-4)
  - MiniMax (M2)
  - Mistral (7B through Large 3, Pixtral, Ministral, Voxtral, Codestral)
  - Moonshot (Kimi K2)
  - NVIDIA (Nemotron Nano)
  - Qwen (Qwen3 series)
  - Stability AI (SD3.5, SDXL, image editing)
  - TwelveLabs (Marengo, Pegasus)
  - Writer (Palmyra X4/X5)
- **Expected Count**: ~100+ models

### Google Vertex AI
- **Documentation**: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models
- **Includes**: Gemini + partner models (Claude, Llama)

### Azure AI Foundry
- **Documentation**: https://azure.microsoft.com/en-us/pricing/details/ai-foundry-models/
- **Includes**: OpenAI, Mistral, DeepSeek, Meta models

---

## Tier 4: Specialized Providers

### Stability AI (Images)
- **Documentation**: https://platform.stability.ai/docs/api-reference
- **Core Models**: https://stability.ai/core-models
- **Key Models**:
  - Stable Diffusion 3.5 Large/Medium/Turbo
  - Stable Diffusion 3 Large/Medium
  - SDXL, SDXL Turbo
  - Stable Image Core/Ultra
  - Image editing (inpaint, outpaint, style, control)
- **Expected Count**: ~15-20 models

### ElevenLabs (Voice/TTS)
- **Documentation**: https://elevenlabs.io/docs/overview/models
- **API Endpoint**: https://elevenlabs.io/docs/api-reference/models/list
- **Key Models**:
  - Eleven Flash v2.5 (75ms latency, 32 languages)
  - Eleven Turbo v2.5
  - Eleven Multilingual v2 (29 languages)
- **Expected Count**: ~5-10 models

### Replicate
- **Documentation**: https://replicate.com/docs/reference/http
- **API Endpoint**: `GET https://api.replicate.com/v1/models`
- **Official Models**: https://replicate.com/collections/official
- **Key Features**:
  - 100+ official models (always warm)
  - Thousands of community models
  - Unified predictions endpoint
- **Expected Count**: 100+ official, thousands community

---

## Tier 5: Chinese Providers

### Alibaba Qwen (DashScope)
- **Documentation**: https://www.alibabacloud.com/help/en/model-studio/models
- **API Base (Singapore)**: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`
- **Key Models (2025)**:
  - Qwen3-Max (flagship)
  - Qwen-Plus (various snapshots)
  - Qwen-Flash, Qwen-Turbo
  - Qwen-Long (10M context)
  - Qwen3-VL (vision-language)
  - Qwen3-Coder-Flash/Plus
  - Qwen-Math-Plus/Turbo
  - Qwen-MT (92 languages)
  - Text-Embedding v3/v4
- **Expected Count**: ~20-30 models

### Zhipu AI (GLM)
- **API Base**: `https://open.bigmodel.cn/api/paas/v4`
- **Key Models**: GLM-4 series

### Moonshot (Kimi)
- **API Base**: `https://api.moonshot.ai/v1`
- **Key Models**: Kimi K2 (256K context)

### Baichuan
- **API Base**: `https://api.baichuan-ai.com/v1`
- **Key Models**: Baichuan 4 series (192K context)

### MiniMax
- **API Base**: `https://api.minimax.io`
- **Key Models**: ABAB series, M2 (4M context)

### Volcengine (Doubao)
- **API Base**: `https://ark.cn-beijing.volces.com/api/v3`
- **Key Models**: Doubao series (256K context)

### iFlyTek Spark
- **API Base**: `https://spark-api.xf-yun.com`
- **Key Models**: Spark 4 series (128K context)

---

## Quick API Check Commands

```bash
# OpenAI
curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"

# Anthropic
curl https://api.anthropic.com/v1/models -H "x-api-key: $ANTHROPIC_API_KEY" -H "anthropic-version: 2023-06-01"

# Groq
curl https://api.groq.com/openai/v1/models -H "Authorization: Bearer $GROQ_API_KEY"

# OpenRouter
curl https://openrouter.ai/api/v1/models -H "Authorization: Bearer $OPENROUTER_API_KEY"

# Together AI
curl https://api.together.xyz/v1/models -H "Authorization: Bearer $TOGETHER_API_KEY"
```

---

## Summary: Expected vs Current Model Counts

| Provider | Expected | Current | Status |
|----------|----------|---------|--------|
| AWS Bedrock | ~100+ | 103 | ✅ Complete |
| OpenRouter | ~400+ | 353 | ⚠️ Check for updates |
| Together AI | ~200+ | 61 | ⚠️ Need API fetch |
| OpenAI | ~20-25 | 22 | ✅ Good |
| Anthropic | ~10-12 | 8 | ⚠️ Missing Claude 4.x |
| Google Gemini | ~15-20 | 13 | ⚠️ Missing Gemini 3 |
| Mistral | ~15-20 | 18 | ✅ Good |
| Cohere | ~12-15 | 15 | ✅ Good |
| DeepSeek | ~5-8 | 7 | ✅ Good |
| Groq | ~15-20 | 18 | ✅ Good |
| Fireworks | ~50-100 | 19 | ⚠️ Need API fetch |
| Perplexity | ~10-15 | 6 | ⚠️ Missing Sonar models |
| Qwen/DashScope | ~20-30 | 19 | ⚠️ Check for updates |
| Stability AI | ~15-20 | 16 | ✅ Good |
| ElevenLabs | ~5-10 | 10 | ✅ Good |

---

*Last Updated: January 2026*
