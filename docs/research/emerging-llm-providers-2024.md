# Emerging LLM Inference Startups and AI API Providers (2023-2024)

This document catalogs emerging LLM inference startups and AI API providers that launched or gained traction in 2023-2024, focusing on providers not commonly listed in major aggregators like LiteLLM.

**Research Date:** January 2025
**Last Updated:** 2025-01-04

---

## Table of Contents

1. [YC-Backed AI Companies with APIs](#1-yc-backed-ai-companies-with-apis-2023-2024-batches)
2. [Emerging Inference Startups (Seed/Series A)](#2-emerging-inference-startups-seedseries-a-stage)
3. [Open-Source Model Hosting Platforms](#3-new-open-source-model-hosting-platforms)
4. [Regional AI Startups with APIs](#4-regional-ai-startups-with-apis)
5. [Specialized Vertical AI Providers](#5-specialized-vertical-ai-providers)
6. [LLM Gateways and Routers](#6-llm-gateways-and-routers)
7. [Hardware-Accelerated Inference Providers](#7-hardware-accelerated-inference-providers)
8. [AI Infrastructure Cloud Providers](#8-ai-infrastructure-cloud-providers-neoclouds)

---

## 1. YC-Backed AI Companies with APIs (2023-2024 Batches)

### 1.1 Pipeshift
**Batch:** YC S24
**Description:** Modular orchestration platform for building with open-source AI components across any cloud or on-prem.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.pipeshift.io/v1/` (estimated) |
| **Key Models** | Open-source LLMs (Llama, Mistral), embeddings, vision models, audio models |
| **Unique Features** | Fine-tuning + inference platform, modular AI orchestration |
| **Env Variable** | `PIPESHIFT_API_KEY` |
| **Funding** | Seed (YC S24) |

### 1.2 nCompass
**Batch:** YC W24
**Description:** Simplifies hosting and acceleration of large-scale, open-source AI models.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.ncompass.ai/v1/` (estimated) |
| **Key Models** | Open-source models with hardware acceleration |
| **Unique Features** | PhD-level hardware acceleration optimization, cost-efficient inference |
| **Env Variable** | `NCOMPASS_API_KEY` |
| **Funding** | $1.7M Seed (2024) |

### 1.3 Superpowered AI
**Batch:** YC (2023)
**Description:** Simple API for connecting external data sources to LLMs (RAG-focused).

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.superpowered.ai/v1/` |
| **Key Models** | RAG-optimized models, document processing |
| **Unique Features** | Proprietary RAG technology, external data source connectors |
| **Env Variable** | `SUPERPOWERED_API_KEY` |

### 1.4 Greptile
**Batch:** YC
**Description:** API that uses LLMs to answer questions about company codebases.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.greptile.com/v1/` |
| **Key Models** | Code-understanding LLMs |
| **Unique Features** | Codebase-aware Q&A, code search and understanding |
| **Env Variable** | `GREPTILE_API_KEY` |

### 1.5 Unsloth AI
**Batch:** YC S24
**Description:** Open-source platform for fine-tuning and training LLMs.

| Field | Value |
|-------|-------|
| **API Endpoint** | Open-source (self-hosted) |
| **Key Models** | Llama, Mistral, Qwen fine-tuning |
| **Unique Features** | 2x faster training, 80% less memory usage |
| **Env Variable** | N/A (open-source) |
| **GitHub** | `unslothai/unsloth` |

### 1.6 Mem0
**Batch:** YC S24
**Description:** Open-source memory platform for LLM applications.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.mem0.ai/v1/` |
| **Key Models** | Memory layer for any LLM |
| **Unique Features** | Persistent memory for AI agents, cross-session context |
| **Env Variable** | `MEM0_API_KEY` |
| **GitHub** | `mem0ai/mem0` |

### 1.7 Maihem
**Batch:** YC W24
**Description:** AI agents specialized in evaluating other AI products.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.maihem.ai/v1/` |
| **Key Models** | Testing/evaluation agents |
| **Unique Features** | AI product testing, edge case coverage |
| **Env Variable** | `MAIHEM_API_KEY` |

### 1.8 Wordware
**Batch:** YC S24
**Description:** IDE to develop, iterate, and deploy AI experiences.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.wordware.ai/v1/` |
| **Key Models** | Multi-model support |
| **Unique Features** | Visual AI development environment |
| **Env Variable** | `WORDWARE_API_KEY` |

### 1.9 Neuralize
**Batch:** YC S24
**Description:** Deployment platform for on-device AI inference.

| Field | Value |
|-------|-------|
| **API Endpoint** | Edge deployment (not cloud API) |
| **Key Models** | Optimized edge models |
| **Unique Features** | On-device inference, privacy-preserving AI |
| **Env Variable** | `NEURALIZE_API_KEY` |

---

## 2. Emerging Inference Startups (Seed/Series A Stage)

### 2.1 FriendliAI
**Stage:** Seed Extension ($20M)
**Description:** AI inference platform providing "last mile" AI engagement.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.friendli.ai/v1/` |
| **Key Models** | Llama, Mistral, custom models |
| **Unique Features** | Dedicated endpoints, serverless endpoints, container deployment |
| **Env Variable** | `FRIENDLI_API_KEY` |
| **Pricing** | Per-token pricing |

### 2.2 Inference Labs
**Stage:** Pre-Seed ($2.3M, April 2024)
**Description:** AI inference infrastructure on blockchain with Proof of Inference.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.inferencelabs.ai/` (estimated) |
| **Key Models** | Decentralized LLM inference |
| **Unique Features** | Blockchain-verified inference, liquid staking |
| **Env Variable** | `INFERENCE_LABS_API_KEY` |

### 2.3 Runware
**Stage:** Seed ($3M, October 2024)
**Description:** World's fastest media generation API with sub-second generation times.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.runware.ai/v1/` |
| **Key Models** | FLUX, Stable Diffusion 1.5, SDXL |
| **Unique Features** | Sub-second image generation, media-focused |
| **Env Variable** | `RUNWARE_API_KEY` |
| **Pricing** | Pay-per-generation |

### 2.4 Enkrypt AI
**Stage:** Seed ($2.4M, February 2024)
**Description:** Secure enterprise gateway for LLM usage oversight.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.enkryptai.com/v1/` |
| **Key Models** | Gateway to any LLM |
| **Unique Features** | Sentry gateway, LLM usage monitoring, security |
| **Env Variable** | `ENKRYPT_API_KEY` |

### 2.5 Credal AI
**Stage:** Seed ($4.8M, October 2023)
**Description:** Secure connection of internal data to cloud-hosted GenAI models.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.credal.ai/v1/` |
| **Key Models** | Multi-model support with security layer |
| **Unique Features** | Pre-built data connectors, enterprise security |
| **Env Variable** | `CREDAL_API_KEY` |

### 2.6 TensorZero
**Stage:** Seed ($7.3M, 2024)
**Description:** Solves enterprise LLM development challenges.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.tensorzero.com/v1/` |
| **Key Models** | Multi-model orchestration |
| **Unique Features** | Enterprise LLM lifecycle management |
| **Env Variable** | `TENSORZERO_API_KEY` |

---

## 3. New Open-Source Model Hosting Platforms

### 3.1 Featherless.ai
**Description:** Serverless LLM hosting with flat monthly pricing for unlimited tokens.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.featherless.ai/v1/` |
| **Key Models** | Llama 2/3, Mistral, Qwen, DeepSeek, 1000+ HuggingFace models |
| **Unique Features** | Unlimited tokens, flat pricing, serverless |
| **Env Variable** | `FEATHERLESS_API_KEY` |
| **Pricing** | Flat monthly subscription |
| **OpenAI Compatible** | Yes |

### 3.2 DeepInfra
**Stage:** Seed ($8M, November 2023)
**Description:** Pay-per-use API for 100+ open-source models.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.deepinfra.com/v1/openai/` |
| **Key Models** | Llama 3, Mistral, Mixtral, SDXL, Whisper |
| **Unique Features** | OpenAI-compatible, dedicated instances available |
| **Env Variable** | `DEEPINFRA_API_KEY` |
| **Pricing** | Per-token (very competitive) |
| **OpenAI Compatible** | Yes |

### 3.3 Together AI
**Stage:** Series A Extension ($106M, March 2024)
**Description:** Platform for open-source model inference and fine-tuning.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.together.xyz/v1/` |
| **Key Models** | 200+ models: Llama, Mistral, DeepSeek-R1, Code Llama, Stable Diffusion |
| **Unique Features** | 11x lower cost than GPT-4o, LoRA fine-tuning, OpenAI-compatible |
| **Env Variable** | `TOGETHER_API_KEY` |
| **Pricing** | Per-token, competitive |
| **OpenAI Compatible** | Yes |

### 3.4 Novita AI
**Description:** 200+ Model APIs with GPU cloud and serverless options.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.novita.ai/v3/` |
| **Key Models** | DeepSeek V3, Llama, Stable Diffusion, FLUX |
| **Unique Features** | $0.20/M tokens, serverless GPUs, custom deployment |
| **Env Variable** | `NOVITA_API_KEY` |
| **Pricing** | From $0.20 per million tokens |
| **OpenAI Compatible** | Yes |

### 3.5 SiliconFlow
**Description:** High-performance AI infrastructure for LLMs and multimodal models.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.siliconflow.cn/v1/` |
| **Key Models** | DeepSeek, GLM, Qwen, FLUX.1 series |
| **Unique Features** | 2.3x faster inference, 32% lower latency vs competitors |
| **Env Variable** | `SILICONFLOW_API_KEY` |
| **Pricing** | Highly competitive, China-focused |
| **OpenAI Compatible** | Yes |

### 3.6 Nscale
**Description:** Fully managed serverless inference platform.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.nscale.com/v1/` |
| **Key Models** | Popular GenAI models |
| **Unique Features** | Zero rate limits, pay-per-use, instant access |
| **Env Variable** | `NSCALE_API_KEY` |
| **OpenAI Compatible** | Yes |

### 3.7 RunPod
**Description:** Serverless GPU cloud with vLLM integration.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.runpod.ai/v2/{endpoint_id}/` |
| **Key Models** | Any HuggingFace model via vLLM |
| **Unique Features** | vLLM workers, OpenAI-compatible, auto-scaling |
| **Env Variable** | `RUNPOD_API_KEY` |
| **Pricing** | Per-second GPU billing |
| **OpenAI Compatible** | Yes (via vLLM worker) |

---

## 4. Regional AI Startups with APIs

### 4.1 Southeast Asia

#### Kata.ai (Indonesia)
**Description:** Conversational AI platform for businesses.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.kata.ai/v1/` |
| **Key Models** | Chatbot/NLP models for Indonesian language |
| **Unique Features** | Indonesian language optimization, WhatsApp integration |
| **Env Variable** | `KATA_API_KEY` |
| **Region** | Indonesia |

#### Nebius AI Studio (Singapore Hub)
**Description:** AI cloud platform expanding into Southeast Asia.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.studio.nebius.ai/v1/` |
| **Key Models** | Llama, Mistral, Qwen |
| **Unique Features** | EU data compliance, Singapore regional hub |
| **Env Variable** | `NEBIUS_API_KEY` |
| **Region** | Singapore/EU |

### 4.2 Middle East

#### Falcon (UAE - Technology Innovation Institute)
**Description:** Open-source LLM from Abu Dhabi.

| Field | Value |
|-------|-------|
| **API Endpoint** | Via AI71 platform: `https://api.ai71.ai/v1/` |
| **Key Models** | Falcon 2 (up to 180B parameters) |
| **Unique Features** | Open-source (Apache 2.0), multilingual, topped HuggingFace leaderboard |
| **Env Variable** | `AI71_API_KEY` |
| **Region** | UAE |

#### Jais (UAE - G42/MBZUAI)
**Description:** Arabic-optimized LLM from Abu Dhabi.

| Field | Value |
|-------|-------|
| **API Endpoint** | Via G42 Cloud |
| **Key Models** | Jais 70B (Arabic-English bilingual) |
| **Unique Features** | Best-in-class Arabic understanding |
| **Env Variable** | `G42_API_KEY` |
| **Region** | UAE |

#### ALLaM (Saudi Arabia - SDAIA)
**Description:** Arabic Large Language Model on IBM watsonx.

| Field | Value |
|-------|-------|
| **API Endpoint** | Via IBM watsonx.ai |
| **Key Models** | ALLaM (Arabic-first) |
| **Unique Features** | Open-source Arabic LLM, cultural context |
| **Env Variable** | Via IBM watsonx credentials |
| **Region** | Saudi Arabia |

#### Kawn by Misraj AI (Saudi Arabia)
**Description:** Breakthrough Arabic LLM with dialect support.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.misraj.ai/v1/` (estimated) |
| **Key Models** | Kawn (Arabic LLM), Mutarjim (translation), Lahjawi (dialect) |
| **Unique Features** | 15 Arabic dialect support, bidirectional translation |
| **Env Variable** | `MISRAJ_API_KEY` |
| **Region** | Saudi Arabia |

### 4.3 Africa

#### MazzumaGPT (Ghana)
**Description:** LLM trained on blockchain languages for smart contracts.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.mazzuma.com/gpt/` (estimated) |
| **Key Models** | Blockchain/smart contract specialized LLM |
| **Unique Features** | Smart contract drafting, blockchain languages |
| **Env Variable** | `MAZZUMA_API_KEY` |
| **Region** | Ghana |

### 4.4 Latin America

#### Lexter (Brazil)
**Description:** First legal LLM company in Brazil (YC-backed).

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.lexter.ai/v1/` (estimated) |
| **Key Models** | Brazilian legal LLM |
| **Unique Features** | Portuguese legal documents, top 5 Brazil law firms |
| **Env Variable** | `LEXTER_API_KEY` |
| **Region** | Brazil |

#### Take Blip (Brazil)
**Description:** Communication-based business solutions and text analytics.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.blip.ai/` |
| **Key Models** | Conversational AI, NLP |
| **Unique Features** | Marketing automation, WhatsApp business integration |
| **Env Variable** | `BLIP_API_KEY` |
| **Region** | Brazil |

---

## 5. Specialized Vertical AI Providers

### 5.1 Legal AI

#### Harvey
**Description:** AI for legal professionals, partnered with OpenAI.

| Field | Value |
|-------|-------|
| **API Endpoint** | Enterprise only (no public API) |
| **Key Models** | Custom legal LLM (GPT-based) |
| **Unique Features** | Contract analysis, litigation support, legal reasoning |
| **Env Variable** | N/A (Enterprise) |

#### DeepJudge
**Description:** Precision AI search for legal teams.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.deepjudge.ai/v1/` |
| **Key Models** | Legal search and analysis LLM |
| **Unique Features** | Internal knowledge base search, LLM agents for legal |
| **Env Variable** | `DEEPJUDGE_API_KEY` |

#### Legora
**Description:** Collaborative AI workspace for lawyers.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.legora.com/v1/` |
| **Key Models** | Document analysis LLM |
| **Unique Features** | Analyze 10K+ documents simultaneously, markup suggestions |
| **Env Variable** | `LEGORA_API_KEY` |

#### Lexlegis.AI (India)
**Description:** Legal-specific LLM for Indian law.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.lexlegis.ai/v1/` |
| **Key Models** | Indian legal LLM (10M+ documents trained) |
| **Unique Features** | Indian case law search with citations |
| **Env Variable** | `LEXLEGIS_API_KEY` |
| **Region** | India |

#### Pincites (YC)
**Description:** Contract review acceleration using LLMs.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.pincites.com/v1/` |
| **Key Models** | Contract analysis LLM |
| **Unique Features** | First-pass contract review, risk identification |
| **Env Variable** | `PINCITES_API_KEY` |

### 5.2 Healthcare AI

#### OpenEvidence
**Description:** Medical AI API from Mayo Clinic Platform Accelerate.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.openevidence.com/v1/` |
| **Key Models** | Medical domain LLM |
| **Unique Features** | Only API purpose-built for medical domain, powers Elsevier's ClinicalKey AI |
| **Env Variable** | `OPENEVIDENCE_API_KEY` |

#### Hippocratic AI
**Description:** Safety-focused LLM for healthcare.

| Field | Value |
|-------|-------|
| **API Endpoint** | Enterprise only |
| **Key Models** | 4.1T+ parameter constellation architecture |
| **Unique Features** | Healthcare safety focus, specialized support models |
| **Env Variable** | N/A (Enterprise) |
| **Funding** | $50M Seed |

#### Abridge
**Description:** Clinical documentation AI integrated with Epic.

| Field | Value |
|-------|-------|
| **API Endpoint** | Enterprise/Epic integration |
| **Key Models** | Clinical conversation summarization |
| **Unique Features** | Epic EHR integration, ambient clinical documentation |
| **Env Variable** | N/A (Enterprise) |

#### Corti
**Description:** Healthcare-native APIs for clinical workflows.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.corti.ai/v1/` |
| **Key Models** | Clinical workflow LLMs |
| **Unique Features** | Published research at NeurIPS, ICML, healthcare-native |
| **Env Variable** | `CORTI_API_KEY` |

#### John Snow Labs Healthcare LLM
**Description:** Medical LLMs for clinical text processing.

| Field | Value |
|-------|-------|
| **API Endpoint** | Self-hosted or Databricks |
| **Key Models** | Healthcare LLM, Medical NER, Clinical summarization |
| **Unique Features** | No external data sharing, on-premise deployment |
| **Env Variable** | `JSL_LICENSE_KEY` |

### 5.3 Finance AI

#### Rogo
**Description:** LLMs built specifically for finance professionals.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.rogo.ai/v1/` |
| **Key Models** | Finance-tuned LLMs |
| **Unique Features** | Professionally labeled finance data, flexible deployment |
| **Env Variable** | `ROGO_API_KEY` |

#### Ntropy
**Description:** Transaction enrichment API for financial data.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.ntropy.com/v2/` |
| **Key Models** | Transaction classification/enrichment |
| **Unique Features** | Millisecond latency, 10,000x lower cost than alternatives |
| **Env Variable** | `NTROPY_API_KEY` |

#### NayaOne
**Description:** AI sandbox for financial institutions.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.nayaone.com/v1/` |
| **Key Models** | Multi-model testing environment |
| **Unique Features** | Synthetic data generation, fintech marketplace |
| **Env Variable** | `NAYAONE_API_KEY` |
| **Region** | UK/London |

#### FinGPT (Open Source)
**Description:** Open-source financial LLM.

| Field | Value |
|-------|-------|
| **API Endpoint** | Self-hosted |
| **Key Models** | Finance-tuned open models |
| **Unique Features** | Real-time NLP data processing, LoRA fine-tuning |
| **Env Variable** | N/A (self-hosted) |
| **GitHub** | `AI4Finance-Foundation/FinGPT` |

---

## 6. LLM Gateways and Routers

### 6.1 OpenRouter
**Description:** Universal API for 300+ AI models from 60+ providers.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://openrouter.ai/api/v1/` |
| **Key Models** | 400+ models from all major providers |
| **Unique Features** | Automatic routing based on cost/speed/quality, failover handling |
| **Env Variable** | `OPENROUTER_API_KEY` |
| **Funding** | $40M Series A (June 2025) |
| **OpenAI Compatible** | Yes |

### 6.2 Unify AI
**Description:** Dynamic LLM routing for optimal cost/latency/quality.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.unify.ai/v0/` |
| **Key Models** | Routes to 50+ providers |
| **Unique Features** | Neural network router, router string concept |
| **Env Variable** | `UNIFY_API_KEY` |
| **Funding** | $8M (May 2024) |
| **OpenAI Compatible** | Yes |

### 6.3 Martian
**Description:** First LLM router, dynamic model selection.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.withmartian.com/v1/` |
| **Key Models** | Routes to multiple providers |
| **Unique Features** | Cost controls (max_cost), willingness_to_pay parameter |
| **Env Variable** | `MARTIAN_API_KEY` |
| **Funding** | $9M (NEA, General Catalyst) |
| **OpenAI Compatible** | Yes |

### 6.4 Portkey AI
**Description:** Enterprise AI gateway with guardrails.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.portkey.ai/v1/` |
| **Key Models** | 1600+ LLMs supported |
| **Unique Features** | Load balancing, 50+ AI guardrails, request timeouts |
| **Env Variable** | `PORTKEY_API_KEY` |
| **OpenAI Compatible** | Yes |

### 6.5 LLM Gateway (llmgateway.io)
**Description:** Open-source alternative to OpenRouter.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.llmgateway.io/v1/` |
| **Key Models** | OpenAI, Anthropic, Google, more |
| **Unique Features** | Open-source, unified API |
| **Env Variable** | `LLMGATEWAY_API_KEY` |
| **OpenAI Compatible** | Yes |

### 6.6 Helicone
**Description:** LLM gateway written in Rust with ultra-low latency.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://oai.helicone.ai/v1/` (proxy) |
| **Key Models** | Proxy to any provider |
| **Unique Features** | 8ms P50 latency, observability, Rust-based |
| **Env Variable** | `HELICONE_API_KEY` |
| **OpenAI Compatible** | Yes |

---

## 7. Hardware-Accelerated Inference Providers

### 7.1 Groq
**Description:** LPU (Language Processing Unit) powered inference.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.groq.com/openai/v1/` |
| **Key Models** | Llama 3, Mixtral, Gemma |
| **Unique Features** | 750+ tokens/sec on Llama 3.1 8B, LPU hardware |
| **Env Variable** | `GROQ_API_KEY` |
| **Funding** | $640M Series D (2024), $2.8B valuation |
| **OpenAI Compatible** | Yes |

### 7.2 Cerebras
**Description:** Wafer-Scale Engine powered inference.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.cerebras.ai/v1/` |
| **Key Models** | Llama 3.1 (8B, 70B, 405B) |
| **Unique Features** | 1800 tokens/sec on 8B, 969 tokens/sec on 405B (record) |
| **Env Variable** | `CEREBRAS_API_KEY` |
| **OpenAI Compatible** | Yes |

### 7.3 SambaNova
**Description:** RDU (Reconfigurable Dataflow Unit) inference.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.sambanova.ai/v1/` |
| **Key Models** | Llama 3.1 (8B, 70B, 405B) |
| **Unique Features** | 580 tokens/sec on 70B, 132 tokens/sec on 405B in 16-bit |
| **Env Variable** | `SAMBANOVA_API_KEY` |
| **OpenAI Compatible** | Yes |

### 7.4 FuriosaAI
**Description:** Korean AI chip startup for inference.

| Field | Value |
|-------|-------|
| **API Endpoint** | Hardware/SDK focused |
| **Key Models** | Via RNGD chip |
| **Unique Features** | RNGD chip for LLM/multimodal inference |
| **Env Variable** | N/A (hardware) |
| **Region** | South Korea |

---

## 8. AI Infrastructure Cloud Providers (Neoclouds)

### 8.1 CoreWeave
**Description:** AI-native cloud with GPU infrastructure.

| Field | Value |
|-------|-------|
| **API Endpoint** | Kubernetes/managed services |
| **Key Models** | Any model on NVIDIA GPUs |
| **Unique Features** | Managed Kubernetes, Slurm, inference partnerships |
| **Env Variable** | Via Kubernetes credentials |

### 8.2 Lambda Labs
**Description:** GPU cloud with ML frameworks pre-installed.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://cloud.lambda.ai/api/v1/` |
| **Key Models** | Any model on A100/H100 GPUs |
| **Unique Features** | One-click GPU clusters, NVIDIA investor |
| **Env Variable** | `LAMBDA_API_KEY` |

### 8.3 Lepton AI
**Description:** Global GPU compute network via single platform.

| Field | Value |
|-------|-------|
| **API Endpoint** | `https://api.lepton.ai/v1/` |
| **Key Models** | Public models + custom deployment |
| **Unique Features** | NVIDIA DGX Cloud integration, global GPU network |
| **Env Variable** | `LEPTON_API_KEY` |
| **OpenAI Compatible** | Yes |

---

## Summary: Environment Variable Conventions

| Provider | Environment Variable | API Base URL |
|----------|---------------------|--------------|
| Together AI | `TOGETHER_API_KEY` | `https://api.together.xyz/v1/` |
| DeepInfra | `DEEPINFRA_API_KEY` | `https://api.deepinfra.com/v1/openai/` |
| Groq | `GROQ_API_KEY` | `https://api.groq.com/openai/v1/` |
| Cerebras | `CEREBRAS_API_KEY` | `https://api.cerebras.ai/v1/` |
| SambaNova | `SAMBANOVA_API_KEY` | `https://api.sambanova.ai/v1/` |
| Featherless | `FEATHERLESS_API_KEY` | `https://api.featherless.ai/v1/` |
| Novita AI | `NOVITA_API_KEY` | `https://api.novita.ai/v3/` |
| SiliconFlow | `SILICONFLOW_API_KEY` | `https://api.siliconflow.cn/v1/` |
| OpenRouter | `OPENROUTER_API_KEY` | `https://openrouter.ai/api/v1/` |
| Unify AI | `UNIFY_API_KEY` | `https://api.unify.ai/v0/` |
| Martian | `MARTIAN_API_KEY` | `https://api.withmartian.com/v1/` |
| Portkey | `PORTKEY_API_KEY` | `https://api.portkey.ai/v1/` |
| Helicone | `HELICONE_API_KEY` | `https://oai.helicone.ai/v1/` |
| RunPod | `RUNPOD_API_KEY` | `https://api.runpod.ai/v2/{endpoint_id}/` |
| Lepton AI | `LEPTON_API_KEY` | `https://api.lepton.ai/v1/` |
| Lambda Labs | `LAMBDA_API_KEY` | `https://cloud.lambda.ai/api/v1/` |
| FriendliAI | `FRIENDLI_API_KEY` | `https://api.friendli.ai/v1/` |
| AI71 (Falcon) | `AI71_API_KEY` | `https://api.ai71.ai/v1/` |
| Perplexity | `PERPLEXITY_API_KEY` | `https://api.perplexity.ai/` |

---

## Key Trends Observed (2024)

1. **Cost Reduction**: Inference costs dropped from $20/M tokens (Nov 2022) to $0.07/M tokens (Oct 2024) for efficient models.

2. **Hardware Differentiation**: Groq (LPU), Cerebras (WSE), and SambaNova (RDU) offer 10-100x faster inference than GPU-based solutions.

3. **Regional LLMs**: Strong emergence of Arabic LLMs (Falcon, Jais, ALLaM, Kawn) and Southeast Asian expansion.

4. **Vertical Specialization**: Legal, medical, and finance AI APIs becoming production-ready.

5. **LLM Routing**: OpenRouter, Unify, and Martian making multi-provider access seamless.

6. **Open Source Dominance**: Llama, Mistral, Qwen, and DeepSeek models available across all platforms.

---

## Sources

- [YC Company Directory](https://www.ycombinator.com/companies)
- [TechCrunch AI Startup Coverage](https://techcrunch.com)
- [Crunchbase Funding Data](https://www.crunchbase.com)
- [Artificial Analysis Benchmarks](https://artificialanalysis.ai)
- [NVIDIA Blog - Generative AI Startups](https://blogs.nvidia.com)
- [PitchBook Analysis](https://pitchbook.com)
- Individual company documentation and API references
