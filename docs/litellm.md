# LiteLLM Supported Providers

A comprehensive list of all LLM providers supported by the LiteLLM Python library. LiteLLM is a unified SDK that enables calling 100+ LLM APIs in OpenAI (or native) format.

**Documentation:** https://docs.litellm.ai/
**GitHub:** https://github.com/BerriAI/litellm
**Providers Database:** https://models.litellm.ai/

---

## Cloud Platforms & Enterprise

### OpenAI
- **Details:** Access to GPT-4, GPT-4 Turbo, GPT-3.5-Turbo, and other OpenAI models
- **Documentation:** https://docs.litellm.ai/docs/providers/openai
- **Auth:** API Key required
- **Features:** Full endpoint support including chat/completions, embeddings, image generation

### Azure OpenAI
- **Details:** Microsoft's implementation of OpenAI models with enterprise features
- **Documentation:** https://docs.litellm.ai/docs/providers/azure
- **Auth:** API Key, Endpoint URL required
- **Features:** Chat completions, embeddings, deployments

### Google (Vertex AI)
- **Details:** Access to Google's Gemini, PaLM 2, and other Google AI models
- **Documentation:** https://docs.litellm.ai/docs/providers/vertex_ai
- **Auth:** Service account credentials or API key
- **Features:** Chat completions, embeddings, function calling

### Google AI Studio
- **Details:** Direct access to Google's models via AI Studio
- **Documentation:** https://docs.litellm.ai/docs/providers/google
- **Auth:** API Key
- **Features:** Gemini models, embeddings

### AWS Bedrock
- **Details:** Access to various models including Claude, Llama, Mistral through AWS
- **Documentation:** https://docs.litellm.ai/docs/providers/bedrock
- **Auth:** AWS credentials (Access Key, Secret Key, Region)
- **Features:** Chat completions, embeddings, streaming

### AWS SageMaker
- **Details:** Deploy and access custom models through AWS SageMaker
- **Documentation:** https://docs.litellm.ai/docs/providers/sagemaker
- **Auth:** AWS credentials
- **Features:** Custom endpoint support

### Anthropic
- **Details:** Access to Claude models (Claude 3, Claude 2, etc.)
- **Documentation:** https://docs.litellm.ai/docs/providers/anthropic
- **Auth:** API Key
- **Features:** Chat completions, streaming, vision capabilities

---

## Specialized AI Services

### Mistral AI
- **Details:** High-performance language models with strong reasoning capabilities
- **Documentation:** https://docs.litellm.ai/docs/providers/mistral
- **Auth:** API Key
- **Features:** Chat completions, embeddings

### Cohere
- **Details:** Text generation and understanding models
- **Documentation:** https://docs.litellm.ai/docs/providers/cohere
- **Auth:** API Key
- **Features:** Chat completions, embeddings, reranking

### Groq
- **Details:** Ultra-fast inference engine for LLMs
- **Documentation:** https://docs.litellm.ai/docs/providers/groq
- **Auth:** API Key
- **Features:** Chat completions, streaming

### Together AI
- **Details:** Access to a collection of open-source and proprietary models
- **Documentation:** https://docs.litellm.ai/docs/providers/together
- **Auth:** API Key
- **Features:** Chat completions, embeddings

### Fireworks AI
- **Details:** Fast inference platform with various model options
- **Documentation:** https://docs.litellm.ai/docs/providers/fireworks
- **Auth:** API Key
- **Features:** Chat completions

### DeepInfra
- **Details:** Serverless GPU inference platform
- **Documentation:** https://docs.litellm.ai/docs/providers/deepinfra
- **Auth:** API Key
- **Features:** Chat completions, embeddings

### Replicate
- **Details:** Access to various open-source models via Replicate
- **Documentation:** https://docs.litellm.ai/docs/providers/replicate
- **Auth:** API Token
- **Features:** Model deployment and inference

### Perplexity AI
- **Details:** Search-augmented language model
- **Documentation:** https://docs.litellm.ai/docs/providers/perplexity
- **Auth:** API Key
- **Features:** Chat completions with web search

### xAI
- **Details:** Access to Grok model and other xAI models
- **Documentation:** https://docs.litellm.ai/docs/providers/xai
- **Auth:** API Key
- **Features:** Chat completions

### Deepseek
- **Details:** Chinese LLM provider with cost-effective models
- **Documentation:** https://docs.litellm.ai/docs/providers/deepseek
- **Auth:** API Key
- **Features:** Chat completions

### SambaNova
- **Details:** Fast inference with proprietary hardware acceleration
- **Documentation:** https://docs.litellm.ai/docs/providers/sambanova
- **Auth:** API Key
- **Features:** Chat completions

### Cerebras
- **Details:** High-performance AI inference
- **Documentation:** https://docs.litellm.ai/docs/providers/cerebras
- **Auth:** API Key
- **Features:** Chat completions

---

## Open Source & Local Deployment

### Ollama
- **Details:** Run LLMs locally with Ollama
- **Documentation:** https://docs.litellm.ai/docs/providers/ollama
- **Auth:** Local setup (no auth needed)
- **Features:** Chat completions, embeddings, local inference

### vLLM
- **Details:** Efficient LLM serving framework for high-throughput inference
- **Documentation:** https://docs.litellm.ai/docs/providers/vllm
- **Auth:** Endpoint URL based
- **Features:** Chat completions, streaming

### LM Studio
- **Details:** Run open-source models locally with UI
- **Documentation:** https://docs.litellm.ai/docs/providers/lm-studio
- **Auth:** Local setup
- **Features:** Chat completions, local inference

### Llamafile
- **Details:** Run models as standalone executables
- **Documentation:** https://docs.litellm.ai/docs/providers/llamafile
- **Auth:** Local setup
- **Features:** Chat completions

### Hugging Face
- **Details:** Access to models from Hugging Face Hub
- **Documentation:** https://docs.litellm.ai/docs/providers/huggingface
- **Auth:** API Token (optional for public models)
- **Features:** Model inference via Hugging Face APIs

### NVIDIA NIM
- **Details:** NVIDIA's NIM microservices for AI inference
- **Documentation:** https://docs.litellm.ai/docs/providers/nvidia_nim
- **Auth:** API Key, local endpoint setup
- **Features:** Chat completions, embeddings

---

## Meta & Open Source Models

### Meta Llama API
- **Details:** Access to Meta's Llama models through Llama API
- **Documentation:** https://docs.litellm.ai/docs/providers/llama_api
- **Auth:** API Key
- **Features:** Chat completions

---

## Enterprise & Data Platforms

### Databricks
- **Details:** Foundation models served through Databricks
- **Documentation:** https://docs.litellm.ai/docs/providers/databricks
- **Auth:** Databricks API credentials
- **Features:** Chat completions, embeddings

### Snowflake
- **Details:** Access to models through Snowflake cortex API
- **Documentation:** https://docs.litellm.ai/docs/providers/snowflake
- **Auth:** Snowflake credentials
- **Features:** Chat completions

### IBM Watsonx
- **Details:** Enterprise AI platform by IBM
- **Documentation:** https://docs.litellm.ai/docs/providers/ibm
- **Auth:** IBM Cloud API Key, Project ID
- **Features:** Chat completions, embeddings

### Oracle Cloud (OCI)
- **Details:** Language models available through Oracle Cloud
- **Documentation:** https://docs.litellm.ai/docs/providers/oci
- **Auth:** OCI credentials
- **Features:** Chat completions

### Volcengine (ByteDance)
- **Details:** ByteDance's cloud AI services
- **Documentation:** https://docs.litellm.ai/docs/providers/volcengine
- **Auth:** API Key
- **Features:** Chat completions

---

## Additional & Regional Providers

### NLP Cloud
- **Details:** Easy-to-use API for various NLP models
- **Documentation:** https://docs.litellm.ai/docs/providers/nlpcloud
- **Auth:** API Token
- **Features:** Chat completions, embeddings

### SAP Generative AI Hub
- **Details:** SAP's enterprise AI hub
- **Documentation:** https://docs.litellm.ai/docs/providers/sap
- **Auth:** SAP cloud credentials
- **Features:** Model access through SAP platform

### DataRobot
- **Details:** ML operations and deployment platform
- **Documentation:** https://docs.litellm.ai/docs/providers/datarobot
- **Auth:** API credentials
- **Features:** Model inference

### Anyscale
- **Details:** Ray-based serving platform
- **Documentation:** https://docs.litellm.ai/docs/providers/anyscale
- **Auth:** API Key
- **Features:** Chat completions

### Azure AI
- **Details:** Microsoft's comprehensive AI platform
- **Documentation:** https://docs.litellm.ai/docs/providers/azure_ai
- **Auth:** Azure credentials
- **Features:** Various AI services integration

---

## Multi-Provider Gateways

### OpenRouter
- **Details:** Access to 100+ models across multiple providers (Anthropic, OpenAI, Google, Meta, Mistral, etc.)
- **Documentation:** https://docs.litellm.ai/docs/providers/openrouter
- **Auth:** `OPENROUTER_API_KEY`
- **Features:** Unified access to multiple providers, load balancing across providers
- **Usage:** `model="openrouter/<model-name>"`

---

## Custom & OpenAI-Compatible Providers

### OpenAI-Compatible Endpoints
- **Details:** Any OpenAI-compatible LLM endpoint can be integrated
- **Documentation:** https://docs.litellm.ai/docs/providers/openai_compatible
- **Auth:** API Key (provider-dependent)
- **Features:** Full endpoint support for compatible services
- **Note:** Add support by editing a single JSON file for straightforward integrations

---

## Key Features Across Providers

### Supported Endpoints
LiteLLM providers support various endpoints:
- `/chat/completions` - Chat-based interactions
- `/messages` - Message API (Anthropic-style)
- `/responses` - Response generation
- `/embeddings` - Text embeddings
- `/image/generations` - Image generation
- `/audio/transcriptions` - Audio transcription
- `/audio/speech` - Text-to-speech
- `/moderations` - Content moderation
- `/batches` - Batch processing
- `/rerank` - Reranking

### Common Features
- **Fallback support**: Automatic fallback to alternative providers
- **Retry logic**: Built-in retry mechanisms
- **Cost tracking**: Monitor spending across providers
- **Load balancing**: Distribute requests across multiple providers
- **Streaming**: Real-time response streaming where available
- **Vision support**: Image/vision capabilities for compatible models

---

## Provider Registration

To integrate a new provider:
- **Documentation**: https://docs.litellm.ai/docs/provider_registration/
- Contributers can add custom providers by following the registration process
- Share models pricing and context window information

---

## Getting Started

1. **Install LiteLLM**:
   ```bash
   pip install litellm
   ```

2. **Basic Usage**:
   ```python
   from litellm import completion

   response = completion(
       model="openai/gpt-4",
       messages=[{"role": "user", "content": "Hello!"}],
       api_key="your-api-key"
   )
   ```

3. **Using Different Providers**:
   ```python
   # Azure
   response = completion(model="azure/gpt-4", ...)

   # Anthropic Claude
   response = completion(model="claude-3-sonnet", ...)

   # Groq
   response = completion(model="groq/mixtral-8x7b-32768", ...)
   ```

---

## Resources

- **Official Docs**: https://docs.litellm.ai/
- **GitHub Repository**: https://github.com/BerriAI/litellm
- **Models Database**: https://models.litellm.ai/
- **PyPI Package**: https://pypi.org/project/litellm/
- **GitHub Issues**: https://github.com/BerriAI/litellm/issues

---

## Last Updated
December 2024 - Based on LiteLLM documentation v1.0+

*Note: This list contains 50+ providers. For the most current list and specific model availability, visit the [LiteLLM documentation](https://docs.litellm.ai/docs/providers) or check the [Models Database](https://models.litellm.ai/).*
