# Supported Providers

LLMKit supports **100+ LLM providers** through a unified interface.

## Quick Reference

| Provider | Environment Variable | Example Model |
|----------|---------------------|---------------|
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4-20250514` |
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o` |
| Google AI | `GOOGLE_API_KEY` | `google/gemini-2.0-flash` |
| AWS Bedrock | `AWS_ACCESS_KEY_ID` | `bedrock/anthropic.claude-3-sonnet` |
| Azure OpenAI | `AZURE_API_KEY` | `azure/gpt-4o` |
| Google Vertex AI | `GOOGLE_APPLICATION_CREDENTIALS` | `vertex/gemini-pro` |
| Mistral | `MISTRAL_API_KEY` | `mistral/mistral-large` |
| Groq | `GROQ_API_KEY` | `groq/llama-3.3-70b-versatile` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek/deepseek-chat` |
| Cohere | `COHERE_API_KEY` | `cohere/command-r-plus` |

## All Providers by Category

### Major Cloud Providers

| Provider | Env Variable | Features |
|----------|-------------|----------|
| **Anthropic** | `ANTHROPIC_API_KEY` | Streaming, Tools, Vision, Caching, Thinking |
| **OpenAI** | `OPENAI_API_KEY` | Streaming, Tools, Vision, Structured Output |
| **Google AI** | `GOOGLE_API_KEY` | Streaming, Tools, Vision, Caching |
| **AWS Bedrock** | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Streaming, Tools, Vision |
| **Azure OpenAI** | `AZURE_API_KEY`, `AZURE_ENDPOINT` | Streaming, Tools, Vision |
| **Google Vertex AI** | `GOOGLE_APPLICATION_CREDENTIALS` | Streaming, Tools, Vision, Thinking |

### Inference Platforms

| Provider | Env Variable | Features |
|----------|-------------|----------|
| **Together AI** | `TOGETHER_API_KEY` | Streaming, Tools, Vision |
| **Fireworks** | `FIREWORKS_API_KEY` | Streaming, Tools, Vision |
| **Groq** | `GROQ_API_KEY` | Streaming, Tools (ultra-fast) |
| **Cerebras** | `CEREBRAS_API_KEY` | Streaming, Tools |
| **SambaNova** | `SAMBANOVA_API_KEY` | Streaming |
| **DeepInfra** | `DEEPINFRA_API_KEY` | Streaming, Tools, Vision |
| **Anyscale** | `ANYSCALE_API_KEY` | Streaming, Tools |
| **Replicate** | `REPLICATE_API_TOKEN` | Streaming |
| **RunPod** | `RUNPOD_API_KEY` | Streaming |
| **OctoAI** | `OCTOAI_API_KEY` | Streaming, Tools, Vision |
| **Lepton** | `LEPTON_API_KEY` | Streaming |

### Specialized AI Providers

| Provider | Env Variable | Features |
|----------|-------------|----------|
| **Mistral** | `MISTRAL_API_KEY` | Streaming, Tools, Code |
| **DeepSeek** | `DEEPSEEK_API_KEY` | Streaming, Tools, Thinking |
| **Cohere** | `COHERE_API_KEY` | Streaming, Tools, RAG |
| **AI21** | `AI21_API_KEY` | Streaming |
| **Perplexity** | `PERPLEXITY_API_KEY` | Streaming, Search-augmented |
| **Writer** | `WRITER_API_KEY` | Streaming, Tools |
| **xAI (Grok)** | `XAI_API_KEY` | Streaming, Tools, Vision |
| **Aleph Alpha** | `ALEPH_ALPHA_API_KEY` | Streaming |

### Aggregators & Routers

| Provider | Env Variable | Features |
|----------|-------------|----------|
| **OpenRouter** | `OPENROUTER_API_KEY` | Access to 100+ models |
| **Unify** | `UNIFY_API_KEY` | Smart routing |
| **Portkey** | `PORTKEY_API_KEY` | Gateway with observability |
| **Helicone** | `HELICONE_API_KEY` | Logging & analytics |
| **SiliconFlow** | `SILICONFLOW_API_KEY` | Chinese model access |

### Enterprise Platforms

| Provider | Env Variable | Features |
|----------|-------------|----------|
| **Databricks** | `DATABRICKS_TOKEN` | Streaming, Tools |
| **Snowflake** | `SNOWFLAKE_API_KEY` | Enterprise |
| **IBM WatsonX** | `WATSONX_API_KEY` | Enterprise |
| **SAP AI Core** | `SAP_AI_CORE_API_KEY` | Enterprise |
| **DataRobot** | `DATAROBOT_API_KEY` | MLOps |
| **Oracle OCI** | `OCI_API_KEY` | Enterprise |

### Chinese Providers

| Provider | Env Variable | Features |
|----------|-------------|----------|
| **Alibaba (Qwen)** | `DASHSCOPE_API_KEY` | Streaming, Tools, Vision |
| **Baidu (ERNIE)** | `BAIDU_API_KEY` | Streaming, Tools |
| **Zhipu (GLM)** | `ZHIPU_API_KEY` | Streaming, Tools, Vision |
| **Moonshot** | `MOONSHOT_API_KEY` | Streaming, Tools |
| **MiniMax** | `MINIMAX_API_KEY` | Streaming, Tools |
| **Yi** | `YI_API_KEY` | Streaming, Tools, Vision |
| **Baichuan** | `BAICHUAN_API_KEY` | Streaming |
| **Stepfun** | `STEPFUN_API_KEY` | Streaming |
| **Volcengine** | `VOLC_ACCESSKEY` | Streaming, Tools, Vision |
| **Hunyuan** | `HUNYUAN_API_KEY` | Streaming |
| **Spark (iFlytek)** | `SPARK_API_KEY` | Streaming |

### Regional Providers

| Provider | Region | Env Variable |
|----------|--------|-------------|
| **Maritaca** | Brazil | `MARITALK_API_KEY` |
| **Naver Clova** | Korea | `CLOVASTUDIO_API_KEY` |
| **Upstage** | Korea | `UPSTAGE_API_KEY` |
| **Yandex** | Russia | `YANDEX_API_KEY` |
| **GigaChat** | Russia | `GIGACHAT_API_KEY` |
| **SEA-LION** | Singapore | `SEA_LION_API_KEY` |
| **Kakao** | Korea | `KAKAO_API_KEY` |
| **Sarvam** | India | `SARVAM_API_KEY` |
| **Krutrim** | India | `KRUTRIM_API_KEY` |

### European Providers

| Provider | Env Variable | Features |
|----------|-------------|----------|
| **LightOn** | `LIGHTON_API_KEY` | GDPR-compliant |
| **IONOS** | `IONOS_API_KEY` | EU hosting |
| **Scaleway** | `SCALEWAY_API_KEY` | EU hosting |
| **OVHCloud** | `OVH_API_KEY` | EU hosting |
| **Tilde** | `TILDE_API_KEY` | Baltic languages |
| **SiloAI** | `SILOAI_API_KEY` | Nordic |
| **SwissAI** | `SWISSAI_API_KEY` | Swiss hosting |

### Local & Self-Hosted

| Provider | Default URL | Features |
|----------|-------------|----------|
| **Ollama** | `http://localhost:11434` | Local models |
| **LM Studio** | `http://localhost:1234` | Local models |
| **vLLM** | `http://localhost:8000` | High-performance serving |
| **LocalAI** | `http://localhost:8080` | OpenAI-compatible |
| **Llamafile** | `http://localhost:8080` | Single-file LLMs |
| **Jan** | `http://localhost:1337` | Desktop app |
| **Xinference** | `http://localhost:9997` | Distributed inference |
| **TGI** | `http://localhost:8080` | HuggingFace serving |
| **GPT4All** | Local | Desktop app |
| **Petals** | Distributed | Collaborative inference |

### Audio Providers

| Provider | Env Variable | Features |
|----------|-------------|----------|
| **Deepgram** | `DEEPGRAM_API_KEY` | Speech-to-text |
| **ElevenLabs** | `ELEVENLABS_API_KEY` | Text-to-speech, Voice cloning |
| **AssemblyAI** | `ASSEMBLYAI_API_KEY` | Transcription, Speaker diarization |
| **PlayHT** | `PLAYHT_API_KEY` | Text-to-speech |
| **Resemble** | `RESEMBLE_API_KEY` | Voice cloning |
| **Rev AI** | `REVAI_API_KEY` | Transcription |
| **Speechmatics** | `SPEECHMATICS_API_KEY` | Transcription |

### Image Providers

| Provider | Env Variable | Features |
|----------|-------------|----------|
| **Stability AI** | `STABILITY_API_KEY` | Image generation |
| **Fal** | `FAL_API_KEY` | Fast image generation |
| **Leonardo** | `LEONARDO_API_KEY` | Image generation |
| **Ideogram** | `IDEOGRAM_API_KEY` | Text-in-image |
| **Black Forest Labs** | `BFL_API_KEY` | FLUX models |
| **Recraft** | `RECRAFT_API_KEY` | Professional images |
| **Clarifai** | `CLARIFAI_API_KEY` | Vision AI |

### Video Providers

| Provider | Env Variable | Features |
|----------|-------------|----------|
| **RunwayML** | `RUNWAYML_API_KEY` | Video generation |
| **Pika** | `PIKA_API_KEY` | Video generation |
| **Luma** | `LUMA_API_KEY` | Video generation |
| **Kling** | `KLING_API_KEY` | Video generation |
| **HeyGen** | `HEYGEN_API_KEY` | Avatar videos |
| **D-ID** | `DID_API_KEY` | Avatar videos |
| **Twelve Labs** | `TWELVE_LABS_API_KEY` | Video understanding |
| **Runware** | `RUNWARE_API_KEY` | Video aggregator |

### Embedding Providers

| Provider | Env Variable | Features |
|----------|-------------|----------|
| **Voyage AI** | `VOYAGE_API_KEY` | High-quality embeddings |
| **Jina AI** | `JINA_API_KEY` | Multilingual embeddings |
| **Cohere** | `COHERE_API_KEY` | Embed + Rerank |

## Usage

### Single Provider

```rust
use llmkit::LLMKitClient;

// Using environment variable
let client = LLMKitClient::from_env()?;
let response = client.complete(request).await?;
```

### Multiple Providers

```rust
use llmkit::ClientBuilder;

let client = ClientBuilder::new()
    .with_anthropic_from_env()?
    .with_openai_from_env()?
    .with_groq_from_env()?
    .build()?;

// Route to different providers using "provider/model" format
let response = client.complete("anthropic/claude-sonnet-4-20250514", messages).await?;
let response = client.complete("openai/gpt-4o", messages).await?;
let response = client.complete("groq/llama-3.3-70b-versatile", messages).await?;
```

### Local Models

```rust
let client = ClientBuilder::new()
    .with_ollama_url("http://localhost:11434")?
    .build()?;

let response = client.complete("ollama/llama3.2", messages).await?;
```

## Feature Flags

Enable only the providers you need:

```toml
[dependencies]
llmkit = { version = "0.1", features = ["anthropic", "openai", "groq"] }

# Or enable all
llmkit = { version = "0.1", features = ["all-providers"] }
```
