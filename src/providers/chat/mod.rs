//! Chat and completion providers.
//!
//! This module contains providers that implement the standard Provider trait
//! for chat/completion requests across various LLM services.

#[cfg(feature = "anthropic")]
pub mod anthropic;

#[cfg(feature = "openai")]
pub mod openai;

#[cfg(feature = "openrouter")]
pub mod openrouter;

#[cfg(feature = "ollama")]
pub mod ollama;

#[cfg(feature = "groq")]
pub mod groq;

#[cfg(feature = "mistral")]
pub mod mistral;

#[cfg(feature = "azure")]
pub mod azure;

#[cfg(feature = "bedrock")]
pub mod bedrock;

#[cfg(feature = "openai-compatible")]
pub mod openai_compatible;

#[cfg(feature = "google")]
pub mod google;

#[cfg(feature = "vertex")]
pub mod vertex;

#[cfg(feature = "cohere")]
pub mod cohere;

#[cfg(feature = "ai21")]
pub mod ai21;

#[cfg(feature = "huggingface")]
pub mod huggingface;

#[cfg(feature = "replicate")]
pub mod replicate;

#[cfg(feature = "baseten")]
pub mod baseten;

#[cfg(feature = "runpod")]
pub mod runpod;

#[cfg(feature = "cloudflare")]
pub mod cloudflare;

#[cfg(feature = "watsonx")]
pub mod watsonx;

#[cfg(feature = "databricks")]
pub mod databricks;

#[cfg(feature = "datarobot")]
pub mod datarobot;

#[cfg(feature = "cerebras")]
pub mod cerebras;

#[cfg(feature = "sagemaker")]
pub mod sagemaker;

#[cfg(feature = "snowflake")]
pub mod snowflake;

#[cfg(feature = "sambanova")]
pub mod sambanova;

#[cfg(feature = "fireworks")]
pub mod fireworks;

#[cfg(feature = "deepseek")]
pub mod deepseek;

#[cfg(feature = "aleph-alpha")]
pub mod aleph_alpha;

#[cfg(feature = "nlp-cloud")]
pub mod nlp_cloud;

#[cfg(feature = "yandex")]
pub mod yandex;

#[cfg(feature = "gigachat")]
pub mod gigachat;

#[cfg(feature = "clova")]
pub mod clova;

#[cfg(feature = "maritaca")]
pub mod maritaca;

#[cfg(feature = "writer")]
pub mod writer;

#[cfg(feature = "perplexity")]
pub mod perplexity;

#[cfg(feature = "baidu")]
pub mod baidu;

#[cfg(feature = "alibaba")]
pub mod alibaba;

#[cfg(feature = "vllm")]
pub mod vllm;

#[cfg(feature = "oracle")]
pub mod oracle;

#[cfg(feature = "sap")]
pub mod sap;

#[cfg(feature = "xai")]
pub mod xai;

#[cfg(feature = "deepinfra")]
pub mod deepinfra;

#[cfg(feature = "nvidia-nim")]
pub mod nvidia_nim;

// Tier 1 - High Priority Inference
#[cfg(feature = "anyscale")]
pub mod anyscale;

#[cfg(feature = "github")]
pub mod github_models;

#[cfg(feature = "friendli")]
pub mod friendli;

#[cfg(feature = "hyperbolic")]
pub mod hyperbolic;

#[cfg(feature = "lambda")]
pub mod lambda_ai;

#[cfg(feature = "novita")]
pub mod novita;

#[cfg(feature = "nebius")]
pub mod nebius;

#[cfg(feature = "lepton")]
pub mod lepton;

#[cfg(feature = "stability")]
pub mod stability;

#[cfg(feature = "voyage")]
pub mod voyage;

#[cfg(feature = "jina")]
pub mod jina;

#[cfg(feature = "deepgram")]
pub mod deepgram;

#[cfg(feature = "elevenlabs")]
pub mod elevenlabs;

#[cfg(feature = "gpt4all")]
pub mod gpt4all;

// Tier 2 - Chinese Providers
#[cfg(feature = "minimax")]
pub mod minimax;

#[cfg(feature = "moonshot")]
pub mod moonshot;

#[cfg(feature = "zhipu")]
pub mod zhipu;

#[cfg(feature = "volcengine")]
pub mod volcengine;

#[cfg(feature = "baichuan")]
pub mod baichuan_ai;

#[cfg(feature = "stepfun")]
pub mod stepfun;

#[cfg(feature = "yi")]
pub mod yi;

#[cfg(feature = "spark")]
pub mod spark;

// Tier 3 - Local/Self-Hosted
#[cfg(feature = "lm-studio")]
pub mod lm_studio;

#[cfg(feature = "llamafile")]
pub mod llamafile;

#[cfg(feature = "xinference")]
pub mod xinference;

#[cfg(feature = "localai")]
pub mod localai;

#[cfg(feature = "jan")]
pub mod jan;

#[cfg(feature = "petals")]
pub mod petals;

#[cfg(feature = "triton")]
pub mod triton;

#[cfg(feature = "tgi")]
pub mod tgi;

// Tier 4 - Enterprise/Specialized
#[cfg(feature = "predibase")]
pub mod predibase;

#[cfg(feature = "octoai")]
pub mod octoai;

#[cfg(feature = "featherless")]
pub mod featherless;

#[cfg(feature = "ovhcloud")]
pub mod ovhcloud;

#[cfg(feature = "scaleway")]
pub mod scaleway;

#[cfg(feature = "crusoe")]
pub mod crusoe;

#[cfg(feature = "cerebrium")]
pub mod cerebrium;

#[cfg(feature = "lightning")]
pub mod lightning;

#[cfg(feature = "assemblyai")]
pub mod assemblyai;

#[cfg(feature = "runwayml")]
pub mod runwayml;

// Tier 5 - Asian Regional Providers
#[cfg(feature = "naver")]
pub mod naver;

#[cfg(feature = "kakao")]
pub mod kakao;

#[cfg(feature = "lg-exaone")]
pub mod lg_exaone;

#[cfg(feature = "plamo")]
pub mod plamo;

#[cfg(feature = "sarvam")]
pub mod sarvam;

#[cfg(feature = "krutrim")]
pub mod krutrim;

#[cfg(feature = "ntt")]
pub mod ntt;

#[cfg(feature = "softbank")]
pub mod softbank;

// Tier 6 - European Sovereign AI
#[cfg(feature = "ionos")]
pub mod ionos;

#[cfg(feature = "tilde")]
pub mod tilde;

#[cfg(feature = "silo-ai")]
pub mod silo_ai;

#[cfg(feature = "swiss-ai")]
pub mod swiss_ai;

// Tier 7 - Router/Gateway/Meta Providers
#[cfg(feature = "unify")]
pub mod unify;

#[cfg(feature = "martian")]
pub mod martian;

#[cfg(feature = "portkey")]
pub mod portkey;

#[cfg(feature = "helicone")]
pub mod helicone;

#[cfg(feature = "siliconflow")]
pub mod siliconflow;

// Tier 8 - Video AI Providers
#[cfg(feature = "pika")]
pub mod pika;

#[cfg(feature = "luma")]
pub mod luma;

#[cfg(feature = "kling")]
pub mod kling;

#[cfg(feature = "heygen")]
pub mod heygen;

#[cfg(feature = "did")]
pub mod did;

#[cfg(feature = "twelve-labs")]
pub mod twelve_labs;

// Tier 9 - Audio AI Providers
#[cfg(feature = "rev")]
pub mod rev;

#[cfg(feature = "speechmatics")]
pub mod speechmatics;

#[cfg(feature = "playht")]
pub mod playht;

#[cfg(feature = "resemble")]
pub mod resemble;

// Tier 10 - Image AI Providers
#[cfg(feature = "leonardo")]
pub mod leonardo;

#[cfg(feature = "ideogram")]
pub mod ideogram;

#[cfg(feature = "black-forest-labs")]
pub mod black_forest_labs;

#[cfg(feature = "clarifai")]
pub mod clarifai;

#[cfg(feature = "fal")]
pub mod fal;

// Tier 11 - Infrastructure Providers
#[cfg(feature = "modal")]
pub mod modal;

#[cfg(feature = "coreweave")]
pub mod coreweave;

#[cfg(feature = "tensordock")]
pub mod tensordock;

#[cfg(feature = "beam")]
pub mod beam;

#[cfg(feature = "vastai")]
pub mod vastai;

// Tier 12 - Emerging Startups
#[cfg(feature = "nscale")]
pub mod nscale;

#[cfg(feature = "runware")]
pub mod runware;

#[cfg(feature = "ai71")]
pub mod ai71;

// Contingent providers (pending API access)
pub mod chatlaw;
pub mod latamgpt;
pub mod lighton;

// Re-exports for convenience

#[cfg(feature = "anthropic")]
pub use anthropic::AnthropicProvider;

#[cfg(feature = "openai")]
pub use openai::OpenAIProvider;

#[cfg(feature = "openrouter")]
pub use openrouter::OpenRouterProvider;

#[cfg(feature = "ollama")]
pub use ollama::OllamaProvider;

#[cfg(feature = "groq")]
pub use groq::GroqProvider;

#[cfg(feature = "mistral")]
pub use mistral::{MistralConfig, MistralProvider, MistralRegion};

#[cfg(feature = "azure")]
pub use azure::{AzureConfig, AzureOpenAIProvider};

#[cfg(feature = "bedrock")]
pub use bedrock::{BedrockBuilder, BedrockConfig, BedrockProvider, ModelFamily};

#[cfg(feature = "openai-compatible")]
pub use openai_compatible::{known_providers, OpenAICompatibleProvider, ProviderInfo};

#[cfg(feature = "google")]
pub use google::GoogleProvider;

#[cfg(feature = "vertex")]
pub use vertex::{VertexConfig, VertexProvider};

#[cfg(feature = "cohere")]
pub use cohere::CohereProvider;

#[cfg(feature = "ai21")]
pub use ai21::AI21Provider;

#[cfg(feature = "huggingface")]
pub use huggingface::HuggingFaceProvider;

#[cfg(feature = "replicate")]
pub use replicate::ReplicateProvider;

#[cfg(feature = "baseten")]
pub use baseten::BasetenProvider;

#[cfg(feature = "runpod")]
pub use runpod::RunPodProvider;

#[cfg(feature = "cloudflare")]
pub use cloudflare::CloudflareProvider;

#[cfg(feature = "watsonx")]
pub use watsonx::WatsonxProvider;

#[cfg(feature = "databricks")]
pub use databricks::DatabricksProvider;

#[cfg(feature = "datarobot")]
pub use datarobot::DataRobotProvider;

#[cfg(feature = "cerebras")]
pub use cerebras::CerebrasProvider;

#[cfg(feature = "sagemaker")]
pub use sagemaker::SageMakerProvider;

#[cfg(feature = "snowflake")]
pub use snowflake::SnowflakeProvider;

#[cfg(feature = "sambanova")]
pub use sambanova::SambaNovaProvider;

#[cfg(feature = "fireworks")]
pub use fireworks::FireworksProvider;

#[cfg(feature = "deepseek")]
pub use deepseek::DeepSeekProvider;

#[cfg(feature = "aleph-alpha")]
pub use aleph_alpha::AlephAlphaProvider;

#[cfg(feature = "nlp-cloud")]
pub use nlp_cloud::NlpCloudProvider;

#[cfg(feature = "yandex")]
pub use yandex::YandexProvider;

#[cfg(feature = "gigachat")]
pub use gigachat::GigaChatProvider;

#[cfg(feature = "clova")]
pub use clova::ClovaProvider;

#[cfg(feature = "maritaca")]
pub use maritaca::MaritacaProvider;

#[cfg(feature = "writer")]
pub use writer::WriterProvider;

#[cfg(feature = "perplexity")]
pub use perplexity::{
    Citation, PerplexityModelInfo, PerplexityProvider, PerplexitySearchMode,
    SearchAugmentedResponse,
};

#[cfg(feature = "baidu")]
pub use baidu::{ApiVersion, BaiduModelInfo, BaiduProvider};

#[cfg(feature = "alibaba")]
pub use alibaba::{AlibabaModelInfo, AlibabaProvider, ModelSpecialization};

#[cfg(feature = "vllm")]
pub use vllm::{SchedulingPolicy, ServerStats, VLLMProvider};

#[cfg(feature = "oracle")]
pub use oracle::{DeploymentType, OracleEndpointConfig, OracleModelInfo, OracleOCIProvider};

#[cfg(feature = "sap")]
pub use sap::{IntegrationType, SAPConsumptionPlan, SAPGenerativeAIProvider, SAPModelInfo};

#[cfg(feature = "xai")]
pub use xai::{XAIModelInfo, XAIProvider};

#[cfg(feature = "deepinfra")]
pub use deepinfra::{DeepInfraModelInfo, DeepInfraProvider};

#[cfg(feature = "nvidia-nim")]
pub use nvidia_nim::{NvidiaNIMModelInfo, NvidiaNIMProvider};

// Tier 1 - High Priority Inference re-exports
#[cfg(feature = "anyscale")]
pub use anyscale::{AnyscaleModelInfo, AnyscaleProvider};

#[cfg(feature = "github")]
pub use github_models::{GitHubModelInfo, GitHubModelsProvider};

#[cfg(feature = "friendli")]
pub use friendli::{FriendliModelInfo, FriendliProvider};

#[cfg(feature = "hyperbolic")]
pub use hyperbolic::{HyperbolicModelInfo, HyperbolicProvider};

#[cfg(feature = "lambda")]
pub use lambda_ai::{LambdaModelInfo, LambdaProvider};

#[cfg(feature = "novita")]
pub use novita::{NovitaModelInfo, NovitaProvider};

#[cfg(feature = "nebius")]
pub use nebius::{NebiusModelInfo, NebiusProvider};

#[cfg(feature = "lepton")]
pub use lepton::{LeptonModelInfo, LeptonProvider};

#[cfg(feature = "stability")]
pub use stability::{StabilityModelInfo, StabilityProvider};

#[cfg(feature = "voyage")]
pub use voyage::{VoyageModelInfo, VoyageProvider};

#[cfg(feature = "jina")]
pub use jina::{JinaModelInfo, JinaProvider};

#[cfg(feature = "deepgram")]
pub use deepgram::{DeepgramModelInfo, DeepgramProvider};

#[cfg(feature = "elevenlabs")]
pub use elevenlabs::{ElevenLabsModelInfo, ElevenLabsProvider};

#[cfg(feature = "gpt4all")]
pub use gpt4all::{GPT4AllModelInfo, GPT4AllProvider};

// Tier 2 - Chinese Providers re-exports
#[cfg(feature = "minimax")]
pub use minimax::{MiniMaxModelInfo, MiniMaxProvider};

#[cfg(feature = "moonshot")]
pub use moonshot::{MoonshotModelInfo, MoonshotProvider};

#[cfg(feature = "zhipu")]
pub use zhipu::{ZhipuModelInfo, ZhipuProvider};

#[cfg(feature = "volcengine")]
pub use volcengine::{VolcengineModelInfo, VolcengineProvider};

#[cfg(feature = "baichuan")]
pub use baichuan_ai::{BaichuanModelInfo, BaichuanProvider};

#[cfg(feature = "stepfun")]
pub use stepfun::{StepfunModelInfo, StepfunProvider};

#[cfg(feature = "yi")]
pub use yi::{YiModelInfo, YiProvider};

#[cfg(feature = "spark")]
pub use spark::{SparkModelInfo, SparkProvider};

// Tier 3 - Local/Self-Hosted re-exports
#[cfg(feature = "lm-studio")]
pub use lm_studio::{LMStudioModelInfo, LMStudioProvider};

#[cfg(feature = "llamafile")]
pub use llamafile::{LlamafileModelInfo, LlamafileProvider};

#[cfg(feature = "xinference")]
pub use xinference::{XinferenceModelInfo, XinferenceProvider};

#[cfg(feature = "localai")]
pub use localai::{LocalAIModelInfo, LocalAIProvider};

#[cfg(feature = "jan")]
pub use jan::{JanModelInfo, JanProvider};

#[cfg(feature = "petals")]
pub use petals::{PetalsModelInfo, PetalsProvider};

#[cfg(feature = "triton")]
pub use triton::{TritonModelInfo, TritonProvider};

#[cfg(feature = "tgi")]
pub use tgi::{TGIModelInfo, TGIProvider};

// Tier 4 - Enterprise/Specialized re-exports
#[cfg(feature = "predibase")]
pub use predibase::{PredibaseModelInfo, PredibaseProvider};

#[cfg(feature = "octoai")]
pub use octoai::{OctoAIModelInfo, OctoAIProvider};

#[cfg(feature = "featherless")]
pub use featherless::{FeatherlessModelInfo, FeatherlessProvider};

#[cfg(feature = "ovhcloud")]
pub use ovhcloud::{OVHCloudModelInfo, OVHCloudProvider};

#[cfg(feature = "scaleway")]
pub use scaleway::{ScalewayModelInfo, ScalewayProvider};

#[cfg(feature = "crusoe")]
pub use crusoe::{CrusoeModelInfo, CrusoeProvider};

#[cfg(feature = "cerebrium")]
pub use cerebrium::{CerebriumModelInfo, CerebriumProvider};

#[cfg(feature = "lightning")]
pub use lightning::{LightningModelInfo, LightningProvider};

#[cfg(feature = "assemblyai")]
pub use assemblyai::{AssemblyAIModelInfo, AssemblyAIProvider};

#[cfg(feature = "runwayml")]
pub use runwayml::{RunwayMLModelInfo, RunwayMLProvider};

// Tier 5 - Asian Regional Providers re-exports
#[cfg(feature = "naver")]
pub use naver::{NaverModelInfo, NaverProvider};

#[cfg(feature = "kakao")]
pub use kakao::{KakaoModelInfo, KakaoProvider};

#[cfg(feature = "lg-exaone")]
pub use lg_exaone::{LGExaoneModelInfo, LGExaoneProvider};

#[cfg(feature = "plamo")]
pub use plamo::{PLaMoModelInfo, PLaMoProvider};

#[cfg(feature = "sarvam")]
pub use sarvam::{SarvamModelInfo, SarvamProvider};

#[cfg(feature = "krutrim")]
pub use krutrim::{KrutrimModelInfo, KrutrimProvider};

#[cfg(feature = "ntt")]
pub use ntt::{NTTModelInfo, NTTProvider};

#[cfg(feature = "softbank")]
pub use softbank::{SoftBankModelInfo, SoftBankProvider};

// Tier 6 - European Sovereign AI re-exports
#[cfg(feature = "ionos")]
pub use ionos::{IONOSModelInfo, IONOSProvider};

#[cfg(feature = "tilde")]
pub use tilde::{TildeModelInfo, TildeProvider};

#[cfg(feature = "silo-ai")]
pub use silo_ai::{SiloAIModelInfo, SiloAIProvider};

#[cfg(feature = "swiss-ai")]
pub use swiss_ai::{SwissAIModelInfo, SwissAIProvider};

// Tier 7 - Router/Gateway/Meta Providers re-exports
#[cfg(feature = "unify")]
pub use unify::{UnifyProvider, UnifyRouterInfo};

#[cfg(feature = "martian")]
pub use martian::{MartianCapabilities, MartianProvider};

#[cfg(feature = "portkey")]
pub use portkey::{PortkeyCapabilities, PortkeyProvider};

#[cfg(feature = "helicone")]
pub use helicone::{HeliconeFeatures, HeliconeProvider};

#[cfg(feature = "siliconflow")]
pub use siliconflow::{SiliconFlowModelInfo, SiliconFlowProvider};

// Tier 8 - Video AI Providers re-exports
#[cfg(feature = "pika")]
pub use pika::{PikaModelInfo, PikaProvider};

#[cfg(feature = "luma")]
pub use luma::{LumaModelInfo, LumaProvider};

#[cfg(feature = "kling")]
pub use kling::{KlingModelInfo, KlingProvider};

#[cfg(feature = "heygen")]
pub use heygen::{HeyGenFeatureInfo, HeyGenProvider};

#[cfg(feature = "did")]
pub use did::{DIDFeatureInfo, DIDProvider};

#[cfg(feature = "twelve-labs")]
pub use twelve_labs::{TwelveLabsModelInfo, TwelveLabsProvider};

// Tier 9 - Audio AI Providers re-exports
#[cfg(feature = "rev")]
pub use rev::{RevProvider, RevServiceInfo};

#[cfg(feature = "speechmatics")]
pub use speechmatics::{SpeechmaticsProvider, SpeechmaticsServiceInfo};

#[cfg(feature = "playht")]
pub use playht::{PlayHTModelInfo, PlayHTProvider};

#[cfg(feature = "resemble")]
pub use resemble::{ResembleFeatureInfo, ResembleProvider};

// Tier 10 - Image AI Providers re-exports
#[cfg(feature = "leonardo")]
pub use leonardo::{LeonardoModelInfo, LeonardoProvider};

#[cfg(feature = "ideogram")]
pub use ideogram::{IdeogramModelInfo, IdeogramProvider};

#[cfg(feature = "black-forest-labs")]
pub use black_forest_labs::{BlackForestLabsModelInfo, BlackForestLabsProvider};

#[cfg(feature = "clarifai")]
pub use clarifai::{ClarifaiCapabilityInfo, ClarifaiProvider};

#[cfg(feature = "fal")]
pub use fal::{FalModelInfo, FalProvider};

// Tier 11 - Infrastructure Providers re-exports
#[cfg(feature = "modal")]
pub use modal::{ModalCapabilities, ModalGPUInfo, ModalProvider};

#[cfg(feature = "coreweave")]
pub use coreweave::{CoreWeaveCapabilities, CoreWeaveGPUInfo, CoreWeaveProvider};

#[cfg(feature = "tensordock")]
pub use tensordock::{TensorDockCapabilities, TensorDockGPUInfo, TensorDockProvider};

#[cfg(feature = "beam")]
pub use beam::{BeamCapabilities, BeamGPUInfo, BeamProvider};

#[cfg(feature = "vastai")]
pub use vastai::{VastAICapabilities, VastAIGPUInfo, VastAIProvider};

// Tier 12 - Emerging Startups re-exports
#[cfg(feature = "nscale")]
pub use nscale::{NscalePlatformInfo, NscaleProvider};

#[cfg(feature = "runware")]
pub use runware::{RunwareModelInfo, RunwareProvider};

#[cfg(feature = "ai71")]
pub use ai71::{AI71ModelInfo, AI71Provider};

// Contingent provider re-exports (pending API access)
pub use chatlaw::ChatLawProvider;
pub use latamgpt::LatamGPTProvider;
pub use lighton::LightOnProvider;
