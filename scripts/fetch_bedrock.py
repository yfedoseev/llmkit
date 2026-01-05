#!/usr/bin/env python3
"""
Create Bedrock models CSV from AWS documentation.
Comprehensive list of all models available on Amazon Bedrock.
Source: https://docs.aws.amazon.com/bedrock/latest/userguide/models-supported.html
"""

import csv
import sys
from datetime import datetime

# Bedrock models compiled from AWS documentation (January 2026)
# Format: (model_id, alias, name, status, input_price, output_price, context_window, max_output, capabilities, quality, description)
BEDROCK_MODELS = [
    # ============================================================
    # AI21 LABS MODELS
    # ============================================================
    ("bedrock/ai21.jamba-1-5-large-v1:0", "jamba-1-5-large", "AI21 Jamba 1.5 Large", "C", "2.00", "8.00", 256000, 4096, "TJ", "verified", "AI21 large hybrid SSM-Transformer model"),
    ("bedrock/ai21.jamba-1-5-mini-v1:0", "jamba-1-5-mini", "AI21 Jamba 1.5 Mini", "C", "0.20", "0.40", 256000, 4096, "TJ", "verified", "AI21 efficient hybrid model"),
    ("bedrock/ai21.jamba-instruct-v1:0", "jamba-instruct", "AI21 Jamba Instruct", "C", "0.50", "0.70", 256000, 4096, "TJ", "verified", "AI21 instruction-tuned Jamba"),
    ("bedrock/ai21.j2-ultra-v1", "jurassic-2-ultra", "AI21 Jurassic-2 Ultra", "L", "18.80", "18.80", 8191, 8191, "-", "verified", "Legacy AI21 flagship model"),
    ("bedrock/ai21.j2-mid-v1", "jurassic-2-mid", "AI21 Jurassic-2 Mid", "L", "12.50", "12.50", 8191, 8191, "-", "verified", "Legacy AI21 mid-tier model"),

    # ============================================================
    # AMAZON MODELS
    # ============================================================
    # Amazon Nova Foundation Models
    ("bedrock/amazon.nova-pro-v1:0", "nova-pro", "Amazon Nova Pro", "C", "0.80", "3.20", 300000, 5000, "SVTJ", "verified", "Advanced multimodal understanding and generation"),
    ("bedrock/amazon.nova-lite-v1:0", "nova-lite", "Amazon Nova Lite", "C", "0.06", "0.24", 300000, 5000, "SVTJ", "verified", "Fast and cost-effective multimodal"),
    ("bedrock/amazon.nova-micro-v1:0", "nova-micro", "Amazon Nova Micro", "C", "0.035", "0.14", 128000, 5000, "TJ", "verified", "Text-only fastest and lowest cost"),
    ("bedrock/amazon.nova-premier-v1:0", "nova-premier", "Amazon Nova Premier", "C", "2.50", "10.00", 1000000, 5000, "SVTJK", "verified", "Most capable Nova for complex tasks"),

    # Amazon Nova Canvas (Image Generation)
    ("bedrock/amazon.nova-canvas-v1:0", "nova-canvas", "Amazon Nova Canvas", "C", "0.04", "-", 0, 0, "I", "verified", "State-of-art image generation"),

    # Amazon Nova Reel (Video Generation)
    ("bedrock/amazon.nova-reel-v1:0", "nova-reel", "Amazon Nova Reel", "C", "0.80", "-", 0, 0, "D", "verified", "Studio-quality video generation"),

    # Amazon Nova Sonic (Speech)
    ("bedrock/amazon.nova-sonic-v1:0", "nova-sonic", "Amazon Nova Sonic", "C", "4.88", "7.50", 0, 0, "A", "verified", "Streaming speech-to-speech"),

    # Amazon Nova 2 Series (2025+)
    ("bedrock/amazon.nova-2-lite-v1:0", "nova-2-lite", "Amazon Nova 2 Lite", "C", "0.08", "0.32", 300000, 5000, "SVTJ", "verified", "Next-gen efficient multimodal"),
    ("bedrock/amazon.nova-2-pro-v1:0", "nova-2-pro", "Amazon Nova 2 Pro", "C", "1.00", "4.00", 300000, 5000, "SVTJK", "verified", "Next-gen advanced multimodal"),

    # Amazon Titan Text Models
    ("bedrock/amazon.titan-text-premier-v1:0", "titan-text-premier", "Amazon Titan Text Premier", "C", "0.50", "1.50", 32000, 3072, "TJ", "verified", "Titan flagship text model"),
    ("bedrock/amazon.titan-text-express-v1", "titan-text-express", "Amazon Titan Text Express", "C", "0.20", "0.60", 8000, 4096, "-", "verified", "Balanced Titan text model"),
    ("bedrock/amazon.titan-text-lite-v1", "titan-text-lite", "Amazon Titan Text Lite", "C", "0.15", "0.20", 4000, 4000, "-", "verified", "Lightweight Titan text"),

    # Amazon Titan Embeddings
    ("bedrock/amazon.titan-embed-text-v2:0", "titan-embed-v2", "Amazon Titan Text Embeddings V2", "C", "0.02", "-", 8192, 1024, "E", "verified", "Latest Titan embeddings"),
    ("bedrock/amazon.titan-embed-text-v1", "titan-embed-v1", "Amazon Titan Text Embeddings V1", "C", "0.10", "-", 8192, 1536, "E", "verified", "Titan embeddings v1"),
    ("bedrock/amazon.titan-embed-image-v1", "titan-embed-image", "Amazon Titan Multimodal Embeddings", "C", "0.80", "-", 128, 1024, "VE", "verified", "Image+text embeddings"),

    # Amazon Titan Image Generator
    ("bedrock/amazon.titan-image-generator-v2:0", "titan-image-v2", "Amazon Titan Image Generator V2", "C", "0.008", "-", 0, 0, "I", "verified", "Advanced image generation"),
    ("bedrock/amazon.titan-image-generator-v1", "titan-image-v1", "Amazon Titan Image Generator V1", "C", "0.01", "-", 0, 0, "I", "verified", "Image generation"),

    # Amazon Rerank
    ("bedrock/amazon.rerank-v1:0", "amazon-rerank", "Amazon Rerank", "C", "1.00", "-", 32000, 0, "R", "verified", "Semantic reranking model"),

    # ============================================================
    # ANTHROPIC MODELS
    # ============================================================
    # Claude 4.5 Series
    ("bedrock/anthropic.claude-sonnet-4-5-v1:0", "claude-4-5-sonnet", "Claude 4.5 Sonnet", "C", "3.00", "15.00", 200000, 8192, "SVTJKC", "verified", "Best for complex coding and analysis"),

    # Claude 4 Series
    ("bedrock/anthropic.claude-opus-4-1-v1:0", "claude-4-1-opus", "Claude 4.1 Opus", "C", "15.00", "75.00", 200000, 32000, "SVTJKC", "verified", "Most powerful Claude model"),
    ("bedrock/anthropic.claude-opus-4-v1:0", "claude-4-opus", "Claude 4 Opus", "C", "15.00", "75.00", 200000, 32000, "SVTJKC", "verified", "Claude 4 flagship"),
    ("bedrock/anthropic.claude-sonnet-4-v1:0", "claude-4-sonnet", "Claude 4 Sonnet", "C", "3.00", "15.00", 200000, 8192, "SVTJKC", "verified", "Claude 4 balanced model"),

    # Claude 3.7 Series
    ("bedrock/anthropic.claude-3-7-sonnet-v1:0", "claude-3-7-sonnet", "Claude 3.7 Sonnet", "C", "3.00", "15.00", 200000, 8192, "SVTJKC", "verified", "Enhanced Claude 3.5 successor"),

    # Claude 3.5 Series
    ("bedrock/anthropic.claude-3-5-sonnet-v2:0", "claude-3-5-sonnet-v2", "Claude 3.5 Sonnet V2", "C", "3.00", "15.00", 200000, 8192, "SVTJKC", "verified", "Latest 3.5 Sonnet with computer use"),
    ("bedrock/anthropic.claude-3-5-sonnet-v1:0", "claude-3-5-sonnet-v1", "Claude 3.5 Sonnet V1", "C", "3.00", "15.00", 200000, 8192, "SVTJC", "verified", "Original Claude 3.5 Sonnet"),
    ("bedrock/anthropic.claude-3-5-haiku-v1:0", "claude-3-5-haiku", "Claude 3.5 Haiku", "C", "0.80", "4.00", 200000, 8192, "SVTJC", "verified", "Fast Claude 3.5 model"),

    # Claude 3 Series
    ("bedrock/anthropic.claude-3-opus-v1:0", "claude-3-opus", "Claude 3 Opus", "C", "15.00", "75.00", 200000, 4096, "SVTJC", "verified", "Claude 3 most capable"),
    ("bedrock/anthropic.claude-3-sonnet-v1:0", "claude-3-sonnet", "Claude 3 Sonnet", "L", "3.00", "15.00", 200000, 4096, "SVTJC", "verified", "Claude 3 balanced"),
    ("bedrock/anthropic.claude-3-haiku-v1:0", "claude-3-haiku", "Claude 3 Haiku", "C", "0.25", "1.25", 200000, 4096, "SVTJC", "verified", "Claude 3 fastest"),

    # Claude 2 Legacy
    ("bedrock/anthropic.claude-v2:1", "claude-2-1", "Claude 2.1", "L", "8.00", "24.00", 200000, 4096, "T", "verified", "Legacy Claude 2.1"),
    ("bedrock/anthropic.claude-v2", "claude-2", "Claude 2", "L", "8.00", "24.00", 100000, 4096, "-", "verified", "Legacy Claude 2"),
    ("bedrock/anthropic.claude-instant-v1", "claude-instant", "Claude Instant", "L", "0.80", "2.40", 100000, 4096, "-", "verified", "Legacy fast Claude"),

    # ============================================================
    # COHERE MODELS
    # ============================================================
    ("bedrock/cohere.command-r-plus-v1:0", "command-r-plus", "Cohere Command R+", "C", "2.50", "10.00", 128000, 4000, "TJS", "verified", "Cohere flagship RAG model"),
    ("bedrock/cohere.command-r-v1:0", "command-r", "Cohere Command R", "C", "0.50", "1.50", 128000, 4000, "TJS", "verified", "Cohere efficient RAG model"),
    ("bedrock/cohere.command-light-text-v14", "command-light", "Cohere Command Light", "L", "0.30", "0.60", 4096, 4096, "-", "verified", "Legacy Cohere light model"),
    ("bedrock/cohere.command-text-v14", "command-text", "Cohere Command", "L", "1.50", "2.00", 4096, 4096, "-", "verified", "Legacy Cohere command"),

    # Cohere Embeddings
    ("bedrock/cohere.embed-english-v3", "cohere-embed-en-v3", "Cohere Embed English V3", "C", "0.10", "-", 512, 1024, "E", "verified", "English embeddings"),
    ("bedrock/cohere.embed-multilingual-v3", "cohere-embed-multi-v3", "Cohere Embed Multilingual V3", "C", "0.10", "-", 512, 1024, "E", "verified", "Multilingual embeddings"),

    # Cohere Rerank
    ("bedrock/cohere.rerank-v3-5:0", "cohere-rerank-v3-5", "Cohere Rerank 3.5", "C", "2.00", "-", 4096, 0, "R", "verified", "Latest Cohere reranker"),
    ("bedrock/cohere.rerank-multilingual-v3:0", "cohere-rerank-multi", "Cohere Rerank Multilingual", "C", "2.00", "-", 4096, 0, "R", "verified", "Multilingual reranking"),

    # ============================================================
    # DEEPSEEK MODELS
    # ============================================================
    ("bedrock/deepseek.deepseek-r1-v1:0", "deepseek-r1", "DeepSeek R1", "C", "1.40", "5.60", 128000, 16384, "TJK", "verified", "DeepSeek reasoning model"),
    ("bedrock/deepseek.deepseek-v3-1-v1:0", "deepseek-v3-1", "DeepSeek V3.1", "C", "0.27", "1.10", 128000, 16384, "TJS", "verified", "DeepSeek efficient MoE"),

    # ============================================================
    # GOOGLE MODELS (via Bedrock)
    # ============================================================
    ("bedrock/google.gemma-3-27b-it-v1:0", "gemma-3-27b", "Google Gemma 3 27B IT", "C", "0.30", "0.35", 128000, 8192, "VT", "verified", "Gemma 3 large model"),
    ("bedrock/google.gemma-3-12b-it-v1:0", "gemma-3-12b", "Google Gemma 3 12B IT", "C", "0.10", "0.15", 128000, 8192, "VT", "verified", "Gemma 3 medium model"),
    ("bedrock/google.gemma-3-4b-it-v1:0", "gemma-3-4b", "Google Gemma 3 4B IT", "C", "0.06", "0.08", 128000, 8192, "-", "verified", "Gemma 3 small model"),
    ("bedrock/google.gemma-2-27b-it-v1:0", "gemma-2-27b", "Google Gemma 2 27B IT", "C", "0.30", "0.35", 8192, 8192, "-", "verified", "Gemma 2 large model"),
    ("bedrock/google.gemma-7b-it-v1:0", "gemma-7b", "Google Gemma 7B IT", "L", "0.07", "0.14", 8192, 8192, "-", "verified", "Legacy Gemma instruction"),

    # ============================================================
    # LUMA AI MODELS
    # ============================================================
    ("bedrock/luma.ray-v2:0", "luma-ray-v2", "Luma Ray V2", "C", "0.65", "-", 0, 0, "D", "verified", "Fast video generation"),

    # ============================================================
    # META LLAMA MODELS
    # ============================================================
    # Llama 4 Series
    ("bedrock/meta.llama4-maverick-17b-instruct-v1:0", "llama-4-maverick", "Llama 4 Maverick 17B", "C", "0.22", "0.88", 128000, 8192, "SVTJ", "verified", "Llama 4 multimodal specialist"),
    ("bedrock/meta.llama4-scout-17b-instruct-v1:0", "llama-4-scout", "Llama 4 Scout 17B", "C", "0.17", "0.68", 1000000, 8192, "SVTJ", "verified", "Llama 4 long context"),

    # Llama 3.3 Series
    ("bedrock/meta.llama3-3-70b-instruct-v1:0", "llama-3-3-70b", "Llama 3.3 70B Instruct", "C", "0.72", "0.72", 128000, 8192, "TJS", "verified", "Llama 3.3 flagship"),

    # Llama 3.2 Series
    ("bedrock/meta.llama3-2-90b-instruct-v1:0", "llama-3-2-90b-vision", "Llama 3.2 90B Vision", "C", "2.00", "2.00", 128000, 4096, "SVTJ", "verified", "Llama 3.2 large vision"),
    ("bedrock/meta.llama3-2-11b-instruct-v1:0", "llama-3-2-11b-vision", "Llama 3.2 11B Vision", "C", "0.16", "0.16", 128000, 4096, "SVTJ", "verified", "Llama 3.2 small vision"),
    ("bedrock/meta.llama3-2-3b-instruct-v1:0", "llama-3-2-3b", "Llama 3.2 3B Instruct", "C", "0.15", "0.15", 128000, 4096, "TJ", "verified", "Llama 3.2 tiny"),
    ("bedrock/meta.llama3-2-1b-instruct-v1:0", "llama-3-2-1b", "Llama 3.2 1B Instruct", "C", "0.10", "0.10", 128000, 4096, "-", "verified", "Llama 3.2 smallest"),

    # Llama 3.1 Series
    ("bedrock/meta.llama3-1-405b-instruct-v1:0", "llama-3-1-405b", "Llama 3.1 405B Instruct", "C", "5.32", "16.00", 128000, 8192, "TJS", "verified", "Llama 3.1 largest"),
    ("bedrock/meta.llama3-1-70b-instruct-v1:0", "llama-3-1-70b", "Llama 3.1 70B Instruct", "C", "0.72", "0.72", 128000, 8192, "TJS", "verified", "Llama 3.1 large"),
    ("bedrock/meta.llama3-1-8b-instruct-v1:0", "llama-3-1-8b", "Llama 3.1 8B Instruct", "C", "0.22", "0.22", 128000, 8192, "TJ", "verified", "Llama 3.1 small"),

    # Llama 3 Legacy
    ("bedrock/meta.llama3-70b-instruct-v1:0", "llama-3-70b", "Llama 3 70B Instruct", "L", "2.65", "3.50", 8192, 2048, "TJ", "verified", "Legacy Llama 3 large"),
    ("bedrock/meta.llama3-8b-instruct-v1:0", "llama-3-8b", "Llama 3 8B Instruct", "L", "0.30", "0.60", 8192, 2048, "-", "verified", "Legacy Llama 3 small"),

    # Llama 2 Legacy
    ("bedrock/meta.llama2-70b-chat-v1", "llama-2-70b", "Llama 2 70B Chat", "L", "1.95", "2.56", 4096, 2048, "-", "verified", "Legacy Llama 2"),
    ("bedrock/meta.llama2-13b-chat-v1", "llama-2-13b", "Llama 2 13B Chat", "L", "0.75", "1.00", 4096, 2048, "-", "verified", "Legacy Llama 2 small"),

    # ============================================================
    # MINIMAX MODELS
    # ============================================================
    ("bedrock/minimax.minimax-m2-v1:0", "minimax-m2", "MiniMax M2", "C", "0.80", "3.20", 128000, 8192, "TJ", "verified", "MiniMax flagship model"),

    # ============================================================
    # MISTRAL AI MODELS
    # ============================================================
    # Mistral Large Series
    ("bedrock/mistral.mistral-large-2411-v1:0", "mistral-large-2411", "Mistral Large 24.11", "C", "2.00", "6.00", 128000, 8192, "VTJS", "verified", "Latest Mistral Large"),
    ("bedrock/mistral.mistral-large-2407-v1:0", "mistral-large-2407", "Mistral Large 24.07", "C", "2.00", "6.00", 128000, 8192, "TJS", "verified", "Mistral Large previous"),

    # Pixtral (Vision)
    ("bedrock/mistral.pixtral-large-2502-v1:0", "pixtral-large", "Pixtral Large 25.02", "C", "2.00", "6.00", 128000, 8192, "SVTJS", "verified", "Mistral multimodal flagship"),
    ("bedrock/mistral.pixtral-12b-2409-v1:0", "pixtral-12b", "Pixtral 12B", "C", "0.15", "0.15", 128000, 8192, "SVT", "verified", "Mistral small multimodal"),

    # Ministral (Small)
    ("bedrock/mistral.ministral-8b-2410-v1:0", "ministral-8b", "Ministral 8B", "C", "0.10", "0.10", 128000, 8192, "TJ", "verified", "Mistral efficient edge model"),
    ("bedrock/mistral.ministral-3b-2410-v1:0", "ministral-3b", "Ministral 3B", "C", "0.04", "0.04", 128000, 8192, "-", "verified", "Mistral tiny edge model"),

    # Mistral Small
    ("bedrock/mistral.mistral-small-2409-v1:0", "mistral-small-2409", "Mistral Small 24.09", "C", "0.10", "0.30", 32000, 8192, "TJ", "verified", "Cost-effective Mistral"),

    # Codestral
    ("bedrock/mistral.codestral-2501-v1:0", "codestral-2501", "Codestral 25.01", "C", "0.30", "0.90", 256000, 16384, "TJ", "verified", "Code-specialized Mistral"),

    # Voxtral (Speech)
    ("bedrock/mistral.voxtral-mini-3b-v1:0", "voxtral-mini", "Voxtral Mini 3B", "C", "0.05", "0.05", 32000, 8192, "A", "verified", "Mistral speech model"),

    # Legacy Mistral
    ("bedrock/mistral.mistral-7b-instruct-v0:2", "mistral-7b", "Mistral 7B Instruct", "L", "0.15", "0.20", 32000, 8192, "-", "verified", "Legacy Mistral 7B"),
    ("bedrock/mistral.mixtral-8x7b-instruct-v0:1", "mixtral-8x7b", "Mixtral 8x7B Instruct", "L", "0.45", "0.70", 32000, 8192, "-", "verified", "Legacy Mixtral MoE"),

    # ============================================================
    # MOONSHOT AI MODELS
    # ============================================================
    ("bedrock/moonshot.kimi-k2-thinking-v1:0", "kimi-k2-thinking", "Moonshot Kimi K2 Thinking", "C", "0.60", "2.40", 131072, 16384, "TJK", "verified", "Kimi reasoning model"),

    # ============================================================
    # NVIDIA MODELS
    # ============================================================
    ("bedrock/nvidia.nemotron-nano-8b-v1:0", "nemotron-nano", "NVIDIA Nemotron Nano 8B", "C", "0.10", "0.10", 128000, 4096, "TJ", "verified", "NVIDIA efficient model"),

    # ============================================================
    # QWEN MODELS
    # ============================================================
    ("bedrock/qwen.qwen3-235b-instruct-v1:0", "qwen3-235b", "Qwen3 235B Instruct", "C", "1.50", "4.00", 131072, 8192, "TJK", "verified", "Qwen3 largest model"),
    ("bedrock/qwen.qwen3-72b-instruct-v1:0", "qwen3-72b", "Qwen3 72B Instruct", "C", "0.40", "0.80", 131072, 8192, "TJK", "verified", "Qwen3 large model"),
    ("bedrock/qwen.qwen3-32b-instruct-v1:0", "qwen3-32b", "Qwen3 32B Instruct", "C", "0.20", "0.40", 131072, 8192, "TJK", "verified", "Qwen3 medium model"),
    ("bedrock/qwen.qwen3-14b-instruct-v1:0", "qwen3-14b", "Qwen3 14B Instruct", "C", "0.10", "0.20", 131072, 8192, "TJ", "verified", "Qwen3 small model"),
    ("bedrock/qwen.qwen3-8b-instruct-v1:0", "qwen3-8b", "Qwen3 8B Instruct", "C", "0.06", "0.12", 131072, 8192, "TJ", "verified", "Qwen3 efficient model"),
    ("bedrock/qwen.qwen3-4b-instruct-v1:0", "qwen3-4b", "Qwen3 4B Instruct", "C", "0.04", "0.08", 131072, 8192, "-", "verified", "Qwen3 tiny model"),

    # ============================================================
    # STABILITY AI MODELS
    # ============================================================
    ("bedrock/stability.sd3-5-large-v1:0", "sd3-5-large", "Stable Diffusion 3.5 Large", "C", "0.065", "-", 0, 0, "I", "verified", "SD3.5 large image generation"),
    ("bedrock/stability.sd3-5-large-turbo-v1:0", "sd3-5-turbo", "Stable Diffusion 3.5 Turbo", "C", "0.04", "-", 0, 0, "I", "verified", "SD3.5 fast generation"),
    ("bedrock/stability.sd3-large-v1:0", "sd3-large", "Stable Diffusion 3 Large", "C", "0.08", "-", 0, 0, "I", "verified", "SD3 large generation"),
    ("bedrock/stability.sd3-medium-v1:0", "sd3-medium", "Stable Diffusion 3 Medium", "C", "0.035", "-", 0, 0, "I", "verified", "SD3 medium generation"),
    ("bedrock/stability.stable-image-core-v1:0", "stable-image-core", "Stable Image Core", "C", "0.04", "-", 0, 0, "I", "verified", "Fast image generation"),
    ("bedrock/stability.stable-image-ultra-v1:0", "stable-image-ultra", "Stable Image Ultra", "C", "0.14", "-", 0, 0, "I", "verified", "Highest quality images"),
    ("bedrock/stability.sd3-5-medium-v1:0", "sd3-5-medium", "Stable Diffusion 3.5 Medium", "C", "0.025", "-", 0, 0, "I", "verified", "SD3.5 medium generation"),

    # Image Editing
    ("bedrock/stability.stable-diffusion-xl-v1", "sdxl", "Stable Diffusion XL", "C", "0.04", "-", 0, 0, "I", "verified", "SDXL image generation"),
    ("bedrock/stability.stable-image-edit-v1:0", "stable-image-edit", "Stable Image Edit", "C", "0.04", "-", 0, 0, "I", "verified", "Inpaint and outpaint"),
    ("bedrock/stability.stable-image-style-v1:0", "stable-image-style", "Stable Image Style", "C", "0.04", "-", 0, 0, "I", "verified", "Style transfer"),
    ("bedrock/stability.stable-image-control-v1:0", "stable-image-control", "Stable Image Control", "C", "0.04", "-", 0, 0, "I", "verified", "Sketch and structure to image"),
    ("bedrock/stability.stable-image-background-v1:0", "stable-image-bg", "Stable Image Background", "C", "0.04", "-", 0, 0, "I", "verified", "Background removal/replace"),

    # ============================================================
    # TWELVELABS MODELS
    # ============================================================
    ("bedrock/twelvelabs.marengo-embed-v1:0", "marengo-embed", "TwelveLabs Marengo Embed", "C", "0.025", "-", 0, 1024, "VE", "verified", "Video embeddings"),
    ("bedrock/twelvelabs.pegasus-1-2-v1:0", "pegasus-1-2", "TwelveLabs Pegasus 1.2", "C", "0.50", "1.50", 0, 4096, "D", "verified", "Video understanding"),

    # ============================================================
    # WRITER MODELS
    # ============================================================
    ("bedrock/writer.palmyra-x5-v1:0", "palmyra-x5", "Writer Palmyra X5", "C", "4.00", "12.00", 128000, 8192, "TJS", "verified", "Writer latest flagship"),
    ("bedrock/writer.palmyra-x4-v1:0", "palmyra-x4", "Writer Palmyra X4", "C", "2.00", "6.00", 128000, 8192, "TJS", "verified", "Writer previous flagship"),
]


def validate_model(model: tuple) -> bool:
    """Validate a model record has valid data."""
    if len(model) != 11:
        return False

    id_val, alias, name, status, input_price, output_price, context, max_out, caps, quality, desc = model

    if not id_val or not name or status not in ['C', 'L', 'D']:
        return False

    try:
        if input_price != '-':
            inp = float(input_price)
            if inp < 0 or inp > 100:  # Price sanity check (per million tokens)
                return False
        if output_price != '-':
            out = float(output_price)
            if out < 0 or out > 200:
                return False
    except ValueError:
        return False

    return True


def save_bedrock_csv(output_file: str) -> None:
    """Save Bedrock models to CSV file."""
    fieldnames = [
        'id', 'alias', 'name', 'status', 'input_price', 'output_price',
        'cache_input_price', 'context_window', 'max_output', 'capabilities',
        'quality', 'source', 'updated', 'description',
        'mmlu_score', 'humaneval_score', 'math_score'
    ]

    rows = []
    today = datetime.now().strftime('%Y-%m-%d')

    valid_count = 0
    for model in BEDROCK_MODELS:
        if not validate_model(model):
            print(f"Skipping invalid model: {model[0] if len(model) > 0 else 'unknown'}", file=sys.stderr)
            continue

        id_val, alias, name, status, input_price, output_price, context, max_out, caps, quality, desc = model

        row = {
            'id': id_val,
            'alias': alias,
            'name': name,
            'status': status,
            'input_price': input_price,
            'output_price': output_price,
            'cache_input_price': '-',
            'context_window': context,
            'max_output': max_out,
            'capabilities': caps if caps else '-',
            'quality': quality,
            'source': 'bedrock',
            'updated': today,
            'description': desc,
            'mmlu_score': '-',
            'humaneval_score': '-',
            'math_score': '-'
        }
        rows.append(row)
        valid_count += 1

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {valid_count} Bedrock models to {output_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary by provider
    providers = {}
    for r in rows:
        provider = r['id'].split('/')[1].split('.')[0] if '.' in r['id'].split('/')[1] else r['id'].split('/')[1]
        providers[provider] = providers.get(provider, 0) + 1

    print(f"\n=== Summary by Provider ===")
    for provider, count in sorted(providers.items()):
        print(f"  {provider}: {count} models")

    print(f"\n=== Overall Summary ===")
    print(f"Total models: {len(rows)}")
    current = sum(1 for r in rows if r['status'] == 'C')
    legacy = sum(1 for r in rows if r['status'] == 'L')
    print(f"Current: {current}, Legacy: {legacy}")

    # Capability breakdown
    with_vision = sum(1 for r in rows if 'V' in r['capabilities'])
    with_tools = sum(1 for r in rows if 'T' in r['capabilities'])
    with_thinking = sum(1 for r in rows if 'K' in r['capabilities'])
    with_embed = sum(1 for r in rows if 'E' in r['capabilities'])
    with_image = sum(1 for r in rows if 'I' in r['capabilities'])
    with_audio = sum(1 for r in rows if 'A' in r['capabilities'])
    with_video = sum(1 for r in rows if 'D' in r['capabilities'])
    with_rerank = sum(1 for r in rows if 'R' in r['capabilities'])

    print(f"\n=== Capability Breakdown ===")
    print(f"Vision (V): {with_vision}")
    print(f"Tools (T): {with_tools}")
    print(f"Thinking (K): {with_thinking}")
    print(f"Embeddings (E): {with_embed}")
    print(f"Image Gen (I): {with_image}")
    print(f"Audio (A): {with_audio}")
    print(f"Video (D): {with_video}")
    print(f"Rerank (R): {with_rerank}")


def main():
    print(f"Processing {len(BEDROCK_MODELS)} Bedrock models from AWS documentation...")
    output_file = '/home/yfedoseev/projects/modelsuite/data/models/aggregators/bedrock.csv'
    save_bedrock_csv(output_file)


if __name__ == '__main__':
    main()
