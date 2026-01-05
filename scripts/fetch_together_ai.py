#!/usr/bin/env python3
"""
Fetch Together AI models via their official API.

Together AI provides 200+ open-source models via their API.
API Reference: https://docs.together.ai/reference/models-1
"""

import json
import requests
import csv
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Together AI API endpoint (v1)
TOGETHER_API_URL = "https://api.together.xyz/v1/models"

# Get API key from environment
TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY", "")


def fetch_together_models() -> List[Dict]:
    """Fetch models from Together AI API."""
    if not TOGETHER_API_KEY:
        print("Warning: TOGETHER_API_KEY not set, using fallback list", file=sys.stderr)
        return get_fallback_models()

    try:
        print("Fetching Together AI models via API...", file=sys.stderr)
        headers = {
            "Authorization": f"Bearer {TOGETHER_API_KEY}",
            "Content-Type": "application/json"
        }
        response = requests.get(TOGETHER_API_URL, headers=headers, timeout=60)
        response.raise_for_status()

        models = response.json()
        if isinstance(models, dict):
            models = models.get("data", models.get("models", []))

        print(f"✓ Fetched {len(models)} models from Together AI API", file=sys.stderr)
        return models

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Together AI models: {e}", file=sys.stderr)
        return get_fallback_models()


def get_model_type(model: Dict) -> str:
    """Determine model type/category."""
    model_type = model.get("type", "").lower()
    model_id = model.get("id", "").lower()

    if model_type == "embedding" or "embed" in model_id:
        return "embed"
    elif model_type == "image" or any(x in model_id for x in ["flux", "sdxl", "stable-diffusion", "imagen"]):
        return "image"
    elif model_type == "audio" or any(x in model_id for x in ["whisper", "sonic", "tts"]):
        return "audio"
    elif model_type == "video" or "kling" in model_id:
        return "video"
    elif model_type == "rerank" or "rerank" in model_id:
        return "rerank"
    elif model_type == "moderation" or "guard" in model_id:
        return "moderation"
    else:
        return "chat"


def get_capabilities(model: Dict) -> str:
    """Determine model capabilities."""
    model_id = model.get("id", "").lower()
    model_type = get_model_type(model)
    caps = []

    if model_type == "embed":
        return "E"
    elif model_type == "image":
        return "I"
    elif model_type == "audio":
        return "A"
    elif model_type == "video":
        return "D"
    elif model_type == "rerank":
        return "R"
    elif model_type == "moderation":
        return "M"

    # Chat models
    # Vision
    if any(x in model_id for x in ["vision", "vl", "llava", "pixtral", "qwen2-vl", "llama-3.2-11b", "llama-3.2-90b"]):
        caps.append("V")

    # Tools/function calling
    if any(x in model_id for x in ["instruct", "chat", "turbo", "deepseek", "qwen", "llama-3", "mistral"]):
        caps.append("T")

    # JSON mode
    if any(x in model_id for x in ["instruct", "chat", "coder", "deepseek", "qwen"]):
        caps.append("J")

    # Structured output
    if any(x in model_id for x in ["llama-3.1", "llama-3.2", "llama-3.3", "qwen2", "deepseek-v3"]):
        caps.append("S")

    # Thinking/reasoning
    if any(x in model_id for x in ["deepseek-r1", "qwq", "reasoning"]):
        caps.append("K")

    return "".join(caps) if caps else "T"


def format_price(pricing: Dict, key: str) -> str:
    """Extract and format price from pricing dict."""
    if not pricing:
        return "-"

    # Together AI uses different pricing structures
    price = pricing.get(key, pricing.get(f"{key}_price", 0))
    if price is None or price == 0:
        return "-"

    # Convert to per-million-token pricing if needed
    if isinstance(price, (int, float)):
        # Together AI prices are often per token, convert to per million
        if price < 0.01:
            price = price * 1000000
        return f"{price:.6f}".rstrip('0').rstrip('.')

    return str(price)


def model_to_csv_row(model: Dict) -> Optional[Dict[str, str]]:
    """Convert a Together AI model to CSV row format."""
    model_id = model.get("id", "")
    if not model_id:
        return None

    # Create standardized ID
    if not model_id.startswith("together/"):
        std_id = f"together/{model_id}"
    else:
        std_id = model_id

    # Create alias from model ID
    alias = model_id.split("/")[-1].lower().replace("_", "-")[:60]

    # Get display name
    display_name = model.get("display_name", "") or model.get("name", "") or model_id.split("/")[-1]

    # Get pricing
    pricing = model.get("pricing", {})
    input_price = format_price(pricing, "input")
    output_price = format_price(pricing, "output")

    # Get context length
    context = model.get("context_length", 4096)
    if context is None:
        context = 4096

    # Get max output (if available, otherwise estimate)
    max_output = model.get("max_output", model.get("max_tokens", min(4096, context // 4)))

    # Get capabilities
    capabilities = get_capabilities(model)

    # Get model type for classification
    model_type = get_model_type(model)

    # Get description
    description = model.get("description", "") or f"{display_name} on Together AI"
    description = description[:100].replace("|", "-")

    return {
        "id": std_id,
        "alias": alias,
        "name": display_name,
        "status": "C",
        "input_price": input_price,
        "output_price": output_price,
        "cache_input_price": "-",
        "context_window": str(context),
        "max_output": str(max_output),
        "capabilities": capabilities,
        "quality": "official",
        "source": "together",
        "updated": datetime.now().strftime("%Y-%m-%d"),
        "description": description,
        "mmlu_score": "-",
        "humaneval_score": "-",
        "math_score": "-"
    }


def get_fallback_models() -> List[Dict]:
    """Fallback list of known Together AI models."""
    return [
        # DeepSeek
        {"id": "deepseek-ai/DeepSeek-R1", "display_name": "DeepSeek R1", "context_length": 128000, "pricing": {"input": 0.55, "output": 2.19}},
        {"id": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B", "display_name": "DeepSeek R1 Distill 70B", "context_length": 128000, "pricing": {"input": 0.40, "output": 0.40}},
        {"id": "deepseek-ai/DeepSeek-V3", "display_name": "DeepSeek V3", "context_length": 128000, "pricing": {"input": 0.50, "output": 0.50}},

        # Llama 3.3
        {"id": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "display_name": "Llama 3.3 70B Turbo", "context_length": 128000, "pricing": {"input": 0.88, "output": 0.88}},

        # Llama 3.1
        {"id": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo", "display_name": "Llama 3.1 405B Turbo", "context_length": 130000, "pricing": {"input": 3.50, "output": 3.50}},
        {"id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "display_name": "Llama 3.1 70B Turbo", "context_length": 131072, "pricing": {"input": 0.88, "output": 0.88}},
        {"id": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", "display_name": "Llama 3.1 8B Turbo", "context_length": 131072, "pricing": {"input": 0.18, "output": 0.18}},

        # Llama 3.2 Vision
        {"id": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo", "display_name": "Llama 3.2 90B Vision", "context_length": 128000, "pricing": {"input": 1.20, "output": 1.20}},
        {"id": "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo", "display_name": "Llama 3.2 11B Vision", "context_length": 128000, "pricing": {"input": 0.18, "output": 0.18}},

        # Qwen
        {"id": "Qwen/Qwen2.5-72B-Instruct-Turbo", "display_name": "Qwen 2.5 72B Turbo", "context_length": 131072, "pricing": {"input": 1.20, "output": 1.20}},
        {"id": "Qwen/Qwen2.5-Coder-32B-Instruct", "display_name": "Qwen 2.5 Coder 32B", "context_length": 131072, "pricing": {"input": 0.80, "output": 0.80}},
        {"id": "Qwen/QwQ-32B-Preview", "display_name": "QwQ 32B Preview", "context_length": 32768, "pricing": {"input": 1.20, "output": 1.20}},

        # Mistral
        {"id": "mistralai/Mixtral-8x22B-Instruct-v0.1", "display_name": "Mixtral 8x22B", "context_length": 65536, "pricing": {"input": 1.20, "output": 1.20}},
        {"id": "mistralai/Mixtral-8x7B-Instruct-v0.1", "display_name": "Mixtral 8x7B", "context_length": 32768, "pricing": {"input": 0.60, "output": 0.60}},
        {"id": "mistralai/Mistral-7B-Instruct-v0.3", "display_name": "Mistral 7B v0.3", "context_length": 32768, "pricing": {"input": 0.20, "output": 0.20}},

        # FLUX Image
        {"id": "black-forest-labs/FLUX.1-dev", "display_name": "FLUX.1 Dev", "type": "image", "context_length": 0, "pricing": {"input": 0.025}},
        {"id": "black-forest-labs/FLUX.1-schnell", "display_name": "FLUX.1 Schnell", "type": "image", "context_length": 0, "pricing": {"input": 0.003}},
        {"id": "black-forest-labs/FLUX.1.1-pro", "display_name": "FLUX 1.1 Pro", "type": "image", "context_length": 0, "pricing": {"input": 0.040}},

        # Embeddings
        {"id": "togethercomputer/m2-bert-80M-8k-retrieval", "display_name": "M2 BERT 80M", "type": "embedding", "context_length": 8192, "pricing": {"input": 0.008}},
        {"id": "WhereIsAI/UAE-Large-V1", "display_name": "UAE Large V1", "type": "embedding", "context_length": 512, "pricing": {"input": 0.016}},

        # Code models
        {"id": "Phind/Phind-CodeLlama-34B-v2", "display_name": "Phind CodeLlama 34B", "context_length": 16384, "pricing": {"input": 0.80, "output": 0.80}},
        {"id": "codellama/CodeLlama-70b-Instruct-hf", "display_name": "CodeLlama 70B", "context_length": 4096, "pricing": {"input": 0.90, "output": 0.90}},

        # Other chat models
        {"id": "google/gemma-2-27b-it", "display_name": "Gemma 2 27B", "context_length": 8192, "pricing": {"input": 0.80, "output": 0.80}},
        {"id": "google/gemma-2-9b-it", "display_name": "Gemma 2 9B", "context_length": 8192, "pricing": {"input": 0.30, "output": 0.30}},
        {"id": "databricks/dbrx-instruct", "display_name": "DBRX Instruct", "context_length": 32768, "pricing": {"input": 1.20, "output": 1.20}},
        {"id": "NousResearch/Hermes-3-Llama-3.1-405B-Turbo", "display_name": "Hermes 3 405B", "context_length": 130000, "pricing": {"input": 3.50, "output": 3.50}},

        # Audio
        {"id": "cartesia/sonic", "display_name": "Cartesia Sonic TTS", "type": "audio", "context_length": 0, "pricing": {"input": 0.006}},

        # Video
        {"id": "kwaivgI/kling-1.6-standard", "display_name": "Kling 1.6 Standard", "type": "video", "context_length": 0, "pricing": {"input": 0.10}},
    ]


def save_to_csv(models: List[Dict[str, str]], output_path: Path) -> int:
    """Save models to CSV file."""
    if not models:
        print("No models to save", file=sys.stderr)
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id", "alias", "name", "status", "input_price", "output_price",
        "cache_input_price", "context_window", "max_output", "capabilities",
        "quality", "source", "updated", "description",
        "mmlu_score", "humaneval_score", "math_score"
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(models)

    print(f"✓ Saved {len(models)} models to {output_path}", file=sys.stderr)
    return len(models)


def main():
    """Main entry point."""
    # Fetch models from API
    api_models = fetch_together_models()

    if not api_models:
        print("Error: No models fetched", file=sys.stderr)
        sys.exit(1)

    # Convert to CSV rows
    csv_rows = []
    model_types = {}

    for model in api_models:
        try:
            row = model_to_csv_row(model)
            if row:
                csv_rows.append(row)
                model_type = get_model_type(model)
                model_types[model_type] = model_types.get(model_type, 0) + 1
        except Exception as e:
            print(f"Warning: Error processing model {model.get('id', 'unknown')}: {e}", file=sys.stderr)
            continue

    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "models" / "aggregators" / "together_ai.csv"
    count = save_to_csv(csv_rows, output_path)

    # Print summary
    print(f"\n=== Together AI Summary ===", file=sys.stderr)
    print(f"Total models: {count}", file=sys.stderr)
    for mtype, mcount in sorted(model_types.items()):
        print(f"  {mtype}: {mcount}", file=sys.stderr)

    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
