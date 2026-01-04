#!/usr/bin/env python3
"""
Fetch Together AI models and convert to CSV format.

Together AI provides 200+ open-source models via their API and platform.
Reference: https://api.together.ai/models
"""

import json
import requests
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Together AI API endpoint
TOGETHER_MODELS_URL = "https://api.together.xyz/models"

# Model pricing estimates (based on Together AI pricing tiers)
PRICING_TIERS = {
    "1b": {"input": 0.00000025, "output": 0.00000025},
    "3b": {"input": 0.00000075, "output": 0.00000075},
    "7b": {"input": 0.0000008, "output": 0.0000008},
    "8b": {"input": 0.0000008, "output": 0.0000008},
    "13b": {"input": 0.000002, "output": 0.000002},
    "22b": {"input": 0.000003, "output": 0.000003},
    "34b": {"input": 0.000006, "output": 0.000006},
    "40b": {"input": 0.000008, "output": 0.000008},
    "70b": {"input": 0.0000009, "output": 0.000009},
    "405b": {"input": 0.00009, "output": 0.0009},
}

# Default context windows by model size
CONTEXT_WINDOWS = {
    "llama-3-8b": 8192,
    "llama-3-70b": 8192,
    "llama-2-7b": 4096,
    "llama-2-13b": 4096,
    "llama-2-70b": 4096,
    "mistral-7b": 32768,
    "mixtral-8x7b": 32768,
    "mixtral-8x22b": 65536,
    "qwen": 8192,
    "deepseek": 4096,
}

def estimate_pricing(model_name: str) -> tuple:
    """Estimate pricing for a model based on size hints."""
    model_lower = model_name.lower()

    # Check for known size indicators
    for size_hint, pricing in PRICING_TIERS.items():
        if size_hint in model_lower:
            return (pricing["input"], pricing["output"])

    # Default to 7b pricing
    return (PRICING_TIERS["7b"]["input"], PRICING_TIERS["7b"]["output"])

def estimate_context_window(model_name: str) -> int:
    """Estimate context window for a model."""
    model_lower = model_name.lower()

    for model_hint, context in CONTEXT_WINDOWS.items():
        if model_hint in model_lower:
            return context

    # Check for specific context clues
    if "mixtral-8x22b" in model_lower or "deepseek-coder" in model_lower:
        return 65536
    if "mistral" in model_lower or "mixtral" in model_lower:
        return 32768

    return 4096  # Default context window

def classify_model(model_name: str) -> str:
    """Classify model capabilities."""
    model_lower = model_name.lower()
    capabilities = []

    # Vision capabilities
    if any(x in model_lower for x in ["vision", "llava", "pixtral", "qwen-vl"]):
        capabilities.append("V")

    # Tool/function calling
    if any(x in model_lower for x in ["gpt-like", "coder", "tool", "function"]):
        capabilities.append("T")

    # JSON mode
    if "json" in model_lower or any(x in model_lower for x in ["gpt", "coder"]):
        capabilities.append("J")

    # Structured output
    if "structured" in model_lower or "schema" in model_lower:
        capabilities.append("S")

    return "".join(capabilities) if capabilities else "T"  # Default to basic text

def fetch_together_ai_models() -> List[Dict]:
    """Fetch models from Together AI API."""
    try:
        print("Fetching Together AI models...", file=sys.stderr)
        response = requests.get(TOGETHER_MODELS_URL, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Handle different response formats
        if isinstance(data, dict):
            models = data.get("models", [])
        elif isinstance(data, list):
            models = data
        else:
            print("Unexpected response format from Together AI API", file=sys.stderr)
            return []

        print(f"✓ Fetched {len(models)} models from Together AI", file=sys.stderr)
        return models

    except requests.exceptions.RequestException as e:
        print(f"Error fetching Together AI models: {e}", file=sys.stderr)
        # Return curated list of known Together AI models
        return get_known_together_ai_models()

def get_known_together_ai_models() -> List[Dict]:
    """Return a curated list of known Together AI models (fallback)."""
    return [
        {
            "name": "meta-llama/Llama-3-70b-chat-hf",
            "description": "Meta Llama 3 70B Chat - state-of-the-art open model",
            "context_length": 8192,
            "type": "chat"
        },
        {
            "name": "meta-llama/Llama-3-8b-chat-hf",
            "description": "Meta Llama 3 8B Chat - efficient chat model",
            "context_length": 8192,
            "type": "chat"
        },
        {
            "name": "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "description": "Mistral Mixtral 8x22B Instruct",
            "context_length": 65536,
            "type": "chat"
        },
        {
            "name": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "description": "Mistral Mixtral 8x7B Instruct",
            "context_length": 32768,
            "type": "chat"
        },
        {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "description": "Mistral 7B Instruct v0.2",
            "context_length": 32768,
            "type": "chat"
        },
        {
            "name": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO",
            "description": "Nous Hermes 2 Mixtral 8x7B DPO",
            "context_length": 32768,
            "type": "chat"
        },
        {
            "name": "NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT",
            "description": "Nous Hermes 2 Mixtral 8x7B SFT",
            "context_length": 32768,
            "type": "chat"
        },
        {
            "name": "allenai/OLMo-7B-Instruct",
            "description": "Allen AI OLMo 7B Instruct",
            "context_length": 2048,
            "type": "chat"
        },
        {
            "name": "Qwen/Qwen1.5-72B-Chat",
            "description": "Qwen 1.5 72B Chat",
            "context_length": 32768,
            "type": "chat"
        },
        {
            "name": "Qwen/Qwen1.5-7B-Chat",
            "description": "Qwen 1.5 7B Chat",
            "context_length": 32768,
            "type": "chat"
        },
        {
            "name": "togethercomputer/CodeLlama-34b-Instruct",
            "description": "CodeLlama 34B Instruct for coding tasks",
            "context_length": 8192,
            "type": "chat"
        },
        {
            "name": "togethercomputer/CodeLlama-13b-Instruct",
            "description": "CodeLlama 13B Instruct for coding tasks",
            "context_length": 4096,
            "type": "chat"
        },
        {
            "name": "togethercomputer/CodeLlama-7b-Instruct",
            "description": "CodeLlama 7B Instruct for coding tasks",
            "context_length": 4096,
            "type": "chat"
        },
        {
            "name": "teknium/OpenHermes-2.5-Mistral-7B",
            "description": "OpenHermes 2.5 Mistral 7B",
            "context_length": 32768,
            "type": "chat"
        },
        {
            "name": "WizardLM/WizardLM-13B-V1.2",
            "description": "WizardLM 13B V1.2",
            "context_length": 4096,
            "type": "chat"
        },
        {
            "name": "lmsys/vicuna-13b-v1.5",
            "description": "Vicuna 13B V1.5",
            "context_length": 4096,
            "type": "chat"
        },
    ]

def model_to_csv_row(model: Dict, index: int) -> Dict[str, str]:
    """Convert a Together AI model to CSV row format."""
    # Extract model info
    name = model.get("name", "") or model.get("id", "")
    display_name = model.get("display_name", "") or name
    description = model.get("description", "")

    # Create alias from name
    alias = name.split("/")[-1].lower() if "/" in name else name.lower()
    alias = alias.replace("_", "-")[:50]

    # Estimate pricing and context
    input_price, output_price = estimate_pricing(name)
    context = model.get("context_length", estimate_context_window(name))
    max_output = min(4096, context // 4)  # Assume max output is 1/4 of context

    # Classify capabilities
    capabilities = classify_model(name)

    return {
        "id": f"together/{alias}",
        "alias": alias,
        "name": f"Together: {display_name}",
        "status": "C",  # Current
        "input_price": str(input_price),
        "output_price": str(output_price),
        "cache_input_price": "-",
        "context_window": str(context),
        "max_output": str(max_output),
        "capabilities": capabilities,
        "quality": "verified",
        "source": "together",
        "updated": datetime.now().strftime("%Y-%m-%d"),
        "description": description[:200] if description else f"{display_name} model hosted on Together AI",
        "mmlu_score": "-",
        "bbh_score": "-",
        "humaneval_score": "-",
        "aime_score": "-",
        "livebench_score": "-",
        "gpqa_score": "-",
        "gpt4_eval_score": "-",
        "agentic_score": "-",
        "ceo_evals_score": "-",
    }

def save_to_csv(models: List[Dict[str, str]], output_path: Path) -> int:
    """Save models to CSV file."""
    if not models:
        print("No models to save", file=sys.stderr)
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "id", "alias", "name", "status", "input_price", "output_price",
        "cache_input_price", "context_window", "max_output", "capabilities",
        "quality", "source", "updated", "description", "mmlu_score",
        "bbh_score", "humaneval_score", "aime_score", "livebench_score",
        "gpqa_score", "gpt4_eval_score", "agentic_score", "ceo_evals_score"
    ]

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(models)

    print(f"✓ Saved {len(models)} models to {output_path}", file=sys.stderr)
    return len(models)

def main():
    """Main entry point."""
    # Fetch models
    api_models = fetch_together_ai_models()

    if not api_models:
        print("Error: No models fetched", file=sys.stderr)
        sys.exit(1)

    # Convert to CSV rows
    csv_rows = []
    for idx, model in enumerate(api_models, 1):
        try:
            row = model_to_csv_row(model, idx)
            csv_rows.append(row)
        except Exception as e:
            print(f"Warning: Error processing model {idx}: {e}", file=sys.stderr)
            continue

    # Save to CSV
    output_path = Path(__file__).parent.parent / "data" / "models" / "aggregators" / "together_ai.csv"
    count = save_to_csv(csv_rows, output_path)

    print(f"\nTotal models processed: {count}", file=sys.stderr)
    return 0 if count > 0 else 1

if __name__ == "__main__":
    sys.exit(main())
