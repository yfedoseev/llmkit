#!/usr/bin/env python3
"""
Fetch Fireworks AI models via official API.
Uses pagination to get all models from /v1/accounts/fireworks/models endpoint.
"""

import csv
import os
import sys
import requests
from datetime import datetime
from typing import List, Dict, Any, Optional

# API Configuration
FIREWORKS_API_URL = "https://api.fireworks.ai/v1/accounts/fireworks/models"
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY", "")
PAGE_SIZE = 200

def fetch_all_models() -> List[Dict[str, Any]]:
    """Fetch all models from Fireworks API with pagination."""
    if not FIREWORKS_API_KEY:
        print("Warning: FIREWORKS_API_KEY not set, using fallback list", file=sys.stderr)
        return []

    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }

    all_models = []
    page_token = None
    page_num = 1

    while True:
        url = f"{FIREWORKS_API_URL}?pageSize={PAGE_SIZE}"
        if page_token:
            url += f"&pageToken={page_token}"

        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()

            models = data.get("models", [])
            all_models.extend(models)

            total_size = data.get("totalSize", "?")
            print(f"  Page {page_num}: fetched {len(models)} models (total so far: {len(all_models)}/{total_size})")

            page_token = data.get("nextPageToken")
            if not page_token:
                break
            page_num += 1

        except requests.exceptions.RequestException as e:
            print(f"Error fetching models: {e}", file=sys.stderr)
            break

    return all_models

def map_model_kind(kind: str) -> str:
    """Map Fireworks model kind to capabilities."""
    kind_map = {
        "HF_BASE_MODEL": "T",      # Text generation
        "HF_PEFT_ADDON": "T",      # Fine-tuned text
        "FLUMINA_BASE_MODEL": "I", # Image generation (Flux, etc)
        "EMBEDDING_MODEL": "E",    # Embeddings
        "AUDIO_MODEL": "A",        # Audio
        "VIDEO_MODEL": "D",        # Video
    }
    return kind_map.get(kind, "T")

def get_capabilities(model: Dict[str, Any]) -> str:
    """Determine model capabilities from API response."""
    caps = []

    # Check for vision capability
    if model.get("supportsImageInput", False):
        caps.append("V")

    # Check for tools/function calling
    if model.get("supportsTools", False):
        caps.append("T")

    # JSON mode - assume most chat models support it
    kind = model.get("kind", "")
    if kind in ["HF_BASE_MODEL", "HF_PEFT_ADDON"] and model.get("contextLength", 0) > 0:
        caps.append("J")

    # Special model types
    if "thinking" in model.get("name", "").lower() or "qwq" in model.get("name", "").lower():
        caps.append("K")  # Thinking/reasoning

    # Image generation models
    if kind == "FLUMINA_BASE_MODEL":
        caps = ["I"]

    # Embedding models
    if kind == "EMBEDDING_MODEL":
        caps = ["E"]

    # Audio models
    if "whisper" in model.get("name", "").lower():
        caps = ["A"]

    return "".join(caps) if caps else "-"

def get_model_pricing(model: Dict[str, Any]) -> tuple:
    """Get pricing for model. Returns (input_price, output_price) per 1M tokens."""
    name = model.get("name", "").lower()
    context = model.get("contextLength", 0)
    kind = model.get("kind", "")

    # Image models - per image pricing
    if kind == "FLUMINA_BASE_MODEL":
        if "flux" in name:
            if "pro" in name or "max" in name:
                return 0.05, "-"
            return 0.025, "-"
        return 0.02, "-"

    # Embedding models
    if kind == "EMBEDDING_MODEL":
        return 0.008, "-"

    # Text models - pricing based on size (estimated from model names)
    if "405b" in name:
        return 3.0, 3.0
    elif "671b" in name or "deepseek-v3" in name:
        return 0.9, 0.9
    elif "235b" in name or "480b" in name:
        return 0.9, 0.9
    elif "70b" in name or "72b" in name:
        return 0.9, 0.9
    elif "32b" in name or "30b" in name:
        return 0.9, 0.9
    elif "22b" in name or "8x22b" in name:
        return 1.2, 1.2
    elif "13b" in name:
        return 0.2, 0.2
    elif "8b" in name or "9b" in name or "7b" in name:
        return 0.2, 0.2
    elif "3b" in name or "4b" in name:
        return 0.1, 0.1
    elif "scout" in name:
        return 0.15, 0.6
    elif "maverick" in name:
        return 0.22, 0.88
    elif "r1" in name:
        return 3.0, 8.0
    elif "deepseek" in name:
        return 0.9, 0.9
    elif "kimi" in name:
        return 0.35, 1.4
    elif "minimax" in name:
        return 0.2, 0.2
    elif "glm" in name:
        return 0.35, 0.35
    elif "gpt-oss" in name:
        return 0.5, 0.5

    # Default pricing
    return 0.2, 0.2

def create_alias(model: Dict[str, Any]) -> str:
    """Create a short alias for the model."""
    name = model.get("name", "")
    display = model.get("displayName", name)

    # Extract model ID from full name
    model_id = name.split("/")[-1] if "/" in name else name

    # Clean up common patterns
    alias = model_id.lower()
    alias = alias.replace("accounts/fireworks/models/", "")
    alias = alias.replace("-instruct", "")
    alias = alias.replace("-basic", "")
    alias = alias.replace("-chat", "")

    # Shorten common prefixes
    alias = alias.replace("llama-v3p", "llama-")
    alias = alias.replace("qwen2p5", "qwen-2.5")
    alias = alias.replace("qwen3", "qwen-3")

    return alias[:50]  # Limit length

def process_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process API models into CSV format."""
    rows = []
    seen_ids = set()

    for model in models:
        name = model.get("name", "")
        if not name or name in seen_ids:
            continue
        seen_ids.add(name)

        # Skip non-ready models
        state = model.get("state", "")
        if state not in ["READY", ""]:
            continue

        model_id = name.split("/")[-1] if "/" in name else name
        display_name = model.get("displayName", model_id)
        context_length = model.get("contextLength", 0) or 4096
        kind = model.get("kind", "")

        input_price, output_price = get_model_pricing(model)
        capabilities = get_capabilities(model)

        # Determine max output
        max_output = min(context_length, 16384)
        if kind == "FLUMINA_BASE_MODEL":
            max_output = 0  # Image models
        elif kind == "EMBEDDING_MODEL":
            max_output = model.get("contextLength", 768)

        row = {
            'id': f"fireworks/{model_id}",
            'alias': create_alias(model),
            'name': display_name[:60],
            'status': 'C',
            'input_price': f"{input_price:.6f}" if isinstance(input_price, (int, float)) else input_price,
            'output_price': f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price,
            'cache_input_price': '-',
            'context_window': context_length,
            'max_output': max_output,
            'capabilities': capabilities,
            'quality': 'api',
            'source': 'fireworks',
            'updated': datetime.now().strftime('%Y-%m-%d'),
            'description': (model.get("description", "") or display_name)[:100],
            'mmlu_score': '-',
            'humaneval_score': '-',
            'math_score': '-'
        }
        rows.append(row)

    return rows

# Fallback models when API is unavailable
FALLBACK_MODELS = [
    {"id": "llama4-scout-instruct-basic", "alias": "llama-4-scout", "name": "Llama 4 Scout", "context_window": 131072, "input_price": 0.15, "output_price": 0.60, "capabilities": "VTJS"},
    {"id": "llama4-maverick-instruct-basic", "alias": "llama-4-maverick", "name": "Llama 4 Maverick", "context_window": 131072, "input_price": 0.22, "output_price": 0.88, "capabilities": "VTJS"},
    {"id": "llama-v3p3-70b-instruct", "alias": "llama-3.3-70b", "name": "Llama 3.3 70B", "context_window": 131072, "input_price": 0.90, "output_price": 0.90, "capabilities": "TJS"},
    {"id": "deepseek-r1-0528", "alias": "deepseek-r1", "name": "DeepSeek R1", "context_window": 163840, "input_price": 3.0, "output_price": 8.0, "capabilities": "TJK"},
    {"id": "deepseek-v3-0324", "alias": "deepseek-v3", "name": "DeepSeek V3", "context_window": 163840, "input_price": 0.9, "output_price": 0.9, "capabilities": "TJS"},
    {"id": "qwen3-235b-a22b", "alias": "qwen-3-235b", "name": "Qwen 3 235B", "context_window": 131072, "input_price": 0.9, "output_price": 0.9, "capabilities": "TJS"},
    {"id": "mixtral-8x22b-instruct", "alias": "mixtral-8x22b", "name": "Mixtral 8x22B", "context_window": 65536, "input_price": 1.2, "output_price": 1.2, "capabilities": "TJS"},
    {"id": "flux-1-dev-fp8", "alias": "flux-1-dev", "name": "FLUX.1 Dev", "context_window": 0, "input_price": 0.025, "output_price": "-", "capabilities": "I"},
    {"id": "flux-1-schnell-fp8", "alias": "flux-1-schnell", "name": "FLUX.1 Schnell", "context_window": 0, "input_price": 0.025, "output_price": "-", "capabilities": "I"},
]

def process_fallback_models() -> List[Dict[str, Any]]:
    """Process fallback models when API is unavailable."""
    rows = []
    for model in FALLBACK_MODELS:
        output_price = model['output_price']
        row = {
            'id': f"fireworks/{model['id']}",
            'alias': model['alias'],
            'name': model['name'],
            'status': 'C',
            'input_price': f"{model['input_price']:.6f}",
            'output_price': f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price,
            'cache_input_price': '-',
            'context_window': model['context_window'],
            'max_output': min(model['context_window'], 16384) if model['context_window'] > 0 else 0,
            'capabilities': model['capabilities'],
            'quality': 'official',
            'source': 'fireworks',
            'updated': datetime.now().strftime('%Y-%m-%d'),
            'description': model['name'],
            'mmlu_score': '-',
            'humaneval_score': '-',
            'math_score': '-'
        }
        rows.append(row)
    return rows

def save_csv(rows: List[Dict[str, Any]], output_file: str) -> None:
    """Save models to CSV file."""
    if not rows:
        print("No models to save", file=sys.stderr)
        return

    fieldnames = [
        'id', 'alias', 'name', 'status', 'input_price', 'output_price',
        'cache_input_price', 'context_window', 'max_output', 'capabilities',
        'quality', 'source', 'updated', 'description',
        'mmlu_score', 'humaneval_score', 'math_score'
    ]

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} models to {output_file}")

def main():
    print("Fetching Fireworks AI models via API...")

    # Try API first
    api_models = fetch_all_models()

    if api_models:
        rows = process_models(api_models)
        print(f"\nProcessed {len(rows)} models from API")
    else:
        print("Using fallback model list...")
        rows = process_fallback_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/inference/fireworks.csv'
    save_csv(rows, output_file)

    # Print summary by type
    kinds = {}
    for row in rows:
        caps = row['capabilities']
        if 'I' in caps:
            kind = 'image'
        elif 'E' in caps:
            kind = 'embedding'
        elif 'A' in caps:
            kind = 'audio'
        elif 'D' in caps:
            kind = 'video'
        else:
            kind = 'chat'
        kinds[kind] = kinds.get(kind, 0) + 1

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")
    for kind, count in sorted(kinds.items()):
        print(f"  {kind}: {count}")

if __name__ == '__main__':
    main()
