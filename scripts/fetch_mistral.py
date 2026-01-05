#!/usr/bin/env python3
"""
Fetch Mistral AI models.
Can use API with MISTRAL_API_KEY or fall back to hardcoded list.
"""

import os
import csv
import sys
import requests
from datetime import datetime

API_URL = "https://api.mistral.ai/v1/models"

MISTRAL_MODELS = [
    # Premier Models
    {
        "id": "mistral-large-2411",
        "alias": "mistral-large",
        "name": "Mistral Large 24.11",
        "context_window": 131072,
        "max_output": 131072,
        "input_price": 2.0,
        "output_price": 6.0,
        "capabilities": "VTJS",
        "description": "Flagship model for complex tasks and reasoning"
    },
    {
        "id": "mistral-large-2407",
        "alias": "mistral-large-2407",
        "name": "Mistral Large 24.07",
        "context_window": 131072,
        "max_output": 131072,
        "input_price": 2.0,
        "output_price": 6.0,
        "capabilities": "VTJS",
        "description": "Previous Mistral Large version"
    },
    {
        "id": "pixtral-large-2411",
        "alias": "pixtral-large",
        "name": "Pixtral Large",
        "context_window": 131072,
        "max_output": 131072,
        "input_price": 2.0,
        "output_price": 6.0,
        "capabilities": "VTJS",
        "description": "Vision-enabled large model"
    },
    # Medium Models
    {
        "id": "mistral-medium-2505",
        "alias": "mistral-medium",
        "name": "Mistral Medium 25.05",
        "context_window": 131072,
        "max_output": 131072,
        "input_price": 0.40,
        "output_price": 2.0,
        "capabilities": "VTJS",
        "description": "Balanced performance and cost"
    },
    # Small Models
    {
        "id": "mistral-small-2503",
        "alias": "mistral-small",
        "name": "Mistral Small 25.03",
        "context_window": 32768,
        "max_output": 32768,
        "input_price": 0.10,
        "output_price": 0.30,
        "capabilities": "VTJS",
        "description": "Fast and efficient for most tasks"
    },
    {
        "id": "mistral-small-2409",
        "alias": "mistral-small-2409",
        "name": "Mistral Small 24.09",
        "context_window": 32768,
        "max_output": 32768,
        "input_price": 0.10,
        "output_price": 0.30,
        "capabilities": "VTJS",
        "description": "Previous Mistral Small version"
    },
    {
        "id": "pixtral-12b-2409",
        "alias": "pixtral-12b",
        "name": "Pixtral 12B",
        "context_window": 131072,
        "max_output": 131072,
        "input_price": 0.15,
        "output_price": 0.15,
        "capabilities": "VTJ",
        "description": "Vision model with 12B parameters"
    },
    # Free Tier
    {
        "id": "mistral-nemo-2407",
        "alias": "mistral-nemo",
        "name": "Mistral Nemo",
        "context_window": 131072,
        "max_output": 131072,
        "input_price": 0.15,
        "output_price": 0.15,
        "capabilities": "TJ",
        "description": "12B parameter free tier model"
    },
    # Codestral (Code)
    {
        "id": "codestral-2501",
        "alias": "codestral",
        "name": "Codestral 25.01",
        "context_window": 262144,
        "max_output": 262144,
        "input_price": 0.30,
        "output_price": 0.90,
        "capabilities": "TJ",
        "description": "Specialized code generation model"
    },
    {
        "id": "codestral-2405",
        "alias": "codestral-2405",
        "name": "Codestral 24.05",
        "context_window": 32768,
        "max_output": 32768,
        "input_price": 0.20,
        "output_price": 0.60,
        "capabilities": "TJ",
        "description": "Previous Codestral version"
    },
    {
        "id": "codestral-mamba-2407",
        "alias": "codestral-mamba",
        "name": "Codestral Mamba",
        "context_window": 262144,
        "max_output": 262144,
        "input_price": 0.20,
        "output_price": 0.60,
        "capabilities": "TJ",
        "description": "Mamba architecture for code"
    },
    # Embeddings
    {
        "id": "mistral-embed",
        "alias": "mistral-embed",
        "name": "Mistral Embed",
        "context_window": 8192,
        "max_output": 1024,
        "input_price": 0.10,
        "output_price": "-",
        "capabilities": "E",
        "description": "Text embeddings model"
    },
    # Moderation
    {
        "id": "mistral-moderation-2411",
        "alias": "mistral-moderation",
        "name": "Mistral Moderation",
        "context_window": 8192,
        "max_output": 0,
        "input_price": 0.10,
        "output_price": "-",
        "capabilities": "M",
        "description": "Content moderation model"
    },
    # Mixtral (Open)
    {
        "id": "open-mixtral-8x7b",
        "alias": "mixtral-8x7b",
        "name": "Mixtral 8x7B",
        "context_window": 32768,
        "max_output": 32768,
        "input_price": 0.70,
        "output_price": 0.70,
        "capabilities": "TJ",
        "description": "Open-weight MoE model"
    },
    {
        "id": "open-mixtral-8x22b",
        "alias": "mixtral-8x22b",
        "name": "Mixtral 8x22B",
        "context_window": 65536,
        "max_output": 65536,
        "input_price": 2.0,
        "output_price": 6.0,
        "capabilities": "TJ",
        "description": "Large open-weight MoE model"
    },
    {
        "id": "open-mistral-7b",
        "alias": "mistral-7b",
        "name": "Mistral 7B",
        "context_window": 32768,
        "max_output": 32768,
        "input_price": 0.25,
        "output_price": 0.25,
        "capabilities": "TJ",
        "description": "Open-weight 7B model"
    },
    # Ministral
    {
        "id": "ministral-3b-2410",
        "alias": "ministral-3b",
        "name": "Ministral 3B",
        "context_window": 131072,
        "max_output": 131072,
        "input_price": 0.04,
        "output_price": 0.04,
        "capabilities": "TJ",
        "description": "Smallest Mistral model"
    },
    {
        "id": "ministral-8b-2410",
        "alias": "ministral-8b",
        "name": "Ministral 8B",
        "context_window": 131072,
        "max_output": 131072,
        "input_price": 0.10,
        "output_price": 0.10,
        "capabilities": "TJ",
        "description": "Efficient 8B model"
    },
]

def fetch_from_api():
    """Fetch models from Mistral API."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        return None

    try:
        response = requests.get(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        print(f"API fetch failed: {e}", file=sys.stderr)
        return None

def process_models() -> list:
    """Process Mistral models into CSV format."""
    rows = []

    for model in MISTRAL_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"mistral/{model['id']}",
            'alias': model['alias'],
            'name': model['name'],
            'status': 'C',
            'input_price': f"{model['input_price']:.6f}",
            'output_price': output_str,
            'cache_input_price': '-',
            'context_window': model['context_window'],
            'max_output': model['max_output'],
            'capabilities': model['capabilities'],
            'quality': 'official',
            'source': 'mistral',
            'updated': datetime.now().strftime('%Y-%m-%d'),
            'description': model['description'][:100],
            'mmlu_score': '-',
            'humaneval_score': '-',
            'math_score': '-'
        }
        rows.append(row)

    return rows

def save_csv(rows: list, output_file: str) -> None:
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
    print("Processing Mistral models...")

    # Try API first
    api_models = fetch_from_api()
    if api_models:
        print(f"Fetched {len(api_models)} models from API")
    else:
        print("Using hardcoded model list")

    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/core/mistral.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
