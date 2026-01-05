#!/usr/bin/env python3
"""
Fetch Nebius AI models.
Nebius is a European cloud provider with LLM inference.
"""

import csv
import sys
from datetime import datetime

NEBIUS_MODELS = [
    # Llama 3 Series
    {
        "id": "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "alias": "llama-3.1-405b",
        "name": "Llama 3.1 405B",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 2.35,
        "output_price": 2.35,
        "capabilities": "TJS",
        "description": "Largest Llama on Nebius"
    },
    {
        "id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "alias": "llama-3.1-70b",
        "name": "Llama 3.1 70B",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.35,
        "output_price": 0.40,
        "capabilities": "TJS",
        "description": "Llama 3.1 70B on Nebius"
    },
    {
        "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "alias": "llama-3.1-8b",
        "name": "Llama 3.1 8B",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.06,
        "output_price": 0.06,
        "capabilities": "TJS",
        "description": "Llama 3.1 8B on Nebius"
    },
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "alias": "llama-3.3-70b",
        "name": "Llama 3.3 70B",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.35,
        "output_price": 0.40,
        "capabilities": "TJS",
        "description": "Llama 3.3 70B on Nebius"
    },
    # Llama Vision
    {
        "id": "meta-llama/Llama-3.2-90B-Vision-Instruct",
        "alias": "llama-3.2-90b-vision",
        "name": "Llama 3.2 90B Vision",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.35,
        "output_price": 0.40,
        "capabilities": "VTJS",
        "description": "Vision model on Nebius"
    },
    {
        "id": "meta-llama/Llama-3.2-11B-Vision-Instruct",
        "alias": "llama-3.2-11b-vision",
        "name": "Llama 3.2 11B Vision",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.06,
        "output_price": 0.06,
        "capabilities": "VTJS",
        "description": "Compact vision model"
    },
    # Qwen Series
    {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "alias": "qwen-2.5-72b",
        "name": "Qwen 2.5 72B",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.40,
        "output_price": 0.40,
        "capabilities": "TJS",
        "description": "Qwen 2.5 72B on Nebius"
    },
    {
        "id": "Qwen/QwQ-32B-Preview",
        "alias": "qwq-32b",
        "name": "QwQ 32B",
        "context_window": 32768,
        "max_output": 32768,
        "input_price": 0.15,
        "output_price": 0.15,
        "capabilities": "TJK",
        "description": "QwQ reasoning model"
    },
    # DeepSeek
    {
        "id": "deepseek-ai/DeepSeek-V3",
        "alias": "deepseek-v3",
        "name": "DeepSeek V3",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.35,
        "output_price": 0.90,
        "capabilities": "TJS",
        "description": "DeepSeek V3 on Nebius"
    },
    {
        "id": "deepseek-ai/DeepSeek-R1",
        "alias": "deepseek-r1",
        "name": "DeepSeek R1",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.35,
        "output_price": 2.15,
        "capabilities": "TJK",
        "description": "DeepSeek R1 reasoning"
    },
    # Mistral
    {
        "id": "mistralai/Mistral-Large-Instruct-2411",
        "alias": "mistral-large",
        "name": "Mistral Large",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.90,
        "output_price": 2.70,
        "capabilities": "TJS",
        "description": "Mistral Large on Nebius"
    },
    {
        "id": "mistralai/Mistral-Nemo-Instruct-2407",
        "alias": "mistral-nemo",
        "name": "Mistral Nemo",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.06,
        "output_price": 0.06,
        "capabilities": "TJS",
        "description": "Mistral Nemo on Nebius"
    },
    # Phi
    {
        "id": "microsoft/Phi-3.5-MoE-instruct",
        "alias": "phi-3.5-moe",
        "name": "Phi 3.5 MoE",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.12,
        "output_price": 0.12,
        "capabilities": "TJS",
        "description": "Microsoft Phi 3.5 MoE"
    },
    # Embeddings
    {
        "id": "BAAI/bge-m3",
        "alias": "bge-m3",
        "name": "BGE M3",
        "context_window": 8192,
        "max_output": 1024,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "E",
        "description": "BGE M3 embeddings"
    },
    {
        "id": "BAAI/bge-multilingual-gemma2",
        "alias": "bge-gemma2",
        "name": "BGE Multilingual Gemma2",
        "context_window": 8192,
        "max_output": 1024,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "E",
        "description": "Multilingual embeddings"
    },
]

def process_models() -> list:
    """Process Nebius models into CSV format."""
    rows = []

    for model in NEBIUS_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        model_id = model['id'].split('/')[-1]
        row = {
            'id': f"nebius/{model_id}",
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
            'source': 'nebius',
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
    print("Processing Nebius models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/inference/nebius.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
