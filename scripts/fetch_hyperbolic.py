#!/usr/bin/env python3
"""
Fetch Hyperbolic models.
Hyperbolic provides affordable LLM inference.
"""

import csv
import sys
from datetime import datetime

HYPERBOLIC_MODELS = [
    # DeepSeek
    {
        "id": "deepseek-ai/DeepSeek-V3",
        "alias": "deepseek-v3",
        "name": "DeepSeek V3",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.50,
        "output_price": 1.00,
        "capabilities": "TJS",
        "description": "DeepSeek V3 on Hyperbolic"
    },
    {
        "id": "deepseek-ai/DeepSeek-R1",
        "alias": "deepseek-r1",
        "name": "DeepSeek R1",
        "context_window": 65536,
        "max_output": 8192,
        "input_price": 0.50,
        "output_price": 2.00,
        "capabilities": "TJK",
        "description": "DeepSeek R1 reasoning"
    },
    {
        "id": "deepseek-ai/DeepSeek-R1-Zero",
        "alias": "deepseek-r1-zero",
        "name": "DeepSeek R1 Zero",
        "context_window": 65536,
        "max_output": 8192,
        "input_price": 0.50,
        "output_price": 2.00,
        "capabilities": "TJK",
        "description": "DeepSeek R1 Zero-shot"
    },
    # Llama 3 Series
    {
        "id": "meta-llama/Meta-Llama-3.1-405B-Instruct",
        "alias": "llama-3.1-405b",
        "name": "Llama 3.1 405B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 3.00,
        "output_price": 3.00,
        "capabilities": "TJS",
        "description": "Llama 3.1 405B on Hyperbolic"
    },
    {
        "id": "meta-llama/Meta-Llama-3.1-70B-Instruct",
        "alias": "llama-3.1-70b",
        "name": "Llama 3.1 70B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.40,
        "output_price": 0.40,
        "capabilities": "TJS",
        "description": "Llama 3.1 70B on Hyperbolic"
    },
    {
        "id": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "alias": "llama-3.1-8b",
        "name": "Llama 3.1 8B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.04,
        "output_price": 0.04,
        "capabilities": "TJS",
        "description": "Llama 3.1 8B on Hyperbolic"
    },
    {
        "id": "meta-llama/Llama-3.3-70B-Instruct",
        "alias": "llama-3.3-70b",
        "name": "Llama 3.3 70B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.40,
        "output_price": 0.40,
        "capabilities": "TJS",
        "description": "Llama 3.3 70B on Hyperbolic"
    },
    # Qwen
    {
        "id": "Qwen/Qwen2.5-72B-Instruct",
        "alias": "qwen-2.5-72b",
        "name": "Qwen 2.5 72B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.40,
        "output_price": 0.40,
        "capabilities": "TJS",
        "description": "Qwen 2.5 72B on Hyperbolic"
    },
    {
        "id": "Qwen/QwQ-32B-Preview",
        "alias": "qwq-32b",
        "name": "QwQ 32B",
        "context_window": 32768,
        "max_output": 32768,
        "input_price": 0.20,
        "output_price": 0.20,
        "capabilities": "TJK",
        "description": "QwQ reasoning model"
    },
    {
        "id": "Qwen/Qwen2.5-Coder-32B-Instruct",
        "alias": "qwen-coder-32b",
        "name": "Qwen 2.5 Coder 32B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.20,
        "output_price": 0.20,
        "capabilities": "TJS",
        "description": "Qwen Coder on Hyperbolic"
    },
    # Hermes
    {
        "id": "NousResearch/Hermes-3-Llama-3.1-70B",
        "alias": "hermes-3-70b",
        "name": "Hermes 3 70B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.40,
        "output_price": 0.40,
        "capabilities": "TJS",
        "description": "Hermes 3 Llama 70B"
    },
    # Image
    {
        "id": "stabilityai/stable-diffusion-xl-base-1.0",
        "alias": "sdxl",
        "name": "SDXL",
        "context_window": 77,
        "max_output": 0,
        "input_price": 0.002,
        "output_price": "-",
        "capabilities": "I",
        "description": "SDXL image generation"
    },
    {
        "id": "black-forest-labs/FLUX.1-dev",
        "alias": "flux-dev",
        "name": "FLUX.1 Dev",
        "context_window": 77,
        "max_output": 0,
        "input_price": 0.003,
        "output_price": "-",
        "capabilities": "I",
        "description": "FLUX image generation"
    },
]

def process_models() -> list:
    """Process Hyperbolic models into CSV format."""
    rows = []

    for model in HYPERBOLIC_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        model_id = model['id'].split('/')[-1]
        row = {
            'id': f"hyperbolic/{model_id}",
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
            'source': 'hyperbolic',
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
    print("Processing Hyperbolic models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/inference/hyperbolic.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
