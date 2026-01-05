#!/usr/bin/env python3
"""
Fetch Novita AI models.
Novita provides 200+ models with flat pricing.
"""

import csv
import sys
from datetime import datetime

NOVITA_MODELS = [
    # Llama Series (Flat pricing)
    {
        "id": "meta-llama/llama-3.3-70b-instruct",
        "alias": "llama-3.3-70b",
        "name": "Llama 3.3 70B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.19,
        "output_price": 0.19,
        "capabilities": "TJS",
        "description": "Llama 3.3 70B on Novita"
    },
    {
        "id": "meta-llama/llama-3.1-405b-instruct",
        "alias": "llama-3.1-405b",
        "name": "Llama 3.1 405B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 1.00,
        "output_price": 1.00,
        "capabilities": "TJS",
        "description": "Llama 3.1 405B on Novita"
    },
    {
        "id": "meta-llama/llama-3.1-70b-instruct",
        "alias": "llama-3.1-70b",
        "name": "Llama 3.1 70B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.19,
        "output_price": 0.19,
        "capabilities": "TJS",
        "description": "Llama 3.1 70B on Novita"
    },
    {
        "id": "meta-llama/llama-3.1-8b-instruct",
        "alias": "llama-3.1-8b",
        "name": "Llama 3.1 8B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.03,
        "output_price": 0.03,
        "capabilities": "TJS",
        "description": "Llama 3.1 8B on Novita"
    },
    # DeepSeek
    {
        "id": "deepseek/deepseek-v3",
        "alias": "deepseek-v3",
        "name": "DeepSeek V3",
        "context_window": 65536,
        "max_output": 8192,
        "input_price": 0.29,
        "output_price": 0.87,
        "capabilities": "TJS",
        "description": "DeepSeek V3 on Novita"
    },
    {
        "id": "deepseek/deepseek-r1",
        "alias": "deepseek-r1",
        "name": "DeepSeek R1",
        "context_window": 65536,
        "max_output": 8192,
        "input_price": 0.29,
        "output_price": 1.20,
        "capabilities": "TJK",
        "description": "DeepSeek R1 reasoning on Novita"
    },
    {
        "id": "deepseek/deepseek-r1-distill-llama-70b",
        "alias": "deepseek-r1-70b",
        "name": "DeepSeek R1 Distill 70B",
        "context_window": 65536,
        "max_output": 8192,
        "input_price": 0.19,
        "output_price": 0.19,
        "capabilities": "TJK",
        "description": "DeepSeek R1 distilled"
    },
    # Qwen
    {
        "id": "qwen/qwen-2.5-72b-instruct",
        "alias": "qwen-2.5-72b",
        "name": "Qwen 2.5 72B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.29,
        "output_price": 0.29,
        "capabilities": "TJS",
        "description": "Qwen 2.5 72B on Novita"
    },
    {
        "id": "qwen/qwen-2.5-coder-32b-instruct",
        "alias": "qwen-coder-32b",
        "name": "Qwen 2.5 Coder 32B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.10,
        "output_price": 0.10,
        "capabilities": "TJS",
        "description": "Qwen Coder on Novita"
    },
    {
        "id": "qwen/qwq-32b-preview",
        "alias": "qwq-32b",
        "name": "QwQ 32B",
        "context_window": 32768,
        "max_output": 32768,
        "input_price": 0.10,
        "output_price": 0.10,
        "capabilities": "TJK",
        "description": "QwQ reasoning on Novita"
    },
    # Mistral
    {
        "id": "mistralai/mistral-large-2411",
        "alias": "mistral-large",
        "name": "Mistral Large",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.80,
        "output_price": 2.40,
        "capabilities": "TJS",
        "description": "Mistral Large on Novita"
    },
    {
        "id": "mistralai/mixtral-8x22b-instruct",
        "alias": "mixtral-8x22b",
        "name": "Mixtral 8x22B",
        "context_window": 65536,
        "max_output": 8192,
        "input_price": 0.29,
        "output_price": 0.29,
        "capabilities": "TJS",
        "description": "Mixtral 8x22B on Novita"
    },
    {
        "id": "mistralai/codestral-2501",
        "alias": "codestral",
        "name": "Codestral",
        "context_window": 262144,
        "max_output": 8192,
        "input_price": 0.10,
        "output_price": 0.30,
        "capabilities": "TJS",
        "description": "Codestral on Novita"
    },
    # Image Generation
    {
        "id": "black-forest-labs/flux-1-schnell",
        "alias": "flux-schnell",
        "name": "FLUX.1 Schnell",
        "context_window": 77,
        "max_output": 0,
        "input_price": 0.003,
        "output_price": "-",
        "capabilities": "I",
        "description": "Fast FLUX image generation"
    },
    {
        "id": "black-forest-labs/flux-1-dev",
        "alias": "flux-dev",
        "name": "FLUX.1 Dev",
        "context_window": 77,
        "max_output": 0,
        "input_price": 0.025,
        "output_price": "-",
        "capabilities": "I",
        "description": "FLUX Dev image generation"
    },
    {
        "id": "stability-ai/sdxl",
        "alias": "sdxl",
        "name": "SDXL",
        "context_window": 77,
        "max_output": 0,
        "input_price": 0.002,
        "output_price": "-",
        "capabilities": "I",
        "description": "SDXL on Novita"
    },
    {
        "id": "stability-ai/sd3-medium",
        "alias": "sd3-medium",
        "name": "SD3 Medium",
        "context_window": 77,
        "max_output": 0,
        "input_price": 0.03,
        "output_price": "-",
        "capabilities": "I",
        "description": "SD3 Medium on Novita"
    },
    # Embeddings
    {
        "id": "sentence-transformers/all-MiniLM-L6-v2",
        "alias": "minilm-l6",
        "name": "MiniLM L6 v2",
        "context_window": 512,
        "max_output": 384,
        "input_price": 0.01,
        "output_price": "-",
        "capabilities": "E",
        "description": "Lightweight embeddings"
    },
    {
        "id": "BAAI/bge-large-en-v1.5",
        "alias": "bge-large",
        "name": "BGE Large EN",
        "context_window": 512,
        "max_output": 1024,
        "input_price": 0.01,
        "output_price": "-",
        "capabilities": "E",
        "description": "BGE embeddings on Novita"
    },
]

def process_models() -> list:
    """Process Novita models into CSV format."""
    rows = []

    for model in NOVITA_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        model_id = model['id'].split('/')[-1]
        row = {
            'id': f"novita/{model_id}",
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
            'source': 'novita',
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
    print("Processing Novita models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/inference/novita.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
