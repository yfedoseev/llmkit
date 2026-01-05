#!/usr/bin/env python3
"""
Fetch SambaNova models.
SambaNova provides ultra-fast inference on custom RDU hardware.
"""

import csv
import sys
from datetime import datetime

SAMBANOVA_MODELS = [
    # Meta Llama 3.3
    {
        "id": "Meta-Llama-3.3-70B-Instruct",
        "alias": "llama-3.3-70b",
        "name": "Llama 3.3 70B Instruct",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.40,
        "output_price": 0.80,
        "capabilities": "TJS",
        "description": "Meta Llama 3.3 70B on SambaNova RDU"
    },
    # Meta Llama 3.1 Series
    {
        "id": "Meta-Llama-3.1-405B-Instruct",
        "alias": "llama-3.1-405b",
        "name": "Llama 3.1 405B Instruct",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 5.0,
        "output_price": 10.0,
        "capabilities": "TJS",
        "description": "Largest Llama model, 405B parameters"
    },
    {
        "id": "Meta-Llama-3.1-70B-Instruct",
        "alias": "llama-3.1-70b",
        "name": "Llama 3.1 70B Instruct",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.40,
        "output_price": 0.80,
        "capabilities": "TJS",
        "description": "Meta Llama 3.1 70B on SambaNova"
    },
    {
        "id": "Meta-Llama-3.1-8B-Instruct",
        "alias": "llama-3.1-8b",
        "name": "Llama 3.1 8B Instruct",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.10,
        "output_price": 0.20,
        "capabilities": "TJS",
        "description": "Meta Llama 3.1 8B on SambaNova"
    },
    # Llama 3.2 Vision
    {
        "id": "Llama-3.2-90B-Vision-Instruct",
        "alias": "llama-3.2-90b-vision",
        "name": "Llama 3.2 90B Vision",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.60,
        "output_price": 1.20,
        "capabilities": "VTJS",
        "description": "Llama 3.2 90B with vision capabilities"
    },
    {
        "id": "Llama-3.2-11B-Vision-Instruct",
        "alias": "llama-3.2-11b-vision",
        "name": "Llama 3.2 11B Vision",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.15,
        "output_price": 0.30,
        "capabilities": "VTJS",
        "description": "Compact Llama 3.2 with vision"
    },
    # Qwen Series
    {
        "id": "Qwen2.5-72B-Instruct",
        "alias": "qwen-2.5-72b",
        "name": "Qwen 2.5 72B Instruct",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.40,
        "output_price": 0.80,
        "capabilities": "TJS",
        "description": "Alibaba Qwen 2.5 72B on SambaNova"
    },
    {
        "id": "Qwen2.5-Coder-32B-Instruct",
        "alias": "qwen-coder-32b",
        "name": "Qwen 2.5 Coder 32B",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.20,
        "output_price": 0.40,
        "capabilities": "TJS",
        "description": "Qwen 2.5 specialized for code"
    },
    {
        "id": "QwQ-32B",
        "alias": "qwq-32b",
        "name": "QwQ 32B",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.20,
        "output_price": 0.40,
        "capabilities": "TJK",
        "description": "Alibaba QwQ reasoning model"
    },
    # DeepSeek
    {
        "id": "DeepSeek-R1",
        "alias": "deepseek-r1",
        "name": "DeepSeek R1",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.60,
        "output_price": 1.20,
        "capabilities": "TJK",
        "description": "DeepSeek R1 reasoning model"
    },
    {
        "id": "DeepSeek-R1-Distill-Llama-70B",
        "alias": "deepseek-r1-70b",
        "name": "DeepSeek R1 Distill 70B",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.40,
        "output_price": 0.80,
        "capabilities": "TJK",
        "description": "DeepSeek R1 distilled to Llama 70B"
    },
]

def process_models() -> list:
    """Process SambaNova models into CSV format."""
    rows = []

    for model in SAMBANOVA_MODELS:
        row = {
            'id': f"sambanova/{model['id']}",
            'alias': model['alias'],
            'name': model['name'],
            'status': 'C',
            'input_price': f"{model['input_price']:.6f}",
            'output_price': f"{model['output_price']:.6f}",
            'cache_input_price': '-',
            'context_window': model['context_window'],
            'max_output': model['max_output'],
            'capabilities': model['capabilities'],
            'quality': 'official',
            'source': 'sambanova',
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
    print("Processing SambaNova models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/inference/sambanova.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
