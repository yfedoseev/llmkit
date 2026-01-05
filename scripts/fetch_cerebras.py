#!/usr/bin/env python3
"""
Fetch Cerebras models.
Cerebras provides ultra-fast inference on custom Wafer-Scale hardware.
"""

import csv
import sys
from datetime import datetime

CEREBRAS_MODELS = [
    # Llama 3.1 Series
    {
        "id": "llama3.1-8b",
        "alias": "llama-3.1-8b",
        "name": "Llama 3.1 8B",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.10,
        "output_price": 0.10,
        "capabilities": "TJS",
        "description": "Meta Llama 3.1 8B on Cerebras WSE, ultra-fast"
    },
    {
        "id": "llama3.1-70b",
        "alias": "llama-3.1-70b",
        "name": "Llama 3.1 70B",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.60,
        "output_price": 0.60,
        "capabilities": "TJS",
        "description": "Meta Llama 3.1 70B on Cerebras WSE"
    },
    # Llama 3.3 Series
    {
        "id": "llama-3.3-70b",
        "alias": "llama-3.3-70b",
        "name": "Llama 3.3 70B",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.85,
        "output_price": 1.20,
        "capabilities": "TJS",
        "description": "Latest Llama 3.3 on Cerebras WSE"
    },
    # Qwen Series
    {
        "id": "qwen-2.5-32b",
        "alias": "qwen-2.5-32b",
        "name": "Qwen 2.5 32B",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.20,
        "output_price": 0.20,
        "capabilities": "TJS",
        "description": "Alibaba Qwen 2.5 32B on Cerebras"
    },
    {
        "id": "qwen-2.5-coder-32b",
        "alias": "qwen-coder-32b",
        "name": "Qwen 2.5 Coder 32B",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.20,
        "output_price": 0.20,
        "capabilities": "TJS",
        "description": "Qwen 2.5 Coder for code generation"
    },
    # DeepSeek R1
    {
        "id": "deepseek-r1-distill-llama-70b",
        "alias": "deepseek-r1-70b",
        "name": "DeepSeek R1 Distill 70B",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.85,
        "output_price": 1.20,
        "capabilities": "TJK",
        "description": "DeepSeek R1 reasoning distilled to Llama"
    },
]

def process_models() -> list:
    """Process Cerebras models into CSV format."""
    rows = []

    for model in CEREBRAS_MODELS:
        row = {
            'id': f"cerebras/{model['id']}",
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
            'source': 'cerebras',
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
    print("Processing Cerebras models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/inference/cerebras.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
