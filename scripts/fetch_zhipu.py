#!/usr/bin/env python3
"""
Fetch Zhipu/GLM models.
Zhipu AI is a Chinese AI company known for GLM models with 200K context.
"""

import csv
import sys
from datetime import datetime

ZHIPU_MODELS = [
    # GLM-4 Series (Latest)
    {
        "id": "glm-4-plus",
        "alias": "glm-4-plus",
        "name": "GLM-4 Plus",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.69,
        "output_price": 0.69,
        "capabilities": "VTJS",
        "description": "Enhanced GLM-4 for complex tasks"
    },
    {
        "id": "glm-4-0520",
        "alias": "glm-4",
        "name": "GLM-4",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.14,
        "output_price": 0.14,
        "capabilities": "VTJS",
        "description": "Standard GLM-4 model"
    },
    {
        "id": "glm-4-air",
        "alias": "glm-4-air",
        "name": "GLM-4 Air",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.0014,
        "output_price": 0.0014,
        "capabilities": "TJS",
        "description": "Lightweight GLM-4 variant"
    },
    {
        "id": "glm-4-airx",
        "alias": "glm-4-airx",
        "name": "GLM-4 AirX",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.014,
        "output_price": 0.014,
        "capabilities": "TJS",
        "description": "Fast inference GLM-4"
    },
    {
        "id": "glm-4-flash",
        "alias": "glm-4-flash",
        "name": "GLM-4 Flash",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.0,
        "output_price": 0.0,
        "capabilities": "TJS",
        "description": "Free tier GLM-4"
    },
    {
        "id": "glm-4-flashx",
        "alias": "glm-4-flashx",
        "name": "GLM-4 FlashX",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.0,
        "output_price": 0.0,
        "capabilities": "TJS",
        "description": "Free tier fast GLM-4"
    },
    {
        "id": "glm-4-long",
        "alias": "glm-4-long",
        "name": "GLM-4 Long",
        "context_window": 1000000,
        "max_output": 8192,
        "input_price": 0.0014,
        "output_price": 0.0014,
        "capabilities": "TJS",
        "description": "1M context for ultra-long documents"
    },
    # GLM-4V (Vision)
    {
        "id": "glm-4v-plus",
        "alias": "glm-4v-plus",
        "name": "GLM-4V Plus",
        "context_window": 8192,
        "max_output": 4096,
        "input_price": 0.14,
        "output_price": 0.14,
        "capabilities": "VTJS",
        "description": "Enhanced vision model"
    },
    {
        "id": "glm-4v",
        "alias": "glm-4v",
        "name": "GLM-4V",
        "context_window": 8192,
        "max_output": 4096,
        "input_price": 0.069,
        "output_price": 0.069,
        "capabilities": "VTJ",
        "description": "Standard vision model"
    },
    {
        "id": "glm-4v-flash",
        "alias": "glm-4v-flash",
        "name": "GLM-4V Flash",
        "context_window": 8192,
        "max_output": 4096,
        "input_price": 0.0,
        "output_price": 0.0,
        "capabilities": "VTJ",
        "description": "Free tier vision model"
    },
    # CodeGeeX
    {
        "id": "codegeex-4",
        "alias": "codegeex-4",
        "name": "CodeGeeX 4",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.0014,
        "output_price": 0.0014,
        "capabilities": "TJS",
        "description": "Specialized code generation"
    },
    # GLM-Z1 (Reasoning)
    {
        "id": "glm-z1-preview",
        "alias": "glm-z1",
        "name": "GLM-Z1 Preview",
        "context_window": 16384,
        "max_output": 16384,
        "input_price": 0.69,
        "output_price": 0.69,
        "capabilities": "TJK",
        "description": "Reasoning model with thinking"
    },
    {
        "id": "glm-z1-air-preview",
        "alias": "glm-z1-air",
        "name": "GLM-Z1 Air Preview",
        "context_window": 16384,
        "max_output": 16384,
        "input_price": 0.069,
        "output_price": 0.069,
        "capabilities": "TJK",
        "description": "Lightweight reasoning model"
    },
    {
        "id": "glm-z1-flash-preview",
        "alias": "glm-z1-flash",
        "name": "GLM-Z1 Flash Preview",
        "context_window": 16384,
        "max_output": 16384,
        "input_price": 0.0,
        "output_price": 0.0,
        "capabilities": "TJK",
        "description": "Free tier reasoning model"
    },
    # Embeddings
    {
        "id": "embedding-3",
        "alias": "embedding-3",
        "name": "Embedding 3",
        "context_window": 8192,
        "max_output": 2048,
        "input_price": 0.0007,
        "output_price": "-",
        "capabilities": "E",
        "description": "Text embeddings"
    },
]

def process_models() -> list:
    """Process Zhipu models into CSV format."""
    rows = []

    for model in ZHIPU_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"zhipu/{model['id']}",
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
            'source': 'zhipu',
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
    print("Processing Zhipu models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/chinese/zhipu.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
