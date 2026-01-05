#!/usr/bin/env python3
"""
Fetch Volcengine/ByteDance models.
Volcengine is ByteDance's cloud platform hosting Doubao models.
"""

import csv
import sys
from datetime import datetime

VOLCENGINE_MODELS = [
    # Doubao Pro Series
    {
        "id": "doubao-pro-256k",
        "alias": "doubao-pro-256k",
        "name": "Doubao Pro 256K",
        "context_window": 262144,
        "max_output": 8192,
        "input_price": 0.69,
        "output_price": 1.24,
        "capabilities": "VTJS",
        "description": "Flagship 256K context model"
    },
    {
        "id": "doubao-pro-128k",
        "alias": "doubao-pro-128k",
        "name": "Doubao Pro 128K",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.69,
        "output_price": 1.24,
        "capabilities": "VTJS",
        "description": "High-performance 128K model"
    },
    {
        "id": "doubao-pro-32k",
        "alias": "doubao-pro-32k",
        "name": "Doubao Pro 32K",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.11,
        "output_price": 0.24,
        "capabilities": "VTJS",
        "description": "Balanced 32K model"
    },
    {
        "id": "doubao-pro-4k",
        "alias": "doubao-pro-4k",
        "name": "Doubao Pro 4K",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.11,
        "output_price": 0.24,
        "capabilities": "VTJS",
        "description": "Fast 4K model"
    },
    # Doubao Lite Series
    {
        "id": "doubao-lite-128k",
        "alias": "doubao-lite-128k",
        "name": "Doubao Lite 128K",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.041,
        "output_price": 0.124,
        "capabilities": "TJS",
        "description": "Lightweight 128K model"
    },
    {
        "id": "doubao-lite-32k",
        "alias": "doubao-lite-32k",
        "name": "Doubao Lite 32K",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.041,
        "output_price": 0.124,
        "capabilities": "TJS",
        "description": "Lightweight 32K model"
    },
    {
        "id": "doubao-lite-4k",
        "alias": "doubao-lite-4k",
        "name": "Doubao Lite 4K",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.041,
        "output_price": 0.124,
        "capabilities": "TJS",
        "description": "Lightweight 4K model"
    },
    # Doubao Vision
    {
        "id": "doubao-vision-pro-32k",
        "alias": "doubao-vision-pro",
        "name": "Doubao Vision Pro 32K",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.27,
        "output_price": 0.27,
        "capabilities": "VTJS",
        "description": "Vision-language model"
    },
    {
        "id": "doubao-vision-lite-32k",
        "alias": "doubao-vision-lite",
        "name": "Doubao Vision Lite 32K",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.11,
        "output_price": 0.11,
        "capabilities": "VTJ",
        "description": "Lightweight vision model"
    },
    # Thinking Models
    {
        "id": "doubao-1.5-thinking-pro",
        "alias": "doubao-thinking-pro",
        "name": "Doubao 1.5 Thinking Pro",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.55,
        "output_price": 2.19,
        "capabilities": "VTJSK",
        "description": "Chain-of-thought reasoning model"
    },
    {
        "id": "doubao-1.5-thinking-pro-m",
        "alias": "doubao-thinking-m",
        "name": "Doubao 1.5 Thinking Pro M",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.28,
        "output_price": 0.82,
        "capabilities": "VTJSK",
        "description": "Medium thinking model"
    },
    # Embeddings
    {
        "id": "doubao-embedding",
        "alias": "doubao-embed",
        "name": "Doubao Embedding",
        "context_window": 4096,
        "max_output": 2048,
        "input_price": 0.0007,
        "output_price": "-",
        "capabilities": "E",
        "description": "Text embeddings"
    },
    {
        "id": "doubao-embedding-large",
        "alias": "doubao-embed-large",
        "name": "Doubao Embedding Large",
        "context_window": 4096,
        "max_output": 2048,
        "input_price": 0.0007,
        "output_price": "-",
        "capabilities": "E",
        "description": "Large text embeddings"
    },
]

def process_models() -> list:
    """Process Volcengine models into CSV format."""
    rows = []

    for model in VOLCENGINE_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"volcengine/{model['id']}",
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
            'source': 'volcengine',
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
    print("Processing Volcengine models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/chinese/volcengine.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
