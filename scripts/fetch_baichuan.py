#!/usr/bin/env python3
"""
Fetch Baichuan models.
Baichuan Intelligence is a Chinese AI company with up to 192K context.
"""

import csv
import sys
from datetime import datetime

BAICHUAN_MODELS = [
    # Baichuan4 Series
    {
        "id": "Baichuan4-Turbo",
        "alias": "baichuan4-turbo",
        "name": "Baichuan4 Turbo",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.14,
        "output_price": 0.14,
        "capabilities": "VTJS",
        "description": "Flagship turbo model"
    },
    {
        "id": "Baichuan4-Air",
        "alias": "baichuan4-air",
        "name": "Baichuan4 Air",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.014,
        "output_price": 0.014,
        "capabilities": "TJS",
        "description": "Lightweight efficient model"
    },
    # Baichuan3 Series
    {
        "id": "Baichuan3-Turbo",
        "alias": "baichuan3-turbo",
        "name": "Baichuan3 Turbo",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.017,
        "output_price": 0.017,
        "capabilities": "TJS",
        "description": "Previous gen turbo model"
    },
    {
        "id": "Baichuan3-Turbo-128k",
        "alias": "baichuan3-128k",
        "name": "Baichuan3 Turbo 128K",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.034,
        "output_price": 0.034,
        "capabilities": "TJS",
        "description": "128K context model"
    },
    # Baichuan2 Series
    {
        "id": "Baichuan2-Turbo",
        "alias": "baichuan2-turbo",
        "name": "Baichuan2 Turbo",
        "context_window": 32768,
        "max_output": 4096,
        "input_price": 0.011,
        "output_price": 0.011,
        "capabilities": "TJ",
        "description": "Legacy turbo model"
    },
    {
        "id": "Baichuan2-Turbo-192k",
        "alias": "baichuan2-192k",
        "name": "Baichuan2 Turbo 192K",
        "context_window": 196608,
        "max_output": 4096,
        "input_price": 0.022,
        "output_price": 0.022,
        "capabilities": "TJ",
        "description": "192K context legacy model"
    },
    # Embeddings
    {
        "id": "Baichuan-Text-Embedding",
        "alias": "baichuan-embed",
        "name": "Baichuan Text Embedding",
        "context_window": 512,
        "max_output": 1024,
        "input_price": 0.0007,
        "output_price": "-",
        "capabilities": "E",
        "description": "Text embeddings"
    },
]

def process_models() -> list:
    """Process Baichuan models into CSV format."""
    rows = []

    for model in BAICHUAN_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"baichuan/{model['id']}",
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
            'source': 'baichuan',
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
    print("Processing Baichuan models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/chinese/baichuan.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
