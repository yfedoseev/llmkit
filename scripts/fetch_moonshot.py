#!/usr/bin/env python3
"""
Fetch Moonshot/Kimi models.
Moonshot AI is a Chinese AI company known for Kimi chatbot with 256K context.
"""

import csv
import sys
from datetime import datetime

MOONSHOT_MODELS = [
    # Kimi K2 (Latest - 1T MoE)
    {
        "id": "kimi-k2-0711-preview",
        "alias": "kimi-k2",
        "name": "Kimi K2",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.55,
        "output_price": 2.0,
        "capabilities": "VTJSK",
        "description": "1T MoE model with thinking capabilities"
    },
    # Moonshot v1 Series
    {
        "id": "moonshot-v1-256k",
        "alias": "moonshot-256k",
        "name": "Moonshot V1 256K",
        "context_window": 262144,
        "max_output": 8192,
        "input_price": 0.84,
        "output_price": 0.84,
        "capabilities": "TJS",
        "description": "256K context for long document processing"
    },
    {
        "id": "moonshot-v1-128k",
        "alias": "moonshot-128k",
        "name": "Moonshot V1 128K",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.56,
        "output_price": 0.56,
        "capabilities": "TJS",
        "description": "128K context balanced model"
    },
    {
        "id": "moonshot-v1-32k",
        "alias": "moonshot-32k",
        "name": "Moonshot V1 32K",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.17,
        "output_price": 0.17,
        "capabilities": "TJS",
        "description": "32K context efficient model"
    },
    {
        "id": "moonshot-v1-8k",
        "alias": "moonshot-8k",
        "name": "Moonshot V1 8K",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.08,
        "output_price": 0.08,
        "capabilities": "TJS",
        "description": "8K context fast model"
    },
    # Web search
    {
        "id": "moonshot-v1-8k-web",
        "alias": "moonshot-web",
        "name": "Moonshot V1 Web",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.08,
        "output_price": 0.08,
        "capabilities": "TJS",
        "description": "8K with web search capability"
    },
]

def process_models() -> list:
    """Process Moonshot models into CSV format."""
    rows = []

    for model in MOONSHOT_MODELS:
        row = {
            'id': f"moonshot/{model['id']}",
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
            'source': 'moonshot',
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
    print("Processing Moonshot models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/chinese/moonshot.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
