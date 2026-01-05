#!/usr/bin/env python3
"""
Fetch Yi/01.AI models.
01.AI is a Chinese AI company founded by Kai-Fu Lee.
"""

import csv
import sys
from datetime import datetime

YI_MODELS = [
    # Yi-Lightning (Latest)
    {
        "id": "yi-lightning",
        "alias": "yi-lightning",
        "name": "Yi Lightning",
        "context_window": 16384,
        "max_output": 16384,
        "input_price": 0.14,
        "output_price": 0.14,
        "capabilities": "TJS",
        "description": "Fast and affordable model"
    },
    # Yi-Large Series
    {
        "id": "yi-large",
        "alias": "yi-large",
        "name": "Yi Large",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.42,
        "output_price": 0.42,
        "capabilities": "TJS",
        "description": "Flagship large model"
    },
    {
        "id": "yi-large-rag",
        "alias": "yi-large-rag",
        "name": "Yi Large RAG",
        "context_window": 16384,
        "max_output": 8192,
        "input_price": 0.35,
        "output_price": 0.35,
        "capabilities": "TJS",
        "description": "RAG-optimized model"
    },
    {
        "id": "yi-large-turbo",
        "alias": "yi-large-turbo",
        "name": "Yi Large Turbo",
        "context_window": 16384,
        "max_output": 8192,
        "input_price": 0.17,
        "output_price": 0.17,
        "capabilities": "TJS",
        "description": "Fast large model"
    },
    {
        "id": "yi-large-fc",
        "alias": "yi-large-fc",
        "name": "Yi Large FC",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.42,
        "output_price": 0.42,
        "capabilities": "TJS",
        "description": "Function calling model"
    },
    # Yi-Medium Series
    {
        "id": "yi-medium",
        "alias": "yi-medium",
        "name": "Yi Medium",
        "context_window": 16384,
        "max_output": 8192,
        "input_price": 0.035,
        "output_price": 0.035,
        "capabilities": "TJ",
        "description": "Balanced medium model"
    },
    {
        "id": "yi-medium-200k",
        "alias": "yi-medium-200k",
        "name": "Yi Medium 200K",
        "context_window": 204800,
        "max_output": 8192,
        "input_price": 0.17,
        "output_price": 0.17,
        "capabilities": "TJ",
        "description": "200K context model"
    },
    # Yi-Spark
    {
        "id": "yi-spark",
        "alias": "yi-spark",
        "name": "Yi Spark",
        "context_window": 16384,
        "max_output": 8192,
        "input_price": 0.0014,
        "output_price": 0.0014,
        "capabilities": "TJ",
        "description": "Lightweight affordable model"
    },
    # Yi-Vision
    {
        "id": "yi-vision",
        "alias": "yi-vision",
        "name": "Yi Vision",
        "context_window": 16384,
        "max_output": 8192,
        "input_price": 0.08,
        "output_price": 0.08,
        "capabilities": "VTJ",
        "description": "Vision-language model"
    },
]

def process_models() -> list:
    """Process Yi models into CSV format."""
    rows = []

    for model in YI_MODELS:
        row = {
            'id': f"yi/{model['id']}",
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
            'source': 'yi',
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
    print("Processing Yi models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/chinese/yi.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
