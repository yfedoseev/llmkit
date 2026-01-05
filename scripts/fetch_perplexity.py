#!/usr/bin/env python3
"""
Fetch Perplexity models.
Uses hardcoded list from official documentation.
Perplexity specializes in search-augmented AI.
"""

import csv
import sys
from datetime import datetime

PERPLEXITY_MODELS = [
    # Sonar Series (Search-augmented)
    {
        "id": "sonar-pro",
        "alias": "sonar-pro",
        "name": "Sonar Pro",
        "context_window": 200000,
        "max_output": 8000,
        "input_price": 3.0,
        "output_price": 15.0,
        "search_price": 5.0,
        "capabilities": "TJS",
        "description": "Advanced search model for complex research tasks"
    },
    {
        "id": "sonar",
        "alias": "sonar",
        "name": "Sonar",
        "context_window": 127072,
        "max_output": 8000,
        "input_price": 1.0,
        "output_price": 1.0,
        "search_price": 5.0,
        "capabilities": "TJS",
        "description": "Fast search-augmented model"
    },
    {
        "id": "sonar-deep-research",
        "alias": "sonar-deep",
        "name": "Sonar Deep Research",
        "context_window": 127072,
        "max_output": 8000,
        "input_price": 2.0,
        "output_price": 8.0,
        "search_price": 5.0,
        "capabilities": "TJK",
        "description": "Multi-step reasoning with comprehensive research"
    },
    {
        "id": "sonar-reasoning-pro",
        "alias": "sonar-reasoning-pro",
        "name": "Sonar Reasoning Pro",
        "context_window": 127072,
        "max_output": 8000,
        "input_price": 2.0,
        "output_price": 8.0,
        "search_price": 5.0,
        "capabilities": "TJK",
        "description": "Chain-of-thought reasoning with search"
    },
    {
        "id": "sonar-reasoning",
        "alias": "sonar-reasoning",
        "name": "Sonar Reasoning",
        "context_window": 127072,
        "max_output": 8000,
        "input_price": 1.0,
        "output_price": 5.0,
        "search_price": 5.0,
        "capabilities": "TJK",
        "description": "Fast reasoning with search augmentation"
    },
    # R1 Series (Reasoning without search)
    {
        "id": "r1-1776",
        "alias": "r1-1776",
        "name": "R1 1776",
        "context_window": 127072,
        "max_output": 8000,
        "input_price": 2.0,
        "output_price": 8.0,
        "search_price": "-",
        "capabilities": "TJK",
        "description": "Post-trained reasoning model"
    },
]

def process_models() -> list:
    """Process Perplexity models into CSV format."""
    rows = []

    for model in PERPLEXITY_MODELS:
        row = {
            'id': f"perplexity/{model['id']}",
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
            'source': 'perplexity',
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
    print("Processing Perplexity models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/inference/perplexity.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
