#!/usr/bin/env python3
"""
Fetch Voyage AI models.
Voyage AI provides text embedding models.
"""

import csv
import sys
from datetime import datetime

VOYAGE_MODELS = [
    # Voyage 3.5 (Latest)
    {
        "id": "voyage-3.5",
        "alias": "voyage-3.5",
        "name": "Voyage 3.5",
        "context_window": 32000,
        "max_output": 1024,
        "input_price": 0.06,
        "output_price": "-",
        "capabilities": "E",
        "description": "Latest embedding model, 1024 dimensions"
    },
    # Voyage 3
    {
        "id": "voyage-3",
        "alias": "voyage-3",
        "name": "Voyage 3",
        "context_window": 32000,
        "max_output": 1024,
        "input_price": 0.06,
        "output_price": "-",
        "capabilities": "E",
        "description": "General purpose embeddings"
    },
    {
        "id": "voyage-3-lite",
        "alias": "voyage-3-lite",
        "name": "Voyage 3 Lite",
        "context_window": 32000,
        "max_output": 512,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "E",
        "description": "Lightweight embeddings, 512 dimensions"
    },
    # Code Embeddings
    {
        "id": "voyage-code-3",
        "alias": "voyage-code-3",
        "name": "Voyage Code 3",
        "context_window": 32000,
        "max_output": 1024,
        "input_price": 0.18,
        "output_price": "-",
        "capabilities": "E",
        "description": "Code-optimized embeddings"
    },
    # Finance Embeddings
    {
        "id": "voyage-finance-2",
        "alias": "voyage-finance",
        "name": "Voyage Finance 2",
        "context_window": 32000,
        "max_output": 1024,
        "input_price": 0.12,
        "output_price": "-",
        "capabilities": "E",
        "description": "Finance-optimized embeddings"
    },
    # Law Embeddings
    {
        "id": "voyage-law-2",
        "alias": "voyage-law",
        "name": "Voyage Law 2",
        "context_window": 16000,
        "max_output": 1024,
        "input_price": 0.12,
        "output_price": "-",
        "capabilities": "E",
        "description": "Legal document embeddings"
    },
    # Multilingual
    {
        "id": "voyage-multilingual-2",
        "alias": "voyage-multilingual",
        "name": "Voyage Multilingual 2",
        "context_window": 32000,
        "max_output": 1024,
        "input_price": 0.12,
        "output_price": "-",
        "capabilities": "E",
        "description": "Multilingual embeddings"
    },
    # Voyage 2
    {
        "id": "voyage-02",
        "alias": "voyage-2",
        "name": "Voyage 2",
        "context_window": 16000,
        "max_output": 1024,
        "input_price": 0.10,
        "output_price": "-",
        "capabilities": "E",
        "description": "Previous generation embeddings"
    },
    {
        "id": "voyage-large-2",
        "alias": "voyage-large-2",
        "name": "Voyage Large 2",
        "context_window": 16000,
        "max_output": 1536,
        "input_price": 0.12,
        "output_price": "-",
        "capabilities": "E",
        "description": "Large embeddings, 1536 dimensions"
    },
    {
        "id": "voyage-large-2-instruct",
        "alias": "voyage-large-instruct",
        "name": "Voyage Large 2 Instruct",
        "context_window": 16000,
        "max_output": 1024,
        "input_price": 0.12,
        "output_price": "-",
        "capabilities": "E",
        "description": "Instruction-tuned embeddings"
    },
    # Reranker
    {
        "id": "rerank-2",
        "alias": "rerank-2",
        "name": "Rerank 2",
        "context_window": 32000,
        "max_output": 0,
        "input_price": 0.05,
        "output_price": "-",
        "capabilities": "R",
        "description": "Document reranking, $0.05/1M tokens"
    },
    {
        "id": "rerank-2-lite",
        "alias": "rerank-2-lite",
        "name": "Rerank 2 Lite",
        "context_window": 32000,
        "max_output": 0,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "R",
        "description": "Lightweight reranking, $0.02/1M tokens"
    },
]

def process_models() -> list:
    """Process Voyage AI models into CSV format."""
    rows = []

    for model in VOYAGE_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"voyage/{model['id']}",
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
            'source': 'voyage',
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
    print("Processing Voyage AI models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/specialized/voyage.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
