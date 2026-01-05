#!/usr/bin/env python3
"""
Fetch Cohere models.
Uses hardcoded list from official documentation.
"""

import csv
import sys
from datetime import datetime

COHERE_MODELS = [
    # Command R Series
    {
        "id": "command-r-plus-08-2024",
        "alias": "command-r-plus",
        "name": "Command R+",
        "context_window": 128000,
        "max_output": 4096,
        "input_price": 2.50,
        "output_price": 10.0,
        "capabilities": "VTJS",
        "description": "Most capable Command model for complex tasks"
    },
    {
        "id": "command-r-08-2024",
        "alias": "command-r",
        "name": "Command R",
        "context_window": 128000,
        "max_output": 4096,
        "input_price": 0.15,
        "output_price": 0.60,
        "capabilities": "VTJS",
        "description": "Balanced performance for RAG and tool use"
    },
    {
        "id": "command-r7b-12-2024",
        "alias": "command-r7b",
        "name": "Command R7B",
        "context_window": 128000,
        "max_output": 4096,
        "input_price": 0.0375,
        "output_price": 0.15,
        "capabilities": "TJS",
        "description": "Smallest Command model, highly efficient"
    },
    # Command A
    {
        "id": "command-a-03-2025",
        "alias": "command-a",
        "name": "Command A",
        "context_window": 256000,
        "max_output": 8192,
        "input_price": 2.50,
        "output_price": 10.0,
        "capabilities": "VTJS",
        "description": "Agent-focused model with 256K context"
    },
    # Legacy Command
    {
        "id": "command",
        "alias": "command",
        "name": "Command",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 1.0,
        "output_price": 2.0,
        "capabilities": "TJ",
        "description": "Legacy command model"
    },
    {
        "id": "command-light",
        "alias": "command-light",
        "name": "Command Light",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.30,
        "output_price": 0.60,
        "capabilities": "TJ",
        "description": "Legacy lightweight command model"
    },
    # Aya Series (Multilingual)
    {
        "id": "aya-expanse-32b",
        "alias": "aya-expanse-32b",
        "name": "Aya Expanse 32B",
        "context_window": 128000,
        "max_output": 4096,
        "input_price": 0.50,
        "output_price": 1.50,
        "capabilities": "TJ",
        "description": "Multilingual model supporting 23 languages"
    },
    {
        "id": "aya-expanse-8b",
        "alias": "aya-expanse-8b",
        "name": "Aya Expanse 8B",
        "context_window": 8192,
        "max_output": 4096,
        "input_price": 0.05,
        "output_price": 0.10,
        "capabilities": "TJ",
        "description": "Efficient multilingual model"
    },
    # Embeddings
    {
        "id": "embed-english-v3.0",
        "alias": "embed-english-v3",
        "name": "Embed English v3",
        "context_window": 512,
        "max_output": 1024,
        "input_price": 0.10,
        "output_price": "-",
        "capabilities": "E",
        "description": "English text embeddings"
    },
    {
        "id": "embed-multilingual-v3.0",
        "alias": "embed-multilingual-v3",
        "name": "Embed Multilingual v3",
        "context_window": 512,
        "max_output": 1024,
        "input_price": 0.10,
        "output_price": "-",
        "capabilities": "E",
        "description": "Multilingual text embeddings"
    },
    {
        "id": "embed-english-light-v3.0",
        "alias": "embed-english-light",
        "name": "Embed English Light v3",
        "context_window": 512,
        "max_output": 384,
        "input_price": 0.10,
        "output_price": "-",
        "capabilities": "E",
        "description": "Lightweight English embeddings"
    },
    {
        "id": "embed-multilingual-light-v3.0",
        "alias": "embed-multilingual-light",
        "name": "Embed Multilingual Light v3",
        "context_window": 512,
        "max_output": 384,
        "input_price": 0.10,
        "output_price": "-",
        "capabilities": "E",
        "description": "Lightweight multilingual embeddings"
    },
    # Rerank
    {
        "id": "rerank-v3.5",
        "alias": "rerank-v3.5",
        "name": "Rerank v3.5",
        "context_window": 4096,
        "max_output": 0,
        "input_price": 2.0,
        "output_price": "-",
        "capabilities": "R",
        "description": "Document reranking model, $2/1K searches"
    },
    {
        "id": "rerank-english-v3.0",
        "alias": "rerank-english-v3",
        "name": "Rerank English v3",
        "context_window": 4096,
        "max_output": 0,
        "input_price": 2.0,
        "output_price": "-",
        "capabilities": "R",
        "description": "English document reranking"
    },
    {
        "id": "rerank-multilingual-v3.0",
        "alias": "rerank-multilingual-v3",
        "name": "Rerank Multilingual v3",
        "context_window": 4096,
        "max_output": 0,
        "input_price": 2.0,
        "output_price": "-",
        "capabilities": "R",
        "description": "Multilingual document reranking"
    },
]

def process_models() -> list:
    """Process Cohere models into CSV format."""
    rows = []

    for model in COHERE_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"cohere/{model['id']}",
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
            'source': 'cohere',
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
    print("Processing Cohere models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/core/cohere.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
