#!/usr/bin/env python3
"""
Fetch Jina AI models.
Jina AI provides embedding and reranking models.
"""

import csv
import sys
from datetime import datetime

JINA_MODELS = [
    # Jina Embeddings v3
    {
        "id": "jina-embeddings-v3",
        "alias": "jina-embed-v3",
        "name": "Jina Embeddings v3",
        "context_window": 8192,
        "max_output": 1024,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "E",
        "description": "Latest multilingual embeddings"
    },
    # Jina Embeddings v2
    {
        "id": "jina-embeddings-v2-base-en",
        "alias": "jina-embed-v2-en",
        "name": "Jina Embeddings v2 Base EN",
        "context_window": 8192,
        "max_output": 768,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "E",
        "description": "English embeddings, 768 dimensions"
    },
    {
        "id": "jina-embeddings-v2-base-de",
        "alias": "jina-embed-v2-de",
        "name": "Jina Embeddings v2 Base DE",
        "context_window": 8192,
        "max_output": 768,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "E",
        "description": "German embeddings"
    },
    {
        "id": "jina-embeddings-v2-base-es",
        "alias": "jina-embed-v2-es",
        "name": "Jina Embeddings v2 Base ES",
        "context_window": 8192,
        "max_output": 768,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "E",
        "description": "Spanish embeddings"
    },
    {
        "id": "jina-embeddings-v2-base-zh",
        "alias": "jina-embed-v2-zh",
        "name": "Jina Embeddings v2 Base ZH",
        "context_window": 8192,
        "max_output": 768,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "E",
        "description": "Chinese embeddings"
    },
    {
        "id": "jina-embeddings-v2-base-code",
        "alias": "jina-embed-v2-code",
        "name": "Jina Embeddings v2 Code",
        "context_window": 8192,
        "max_output": 768,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "E",
        "description": "Code embeddings"
    },
    {
        "id": "jina-embeddings-v2-small-en",
        "alias": "jina-embed-v2-small",
        "name": "Jina Embeddings v2 Small EN",
        "context_window": 8192,
        "max_output": 512,
        "input_price": 0.01,
        "output_price": "-",
        "capabilities": "E",
        "description": "Small English embeddings"
    },
    # Jina CLIP (Vision)
    {
        "id": "jina-clip-v2",
        "alias": "jina-clip-v2",
        "name": "Jina CLIP v2",
        "context_window": 8192,
        "max_output": 1024,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "VE",
        "description": "Multimodal image-text embeddings"
    },
    {
        "id": "jina-clip-v1",
        "alias": "jina-clip-v1",
        "name": "Jina CLIP v1",
        "context_window": 8192,
        "max_output": 768,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "VE",
        "description": "Image-text embeddings"
    },
    # ColBERT (Late Interaction)
    {
        "id": "jina-colbert-v2",
        "alias": "jina-colbert-v2",
        "name": "Jina ColBERT v2",
        "context_window": 8192,
        "max_output": 128,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "E",
        "description": "Late interaction retrieval"
    },
    {
        "id": "jina-colbert-v1-en",
        "alias": "jina-colbert-v1",
        "name": "Jina ColBERT v1 EN",
        "context_window": 8192,
        "max_output": 128,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "E",
        "description": "English ColBERT"
    },
    # Reranker
    {
        "id": "jina-reranker-v2-base-multilingual",
        "alias": "jina-rerank-v2",
        "name": "Jina Reranker v2 Multilingual",
        "context_window": 8192,
        "max_output": 0,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "R",
        "description": "Multilingual reranking"
    },
    {
        "id": "jina-reranker-v1-base-en",
        "alias": "jina-rerank-v1",
        "name": "Jina Reranker v1 EN",
        "context_window": 8192,
        "max_output": 0,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "R",
        "description": "English reranking"
    },
    {
        "id": "jina-reranker-v1-turbo-en",
        "alias": "jina-rerank-turbo",
        "name": "Jina Reranker v1 Turbo",
        "context_window": 8192,
        "max_output": 0,
        "input_price": 0.01,
        "output_price": "-",
        "capabilities": "R",
        "description": "Fast English reranking"
    },
    {
        "id": "jina-reranker-v1-tiny-en",
        "alias": "jina-rerank-tiny",
        "name": "Jina Reranker v1 Tiny",
        "context_window": 8192,
        "max_output": 0,
        "input_price": 0.005,
        "output_price": "-",
        "capabilities": "R",
        "description": "Tiny reranking model"
    },
    # Reader/Parser
    {
        "id": "jina-reader-v1",
        "alias": "jina-reader",
        "name": "Jina Reader v1",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0,
        "output_price": "-",
        "capabilities": "-",
        "description": "URL/PDF to markdown"
    },
]

def process_models() -> list:
    """Process Jina AI models into CSV format."""
    rows = []

    for model in JINA_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"jina/{model['id']}",
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
            'source': 'jina',
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
    print("Processing Jina AI models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/specialized/jina.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
