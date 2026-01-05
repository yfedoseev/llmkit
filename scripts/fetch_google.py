#!/usr/bin/env python3
"""
Fetch Google Gemini models.
Uses hardcoded list from official documentation with accurate pricing.
"""

import csv
import sys
from datetime import datetime

GOOGLE_MODELS = [
    # Gemini 3 Series (Preview - Latest)
    {
        "id": "gemini-3-pro-preview",
        "alias": "gemini-3-pro",
        "name": "Gemini 3 Pro",
        "context_window": 1048576,
        "max_output": 65536,
        "input_price": 1.50,
        "output_price": 12.0,
        "cache_read_price": 0.375,
        "capabilities": "VTJSK",
        "description": "Latest reasoning-first model for complex agentic workflows and coding"
    },
    {
        "id": "gemini-3-flash-preview",
        "alias": "gemini-3-flash",
        "name": "Gemini 3 Flash",
        "context_window": 1048576,
        "max_output": 65536,
        "input_price": 0.20,
        "output_price": 0.80,
        "cache_read_price": 0.05,
        "capabilities": "VTJSK",
        "description": "Best for complex multimodal, agentic problems with strong reasoning"
    },
    # Gemini 2.5 Series
    {
        "id": "gemini-2.5-pro-preview-06-05",
        "alias": "gemini-2.5-pro",
        "name": "Gemini 2.5 Pro",
        "context_window": 1048576,
        "max_output": 65536,
        "input_price": 1.25,
        "output_price": 10.0,
        "cache_read_price": 0.31,
        "capabilities": "VTJSK",
        "description": "Latest flagship model with thinking, 1M context"
    },
    {
        "id": "gemini-2.5-flash-preview-05-20",
        "alias": "gemini-2.5-flash",
        "name": "Gemini 2.5 Flash",
        "context_window": 1048576,
        "max_output": 65536,
        "input_price": 0.15,
        "output_price": 0.60,
        "cache_read_price": 0.0375,
        "capabilities": "VTJSK",
        "description": "Fast thinking model with 1M context"
    },
    # Gemini 2.0 Series
    {
        "id": "gemini-2.0-flash",
        "alias": "gemini-2.0-flash",
        "name": "Gemini 2.0 Flash",
        "context_window": 1048576,
        "max_output": 8192,
        "input_price": 0.10,
        "output_price": 0.40,
        "cache_read_price": 0.025,
        "capabilities": "VTJS",
        "description": "Fast multimodal model with tool use"
    },
    {
        "id": "gemini-2.0-flash-lite",
        "alias": "gemini-2.0-flash-lite",
        "name": "Gemini 2.0 Flash Lite",
        "context_window": 1048576,
        "max_output": 8192,
        "input_price": 0.075,
        "output_price": 0.30,
        "cache_read_price": "-",
        "capabilities": "VTJ",
        "description": "Lightweight and cost-effective Flash variant"
    },
    # Gemini 1.5 Series
    {
        "id": "gemini-1.5-pro",
        "alias": "gemini-1.5-pro",
        "name": "Gemini 1.5 Pro",
        "context_window": 2097152,
        "max_output": 8192,
        "input_price": 1.25,
        "output_price": 5.0,
        "cache_read_price": 0.31,
        "capabilities": "VTJSC",
        "description": "2M context for complex reasoning and analysis"
    },
    {
        "id": "gemini-1.5-flash",
        "alias": "gemini-1.5-flash",
        "name": "Gemini 1.5 Flash",
        "context_window": 1048576,
        "max_output": 8192,
        "input_price": 0.075,
        "output_price": 0.30,
        "cache_read_price": 0.01875,
        "capabilities": "VTJSC",
        "description": "Fast and efficient with 1M context"
    },
    {
        "id": "gemini-1.5-flash-8b",
        "alias": "gemini-1.5-flash-8b",
        "name": "Gemini 1.5 Flash 8B",
        "context_window": 1048576,
        "max_output": 8192,
        "input_price": 0.0375,
        "output_price": 0.15,
        "cache_read_price": 0.01,
        "capabilities": "VTJS",
        "description": "Smallest Flash variant, highly efficient"
    },
    # Gemini 1.0 Pro
    {
        "id": "gemini-1.0-pro",
        "alias": "gemini-1.0-pro",
        "name": "Gemini 1.0 Pro",
        "context_window": 32760,
        "max_output": 8192,
        "input_price": 0.50,
        "output_price": 1.50,
        "cache_read_price": "-",
        "capabilities": "TJ",
        "description": "Original Gemini model, legacy support"
    },
    # Embeddings
    {
        "id": "text-embedding-004",
        "alias": "text-embedding-004",
        "name": "Text Embedding 004",
        "context_window": 2048,
        "max_output": 768,
        "input_price": 0.00001,
        "output_price": "-",
        "cache_read_price": "-",
        "capabilities": "E",
        "description": "Text embeddings, 768 dimensions"
    },
    {
        "id": "text-multilingual-embedding-002",
        "alias": "multilingual-embed",
        "name": "Multilingual Embedding 002",
        "context_window": 2048,
        "max_output": 768,
        "input_price": 0.00001,
        "output_price": "-",
        "cache_read_price": "-",
        "capabilities": "E",
        "description": "Multilingual text embeddings"
    },
    # Imagen
    {
        "id": "imagen-3.0-generate-002",
        "alias": "imagen-3",
        "name": "Imagen 3",
        "context_window": 480,
        "max_output": 0,
        "input_price": 0.04,
        "output_price": "-",
        "cache_read_price": "-",
        "capabilities": "I",
        "description": "Image generation, $0.04/image"
    },
    {
        "id": "imagen-3.0-fast-generate-001",
        "alias": "imagen-3-fast",
        "name": "Imagen 3 Fast",
        "context_window": 480,
        "max_output": 0,
        "input_price": 0.02,
        "output_price": "-",
        "cache_read_price": "-",
        "capabilities": "I",
        "description": "Fast image generation, $0.02/image"
    },
    # Veo
    {
        "id": "veo-2.0-generate-001",
        "alias": "veo-2",
        "name": "Veo 2",
        "context_window": 480,
        "max_output": 0,
        "input_price": 0.35,
        "output_price": "-",
        "cache_read_price": "-",
        "capabilities": "D",
        "description": "Video generation, $0.35/second"
    },
]

def process_models() -> list:
    """Process Google models into CSV format."""
    rows = []

    for model in GOOGLE_MODELS:
        cache_price = model.get('cache_read_price', '-')
        cache_str = f"{cache_price:.6f}" if isinstance(cache_price, (int, float)) else cache_price
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"google/{model['id']}",
            'alias': model['alias'],
            'name': model['name'],
            'status': 'C',
            'input_price': f"{model['input_price']:.6f}",
            'output_price': output_str,
            'cache_input_price': cache_str,
            'context_window': model['context_window'],
            'max_output': model['max_output'],
            'capabilities': model['capabilities'],
            'quality': 'official',
            'source': 'google',
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
    print("Processing Google Gemini models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/core/google.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
