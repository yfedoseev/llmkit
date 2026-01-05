#!/usr/bin/env python3
"""
Fetch Claude models from Anthropic.
Uses hardcoded list from official documentation since no public models API.
"""

import csv
import sys
from datetime import datetime

ANTHROPIC_MODELS = [
    # Claude 4.5 Sonnet (Latest - November 2025)
    {
        "id": "claude-sonnet-4-5-20250929",
        "alias": "claude-4.5-sonnet",
        "name": "Claude 4.5 Sonnet",
        "context_window": 200000,
        "max_output": 16384,
        "input_price": 3.0,
        "output_price": 15.0,
        "cache_read_price": 0.3,
        "cache_write_price": 3.75,
        "capabilities": "VTJSKC",
        "description": "Best for complex coding and analysis, supports 1M context with beta header"
    },
    # Claude 4.1 Opus (August 2025)
    {
        "id": "claude-opus-4-1-20250805",
        "alias": "claude-4.1-opus",
        "name": "Claude 4.1 Opus",
        "context_window": 200000,
        "max_output": 32000,
        "input_price": 15.0,
        "output_price": 75.0,
        "cache_read_price": 1.5,
        "cache_write_price": 18.75,
        "capabilities": "VTJSKC",
        "description": "Most powerful Claude for agentic tasks, coding, and reasoning"
    },
    # Claude 4 Opus (May 2025)
    {
        "id": "claude-opus-4-20250514",
        "alias": "claude-4-opus",
        "name": "Claude 4 Opus",
        "context_window": 200000,
        "max_output": 32000,
        "input_price": 15.0,
        "output_price": 75.0,
        "cache_read_price": 1.5,
        "cache_write_price": 18.75,
        "capabilities": "VTJSKC",
        "description": "Claude 4 flagship for complex tasks, research, and analysis"
    },
    # Claude 4 Sonnet (May 2025)
    {
        "id": "claude-sonnet-4-20250514",
        "alias": "claude-4-sonnet",
        "name": "Claude 4 Sonnet",
        "context_window": 200000,
        "max_output": 64000,
        "input_price": 3.0,
        "output_price": 15.0,
        "cache_read_price": 0.3,
        "cache_write_price": 3.75,
        "capabilities": "VTJSKC",
        "description": "Balanced intelligence and speed for everyday tasks"
    },
    # Claude 4.5 Haiku (October 2025)
    {
        "id": "claude-haiku-4-5-20251015",
        "alias": "claude-4.5-haiku",
        "name": "Claude 4.5 Haiku",
        "context_window": 200000,
        "max_output": 8192,
        "input_price": 1.0,
        "output_price": 5.0,
        "cache_read_price": 0.1,
        "cache_write_price": 1.25,
        "capabilities": "VTJSC",
        "description": "Fast model optimized for low latency and cost"
    },
    # Claude 3.7 Sonnet (Extended thinking)
    {
        "id": "claude-3-7-sonnet-20250219",
        "alias": "claude-3.7-sonnet",
        "name": "Claude 3.7 Sonnet",
        "context_window": 200000,
        "max_output": 128000,
        "input_price": 3.0,
        "output_price": 15.0,
        "cache_read_price": 0.3,
        "cache_write_price": 3.75,
        "capabilities": "VTJSKC",
        "description": "Extended thinking with 128K output for complex reasoning"
    },
    # Claude 3.5 Sonnet (Latest)
    {
        "id": "claude-3-5-sonnet-20241022",
        "alias": "claude-3.5-sonnet",
        "name": "Claude 3.5 Sonnet",
        "context_window": 200000,
        "max_output": 8192,
        "input_price": 3.0,
        "output_price": 15.0,
        "cache_read_price": 0.3,
        "cache_write_price": 3.75,
        "capabilities": "VTJSC",
        "description": "Best for coding, analysis, and complex reasoning tasks"
    },
    # Claude 3.5 Haiku
    {
        "id": "claude-3-5-haiku-20241022",
        "alias": "claude-3.5-haiku",
        "name": "Claude 3.5 Haiku",
        "context_window": 200000,
        "max_output": 8192,
        "input_price": 0.80,
        "output_price": 4.0,
        "cache_read_price": 0.08,
        "cache_write_price": 1.0,
        "capabilities": "VTJSC",
        "description": "Fast and affordable for high-volume tasks"
    },
    # Claude 3 Opus
    {
        "id": "claude-3-opus-20240229",
        "alias": "claude-3-opus",
        "name": "Claude 3 Opus",
        "context_window": 200000,
        "max_output": 4096,
        "input_price": 15.0,
        "output_price": 75.0,
        "cache_read_price": 1.5,
        "cache_write_price": 18.75,
        "capabilities": "VTJC",
        "description": "Powerful model for complex tasks requiring deep understanding"
    },
    # Claude 3 Sonnet
    {
        "id": "claude-3-sonnet-20240229",
        "alias": "claude-3-sonnet",
        "name": "Claude 3 Sonnet",
        "context_window": 200000,
        "max_output": 4096,
        "input_price": 3.0,
        "output_price": 15.0,
        "cache_read_price": 0.3,
        "cache_write_price": 3.75,
        "capabilities": "VTJ",
        "description": "Balanced performance for wide range of tasks"
    },
    # Claude 3 Haiku
    {
        "id": "claude-3-haiku-20240307",
        "alias": "claude-3-haiku",
        "name": "Claude 3 Haiku",
        "context_window": 200000,
        "max_output": 4096,
        "input_price": 0.25,
        "output_price": 1.25,
        "cache_read_price": 0.03,
        "cache_write_price": 0.30,
        "capabilities": "VTJ",
        "description": "Fastest and most compact Claude 3 model"
    },
]

def process_models() -> list:
    """Process Anthropic models into CSV format."""
    rows = []

    for model in ANTHROPIC_MODELS:
        row = {
            'id': f"anthropic/{model['id']}",
            'alias': model['alias'],
            'name': model['name'],
            'status': 'C',
            'input_price': f"{model['input_price']:.6f}",
            'output_price': f"{model['output_price']:.6f}",
            'cache_input_price': f"{model.get('cache_read_price', '-')}",
            'context_window': model['context_window'],
            'max_output': model['max_output'],
            'capabilities': model['capabilities'],
            'quality': 'official',
            'source': 'anthropic',
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
    print("Processing Anthropic Claude models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/core/anthropic.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
