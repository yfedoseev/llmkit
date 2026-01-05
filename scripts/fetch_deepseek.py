#!/usr/bin/env python3
"""
Fetch DeepSeek models.
Uses hardcoded list from official documentation.
"""

import csv
import sys
from datetime import datetime

DEEPSEEK_MODELS = [
    # DeepSeek V3
    {
        "id": "deepseek-chat",
        "alias": "deepseek-v3",
        "name": "DeepSeek V3",
        "context_window": 65536,
        "max_output": 8192,
        "input_price": 0.27,
        "output_price": 1.10,
        "cache_read_price": 0.07,
        "capabilities": "VTJSC",
        "description": "671B MoE model, best performance at lowest cost"
    },
    {
        "id": "deepseek-chat-v3-0324",
        "alias": "deepseek-v3-0324",
        "name": "DeepSeek V3 0324",
        "context_window": 65536,
        "max_output": 8192,
        "input_price": 0.27,
        "output_price": 1.10,
        "cache_read_price": 0.07,
        "capabilities": "VTJSC",
        "description": "DeepSeek V3 March 2024 version"
    },
    # DeepSeek R1 (Reasoning)
    {
        "id": "deepseek-reasoner",
        "alias": "deepseek-r1",
        "name": "DeepSeek R1",
        "context_window": 65536,
        "max_output": 8192,
        "input_price": 0.55,
        "output_price": 2.19,
        "cache_read_price": 0.14,
        "capabilities": "VTJSK",
        "description": "Reasoning model with chain-of-thought"
    },
    {
        "id": "deepseek-reasoner-0528",
        "alias": "deepseek-r1-0528",
        "name": "DeepSeek R1 0528",
        "context_window": 65536,
        "max_output": 8192,
        "input_price": 0.55,
        "output_price": 2.19,
        "cache_read_price": 0.14,
        "capabilities": "VTJSK",
        "description": "DeepSeek R1 May 2028 version"
    },
    # DeepSeek Coder
    {
        "id": "deepseek-coder",
        "alias": "deepseek-coder",
        "name": "DeepSeek Coder",
        "context_window": 65536,
        "max_output": 8192,
        "input_price": 0.14,
        "output_price": 0.28,
        "cache_read_price": 0.035,
        "capabilities": "TJS",
        "description": "Specialized code generation model"
    },
    # DeepSeek V2 (Legacy)
    {
        "id": "deepseek-chat-v2",
        "alias": "deepseek-v2",
        "name": "DeepSeek V2",
        "context_window": 32768,
        "max_output": 4096,
        "input_price": 0.14,
        "output_price": 0.28,
        "cache_read_price": "-",
        "capabilities": "TJS",
        "description": "Previous generation model"
    },
    # DeepSeek V2.5 (Combined)
    {
        "id": "deepseek-chat-v2.5",
        "alias": "deepseek-v2.5",
        "name": "DeepSeek V2.5",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.14,
        "output_price": 0.28,
        "cache_read_price": "-",
        "capabilities": "TJS",
        "description": "Combined chat and code capabilities"
    },
]

def process_models() -> list:
    """Process DeepSeek models into CSV format."""
    rows = []

    for model in DEEPSEEK_MODELS:
        cache_price = model.get('cache_read_price', '-')
        cache_str = f"{cache_price:.6f}" if isinstance(cache_price, (int, float)) else cache_price

        row = {
            'id': f"deepseek/{model['id']}",
            'alias': model['alias'],
            'name': model['name'],
            'status': 'C',
            'input_price': f"{model['input_price']:.6f}",
            'output_price': f"{model['output_price']:.6f}",
            'cache_input_price': cache_str,
            'context_window': model['context_window'],
            'max_output': model['max_output'],
            'capabilities': model['capabilities'],
            'quality': 'official',
            'source': 'deepseek',
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
    print("Processing DeepSeek models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/core/deepseek.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
