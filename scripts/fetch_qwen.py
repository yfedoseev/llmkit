#!/usr/bin/env python3
"""
Fetch Qwen/DashScope models.
Alibaba Cloud's Qwen series with up to 262K context.
"""

import csv
import sys
from datetime import datetime

QWEN_MODELS = [
    # Qwen3 Series (Latest)
    {
        "id": "qwen3-235b-a22b",
        "alias": "qwen3-235b",
        "name": "Qwen3 235B A22B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.55,
        "output_price": 2.19,
        "capabilities": "VTJSK",
        "description": "235B MoE with thinking capabilities"
    },
    {
        "id": "qwen3-32b",
        "alias": "qwen3-32b",
        "name": "Qwen3 32B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.28,
        "output_price": 0.82,
        "capabilities": "VTJSK",
        "description": "32B dense model with thinking"
    },
    # Qwen2.5 Series
    {
        "id": "qwen2.5-72b-instruct",
        "alias": "qwen2.5-72b",
        "name": "Qwen 2.5 72B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.55,
        "output_price": 1.65,
        "capabilities": "VTJS",
        "description": "Flagship 72B instruct model"
    },
    {
        "id": "qwen2.5-32b-instruct",
        "alias": "qwen2.5-32b",
        "name": "Qwen 2.5 32B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.28,
        "output_price": 0.55,
        "capabilities": "VTJS",
        "description": "Balanced 32B model"
    },
    {
        "id": "qwen2.5-14b-instruct",
        "alias": "qwen2.5-14b",
        "name": "Qwen 2.5 14B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.055,
        "output_price": 0.11,
        "capabilities": "TJS",
        "description": "Efficient 14B model"
    },
    {
        "id": "qwen2.5-7b-instruct",
        "alias": "qwen2.5-7b",
        "name": "Qwen 2.5 7B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.014,
        "output_price": 0.041,
        "capabilities": "TJS",
        "description": "Compact 7B model"
    },
    {
        "id": "qwen2.5-3b-instruct",
        "alias": "qwen2.5-3b",
        "name": "Qwen 2.5 3B",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.0042,
        "output_price": 0.0083,
        "capabilities": "TJ",
        "description": "Tiny 3B model"
    },
    # Qwen2.5 Coder
    {
        "id": "qwen2.5-coder-32b-instruct",
        "alias": "qwen-coder-32b",
        "name": "Qwen 2.5 Coder 32B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.28,
        "output_price": 0.55,
        "capabilities": "TJS",
        "description": "Code-specialized 32B model"
    },
    {
        "id": "qwen2.5-coder-14b-instruct",
        "alias": "qwen-coder-14b",
        "name": "Qwen 2.5 Coder 14B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.055,
        "output_price": 0.11,
        "capabilities": "TJS",
        "description": "Code-specialized 14B model"
    },
    {
        "id": "qwen2.5-coder-7b-instruct",
        "alias": "qwen-coder-7b",
        "name": "Qwen 2.5 Coder 7B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.014,
        "output_price": 0.041,
        "capabilities": "TJS",
        "description": "Code-specialized 7B model"
    },
    # Qwen2.5 Math
    {
        "id": "qwen2.5-math-72b-instruct",
        "alias": "qwen-math-72b",
        "name": "Qwen 2.5 Math 72B",
        "context_window": 4096,
        "max_output": 4096,
        "input_price": 0.55,
        "output_price": 1.65,
        "capabilities": "TJS",
        "description": "Math-specialized 72B model"
    },
    # Qwen-VL (Vision)
    {
        "id": "qwen-vl-max-0809",
        "alias": "qwen-vl-max",
        "name": "Qwen VL Max",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.28,
        "output_price": 0.82,
        "capabilities": "VTJS",
        "description": "Best vision-language model"
    },
    {
        "id": "qwen-vl-plus-0809",
        "alias": "qwen-vl-plus",
        "name": "Qwen VL Plus",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.11,
        "output_price": 0.33,
        "capabilities": "VTJS",
        "description": "Enhanced vision-language model"
    },
    {
        "id": "qwen2-vl-72b-instruct",
        "alias": "qwen2-vl-72b",
        "name": "Qwen2 VL 72B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.55,
        "output_price": 1.65,
        "capabilities": "VTJS",
        "description": "Large vision-language model"
    },
    {
        "id": "qwen2-vl-7b-instruct",
        "alias": "qwen2-vl-7b",
        "name": "Qwen2 VL 7B",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.014,
        "output_price": 0.041,
        "capabilities": "VTJS",
        "description": "Compact vision-language model"
    },
    # Qwen Long
    {
        "id": "qwen-long",
        "alias": "qwen-long",
        "name": "Qwen Long",
        "context_window": 10000000,
        "max_output": 8192,
        "input_price": 0.0007,
        "output_price": 0.0028,
        "capabilities": "TJS",
        "description": "10M context for ultra-long documents"
    },
    # QwQ (Reasoning)
    {
        "id": "qwq-32b-preview",
        "alias": "qwq-32b",
        "name": "QwQ 32B",
        "context_window": 32768,
        "max_output": 32768,
        "input_price": 0.28,
        "output_price": 0.82,
        "capabilities": "TJK",
        "description": "Chain-of-thought reasoning model"
    },
    # Qwen Audio
    {
        "id": "qwen2-audio-instruct",
        "alias": "qwen2-audio",
        "name": "Qwen2 Audio",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.028,
        "output_price": 0.082,
        "capabilities": "ATJ",
        "description": "Audio understanding model"
    },
    # Embeddings
    {
        "id": "text-embedding-v3",
        "alias": "qwen-embed-v3",
        "name": "Text Embedding V3",
        "context_window": 8192,
        "max_output": 1024,
        "input_price": 0.0001,
        "output_price": "-",
        "capabilities": "E",
        "description": "Text embeddings"
    },
]

def process_models() -> list:
    """Process Qwen models into CSV format."""
    rows = []

    for model in QWEN_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"qwen/{model['id']}",
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
            'source': 'qwen',
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
    print("Processing Qwen models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/chinese/qwen.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
