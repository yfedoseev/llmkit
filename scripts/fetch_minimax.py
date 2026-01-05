#!/usr/bin/env python3
"""
Fetch MiniMax models.
MiniMax is a Chinese AI company known for ABAB models with 4M context.
"""

import csv
import sys
from datetime import datetime

MINIMAX_MODELS = [
    # ABAB 7 Series
    {
        "id": "abab7-chat-preview",
        "alias": "abab7",
        "name": "ABAB 7",
        "context_window": 4000000,
        "max_output": 16384,
        "input_price": 0.70,
        "output_price": 0.70,
        "capabilities": "VTJS",
        "description": "Flagship 4M context model for complex tasks"
    },
    # ABAB 6.5 Series
    {
        "id": "abab6.5s-chat",
        "alias": "abab6.5s",
        "name": "ABAB 6.5s",
        "context_window": 245760,
        "max_output": 16384,
        "input_price": 0.14,
        "output_price": 0.14,
        "capabilities": "VTJS",
        "description": "Fast 245K context model"
    },
    {
        "id": "abab6.5t-chat",
        "alias": "abab6.5t",
        "name": "ABAB 6.5t",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.07,
        "output_price": 0.07,
        "capabilities": "TJS",
        "description": "Efficient text model"
    },
    {
        "id": "abab6.5g-chat",
        "alias": "abab6.5g",
        "name": "ABAB 6.5g",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.14,
        "output_price": 0.14,
        "capabilities": "TJS",
        "description": "General purpose model"
    },
    # ABAB 6 Series
    {
        "id": "abab6-chat",
        "alias": "abab6",
        "name": "ABAB 6",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.21,
        "output_price": 0.21,
        "capabilities": "VTJ",
        "description": "Previous generation ABAB"
    },
    # ABAB 5.5 Series
    {
        "id": "abab5.5s-chat",
        "alias": "abab5.5s",
        "name": "ABAB 5.5s",
        "context_window": 16384,
        "max_output": 8192,
        "input_price": 0.07,
        "output_price": 0.07,
        "capabilities": "TJ",
        "description": "Legacy efficient model"
    },
    {
        "id": "abab5.5-chat",
        "alias": "abab5.5",
        "name": "ABAB 5.5",
        "context_window": 16384,
        "max_output": 8192,
        "input_price": 0.21,
        "output_price": 0.21,
        "capabilities": "VTJ",
        "description": "Legacy vision model"
    },
    # Speech Models
    {
        "id": "speech-01-turbo",
        "alias": "speech-turbo",
        "name": "Speech 01 Turbo",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.007,
        "output_price": "-",
        "capabilities": "A",
        "description": "Fast TTS model"
    },
    {
        "id": "speech-01-hd",
        "alias": "speech-hd",
        "name": "Speech 01 HD",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.10,
        "output_price": "-",
        "capabilities": "A",
        "description": "High-quality TTS"
    },
    {
        "id": "speech-02-turbo",
        "alias": "speech-02",
        "name": "Speech 02 Turbo",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.007,
        "output_price": "-",
        "capabilities": "A",
        "description": "Next-gen TTS"
    },
    # Video Model
    {
        "id": "video-01",
        "alias": "video-01",
        "name": "Video 01",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.30,
        "output_price": "-",
        "capabilities": "D",
        "description": "Video generation"
    },
    # Music Model
    {
        "id": "music-01",
        "alias": "music-01",
        "name": "Music 01",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.01,
        "output_price": "-",
        "capabilities": "A",
        "description": "Music generation"
    },
    # Embeddings
    {
        "id": "embo-01",
        "alias": "embo-01",
        "name": "Embo 01",
        "context_window": 4096,
        "max_output": 1536,
        "input_price": 0.0001,
        "output_price": "-",
        "capabilities": "E",
        "description": "Text embeddings"
    },
]

def process_models() -> list:
    """Process MiniMax models into CSV format."""
    rows = []

    for model in MINIMAX_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"minimax/{model['id']}",
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
            'source': 'minimax',
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
    print("Processing MiniMax models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/chinese/minimax.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
