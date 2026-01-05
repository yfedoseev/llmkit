#!/usr/bin/env python3
"""
Fetch iFlyTek Spark models.
iFlyTek is a Chinese AI company specializing in speech recognition.
"""

import csv
import sys
from datetime import datetime

SPARK_MODELS = [
    # Spark 4.0 Series
    {
        "id": "spark-4.0-ultra",
        "alias": "spark-4-ultra",
        "name": "Spark 4.0 Ultra",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.69,
        "output_price": 0.69,
        "capabilities": "VTJS",
        "description": "Flagship 128K context model"
    },
    {
        "id": "spark-4.0-max-32k",
        "alias": "spark-4-max",
        "name": "Spark 4.0 Max 32K",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.42,
        "output_price": 0.42,
        "capabilities": "VTJS",
        "description": "32K context model"
    },
    # Spark 3.5 Series
    {
        "id": "spark-3.5-max",
        "alias": "spark-3.5-max",
        "name": "Spark 3.5 Max",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.42,
        "output_price": 0.42,
        "capabilities": "TJS",
        "description": "Previous gen 128K model"
    },
    {
        "id": "spark-3.5-pro",
        "alias": "spark-3.5-pro",
        "name": "Spark 3.5 Pro",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.14,
        "output_price": 0.14,
        "capabilities": "TJS",
        "description": "Balanced 32K model"
    },
    # Spark Lite
    {
        "id": "spark-lite",
        "alias": "spark-lite",
        "name": "Spark Lite",
        "context_window": 8192,
        "max_output": 4096,
        "input_price": 0.0,
        "output_price": 0.0,
        "capabilities": "TJ",
        "description": "Free lightweight model"
    },
    # Spark Vision
    {
        "id": "spark-vision-4.0",
        "alias": "spark-vision-4",
        "name": "Spark Vision 4.0",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.42,
        "output_price": 0.42,
        "capabilities": "VTJ",
        "description": "Vision-language model"
    },
    # Spark Code
    {
        "id": "spark-code",
        "alias": "spark-code",
        "name": "Spark Code",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.21,
        "output_price": 0.21,
        "capabilities": "TJS",
        "description": "Code generation model"
    },
    # Speech Models
    {
        "id": "spark-asr-v2",
        "alias": "spark-asr",
        "name": "Spark ASR v2",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.03,
        "output_price": "-",
        "capabilities": "A",
        "description": "Speech recognition"
    },
    {
        "id": "spark-tts-v2",
        "alias": "spark-tts",
        "name": "Spark TTS v2",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.01,
        "output_price": "-",
        "capabilities": "A",
        "description": "Text-to-speech"
    },
    # Embeddings
    {
        "id": "spark-embedding-v1",
        "alias": "spark-embed",
        "name": "Spark Embedding v1",
        "context_window": 512,
        "max_output": 768,
        "input_price": 0.0003,
        "output_price": "-",
        "capabilities": "E",
        "description": "Text embeddings"
    },
]

def process_models() -> list:
    """Process iFlyTek Spark models into CSV format."""
    rows = []

    for model in SPARK_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"spark/{model['id']}",
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
            'source': 'spark',
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
    print("Processing iFlyTek Spark models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/chinese/spark.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
