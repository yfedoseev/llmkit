#!/usr/bin/env python3
"""
Fetch Groq models.
Can use API with GROQ_API_KEY or fall back to hardcoded list.
"""

import os
import csv
import sys
import requests
from datetime import datetime

API_URL = "https://api.groq.com/openai/v1/models"

GROQ_MODELS = [
    # Llama 4 Series
    {
        "id": "llama-4-scout-17b-16e-instruct",
        "alias": "llama-4-scout",
        "name": "Llama 4 Scout 17B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.11,
        "output_price": 0.34,
        "capabilities": "VTJS",
        "description": "Meta's Llama 4 Scout 17B on Groq hardware"
    },
    {
        "id": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "alias": "llama-4-maverick",
        "name": "Llama 4 Maverick 17B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.20,
        "output_price": 0.60,
        "capabilities": "VTJS",
        "description": "Meta's Llama 4 Maverick 17B on Groq"
    },
    # Llama 3.3 Series
    {
        "id": "llama-3.3-70b-versatile",
        "alias": "llama-3.3-70b",
        "name": "Llama 3.3 70B",
        "context_window": 131072,
        "max_output": 32768,
        "input_price": 0.59,
        "output_price": 0.79,
        "capabilities": "TJS",
        "description": "Meta's Llama 3.3 70B, ultra-fast inference"
    },
    {
        "id": "llama-3.3-70b-specdec",
        "alias": "llama-3.3-70b-spec",
        "name": "Llama 3.3 70B SpecDec",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.59,
        "output_price": 0.99,
        "capabilities": "TJS",
        "description": "Llama 3.3 70B with speculative decoding"
    },
    # Llama 3.2 Series
    {
        "id": "llama-3.2-90b-vision-preview",
        "alias": "llama-3.2-90b-vision",
        "name": "Llama 3.2 90B Vision",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.90,
        "output_price": 0.90,
        "capabilities": "VTJ",
        "description": "Vision-enabled Llama 3.2 90B"
    },
    {
        "id": "llama-3.2-11b-vision-preview",
        "alias": "llama-3.2-11b-vision",
        "name": "Llama 3.2 11B Vision",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.18,
        "output_price": 0.18,
        "capabilities": "VTJ",
        "description": "Compact vision model"
    },
    {
        "id": "llama-3.2-3b-preview",
        "alias": "llama-3.2-3b",
        "name": "Llama 3.2 3B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.06,
        "output_price": 0.06,
        "capabilities": "TJ",
        "description": "Smallest Llama 3.2 model"
    },
    {
        "id": "llama-3.2-1b-preview",
        "alias": "llama-3.2-1b",
        "name": "Llama 3.2 1B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.04,
        "output_price": 0.04,
        "capabilities": "TJ",
        "description": "Tiny Llama model for edge devices"
    },
    # Llama 3.1 Series
    {
        "id": "llama-3.1-70b-versatile",
        "alias": "llama-3.1-70b",
        "name": "Llama 3.1 70B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.59,
        "output_price": 0.79,
        "capabilities": "TJS",
        "description": "Llama 3.1 70B on Groq hardware"
    },
    {
        "id": "llama-3.1-8b-instant",
        "alias": "llama-3.1-8b",
        "name": "Llama 3.1 8B",
        "context_window": 131072,
        "max_output": 8192,
        "input_price": 0.05,
        "output_price": 0.08,
        "capabilities": "TJS",
        "description": "Fast and efficient Llama 3.1 8B"
    },
    # Mixtral
    {
        "id": "mixtral-8x7b-32768",
        "alias": "mixtral-8x7b",
        "name": "Mixtral 8x7B",
        "context_window": 32768,
        "max_output": 8192,
        "input_price": 0.24,
        "output_price": 0.24,
        "capabilities": "TJ",
        "description": "Mistral's MoE model on Groq"
    },
    # Gemma
    {
        "id": "gemma2-9b-it",
        "alias": "gemma2-9b",
        "name": "Gemma 2 9B",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 0.20,
        "output_price": 0.20,
        "capabilities": "TJ",
        "description": "Google's Gemma 2 9B on Groq"
    },
    # DeepSeek
    {
        "id": "deepseek-r1-distill-llama-70b",
        "alias": "deepseek-r1-70b",
        "name": "DeepSeek R1 Distill 70B",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.75,
        "output_price": 0.99,
        "capabilities": "TJK",
        "description": "DeepSeek R1 distilled to Llama 70B"
    },
    {
        "id": "deepseek-r1-distill-qwen-32b",
        "alias": "deepseek-r1-32b",
        "name": "DeepSeek R1 Distill 32B",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.69,
        "output_price": 0.69,
        "capabilities": "TJK",
        "description": "DeepSeek R1 distilled to Qwen 32B"
    },
    # Qwen
    {
        "id": "qwen-qwq-32b",
        "alias": "qwq-32b",
        "name": "Qwen QWQ 32B",
        "context_window": 131072,
        "max_output": 16384,
        "input_price": 0.29,
        "output_price": 0.39,
        "capabilities": "TJK",
        "description": "Alibaba's Qwen QWQ reasoning model"
    },
    # Whisper
    {
        "id": "whisper-large-v3",
        "alias": "whisper-v3",
        "name": "Whisper Large v3",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.111,
        "output_price": "-",
        "capabilities": "A",
        "description": "Speech-to-text, $0.111/hour"
    },
    {
        "id": "whisper-large-v3-turbo",
        "alias": "whisper-v3-turbo",
        "name": "Whisper Large v3 Turbo",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.04,
        "output_price": "-",
        "capabilities": "A",
        "description": "Fast speech-to-text, $0.04/hour"
    },
    # Distil Whisper
    {
        "id": "distil-whisper-large-v3-en",
        "alias": "distil-whisper",
        "name": "Distil Whisper Large v3",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "A",
        "description": "Efficient English speech-to-text, $0.02/hour"
    },
]

def fetch_from_api():
    """Fetch models from Groq API."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        response = requests.get(
            API_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30
        )
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        print(f"API fetch failed: {e}", file=sys.stderr)
        return None

def process_models() -> list:
    """Process Groq models into CSV format."""
    rows = []

    for model in GROQ_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"groq/{model['id']}",
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
            'source': 'groq',
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
    print("Processing Groq models...")

    # Try API first
    api_models = fetch_from_api()
    if api_models:
        print(f"Fetched {len(api_models)} models from API")
    else:
        print("Using hardcoded model list")

    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/inference/groq.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
