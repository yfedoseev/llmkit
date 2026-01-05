#!/usr/bin/env python3
"""
Fetch OpenAI models.
Uses hardcoded list from official documentation since models API doesn't include pricing.
"""

import csv
import sys
from datetime import datetime

OPENAI_MODELS = [
    # GPT-5 Series (Latest - 2025)
    {
        "id": "gpt-5",
        "alias": "gpt-5",
        "name": "GPT-5",
        "context_window": 1000000,
        "max_output": 100000,
        "input_price": 1.25,
        "output_price": 10.0,
        "cache_read_price": 0.30,
        "capabilities": "VTJSKC",
        "description": "Most capable OpenAI model, unified reasoning and language"
    },
    {
        "id": "gpt-5-mini",
        "alias": "gpt-5-mini",
        "name": "GPT-5 Mini",
        "context_window": 1000000,
        "max_output": 100000,
        "input_price": 0.30,
        "output_price": 1.25,
        "cache_read_price": 0.075,
        "capabilities": "VTJSKC",
        "description": "Efficient GPT-5 for high-volume tasks"
    },
    # GPT-4.1 Series (April 2025)
    {
        "id": "gpt-4.1",
        "alias": "gpt-4.1",
        "name": "GPT-4.1",
        "context_window": 1047576,
        "max_output": 32768,
        "input_price": 2.0,
        "output_price": 8.0,
        "cache_read_price": 0.5,
        "capabilities": "VTJSC",
        "description": "Most capable GPT-4 with 1M context, flagship model"
    },
    {
        "id": "gpt-4.1-mini",
        "alias": "gpt-4.1-mini",
        "name": "GPT-4.1 Mini",
        "context_window": 1047576,
        "max_output": 32768,
        "input_price": 0.40,
        "output_price": 1.60,
        "cache_read_price": 0.10,
        "capabilities": "VTJSC",
        "description": "Affordable 1M context model for high volume tasks"
    },
    {
        "id": "gpt-4.1-nano",
        "alias": "gpt-4.1-nano",
        "name": "GPT-4.1 Nano",
        "context_window": 1047576,
        "max_output": 32768,
        "input_price": 0.10,
        "output_price": 0.40,
        "cache_read_price": 0.025,
        "capabilities": "VTJS",
        "description": "Fastest and cheapest GPT-4.1 variant"
    },
    # o-Series (Reasoning Models)
    {
        "id": "o3",
        "alias": "o3",
        "name": "o3",
        "context_window": 200000,
        "max_output": 100000,
        "input_price": 10.0,
        "output_price": 40.0,
        "cache_read_price": 2.5,
        "capabilities": "VTJSK",
        "description": "Most powerful reasoning model for complex problems"
    },
    {
        "id": "o3-pro",
        "alias": "o3-pro",
        "name": "o3 Pro",
        "context_window": 200000,
        "max_output": 100000,
        "input_price": 20.0,
        "output_price": 80.0,
        "cache_read_price": 5.0,
        "capabilities": "VTJSK",
        "description": "Extended compute for hard reasoning problems"
    },
    {
        "id": "o3-mini",
        "alias": "o3-mini",
        "name": "o3 Mini",
        "context_window": 200000,
        "max_output": 100000,
        "input_price": 1.10,
        "output_price": 4.40,
        "cache_read_price": 0.275,
        "capabilities": "VTJSK",
        "description": "Fast and affordable reasoning model"
    },
    {
        "id": "o4-mini",
        "alias": "o4-mini",
        "name": "o4 Mini",
        "context_window": 200000,
        "max_output": 100000,
        "input_price": 1.10,
        "output_price": 4.40,
        "cache_read_price": 0.275,
        "capabilities": "VTJSK",
        "description": "Latest reasoning model, balanced speed and capability"
    },
    {
        "id": "o1",
        "alias": "o1",
        "name": "o1",
        "context_window": 200000,
        "max_output": 100000,
        "input_price": 15.0,
        "output_price": 60.0,
        "cache_read_price": 3.75,
        "capabilities": "VTJSK",
        "description": "Deep reasoning for math, science, and coding"
    },
    {
        "id": "o1-pro",
        "alias": "o1-pro",
        "name": "o1 Pro",
        "context_window": 200000,
        "max_output": 100000,
        "input_price": 150.0,
        "output_price": 600.0,
        "cache_read_price": 37.5,
        "capabilities": "VTJSK",
        "description": "Extended compute for hardest problems"
    },
    {
        "id": "o1-mini",
        "alias": "o1-mini",
        "name": "o1 Mini",
        "context_window": 128000,
        "max_output": 65536,
        "input_price": 1.10,
        "output_price": 4.40,
        "cache_read_price": 0.55,
        "capabilities": "VTJSK",
        "description": "Fast reasoning for STEM tasks"
    },
    # GPT-4o Series
    {
        "id": "gpt-4o",
        "alias": "gpt-4o",
        "name": "GPT-4o",
        "context_window": 128000,
        "max_output": 16384,
        "input_price": 2.50,
        "output_price": 10.0,
        "cache_read_price": 1.25,
        "capabilities": "VTJSC",
        "description": "Flagship multimodal model for text, vision, and audio"
    },
    {
        "id": "gpt-4o-mini",
        "alias": "gpt-4o-mini",
        "name": "GPT-4o Mini",
        "context_window": 128000,
        "max_output": 16384,
        "input_price": 0.15,
        "output_price": 0.60,
        "cache_read_price": 0.075,
        "capabilities": "VTJSC",
        "description": "Affordable multimodal model for lightweight tasks"
    },
    {
        "id": "gpt-4o-audio-preview",
        "alias": "gpt-4o-audio",
        "name": "GPT-4o Audio Preview",
        "context_window": 128000,
        "max_output": 16384,
        "input_price": 2.50,
        "output_price": 10.0,
        "cache_read_price": "-",
        "capabilities": "VTJS",
        "description": "GPT-4o with native audio understanding"
    },
    # GPT-4 Turbo
    {
        "id": "gpt-4-turbo",
        "alias": "gpt-4-turbo",
        "name": "GPT-4 Turbo",
        "context_window": 128000,
        "max_output": 4096,
        "input_price": 10.0,
        "output_price": 30.0,
        "cache_read_price": "-",
        "capabilities": "VTJS",
        "description": "Previous generation GPT-4 with vision"
    },
    # GPT-4
    {
        "id": "gpt-4",
        "alias": "gpt-4",
        "name": "GPT-4",
        "context_window": 8192,
        "max_output": 8192,
        "input_price": 30.0,
        "output_price": 60.0,
        "cache_read_price": "-",
        "capabilities": "TJS",
        "description": "Original GPT-4 model"
    },
    # GPT-3.5 Turbo
    {
        "id": "gpt-3.5-turbo",
        "alias": "gpt-3.5-turbo",
        "name": "GPT-3.5 Turbo",
        "context_window": 16385,
        "max_output": 4096,
        "input_price": 0.50,
        "output_price": 1.50,
        "cache_read_price": "-",
        "capabilities": "TJ",
        "description": "Legacy model, fast and affordable"
    },
    # Embeddings
    {
        "id": "text-embedding-3-large",
        "alias": "embed-3-large",
        "name": "Text Embedding 3 Large",
        "context_window": 8191,
        "max_output": 3072,
        "input_price": 0.13,
        "output_price": "-",
        "cache_read_price": "-",
        "capabilities": "E",
        "description": "Most capable embedding model, 3072 dimensions"
    },
    {
        "id": "text-embedding-3-small",
        "alias": "embed-3-small",
        "name": "Text Embedding 3 Small",
        "context_window": 8191,
        "max_output": 1536,
        "input_price": 0.02,
        "output_price": "-",
        "cache_read_price": "-",
        "capabilities": "E",
        "description": "Efficient embedding model, 1536 dimensions"
    },
    # Audio Models
    {
        "id": "whisper-1",
        "alias": "whisper-1",
        "name": "Whisper",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.006,
        "output_price": "-",
        "cache_read_price": "-",
        "capabilities": "A",
        "description": "Speech-to-text model, $0.006/minute"
    },
    {
        "id": "tts-1",
        "alias": "tts-1",
        "name": "TTS-1",
        "context_window": 4096,
        "max_output": 0,
        "input_price": 15.0,
        "output_price": "-",
        "cache_read_price": "-",
        "capabilities": "A",
        "description": "Text-to-speech, $15/1M characters"
    },
    {
        "id": "tts-1-hd",
        "alias": "tts-1-hd",
        "name": "TTS-1 HD",
        "context_window": 4096,
        "max_output": 0,
        "input_price": 30.0,
        "output_price": "-",
        "cache_read_price": "-",
        "capabilities": "A",
        "description": "High-quality text-to-speech, $30/1M characters"
    },
    # Image Models
    {
        "id": "dall-e-3",
        "alias": "dall-e-3",
        "name": "DALL-E 3",
        "context_window": 4000,
        "max_output": 0,
        "input_price": 0.04,
        "output_price": "-",
        "cache_read_price": "-",
        "capabilities": "I",
        "description": "Image generation, $0.04-0.12/image"
    },
    {
        "id": "gpt-image-1",
        "alias": "gpt-image-1",
        "name": "GPT Image 1",
        "context_window": 32000,
        "max_output": 0,
        "input_price": 5.0,
        "output_price": "-",
        "cache_read_price": "-",
        "capabilities": "I",
        "description": "Advanced image generation with text input"
    },
]

def process_models() -> list:
    """Process OpenAI models into CSV format."""
    rows = []

    for model in OPENAI_MODELS:
        cache_price = model.get('cache_read_price', '-')
        cache_str = f"{cache_price:.6f}" if isinstance(cache_price, (int, float)) else cache_price
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"openai/{model['id']}",
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
            'source': 'openai',
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
    print("Processing OpenAI models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/core/openai.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
