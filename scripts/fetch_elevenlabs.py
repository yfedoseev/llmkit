#!/usr/bin/env python3
"""
Fetch ElevenLabs models.
ElevenLabs provides text-to-speech and voice AI models.
"""

import csv
import sys
from datetime import datetime

ELEVENLABS_MODELS = [
    # Multilingual v2
    {
        "id": "eleven_multilingual_v2",
        "alias": "multilingual-v2",
        "name": "Multilingual v2",
        "context_window": 5000,
        "max_output": 0,
        "input_price": 0.18,
        "output_price": "-",
        "capabilities": "A",
        "description": "29 languages, most natural, $0.18/1K chars"
    },
    # Turbo v2.5
    {
        "id": "eleven_turbo_v2_5",
        "alias": "turbo-v2.5",
        "name": "Turbo v2.5",
        "context_window": 5000,
        "max_output": 0,
        "input_price": 0.18,
        "output_price": "-",
        "capabilities": "A",
        "description": "32 languages, low latency, $0.18/1K chars"
    },
    # Turbo v2
    {
        "id": "eleven_turbo_v2",
        "alias": "turbo-v2",
        "name": "Turbo v2",
        "context_window": 5000,
        "max_output": 0,
        "input_price": 0.18,
        "output_price": "-",
        "capabilities": "A",
        "description": "English only, fastest, $0.18/1K chars"
    },
    # English v1
    {
        "id": "eleven_monolingual_v1",
        "alias": "english-v1",
        "name": "English v1",
        "context_window": 5000,
        "max_output": 0,
        "input_price": 0.18,
        "output_price": "-",
        "capabilities": "A",
        "description": "English only, legacy model, $0.18/1K chars"
    },
    # Flash v2.5
    {
        "id": "eleven_flash_v2_5",
        "alias": "flash-v2.5",
        "name": "Flash v2.5",
        "context_window": 5000,
        "max_output": 0,
        "input_price": 0.09,
        "output_price": "-",
        "capabilities": "A",
        "description": "Ultra-low latency, $0.09/1K chars"
    },
    # Flash v2
    {
        "id": "eleven_flash_v2",
        "alias": "flash-v2",
        "name": "Flash v2",
        "context_window": 5000,
        "max_output": 0,
        "input_price": 0.09,
        "output_price": "-",
        "capabilities": "A",
        "description": "Fast English, $0.09/1K chars"
    },
    # Sound Effects
    {
        "id": "eleven_sound_effects",
        "alias": "sound-effects",
        "name": "Sound Effects",
        "context_window": 450,
        "max_output": 0,
        "input_price": 0.18,
        "output_price": "-",
        "capabilities": "A",
        "description": "Sound effect generation, $0.18/1K chars"
    },
    # Voice Isolation
    {
        "id": "eleven_voice_isolation",
        "alias": "voice-isolation",
        "name": "Voice Isolation",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.50,
        "output_price": "-",
        "capabilities": "A",
        "description": "Isolate voices from audio, $0.50/min"
    },
    # Speech to Speech
    {
        "id": "eleven_speech_to_speech_v2",
        "alias": "speech-to-speech",
        "name": "Speech to Speech v2",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.25,
        "output_price": "-",
        "capabilities": "A",
        "description": "Voice conversion, $0.25/1K chars"
    },
    # Conversational
    {
        "id": "eleven_conversational_v1",
        "alias": "conversational-v1",
        "name": "Conversational v1",
        "context_window": 5000,
        "max_output": 0,
        "input_price": 0.18,
        "output_price": "-",
        "capabilities": "A",
        "description": "Optimized for dialogue, $0.18/1K chars"
    },
]

def process_models() -> list:
    """Process ElevenLabs models into CSV format."""
    rows = []

    for model in ELEVENLABS_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"elevenlabs/{model['id']}",
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
            'source': 'elevenlabs',
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
    print("Processing ElevenLabs models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/specialized/elevenlabs.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
