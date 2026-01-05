#!/usr/bin/env python3
"""
Fetch Deepgram models.
Deepgram provides speech-to-text (STT) models.
"""

import csv
import sys
from datetime import datetime

DEEPGRAM_MODELS = [
    # Nova 3 Series (Latest)
    {
        "id": "nova-3",
        "alias": "nova-3",
        "name": "Nova 3",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0043,
        "output_price": "-",
        "capabilities": "A",
        "description": "Most accurate STT, $0.0043/min"
    },
    {
        "id": "nova-3-medical",
        "alias": "nova-3-medical",
        "name": "Nova 3 Medical",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0050,
        "output_price": "-",
        "capabilities": "A",
        "description": "Medical transcription, $0.005/min"
    },
    # Nova 2 Series
    {
        "id": "nova-2",
        "alias": "nova-2",
        "name": "Nova 2",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0043,
        "output_price": "-",
        "capabilities": "A",
        "description": "Previous gen STT, $0.0043/min"
    },
    {
        "id": "nova-2-general",
        "alias": "nova-2-general",
        "name": "Nova 2 General",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0043,
        "output_price": "-",
        "capabilities": "A",
        "description": "General purpose, $0.0043/min"
    },
    {
        "id": "nova-2-meeting",
        "alias": "nova-2-meeting",
        "name": "Nova 2 Meeting",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0043,
        "output_price": "-",
        "capabilities": "A",
        "description": "Meeting transcription, $0.0043/min"
    },
    {
        "id": "nova-2-phonecall",
        "alias": "nova-2-phone",
        "name": "Nova 2 Phone",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0043,
        "output_price": "-",
        "capabilities": "A",
        "description": "Phone call transcription, $0.0043/min"
    },
    {
        "id": "nova-2-voicemail",
        "alias": "nova-2-voicemail",
        "name": "Nova 2 Voicemail",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0043,
        "output_price": "-",
        "capabilities": "A",
        "description": "Voicemail transcription, $0.0043/min"
    },
    {
        "id": "nova-2-finance",
        "alias": "nova-2-finance",
        "name": "Nova 2 Finance",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0043,
        "output_price": "-",
        "capabilities": "A",
        "description": "Financial transcription, $0.0043/min"
    },
    {
        "id": "nova-2-conversational-ai",
        "alias": "nova-2-conv",
        "name": "Nova 2 Conversational",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0043,
        "output_price": "-",
        "capabilities": "A",
        "description": "Conversational AI, $0.0043/min"
    },
    {
        "id": "nova-2-drivethru",
        "alias": "nova-2-drive",
        "name": "Nova 2 Drive-thru",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0043,
        "output_price": "-",
        "capabilities": "A",
        "description": "Drive-thru ordering, $0.0043/min"
    },
    {
        "id": "nova-2-atc",
        "alias": "nova-2-atc",
        "name": "Nova 2 ATC",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0043,
        "output_price": "-",
        "capabilities": "A",
        "description": "Air traffic control, $0.0043/min"
    },
    # Whisper Cloud
    {
        "id": "whisper-cloud",
        "alias": "whisper-cloud",
        "name": "Whisper Cloud",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0048,
        "output_price": "-",
        "capabilities": "A",
        "description": "OpenAI Whisper on Deepgram, $0.0048/min"
    },
    {
        "id": "whisper-cloud-large",
        "alias": "whisper-large",
        "name": "Whisper Large",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0048,
        "output_price": "-",
        "capabilities": "A",
        "description": "Whisper Large, $0.0048/min"
    },
    {
        "id": "whisper-cloud-medium",
        "alias": "whisper-medium",
        "name": "Whisper Medium",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0048,
        "output_price": "-",
        "capabilities": "A",
        "description": "Whisper Medium, $0.0048/min"
    },
    # Enhanced
    {
        "id": "enhanced",
        "alias": "enhanced",
        "name": "Enhanced",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0145,
        "output_price": "-",
        "capabilities": "A",
        "description": "Enhanced accuracy, $0.0145/min"
    },
    # Base
    {
        "id": "base",
        "alias": "base",
        "name": "Base",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0125,
        "output_price": "-",
        "capabilities": "A",
        "description": "Base model, $0.0125/min"
    },
    # Text-to-Speech
    {
        "id": "aura-asteria-en",
        "alias": "aura-asteria",
        "name": "Aura Asteria",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0150,
        "output_price": "-",
        "capabilities": "A",
        "description": "TTS Asteria voice, $0.015/1K chars"
    },
    {
        "id": "aura-luna-en",
        "alias": "aura-luna",
        "name": "Aura Luna",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0150,
        "output_price": "-",
        "capabilities": "A",
        "description": "TTS Luna voice, $0.015/1K chars"
    },
    {
        "id": "aura-stella-en",
        "alias": "aura-stella",
        "name": "Aura Stella",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0150,
        "output_price": "-",
        "capabilities": "A",
        "description": "TTS Stella voice, $0.015/1K chars"
    },
    {
        "id": "aura-orion-en",
        "alias": "aura-orion",
        "name": "Aura Orion",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.0150,
        "output_price": "-",
        "capabilities": "A",
        "description": "TTS Orion voice, $0.015/1K chars"
    },
]

def process_models() -> list:
    """Process Deepgram models into CSV format."""
    rows = []

    for model in DEEPGRAM_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"deepgram/{model['id']}",
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
            'source': 'deepgram',
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
    print("Processing Deepgram models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/specialized/deepgram.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
