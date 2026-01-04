#!/usr/bin/env python3
"""
Fetch models from OpenRouter API and generate CSV file.
"""

import json
import requests
import csv
import sys
from typing import Optional
from datetime import datetime

def map_capabilities(supported_params: list, input_modalities: list, output_modalities: list) -> str:
    """Map OpenRouter API parameters to capability flags."""
    capabilities = []

    # Vision
    if 'image' in input_modalities:
        capabilities.append('V')

    # Tools
    if 'tools' in supported_params:
        capabilities.append('T')

    # JSON/Structured Output
    if 'response_format' in supported_params or 'structured_outputs' in supported_params:
        capabilities.append('J')

    # Structured Output (separate flag)
    if 'structured_outputs' in supported_params:
        if 'S' not in capabilities:
            capabilities.append('S')

    # Thinking/Reasoning
    if 'reasoning' in supported_params or 'include_reasoning' in supported_params:
        capabilities.append('K')

    return ''.join(sorted(set(capabilities))) if capabilities else '-'

def get_model_status(model_id: str, created: int) -> str:
    """Determine model status based on creation date and name."""
    current_timestamp = datetime.now().timestamp()
    days_old = (current_timestamp - created) / 86400

    # Models older than 6 months might be legacy
    if days_old > 180 and any(x in model_id.lower() for x in ['deprecated', 'legacy', 'old']):
        return 'L'

    return 'C'  # Default to Current

def fetch_openrouter_models() -> Optional[list]:
    """Fetch all models from OpenRouter API."""
    try:
        response = requests.get('https://openrouter.ai/api/v1/models', timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get('data', [])
    except Exception as e:
        print(f"Error fetching OpenRouter models: {e}", file=sys.stderr)
        return None

def process_models(models: list) -> list:
    """Process OpenRouter models into CSV format."""
    rows = []

    for model in models:
        try:
            model_id = model.get('id', '')
            if not model_id:
                continue

            name = model.get('name', model_id)
            description = model.get('description', '')[:100]  # Truncate to 100 chars

            pricing = model.get('pricing', {})
            input_price = float(pricing.get('prompt', 0))
            output_price = float(pricing.get('completion', 0))

            context_window = model.get('context_length', 4096)
            # Get max_output from top_provider, fall back to context window / 4, minimum 1024
            max_output = model.get('top_provider', {}).get('max_completion_tokens')
            if not max_output:
                max_output = max(1024, context_window // 4)  # Default: 1/4 of context window, min 1024

            supported_params = model.get('supported_parameters', [])
            input_modalities = model.get('architecture', {}).get('input_modalities', [])
            output_modalities = model.get('architecture', {}).get('output_modalities', [])

            capabilities = map_capabilities(supported_params, input_modalities, output_modalities)
            created = model.get('created', int(datetime.now().timestamp()))
            status = get_model_status(model_id, created)

            # Create short alias from model ID
            alias = model_id.split('/')[-1][:20] if '/' in model_id else model_id[:20]

            row = {
                'id': model_id,
                'alias': alias,
                'name': name,
                'status': status,
                'input_price': f"{input_price:.8f}" if input_price > 0 else '-',
                'output_price': f"{output_price:.8f}" if output_price > 0 else '-',
                'cache_input_price': '-',
                'context_window': context_window,
                'max_output': max_output,
                'capabilities': capabilities,
                'quality': 'verified',
                'source': 'openrouter',
                'updated': datetime.fromtimestamp(created).strftime('%Y-%m-%d'),
                'description': description,
                'mmlu_score': '-',
                'humaneval_score': '-',
                'math_score': '-'
            }
            rows.append(row)
        except Exception as e:
            print(f"Error processing model {model.get('id', 'unknown')}: {e}", file=sys.stderr)
            continue

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

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {len(rows)} models to {output_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}", file=sys.stderr)

def main():
    print("Fetching OpenRouter models...")
    models = fetch_openrouter_models()

    if not models:
        print("Failed to fetch models", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(models)} models...")
    rows = process_models(models)

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/aggregators/openrouter.csv'
    save_csv(rows, output_file)

    # Print summary
    current = sum(1 for r in rows if r['status'] == 'C')
    legacy = sum(1 for r in rows if r['status'] == 'L')
    with_vision = sum(1 for r in rows if 'V' in r['capabilities'])
    with_tools = sum(1 for r in rows if 'T' in r['capabilities'])

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")
    print(f"Current: {current}, Legacy: {legacy}")
    print(f"With Vision: {with_vision}, With Tools: {with_tools}")

if __name__ == '__main__':
    main()
