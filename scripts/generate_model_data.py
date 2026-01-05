#!/usr/bin/env python3
"""
Generate MODEL_DATA for models.rs from all CSV files.
Converts CSV format to MODEL_DATA DSL format and generates the complete Rust constant.
"""

import csv
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATA_DIR = Path("/home/yfedoseev/projects/modelsuite/data/models")
OUTPUT_FILE = Path("/home/yfedoseev/projects/modelsuite/scripts/model_data_complete.txt")

def read_all_csvs():
    """Read all CSV files and return models grouped by provider."""
    all_models = []

    for csv_file in DATA_DIR.rglob("*.csv"):
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('id'):
                        row['_source_file'] = str(csv_file.relative_to(DATA_DIR))
                        all_models.append(row)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    return all_models

def format_price(price_str):
    """Format price value."""
    if not price_str or price_str == '-' or price_str == '':
        return ""
    try:
        price = float(price_str)
        if price == 0:
            return "0"
        elif price >= 1.0:
            return f"{price:.1f}"
        elif price >= 0.01:
            return f"{price:.2f}"
        else:
            return f"{price:.4f}"
    except ValueError:
        return ""

def get_provider_from_id(model_id):
    """Extract provider from model ID."""
    if '/' in model_id:
        return model_id.split('/')[0]
    return "unknown"

def csv_to_model_data(row):
    """Convert a CSV row to MODEL_DATA format."""
    model_id = (row.get('id') or '').strip()
    if not model_id:
        return None

    alias = (row.get('alias') or '').strip()
    name = (row.get('name') or '')[:60].strip()
    status = (row.get('status') or 'C').strip() or 'C'

    # Format pricing: input,output[,cache]
    input_price = format_price(row.get('input_price', ''))
    output_price = format_price(row.get('output_price', ''))
    cache_price = format_price(row.get('cache_input_price', ''))

    pricing_parts = []
    if input_price:
        pricing_parts.append(input_price)
    if output_price:
        pricing_parts.append(output_price)
    elif input_price:
        pricing_parts.append("")  # Need placeholder if input exists
    if cache_price and cache_price != '-':
        if len(pricing_parts) < 2:
            pricing_parts.extend([""] * (2 - len(pricing_parts)))
        pricing_parts.append(cache_price)

    pricing = ",".join(pricing_parts) if pricing_parts else "-"

    # Format context: max_context,max_output
    context_window = row.get('context_window', '0')
    max_output = row.get('max_output', '0')
    try:
        ctx = int(float(context_window)) if context_window else 0
        out = int(float(max_output)) if max_output else 0
        context = f"{ctx},{out}"
    except:
        context = "0,0"

    # Capabilities
    caps = (row.get('capabilities') or '-').strip()
    if not caps:
        caps = '-'

    # Benchmarks (not in CSV)
    benchmarks = '-'

    # Description (clean up)
    description = (row.get('description') or '')[:80]
    description = description.replace('|', '-').replace('\n', ' ').replace('\r', '').strip()
    if not description:
        description = name

    # Classify: Y for chat models, N for special types
    classify = 'Y'
    caps_upper = caps.upper()
    if 'E' in caps_upper:  # Embedding
        classify = 'N'
    if caps_upper == 'I':  # Image only
        classify = 'N'
    if caps_upper == 'A':  # Audio only
        classify = 'N'
    if caps_upper == 'D':  # Video only
        classify = 'N'
    if caps_upper == 'M':  # Moderation only
        classify = 'N'
    if 'K' in caps_upper and len(caps_upper) <= 3:  # Thinking-only models
        classify = 'N'

    return f"{model_id}|{alias}|{name}|{status}|{pricing}|{context}|{caps}|{benchmarks}|{description}|{classify}"

def get_provider_display_name(provider):
    """Get display name for provider."""
    names = {
        'anthropic': 'ANTHROPIC - Direct API',
        'openai': 'OPENAI - Direct API',
        'google': 'GOOGLE - Direct API',
        'mistral': 'MISTRAL - Direct API',
        'cohere': 'COHERE - Direct API',
        'deepseek': 'DEEPSEEK - Direct API',
        'groq': 'GROQ - Fast Inference',
        'cerebras': 'CEREBRAS - Ultra-fast Inference',
        'sambanova': 'SAMBANOVA - Fast Inference',
        'fireworks': 'FIREWORKS - Fast Inference',
        'together': 'TOGETHER AI - Aggregator',
        'openrouter': 'OPENROUTER - Aggregator',
        'bedrock': 'AWS BEDROCK - Cloud',
        'perplexity': 'PERPLEXITY - Search AI',
        'ai21': 'AI21 - Jamba',
        'replicate': 'REPLICATE - Model Hub',
        'huggingface': 'HUGGINGFACE - Inference API',
        'stability': 'STABILITY AI - Image Generation',
        'elevenlabs': 'ELEVENLABS - Voice AI',
        'deepgram': 'DEEPGRAM - Speech-to-Text',
        'voyage': 'VOYAGE AI - Embeddings',
        'jina': 'JINA AI - Embeddings',
        'novita': 'NOVITA AI - Inference',
        'nebius': 'NEBIUS - Inference',
        'hyperbolic': 'HYPERBOLIC - Inference',
        'qwen': 'QWEN/DASHSCOPE - Alibaba',
        'zhipu': 'ZHIPU/GLM - Chinese AI',
        'minimax': 'MINIMAX - Chinese AI',
        'moonshot': 'MOONSHOT/KIMI - Chinese AI',
        'baichuan': 'BAICHUAN - Chinese AI',
        'yi': 'YI/01.AI - Chinese AI',
        'volcengine': 'VOLCENGINE/DOUBAO - ByteDance',
        'spark': 'SPARK/IFLYTEK - Chinese AI',
    }
    return names.get(provider, provider.upper())

def generate_model_data():
    """Generate complete MODEL_DATA content."""
    print("Reading all CSV files...")
    all_models = read_all_csvs()
    print(f"Found {len(all_models)} total entries")

    # Deduplicate by model ID
    seen_ids = {}
    for model in all_models:
        model_id = model.get('id', '').strip()
        if model_id and model_id not in seen_ids:
            seen_ids[model_id] = model

    print(f"After deduplication: {len(seen_ids)} unique models")

    # Group by provider
    by_provider = defaultdict(list)
    for model_id, model in seen_ids.items():
        provider = get_provider_from_id(model_id)
        by_provider[provider].append(model)

    # Sort providers
    provider_order = [
        'anthropic', 'openai', 'google', 'mistral', 'deepseek', 'cohere',
        'groq', 'cerebras', 'sambanova', 'fireworks', 'perplexity',
        'together', 'openrouter', 'bedrock',
        'ai21', 'replicate', 'huggingface',
        'qwen', 'zhipu', 'minimax', 'moonshot', 'baichuan', 'yi', 'volcengine', 'spark',
        'stability', 'elevenlabs', 'deepgram', 'voyage', 'jina',
        'novita', 'nebius', 'hyperbolic',
    ]

    # Add any providers not in the order
    all_providers = set(by_provider.keys())
    for p in all_providers:
        if p not in provider_order:
            provider_order.append(p)

    # Generate output
    lines = []
    lines.append("# =============================================================================")
    lines.append("# MODEL REGISTRY - Auto-generated from CSV files")
    lines.append(f"# Format: id|alias|name|status|pricing|context|caps|benchmarks|description|classify")
    lines.append(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"# Total models: {len(seen_ids)}")
    lines.append("# =============================================================================")
    lines.append("")

    model_count = 0
    for provider in provider_order:
        if provider not in by_provider:
            continue

        models = by_provider[provider]
        if not models:
            continue

        # Sort models by ID
        models.sort(key=lambda x: x.get('id', ''))

        lines.append(f"# =============================================================================")
        lines.append(f"# {get_provider_display_name(provider)} ({len(models)} models)")
        lines.append(f"# =============================================================================")

        for model in models:
            line = csv_to_model_data(model)
            if line:
                lines.append(line)
                model_count += 1

        lines.append("")

    print(f"Generated {model_count} model entries")

    return "\n".join(lines)

def main():
    content = generate_model_data()

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"\nOutput written to: {OUTPUT_FILE}")
    print(f"\nTo update models.rs, copy the content between the MODEL_DATA markers.")

if __name__ == '__main__':
    main()
