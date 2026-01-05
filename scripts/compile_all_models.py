#!/usr/bin/env python3
"""
Compile all CSV model files into MODEL_DATA format for models.rs.

This script reads all CSV files from data/models/ subdirectories and
outputs Rust code for the MODEL_DATA constant.
"""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict

# Base directory for model CSVs
BASE_DIR = Path("/home/yfedoseev/projects/modelsuite/data/models")

# Output file for compiled models
OUTPUT_FILE = Path("/home/yfedoseev/projects/modelsuite/scripts/compiled_models.txt")

def load_csv(filepath: Path) -> List[Dict[str, Any]]:
    """Load a CSV file and return list of dicts."""
    if not filepath.exists():
        print(f"Warning: {filepath} not found", file=sys.stderr)
        return []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            return list(reader)
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return []

def format_price(price_str: str) -> str:
    """Format price for MODEL_DATA."""
    if price_str == '-' or price_str == '' or price_str is None:
        return '-'
    try:
        price = float(price_str)
        if price == 0:
            return '0'
        # Format with appropriate precision
        if price < 0.001:
            return f"{price:.6f}"
        elif price < 1:
            return f"{price:.4f}"
        else:
            return f"{price:.2f}"
    except:
        return '-'

def format_context(context_str: str) -> str:
    """Format context window for MODEL_DATA (in K units)."""
    try:
        context = int(context_str)
        if context == 0:
            return '-'
        if context >= 1000000:
            return f"{context // 1000}K"
        elif context >= 1000:
            return f"{context // 1000}K"
        else:
            return str(context)
    except:
        return '-'

def format_output(output_str: str) -> str:
    """Format max output for MODEL_DATA."""
    try:
        output = int(output_str)
        if output == 0:
            return '-'
        if output >= 1000:
            return f"{output // 1000}K"
        else:
            return str(output)
    except:
        return '-'

def model_to_pipe_format(model: Dict[str, Any]) -> str:
    """Convert a model dict to pipe-delimited format.

    Format: id|alias|name|status|in_price/out_price/cache|context/output|caps|benchmarks|desc|class
    """
    model_id = model.get('id', '')
    alias = model.get('alias', model_id.split('/')[-1])
    name = model.get('name', alias)
    status = model.get('status', 'C')

    # Pricing
    in_price = format_price(model.get('input_price', '-'))
    out_price = format_price(model.get('output_price', '-'))
    cache_price = format_price(model.get('cache_input_price', '-'))

    if cache_price != '-':
        pricing = f"{in_price}/{out_price}/{cache_price}"
    else:
        pricing = f"{in_price}/{out_price}"

    # Context/Output
    context = format_context(model.get('context_window', '0'))
    max_output = format_output(model.get('max_output', '0'))
    context_output = f"{context}/{max_output}"

    # Capabilities
    caps = model.get('capabilities', '-')
    if not caps or caps == '':
        caps = '-'

    # Benchmarks (placeholder)
    benchmarks = "-/-/-"

    # Description (truncate)
    desc = model.get('description') or ''
    desc = desc[:80].replace('|', '-')

    # Classification
    classify = "chat"  # Default
    if 'E' in caps:
        classify = "embed"
    elif 'I' in caps:
        classify = "image"
    elif 'A' in caps:
        classify = "audio"
    elif 'D' in caps:
        classify = "video"
    elif 'R' in caps:
        classify = "rerank"

    return f'        "{model_id}|{alias}|{name}|{status}|{pricing}|{context_output}|{caps}|{benchmarks}|{desc}|{classify}",'

def scan_directories() -> Dict[str, List[Path]]:
    """Scan for all CSV files organized by category."""
    categories = defaultdict(list)

    if not BASE_DIR.exists():
        print(f"Base directory {BASE_DIR} not found", file=sys.stderr)
        return categories

    for subdir in BASE_DIR.iterdir():
        if subdir.is_dir():
            for csv_file in subdir.glob("*.csv"):
                categories[subdir.name].append(csv_file)

    return categories

def main():
    print("=== Compiling All Models ===")
    print(f"Base directory: {BASE_DIR}")

    # Scan for CSV files
    categories = scan_directories()

    if not categories:
        print("No CSV files found!", file=sys.stderr)
        sys.exit(1)

    # Collect all models
    all_models = []
    stats = {}

    for category, files in sorted(categories.items()):
        print(f"\n=== Category: {category} ===")
        category_count = 0

        for csv_file in sorted(files):
            models = load_csv(csv_file)
            if models:
                print(f"  {csv_file.name}: {len(models)} models")
                all_models.extend(models)
                category_count += len(models)

        stats[category] = category_count

    print(f"\n=== Summary ===")
    for category, count in sorted(stats.items()):
        print(f"  {category}: {count} models")
    print(f"  TOTAL: {len(all_models)} models")

    # Deduplicate by model ID
    seen_ids = set()
    unique_models = []
    for model in all_models:
        model_id = model.get('id', '')
        if model_id and model_id not in seen_ids:
            seen_ids.add(model_id)
            unique_models.append(model)

    print(f"\n  After deduplication: {len(unique_models)} models")

    # Generate output
    output_lines = []
    output_lines.append("// Auto-generated MODEL_DATA entries")
    output_lines.append("// Generated by compile_all_models.py")
    output_lines.append(f"// Total models: {len(unique_models)}")
    output_lines.append("")
    output_lines.append("const MODEL_DATA_NEW: &[&str] = &[")

    # Group by provider prefix for organization
    by_provider = defaultdict(list)
    for model in unique_models:
        model_id = model.get('id', '')
        prefix = model_id.split('/')[0] if '/' in model_id else 'other'
        by_provider[prefix].append(model)

    for provider, models in sorted(by_provider.items()):
        output_lines.append(f"    // === {provider.upper()} ({len(models)} models) ===")
        for model in models:
            output_lines.append(model_to_pipe_format(model))
        output_lines.append("")

    output_lines.append("];")

    # Write output
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"\n=== Output written to {OUTPUT_FILE} ===")
    print(f"Total unique models: {len(unique_models)}")

if __name__ == '__main__':
    main()
