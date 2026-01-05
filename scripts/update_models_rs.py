#!/usr/bin/env python3
"""
Update models.rs with data from CSV files.
Converts CSV format to MODEL_DATA DSL format.
"""

import csv
import os
import re
from pathlib import Path

DATA_DIR = Path("/home/yfedoseev/projects/modelsuite/data/models")
MODELS_RS = Path("/home/yfedoseev/projects/modelsuite/src/models.rs")

def read_csv_models(csv_path: Path) -> list:
    """Read models from a CSV file."""
    models = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                models.append(row)
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
    return models

def format_price(price_str: str) -> str:
    """Format price value."""
    if not price_str or price_str == '-':
        return ""
    try:
        price = float(price_str)
        if price >= 1.0:
            return f"{price:.1f}"
        elif price >= 0.1:
            return f"{price:.2f}"
        else:
            return f"{price:.3f}"
    except ValueError:
        return price_str

def csv_to_model_data(row: dict) -> str:
    """Convert a CSV row to MODEL_DATA format."""
    model_id = row.get('id', '')
    alias = row.get('alias', '')
    name = row.get('name', '')[:60]  # Limit name length
    status = row.get('status', 'C')

    # Format pricing
    input_price = format_price(row.get('input_price', ''))
    output_price = format_price(row.get('output_price', ''))
    cache_price = format_price(row.get('cache_input_price', ''))

    if cache_price and cache_price != '-':
        pricing = f"{input_price},{output_price},{cache_price}"
    else:
        pricing = f"{input_price},{output_price}"

    # Format context
    context_window = row.get('context_window', '0')
    max_output = row.get('max_output', '0')
    context = f"{context_window},{max_output}"

    # Capabilities
    caps = row.get('capabilities', '-') or '-'

    # Benchmarks (we don't have these in CSV)
    benchmarks = '-'

    # Description (clean up)
    description = row.get('description', '')[:80]
    description = description.replace('|', '-').replace('\n', ' ').strip()

    # Classify (default to Y for chat models)
    classify = 'Y'
    if 'embedding' in caps.lower() or 'E' in caps:
        classify = 'N'
    if 'I' in caps:  # Image
        classify = 'N'
    if 'A' in caps:  # Audio
        classify = 'N'

    return f"{model_id}|{alias}|{name}|{status}|{pricing}|{context}|{caps}|{benchmarks}|{description}|{classify}"

def generate_provider_section(provider_name: str, csv_path: Path, header_url: str = "") -> str:
    """Generate a MODEL_DATA section for a provider."""
    models = read_csv_models(csv_path)
    if not models:
        return ""

    lines = []
    lines.append(f"# =============================================================================")
    if header_url:
        lines.append(f"# {provider_name.upper()} ({header_url})")
    else:
        lines.append(f"# {provider_name.upper()}")
    lines.append(f"# =============================================================================")

    for row in models:
        line = csv_to_model_data(row)
        lines.append(line)

    return "\n".join(lines)

def update_fireworks_section(models_rs_content: str) -> str:
    """Update the Fireworks section in models.rs."""
    fireworks_csv = DATA_DIR / "inference" / "fireworks.csv"
    if not fireworks_csv.exists():
        print(f"Fireworks CSV not found: {fireworks_csv}")
        return models_rs_content

    # Generate new Fireworks section
    new_section = generate_provider_section(
        "FIREWORKS - Fast Inference",
        fireworks_csv,
        "fireworks.ai"
    )

    # Find and replace Fireworks section
    pattern = r'# =+\n# FIREWORKS[^\n]*\n# =+\n(.*?)(?=# =+|"#;)'

    # Simple approach: find the Fireworks section and replace models
    lines = models_rs_content.split('\n')
    new_lines = []
    in_fireworks = False
    skip_until_next_section = False

    for line in lines:
        if '# FIREWORKS' in line:
            in_fireworks = True
            skip_until_next_section = True
            # Add the new Fireworks section
            new_lines.append(new_section)
            continue

        if skip_until_next_section:
            if line.startswith('# =====') and '# FIREWORKS' not in ''.join(new_lines[-3:]):
                skip_until_next_section = False
                new_lines.append(line)
            continue

        new_lines.append(line)

    return '\n'.join(new_lines)

def main():
    print("Updating models.rs with Fireworks data...")

    # Read current models.rs
    if not MODELS_RS.exists():
        print(f"models.rs not found: {MODELS_RS}")
        return

    with open(MODELS_RS, 'r', encoding='utf-8') as f:
        content = f.read()

    # Count current Fireworks models
    fireworks_pattern = r'^fireworks/[^|]+\|'
    current_count = len(re.findall(fireworks_pattern, content, re.MULTILINE))
    print(f"Current Fireworks models in models.rs: {current_count}")

    # Generate new Fireworks section
    fireworks_csv = DATA_DIR / "inference" / "fireworks.csv"
    models = read_csv_models(fireworks_csv)
    print(f"Fireworks models in CSV: {len(models)}")

    # Generate new entries
    new_entries = []
    for row in models:
        line = csv_to_model_data(row)
        new_entries.append(line)

    # Save to intermediate file for review
    output_file = "/home/yfedoseev/projects/modelsuite/scripts/fireworks_entries.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# =============================================================================\n")
        f.write("# FIREWORKS - Fast Inference (fireworks.ai)\n")
        f.write("# =============================================================================\n")
        for entry in new_entries:
            f.write(entry + "\n")

    print(f"\nGenerated {len(new_entries)} Fireworks entries")
    print(f"Saved to: {output_file}")
    print("\nTo update models.rs, run:")
    print("  python3 scripts/replace_fireworks.py")

if __name__ == '__main__':
    main()
