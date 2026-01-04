#!/usr/bin/env python3
"""
Convert Phase 1 CSV model data to pipe-delimited format for models.rs integration.

This script reads CSV files from data/models/ and converts them to the pipe-delimited
format expected by the MODEL_DATA constant in src/models.rs.
"""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Expected benchmark fields in output format
BENCHMARK_FIELDS = ['mmlu', 'bbh', 'humaneval', 'aime', 'livebench', 'gpqa', 'gpt4_eval', 'agentic', 'ceo_evals', 'context_tokens']

def extract_provider_from_id(model_id: str) -> str:
    """Extract provider prefix from model ID."""
    if '/' in model_id:
        return model_id.split('/')[0]
    # For models without explicit provider, infer from context
    return 'openrouter'

def parse_capabilities(cap_string: str) -> str:
    """Normalize capabilities string."""
    if not cap_string or cap_string == '-':
        return ''
    return cap_string.upper()

def normalize_price(price_str: str) -> str:
    """Normalize price to expected format."""
    if not price_str or price_str == '-' or price_str == '':
        return '0'
    try:
        # Convert to float and back to string to normalize
        price_float = float(price_str)
        # Return with scientific notation for very small numbers
        if price_float < 0.00001 and price_float > 0:
            return f"{price_float:.10e}".rstrip('0').rstrip('.')
        return str(price_float)
    except ValueError:
        return '0'

def sanitize_description(desc: str) -> str:
    """Sanitize description text for Rust raw string literals."""
    if not desc:
        return desc

    # Remove problematic characters for Rust raw strings
    # Keep only basic ASCII letters, numbers, spaces, and basic punctuation
    allowed = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:()[]{}!?-_/')

    desc = ''.join(c for c in desc if c in allowed)

    # Clean up excessive whitespace
    import re
    desc = re.sub(r'\s+', ' ', desc).strip()

    return desc

def build_benchmarks_string(row: Dict[str, str]) -> str:
    """Build benchmarks string from CSV row."""
    benchmarks = []

    # Map CSV benchmark columns to expected fields
    benchmark_mappings = {
        'mmlu_score': 'mmlu',
        'bbh_score': 'bbh',
        'humaneval_score': 'humaneval',
        'aime_score': 'aime',
        'livebench_score': 'livebench',
        'gpqa_score': 'gpqa',
        'gpt4_eval_score': 'gpt4_eval',
        'agentic_score': 'agentic',
        'ceo_evals_score': 'ceo_evals',
    }

    for csv_col, field_name in benchmark_mappings.items():
        value = row.get(csv_col, '-')
        if not value or value == '-':
            benchmarks.append('-')
        else:
            try:
                # Ensure it's a valid number
                float(value)
                benchmarks.append(value)
            except ValueError:
                benchmarks.append('-')

    # Add context window tokens if not already included
    context = row.get('context_window', '0')
    benchmarks.append(context)

    return ','.join(benchmarks)

def csv_row_to_pipe_format(row: Dict[str, str], source: str) -> str:
    """Convert a single CSV row to pipe-delimited format."""
    model_id = row['id'].strip()
    provider = extract_provider_from_id(model_id)

    # Build full provider/id if not already present
    if '/' not in model_id:
        full_id = f"{provider}/{model_id}"
    else:
        full_id = model_id

    alias = row.get('alias', '').strip() or '-'
    name = row.get('name', '').strip() or '-'
    status = row.get('status', 'C').strip() or 'C'

    # Prices
    input_price = normalize_price(row.get('input_price', '0'))
    output_price = normalize_price(row.get('output_price', '0'))
    cache_price = row.get('cache_input_price', '-').strip()
    if cache_price and cache_price != '-':
        cache_price = normalize_price(cache_price)
        price_str = f"{input_price},{output_price},{cache_price}"
    else:
        price_str = f"{input_price},{output_price}"

    # Context and output
    context_window = row.get('context_window', '0').strip() or '0'
    max_output = row.get('max_output', '0').strip() or '0'
    context_str = f"{context_window},{max_output}"

    # Capabilities
    capabilities = parse_capabilities(row.get('capabilities', ''))

    # Benchmarks
    benchmarks = build_benchmarks_string(row)

    # Description - sanitize, escape quotes and limit length
    description = row.get('description', '').strip()
    description = sanitize_description(description)  # Sanitize Unicode characters
    if len(description) > 200:
        description = description[:197] + '...'
    description = description.replace('"', '\\"')

    # Can classify - default to Y
    can_classify = row.get('can_classify', 'Y').strip() or 'Y'

    # Build the pipe-delimited line
    line = f"{full_id}|{alias}|{name}|{status}|{price_str}|{context_str}|{capabilities}|{benchmarks}|{description}|{can_classify}"

    return line

def process_csv_file(csv_path: Path, source: str) -> List[str]:
    """Process a single CSV file and return pipe-delimited lines."""
    lines = []

    print(f"Processing {csv_path.name}...", file=sys.stderr)

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (skip header)
                try:
                    line = csv_row_to_pipe_format(row, source)
                    lines.append(line)
                except Exception as e:
                    print(f"Error processing row {row_num} in {csv_path.name}: {e}", file=sys.stderr)
                    continue

        print(f"✓ Processed {len(lines)} models from {csv_path.name}", file=sys.stderr)
        return lines
    except Exception as e:
        print(f"Error reading {csv_path}: {e}", file=sys.stderr)
        return []

def generate_model_data_section() -> str:
    """Generate the complete MODEL_DATA addition for all Phase 1-2 CSV files."""
    all_lines = []
    data_dir = Path(__file__).parent.parent / 'data' / 'models'

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Process OpenRouter models
    or_file = data_dir / 'aggregators' / 'openrouter.csv'
    if or_file.exists():
        or_lines = process_csv_file(or_file, 'openrouter')
        if or_lines:
            all_lines.append("\n# =============================================================================")
            all_lines.append("# OPENROUTER (Meta-aggregator with 353+ models)")
            all_lines.append("# =============================================================================")
            all_lines.extend(or_lines)

    # Process Bedrock models
    br_file = data_dir / 'aggregators' / 'bedrock.csv'
    if br_file.exists():
        br_lines = process_csv_file(br_file, 'bedrock')
        if br_lines:
            all_lines.append("\n# =============================================================================")
            all_lines.append("# AWS BEDROCK (Enterprise provider with 48+ models)")
            all_lines.append("# =============================================================================")
            all_lines.extend(br_lines)

    # Process Together AI models
    tai_file = data_dir / 'aggregators' / 'together_ai.csv'
    if tai_file.exists():
        tai_lines = process_csv_file(tai_file, 'together')
        if tai_lines:
            all_lines.append("\n# =============================================================================")
            all_lines.append("# TOGETHER AI (Open-source models with 61+ models)")
            all_lines.append("# =============================================================================")
            all_lines.extend(tai_lines)

    # Process Latest releases
    lr_file = data_dir / 'core' / 'latest_releases.csv'
    if lr_file.exists():
        lr_lines = process_csv_file(lr_file, 'latest')
        if lr_lines:
            all_lines.append("\n# =============================================================================")
            all_lines.append("# LATEST RELEASES (Frontier models from January 2026)")
            all_lines.append("# =============================================================================")
            all_lines.extend(lr_lines)

    return '\n'.join(all_lines)

def main():
    """Main entry point."""
    print("Generating model data...", file=sys.stderr)
    output = generate_model_data_section()

    if not output:
        print("Error: No CSV files processed successfully", file=sys.stderr)
        sys.exit(1)

    # Print to stdout for piping to a file
    sys.stdout.write(output)

    print(f"\nGeneration complete. Total lines: {len(output.splitlines())}", file=sys.stderr)

if __name__ == '__main__':
    main()
