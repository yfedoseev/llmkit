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

def normalize_price(price_str: str, convert_to_per_million: bool = True) -> str:
    """Normalize price to expected format (per 1M tokens).

    Args:
        price_str: Price string from CSV
        convert_to_per_million: If True, assumes input is per-token and converts to per-1M
    """
    if not price_str or price_str == '-' or price_str == '':
        return '0'
    try:
        price_float = float(price_str)

        # If price is very small (< 0.001), it's likely per-token pricing
        # Convert to per-1M tokens format for consistency
        if convert_to_per_million and price_float > 0 and price_float < 0.001:
            price_float = price_float * 1_000_000

        # Round to reasonable precision and format
        if price_float == 0:
            return '0'
        elif price_float < 0.01:
            return f"{price_float:.4f}"
        elif price_float < 1:
            return f"{price_float:.2f}"
        else:
            return f"{price_float:.2f}".rstrip('0').rstrip('.')
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
    """Generate the complete MODEL_DATA addition for all CSV files."""
    all_lines = []
    data_dir = Path(__file__).parent.parent / 'data' / 'models'

    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Define all CSV files to process with their section headers
    csv_configs = [
        ('aggregators/openrouter.csv', 'OPENROUTER', 'Meta-aggregator'),
        ('aggregators/bedrock.csv', 'AWS BEDROCK', 'Enterprise provider'),
        ('aggregators/together_ai.csv', 'TOGETHER AI', 'Open-source models'),
        ('inference/groq.csv', 'GROQ', 'Fast inference'),
        ('inference/fireworks.csv', 'FIREWORKS', 'Fast inference'),
        ('inference/cerebras.csv', 'CEREBRAS', 'Fast inference'),
        ('inference/sambanova.csv', 'SAMBANOVA', 'Enterprise inference'),
        ('inference/perplexity.csv', 'PERPLEXITY', 'Search-augmented'),
        ('inference/hyperbolic.csv', 'HYPERBOLIC', 'Inference platform'),
        ('inference/novita.csv', 'NOVITA', 'Inference platform'),
        ('inference/nebius.csv', 'NEBIUS', 'Cloud inference'),
        ('core/google.csv', 'GOOGLE', 'Direct API'),
        ('core/mistral.csv', 'MISTRAL', 'Direct API'),
        ('core/cohere.csv', 'COHERE', 'Direct API'),
        ('core/deepseek.csv', 'DEEPSEEK', 'Direct API'),
        ('core/latest_releases.csv', 'LATEST RELEASES', 'Frontier models'),
        ('chinese/qwen.csv', 'QWEN', 'Chinese provider'),
        ('chinese/baichuan.csv', 'BAICHUAN', 'Chinese provider'),
        ('chinese/zhipu.csv', 'ZHIPU', 'Chinese provider'),
        ('chinese/minimax.csv', 'MINIMAX', 'Chinese provider'),
        ('chinese/moonshot.csv', 'MOONSHOT', 'Chinese provider'),
        ('chinese/yi.csv', 'YI', 'Chinese provider'),
        ('chinese/volcengine.csv', 'VOLCENGINE', 'Chinese provider'),
        ('chinese/spark.csv', 'SPARK', 'Chinese provider'),
        ('specialized/stability.csv', 'STABILITY', 'Image generation'),
        ('specialized/elevenlabs.csv', 'ELEVENLABS', 'Voice synthesis'),
        ('specialized/voyage.csv', 'VOYAGE', 'Embeddings'),
        ('specialized/jina.csv', 'JINA', 'Embeddings'),
        ('specialized/deepgram.csv', 'DEEPGRAM', 'Speech-to-text'),
    ]

    for csv_path, section_name, description in csv_configs:
        full_path = data_dir / csv_path
        if full_path.exists():
            lines = process_csv_file(full_path, section_name.lower())
            if lines:
                all_lines.append(f"\n# =============================================================================")
                all_lines.append(f"# {section_name} ({description})")
                all_lines.append("# =============================================================================")
                all_lines.extend(lines)

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
