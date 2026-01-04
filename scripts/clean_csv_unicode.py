#!/usr/bin/env python3
"""
Clean Unicode characters from CSV files for safe Rust string literal handling.

This script sanitizes CSV files by replacing fancy Unicode characters with ASCII equivalents.
"""

import csv
import sys
from pathlib import Path
from typing import List, Dict, Any

# Mapping of problematic Unicode characters to ASCII equivalents
UNICODE_REPLACEMENTS = {
    '\u2018': "'",  # Left single quotation mark
    '\u2019': "'",  # Right single quotation mark
    '\u201c': '"',  # Left double quotation mark
    '\u201d': '"',  # Right double quotation mark
    '\u2013': '-',  # En dash
    '\u2014': '-',  # Em dash
    '\u2010': '-',  # Hyphen
    '\u2011': '-',  # Non-breaking hyphen
    '\u2212': '-',  # Minus sign
    '\u00b4': "'",  # Acute accent
    '`': "'",       # Backtick
    '\xa0': ' ',    # Non-breaking space
}

def sanitize_text(text: str) -> str:
    """Sanitize text by replacing Unicode characters with ASCII equivalents."""
    if not text:
        return text

    # First pass: explicit replacements
    for unicode_char, ascii_char in UNICODE_REPLACEMENTS.items():
        text = text.replace(unicode_char, ascii_char)

    # Second pass: remove/replace any remaining non-ASCII characters
    # Keep only ASCII letters, digits, common punctuation, and whitespace
    safe_chars = []
    for char in text:
        if ord(char) < 128:  # ASCII range
            safe_chars.append(char)
        else:  # Non-ASCII character - remove it entirely
            # Don't add anything - just skip the character
            pass

    result = ''.join(safe_chars)

    # Third pass: fix common issues
    # Remove unclosed brackets/parentheses at the end
    result = result.rstrip('([{')

    # Clean up double spaces
    while '  ' in result:
        result = result.replace('  ', ' ')

    return result.strip()

def clean_csv_file(csv_path: Path) -> int:
    """Clean a CSV file by sanitizing all Unicode characters."""
    if not csv_path.exists():
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        return 0

    print(f"Cleaning {csv_path.name}...", file=sys.stderr)

    try:
        # Read all rows
        rows = []
        fieldnames = None

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                # Sanitize all fields
                cleaned_row = {}
                for key, value in row.items():
                    cleaned_row[key] = sanitize_text(value) if value else value
                rows.append(cleaned_row)

        # Write back cleaned data
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            if fieldnames:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

        print(f"✓ Cleaned {len(rows)} rows in {csv_path.name}", file=sys.stderr)
        return len(rows)

    except Exception as e:
        print(f"Error processing {csv_path}: {e}", file=sys.stderr)
        return 0

def main():
    """Main entry point."""
    data_dir = Path(__file__).parent.parent / 'data' / 'models'

    csv_files = [
        data_dir / 'aggregators' / 'openrouter.csv',
        data_dir / 'aggregators' / 'bedrock.csv',
        data_dir / 'aggregators' / 'together_ai.csv',
        data_dir / 'core' / 'latest_releases.csv',
    ]

    total_rows = 0
    for csv_file in csv_files:
        if csv_file.exists():
            count = clean_csv_file(csv_file)
            total_rows += count

    print(f"\nTotal rows cleaned: {total_rows}", file=sys.stderr)
    return 0

if __name__ == '__main__':
    sys.exit(main())
