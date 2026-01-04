#!/usr/bin/env python3
"""
Validate model CSV files against the defined schema.
"""

import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

# Validation rules
REQUIRED_FIELDS = ['id', 'alias', 'name', 'status', 'context_window', 'max_output']
VALID_STATUS = {'C', 'L', 'D'}
VALID_CAPABILITIES = {'V', 'T', 'J', 'S', 'K', 'C', '-'}
VALID_QUALITY = {'verified', 'partial', 'estimated'}

class ValidationError:
    def __init__(self, row_num: int, field: str, value: str, error: str):
        self.row_num = row_num
        self.field = field
        self.value = value
        self.error = error

    def __str__(self):
        return f"Row {self.row_num}, Field '{self.field}': {self.error} (value: '{self.value}')"

def validate_csv_file(filepath: str) -> Tuple[List[dict], List[ValidationError]]:
    """Validate a CSV file and return rows and errors."""
    rows = []
    errors = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row_num, row in enumerate(reader, start=2):  # Start at 2 (skip header)
                row_errors = validate_row(row, row_num)
                errors.extend(row_errors)

                if not row_errors:
                    rows.append(row)
                else:
                    rows.append(row)  # Still add for duplicate checking

    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return [], [ValidationError(0, 'file', filepath, str(e))]

    return rows, errors

def validate_row(row: dict, row_num: int) -> List[ValidationError]:
    """Validate a single row."""
    errors = []

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in row or not row[field].strip():
            errors.append(ValidationError(row_num, field, row.get(field, ''), "Required field missing or empty"))

    # Validate status
    status = row.get('status', '').strip()
    if status and status not in VALID_STATUS:
        errors.append(ValidationError(row_num, 'status', status, f"Must be one of {VALID_STATUS}"))

    # Validate pricing
    for price_field in ['input_price', 'output_price', 'cache_input_price']:
        price = row.get(price_field, '-').strip()
        if price != '-':
            try:
                p = float(price)
                if p < 0:
                    errors.append(ValidationError(row_num, price_field, price, "Price must be non-negative"))
                elif p > 10:  # Sanity check
                    errors.append(ValidationError(row_num, price_field, price, "Price seems too high (>$10)"))
            except ValueError:
                errors.append(ValidationError(row_num, price_field, price, "Must be a number or '-'"))

    # Validate context window
    try:
        context = int(row.get('context_window', 0))
        if context < 1000 or context > 10000000:
            errors.append(ValidationError(row_num, 'context_window', str(context), "Must be between 1,000 and 10,000,000"))
    except ValueError:
        errors.append(ValidationError(row_num, 'context_window', row.get('context_window', ''), "Must be an integer"))

    # Validate max output
    try:
        max_out = int(row.get('max_output', 0))
        if max_out < 100 or max_out > 10000000:
            errors.append(ValidationError(row_num, 'max_output', str(max_out), "Must be between 100 and 10,000,000"))
    except ValueError:
        errors.append(ValidationError(row_num, 'max_output', row.get('max_output', ''), "Must be an integer"))

    # Validate capabilities
    capabilities = row.get('capabilities', '').strip()
    if capabilities:
        invalid_caps = set(capabilities) - VALID_CAPABILITIES
        if invalid_caps:
            errors.append(ValidationError(row_num, 'capabilities', capabilities, f"Invalid chars: {invalid_caps}. Valid: {VALID_CAPABILITIES}"))

    # Validate quality
    quality = row.get('quality', '').strip()
    if quality and quality not in VALID_QUALITY:
        errors.append(ValidationError(row_num, 'quality', quality, f"Must be one of {VALID_QUALITY}"))

    # Validate date format
    updated = row.get('updated', '').strip()
    if updated:
        try:
            datetime.strptime(updated, '%Y-%m-%d')
        except ValueError:
            errors.append(ValidationError(row_num, 'updated', updated, "Must be YYYY-MM-DD format"))

    # Price sanity check: output should be >= input (usually)
    try:
        input_price = float(row.get('input_price', '0'))
        output_price = float(row.get('output_price', '0'))
        if input_price > 0 and output_price > 0 and output_price < input_price * 0.5:
            # Warning only
            pass
    except ValueError:
        pass

    return errors

def check_duplicates(rows: List[dict], filepath: str) -> List[str]:
    """Check for duplicate model IDs."""
    seen_ids = {}
    duplicates = []

    for row in rows:
        model_id = row.get('id', '').strip()
        if model_id:
            if model_id in seen_ids:
                duplicates.append(f"Duplicate ID '{model_id}' at rows {seen_ids[model_id]} and {row.get('_row_num', '?')}")
            else:
                seen_ids[model_id] = row.get('id', '')

    return duplicates

def generate_report(files: List[str]) -> None:
    """Generate validation report for all CSV files."""
    all_rows = {}
    all_errors = {}
    total_models = 0
    total_errors = 0

    print("Validating model CSV files...\n")

    for filepath in files:
        if not Path(filepath).exists():
            print(f"File not found: {filepath}")
            continue

        print(f"Validating {Path(filepath).name}...")
        rows, errors = validate_csv_file(filepath)

        all_rows[filepath] = rows
        all_errors[filepath] = errors
        total_models += len(rows)
        total_errors += len(errors)

        if errors:
            print(f"  Found {len(errors)} validation error(s):")
            for error in errors[:10]:  # Show first 10 errors
                print(f"    {error}")
            if len(errors) > 10:
                print(f"    ... and {len(errors) - 10} more errors")
        else:
            print(f"  ✓ All {len(rows)} models valid")

        duplicates = check_duplicates(rows, filepath)
        if duplicates:
            print(f"  Duplicates found:")
            for dup in duplicates[:5]:
                print(f"    {dup}")

        print()

    # Overall summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total models: {total_models}")
    print(f"Total validation errors: {total_errors}")

    # Capability distribution
    print("\nCapability Distribution:")
    capability_count = {}
    for rows in all_rows.values():
        for row in rows:
            caps = row.get('capabilities', '-').strip()
            for cap in caps.split(','):
                cap = cap.strip()
                capability_count[cap] = capability_count.get(cap, 0) + 1

    for cap in sorted(capability_count.keys()):
        print(f"  {cap}: {capability_count[cap]}")

    # Status distribution
    print("\nStatus Distribution:")
    for rows in all_rows.values():
        current = sum(1 for r in rows if r.get('status') == 'C')
        legacy = sum(1 for r in rows if r.get('status') == 'L')
        deprecated = sum(1 for r in rows if r.get('status') == 'D')
        if current or legacy or deprecated:
            print(f"  Current: {current}, Legacy: {legacy}, Deprecated: {deprecated}")

    # Source distribution
    print("\nSource Distribution:")
    source_count = {}
    for rows in all_rows.values():
        for row in rows:
            source = row.get('source', 'unknown')
            source_count[source] = source_count.get(source, 0) + 1

    for source in sorted(source_count.keys()):
        print(f"  {source}: {source_count[source]}")

def main():
    csv_files = [
        '/home/yfedoseev/projects/modelsuite/data/models/aggregators/openrouter.csv',
        '/home/yfedoseev/projects/modelsuite/data/models/aggregators/bedrock.csv',
        '/home/yfedoseev/projects/modelsuite/data/models/core/latest_releases.csv'
    ]

    generate_report(csv_files)

if __name__ == '__main__':
    main()
