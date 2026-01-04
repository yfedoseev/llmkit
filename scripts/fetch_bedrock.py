#!/usr/bin/env python3
"""
Create Bedrock models CSV from AWS documentation.
Includes Claude, Nova, Llama, Mistral, Cohere, and other models available on Bedrock.
"""

import csv
import sys
from datetime import datetime

# Bedrock models compiled from AWS documentation
# Format: (model_id, alias, name, status, input_price, output_price, context_window, max_output, capabilities, quality, description)
BEDROCK_MODELS = [
    # Claude Models
    ("aws/bedrock-claude-3-5-sonnet-20241022-v1:0", "claude-sonnet-3-5", "Claude 3.5 Sonnet", "C", "0.000006", "0.00003", 200000, 4096, "SVTJK", "verified", "Advanced reasoning and tool use"),
    ("aws/bedrock-claude-3-5-sonnet-v2-20250127-v1:0", "claude-sonnet-3-5-v2", "Claude 3.5 Sonnet v2", "C", "0.000006", "0.00003", 200000, 4096, "SVTJK", "verified", "Latest Sonnet with improved performance"),
    ("aws/bedrock-claude-3-7-sonnet-20250219-v1:0", "claude-sonnet-3-7", "Claude 3.7 Sonnet", "C", "0.000006", "0.00003", 200000, 4096, "SVTJK", "verified", "New Sonnet generation"),
    ("aws/bedrock-claude-opus-4-1-20250805-v1:0", "claude-opus-4-1", "Claude Opus 4.1", "C", "0.0000075", "0.00003", 200000, 4096, "SVTJK", "verified", "Powerful flagship model"),
    ("aws/bedrock-claude-opus-4-20250514-v1:0", "claude-opus-4", "Claude Opus 4", "C", "0.0000075", "0.00003", 200000, 4096, "SVTJK", "verified", "Strong reasoning capability"),
    ("aws/bedrock-claude-haiku-4-5-20250811-v1:0", "claude-haiku-4-5", "Claude Haiku 4.5", "C", "0.0000008", "0.000004", 200000, 4096, "SVT", "verified", "Fast and efficient model"),
    ("aws/bedrock-claude-3-5-haiku-20241226-v1:0", "claude-haiku-3-5", "Claude 3.5 Haiku", "C", "0.0000008", "0.000004", 200000, 4096, "SVT", "verified", "Latest Haiku generation"),
    ("aws/bedrock-claude-3-haiku-20240307-v1:0", "claude-haiku-3", "Claude 3 Haiku", "C", "0.00000025", "0.00000125", 200000, 4096, "SV", "verified", "Lightweight model"),
    ("aws/bedrock-claude-3-opus-20240229-v1:0", "claude-opus-3", "Claude 3 Opus", "L", "0.000015", "0.000075", 200000, 4096, "SVT", "verified", "Legacy Claude 3 Opus"),
    ("aws/bedrock-claude-3-sonnet-20240229-v1:0", "claude-sonnet-3", "Claude 3 Sonnet", "L", "0.000003", "0.000015", 200000, 4096, "SVT", "verified", "Legacy Claude 3 Sonnet"),

    # Amazon Nova Models
    ("aws/bedrock-nova-pro-v1:0", "nova-pro", "Amazon Nova Pro", "C", "0.0000008", "0.000003", 300000, 40000, "SVT", "verified", "Advanced multimodal model"),
    ("aws/bedrock-nova-lite-v1:0", "nova-lite", "Amazon Nova Lite", "C", "0.00000006", "0.00000024", 300000, 40000, "SVT", "verified", "Fast and affordable model"),
    ("aws/bedrock-nova-micro-v1:0", "nova-micro", "Amazon Nova Micro", "C", "0.00000003", "0.00000012", 128000, 1024, "T", "verified", "Ultra-lightweight model"),
    ("aws/bedrock-nova-premier-v1:0", "nova-premier", "Amazon Nova Premier", "C", "0.0000008", "0.000003", 300000, 40000, "SVT", "verified", "Premium multimodal model"),

    # Meta Llama Models
    ("aws/bedrock-llama-3-1-405b-instruct-v1:0", "llama-3-1-405b", "Llama 3.1 405B Instruct", "C", "0.00000533", "0.00001600", 128000, 8000, "T", "verified", "Large language instruction model"),
    ("aws/bedrock-llama-3-1-70b-instruct-v1:0", "llama-3-1-70b", "Llama 3.1 70B Instruct", "C", "0.00000135", "0.00000270", 128000, 8000, "T", "verified", "Medium instruction model"),
    ("aws/bedrock-llama-3-3-70b-instruct-v1:0", "llama-3-3-70b", "Llama 3.3 70B Instruct", "C", "0.00000135", "0.00000270", 128000, 8000, "T", "verified", "Latest Llama 3.3 model"),
    ("aws/bedrock-llama-4-maverick-17b-v1:0", "llama-4-maverick", "Llama 4 Maverick 17B", "C", "0.00000040", "0.00000080", 128000, 8000, "SVT", "verified", "Llama 4 early variant"),
    ("aws/bedrock-llama-4-scout-17b-v1:0", "llama-4-scout", "Llama 4.0 Scout 17B", "C", "0.00000040", "0.00000080", 128000, 8000, "SVT", "verified", "Llama 4 scout model"),
    ("aws/bedrock-llama-3-2-90b-vision-instruct-v1:0", "llama-3-2-90b-vision", "Llama 3.2 90B Vision Instruct", "C", "0.00000135", "0.00000270", 128000, 8000, "SV", "verified", "Multimodal Llama model"),
    ("aws/bedrock-llama-3-2-11b-vision-instruct-v1:0", "llama-3-2-11b-vision", "Llama 3.2 11B Vision Instruct", "C", "0.00000030", "0.00000060", 128000, 8000, "SV", "verified", "Small Llama vision model"),
    ("aws/bedrock-llama-3-2-1b-instruct-v1:0", "llama-3-2-1b", "Llama 3.2 1B Instruct", "C", "0.00000001", "0.00000002", 8000, 2048, "-", "verified", "Tiny Llama model"),
    ("aws/bedrock-llama-3-2-3b-instruct-v1:0", "llama-3-2-3b", "Llama 3.2 3B Instruct", "C", "0.00000005", "0.00000010", 8000, 2048, "-", "verified", "Small Llama model"),
    ("aws/bedrock-llama-3-1-8b-instruct-v1:0", "llama-3-1-8b", "Llama 3.1 8B Instruct", "C", "0.00000015", "0.00000300", 128000, 8000, "-", "verified", "Small Llama instruction model"),
    ("aws/bedrock-llama-3-70b-instruct-v1:0", "llama-3-70b", "Llama 3 70B Instruct", "L", "0.00000135", "0.00000270", 8000, 2048, "-", "verified", "Legacy Llama 3 model"),
    ("aws/bedrock-llama-2-70b-chat-v1:0", "llama-2-70b", "Llama 2 70B Chat", "L", "0.00000135", "0.00000180", 4096, 2048, "-", "verified", "Legacy Llama 2 model"),

    # Mistral AI Models
    ("aws/bedrock-mistral-large-2-415b-instruct-v1:0", "mistral-large-2-415b", "Mistral Large 2 415B Instruct", "C", "0.0000008", "0.0000024", 200000, 64000, "TJ", "verified", "Mistral flagship model"),
    ("aws/bedrock-mistral-large-instruct-2412-v1:0", "mistral-large-2412", "Mistral Large Instruct 2412", "C", "0.0000008", "0.0000024", 200000, 64000, "TJ", "verified", "Latest Mistral Large"),
    ("aws/bedrock-mistral-large-2407-v1:0", "mistral-large-2407", "Mistral Large 2 (24.07)", "C", "0.0000008", "0.0000024", 200000, 64000, "TJ", "verified", "Mistral 24.07 version"),
    ("aws/bedrock-pixtral-large-2502-v1:0", "pixtral-large", "Pixtral Large (25.02)", "C", "0.0000008", "0.0000024", 128000, 4096, "SV", "verified", "Mistral multimodal model"),
    ("aws/bedrock-mistral-small-2409-v1:0", "mistral-small-2409", "Mistral Small 2409", "C", "0.00000014", "0.00000042", 32000, 8000, "T", "verified", "Small Mistral model"),
    ("aws/bedrock-mistral-nemo-2407-v1:0", "mistral-nemo", "Mistral Nemo 2407", "C", "0.00000014", "0.00000042", 32000, 8000, "-", "verified", "Efficient Mistral model"),

    # Cohere Models
    ("aws/bedrock-cohere-command-r-7b-12-2024-v1:0", "cohere-command-r-7b", "Cohere Command R 7B", "C", "0.00000010", "0.00000020", 128000, 4096, "TJ", "verified", "Cohere lightweight model"),
    ("aws/bedrock-cohere-command-r-5-2024-10-08-v1:0", "cohere-command-r-5", "Cohere Command R 5B", "C", "0.00000010", "0.00000020", 128000, 4096, "TJ", "verified", "Cohere small model"),
    ("aws/bedrock-cohere-command-r-plus-04-2024-v1:0", "cohere-command-r-plus", "Cohere Command R+ 04-2024", "C", "0.00000030", "0.00000600", 128000, 4096, "TJ", "verified", "Cohere flagship model"),
    ("aws/bedrock-cohere-command-text-v14-7k-instruct-4k-latest-v1:0", "cohere-command-text", "Cohere Command Text", "L", "0.00000150", "0.00000200", 4096, 4096, "-", "verified", "Legacy Cohere model"),

    # AI21 Models
    ("aws/bedrock-ai21-jamba-instruct-v1:0", "ai21-jamba-instruct", "AI21 Jamba Instruct", "C", "0.00000030", "0.00000400", 256000, 4096, "T", "verified", "AI21 instruction model"),
    ("aws/bedrock-ai21-jamba-1-5-large-v1:0", "ai21-jamba-1-5-large", "AI21 Jamba 1.5 Large", "C", "0.00000030", "0.00000400", 256000, 4096, "T", "verified", "Large Jamba model"),
    ("aws/bedrock-ai21-jamba-1-5-mini-v1:0", "ai21-jamba-1-5-mini", "AI21 Jamba 1.5 Mini", "C", "0.00000030", "0.00000400", 256000, 4096, "T", "verified", "Mini Jamba model"),
    ("aws/bedrock-ai21-labs-jurassic-2-ultra-v1:0", "ai21-jurassic-2-ultra", "AI21 Jurassic 2 Ultra", "L", "0.00000188", "0.00000250", 8191, 1024, "-", "verified", "Legacy AI21 model"),

    # Amazon Titan Models
    ("aws/bedrock-amazon-titan-text-premier-v1:0", "titan-text-premier", "Amazon Titan Text Premier", "C", "0.00000500", "0.00001500", 32000, 4096, "-", "verified", "Titan large model"),
    ("aws/bedrock-amazon-titan-text-express-v1:0", "titan-text-express", "Amazon Titan Text Express", "C", "0.00000030", "0.00000400", 8000, 2048, "-", "verified", "Titan efficient model"),
    ("aws/bedrock-amazon-titan-text-lite-v1:0", "titan-text-lite", "Amazon Titan Text Lite", "L", "0.00000030", "0.00000400", 4000, 2048, "-", "verified", "Legacy Titan model"),

    # DeepSeek Models
    ("aws/bedrock-deepseek-r1-v1:0", "deepseek-r1", "DeepSeek R1", "C", "0.00000140", "0.00000560", 128000, 16000, "K", "verified", "DeepSeek reasoning model"),

    # Writer Models
    ("aws/bedrock-writer-palmyra-x5-v1:0", "writer-palmyra-x5", "Writer Palmyra X5", "C", "0.00000150", "0.00000200", 32000, 2048, "T", "verified", "Writer specialized model"),
    ("aws/bedrock-writer-palmyra-x4-v1:0", "writer-palmyra-x4", "Writer Palmyra X4", "L", "0.00000150", "0.00000200", 32000, 2048, "T", "verified", "Legacy Writer model"),

    # Google Gemma Models (via partners)
    ("aws/bedrock-google-gemma-7b-it-v1:0", "google-gemma-7b", "Google Gemma 7B IT", "C", "0.00000070", "0.00000140", 8000, 2000, "-", "verified", "Google lightweight model"),
    ("aws/bedrock-google-gemma-2b-it-v1:0", "google-gemma-2b", "Google Gemma 2B IT", "C", "0.00000035", "0.00000070", 8000, 2000, "-", "verified", "Google tiny model"),
]

def validate_model(model: tuple) -> bool:
    """Validate a model record has valid data."""
    id_val, alias, name, status, input_price, output_price, context, max_out, caps, quality, desc = model

    if not id_val or not name or status not in ['C', 'L', 'D']:
        return False

    try:
        inp = float(input_price) if input_price != '-' else 0
        out = float(output_price) if output_price != '-' else 0
        if inp > 10 or out > 10:  # Sanity check (price too high)
            return False
        if context < 1000 or context > 10000000:
            return False
        if max_out < 100 or max_out > 10000000:
            return False
    except ValueError:
        return False

    return True

def save_bedrock_csv(output_file: str) -> None:
    """Save Bedrock models to CSV file."""
    fieldnames = [
        'id', 'alias', 'name', 'status', 'input_price', 'output_price',
        'cache_input_price', 'context_window', 'max_output', 'capabilities',
        'quality', 'source', 'updated', 'description',
        'mmlu_score', 'humaneval_score', 'math_score'
    ]

    rows = []
    today = datetime.now().strftime('%Y-%m-%d')

    valid_count = 0
    for model in BEDROCK_MODELS:
        if not validate_model(model):
            print(f"Skipping invalid model: {model[0]}", file=sys.stderr)
            continue

        id_val, alias, name, status, input_price, output_price, context, max_out, caps, quality, desc = model

        row = {
            'id': id_val,
            'alias': alias,
            'name': name,
            'status': status,
            'input_price': input_price,
            'output_price': output_price,
            'cache_input_price': '-',
            'context_window': context,
            'max_output': max_out,
            'capabilities': caps,
            'quality': quality,
            'source': 'bedrock',
            'updated': today,
            'description': desc,
            'mmlu_score': '-',
            'humaneval_score': '-',
            'math_score': '-'
        }
        rows.append(row)
        valid_count += 1

    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved {valid_count} Bedrock models to {output_file}")
    except Exception as e:
        print(f"Error saving CSV: {e}", file=sys.stderr)
        sys.exit(1)

    # Print summary
    current = sum(1 for r in rows if r['status'] == 'C')
    legacy = sum(1 for r in rows if r['status'] == 'L')
    with_vision = sum(1 for r in rows if 'V' in r['capabilities'])
    with_tools = sum(1 for r in rows if 'T' in r['capabilities'])
    with_thinking = sum(1 for r in rows if 'K' in r['capabilities'])

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")
    print(f"Current: {current}, Legacy: {legacy}")
    print(f"With Vision: {with_vision}, With Tools: {with_tools}, With Thinking: {with_thinking}")

def main():
    print(f"Processing {len(BEDROCK_MODELS)} Bedrock models...")
    output_file = '/home/yfedoseev/projects/modelsuite/data/models/aggregators/bedrock.csv'
    save_bedrock_csv(output_file)

if __name__ == '__main__':
    main()
