#!/usr/bin/env python3
"""
Fetch Stability AI models.
Stability AI provides image generation models (Stable Diffusion, SD3, etc.)
"""

import csv
import sys
from datetime import datetime

STABILITY_MODELS = [
    # Ultra Models
    {
        "id": "stable-image-ultra",
        "alias": "stable-ultra",
        "name": "Stable Image Ultra",
        "context_window": 10000,
        "max_output": 0,
        "input_price": 0.08,
        "output_price": "-",
        "capabilities": "I",
        "description": "Highest quality image generation, $0.08/image"
    },
    # SD3.5 Series
    {
        "id": "sd3.5-large",
        "alias": "sd3.5-large",
        "name": "SD 3.5 Large",
        "context_window": 10000,
        "max_output": 0,
        "input_price": 0.065,
        "output_price": "-",
        "capabilities": "I",
        "description": "SD 3.5 Large, $0.065/image"
    },
    {
        "id": "sd3.5-large-turbo",
        "alias": "sd3.5-turbo",
        "name": "SD 3.5 Large Turbo",
        "context_window": 10000,
        "max_output": 0,
        "input_price": 0.04,
        "output_price": "-",
        "capabilities": "I",
        "description": "Fast SD 3.5, $0.04/image"
    },
    {
        "id": "sd3.5-medium",
        "alias": "sd3.5-medium",
        "name": "SD 3.5 Medium",
        "context_window": 10000,
        "max_output": 0,
        "input_price": 0.035,
        "output_price": "-",
        "capabilities": "I",
        "description": "SD 3.5 Medium, $0.035/image"
    },
    # SD3 Series
    {
        "id": "sd3-large",
        "alias": "sd3-large",
        "name": "SD 3 Large",
        "context_window": 10000,
        "max_output": 0,
        "input_price": 0.065,
        "output_price": "-",
        "capabilities": "I",
        "description": "SD 3 Large, $0.065/image"
    },
    {
        "id": "sd3-large-turbo",
        "alias": "sd3-turbo",
        "name": "SD 3 Large Turbo",
        "context_window": 10000,
        "max_output": 0,
        "input_price": 0.04,
        "output_price": "-",
        "capabilities": "I",
        "description": "Fast SD 3, $0.04/image"
    },
    {
        "id": "sd3-medium",
        "alias": "sd3-medium",
        "name": "SD 3 Medium",
        "context_window": 10000,
        "max_output": 0,
        "input_price": 0.035,
        "output_price": "-",
        "capabilities": "I",
        "description": "SD 3 Medium, $0.035/image"
    },
    # Core Model
    {
        "id": "stable-image-core",
        "alias": "stable-core",
        "name": "Stable Image Core",
        "context_window": 10000,
        "max_output": 0,
        "input_price": 0.03,
        "output_price": "-",
        "capabilities": "I",
        "description": "Efficient core model, $0.03/image"
    },
    # SDXL
    {
        "id": "stable-diffusion-xl-1024-v1-0",
        "alias": "sdxl-1.0",
        "name": "SDXL 1.0",
        "context_window": 77,
        "max_output": 0,
        "input_price": 0.002,
        "output_price": "-",
        "capabilities": "I",
        "description": "SDXL, $0.002-0.006 depending on steps"
    },
    {
        "id": "stable-diffusion-xl-1024-v0-9",
        "alias": "sdxl-0.9",
        "name": "SDXL 0.9",
        "context_window": 77,
        "max_output": 0,
        "input_price": 0.002,
        "output_price": "-",
        "capabilities": "I",
        "description": "SDXL 0.9 beta"
    },
    # SD 1.6
    {
        "id": "stable-diffusion-v1-6",
        "alias": "sd-1.6",
        "name": "SD 1.6",
        "context_window": 77,
        "max_output": 0,
        "input_price": 0.002,
        "output_price": "-",
        "capabilities": "I",
        "description": "Legacy SD 1.6"
    },
    # Edit/Control
    {
        "id": "stable-diffusion-inpaint",
        "alias": "sd-inpaint",
        "name": "SD Inpaint",
        "context_window": 77,
        "max_output": 0,
        "input_price": 0.002,
        "output_price": "-",
        "capabilities": "I",
        "description": "Image inpainting"
    },
    {
        "id": "stable-image-control",
        "alias": "stable-control",
        "name": "Stable Image Control",
        "context_window": 10000,
        "max_output": 0,
        "input_price": 0.04,
        "output_price": "-",
        "capabilities": "I",
        "description": "Controlled image generation"
    },
    # Upscale
    {
        "id": "stable-image-upscale",
        "alias": "stable-upscale",
        "name": "Stable Image Upscale",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.02,
        "output_price": "-",
        "capabilities": "I",
        "description": "Image upscaling, $0.02/image"
    },
    {
        "id": "stable-fast-upscale",
        "alias": "fast-upscale",
        "name": "Fast Upscale",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.01,
        "output_price": "-",
        "capabilities": "I",
        "description": "Fast upscaling, $0.01/image"
    },
    # Video
    {
        "id": "stable-video-diffusion",
        "alias": "svd",
        "name": "Stable Video Diffusion",
        "context_window": 0,
        "max_output": 0,
        "input_price": 0.20,
        "output_price": "-",
        "capabilities": "D",
        "description": "Video generation"
    },
]

def process_models() -> list:
    """Process Stability AI models into CSV format."""
    rows = []

    for model in STABILITY_MODELS:
        output_price = model['output_price']
        output_str = f"{output_price:.6f}" if isinstance(output_price, (int, float)) else output_price

        row = {
            'id': f"stability/{model['id']}",
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
            'source': 'stability',
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
    print("Processing Stability AI models...")
    rows = process_models()

    output_file = '/home/yfedoseev/projects/modelsuite/data/models/specialized/stability.csv'
    save_csv(rows, output_file)

    print(f"\n=== Summary ===")
    print(f"Total models: {len(rows)}")

if __name__ == '__main__':
    main()
