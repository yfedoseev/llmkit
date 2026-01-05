#!/usr/bin/env python3
"""
Example: Image Generation with ModelSuite Python bindings

This example demonstrates how to use ModelSuite to generate images
from text prompts using various providers (FAL AI, Recraft, Stability AI, Runway ML).

Providers:
- FAL AI: FLUX and Stable Diffusion 3 models
- Recraft: Vector and design-focused image generation
- Stability AI: SDXL and other Stable Diffusion models
- RunwayML: Various image generation models
"""

import os
import sys
import time
from pathlib import Path

# Add the package to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modelsuite-python'))

from modelsuite import (
    ModelSuiteClient,
    ImageGenerationRequest,
    ImageSize,
    ImageQuality,
    ImageStyle,
    ImageFormat,
)


def main():
    """Main example function."""
    print("🎨 ModelSuite Image Generation Example")
    print("=" * 50)

    # Initialize client from environment
    # Requires provider-specific API keys (e.g., OPENAI_API_KEY for DALL-E)
    client = ModelSuiteClient.from_env()
    print("✓ Client initialized from environment")

    # Example 1: Simple image generation request
    print("\n--- Example 1: Simple Image Generation ---")
    req = ImageGenerationRequest("dall-e-3", "A serene landscape with mountains")
    print(f"Model: dall-e-3")
    print(f"Prompt: {req.prompt}")

    response = client.generate_image(req)
    print(f"Generated {response.count} image(s)")
    if response.first():
        print(f"Image URL: {response.first().url}")

    # Example 2: Image generation with specific quality
    print("\n--- Example 2: High-Quality Image ---")
    req2 = (
        ImageGenerationRequest("dall-e-3", "A portrait of a Renaissance noble")
        .with_quality(ImageQuality.Hd)
        .with_style(ImageStyle.Vivid)
    )
    print(f"Prompt: {req2.prompt}")
    print(f"Quality: HD")
    print(f"Style: Vivid")

    response2 = client.generate_image(req2)
    print(f"Generated {response2.count} image(s)")
    print(f"Created at: {response2.created}")

    # Example 3: Multiple images with specific dimensions
    print("\n--- Example 3: Multiple Images with Custom Size ---")
    req3 = (
        ImageGenerationRequest("fal-ai/flux/dev", "Abstract geometric patterns")
        .with_n(2)
        .with_size(ImageSize.Landscape1792x1024)
    )
    print(f"Model: fal-ai/flux/dev")
    print(f"Prompt: {req3.prompt}")
    print(f"Number of images: 2")
    print(f"Size: 1792x1024 (landscape)")

    response3 = client.generate_image(req3)
    print(f"Generated {response3.count} image(s)")
    print(f"Total size: {response3.total_size} bytes")

    # Example 4: Base64 encoded image output
    print("\n--- Example 4: Base64 Encoded Output ---")
    req4 = (
        ImageGenerationRequest("dall-e-3", "A futuristic cityscape at night")
        .with_format(ImageFormat.B64Json)
    )
    print(f"Prompt: {req4.prompt}")
    print(f"Format: Base64-encoded")

    response4 = client.generate_image(req4)
    print(f"Generated {response4.count} image(s)")
    if response4.first() and response4.first().b64_json:
        print(f"B64 data size: {len(response4.first().b64_json)} characters")

    # Example 5: Stability AI with negative prompt
    print("\n--- Example 5: Using Negative Prompts ---")
    req5 = (
        ImageGenerationRequest(
            "stability-ai/stable-diffusion-xl", "A serene ocean landscape"
        )
        .with_negative_prompt("blurry, watermark, low quality")
        .with_n(1)
    )
    print(f"Model: stability-ai/stable-diffusion-xl")
    print(f"Prompt: {req5.prompt}")
    print(f"Negative prompt: {req5.negative_prompt}")

    response5 = client.generate_image(req5)
    print(f"Generated {response5.count} image(s)")

    # Example 6: Portrait size image
    print("\n--- Example 6: Portrait Orientation ---")
    req6 = (
        ImageGenerationRequest(
            "dall-e-3", "A detailed portrait of a character from a fantasy novel"
        )
        .with_size(ImageSize.Portrait1024x1792)
        .with_quality(ImageQuality.Hd)
    )
    print(f"Prompt: {req6.prompt}")
    print(f"Size: 1024x1792 (portrait)")
    print(f"Quality: HD")

    response6 = client.generate_image(req6)
    print(f"Generated {response6.count} image(s)")

    # Example 7: Recraft (vector/design focused)
    print("\n--- Example 7: Vector/Design Generation (Recraft) ---")
    req7 = (
        ImageGenerationRequest(
            "recraft-v3", "A minimalist logo for a tech startup"
        )
        .with_n(1)
    )
    print(f"Model: recraft-v3 (vector generation)")
    print(f"Prompt: {req7.prompt}")

    response7 = client.generate_image(req7)
    print(f"Generated {response7.count} image(s)")

    # Example 8: Different image sizes
    print("\n--- Example 8: Various Image Sizes ---")
    sizes = [
        ("Square (256x256)", ImageSize.Square256),
        ("Square (512x512)", ImageSize.Square512),
        ("Square (1024x1024)", ImageSize.Square1024),
        ("Portrait (1024x1792)", ImageSize.Portrait1024x1792),
        ("Landscape (1792x1024)", ImageSize.Landscape1792x1024),
    ]

    for size_name, size in sizes[:2]:  # Just show 2 for demo
        req = ImageGenerationRequest("dall-e-3", "A test image").with_size(size)
        response = client.generate_image(req)
        print(f"✓ {size_name}: {response.count} image(s)")

    # Example 9: Batch generation with different prompts
    print("\n--- Example 9: Batch Generation ---")
    prompts = [
        ("A red apple on a white table", "dall-e-3"),
        ("A galaxy seen from space", "fal-ai/flux/dev"),
        ("A steampunk airship", "stability-ai/stable-diffusion-xl"),
    ]

    print(f"Generating {len(prompts)} images with different prompts...")
    responses = []
    for prompt, model in prompts:
        req = ImageGenerationRequest(model, prompt)
        response = client.generate_image(req)
        responses.append(response)
        print(f"  ✓ {prompt[:40]}... ({model})")

    print(f"Total images generated: {sum(r.count for r in responses)}")

    # Example 10: Image with seed for reproducibility
    print("\n--- Example 10: Reproducible Generation (with seed) ---")
    req10a = (
        ImageGenerationRequest("stability-ai/stable-diffusion-xl", "A cat sitting")
        .with_seed(12345)
    )
    req10b = (
        ImageGenerationRequest("stability-ai/stable-diffusion-xl", "A cat sitting")
        .with_seed(12345)
    )

    print(f"Generating two images with the same seed (12345)...")
    print(f"Note: Same seed should produce identical or very similar images")
    response10a = client.generate_image(req10a)
    response10b = client.generate_image(req10b)
    print(f"✓ Generated image 1: {response10a.first().url if response10a.first() else 'N/A'}")
    print(f"✓ Generated image 2: {response10b.first().url if response10b.first() else 'N/A'}")

    print("\n" + "=" * 50)
    print("✓ Image generation examples completed!")


def save_image_to_file(response, filename: str) -> None:
    """
    Save image from response to a file.

    Args:
        response: ImageGenerationResponse object
        filename: Output filename
    """
    if response.first() and response.first().b64_json:
        import base64

        image_data = base64.b64decode(response.first().b64_json)
        with open(filename, "wb") as f:
            f.write(image_data)
        print(f"✓ Image saved to {filename}")
    elif response.first() and response.first().url:
        print(f"ℹ Image available at: {response.first().url}")
        print(f"Use a download tool or requests library to save:")
        print(f"  import requests")
        print(f"  response = requests.get('{response.first().url}')")
        print(f"  with open('{filename}', 'wb') as f:")
        print(f"      f.write(response.content)")
    else:
        print("✗ No image data available in response")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
