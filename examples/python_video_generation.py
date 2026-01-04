#!/usr/bin/env python3
"""
Example: Video Generation with LLMKit Python bindings

This example demonstrates how to use LLMKit to generate videos
from text prompts using various providers (Runware, DiffusionRouter).

Providers:
- Runware: Multiple video models (Runway Gen-3, Kling, Pika, Hailuo, Leonardo)
- DiffusionRouter: Stable Diffusion Video (coming February 2026)
"""

import os
import sys
import time
from pathlib import Path

# Add the package to path for local development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'llmkit-python'))

from llmkit import LLMKitClient, VideoGenerationRequest, VideoModel


def main():
    """Main example function."""
    print("🎬 LLMKit Video Generation Example")
    print("=" * 50)

    # Initialize client from environment
    # Requires LLMKIT_API_KEY environment variable
    client = LLMKitClient.from_env()
    print("✓ Client initialized from environment")

    # Example 1: Simple video generation request
    print("\n--- Example 1: Simple Video Generation ---")
    req = VideoGenerationRequest("A serene landscape with mountains and a clear sky")
    print(f"Prompt: {req.prompt}")

    response = client.generate_video(req)
    print(f"Video URL: {response.video_url}")
    print(f"Format: {response.format}")
    print(f"Task ID: {response.task_id}")
    print(f"Status: {response.status}")

    # Example 2: Video generation with specific model
    print("\n--- Example 2: Video with Specific Model ---")
    req2 = (
        VideoGenerationRequest("An abstract geometric animation with rotating shapes")
        .with_model("kling-2.0")
    )
    print(f"Prompt: {req2.prompt}")
    print(f"Model: Kling 2.0")

    response2 = client.generate_video(req2)
    print(f"Video URL: {response2.video_url}")
    print(f"Status: {response2.status}")

    # Example 3: Fully configured video generation request
    print("\n--- Example 3: Fully Configured Request ---")
    req3 = (
        VideoGenerationRequest("A person dancing in a vibrant neon club")
        .with_model("runway-gen-4")
        .with_duration(10)
        .with_width(1280)
        .with_height(720)
    )
    print(f"Prompt: {req3.prompt}")
    print(f"Model: Runway Gen-4")
    print(f"Duration: 10 seconds")
    print(f"Resolution: 1280x720")

    response3 = client.generate_video(req3)
    print(f"Video URL: {response3.video_url}")
    print(f"Duration: {response3.duration}s")
    print(f"Resolution: {response3.width}x{response3.height}")
    print(f"Size: {response3.size} bytes")
    print(f"Status: {response3.status}")

    # Example 4: Polling for async video generation
    print("\n--- Example 4: Polling for Task Completion ---")
    req4 = VideoGenerationRequest(
        "A spaceship flying through a wormhole in space"
    ).with_model("pika-1.0")

    response4 = client.generate_video(req4)
    print(f"Task ID: {response4.task_id}")
    print(f"Initial status: {response4.status}")

    # Simulate polling (in production, use actual polling API)
    print("Simulating polling...")
    for i in range(5):
        time.sleep(1)
        print(f"  Poll {i+1}: Status would be checked via API")
        if i == 4:
            print("  → Task would be completed")

    # Example 5: Multiple video generation requests
    print("\n--- Example 5: Multiple Video Generations ---")
    prompts = [
        "A cat playing with yarn in a cozy living room",
        "Ocean waves crashing on a rocky beach at sunset",
        "Urban street art mural with vibrant colors",
    ]

    models = ["runway-gen-4", "kling-2.0", "pika-1.0"]

    print(f"Generating {len(prompts)} videos with different models...")
    responses = []

    for prompt, model in zip(prompts, models):
        req = VideoGenerationRequest(prompt).with_model(model)
        response = client.generate_video(req)
        responses.append(response)
        print(f"  ✓ Generated: {prompt[:40]}... ({model})")
        print(f"    → Task ID: {response.task_id}")

    # Example 6: Video generation with high quality settings
    print("\n--- Example 6: High Quality Video ---")
    req6 = (
        VideoGenerationRequest("Professional product demonstration video")
        .with_model("leonardo-ultra")
        .with_duration(15)
        .with_width(1920)
        .with_height(1080)
    )
    print(f"Prompt: {req6.prompt}")
    print("Settings:")
    print("  - Model: Leonardo Ultra (highest quality)")
    print("  - Duration: 15 seconds")
    print("  - Resolution: 1920x1080 (Full HD)")

    response6 = client.generate_video(req6)
    print(f"Video URL: {response6.video_url}")
    print(f"Quality: high")
    print(f"Size: {response6.size} bytes")

    # Example 7: Video generation for different use cases
    print("\n--- Example 7: Different Use Cases ---")
    use_cases = [
        ("Marketing", "Product launch celebration with confetti and champagne"),
        ("Education", "Animated physics demonstration of gravitational forces"),
        ("Entertainment", "Sci-fi spaceship battle with explosions and lasers"),
        ("Art", "Abstract surreal dreamscape with melting clocks"),
    ]

    for use_case, prompt in use_cases:
        req = VideoGenerationRequest(prompt)
        response = client.generate_video(req)
        print(f"✓ {use_case}: {prompt[:40]}...")
        print(f"  → Status: {response.status}")

    print("\n" + "=" * 50)
    print("✓ Video generation examples completed!")
    print("\nNote: In production, use polling to wait for async video generation:")
    print("  while response.status != 'completed':")
    print("    response = client.poll_video(response.task_id)")
    print("    time.sleep(5)")


def save_video_to_file(response, filename: str) -> None:
    """
    Save video from response to a file.

    Args:
        response: VideoGenerationResponse object
        filename: Output filename
    """
    if response.video_bytes:
        with open(filename, "wb") as f:
            f.write(response.video_bytes)
        print(f"✓ Video saved to {filename}")
    elif response.video_url:
        print(f"ℹ Video available at: {response.video_url}")
        print(f"Use a download tool or requests library to save:")
        print(f"  import requests")
        print(f"  response = requests.get('{response.video_url}')")
        print(f"  with open('{filename}', 'wb') as f:")
        print(f"      f.write(response.content)")
    else:
        print("✗ No video data available in response")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        sys.exit(1)
