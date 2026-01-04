"""
Vision / Image Analysis Example

Demonstrates image input capabilities with LLMKit.

Requirements:
- Set OPENAI_API_KEY environment variable (GPT-4o has vision)
- Optionally have a local image file to test

Run:
    python 04_vision.py
"""

import base64
from pathlib import Path
from modelsuite import LLMKitClient, Message, CompletionRequest, ContentBlock


def analyze_image_from_url():
    """Analyze an image from a URL."""
    client = LLMKitClient.from_env()

    # Create a message with an image URL
    # Note: Not all providers support URL-based images
    message = Message.user_with_content([
        ContentBlock.text("What do you see in this image? Describe it briefly."),
        ContentBlock.image_url(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/"
            "Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg"
        ),
    ])

    # Use "provider/model" format for explicit provider routing
    request = CompletionRequest(
        model="openai/gpt-4o",
        messages=[message],
        max_tokens=500,
    )

    print("Analyzing image from URL...")
    response = client.complete(request)
    print(f"\nDescription:\n{response.text_content()}")


def analyze_local_image(image_path: str):
    """Analyze a local image file."""
    client = LLMKitClient.from_env()

    # Read and encode the image
    path = Path(image_path)
    if not path.exists():
        print(f"Image not found: {image_path}")
        print("Creating a simple test with a placeholder...")

        # For demo, we'll skip if no image
        return

    # Detect media type
    suffix = path.suffix.lower()
    media_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    media_type = media_types.get(suffix, "image/png")

    # Read and encode
    with open(path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    # Create message with image
    message = Message.user_with_content([
        ContentBlock.text("Analyze this image. What's in it?"),
        ContentBlock.image(media_type, image_data),
    ])

    request = CompletionRequest(
        model="openai/gpt-4o",
        messages=[message],
        max_tokens=500,
    )

    print(f"Analyzing local image: {image_path}")
    response = client.complete(request)
    print(f"\nAnalysis:\n{response.text_content()}")


def multi_image_comparison():
    """Compare multiple images."""
    client = LLMKitClient.from_env()

    # Example with multiple image URLs
    message = Message.user_with_content([
        ContentBlock.text(
            "I'm going to show you two images. "
            "Please compare them and describe the differences."
        ),
        ContentBlock.image_url(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/"
            "Cat03.jpg/120px-Cat03.jpg"
        ),
        ContentBlock.image_url(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/"
            "YellowLabradorLooking_new.jpg/120px-YellowLabradorLooking_new.jpg"
        ),
    ])

    request = CompletionRequest(
        model="openai/gpt-4o",
        messages=[message],
        max_tokens=500,
    )

    print("Comparing two images...")
    response = client.complete(request)
    print(f"\nComparison:\n{response.text_content()}")


def main():
    print("=" * 50)
    print("Vision Example 1: Analyze image from URL")
    print("=" * 50)
    analyze_image_from_url()

    print("\n" + "=" * 50)
    print("Vision Example 2: Compare multiple images")
    print("=" * 50)
    multi_image_comparison()

    # Uncomment to test local image:
    # print("\n" + "=" * 50)
    # print("Vision Example 3: Analyze local image")
    # print("=" * 50)
    # analyze_local_image("path/to/your/image.png")


if __name__ == "__main__":
    main()
