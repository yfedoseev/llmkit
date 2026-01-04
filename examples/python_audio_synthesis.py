#!/usr/bin/env python3
"""
LLMKit Audio Synthesis Example

This example demonstrates how to use the LLMKit Python bindings to synthesize
speech from text using ElevenLabs provider.

Requirements:
- llmkit Python bindings
- ELEVENLABS_API_KEY environment variable
- Speaker voice ID (available from ElevenLabs API)

Usage:
    python python_audio_synthesis.py "Your text here" [output_file]
"""

import sys
import os
from pathlib import Path

try:
    from modelsuite import (
        LLMKitClient,
        SynthesisRequest,
        SynthesizeOptions,
        VoiceSettings,
        LatencyMode,
    )
except ImportError:
    print("Error: llmkit package not found. Please install it first.")
    sys.exit(1)


def synthesize_speech(client: LLMKitClient, text: str, output_file: Path) -> None:
    """Synthesize speech from text using ElevenLabs."""
    print("\n" + "=" * 70)
    print("TEXT-TO-SPEECH SYNTHESIS")
    print("=" * 70)

    print(f"\nInput text: {text}")

    # Create synthesis request
    request = SynthesisRequest(text)

    # Optionally set voice ID (example uses Rachel)
    # Voice IDs available: 21m00Tcm4TlvDq8ikWAM (Rachel), pNInY14gQrG92XwBIHVr, etc.
    request = request.with_voice("21m00Tcm4TlvDq8ikWAM")

    # Create synthesis options
    options = SynthesizeOptions()
    options = options.with_latency_mode(LatencyMode.Balanced)
    options = options.with_output_format("mp3_44100_64")

    # Optionally customize voice settings
    voice_settings = VoiceSettings(stability=0.5, similarity_boost=0.75)
    options = options.with_voice_settings(voice_settings)

    try:
        # Synthesize speech
        print("\nSynthesizing speech...")
        response = client.synthesize_speech(request)

        # Display results
        print(f"\nSynthesis complete!")
        print(f"  Format: {response.format}")
        print(f"  Size: {response.size} bytes")
        if response.duration:
            print(f"  Duration: {response.duration:.2f}s")

        # Save to file
        print(f"\nSaving audio to: {output_file}")
        with open(output_file, "wb") as f:
            f.write(response.audio_bytes)

        print(f"✓ Audio file saved successfully!")
        print(f"  File size: {output_file.stat().st_size / 1024:.1f} KB")

    except Exception as e:
        print(f"Error during synthesis: {e}")


def main() -> None:
    """Main example function."""
    # Parse arguments
    if len(sys.argv) < 2:
        print("Usage: python python_audio_synthesis.py 'Your text here' [output_file]")
        print("\nExamples:")
        print("  python python_audio_synthesis.py 'Hello, world!'")
        print("  python python_audio_synthesis.py 'Tell me a joke' output.mp3")
        sys.exit(1)

    text = sys.argv[1]
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("synthesized.mp3")

    # Check if output file already exists
    if output_file.exists():
        response = input(f"File {output_file} already exists. Overwrite? [y/N]: ")
        if response.lower() != "y":
            print("Cancelled.")
            sys.exit(0)

    # Initialize LLMKit client
    print("Initializing LLMKit client...")
    try:
        client = LLMKitClient.from_env()
    except Exception as e:
        print(f"Error initializing client: {e}")
        print("\nMake sure you have set the required environment variable:")
        print("  - ELEVENLABS_API_KEY")
        sys.exit(1)

    # Synthesize speech
    synthesize_speech(client, text, output_file)

    print("\n" + "=" * 70)
    print("Synthesis example complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
