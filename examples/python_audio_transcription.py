#!/usr/bin/env python3
"""
ModelSuite Audio Transcription Example

This example demonstrates how to use the ModelSuite Python bindings to transcribe
audio files using various providers (Deepgram, AssemblyAI).

Requirements:
- modelsuite Python bindings
- DEEPGRAM_API_KEY or ASSEMBLYAI_API_KEY environment variable

Usage:
    python python_audio_transcription.py <audio_file>
"""

import sys
import os
from pathlib import Path

try:
    from modelsuite import (
        ModelSuiteClient,
        TranscriptionRequest,
        TranscribeOptions,
        DeepgramVersion,
    )
except ImportError:
    print("Error: modelsuite package not found. Please install it first.")
    sys.exit(1)


def transcribe_with_deepgram(client: ModelSuiteClient, audio_bytes: bytes) -> None:
    """Transcribe audio using Deepgram provider."""
    print("\n" + "=" * 70)
    print("TRANSCRIPTION WITH DEEPGRAM")
    print("=" * 70)

    # Create transcription request
    request = TranscriptionRequest(audio_bytes)
    request = request.with_model("nova-3")

    try:
        # Transcribe audio
        response = client.transcribe_audio(request)

        # Display results
        print(f"\nTranscript:")
        print(f"  {response.transcript}")
        print(f"\nConfidence: {response.confidence:.2%}")
        print(f"Duration: {response.duration:.2f}s")
        print(f"Word count: {response.word_count}")

        # Display word-level details if available
        if response.words:
            print(f"\nWord-level details (first 5 words):")
            for word in response.words[:5]:
                print(
                    f"  '{word.word}': {word.start:.2f}s-{word.end:.2f}s "
                    f"(confidence: {word.confidence:.2%})"
                )

    except Exception as e:
        print(f"Error during transcription: {e}")


def main() -> None:
    """Main example function."""
    # Check arguments
    if len(sys.argv) < 2:
        print("Usage: python python_audio_transcription.py <audio_file>")
        print("\nExample:")
        print("  python python_audio_transcription.py speech.wav")
        sys.exit(1)

    audio_file = Path(sys.argv[1])

    # Check if file exists
    if not audio_file.exists():
        print(f"Error: Audio file not found: {audio_file}")
        sys.exit(1)

    # Check file size (warn if very large)
    file_size = audio_file.stat().st_size
    if file_size > 50 * 1024 * 1024:  # 50 MB
        print(f"Warning: Large audio file ({file_size / 1024 / 1024:.1f} MB)")
        print("         Transcription may take longer")

    # Read audio file
    print(f"Reading audio file: {audio_file}")
    with open(audio_file, "rb") as f:
        audio_bytes = f.read()

    print(f"Audio file size: {len(audio_bytes) / 1024:.1f} KB")

    # Initialize ModelSuite client
    print("\nInitializing ModelSuite client...")
    try:
        client = ModelSuiteClient.from_env()
    except Exception as e:
        print(f"Error initializing client: {e}")
        print("\nMake sure you have set the required environment variables:")
        print("  - DEEPGRAM_API_KEY (for Deepgram provider)")
        print("  - ASSEMBLYAI_API_KEY (for AssemblyAI provider)")
        sys.exit(1)

    # Transcribe with Deepgram
    transcribe_with_deepgram(client, audio_bytes)

    print("\n" + "=" * 70)
    print("Transcription complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
