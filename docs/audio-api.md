# ModelSuite Audio API Documentation

## Overview

ModelSuite provides unified access to multiple audio providers through a single, consistent API. This documentation covers:

- **Speech-to-Text (STT)**: Convert audio to text using Deepgram and AssemblyAI
- **Text-to-Speech (TTS)**: Convert text to audio using ElevenLabs

## Quick Start

### Python

```python
from modelsuite import ModelSuiteClient, TranscriptionRequest, SynthesisRequest

# Initialize client
client = ModelSuiteClient.from_env()

# Transcribe audio
with open("audio.wav", "rb") as f:
    audio_bytes = f.read()

request = TranscriptionRequest(audio_bytes)
response = client.transcribe_audio(request)
print(response.transcript)

# Synthesize speech
request = SynthesisRequest("Hello, world!")
response = client.synthesize_speech(request)
with open("output.mp3", "wb") as f:
    f.write(response.audio_bytes)
```

### TypeScript

```typescript
import { ModelSuiteClient, TranscriptionRequest, SynthesisRequest } from 'modelsuite';
import fs from 'fs';

// Initialize client
const client = ModelSuiteClient.fromEnv();

// Transcribe audio
const audioBytes = fs.readFileSync('audio.wav');
const request = new TranscriptionRequest(audioBytes);
const response = await client.transcribeAudio(request);
console.log(response.transcript);

// Synthesize speech
const request = new SynthesisRequest('Hello, world!');
const response = await client.synthesizeSpeech(request);
fs.writeFileSync('output.mp3', Buffer.from(response.audioBytes));
```

## Speech-to-Text (Transcription)

### Supported Providers

| Provider | Languages | Features |
|----------|-----------|----------|
| **Deepgram** | 99+ languages | Real-time, VAD, Diarization, Smart formatting |
| **AssemblyAI** | 99+ languages | Entity detection, Sentiment analysis, Diarization |

### TranscriptionRequest

Create a request to transcribe audio:

```python
from modelsuite import TranscriptionRequest

# Basic request
request = TranscriptionRequest(audio_bytes)

# With model selection
request = request.with_model("nova-3")  # Deepgram model

# With language
request = request.with_language("en")
```

### TranscribeOptions

Configure Deepgram transcription options:

```python
from modelsuite import TranscribeOptions

opts = TranscribeOptions()
opts = opts.with_model("nova-3")
opts = opts.with_smart_format(True)
opts = opts.with_diarize(True)
opts = opts.with_language("en")
opts = opts.with_punctuate(True)
```

**Available Options:**

- `model`: Deepgram model to use (e.g., "nova-3", "nova-2")
- `smart_format`: Enable smart formatting for punctuation and capitalization
- `diarize`: Enable speaker diarization to identify different speakers
- `language`: Language code (e.g., "en", "es", "fr")
- `punctuate`: Enable automatic punctuation addition

### TranscriptionConfig

Configure AssemblyAI transcription options:

```python
from modelsuite import TranscriptionConfig, AudioLanguage

config = TranscriptionConfig()
config = config.with_language(AudioLanguage.Spanish)
config = config.with_diarization(True)
config = config.with_entity_detection(True)
config = config.with_sentiment_analysis(True)
```

**Available Options:**

- `language`: Language for transcription (see AudioLanguage enum)
- `enable_diarization`: Identify different speakers
- `enable_entity_detection`: Detect named entities
- `enable_sentiment_analysis`: Analyze sentiment of speech

### TranscribeResponse

Response from transcription:

```python
response = client.transcribe_audio(request)

# Access transcript
text = response.transcript
confidence = response.confidence  # Overall confidence (0.0-1.0)
words = response.words            # List of Word objects with timing
duration = response.duration      # Audio duration in seconds
word_count = response.word_count  # Number of words transcribed
```

### Word Details

Access word-level information:

```python
for word in response.words:
    print(f"'{word.word}' at {word.start:.2f}s-{word.end:.2f}s")
    print(f"  Confidence: {word.confidence:.2%}")
    print(f"  Duration: {word.duration:.2f}s")
    if word.speaker:
        print(f"  Speaker: {word.speaker}")
```

## Text-to-Speech (Synthesis)

### Supported Providers

| Provider | Features | Voice Count |
|----------|----------|-------------|
| **ElevenLabs** | Multiple voices, latency modes, voice cloning | 100+ voices |

### SynthesisRequest

Create a request to synthesize speech:

```python
from modelsuite import SynthesisRequest

# Basic request
request = SynthesisRequest("Hello, world!")

# With voice ID
request = request.with_voice("21m00Tcm4TlvDq8ikWAM")  # Rachel

# With model
request = request.with_model("eleven_monolingual_v1")
```

### SynthesizeOptions

Configure ElevenLabs synthesis options:

```python
from modelsuite import SynthesizeOptions, VoiceSettings, LatencyMode

opts = SynthesizeOptions()
opts = opts.with_model("eleven_monolingual_v1")
opts = opts.with_latency_mode(LatencyMode.Balanced)
opts = opts.with_output_format("mp3_44100_64")

# Customize voice settings
voice = VoiceSettings(stability=0.5, similarity_boost=0.75)
voice = voice.with_style(0.5)
voice = voice.with_speaker_boost(True)
opts = opts.with_voice_settings(voice)
```

**Latency Modes:**

- `LowestLatency` (0): Fastest, lowest quality
- `LowLatency` (1): Fast, good quality
- `Balanced` (2): Default, balanced speed and quality
- `HighQuality` (3): High quality, slower
- `HighestQuality` (4): Best quality, slowest

**Voice Settings:**

- `stability` (0.0-1.0): How consistent the voice sounds
- `similarity_boost` (0.0-1.0): How similar to the original voice
- `style` (0.0-1.0): Stylization of speech
- `use_speaker_boost`: Enable speaker boost for consistency

### VoiceSettings

Control voice characteristics:

```python
from modelsuite import VoiceSettings

settings = VoiceSettings(stability=0.5, similarity_boost=0.75)
settings = settings.with_style(0.5)
settings = settings.with_speaker_boost(True)
```

### SynthesizeResponse

Response from synthesis:

```python
response = client.synthesize_speech(request)

# Access audio
audio_bytes = response.audio_bytes  # Bytes of audio data
format = response.format            # Audio format (e.g., "mp3")
size = response.size                # Size in bytes
duration = response.duration        # Duration in seconds (if available)
```

### Save Audio to File

```python
import os

# Save synthesized audio
with open("output.mp3", "wb") as f:
    f.write(response.audio_bytes)

# Print file info
file_size = os.path.getsize("output.mp3")
print(f"Saved {file_size / 1024:.1f} KB to output.mp3")
```

## Audio Formats

### Supported Input Formats (STT)

- WAV (.wav)
- MP3 (.mp3)
- Ogg Vorbis (.ogg)
- Flac (.flac)
- uLaw (.ulaw)
- Other common audio formats

### Supported Output Formats (TTS)

- MP3: `mp3_22050_32`, `mp3_44100_64`, `mp3_96_192` (default: `mp3_44100_64`)
- PCM: `pcm_16000`, `pcm_24000`, `pcm_44100`, `pcm_48000`
- uLaw: `ulaw_8000`

## Error Handling

### Python

```python
from modelsuite import ModelSuiteError, ModelSuiteClient, TranscriptionRequest

try:
    client = ModelSuiteClient.from_env()
    request = TranscriptionRequest(audio_bytes)
    response = client.transcribe_audio(request)
except ModelSuiteError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### TypeScript

```typescript
import { ModelSuiteClient, TranscriptionRequest } from 'modelsuite';

try {
  const client = ModelSuiteClient.fromEnv();
  const request = new TranscriptionRequest(audioBytes);
  const response = await client.transcribeAudio(request);
} catch (error) {
  console.error(`Error: ${error}`);
}
```

## Environment Variables

### Speech-to-Text

```bash
# Deepgram
export DEEPGRAM_API_KEY="your-deepgram-api-key"

# AssemblyAI
export ASSEMBLYAI_API_KEY="your-assemblyai-api-key"
```

### Text-to-Speech

```bash
# ElevenLabs
export ELEVENLABS_API_KEY="your-elevenlabs-api-key"
```

## Examples

### Complete Transcription Example

**Python:**

```python
from modelsuite import ModelSuiteClient, TranscriptionRequest, TranscribeOptions

# Read audio file
with open("speech.wav", "rb") as f:
    audio_bytes = f.read()

# Initialize client
client = ModelSuiteClient.from_env()

# Create and configure request
request = TranscriptionRequest(audio_bytes)
request = request.with_model("nova-3")
request = request.with_language("en")

# Transcribe
response = client.transcribe_audio(request)

# Print results
print(f"Transcript: {response.transcript}")
print(f"Confidence: {response.confidence:.2%}")
print(f"Duration: {response.duration:.2f}s")
print(f"Words: {response.word_count}")

# Print first 5 words with timing
for word in response.words[:5]:
    print(f"  {word.word} ({word.start:.2f}s)")
```

**TypeScript:**

```typescript
import { ModelSuiteClient, TranscriptionRequest } from 'modelsuite';
import fs from 'fs';

// Read audio file
const audioBytes = fs.readFileSync('speech.wav');

// Initialize client
const client = ModelSuiteClient.fromEnv();

// Create and configure request
const request = new TranscriptionRequest(audioBytes);
request.with_model('nova-3');
request.with_language('en');

// Transcribe
const response = await client.transcribeAudio(request);

// Print results
console.log(`Transcript: ${response.transcript}`);
console.log(`Confidence: ${(response.confidence ?? 0) * 100}%`);
if (response.duration) {
  console.log(`Duration: ${response.duration.toFixed(2)}s`);
}
console.log(`Words: ${response.word_count}`);

// Print first 5 words with timing
for (const word of response.words.slice(0, 5)) {
  console.log(`  ${word.word} (${word.start.toFixed(2)}s)`);
}
```

### Complete Synthesis Example

**Python:**

```python
from modelsuite import ModelSuiteClient, SynthesisRequest, SynthesizeOptions, VoiceSettings

# Initialize client
client = ModelSuiteClient.from_env()

# Create request
text = "Welcome to ModelSuite! This is a test of the text to speech API."
request = SynthesisRequest(text)
request = request.with_voice("21m00Tcm4TlvDq8ikWAM")

# Customize options
opts = SynthesizeOptions()
opts = opts.with_latency_mode(LatencyMode.Balanced)

# Customize voice
voice = VoiceSettings(stability=0.5, similarity_boost=0.75)
opts = opts.with_voice_settings(voice)

# Synthesize
response = client.synthesize_speech(request)

# Save to file
with open("output.mp3", "wb") as f:
    f.write(response.audio_bytes)

print(f"Saved {response.size} bytes to output.mp3")
```

**TypeScript:**

```typescript
import { ModelSuiteClient, SynthesisRequest, SynthesizeOptions, VoiceSettings, LatencyMode } from 'modelsuite';
import fs from 'fs';

// Initialize client
const client = ModelSuiteClient.fromEnv();

// Create request
const text = 'Welcome to ModelSuite! This is a test of the text to speech API.';
const request = new SynthesisRequest(text);
request.with_voice('21m00Tcm4TlvDq8ikWAM');

// Customize options
const opts = new SynthesizeOptions();
opts.with_latency_mode(LatencyMode.Balanced);

// Customize voice
const voice = new VoiceSettings(0.5, 0.75);
opts.with_voice_settings(voice);

// Synthesize
const response = await client.synthesizeSpeech(request);

// Save to file
fs.writeFileSync('output.mp3', Buffer.from(response.audioBytes));
console.log(`Saved ${response.size} bytes to output.mp3`);
```

## Limits and Considerations

### File Size Limits

- **Maximum audio file size**: 50 MB (varies by provider)
- **Recommended maximum**: 25 MB for optimal performance

### Language Support

- Most providers support 50+ languages
- See provider documentation for complete language lists
- Specify language code for best results (e.g., "en", "es", "fr")

### Cost Optimization

- Use appropriate latency modes for TTS (higher quality = higher cost)
- Consider batch processing for multiple audio files
- Monitor provider pricing and quotas

## Troubleshooting

### "No API key found" Error

```
Error: No API key found for provider
```

**Solution:** Set the required environment variable:

```bash
export DEEPGRAM_API_KEY="your-key"  # for transcription
export ELEVENLABS_API_KEY="your-key"  # for synthesis
```

### "Audio format not supported" Error

**Solution:** Convert audio to a supported format (WAV, MP3, OGG, or FLAC)

### Transcription accuracy issues

**Solutions:**
1. Use higher quality audio (44.1 kHz or higher)
2. Enable `smart_format` for better punctuation
3. Specify the language code
4. Use the latest model version

### Synthesis quality issues

**Solutions:**
1. Try different latency modes
2. Adjust voice settings (stability, similarity_boost)
3. Experiment with different voices
4. Check output format compatibility

## API Reference

### Classes and Methods

#### Python

- `TranscriptionRequest(audio_bytes: bytes)` - Create transcription request
  - `with_model(model: str)` - Set model
  - `with_language(language: str)` - Set language

- `TranscribeOptions()` - Configure Deepgram options
  - `with_model(model: str)`
  - `with_smart_format(enabled: bool)`
  - `with_diarize(enabled: bool)`
  - `with_language(language: str)`
  - `with_punctuate(enabled: bool)`

- `TranscribeResponse` - Transcription response
  - `transcript: str` - Transcribed text
  - `confidence: float` - Overall confidence
  - `words: List[Word]` - Word-level details
  - `duration: float` - Audio duration
  - `word_count: int` - Number of words

- `SynthesisRequest(text: str)` - Create synthesis request
  - `with_voice(voice_id: str)` - Set voice
  - `with_model(model: str)` - Set model

- `SynthesizeResponse` - Synthesis response
  - `audio_bytes: bytes` - Audio data
  - `format: str` - Audio format
  - `duration: float` - Duration (if available)
  - `size: int` - Byte size

#### TypeScript

- `TranscriptionRequest(audioBytes: Uint8Array)` - Create transcription request
  - `with_model(model: string)` - Set model
  - `with_language(language: string)` - Set language

- `TranscribeOptions()` - Configure Deepgram options
  - `with_model(model: string)`
  - `with_smart_format(enabled: boolean)`
  - `with_diarize(enabled: boolean)`
  - `with_language(language: string)`
  - `with_punctuate(enabled: boolean)`

- `TranscribeResponse` - Transcription response
  - `transcript: string` - Transcribed text
  - `confidence: number` - Overall confidence
  - `words: Word[]` - Word-level details
  - `duration: number` - Audio duration
  - `word_count: number` - Number of words

- `SynthesisRequest(text: string)` - Create synthesis request
  - `with_voice(voiceId: string)` - Set voice
  - `with_model(model: string)` - Set model

- `SynthesizeResponse` - Synthesis response
  - `audioBytes: Uint8Array` - Audio data
  - `format: string` - Audio format
  - `duration: number` - Duration (if available)
  - `size: number` - Byte size

## Related Documentation

- [ModelSuite Overview](../README.md)
- [Deepgram API Documentation](https://developers.deepgram.com/)
- [AssemblyAI API Documentation](https://www.assemblyai.com/docs)
- [ElevenLabs API Documentation](https://elevenlabs.io/docs)
