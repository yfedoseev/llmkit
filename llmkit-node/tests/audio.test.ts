/**
 * Unit tests for LLMKit audio bindings
 *
 * These tests verify audio API functionality without making actual audio calls.
 */

import { describe, it, expect } from 'vitest';
import {
  JsDeepgramVersion as DeepgramVersion,
  JsTranscribeOptions as TranscribeOptions,
  JsWord as Word,
  JsTranscribeResponse as TranscribeResponse,
  JsLatencyMode as LatencyMode,
  JsVoiceSettings as VoiceSettings,
  JsSynthesizeOptions as SynthesizeOptions,
  JsSynthesizeResponse as SynthesizeResponse,
  JsAudioLanguage as AudioLanguage,
  JsTranscriptionConfig as TranscriptionConfig,
  JsTranscriptionRequest as TranscriptionRequest,
  JsSynthesisRequest as SynthesisRequest,
  JsLlmKitClient as LLMKitClient,
} from '../index';

// =============================================================================
// DeepgramVersion Tests
// =============================================================================

describe('DeepgramVersion', () => {
  it('has correct V1 value', () => {
    expect(DeepgramVersion.V1).toBe(0);
  });

  it('has correct V3 value', () => {
    expect(DeepgramVersion.V3).toBe(1);
  });
});

// =============================================================================
// TranscribeOptions Tests
// =============================================================================

describe('TranscribeOptions', () => {
  it('creates with default values', () => {
    const opts = new TranscribeOptions();
    expect(opts.model).toBeNull();
    expect(opts.smart_format).toBe(false);
    expect(opts.diarize).toBe(false);
    expect(opts.language).toBeNull();
    expect(opts.punctuate).toBe(false);
  });

  it('sets model via with_model', () => {
    const opts = new TranscribeOptions();
    const updated = opts.with_model('nova-3');
    expect(updated.model).toBe('nova-3');
  });

  it('sets smart_format via with_smart_format', () => {
    const opts = new TranscribeOptions();
    const updated = opts.with_smart_format(true);
    expect(updated.smart_format).toBe(true);
  });

  it('sets diarize via with_diarize', () => {
    const opts = new TranscribeOptions();
    const updated = opts.with_diarize(true);
    expect(updated.diarize).toBe(true);
  });

  it('sets language via with_language', () => {
    const opts = new TranscribeOptions();
    const updated = opts.with_language('en');
    expect(updated.language).toBe('en');
  });

  it('sets punctuate via with_punctuate', () => {
    const opts = new TranscribeOptions();
    const updated = opts.with_punctuate(true);
    expect(updated.punctuate).toBe(true);
  });

  it('supports builder chaining', () => {
    const opts = new TranscribeOptions()
      .with_model('nova-3')
      .with_smart_format(true)
      .with_language('en');

    expect(opts.model).toBe('nova-3');
    expect(opts.smart_format).toBe(true);
    expect(opts.language).toBe('en');
  });
});

// =============================================================================
// Word Tests
// =============================================================================

describe('Word', () => {
  it('calculates duration correctly', () => {
    const word = new Word();
    word.start = 0.0;
    word.end = 0.5;
    word.word = 'hello';
    word.confidence = 0.95;

    expect(word.duration).toBe(0.5);
  });

  it('handles different time ranges', () => {
    const word = new Word();
    word.start = 2.0;
    word.end = 3.5;
    word.duration;

    expect(word.duration).toBe(1.5);
  });

  it('handles speaker diarization', () => {
    const word = new Word();
    word.word = 'hello';
    word.start = 0.0;
    word.end = 0.5;
    word.confidence = 0.95;
    word.speaker = 1;

    expect(word.speaker).toBe(1);
  });
});

// =============================================================================
// TranscribeResponse Tests
// =============================================================================

describe('TranscribeResponse', () => {
  it('counts words correctly', () => {
    const response = new TranscribeResponse();
    response.transcript = 'hello world';
    response.confidence = 0.95;

    const word1 = new Word();
    word1.word = 'hello';
    word1.start = 0.0;
    word1.end = 0.5;
    word1.confidence = 0.95;

    const word2 = new Word();
    word2.word = 'world';
    word2.start = 0.5;
    word2.end = 1.0;
    word2.confidence = 0.94;

    response.words = [word1, word2];
    expect(response.word_count).toBe(2);
  });

  it('handles empty word list', () => {
    const response = new TranscribeResponse();
    response.transcript = 'hello';
    response.confidence = 0.95;
    response.words = [];

    expect(response.word_count).toBe(0);
  });
});

// =============================================================================
// LatencyMode Tests
// =============================================================================

describe('LatencyMode', () => {
  it('has correct enum values', () => {
    expect(LatencyMode.LowestLatency).toBe(0);
    expect(LatencyMode.LowLatency).toBe(1);
    expect(LatencyMode.Balanced).toBe(2);
    expect(LatencyMode.HighQuality).toBe(3);
    expect(LatencyMode.HighestQuality).toBe(4);
  });
});

// =============================================================================
// VoiceSettings Tests
// =============================================================================

describe('VoiceSettings', () => {
  it('creates with default values', () => {
    const settings = new VoiceSettings();
    expect(settings.stability).toBe(0.5);
    expect(settings.similarity_boost).toBe(0.75);
    expect(settings.style).toBeNull();
    expect(settings.use_speaker_boost).toBe(false);
  });

  it('creates with custom values', () => {
    const settings = new VoiceSettings(0.7, 0.8);
    expect(settings.stability).toBe(0.7);
    expect(settings.similarity_boost).toBe(0.8);
  });

  it('sets style via with_style', () => {
    const settings = new VoiceSettings();
    const updated = settings.with_style(0.5);
    expect(updated.style).toBe(0.5);
  });

  it('sets speaker boost via with_speaker_boost', () => {
    const settings = new VoiceSettings();
    const updated = settings.with_speaker_boost(true);
    expect(updated.use_speaker_boost).toBe(true);
  });
});

// =============================================================================
// SynthesizeOptions Tests
// =============================================================================

describe('SynthesizeOptions', () => {
  it('creates with default values', () => {
    const opts = new SynthesizeOptions();
    expect(opts.model_id).toBeNull();
    expect(opts.voice_settings).not.toBeNull();
    expect(opts.latency_mode).toBe(LatencyMode.Balanced);
    expect(opts.output_format).toBe('mp3_44100_64');
  });

  it('sets model via with_model', () => {
    const opts = new SynthesizeOptions();
    const updated = opts.with_model('eleven_monolingual_v1');
    expect(updated.model_id).toBe('eleven_monolingual_v1');
  });

  it('sets voice settings via with_voice_settings', () => {
    const opts = new SynthesizeOptions();
    const settings = new VoiceSettings(0.6);
    const updated = opts.with_voice_settings(settings);
    expect(updated.voice_settings?.stability).toBe(0.6);
  });

  it('sets latency mode via with_latency_mode', () => {
    const opts = new SynthesizeOptions();
    const updated = opts.with_latency_mode(LatencyMode.HighQuality);
    expect(updated.latency_mode).toBe(LatencyMode.HighQuality);
  });

  it('sets output format via with_output_format', () => {
    const opts = new SynthesizeOptions();
    const updated = opts.with_output_format('wav');
    expect(updated.output_format).toBe('wav');
  });

  it('supports builder chaining', () => {
    const opts = new SynthesizeOptions()
      .with_model('eleven_monolingual_v1')
      .with_latency_mode(LatencyMode.HighQuality)
      .with_output_format('mp3');

    expect(opts.model_id).toBe('eleven_monolingual_v1');
    expect(opts.latency_mode).toBe(LatencyMode.HighQuality);
    expect(opts.output_format).toBe('mp3');
  });
});

// =============================================================================
// SynthesizeResponse Tests
// =============================================================================

describe('SynthesizeResponse', () => {
  it('calculates size correctly', () => {
    const response = new SynthesizeResponse();
    response.audio_bytes = new Uint8Array(1000);
    response.format = 'mp3';

    expect(response.size).toBe(1000);
  });

  it('handles empty audio bytes', () => {
    const response = new SynthesizeResponse();
    response.audio_bytes = new Uint8Array(0);
    response.format = 'mp3';

    expect(response.size).toBe(0);
  });

  it('stores duration correctly', () => {
    const response = new SynthesizeResponse();
    response.audio_bytes = new Uint8Array(1000);
    response.format = 'mp3';
    response.duration = 2.5;

    expect(response.duration).toBe(2.5);
  });
});

// =============================================================================
// AudioLanguage Tests
// =============================================================================

describe('AudioLanguage', () => {
  it('has correct enum values', () => {
    expect(AudioLanguage.English).toBe(0);
    expect(AudioLanguage.Spanish).toBe(1);
    expect(AudioLanguage.French).toBe(2);
    expect(AudioLanguage.German).toBe(3);
    expect(AudioLanguage.ChineseSimplified).toBe(4);
    expect(AudioLanguage.ChineseTraditional).toBe(5);
    expect(AudioLanguage.Japanese).toBe(6);
  });
});

// =============================================================================
// TranscriptionConfig Tests
// =============================================================================

describe('TranscriptionConfig', () => {
  it('creates with default values', () => {
    const config = new TranscriptionConfig();
    expect(config.language).toBeNull();
    expect(config.enable_diarization).toBe(false);
    expect(config.enable_entity_detection).toBe(false);
    expect(config.enable_sentiment_analysis).toBe(false);
  });

  it('sets language via with_language', () => {
    const config = new TranscriptionConfig();
    const updated = config.with_language(AudioLanguage.Spanish);
    expect(updated.language).toBe(AudioLanguage.Spanish);
  });

  it('sets diarization via with_diarization', () => {
    const config = new TranscriptionConfig();
    const updated = config.with_diarization(true);
    expect(updated.enable_diarization).toBe(true);
  });

  it('sets entity detection via with_entity_detection', () => {
    const config = new TranscriptionConfig();
    const updated = config.with_entity_detection(true);
    expect(updated.enable_entity_detection).toBe(true);
  });

  it('sets sentiment analysis via with_sentiment_analysis', () => {
    const config = new TranscriptionConfig();
    const updated = config.with_sentiment_analysis(true);
    expect(updated.enable_sentiment_analysis).toBe(true);
  });

  it('supports builder chaining', () => {
    const config = new TranscriptionConfig()
      .with_language(AudioLanguage.Spanish)
      .with_diarization(true)
      .with_entity_detection(true);

    expect(config.language).toBe(AudioLanguage.Spanish);
    expect(config.enable_diarization).toBe(true);
    expect(config.enable_entity_detection).toBe(true);
  });
});

// =============================================================================
// TranscriptionRequest Tests
// =============================================================================

describe('TranscriptionRequest', () => {
  it('creates with audio bytes', () => {
    const audioBytes = new Uint8Array(100);
    const request = new TranscriptionRequest(audioBytes);

    expect(request.audio_bytes).toEqual(audioBytes);
    expect(request.model).toBeNull();
    expect(request.language).toBeNull();
  });

  it('sets model via with_model', () => {
    const audioBytes = new Uint8Array(100);
    const request = new TranscriptionRequest(audioBytes);
    const updated = request.with_model('nova-3');

    expect(updated.model).toBe('nova-3');
  });

  it('sets language via with_language', () => {
    const audioBytes = new Uint8Array(100);
    const request = new TranscriptionRequest(audioBytes);
    const updated = request.with_language('en');

    expect(updated.language).toBe('en');
  });

  it('supports builder chaining', () => {
    const audioBytes = new Uint8Array(100);
    const request = new TranscriptionRequest(audioBytes)
      .with_model('nova-3')
      .with_language('en');

    expect(request.model).toBe('nova-3');
    expect(request.language).toBe('en');
  });
});

// =============================================================================
// SynthesisRequest Tests
// =============================================================================

describe('SynthesisRequest', () => {
  it('creates with text', () => {
    const request = new SynthesisRequest('Hello, world!');

    expect(request.text).toBe('Hello, world!');
    expect(request.voice_id).toBeNull();
    expect(request.model).toBeNull();
  });

  it('sets voice via with_voice', () => {
    const request = new SynthesisRequest('Hello');
    const updated = request.with_voice('pNInY14gQrG92XwBIHVr');

    expect(updated.voice_id).toBe('pNInY14gQrG92XwBIHVr');
  });

  it('sets model via with_model', () => {
    const request = new SynthesisRequest('Hello');
    const updated = request.with_model('eleven_monolingual_v1');

    expect(updated.model).toBe('eleven_monolingual_v1');
  });

  it('supports builder chaining', () => {
    const request = new SynthesisRequest('Hello, world!')
      .with_voice('pNInY14gQrG92XwBIHVr')
      .with_model('eleven_monolingual_v1');

    expect(request.voice_id).toBe('pNInY14gQrG92XwBIHVr');
    expect(request.model).toBe('eleven_monolingual_v1');
  });
});

// =============================================================================
// Client Method Tests
// =============================================================================

describe('LLMKitClient Audio Methods', () => {
  it('has transcribeAudio method', () => {
    expect(LLMKitClient.prototype.transcribe_audio).toBeDefined();
  });

  it('has synthesizeSpeech method', () => {
    expect(LLMKitClient.prototype.synthesize_speech).toBeDefined();
  });
});
