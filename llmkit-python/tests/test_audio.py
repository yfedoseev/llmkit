"""Tests for LLMKit audio bindings."""

import pytest

from llmkit import (  # type: ignore[attr-defined]
    AudioLanguage,
    DeepgramVersion,
    LatencyMode,
    LLMKitClient,
    SynthesisRequest,
    SynthesizeOptions,
    SynthesizeResponse,
    TranscribeOptions,
    TranscribeResponse,
    TranscriptionConfig,
    TranscriptionRequest,
    VoiceSettings,
    Word,
)


class TestDeepgramVersion:
    """Test DeepgramVersion enum."""

    def test_deepgram_version_v1(self) -> None:
        assert DeepgramVersion.V1 == 0

    def test_deepgram_version_v3(self) -> None:
        assert DeepgramVersion.V3 == 1


class TestTranscribeOptions:
    """Test TranscribeOptions configuration."""

    def test_transcribe_options_default(self) -> None:
        opts = TranscribeOptions()
        assert opts.model is None
        assert opts.smart_format is False
        assert opts.diarize is False
        assert opts.language is None
        assert opts.punctuate is False

    def test_transcribe_options_with_model(self) -> None:
        opts = TranscribeOptions()
        opts = opts.with_model("nova-3")
        assert opts.model == "nova-3"

    def test_transcribe_options_with_smart_format(self) -> None:
        opts = TranscribeOptions()
        opts = opts.with_smart_format(True)
        assert opts.smart_format is True

    def test_transcribe_options_with_diarize(self) -> None:
        opts = TranscribeOptions()
        opts = opts.with_diarize(True)
        assert opts.diarize is True

    def test_transcribe_options_with_language(self) -> None:
        opts = TranscribeOptions()
        opts = opts.with_language("en")
        assert opts.language == "en"

    def test_transcribe_options_with_punctuate(self) -> None:
        opts = TranscribeOptions()
        opts = opts.with_punctuate(True)
        assert opts.punctuate is True

    def test_transcribe_options_builder_chain(self) -> None:
        opts = TranscribeOptions().with_model("nova-3").with_smart_format(True).with_language("en")
        assert opts.model == "nova-3"
        assert opts.smart_format is True
        assert opts.language == "en"

    def test_transcribe_options_repr(self) -> None:
        opts = TranscribeOptions()
        repr_str = repr(opts)
        assert "TranscribeOptions" in repr_str


class TestWord:
    """Test Word class for transcription details."""

    def test_word_creation(self) -> None:
        word = Word()
        word.word = "hello"
        word.start = 0.0
        word.end = 0.5
        word.confidence = 0.95
        word.speaker = None

        assert word.word == "hello"
        assert word.start == 0.0
        assert word.end == 0.5
        assert word.confidence == 0.95
        assert word.duration == 0.5

    def test_word_with_speaker(self) -> None:
        word = Word()
        word.word = "hello"
        word.start = 0.0
        word.end = 0.5
        word.confidence = 0.95
        word.speaker = 1

        assert word.speaker == 1


class TestTranscribeResponse:
    """Test TranscribeResponse type."""

    def test_transcribe_response_creation(self) -> None:
        response = TranscribeResponse()
        response.transcript = "Hello, how are you?"
        response.confidence = 0.95
        response.words = []
        response.duration = 2.5
        response.metadata = None

        assert response.transcript == "Hello, how are you?"
        assert response.confidence == 0.95
        assert response.word_count == 0
        assert response.duration == 2.5

    def test_transcribe_response_with_words(self) -> None:
        word = Word()
        word.word = "hello"
        word.start = 0.0
        word.end = 0.5
        word.confidence = 0.95
        word.speaker = None

        response = TranscribeResponse()
        response.transcript = "hello"
        response.confidence = 0.95
        response.words = [word]
        response.duration = 0.5
        response.metadata = None

        assert response.word_count == 1


class TestLatencyMode:
    """Test LatencyMode enum."""

    def test_latency_mode_values(self) -> None:
        assert LatencyMode.LowestLatency == 0
        assert LatencyMode.LowLatency == 1
        assert LatencyMode.Balanced == 2
        assert LatencyMode.HighQuality == 3
        assert LatencyMode.HighestQuality == 4


class TestVoiceSettings:
    """Test VoiceSettings configuration."""

    def test_voice_settings_default(self) -> None:
        settings = VoiceSettings()
        assert settings.stability == 0.5
        assert settings.similarity_boost == 0.75
        assert settings.style is None
        assert settings.use_speaker_boost is False

    def test_voice_settings_custom(self) -> None:
        settings = VoiceSettings(stability=0.7, similarity_boost=0.8)
        assert settings.stability == pytest.approx(0.7, rel=1e-5)
        assert settings.similarity_boost == pytest.approx(0.8, rel=1e-5)

    def test_voice_settings_with_style(self) -> None:
        settings = VoiceSettings()
        settings = settings.with_style(0.5)
        assert settings.style == 0.5

    def test_voice_settings_with_speaker_boost(self) -> None:
        settings = VoiceSettings()
        settings = settings.with_speaker_boost(True)
        assert settings.use_speaker_boost is True


class TestSynthesizeOptions:
    """Test SynthesizeOptions configuration."""

    def test_synthesize_options_default(self) -> None:
        opts = SynthesizeOptions()
        assert opts.model_id is None
        assert opts.voice_settings is not None
        assert opts.latency_mode == LatencyMode.Balanced
        assert opts.output_format == "mp3_44100_64"

    def test_synthesize_options_with_model(self) -> None:
        opts = SynthesizeOptions()
        opts = opts.with_model("eleven_monolingual_v1")
        assert opts.model_id == "eleven_monolingual_v1"

    def test_synthesize_options_with_voice_settings(self) -> None:
        settings = VoiceSettings(stability=0.6)
        opts = SynthesizeOptions()
        opts = opts.with_voice_settings(settings)
        # Compare by properties since VoiceSettings doesn't implement __eq__
        assert opts.voice_settings.stability == pytest.approx(settings.stability, rel=1e-5)
        assert opts.voice_settings.similarity_boost == pytest.approx(settings.similarity_boost, rel=1e-5)

    def test_synthesize_options_with_latency_mode(self) -> None:
        opts = SynthesizeOptions()
        opts = opts.with_latency_mode(LatencyMode.HighQuality)
        assert opts.latency_mode == LatencyMode.HighQuality

    def test_synthesize_options_with_output_format(self) -> None:
        opts = SynthesizeOptions()
        opts = opts.with_output_format("wav")
        assert opts.output_format == "wav"


class TestSynthesizeResponse:
    """Test SynthesizeResponse type."""

    def test_synthesize_response_creation(self) -> None:
        response = SynthesizeResponse()
        response.audio_bytes = b"fake audio data"
        response.format = "mp3"
        response.duration = 2.5

        assert response.audio_bytes == b"fake audio data"
        assert response.format == "mp3"
        assert response.duration == 2.5
        assert response.size == len(b"fake audio data")

    def test_synthesize_response_size(self) -> None:
        response = SynthesizeResponse()
        response.audio_bytes = b"\x00" * 1000
        response.format = "mp3"
        response.duration = None

        assert response.size == 1000


class TestAudioLanguage:
    """Test AudioLanguage enum."""

    def test_audio_language_values(self) -> None:
        assert AudioLanguage.English == 0
        assert AudioLanguage.Spanish == 1
        assert AudioLanguage.French == 2
        assert AudioLanguage.German == 3
        assert AudioLanguage.ChineseSimplified == 4
        assert AudioLanguage.ChineseTraditional == 5
        assert AudioLanguage.Japanese == 6


class TestTranscriptionConfig:
    """Test TranscriptionConfig configuration."""

    def test_transcription_config_default(self) -> None:
        config = TranscriptionConfig()
        assert config.language is None
        assert config.enable_diarization is False
        assert config.enable_entity_detection is False
        assert config.enable_sentiment_analysis is False

    def test_transcription_config_with_language(self) -> None:
        config = TranscriptionConfig()
        config = config.with_language(AudioLanguage.English)
        assert config.language == AudioLanguage.English

    def test_transcription_config_with_diarization(self) -> None:
        config = TranscriptionConfig()
        config = config.with_diarization(True)
        assert config.enable_diarization is True

    def test_transcription_config_with_entity_detection(self) -> None:
        config = TranscriptionConfig()
        config = config.with_entity_detection(True)
        assert config.enable_entity_detection is True

    def test_transcription_config_with_sentiment_analysis(self) -> None:
        config = TranscriptionConfig()
        config = config.with_sentiment_analysis(True)
        assert config.enable_sentiment_analysis is True

    def test_transcription_config_builder_chain(self) -> None:
        config = (
            TranscriptionConfig()
            .with_language(AudioLanguage.Spanish)
            .with_diarization(True)
            .with_entity_detection(True)
        )
        assert config.language == AudioLanguage.Spanish
        assert config.enable_diarization is True
        assert config.enable_entity_detection is True


class TestTranscriptionRequest:
    """Test TranscriptionRequest type."""

    def test_transcription_request_creation(self) -> None:
        audio_bytes = b"fake audio data"
        request = TranscriptionRequest(audio_bytes)

        assert request.audio_bytes == audio_bytes
        assert request.model is None
        assert request.language is None

    def test_transcription_request_with_model(self) -> None:
        audio_bytes = b"fake audio data"
        request = TranscriptionRequest(audio_bytes)
        request = request.with_model("nova-3")

        assert request.model == "nova-3"

    def test_transcription_request_with_language(self) -> None:
        audio_bytes = b"fake audio data"
        request = TranscriptionRequest(audio_bytes)
        request = request.with_language("en")

        assert request.language == "en"

    def test_transcription_request_repr(self) -> None:
        audio_bytes = b"fake audio data"
        request = TranscriptionRequest(audio_bytes)
        repr_str = repr(request)
        assert "TranscriptionRequest" in repr_str
        assert str(len(audio_bytes)) in repr_str


class TestSynthesisRequest:
    """Test SynthesisRequest type."""

    def test_synthesis_request_creation(self) -> None:
        request = SynthesisRequest("Hello, world!")

        assert request.text == "Hello, world!"
        assert request.voice_id is None
        assert request.model is None

    def test_synthesis_request_with_voice(self) -> None:
        request = SynthesisRequest("Hello, world!")
        request = request.with_voice("pNInY14gQrG92XwBIHVr")

        assert request.voice_id == "pNInY14gQrG92XwBIHVr"

    def test_synthesis_request_with_model(self) -> None:
        request = SynthesisRequest("Hello, world!")
        request = request.with_model("eleven_monolingual_v1")

        assert request.model == "eleven_monolingual_v1"

    def test_synthesis_request_builder_chain(self) -> None:
        request = (
            SynthesisRequest("Hello, world!")
            .with_voice("pNInY14gQrG92XwBIHVr")
            .with_model("eleven_monolingual_v1")
        )

        assert request.voice_id == "pNInY14gQrG92XwBIHVr"
        assert request.model == "eleven_monolingual_v1"

    def test_synthesis_request_repr(self) -> None:
        request = SynthesisRequest("Hello, world!")
        repr_str = repr(request)
        assert "SynthesisRequest" in repr_str
        assert "Hello" in repr_str


class TestAudioClientMethods:
    """Test audio methods on LLMKitClient."""

    def test_client_has_transcribe_audio_method(self) -> None:
        """Check that LLMKitClient has transcribe_audio method."""
        assert hasattr(LLMKitClient, "transcribe_audio")

    def test_client_has_synthesize_speech_method(self) -> None:
        """Check that LLMKitClient has synthesize_speech method."""
        assert hasattr(LLMKitClient, "synthesize_speech")


class TestAudioImports:
    """Test that all audio types can be imported."""

    def test_import_all_audio_types(self) -> None:
        """Verify all audio types are importable."""
        from llmkit import (  # type: ignore[attr-defined]
            AudioLanguage,
            DeepgramVersion,
            LatencyMode,
            SynthesisRequest,
            SynthesizeOptions,
            SynthesizeResponse,
            TranscribeOptions,
            TranscribeResponse,
            TranscriptionConfig,
            TranscriptionRequest,
            Voice,
            VoiceSettings,
            Word,
        )

        # Just verify imports succeeded
        assert AudioLanguage is not None
        assert DeepgramVersion is not None
        assert LatencyMode is not None
        assert SynthesisRequest is not None
        assert SynthesizeOptions is not None
        assert SynthesizeResponse is not None
        assert TranscribeOptions is not None
        assert TranscribeResponse is not None
        assert TranscriptionConfig is not None
        assert TranscriptionRequest is not None
        assert Voice is not None
        assert VoiceSettings is not None
        assert Word is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
