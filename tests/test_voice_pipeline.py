import pytest

from src.voice.contracts import STTService, TTSService
from src.voice.pipeline import VoicePipeline


class FakeSTT(STTService):
    def __init__(self, transcript: str = "My payment was charged twice", error: Exception | None = None):
        self._transcript = transcript
        self._error = error

    async def initialize(self) -> None:
        return None

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        if self._error is not None:
            raise self._error
        return self._transcript


class FakeTTS(TTSService):
    def __init__(self, audio: bytes = b"fake-audio", media_type: str = "audio/mpeg", error: Exception | None = None):
        self._audio = audio
        self._media_type = media_type
        self._error = error

    async def initialize(self) -> None:
        return None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        if self._error is not None:
            raise self._error
        return self._audio, self._media_type


@pytest.mark.asyncio
async def test_voice_pipeline_contracts() -> None:
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())

    transcript, transcribe_ms = await pipeline.transcribe(b"fake-input", "audio/wav")
    audio, media_type, synthesize_ms = await pipeline.synthesize("Agent response")

    assert transcript == "My payment was charged twice"
    assert audio == b"fake-audio"
    assert media_type == "audio/mpeg"
    assert transcribe_ms >= 0
    assert synthesize_ms >= 0


@pytest.mark.asyncio
async def test_transcribe_rejects_empty_audio() -> None:
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    with pytest.raises(ValueError):
        await pipeline.transcribe(b"", "audio/wav")


@pytest.mark.asyncio
async def test_transcribe_rejects_blank_transcript() -> None:
    pipeline = VoicePipeline(FakeSTT(transcript="   "), FakeTTS())
    with pytest.raises(ValueError, match="No understandable speech"):
        await pipeline.transcribe(b"fake-input", "audio/wav")


@pytest.mark.asyncio
async def test_transcribe_propagates_stt_failure() -> None:
    pipeline = VoicePipeline(FakeSTT(error=RuntimeError("upstream STT error")), FakeTTS())
    with pytest.raises(RuntimeError, match="upstream STT error"):
        await pipeline.transcribe(b"fake-input", "audio/wav")


@pytest.mark.asyncio
async def test_synthesize_rejects_empty_text() -> None:
    pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    with pytest.raises(ValueError):
        await pipeline.synthesize("   ")


@pytest.mark.asyncio
async def test_synthesize_propagates_tts_failure() -> None:
    pipeline = VoicePipeline(FakeSTT(), FakeTTS(error=RuntimeError("upstream TTS error")))
    with pytest.raises(RuntimeError, match="upstream TTS error"):
        await pipeline.synthesize("Agent response")
