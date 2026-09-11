from __future__ import annotations

import io

import edge_tts

from src.config import Settings

from .contracts import TTSService


class EdgeTTSService(TTSService):
    """TTS adapter backed by Microsoft Edge's free neural voices.

    Chosen because it needs no API key or paid account (nothing to leak in
    the submission) while still producing natural-sounding speech, and the
    ``edge-tts`` package is a thin async wrapper around a public streaming
    endpoint rather than a full conversational platform.
    """

    def __init__(self, settings: Settings) -> None:
        self._voice = settings.tts_voice

    async def initialize(self) -> None:
        return None

    async def synthesize(self, text: str) -> tuple[bytes, str]:
        communicator = edge_tts.Communicate(text, self._voice)
        buffer = io.BytesIO()
        async for chunk in communicator.stream():
            if chunk["type"] == "audio":
                buffer.write(chunk["data"])
        audio_bytes = buffer.getvalue()
        if not audio_bytes:
            raise RuntimeError("Edge TTS returned no audio data")
        return audio_bytes, "audio/mpeg"

    async def cleanup(self) -> None:
        return None
