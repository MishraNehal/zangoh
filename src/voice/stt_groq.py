from __future__ import annotations

import httpx

from src.config import Settings

from .contracts import STTService

GROQ_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

_EXTENSION_BY_MEDIA_TYPE = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/webm": "webm",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/ogg": "ogg",
}


class GroqWhisperSTT(STTService):
    """STT adapter that calls Groq's hosted Whisper endpoint directly.

    Chosen because the starter project already talks to Groq (an
    OpenAI-compatible endpoint) for the chat model, so this reuses the same
    account/API key and needs no new SDK -- a single ``httpx`` multipart POST
    against Groq's OpenAI-compatible ``/audio/transcriptions`` route is
    sufficient, and it keeps this adapter to "core Python audio and HTTP
    libraries" per the assignment's allowed list.
    """

    def __init__(self, settings: Settings) -> None:
        # Falls back to the LLM key since both point at the same Groq account
        # by default; a dedicated STT_API_KEY is still supported for anyone
        # who wants to separate the two.
        self._api_key = settings.stt_api_key or settings.llm_api_key
        self._model = settings.stt_model
        self._client: httpx.AsyncClient | None = None

    async def initialize(self) -> None:
        if not self._api_key or self._api_key == "not-required":
            raise ValueError(
                "STT_API_KEY (or LLM_API_KEY) must be set to use the Groq "
                "Whisper adapter"
            )
        self._client = httpx.AsyncClient(timeout=30.0)

    async def transcribe(self, audio_bytes: bytes, media_type: str) -> str:
        if self._client is None:
            raise RuntimeError("GroqWhisperSTT.initialize() was not called")

        extension = _EXTENSION_BY_MEDIA_TYPE.get(media_type, "wav")
        files = {"file": (f"recording.{extension}", audio_bytes, media_type)}
        data = {"model": self._model, "response_format": "json"}
        headers = {"Authorization": f"Bearer {self._api_key}"}

        response = await self._client.post(
            GROQ_TRANSCRIPTIONS_URL, headers=headers, data=data, files=files
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"Groq transcription request failed "
                f"({response.status_code}): {response.text}"
            )

        payload = response.json()
        return payload.get("text", "")

    async def cleanup(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
