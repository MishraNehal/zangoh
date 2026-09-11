from __future__ import annotations

from src.config import Settings

from .contracts import STTService, TTSService
from .stt_groq import GroqWhisperSTT
from .tts_edge import EdgeTTSService


def build_stt_service(settings: Settings) -> STTService:
    """Single seam for swapping the STT provider later without touching
    server.py or VoicePipeline -- only this function needs to change."""
    return GroqWhisperSTT(settings)


def build_tts_service(settings: Settings) -> TTSService:
    """Single seam for swapping the TTS provider later; see build_stt_service."""
    return EdgeTTSService(settings)
