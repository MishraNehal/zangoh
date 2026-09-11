from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

from src.config import load_settings
from src.models import ChatRequest, ChatResponse, Ticket
from src.pipeline import SupportPipeline
from src.utils.errors import AgentProcessingError, ComponentNotReadyError
from src.voice.factory import build_stt_service, build_tts_service
from src.voice.models import SynthesisRequest, TranscriptionResponse
from src.voice.pipeline import VoicePipeline


settings = load_settings()
pipeline = SupportPipeline(settings, Path(__file__).resolve().parents[2] / "knowledge_base")

# Constructed at import time (mirrors `pipeline` above) but not connected to a
# real STT/TTS client until `.initialize()` runs during lifespan startup, so
# importing this module never makes a network call.
voice_pipeline = VoicePipeline(build_stt_service(settings), build_tts_service(settings))


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Startup owns expensive shared initialization. Request handlers reuse the
    # resulting components, while shutdown makes readiness false immediately.
    await pipeline.initialize()
    await voice_pipeline.initialize()
    yield
    pipeline.ready = False
    await voice_pipeline.cleanup()


app = FastAPI(title="Customer Support Ticket Agent", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    if not pipeline.ready:
        raise HTTPException(status_code=503, detail="Support pipeline is not ready")
    return {"status": "ready"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    # Keep this transport boundary thin: Pydantic validates the public request,
    # the pipeline owns orchestration, and known service errors are translated
    # to stable HTTP responses here.
    try:
        return await pipeline.process(request.session_id, request.message)
    except ComponentNotReadyError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except AgentProcessingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/tickets/{ticket_id}", response_model=Ticket)
async def get_ticket(ticket_id: str) -> Ticket:
    # Keep this endpoint read-only. It should return the exact repository record,
    # not ask the model to reconstruct ticket details. Test both the successful
    # lookup and unknown-ID response through FastAPI's test client.
    ticket = pipeline.tickets.get(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


# ---------------------------------------------------------------------------
# Voice enhancement (mid-session requirement). Both routes are additive: they
# never call `pipeline` directly and never construct a ChatResponse. STT only
# turns audio into text for Streamlit to hand to the *existing* POST /chat
# request, and TTS only turns already-generated response text into audio.
# Errors here use the shared VoiceErrorResponse shape rather than raising
# HTTPException, so a transcription/synthesis failure can never look like a
# validation problem with the untouched /chat contract.
# ---------------------------------------------------------------------------


@app.post("/voice/transcribe", response_model=TranscriptionResponse)
async def transcribe_voice(audio: UploadFile = File(...)):
    audio_bytes = await audio.read()
    media_type = audio.content_type or "audio/wav"
    try:
        transcript, elapsed_ms = await voice_pipeline.transcribe(audio_bytes, media_type)
    except ValueError as exc:
        # Empty audio or no intelligible speech -- a client-input problem.
        return JSONResponse(status_code=422, content={"success": False, "error": str(exc)})
    except Exception as exc:
        # Upstream STT provider failure (network, auth, quota, etc).
        return JSONResponse(
            status_code=502,
            content={"success": False, "error": f"Transcription failed: {exc}"},
        )
    return TranscriptionResponse(success=True, transcript=transcript, processing_time_ms=elapsed_ms)


@app.post("/voice/synthesize")
async def synthesize_voice(request: SynthesisRequest):
    # request.message_id isn't needed server-side (synthesis is stateless) --
    # it travels through so the Streamlit client can be certain which chat
    # message the returned audio belongs to, per the "associate each icon and
    # audio result with the correct chat-message ID" requirement.
    try:
        audio_bytes, media_type, _elapsed_ms = await voice_pipeline.synthesize(request.text)
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"success": False, "error": str(exc)})
    except Exception as exc:
        return JSONResponse(
            status_code=502,
            content={"success": False, "error": f"Speech synthesis failed: {exc}"},
        )
    return Response(content=audio_bytes, media_type=media_type)
