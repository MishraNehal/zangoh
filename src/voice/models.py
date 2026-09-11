from pydantic import BaseModel, Field


class TranscriptionResponse(BaseModel):
    success: bool = True
    transcript: str = Field(min_length=1)
    processing_time_ms: int = Field(ge=0)


class SynthesisRequest(BaseModel):
    message_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=4000)


class VoiceErrorResponse(BaseModel):
    """Shape used for both /voice/transcribe and /voice/synthesize failures.

    Matches the ``transcription_error`` example supplied in
    example_responses.json: a stable, low-drama JSON error the Streamlit
    client can render without special-casing per endpoint.
    """

    success: bool = False
    error: str
