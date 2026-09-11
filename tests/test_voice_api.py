import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.voice.pipeline import VoicePipeline
from tests.test_voice_pipeline import FakeSTT, FakeTTS


@pytest.fixture
def client(monkeypatch):
    """A TestClient wired to fake STT/TTS adapters instead of Groq/Edge TTS.

    Deliberately NOT used as a context manager (`with TestClient(...) as c`),
    so FastAPI's lifespan/startup never runs here -- matching how the rest of
    this suite avoids spinning up the real RAG/LLM pipeline just to test one
    endpoint. That also means `pipeline.ready` stays False, which the last
    test below uses to prove the /chat contract is untouched.
    """
    fake_voice_pipeline = VoicePipeline(FakeSTT(), FakeTTS())
    monkeypatch.setattr(server, "voice_pipeline", fake_voice_pipeline)
    return TestClient(server.app)


def test_transcribe_success(client: TestClient) -> None:
    response = client.post(
        "/voice/transcribe",
        files={"audio": ("clip.wav", b"fake-bytes", "audio/wav")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["transcript"] == "My payment was charged twice"
    assert body["processing_time_ms"] >= 0


def test_transcribe_rejects_empty_audio(client: TestClient) -> None:
    response = client.post(
        "/voice/transcribe",
        files={"audio": ("clip.wav", b"", "audio/wav")},
    )

    assert response.status_code == 422
    body = response.json()
    assert body["success"] is False
    assert "error" in body


def test_transcribe_handles_stt_failure(monkeypatch) -> None:
    broken_pipeline = VoicePipeline(FakeSTT(error=RuntimeError("upstream STT error")), FakeTTS())
    monkeypatch.setattr(server, "voice_pipeline", broken_pipeline)
    client = TestClient(server.app)

    response = client.post(
        "/voice/transcribe",
        files={"audio": ("clip.wav", b"fake-bytes", "audio/wav")},
    )

    assert response.status_code == 502
    body = response.json()
    assert body["success"] is False
    assert "upstream STT error" in body["error"]


def test_synthesize_success(client: TestClient) -> None:
    response = client.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": "Your ticket has been created."},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/mpeg")
    assert response.content == b"fake-audio"


def test_synthesize_rejects_blank_text(client: TestClient) -> None:
    response = client.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": ""},
    )

    # Pydantic's min_length=1 on SynthesisRequest.text rejects this before it
    # ever reaches the VoicePipeline.
    assert response.status_code == 422


def test_synthesize_handles_tts_failure(monkeypatch) -> None:
    broken_pipeline = VoicePipeline(FakeSTT(), FakeTTS(error=RuntimeError("upstream TTS error")))
    monkeypatch.setattr(server, "voice_pipeline", broken_pipeline)
    client = TestClient(server.app)

    response = client.post(
        "/voice/synthesize",
        json={"message_id": "msg-1", "text": "Hello there"},
    )

    assert response.status_code == 502
    body = response.json()
    assert body["success"] is False
    assert "upstream TTS error" in body["error"]


def test_voice_and_message_id_round_trip(client: TestClient) -> None:
    """The synthesis request carries a message_id end-to-end so the caller
    can associate the returned audio with the right chat message, even
    though the server itself doesn't need to inspect it."""
    response = client.post(
        "/voice/synthesize",
        json={"message_id": "assistant-message-42", "text": "Ticket CST-2026-0001 has been created."},
    )
    assert response.status_code == 200


def test_typed_chat_contract_is_unmodified_by_voice_routes(client: TestClient) -> None:
    # Without lifespan startup the pipeline is never marked ready, so /chat
    # should still return its existing 503 contract -- proving voice wiring
    # didn't change /chat's request/response shape or error handling.
    response = client.post("/chat", json={"session_id": "s1", "message": "hi"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Support pipeline is not ready"
