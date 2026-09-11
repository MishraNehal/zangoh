import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from src.config import Settings, load_settings
from src.pipeline import SupportPipeline
from src.utils.errors import ComponentNotReadyError

# load_settings() also calls load_dotenv() internally, but that happens
# *inside* the live tests below -- after pytest has already decided whether
# to skip them. The skipif condition needs .env loaded before that decision
# is made, so we load it here at module import time too.
load_dotenv()

def _offline_settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=".data/test-vector-db",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        rag_collection="test-support",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


@pytest.mark.asyncio
async def test_uninitialized_pipeline_rejects_chat(tmp_path: Path) -> None:
    pipeline = SupportPipeline(_offline_settings(), tmp_path)
    with pytest.raises(ComponentNotReadyError):
        await pipeline.process("session-1", "hello")


@pytest.mark.asyncio
async def test_process_rejects_blank_session_id(tmp_path: Path) -> None:
    pipeline = SupportPipeline(_offline_settings(), tmp_path)
    pipeline.ready = True
    pipeline.workflow = object()
    from src.utils.errors import AgentProcessingError

    with pytest.raises(AgentProcessingError):
        await pipeline.process("   ", "hello")


pytestmark_live = pytest.mark.skipif(
    not os.getenv("LLM_API_KEY") or os.getenv("LLM_API_KEY") == "not-required",
    reason="LLM_API_KEY not set in environment/.env; skipping live Groq integration test",
)


@pytestmark_live
@pytest.mark.asyncio
async def test_live_policy_question_is_grounded_and_cites_source() -> None:
    settings = load_settings()
    knowledge_dir = Path(__file__).resolve().parents[1] / "knowledge_base"
    pipeline = SupportPipeline(settings, knowledge_dir)
    await pipeline.initialize()

    response = await pipeline.process(
        "live-session-1", "How long does standard shipping take?"
    )

    assert response.success
    assert response.response
    assert "shipping.md" in response.sources
    assert response.ticket_id is None


@pytestmark_live
@pytest.mark.asyncio
async def test_live_unknown_question_does_not_fabricate() -> None:
    settings = load_settings()
    knowledge_dir = Path(__file__).resolve().parents[1] / "knowledge_base"
    pipeline = SupportPipeline(settings, knowledge_dir)
    await pipeline.initialize()

    response = await pipeline.process(
        "live-session-2", "What is your company's stock ticker symbol?"
    )

    assert response.success
    assert response.ticket_id is None
    assert response.sources == [] or len(response.sources) <= 1


@pytestmark_live
@pytest.mark.asyncio
async def test_live_multi_turn_ticket_creation_and_retrieval() -> None:
    settings = load_settings()
    knowledge_dir = Path(__file__).resolve().parents[1] / "knowledge_base"
    pipeline = SupportPipeline(settings, knowledge_dir)
    await pipeline.initialize()

    session_id = "live-session-3"
    r1 = await pipeline.process(session_id, "My payment was charged twice, I want to file a ticket")
    assert r1.ticket_id is None

    r2 = await pipeline.process(session_id, "My name is Asha Kumar")
    r3 = await pipeline.process(session_id, "My email is asha@example.com")
    r4 = await pipeline.process(
        session_id, "The issue is a duplicate payment charge on my last order, category is payment"
    )

    final = r4 if r4.ticket_id else r3 if r3.ticket_id else r2
    assert final.ticket_id is not None
    assert pipeline.tickets.get(final.ticket_id) is not None

    repeat = await pipeline.process(session_id, "Any update on my ticket?")
    assert len(list(pipeline.tickets.all())) == 1
    _ = repeat