from pathlib import Path

import pytest

from src.config import Settings
from src.pipeline import SupportPipeline
from src.utils.errors import ComponentNotReadyError


def settings() -> Settings:
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
    pipeline = SupportPipeline(settings(), tmp_path)
    with pytest.raises(ComponentNotReadyError):
        await pipeline.process("session-1", "hello")
