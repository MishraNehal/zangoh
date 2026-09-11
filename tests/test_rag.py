from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

import pytest

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.retriever import KnowledgeRetriever
from src.utils.errors import ComponentNotReadyError


def test_loader_preserves_source_names(knowledge_dir) -> None:
    documents = load_support_documents(knowledge_dir)
    assert {item.metadata["source"] for item in documents} == {
        "accounts.md",
        "payments.md",
        "returns.md",
        "shipping.md",
    }


def test_splitter_keeps_source_metadata(knowledge_dir) -> None:
    chunks = split_support_documents(load_support_documents(knowledge_dir))
    assert chunks
    assert all(chunk.metadata.get("source", "").endswith(".md") for chunk in chunks)


def test_loader_raises_for_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_support_documents(tmp_path / "does-not-exist")


def test_loader_raises_when_no_markdown_files_present(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty-kb"
    empty_dir.mkdir()
    with pytest.raises(ValueError):
        load_support_documents(empty_dir)


class FakeEmbeddings:
    """Deterministic, dependency-free stand-in for HuggingFaceEmbeddings."""

    _DIM = 64

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._DIM
        for token in self._tokens(text):
            idx = int(hashlib.sha256(token.encode("utf-8")).hexdigest(), 16) % self._DIM
            vector[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_api_key="not-required",
        llm_model="test-model",
        vector_db_path=str(tmp_path / "vector_db"),
        embedding_model="fake",
        rag_collection="test-support",
        rag_top_k=3,
        api_host="127.0.0.1",
        api_port=8000,
        streamlit_host="127.0.0.1",
        streamlit_port=8501,
    )


@pytest.fixture
def fake_retriever(monkeypatch, tmp_path: Path, knowledge_dir: Path) -> KnowledgeRetriever:
    monkeypatch.setattr("src.rag.retriever.build_embeddings", lambda settings: FakeEmbeddings())
    return KnowledgeRetriever(_settings(tmp_path), knowledge_dir)


@pytest.mark.asyncio
async def test_search_before_initialize_raises(fake_retriever: KnowledgeRetriever) -> None:
    with pytest.raises(ComponentNotReadyError):
        await fake_retriever.search("How long does shipping take?")


@pytest.mark.asyncio
async def test_search_returns_relevant_chunk_with_source(fake_retriever: KnowledgeRetriever) -> None:
    await fake_retriever.initialize()
    results = await fake_retriever.search("How long does standard shipping take?")

    assert results
    assert all({"content", "source"} <= result.keys() for result in results)
    assert any(result["source"] == "shipping.md" for result in results)


@pytest.mark.asyncio
async def test_search_blank_query_returns_no_results(fake_retriever: KnowledgeRetriever) -> None:
    await fake_retriever.initialize()
    assert await fake_retriever.search("   ") == []


@pytest.mark.asyncio
async def test_search_respects_explicit_limit(fake_retriever: KnowledgeRetriever) -> None:
    await fake_retriever.initialize()
    results = await fake_retriever.search("refund return policy", limit=1)
    assert len(results) <= 1


@pytest.mark.asyncio
async def test_initialize_is_idempotent_and_does_not_duplicate_chunks(
    fake_retriever: KnowledgeRetriever,
) -> None:
    await fake_retriever.initialize()
    await fake_retriever.initialize()

    collection = fake_retriever._store._collection  # type: ignore[attr-defined]
    expected_chunk_count = len(
        split_support_documents(load_support_documents(fake_retriever.documents_dir))
    )
    assert collection.count() == expected_chunk_count