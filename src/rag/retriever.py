from __future__ import annotations

import hashlib
from pathlib import Path

from langchain_chroma import Chroma

from src.config import Settings
from src.rag.document_loader import load_support_documents, split_support_documents
from src.rag.embeddings import build_embeddings
from src.utils.errors import ComponentNotReadyError

# Below this relevance score, we treat the chunk as not actually relevant rather
# than force the LLM to fabricate an answer from a weak match.
_RELEVANCE_THRESHOLD = 0.2


class KnowledgeRetriever:
    """Persistent Chroma retrieval scaffold with stable output contracts."""

    def __init__(self, settings: Settings, documents_dir: Path) -> None:
        self.settings = settings
        self.documents_dir = documents_dir
        self._store: Chroma | None = None

    async def initialize(self) -> None:
        documents = split_support_documents(load_support_documents(self.documents_dir))
        embeddings = build_embeddings(self.settings)

        persist_directory = Path(self.settings.vector_db_path)
        persist_directory.mkdir(parents=True, exist_ok=True)

        try:
            store = Chroma(
                collection_name=self.settings.rag_collection,
                embedding_function=embeddings,
                persist_directory=str(persist_directory),
                collection_metadata={"hnsw:space": "cosine"},
            )

            ids = [self._chunk_id(doc) for doc in documents]
            existing_ids: set[str] = set()
            if ids:
                existing = store.get(ids=ids)
                existing_ids = set(existing.get("ids", []))

            new_docs = []
            new_ids = []
            for doc, doc_id in zip(documents, ids):
                if doc_id not in existing_ids:
                    new_docs.append(doc)
                    new_ids.append(doc_id)

            if new_docs:
                store.add_documents(new_docs, ids=new_ids)
        except Exception as exc:  # surfaces as a failed startup, not fake readiness
            raise ComponentNotReadyError(f"Failed to initialize knowledge retriever: {exc}") from exc

        self._store = store

    @staticmethod
    def _chunk_id(document) -> str:
        # Stable ID derived from source + content so re-running initialize() (e.g.
        # on every API restart) does not duplicate the same chunk in Chroma.
        digest = hashlib.sha256(document.page_content.encode("utf-8")).hexdigest()[:16]
        source = document.metadata.get("source", "unknown")
        return f"{source}::{digest}"

    async def search(self, query: str, limit: int | None = None) -> list[dict[str, str]]:
        if self._store is None:
            raise ComponentNotReadyError("Knowledge retriever is not initialized")

        query = (query or "").strip()
        if not query:
            return []

        k = limit if limit is not None else self.settings.rag_top_k
        try:
            results = self._store.similarity_search_with_relevance_scores(query, k=k)
        except Exception as exc:
            raise ComponentNotReadyError(f"Knowledge retrieval failed: {exc}") from exc

        output: list[dict[str, str]] = []
        for doc, score in results:
            if score is not None and score < _RELEVANCE_THRESHOLD:
                continue
            output.append(
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                }
            )
        return output
