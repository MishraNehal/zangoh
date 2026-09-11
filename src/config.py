from __future__ import annotations

from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """Environment-controlled model and retrieval settings.

    Add new provider or embedding options here when required. Keep safe defaults
    suitable for local development and document every new variable in both
    ``.env.example`` and the project README.
    """
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    vector_db_path: str
    embedding_model: str
    rag_collection: str
    rag_top_k: int
    api_host: str
    api_port: int
    streamlit_host: str
    streamlit_port: int
    # Voice enhancement fields default so existing call sites (and tests) that
    # construct Settings without them keep working unchanged.
    stt_model: str = "whisper-large-v3-turbo"
    stt_api_key: str = ""
    tts_voice: str = "en-US-AriaNeural"


def load_settings() -> Settings:
    """Load the small, provider-neutral configuration used by the starter."""
    load_dotenv()
    # Configuration rules:
    # - Never commit a real API key.
    # - Avoid absolute developer-machine paths.
    # - Keep model identity configurable so reviewers can use another compatible
    #   open-source endpoint without editing source code.
    # - Validate required configuration during startup, not on the first request.
    return Settings(
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_api_key=os.getenv("LLM_API_KEY", "not-required"),
        llm_model=os.getenv("LLM_MODEL", "qwen2.5:3b"),
        vector_db_path=os.getenv("VECTOR_DB_PATH", ".data/vector_db"),
        embedding_model=os.getenv(
            "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        rag_collection=os.getenv("RAG_COLLECTION", "customer-support"),
        rag_top_k=int(os.getenv("RAG_TOP_K", "3")),
        api_host=os.getenv("API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("API_PORT", "8000")),
        streamlit_host=os.getenv("STREAMLIT_HOST", "127.0.0.1"),
        streamlit_port=int(os.getenv("STREAMLIT_PORT", "8501")),
        # Voice enhancement (mid-session requirement). STT_API_KEY defaults to
        # empty so the adapter falls back to LLM_API_KEY -- both point at the
        # same Groq account unless a reviewer wants to separate them.
        stt_model=os.getenv("STT_MODEL", "whisper-large-v3-turbo"),
        stt_api_key=os.getenv("STT_API_KEY", ""),
        tts_voice=os.getenv("TTS_VOICE", "en-US-AriaNeural"),
    )
