from langchain_huggingface import HuggingFaceEmbeddings

from src.config import Settings


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    """Create the configured local/open-source embedding adapter."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)
