"""Embedding model — Hugging Face sentence-transformers (local)."""
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a local embedding model. No API key required."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
