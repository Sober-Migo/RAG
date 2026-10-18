"""Local ONNX embeddings via Chroma — free, no HF Inference API, no torch."""
from typing import List

from chromadb.utils import embedding_functions
from langchain_core.embeddings import Embeddings


class ChromaDefaultEmbeddings(Embeddings):
    """Wraps Chroma's DefaultEmbeddingFunction (all-MiniLM-L6-v2 ONNX)."""

    def __init__(self) -> None:
        self._ef = embedding_functions.DefaultEmbeddingFunction()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._ef([text])[0]


def get_embeddings() -> Embeddings:
    return ChromaDefaultEmbeddings()
