"""Chroma vector store helpers."""
from typing import List

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import Chroma

from app.embeddings import get_embeddings


def build_vectorstore(chunks: List[Document]) -> Chroma:
    embeddings = get_embeddings()
    return Chroma.from_documents(documents=chunks, embedding=embeddings)


def get_retriever(chunks: List[Document], k: int = 4) -> VectorStoreRetriever:
    vectordb = build_vectorstore(chunks)
    return vectordb.as_retriever(search_kwargs={"k": k})
