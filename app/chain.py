"""Retrieval-QA chain."""
from pathlib import Path
from typing import Optional, Union

from langchain.chains import RetrievalQA
from langchain_core.language_models.llms import BaseLLM

from app.documents import load_pdf, split_documents
from app.llm import get_llm
from app.vectorstore import get_retriever


def build_qa_chain(file_path: Union[str, Path], llm: Optional[BaseLLM] = None) -> RetrievalQA:
    """Load PDF → chunk → embed → retriever → RetrievalQA chain."""
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    retriever = get_retriever(chunks)
    llm = llm or get_llm()
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )


def answer_question(file_path: Union[str, Path], query: str) -> str:
    """End-to-end: PDF + question → grounded answer string."""
    if not query or not str(query).strip():
        return "Please enter a question about the document."
    chain = build_qa_chain(file_path)
    result = chain.invoke({"query": query.strip()})
    return result.get("result", str(result))
