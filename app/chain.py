"""RAG chain: retrieve + chat generation, with free extractive fallback."""
from pathlib import Path
from typing import Union

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.config import EXTRACTIVE_FALLBACK, HF_TOKEN
from app.documents import load_pdf, split_documents
from app.llm import get_llm
from app.vectorstore import get_retriever


# Prompt body is sent as the user message; system message is set in llm.py
QA_PROMPT = PromptTemplate.from_template(
    """Context:
{context}

Question: {question}

Answer using only the context above."""
)


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def _extractive_answer(file_path: Union[str, Path], query: str) -> str:
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    retriever = get_retriever(chunks, k=3)
    docs = retriever.invoke(query)
    if not docs:
        return "No relevant passages found in the document."
    parts = [
        f"[Passage {i + 1}]\n{d.page_content.strip()}" for i, d in enumerate(docs)
    ]
    return (
        "(Extractive mode — free, no LLM call)\n\n"
        "Most relevant passages for your question:\n\n"
        + "\n\n---\n\n".join(parts)
    )


def build_generative_chain(file_path: Union[str, Path]):
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    retriever = get_retriever(chunks)
    llm = get_llm()
    return (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )


def answer_question(file_path: Union[str, Path], query: str) -> str:
    if not file_path:
        return "Please upload a PDF first."
    if not query or not str(query).strip():
        return "Please enter a question about the document."

    q = query.strip()

    if HF_TOKEN:
        try:
            return build_generative_chain(file_path).invoke(q)
        except Exception as e:
            if not EXTRACTIVE_FALLBACK:
                return f"Error while answering: {type(e).__name__}: {e}"
            return (
                f"Generative LLM unavailable ({type(e).__name__}: {e}).\n"
                f"Falling back to free extractive retrieval:\n\n"
                f"{_extractive_answer(file_path, q)}"
            )

    return _extractive_answer(file_path, q)
