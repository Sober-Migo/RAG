"""Retrieval-QA chain (LCEL — stable on LangChain 0.2.x / Kaggle / Colab)."""
from pathlib import Path
from typing import Optional, Union

from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from app.documents import load_pdf, split_documents
from app.llm import get_llm
from app.vectorstore import get_retriever


QA_PROMPT = PromptTemplate.from_template(
    """Use only the context below to answer the question.
If the answer is not in the context, say you do not know.

Context:
{context}

Question: {question}

Answer:"""
)


def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def build_qa_chain(file_path: Union[str, Path], llm: Optional[BaseLLM] = None):
    """Load PDF → chunk → embed → retriever → LCEL QA chain."""
    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    retriever = get_retriever(chunks)
    llm = llm or get_llm()

    chain = (
        {
            "context": retriever | _format_docs,
            "question": RunnablePassthrough(),
        }
        | QA_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


def answer_question(file_path: Union[str, Path], query: str) -> str:
    """End-to-end: PDF + question → grounded answer string."""
    if not file_path:
        return "Please upload a PDF first."
    if not query or not str(query).strip():
        return "Please enter a question about the document."
    try:
        chain = build_qa_chain(file_path)
        return chain.invoke(query.strip())
    except Exception as e:
        return f"Error while answering: {type(e).__name__}: {e}"
