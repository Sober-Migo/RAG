"""PDF loading and chunking."""
from pathlib import Path
from typing import List, Union

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_OVERLAP, CHUNK_SIZE


def load_pdf(file_path: Union[str, Path]) -> List[Document]:
    """Load a PDF into LangChain documents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    loader = PyPDFLoader(str(path))
    return loader.load()


def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into overlapping chunks for retrieval."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(documents)
