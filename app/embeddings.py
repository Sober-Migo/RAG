"""Embeddings via Hugging Face Inference API (no local model)."""
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from app.config import EMBEDDING_MODEL, HF_TOKEN


def get_embeddings():
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is required. Set it in the environment or .env file. "
            "Create a token at https://huggingface.co/settings/tokens"
        )
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_TOKEN,
        model_name=EMBEDDING_MODEL,
    )
