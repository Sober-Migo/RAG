"""LLM via Hugging Face Inference API only (no local torch/transformers)."""
from langchain_huggingface import HuggingFaceEndpoint

from app.config import HF_TOKEN, LLM_MODEL, MAX_NEW_TOKENS, TEMPERATURE


def get_llm():
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is required. Set it in the environment or .env file. "
            "Create a token at https://huggingface.co/settings/tokens"
        )
    return HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        huggingfacehub_api_token=HF_TOKEN,
        temperature=TEMPERATURE,
        max_new_tokens=MAX_NEW_TOKENS,
        timeout=120,
    )
