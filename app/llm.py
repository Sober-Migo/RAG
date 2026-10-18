"""LLM provider — Hugging Face only (no Watsonx)."""
from langchain_core.language_models.llms import BaseLLM
from langchain_huggingface import HuggingFaceEndpoint, HuggingFacePipeline
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from app.config import (
    HF_TOKEN,
    LLM_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    USE_INFERENCE_API,
)


def get_llm() -> BaseLLM:
    """
    Return an LLM backed by Hugging Face.

    Priority:
    1. HF Inference API if USE_INFERENCE_API=true and HF_TOKEN is set
    2. Local transformers pipeline (default — works offline after first download)
    """
    if USE_INFERENCE_API and HF_TOKEN:
        return HuggingFaceEndpoint(
            repo_id=LLM_MODEL,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            timeout=120,
        )

    # Local pipeline — encoder-decoder (flan-t5 family)
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    gen_kwargs = {
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    if TEMPERATURE and TEMPERATURE > 0:
        gen_kwargs["temperature"] = TEMPERATURE
        gen_kwargs["do_sample"] = True
    else:
        gen_kwargs["do_sample"] = False

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        **gen_kwargs,
    )
    return HuggingFacePipeline(pipeline=pipe)
