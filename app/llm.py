"""LLM provider — Hugging Face only (no Watsonx).

Uses AutoModelForSeq2SeqLM directly so we never import transformers.pipelines,
which pulls torchvision and breaks on Kaggle when torch/torchvision versions drift.
"""
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from app.config import (
    HF_TOKEN,
    LLM_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
    USE_INFERENCE_API,
)


class LocalSeq2SeqLLM(LLM):
    """Thin LangChain wrapper around a local HF seq2seq model (e.g. flan-t5)."""

    model_name: str = Field(default=LLM_MODEL)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS)
    temperature: float = Field(default=TEMPERATURE)
    _tokenizer: Any = None
    _model: Any = None

    @property
    def _llm_type(self) -> str:
        return "local-seq2seq"

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.eval()

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        import torch

        self._load()
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": bool(self.temperature and self.temperature > 0),
        }
        if gen_kwargs["do_sample"]:
            gen_kwargs["temperature"] = float(self.temperature)

        with torch.no_grad():
            output_ids = self._model.generate(**inputs, **gen_kwargs)
        text = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return text.strip()


def get_llm():
    """
    Return an LLM backed by Hugging Face.

    Priority:
    1. HF Inference API if USE_INFERENCE_API=true and HF_TOKEN is set
    2. Local seq2seq model (default) — no transformers.pipeline import
    """
    if USE_INFERENCE_API and HF_TOKEN:
        from langchain_huggingface import HuggingFaceEndpoint

        return HuggingFaceEndpoint(
            repo_id=LLM_MODEL,
            huggingfacehub_api_token=HF_TOKEN,
            temperature=TEMPERATURE,
            max_new_tokens=MAX_NEW_TOKENS,
            timeout=120,
        )

    return LocalSeq2SeqLLM(
        model_name=LLM_MODEL,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
