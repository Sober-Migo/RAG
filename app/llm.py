"""LLM backends: Groq (free tier) > Hugging Face chat > error."""
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    HF_TOKEN,
    LLM_MODEL,
    MAX_NEW_TOKENS,
    TEMPERATURE,
)


SYSTEM_MSG = (
    "You answer questions using only the provided context. "
    "If the answer is not in the context, say you do not know. "
    "Be concise and factual."
)


class GroqLLM(LLM):
    """Groq chat completions — free tier with API key."""

    api_key: str = Field(...)
    model: str = Field(default=GROQ_MODEL)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS)
    temperature: float = Field(default=TEMPERATURE)

    @property
    def _llm_type(self) -> str:
        return "groq-chat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        from groq import Groq

        client = Groq(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else 0.1,
        )
        text = resp.choices[0].message.content or ""
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return text.strip()


class HFInferenceLLM(LLM):
    """HF Inference Providers chat (often requires credits)."""

    model: str = Field(default=LLM_MODEL)
    token: Optional[str] = Field(default=None)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS)
    temperature: float = Field(default=TEMPERATURE)

    @property
    def _llm_type(self) -> str:
        return "hf-chat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if not self.token:
            raise RuntimeError("HF_TOKEN is required for HF generative answers.")

        from huggingface_hub import InferenceClient

        messages = [
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ]
        last_err: Optional[Exception] = None
        text: Optional[str] = None

        for provider in (None, "hf-inference"):
            try:
                kw = {"token": self.token}
                if provider:
                    kw["provider"] = provider
                client = InferenceClient(**kw)
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else 0.1,
                )
                text = resp.choices[0].message.content
                if text:
                    break
            except Exception as e:
                last_err = e

        if not text:
            raise RuntimeError(
                f"HF chat failed for model={self.model!r}. "
                f"Last error: {type(last_err).__name__}: {last_err}"
            )

        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return text.strip()


def get_llm() -> LLM:
    """Prefer Groq (free) → HF (optional) → raise."""
    if GROQ_API_KEY:
        return GroqLLM(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
    if HF_TOKEN:
        return HFInferenceLLM(
            model=LLM_MODEL,
            token=HF_TOKEN,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
    raise RuntimeError(
        "No LLM key set. Set GROQ_API_KEY (free) or HF_TOKEN, "
        "or use extractive mode without a key."
    )
