"""LLM backends: Groq (optional) > HF OpenAI-compatible router > error."""
from typing import Any, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from app.config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    HF_ROUTER_BASE_URL,
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
            stop=stop,
        )
        return (resp.choices[0].message.content or "").strip()


class HuggingFaceRouterLLM(LLM):
    """
    OpenAI-compatible client against Hugging Face Inference Providers router.

    base_url = https://router.huggingface.co/v1
    (never api-inference.huggingface.co — that host is dead)
    """

    model_id: str = Field(default=LLM_MODEL)
    token: str = Field(...)
    base_url: str = Field(default=HF_ROUTER_BASE_URL)
    max_tokens: int = Field(default=MAX_NEW_TOKENS)
    temperature: float = Field(default=TEMPERATURE)

    @property
    def _llm_type(self) -> str:
        return "huggingface_router"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.token)
        response = client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=stop,
        )
        return (response.choices[0].message.content or "").strip()

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_id": self.model_id,
            "base_url": self.base_url,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }


def get_llm() -> LLM:
    """Groq if key set, else HF router if token set."""
    if GROQ_API_KEY:
        return GroqLLM(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
    if HF_TOKEN:
        return HuggingFaceRouterLLM(
            model_id=LLM_MODEL,
            token=HF_TOKEN,
            base_url=HF_ROUTER_BASE_URL,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
        )
    raise RuntimeError(
        "No LLM key set. Set GROQ_API_KEY and/or HF_TOKEN, "
        "or run without keys for extractive mode."
    )
