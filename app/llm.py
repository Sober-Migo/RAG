"""LLM — Groq only (free tier)."""
from typing import Any, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from app.config import GROQ_API_KEY, GROQ_MODEL, MAX_NEW_TOKENS, TEMPERATURE


SYSTEM_MSG = (
    "You answer questions using only the provided context. "
    "If the answer is not in the context, say you do not know. "
    "Be concise and factual."
)


class GroqLLM(LLM):
    """Groq chat completions."""

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


def get_llm() -> GroqLLM:
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is required for generative answers. "
            "Get a free key at https://console.groq.com/keys"
        )
    return GroqLLM(
        api_key=GROQ_API_KEY,
        model=GROQ_MODEL,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
