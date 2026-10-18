"""LLM via Hugging Face InferenceClient (router). Optional — extractive fallback exists."""
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from app.config import HF_TOKEN, LLM_MODEL, MAX_NEW_TOKENS, TEMPERATURE


class HFRouterLLM(LLM):
    """LangChain LLM using huggingface_hub.InferenceClient (new router)."""

    model: str = Field(default=LLM_MODEL)
    token: Optional[str] = Field(default=None)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS)
    temperature: float = Field(default=TEMPERATURE)

    @property
    def _llm_type(self) -> str:
        return "hf-router"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        from huggingface_hub import InferenceClient

        if not self.token:
            raise RuntimeError("HF_TOKEN is required for generative answers.")

        client = InferenceClient(token=self.token)
        # text_generation works for many instruct models on the router
        text = client.text_generation(
            prompt,
            model=self.model,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature if self.temperature > 0 else None,
            do_sample=self.temperature > 0,
        )
        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return text.strip()


def get_llm() -> HFRouterLLM:
    return HFRouterLLM(model=LLM_MODEL, token=HF_TOKEN)
