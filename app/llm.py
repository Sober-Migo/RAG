"""LLM via modern huggingface_hub.InferenceClient (router — not api-inference)."""
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from app.config import HF_TOKEN, LLM_MODEL, MAX_NEW_TOKENS, TEMPERATURE


class HFInferenceLLM(LLM):
    """
    Uses huggingface_hub>=0.28 InferenceClient.

    Tries in order:
      1) chat_completion (OpenAI-style / router)
      2) text_generation
    Never calls the deprecated api-inference.huggingface.co host directly.
    """

    model: str = Field(default=LLM_MODEL)
    token: Optional[str] = Field(default=None)
    max_new_tokens: int = Field(default=MAX_NEW_TOKENS)
    temperature: float = Field(default=TEMPERATURE)

    @property
    def _llm_type(self) -> str:
        return "hf-inference-client"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if not self.token:
            raise RuntimeError(
                "HF_TOKEN is required for generative answers. "
                "Set HF_TOKEN or use extractive mode without a token."
            )

        from huggingface_hub import InferenceClient

        client = InferenceClient(token=self.token)
        text: Optional[str] = None
        last_err: Optional[Exception] = None

        # 1) Chat completions (works with many instruct models on the router)
        try:
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_new_tokens,
                temperature=self.temperature if self.temperature > 0 else None,
            )
            text = resp.choices[0].message.content
        except Exception as e:
            last_err = e

        # 2) Classic text_generation
        if text is None:
            try:
                text = client.text_generation(
                    prompt,
                    model=self.model,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else None,
                    do_sample=bool(self.temperature and self.temperature > 0),
                )
            except Exception as e:
                last_err = e

        if text is None:
            raise RuntimeError(
                f"HF Inference failed for model={self.model!r}. "
                f"Last error: {type(last_err).__name__}: {last_err}"
            )

        if stop:
            for s in stop:
                if s in text:
                    text = text.split(s)[0]
        return text.strip()


def get_llm() -> HFInferenceLLM:
    return HFInferenceLLM(
        model=LLM_MODEL,
        token=HF_TOKEN,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
