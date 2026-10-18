"""LLM via huggingface_hub.InferenceClient — chat/conversational only."""
from typing import Any, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from pydantic import Field

from app.config import HF_TOKEN, LLM_MODEL, MAX_NEW_TOKENS, TEMPERATURE


class HFInferenceLLM(LLM):
    """
    Instruct models on HF Inference Providers often only support the
    *conversational* task (chat.completions), not text-generation.
    This wrapper uses chat only.
    """

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
            raise RuntimeError("HF_TOKEN is required for generative answers.")

        from huggingface_hub import InferenceClient

        messages = [
            {
                "role": "system",
                "content": (
                    "You answer questions using only the provided context. "
                    "If the answer is not in the context, say you do not know."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        last_err: Optional[Exception] = None
        text: Optional[str] = None

        # Try default routing, then explicit hf-inference provider
        for provider in (None, "hf-inference"):
            try:
                client_kwargs = {"token": self.token}
                if provider:
                    client_kwargs["provider"] = provider
                client = InferenceClient(**client_kwargs)

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
                continue

        # Older method name on some hub versions
        if text is None:
            try:
                client = InferenceClient(token=self.token)
                resp = client.chat_completion(
                    messages=messages,
                    model=self.model,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature if self.temperature > 0 else 0.1,
                )
                # response shape varies by version
                if isinstance(resp, dict):
                    text = resp.get("choices", [{}])[0].get("message", {}).get("content")
                else:
                    text = resp.choices[0].message.content
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


def get_llm() -> HFInferenceLLM:
    return HFInferenceLLM(
        model=LLM_MODEL,
        token=HF_TOKEN,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )
