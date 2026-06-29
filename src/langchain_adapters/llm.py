from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable, RunnableConfig

from src.rag.answer_generator import DEFAULT_APP_CONFIG, LLMClient, OllamaClient, load_llm_config
from src.rag.prompts import SYSTEM_PROMPT


MOCK_RESPONSE = "Mock 模式仅验证本地 LangChain 调用链，不生成超出证据的新结论。"


def _message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(str(block) for block in content)


def prompt_parts(value: Any) -> tuple[str, str]:
    if hasattr(value, "to_messages"):
        messages = value.to_messages()
    elif isinstance(value, list) and all(isinstance(item, BaseMessage) for item in value):
        messages = value
    else:
        return SYSTEM_PROMPT, str(value)

    system_parts = [_message_text(message) for message in messages if message.type == "system"]
    user_parts = [_message_text(message) for message in messages if message.type != "system"]
    return "\n\n".join(system_parts) or SYSTEM_PROMPT, "\n\n".join(user_parts)


class OllamaLLMAdapter(Runnable[Any, str]):
    """Runnable wrapper for the existing local Ollama client with mock fallback."""

    def __init__(
        self,
        mode: str = "ollama",
        llm_client: LLMClient | None = None,
        fallback_to_mock: bool = True,
        app_config: Path = DEFAULT_APP_CONFIG,
    ) -> None:
        if mode not in {"ollama", "mock"}:
            raise ValueError("LLM adapter mode must be `ollama` or `mock`.")
        self.requested_mode = mode
        self.active_mode = mode
        self.fallback_to_mock = fallback_to_mock
        self.fallback_reason: str | None = None
        if mode == "ollama" and llm_client is None:
            model, base_url = load_llm_config(app_config)
            llm_client = OllamaClient(model=model, base_url=base_url)
        self.llm_client = llm_client

    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> str:
        del config, kwargs
        system_prompt, user_prompt = prompt_parts(input)
        return self.generate(system_prompt, user_prompt)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if self.active_mode == "mock":
            return MOCK_RESPONSE
        try:
            if self.llm_client is None:
                raise RuntimeError("Ollama client is not initialized.")
            return self.llm_client.generate(system_prompt, user_prompt)
        except Exception as exc:
            if not self.fallback_to_mock:
                raise
            self.active_mode = "mock"
            self.fallback_reason = f"{type(exc).__name__}: {exc}"
            return MOCK_RESPONSE
