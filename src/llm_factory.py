"""Utilities for constructing chat models based on configuration."""

from __future__ import annotations

from typing import Any, Optional

from langchain_community.llms import HuggingFaceEndpoint
from langchain_openai import ChatOpenAI

from config.config import config


class LLMConfigurationError(RuntimeError):
    """Raised when the language model configuration is invalid."""


def create_chat_model(
    *, temperature: float, model_name: Optional[str] = None
):
    """Return a chat-compatible language model based on the configured provider.

    Args:
        temperature: Sampling temperature for the model.
        model_name: Optional override for the model identifier.

    Returns:
        A LangChain ``BaseLanguageModel`` instance ready for invocation.

    Raises:
        LLMConfigurationError: If the provider is unknown or required
            credentials are missing.
    """

    provider = (config.LLM_PROVIDER or "open_source").lower()

    if provider == "openai":
        if not config.OPENAI_API_KEY:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required when LLM_PROVIDER is set to 'openai'."
            )

        return ChatOpenAI(
            temperature=temperature,
            model=model_name or config.LLM_MODEL,
            openai_api_key=config.OPENAI_API_KEY,
        )

    if provider in {"open_source", "mistral", "huggingface"}:
        if not config.HUGGINGFACEHUB_API_TOKEN:
            raise LLMConfigurationError(
                "HUGGINGFACEHUB_API_TOKEN is required for open source models."
            )

        return HuggingFaceEndpoint(
            repo_id=model_name or config.OPEN_SOURCE_MODEL,
            huggingfacehub_api_token=config.HUGGINGFACEHUB_API_TOKEN,
            task="text-generation",
            model_kwargs={
                "temperature": temperature,
                "max_new_tokens": config.LLM_MAX_NEW_TOKENS,
                "do_sample": temperature > 0,
                "return_full_text": False,
            },
        )

    raise LLMConfigurationError(
        f"Unsupported LLM provider '{config.LLM_PROVIDER}'. "
        "Set LLM_PROVIDER to 'openai' or 'open_source'."
    )


def extract_response_content(response: Any) -> str:
    """Normalize model responses to a plain string."""

    if hasattr(response, "content"):
        return response.content  # type: ignore[return-value]

    if isinstance(response, dict) and "content" in response:
        return str(response["content"])

    return str(response)
