"""Utility helpers for the recipe chatbot backend."""

import os
from functools import lru_cache
from pathlib import Path
from typing import List, Dict

from openai import OpenAI
from dotenv import load_dotenv

# Ensure the .env file is loaded as early as possible.
load_dotenv(override=False)

# --- Constants -------------------------------------------------------------------

# Load system prompt from markdown file
_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"
SYSTEM_PROMPT = _PROMPT_PATH.read_text().strip()


# --- Agent wrapper ---------------------------------------------------------------


def _get_env_var(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is required but missing. Check your .env configuration.")
    return value


@lru_cache(maxsize=1)
def _get_vllm_client() -> OpenAI:
    """Return a cached OpenAI client configured to point at the vLLM server."""

    return OpenAI(api_key=_get_env_var("VLLM_API_KEY"), base_url=_get_env_var("VLLM_API_URL"))


@lru_cache(maxsize=1)
def _get_vllm_model_id() -> str:
    """Return the primary model identifier exposed by the vLLM server."""

    models = _get_vllm_client().models.list()
    if not models.data:
        raise RuntimeError("No models available on the vLLM server.")
    return models.data[0].id


def get_agent_response(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:  # noqa: WPS231
    """Call the underlying large-language model via the local vLLM server."""

    # The first message is assumed to be the system prompt if not explicitly provided
    # or if the history is empty. We'll ensure the system prompt is always first.
    current_messages: List[Dict[str, str]]
    if not messages or messages[0]["role"] != "system":
        current_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        current_messages = messages

    completion = _get_vllm_client().chat.completions.create(
        model=_get_vllm_model_id(),
        messages=current_messages,  # Pass the full history
    )

    assistant_reply_content = (completion.choices[0].message.content or "").strip()

    # Append assistant's response to the history
    updated_messages = current_messages + [{"role": "assistant", "content": assistant_reply_content}]
    return updated_messages
