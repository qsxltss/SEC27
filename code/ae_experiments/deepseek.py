from __future__ import annotations

import os
from typing import Any


MODEL_NAME = "deepseek/deepseek-v3.2"
SYSTEM_PROMPT = "You are a helpful assistant."
TEMPERATURE = 0.0
TOP_P = 1.0
SEED = 42
MAX_TOKENS = 4096


class LiveTargetUnavailable(RuntimeError):
    pass


class DeepSeekTargetClient:
    """Lazy OpenAI-compatible client for RQ1 cache-through misses."""

    def __init__(self) -> None:
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise LiveTargetUnavailable(
                "The target-response cache does not match the current retrieval, "
                "and DEEPSEEK_API_KEY is not set. Export DEEPSEEK_API_KEY "
                "and DEEPSEEK_API_BASE to query DeepSeek and populate the "
                "runtime cache."
            )

        try:
            from openai import OpenAI
        except ImportError as error:
            raise LiveTargetUnavailable(
                "Live DeepSeek fallback requires the 'openai' package from "
                "code/requirements.txt."
            ) from error

        api_base = os.getenv("DEEPSEEK_API_BASE", "").strip()
        if not api_base:
            raise LiveTargetUnavailable(
                "The target-response cache does not match the current retrieval, "
                "and DEEPSEEK_API_BASE is not set. Export the base URL of the "
                "OpenAI-compatible endpoint that serves deepseek/deepseek-v3.2."
            )
        timeout = float(os.getenv("DEEPSEEK_API_TIMEOUT", "300"))
        arguments: dict[str, Any] = {
            "api_key": api_key,
            "base_url": api_base,
            "timeout": timeout,
            "max_retries": 2,
        }
        self.client = OpenAI(**arguments)

    def query(self, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        for attempt in range(2):
            completion = self.client.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                seed=SEED,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            response = str(completion.choices[0].message.content or "").strip()
            if response:
                return response
            if attempt == 0:
                continue
        raise LiveTargetUnavailable("DeepSeek returned an empty response twice.")
