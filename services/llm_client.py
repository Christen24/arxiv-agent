"""
Groq LLM adapter — thin wrapper around the Groq Python SDK.

Designed to be swappable: everything talks to `complete()` and
`complete_json()`, not to Groq internals directly.
"""

from __future__ import annotations

import json
import os

from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential


_client: Groq | None = None

DEFAULT_MODEL = "openai/gpt-oss-20b"


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Copy .env.example → .env and add your key."
            )
        _client = Groq(api_key=api_key)
    return _client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def complete(
    prompt: str,
    system: str = "You are a helpful research assistant.",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.3,
    max_tokens: int = 2048,
) -> str:
    """Send a chat completion and return the text content."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def complete_json(
    prompt: str,
    system: str = "You are a helpful research assistant. Respond ONLY with valid JSON.",
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 3000,
) -> dict:
    """Send a chat completion expecting a JSON response. Parses and returns the dict."""
    client = _get_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)
