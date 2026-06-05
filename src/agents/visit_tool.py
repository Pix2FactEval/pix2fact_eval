"""VisitTool — fetch page content via the Jina Reader API."""

from __future__ import annotations

import os
from typing import Any

import requests

from .base import BaseTool

JINA_READER_API_KEY_PLACEHOLDER = "<YOUR_JINA_READER_API_KEY>"
_DEFAULT_BASE_URL = "https://r.jina.ai"


def _resolve_jina_reader_api_key(explicit: str | None) -> str | None:
    if explicit and explicit.strip() and explicit.strip() != JINA_READER_API_KEY_PLACEHOLDER:
        return explicit.strip()
    v = os.getenv("JINA_READER_API_KEY") or os.getenv("JINA_API_KEY")
    if v and v.strip() and v.strip() != JINA_READER_API_KEY_PLACEHOLDER:
        return v.strip()
    return None


class VisitTool(BaseTool):
    """Fetch the main readable content of a web page via Jina Reader.

    Equivalent to:
    ``curl "https://r.jina.ai/https://www.example.com" -H "Authorization: Bearer ..."``
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = _DEFAULT_BASE_URL,
        default_timeout: float | None = None,
    ) -> None:
        self._api_key = _resolve_jina_reader_api_key(api_key)
        self._base_url = base_url.rstrip("/")
        self._default_timeout = default_timeout

    @property
    def name(self) -> str:
        return "visit_url"

    @property
    def description(self) -> str:
        return (
            "Fetch the main readable content of a web page as plain text. "
            "Use this to read an article, documentation page, or any URL in full."
        )

    @property
    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The full URL of the page to visit (http/https).",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Request timeout in seconds (default 60).",
                        },
                    },
                    "required": ["url"],
                },
            },
        }

    def __call__(  # type: ignore[override]
        self,
        url: str,
        timeout: float | None = None,
        **_: Any,
    ) -> str:
        raw = (url or "").strip()
        if not raw:
            return "visit_url: empty URL."
        if not self._api_key:
            return (
                "visit_url: set JINA_READER_API_KEY (or JINA_API_KEY) in the environment "
                f"(must not remain {JINA_READER_API_KEY_PLACEHOLDER!r}), "
                "or pass api_key to VisitTool()."
            )

        if not raw.startswith(("http://", "https://")):
            raw = f"https://{raw.lstrip('/')}"

        reader_url = f"{self._base_url}/{raw}"
        t = (
            timeout
            if timeout is not None
            else (
                self._default_timeout
                if self._default_timeout is not None
                else float(os.getenv("JINA_READER_TIMEOUT", "60"))
            )
        )

        try:
            resp = requests.get(
                reader_url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=t,
            )
        except requests.RequestException as e:
            return f"visit_url request failed: {e}"

        if resp.status_code >= 400:
            return f"visit_url HTTP {resp.status_code}: {(resp.text or '').strip()[:2000]}"

        return (resp.text or "").strip()
