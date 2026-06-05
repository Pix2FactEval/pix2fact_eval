"""SearchTool — web search via the ModelHub crawl API."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import requests

from .base import BaseTool

SEARCH_API_KEY_PLACEHOLDER = "<YOUR_MODELHUB_SEARCH_API_KEY>"


def _resolve_search_api_key(explicit: str | None) -> str | None:
    if explicit and explicit.strip() and explicit.strip() != SEARCH_API_KEY_PLACEHOLDER:
        return explicit.strip()
    v = os.getenv("MODELHUB_SEARCH_API_KEY")
    if v and v.strip() and v.strip() != SEARCH_API_KEY_PLACEHOLDER:
        return v.strip()
    return None


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

@dataclass
class SearchResultItem:
    """One row from ``results.web`` in the crawl search JSON response."""

    title: str
    description: str
    url: str
    snippets: list[str] = field(default_factory=list)


def search_response_to_items(data: dict[str, Any]) -> list[SearchResultItem]:
    """Extract ``SearchResultItem`` rows from a crawl search API JSON object."""
    results = data.get("results")
    if not isinstance(results, dict):
        return []
    web = results.get("web")
    if not isinstance(web, list):
        return []
    out: list[SearchResultItem] = []
    for row in web:
        if not isinstance(row, dict):
            continue
        title = row.get("title") or ""
        desc = row.get("description") or ""
        url = row.get("url") or ""
        snips: list[str] = []
        raw_snips = row.get("snippets")
        if isinstance(raw_snips, str):
            snips = [raw_snips] if raw_snips.strip() else []
        elif isinstance(raw_snips, list):
            snips = [str(s) for s in raw_snips if s is not None]
        elif raw_snips is not None:
            snips = [str(raw_snips)]
        out.append(
            SearchResultItem(
                title=str(title).strip(),
                description=str(desc).strip(),
                url=str(url).strip(),
                snippets=snips,
            )
        )
    return out


def search_results_to_markdown(
    items: list[SearchResultItem],
    *,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Render search hits as markdown (title, link, description, snippets)."""
    lines: list[str] = ["# Web search results", ""]
    if metadata:
        if (q := metadata.get("query")) is not None:
            lines.append(f"**Query:** {q}")
        lat = metadata.get("latency")
        if lat is not None:
            try:
                lines.append(f"**Latency:** {float(lat):.2f}s")
            except (TypeError, ValueError):
                lines.append(f"**Latency:** {lat}")
        if (su := metadata.get("search_uuid")) is not None:
            lines.append(f"**Search id:** {su}")
        if len(lines) > 2:
            lines.append("")

    if not items:
        lines.append("_No web results._")
        return "\n".join(lines)

    for i, it in enumerate(items, start=1):
        head = f"[{it.title}]({it.url})" if it.url else (it.title or f"Result {i}")
        lines.append(f"## {i}. {head}")
        if it.description:
            lines.append(f"**Description:** {it.description}")
        if it.snippets:
            lines.append("**Snippets:**")
            for s in it.snippets:
                lines.append(f"- {s.replace(chr(10), ' ').strip()}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Tool class
# ---------------------------------------------------------------------------

class SearchTool(BaseTool):
    """Web search via the ModelHub crawl API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        search_url: str | None = None,
        default_count: int = 10,
        retry_times: int = 5,
        retry_sleep_seconds: float = 10.0,
    ) -> None:
        self._api_key = _resolve_search_api_key(api_key)
        self._search_url = (
            search_url.strip()
            if isinstance(search_url, str) and search_url.strip()
            else (os.getenv("MODELHUB_SEARCH_URL") or "").strip()
        )
        self._default_count = default_count
        self._retry_times = max(1, int(retry_times))
        self._retry_sleep_seconds = max(0.0, float(retry_sleep_seconds))

    @property
    def name(self) -> str:
        return "search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current or external information. "
            "Optionally limit to specific domains (comma-separated hostnames) or change result count."
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
                        "query": {
                            "type": "string",
                            "description": "Search query string.",
                        },
                        "count": {
                            "type": "integer",
                            "description": "Max number of results to return (default 10).",
                        },
                        "include_domains": {
                            "type": "string",
                            "description": (
                                "Comma-separated domain names to restrict results to, e.g. "
                                "'example.com, news.example.org'."
                            ),
                        },
                    },
                    "required": ["query"],
                },
            },
        }

    def __call__(  # type: ignore[override]
        self,
        query: str,
        count: int | None = None,
        include_domains: str | None = None,
        return_raw_json: bool = False,
        **_: Any,
    ) -> str:
        q = (query or "").strip()
        if not q:
            return "No results (empty query)."
        if not self._api_key:
            return (
                "Search unavailable: set MODELHUB_SEARCH_API_KEY in the environment "
                f"(must not remain {SEARCH_API_KEY_PLACEHOLDER!r}), or pass api_key to SearchTool()."
            )
        if not self._search_url:
            return (
                "Search unavailable: set MODELHUB_SEARCH_URL (full HTTP endpoint for the crawl/search API)."
            )

        try:
            effective_count = int(count) if count is not None else self._default_count
        except (TypeError, ValueError):
            effective_count = self._default_count

        params: dict[str, Any] = {"query": q, "count": effective_count}
        if include_domains and str(include_domains).strip():
            params["include_domains"] = str(include_domains).strip()

        resp: requests.Response | None = None
        last_exception: requests.RequestException | None = None
        for attempt in range(1, self._retry_times + 1):
            try:
                resp = requests.get(
                    self._search_url,
                    headers={"X-API-Key": self._api_key},
                    params=params,
                    timeout=float(os.getenv("MODELHUB_CRAWL_TIMEOUT", "60")),
                )
                break
            except requests.RequestException as e:
                last_exception = e
                if attempt < self._retry_times:
                    time.sleep(self._retry_sleep_seconds)

        if resp is None:
            return f"Search request failed after {self._retry_times} attempts: {last_exception}"

        try:
            data = resp.json()
        except json.JSONDecodeError:
            return resp.text or f"Search returned HTTP {resp.status_code} with empty body."

        if resp.status_code >= 400:
            text = json.dumps(data, ensure_ascii=False) if isinstance(data, dict) else str(data)
            return f"Search error HTTP {resp.status_code}: {text}"

        if not isinstance(data, dict):
            return json.dumps(data, ensure_ascii=False) if data is not None else ""

        if return_raw_json:
            return json.dumps(data, ensure_ascii=False)

        items = search_response_to_items(data)
        meta = data.get("metadata")
        mdict = meta if isinstance(meta, dict) else None
        if items or mdict is not None or "results" in data:
            return search_results_to_markdown(items, metadata=mdict)
        return json.dumps(data, ensure_ascii=False)
