"""Base class for all agent tools."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class BaseTool(ABC):
    """Abstract base for every tool the agent can call.

    Subclasses must declare their ``name``, ``description``, and OpenAI
    function-call ``openai_schema``, and implement ``__call__`` to run the
    tool given keyword arguments parsed from the model's JSON response.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool identifier used as the function name in OpenAI schemas."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Short human-readable description surfaced to the model."""

    @property
    @abstractmethod
    def openai_schema(self) -> dict[str, Any]:
        """Full OpenAI ``tools`` entry (``{"type": "function", "function": {...}}``)."""

    @abstractmethod
    def __call__(self, **kwargs: Any) -> str:
        """Execute the tool and return a plain-text result."""

    def run_from_json(self, raw_args: str) -> str:
        """Parse ``raw_args`` JSON string and delegate to ``__call__``."""
        try:
            kwargs = json.loads(raw_args or "{}")
        except json.JSONDecodeError:
            kwargs = {}
        return self(**kwargs)
