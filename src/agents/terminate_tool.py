"""TerminateTool — signals the agent to stop and return a structured answer."""

from __future__ import annotations

import json
from typing import Any, Literal

from .base import BaseTool

DEFAULT_FAIL_ANSWER = "[NO_DEFINITIVE_ANSWER]"


class TerminateTool(BaseTool):
    """Special tool the model calls to end the agent loop.

    The model provides a structured response matching the PROMPT_TEMPLATE_TOOL_CALL
    schema: observation, search_plan, search_query, comprehensive_answer, final_answer.

    - ``status="success"`` — the model provides a complete structured answer.
    - ``status="fail"``    — the model could not answer; ``final_answer`` falls back
                             to ``default_fail_answer``.
    """

    def __init__(self, *, default_fail_answer: str = DEFAULT_FAIL_ANSWER) -> None:
        self.default_fail_answer = default_fail_answer
        # Populated by the agent after the tool is invoked.
        self.status: Literal["success", "fail"] | None = None
        self.answer: str | None = None
        self.result: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "terminate"

    @property
    def description(self) -> str:
        return (
            "Terminate the task and return the final structured answer. "
            "Call with status='success' and all answer fields when the task is done, "
            "or status='fail' when you cannot answer."
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
                        "status": {
                            "type": "string",
                            "enum": ["success", "fail"],
                            "description": (
                                "'success' if you have a complete answer, "
                                "'fail' if you cannot answer the question."
                            ),
                        },
                        "observation": {
                            "type": "string",
                            "description": (
                                "Describe specific visual details from the image relevant to the question."
                            ),
                        },
                        "search_plan": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Step-by-step plan to find the necessary information online."
                            ),
                        },
                        "search_query": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Exact search queries extracted from the search plan."
                            ),
                        },
                        "comprehensive_answer": {
                            "type": "string",
                            "description": (
                                "Comprehensive final answer integrating observations and search results."
                            ),
                        },
                        "final_answer": {
                            "type": "string",
                            "description": (
                                "Core, direct answer to the question. "
                                "Use '[NO_DEFINITIVE_ANSWER]' if a definitive answer cannot be determined. "
                                "Required when status='success'."
                            ),
                        },
                    },
                    "required": ["status", "final_answer"],
                },
            },
        }

    def __call__(  # type: ignore[override]
        self,
        status: str,
        observation: str | None = None,
        search_plan: list[str] | None = None,
        search_query: list[str] | None = None,
        comprehensive_answer: str | None = None,
        final_answer: str | None = None,
        **_: Any,
    ) -> str:
        self.status = status  # type: ignore[assignment]
        fa = (final_answer or "").strip() or self.default_fail_answer
        if status != "success":
            fa = self.default_fail_answer

        self.answer = fa
        self.result = {
            "status": status,
            "observation": observation or "",
            "search_plan": search_plan or [],
            "search_query": search_query or [],
            "comprehensive_answer": comprehensive_answer or "",
            "final_answer": fa,
        }
        return f"terminate:{status}"

    def to_json(self) -> str:
        """Return the stored result as a JSON string."""
        if self.result is None:
            return json.dumps({"status": "fail", "final_answer": self.default_fail_answer})
        return json.dumps(self.result, ensure_ascii=False)
