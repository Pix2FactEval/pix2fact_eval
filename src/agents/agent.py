"""Simple tool-calling agent backed by an OpenAI-compatible client."""

from __future__ import annotations

import base64
import io
import math
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

from .base import BaseTool
from .search_tool import SearchTool
from .terminate_tool import TerminateTool
from .visit_tool import VisitTool


@dataclass
class AgentResult:
    """Outcome of a single ``Agent.run()`` call."""

    status: str          # "success" | "fail"
    answer: str
    steps: int           # number of tool-call rounds executed
    messages: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    """Tool-calling agent that loops until ``terminate`` is invoked.

    The agent always injects a ``TerminateTool`` into every run so the model
    has a structured way to signal completion.  Any other tools are passed in
    via the constructor.

    Usage::

        agent = Agent(
            client=get_azure_client(),
            deployment="gpt-4o",
            tools=[SearchTool(), VisitTool()],
            system_prompt="You are a helpful research assistant.",
            max_steps=10,
        )
        result = agent.run("Who won the 2024 Nobel Prize in Physics?")
        print(result.answer)
    """

    def __init__(
        self,
        *,
        client: OpenAI | None = None,
        deployment: str | None = None,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        max_steps: int = 20,
        default_fail_answer: str = TerminateTool().default_fail_answer,
        verbose: bool = True,
        enable_thinking: bool = True,
        token_budget: int = 32_768,
        image_token_budget: int = 1_024,
    ) -> None:
        self._client = client
        self._deployment = deployment
        self._tools: list[BaseTool] = tools
        self._system_prompt = system_prompt
        self._max_steps = max_steps
        self._default_fail_answer = default_fail_answer
        self._verbose = verbose
        self._enable_thinking = enable_thinking
        self._token_budget = token_budget
        self._image_token_budget = image_token_budget

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        instruction: str,
        image: str | Path | None = None,
        image_detail: str = "auto",
    ) -> AgentResult:
        """Run the agent and return an ``AgentResult``.

        Args:
            instruction: The user's text question or instruction.
            image:        Optional image to include alongside the instruction.
                          Accepts a URL (``http/https``), a local file path, or a
                          ``data:image/...;base64,...`` string.
            image_detail: OpenAI vision detail level — ``"auto"``, ``"low"``,
                          or ``"high"`` (ignored when no image is supplied).
        """
        terminate = TerminateTool(default_fail_answer=self._default_fail_answer)
        all_tools: list[BaseTool] = [*self._tools, terminate]
        registry = {t.name: t for t in all_tools}
        openai_tools = [t.openai_schema for t in all_tools]

        client = self._client or self._default_client()
        model = self._deployment or "Qwen/Qwen3-6B-27B"

        # Resolve and optionally resize the image before embedding it.
        resolved_image_url: str | None = None
        if image is not None:
            resolved_image_url = self._prepare_image_url(image)

        messages: list[ChatCompletionMessageParam] = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": self._build_user_content(instruction, resolved_image_url, image_detail)})

        extra_body: dict[str, Any] = {}
        if not self._enable_thinking:
            extra_body = {
                "chat_template_kwargs": {
                    "enable_thinking": self._enable_thinking,
                }
            }

        budget_threshold = int(self._token_budget * 0.8)
        total_tokens_used = 0
        budget_exceeded = False

        steps = 0
        while steps < self._max_steps:
            # Switch to terminate-only action space once budget is exhausted.
            active_tools = [terminate] if budget_exceeded else all_tools
            active_openai_tools = [t.openai_schema for t in active_tools]

            self._log(f"[step {steps}] calling model {model!r} (tokens used so far: {total_tokens_used}/{self._token_budget})")
            print("Tools", active_openai_tools)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=active_openai_tools,
                tool_choice="required",  # force the model to always use a tool
                extra_body=extra_body or None,
            )
            choice = response.choices[0]
            msg = choice.message

            # Accumulate token usage and check against budget.
            usage = response.usage
            total_tokens_used = 0
            if usage is not None:
                step_tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
                total_tokens_used += step_tokens
                self._log(
                    f"[step {steps}] tokens this step: {step_tokens} | "
                    f"total tokens used: {total_tokens_used}/{self._token_budget} "
                    f"({total_tokens_used / self._token_budget * 100:.1f}%)"
                )
                if total_tokens_used >= budget_threshold and not budget_exceeded:
                    self._log(
                        f"[budget] {total_tokens_used} tokens >= 80% of budget "
                        f"({budget_threshold}). Restricting action space to terminate only."
                    )
                    budget_exceeded = True

            self._log(f"[step {steps}] response: {response.model_dump_json(indent=2)}")
            if not msg.tool_calls:
                # Unexpected plain reply — treat as implicit success.
                return AgentResult(
                    status="success",
                    answer=(msg.content or "").strip(),
                    steps=steps,
                    messages=list(messages),
                )

            messages.append(self._to_message_dict(msg))
            steps += 1

            terminated = False
            for tc in msg.tool_calls:
                if tc.type != "function":
                    continue
                name = tc.function.name
                raw_args = tc.function.arguments or "{}"
                tool = registry.get(name)

                if tool is None:
                    tool_output = f"Unknown tool: {name}"
                else:
                    self._log(f"  -> {name}({raw_args[:120]})")
                    tool_output = tool.run_from_json(raw_args)
                    self._log(f"     {tool_output[:120]}")

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_output,
                    }
                )

                if name == terminate.name:
                    terminated = True

            if terminated:
                break

        return AgentResult(
            status=terminate.status or "fail",
            answer=terminate.answer or self._default_fail_answer,
            steps=steps,
            messages=list(messages),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _build_user_content(
        instruction: str,
        image_url: str | None,
        detail: str,
    ) -> str | list[dict[str, Any]]:
        """Return a plain string when there is no image, or a multimodal content list."""
        if image_url is None:
            return instruction
        return [
            {"type": "text", "text": instruction},
            {"type": "image_url", "image_url": {"url": image_url, "detail": detail}},
        ]

    def _prepare_image_url(self, image: str | Path) -> str:
        """Load *image*, enforce ``image_token_budget``, and return a data URL.

        Token count is estimated as ``floor(w * h / (32 * 32))``.  When the
        image exceeds the budget the shorter dimension is scaled down while
        preserving the aspect ratio until the token count is strictly below the
        budget.

        Accepted inputs:
        - ``http/https`` URL  → downloaded, then processed
        - ``data:image/...``  → decoded, then processed
        - local file path     → read, then processed
        """
        from PIL import Image as PILImage  # lazy import — only needed when image is provided

        raw = str(image).strip()

        # ---- load raw bytes + infer MIME ----
        if raw.startswith(("http://", "https://")):
            with urllib.request.urlopen(raw, timeout=30) as resp:
                img_bytes = resp.read()
                ct = resp.headers.get_content_type() or "image/jpeg"
            mime = ct.split(";")[0].strip()
        elif raw.startswith("data:"):
            header, b64data = raw.split(",", 1)
            mime = header.split(";")[0].split(":")[1]
            img_bytes = base64.b64decode(b64data)
        else:
            path = Path(raw)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            img_bytes = path.read_bytes()
            suffix = path.suffix.lower().lstrip(".")
            mime = {
                "jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "gif": "image/gif", "webp": "image/webp",
            }.get(suffix, f"image/{suffix}")

        img = PILImage.open(io.BytesIO(img_bytes))
        w, h = img.size
        tokens = w * h // (32 * 32)
        self._log(f"[image] original size={w}x{h}, image_tokens={tokens} (budget={self._image_token_budget})")

        if tokens > self._image_token_budget:
            # Derive scale so that new_w * new_h == image_token_budget * 1024 - epsilon.
            scale = math.sqrt(self._image_token_budget * 32 * 32 / (w * h))
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            # Nudge down until strictly under budget (handles rounding edge cases).
            while new_w * new_h // (32 * 32) >= self._image_token_budget:
                new_w = max(1, new_w - 1)
                new_h = max(1, new_h - 1)
            self._log(
                f"[image] resizing to {new_w}x{new_h} "
                f"(image_tokens={new_w * new_h // (32 * 32)})"
            )
            img = img.resize((new_w, new_h), PILImage.LANCZOS)
            fmt = "PNG" if img.mode in ("RGBA", "LA", "P") else "JPEG"
            mime = "image/png" if fmt == "PNG" else "image/jpeg"
            buf = io.BytesIO()
            img.save(buf, format=fmt)
            img_bytes = buf.getvalue()

        encoded = base64.b64encode(img_bytes).decode("ascii")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def _default_client() -> OpenAI:
        # Local OpenAI-compat servers often ignore auth; OPENAI_COMPAT_API_KEY defaults accordingly.
        return OpenAI(
            base_url=os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:34573/v1"),
            api_key=os.getenv("OPENAI_COMPAT_API_KEY", "EMPTY"),
        )

    @staticmethod
    def _to_message_dict(msg: Any) -> dict[str, Any]:
        data = msg.model_dump(exclude_none=True)
        # OpenAI requires explicit null content when only tool_calls are present.
        if msg.tool_calls and data.get("content") is None:
            data["content"] = None
        return data

    def _log(self, text: str) -> None:
        if self._verbose:
            print(text)
