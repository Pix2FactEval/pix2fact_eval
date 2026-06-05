"""Agent loop and tool registry for Azure OpenAI chat with tool calls."""

from __future__ import annotations

import os
from typing import Any

import dotenv
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionMessageParam

from .base import BaseTool
from .search_tool import SearchTool
from .terminate_tool import TerminateTool
from .visit_tool import VisitTool

dotenv.load_dotenv()

DEFAULT_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-03-01-preview")


def build_tool_registry(*tools: BaseTool) -> dict[str, BaseTool]:
    """Return a ``{name: tool}`` mapping for O(1) dispatch in the agent loop."""
    return {t.name: t for t in tools}


def get_azure_client(
    *,
    api_key: str | None = None,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
) -> AzureOpenAI:
    key = api_key or os.environ.get("AZURE_OPENAI_API_KEY")
    endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
    if not key or not endpoint:
        raise ValueError(
            "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set "
            "(or pass api_key / azure_endpoint)."
        )
    return AzureOpenAI(
        api_key=key,
        azure_endpoint=endpoint.rstrip("/"),
        api_version=api_version or DEFAULT_API_VERSION,
    )


def _assistant_message_dict(msg: Any) -> dict[str, Any]:
    data = msg.model_dump(exclude_none=True)
    # OpenAI requires explicit null content when only tool_calls are present.
    if msg.tool_calls and data.get("content") is None:
        data["content"] = None
    return data


def chat_completion_with_tools(
    user_message: str,
    *,
    tools: list[BaseTool] | None = None,
    client: AzureOpenAI | None = None,
    deployment: str | None = None,
    system_prompt: str | None = "Answer the user's question based on search result.",
    max_steps: int = 20,
) -> str:
    """Run a tool-augmented chat completion loop on Azure OpenAI.

    The model may call any of the supplied ``tools`` zero or more times.
    A ``TerminateTool`` is always appended so the model has a structured way to
    signal completion.  When ``terminate`` is called the loop breaks and the
    result is returned as a JSON string matching the PROMPT_TEMPLATE_TOOL_CALL
    schema.  Pass ``tools=None`` to use the default set
    (``SearchTool`` + ``VisitTool``).
    """
    if tools is None:
        tools = [SearchTool(), VisitTool()]

    terminate = TerminateTool()
    all_tools: list[BaseTool] = [*tools, terminate]

    registry = build_tool_registry(*all_tools)
    openai_tools = [t.openai_schema for t in all_tools]

    c = client or get_azure_client()
    model = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    if not model:
        raise ValueError("Pass deployment=... or set AZURE_OPENAI_DEPLOYMENT.")

    messages: list[ChatCompletionMessageParam] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    steps = 0
    while steps < max_steps:
        print("Call model", model)
        response = c.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
        )
        print("Get response", response)
        choice = response.choices[0]
        msg = choice.message

        if not msg.tool_calls:
            return (msg.content or "").strip()

        messages.append(_assistant_message_dict(msg))
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
                print(f"Tool call: {name}({raw_args})")
                tool_output = tool.run_from_json(raw_args)
                print(f"Tool result: {tool_output[:200]}...")

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
            return terminate.to_json()

    # Exhausted max_steps without termination.
    return terminate.to_json()


def chat_completion_with_search_answer(
    user_message: str,
    *,
    client: AzureOpenAI | None = None,
    deployment: str | None = None,
    system_prompt: str | None = "Answer the user's question based on search result.",
) -> str:
    """Backwards-compatible alias that only exposes the search tool."""
    return chat_completion_with_tools(
        user_message,
        tools=[SearchTool()],
        client=client,
        deployment=deployment,
        system_prompt=system_prompt,
    )
