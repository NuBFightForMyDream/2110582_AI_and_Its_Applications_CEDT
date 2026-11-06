"""Claude mode: the shared parent/child loops (llm_loops.py) on Claude Haiku 5.5."""
import json
import os

import numpy as np

from .llm_loops import Call, Turn, run_loops

MODEL = "claude-haiku-5-5"


class ClaudeProvider:
    def __init__(self, client=None):
        if client is None:
            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError("CLAUDE_NOT_CONFIGURED")
            import anthropic
            client = anthropic.Anthropic(max_retries=3)    # retries 429/5xx; the case deadline bounds the whole run
        self.client = client

    def turn(self, role, instructions, history, tools, timeout):
        from .engine import GuardError
        # Forcing the call gives the loop its one tool per turn and skips thinking tokens. The answers then carry no
        # thinking blocks, so the tool list may grow with each phase. Pydantic validates the arguments in the loop.
        response = self.client.with_options(timeout=timeout).messages.create(
            model=MODEL, max_tokens=2048, system=instructions, messages=history,
            tools=[{"name": name, "description": description, "input_schema": schema.model_json_schema()}
                   for name, (description, schema) in tools.items()],
            tool_choice={"type": "any", "disable_parallel_tool_use": True},
            output_config={"effort": "low"})
        if response.stop_reason == "refusal":
            raise GuardError("MODEL_REFUSED")
        if response.stop_reason == "max_tokens":
            raise GuardError("MODEL_OUTPUT_TRUNCATED")
        calls = [Call(block.name, dict(block.input or {}), block.id) for block in response.content if block.type == "tool_use"]
        return Turn(calls, response.content, response.usage.input_tokens + response.usage.output_tokens, response.model)


class ClaudeMessages:
    @staticmethod
    def first(text):
        return {"role": "user", "content": text}

    @staticmethod
    def assistant(turn, call):
        return {"role": "assistant", "content": turn.content}

    @staticmethod
    def reply(call, payload):
        text = json.dumps(payload, default=lambda o: o.item() if isinstance(o, np.generic) else str(o))
        return {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call.id, "content": text}]}


def provider_factory():
    return ClaudeProvider()


def run_agents(rt, provider=None):
    return run_loops(rt, provider or provider_factory(), ClaudeMessages, "claude_parent_child_tool_calling")
