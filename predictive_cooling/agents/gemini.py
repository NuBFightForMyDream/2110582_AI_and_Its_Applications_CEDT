"""Gemini mode: the shared parent/child loops (llm_loops.py) on the Gemini API."""
import os

from .llm_loops import Call, Turn, run_loops

# Override with GEMINI_MODEL. flash-lite: fast, larger free quota (a case needs ~11 calls); the bigger
# "gemini-flash-latest" often answers 503 "high demand" and free keys get ~20 requests/day on it.
DEFAULT_MODEL = "gemini-3.5-flash-lite"


class GeminiProvider:
    def __init__(self):
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("GEMINI_NOT_CONFIGURED")

    def turn(self, role, instructions, history, tools, timeout):
        from google import genai
        from google.genai import types
        declarations = [types.FunctionDeclaration(name=name, description=description,
                                                  parameters_json_schema=schema.model_json_schema())
                        for name, (description, schema) in tools.items()]
        # Retry rate limits and transient server errors (Gemini often answers 503 "high demand" for a few seconds,
        # and free keys hit 429 per-minute quotas). Waits ≈ 2, 6, 18 s. The case deadline still bounds the whole run.
        retry = types.HttpRetryOptions(attempts=4, initial_delay=2.0, max_delay=20.0, exp_base=3.0, jitter=1.0,
                                       http_status_codes=[429, 500, 502, 503, 504])
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"],
                              http_options=types.HttpOptions(timeout=max(1000, int(timeout * 1000)),
                                                             retry_options=retry))
        try:
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", DEFAULT_MODEL), contents=history,
                config=types.GenerateContentConfig(
                    system_instruction=instructions, tools=[types.Tool(function_declarations=declarations)],
                    tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
                        mode="ANY", allowed_function_names=list(tools))),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    temperature=0, max_output_tokens=1400))
            content = response.candidates[0].content if response.candidates else None
            calls = [Call(p.function_call.name, dict(p.function_call.args or {}), p.function_call.id)
                     for p in (content.parts or []) if p.function_call] if content else []
            return Turn(calls, content, getattr(response.usage_metadata, "total_token_count", 0) or 0,
                        getattr(response, "model_version", None))
        finally:
            client.close()


class GeminiMessages:
    @staticmethod
    def first(text):
        return {"role": "user", "parts": [{"text": text}]}

    @staticmethod
    def assistant(turn, call):
        return turn.content or {"role": "model", "parts": [{"function_call": {"name": call.name, "args": call.args}}]}

    @staticmethod
    def reply(call, payload):
        try:
            from google.genai import types
            return types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
                name=call.name, id=call.id, response={"result": payload}))])
        except ImportError:            # scripted tests without the SDK installed
            return {"role": "user", "parts": [{"function_response": {"name": call.name, "response": {"result": payload}}}]}


def provider_factory():
    return GeminiProvider()


def run_agents(rt, provider=None):
    return run_loops(rt, provider or provider_factory(), GeminiMessages, "gemini_parent_child_tool_calling")
