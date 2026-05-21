"""
LLM client — Gemma 3 / Gemini Flash via Google AI Studio.
Same API endpoint, just different model strings. Swap via env var LLM_MODEL.

Implements an agentic loop: model decides which tools to call, we execute,
feed results back, repeat until model produces a final text answer.
"""
from __future__ import annotations

import json
from typing import Any, Awaitable, Callable

import google.generativeai as genai

from backend.config import settings


ToolHandler = Callable[[str, dict], Awaitable[dict]]


class LLMClient:
    def __init__(self):
        if not settings.GOOGLE_API_KEY:
            print("⚠ GOOGLE_API_KEY not set — LLM calls will fail. Set it in .env")
        else:
            genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model_name = settings.LLM_MODEL

    async def chat_with_tools(
        self,
        message: str,
        history: list[dict],
        tools: list[dict],
        tool_handler: ToolHandler,
        system: str | None = None,
        max_iters: int = 5,
    ) -> dict:
        """
        Agentic loop: ask Gemma → if it wants tools, run them → feed back → repeat.

        Returns a dict with the final reply and the list of tools called along the way
        (so the frontend can show the retrieval trace under each answer).
        """
        # Convert our tool schema to Gemini's expected format
        gemini_tools = [{"function_declarations": tools}]
        model = genai.GenerativeModel(
            model_name=self.model_name,
            tools=gemini_tools,
            system_instruction=system,
        )

        chat = model.start_chat(history=self._to_gemini_history(history))
        tools_used: list[dict] = []
        reply_text = ""

        try:
            response = await self._send(chat, message)
            for _ in range(max_iters):
                fn_calls = self._extract_function_calls(response)
                if not fn_calls:
                    reply_text = self._extract_text(response)
                    break

                # Run each requested tool and feed results back
                tool_responses = []
                for call in fn_calls:
                    name = call["name"]
                    args = call["args"]
                    print(f"  → tool {name}({args})")
                    result = await tool_handler(name, args)
                    tools_used.append({"name": name, "args": args, "result_preview": _preview(result)})
                    tool_responses.append({"function_response": {"name": name, "response": result}})

                response = await self._send(chat, tool_responses)
        except Exception as e:
            return {"reply": f"(error: {e})", "tools_used": tools_used, "error": str(e)}

        return {"reply": reply_text, "tools_used": tools_used}

    async def _send(self, chat, content):
        """Wrap send_message — sync SDK called from async context."""
        import asyncio
        return await asyncio.to_thread(chat.send_message, content)

    @staticmethod
    def _to_gemini_history(history: list[dict]) -> list[dict]:
        out = []
        for h in history:
            role = "user" if h.get("role") == "user" else "model"
            out.append({"role": role, "parts": [{"text": h.get("content", "")}]})
        return out

    @staticmethod
    def _extract_function_calls(response) -> list[dict]:
        calls = []
        for part in (response.candidates[0].content.parts if response.candidates else []):
            fc = getattr(part, "function_call", None)
            if fc and fc.name:
                calls.append({"name": fc.name, "args": dict(fc.args) if fc.args else {}})
        return calls

    @staticmethod
    def _extract_text(response) -> str:
        try:
            return response.text
        except Exception:
            parts = response.candidates[0].content.parts if response.candidates else []
            return "".join(getattr(p, "text", "") for p in parts).strip()


def _preview(obj: Any) -> str:
    s = json.dumps(obj, default=str)
    return s if len(s) < 200 else s[:200] + "…"
