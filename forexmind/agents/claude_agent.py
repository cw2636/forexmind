"""
ForexMind — Claude AI Trading Agent (LangChain 1.x)
====================================================
The conversational brain of ForexMind.
Uses the new LangChain 1.x bind_tools + message-loop pattern with
Anthropic Claude as the LLM.

Conversation flow:
  User: "Should I buy EUR/USD right now?"
  Agent:
    1. Calls get_sessions() → checks if it's a good trading time
    2. Calls get_signal("EUR_USD", "M5") → runs full technical analysis
    3. Calls get_news("EUR_USD") → checks recent news sentiment
    4. Synthesises everything → returns structured JSON signal + explanation

The agent maintains a sliding window of the last MAX_HISTORY messages so
follow-up questions like "Why do you say buy?" work naturally.

Advanced Python Concepts Used:
  - LangChain 1.x bind_tools + manual tool call/response loop
  - Async/await with asyncio.to_thread for sync tool wrappers
  - Sliding-window message history (no external memory library needed)
  - AsyncIterator for streaming token output
  - Structured JSON output parsing with regex fallback
"""

from __future__ import annotations

import asyncio
import json
import re
from collections import deque
from typing import Any, AsyncIterator

from forexmind.agents.prompts import SYSTEM_PROMPT
from forexmind.agents.tools import build_tools
from forexmind.config.settings import get_settings
from forexmind.utils.logger import get_logger

log = get_logger(__name__)

try:
    from langchain_anthropic import ChatAnthropic
    from langchain_core.messages import (
        HumanMessage, AIMessage, ToolMessage, SystemMessage, BaseMessage,
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    log.warning("langchain-anthropic not installed. Run: pip install langchain-anthropic")

# Keep the most recent N conversation turns (human+AI pairs) in memory
MAX_HISTORY = 20


# ── Agent class ───────────────────────────────────────────────────────────────

class ForexMindAgent:
    """
    Conversational trading agent powered by Anthropic Claude.

    Uses LangChain 1.x's bind_tools pattern:
      1. LLM is "bound" to tools so it can request tool calls
      2. We run a manual agentic loop:
           • Send messages → Claude responds
           • If Claude wants tool calls → execute them, append results
           • Repeat until Claude gives a final text response
      3. Conversation history (sliding window) is managed as a deque

    Usage:
        agent = ForexMindAgent()
        response = await agent.chat("Should I buy EUR/USD?")
        signal = agent.extract_signal(response)
    """

    def __init__(self) -> None:
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "langchain-anthropic is required. "
                "Run: pip install langchain-anthropic langchain-core"
            )
        cfg = get_settings()
        if not cfg.claude.is_configured:
            raise ValueError(
                "ANTHROPIC_API_KEY not set in .env — Claude agent cannot start"
            )

        self._cfg = cfg
        self._tools = build_tools()
        # Map tool name → tool object for fast lookup during execution
        self._tool_map = {t.name: t for t in self._tools}

        # Base LLM
        self._llm = ChatAnthropic(
            model=cfg.claude.model,
            temperature=cfg.claude.temperature,
            max_tokens=cfg.claude.max_tokens,
            anthropic_api_key=cfg.claude.api_key,
        )

        # LLM bound to tools — Claude will now include tool_calls in responses
        # when it decides a tool is needed.
        self._llm_with_tools = self._llm.bind_tools(self._tools)

        # Sliding-window conversation history (excludes the system message)
        # Each element is a BaseMessage (HumanMessage | AIMessage | ToolMessage)
        self._history: deque[BaseMessage] = deque(maxlen=MAX_HISTORY * 2)

        log.info(
            f"ForexMind agent ready | model={cfg.claude.model} "
            f"| tools={[t.name for t in self._tools]}"
        )

    # ── Public interface ──────────────────────────────────────────────────────

    async def chat(self, user_message: str) -> str:
        """
        Send a message and receive a complete response.

        Internally runs the full agentic loop (including tool calls)
        and returns the final text response.
        """
        # Build message list: [system] + history + [new human message]
        messages = self._build_messages(user_message)

        try:
            final_response = await self._run_agent_loop(messages)
        except Exception as e:
            log.error(f"Agent loop error: {e}", exc_info=True)
            final_response = (
                f"I ran into a problem: {e}. "
                "Please check that your API keys are set correctly in .env"
            )

        # Update history with this turn
        self._history.append(HumanMessage(content=user_message))
        self._history.append(AIMessage(content=final_response))

        return final_response

    async def stream_chat(self, user_message: str) -> AsyncIterator[str]:
        """
        Stream the response token-by-token using LangChain event streaming.

        Yields string chunks as they arrive from Claude.
        Tool call intermediate steps are NOT streamed (they run silently).
        The final text answer IS streamed.
        """
        messages = self._build_messages(user_message)

        # First: run tool calls (may take a while for data fetching)
        try:
            messages = await self._resolve_tool_calls(messages, max_iterations=6)
        except Exception as e:
            log.error(f"Tool resolution error: {e}")

        full_response_parts: list[str] = []

        try:
            # Now stream the final answer
            async for chunk in self._llm.astream(messages):
                content = chunk.content
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            text = block.get("text", "")
                            full_response_parts.append(text)
                            yield text
                elif isinstance(content, str) and content:
                    full_response_parts.append(content)
                    yield content
        except Exception as e:
            yield f"\n[Streaming error: {e}]"
            return

        # Save conversation turn
        full_response = "".join(full_response_parts)
        self._history.append(HumanMessage(content=user_message))
        self._history.append(AIMessage(content=full_response))

    def extract_signal(self, response: str) -> dict[str, Any] | None:
        """
        Parse a JSON signal block from the agent's text response.

        Returns a dict with keys like: action, entry, stop_loss, take_profit.
        Returns None if no parseable JSON is found.
        """
        # Pattern 1: ```json { ... } ```
        match = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Pattern 2: any JSON object containing "action" key
        match2 = re.search(r"\{[^{}]*\"action\"[^{}]*\}", response, re.DOTALL)
        if match2:
            try:
                return json.loads(match2.group())
            except json.JSONDecodeError:
                pass

        return None

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self._history.clear()
        log.info("Agent memory cleared")

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_messages(self, user_message: str) -> list[BaseMessage]:
        """Build the full message list for one LLM call."""
        return [
            SystemMessage(content=SYSTEM_PROMPT),
            *list(self._history),
            HumanMessage(content=user_message),
        ]

    async def _run_agent_loop(
        self, messages: list[BaseMessage], max_iterations: int = 6
    ) -> str:
        """
        Core agentic loop.

        1. Invoke LLM with current messages
        2. If LLM requested tool calls → execute them, append results
        3. Re-invoke LLM with updated messages (including tool results)
        4. Repeat until no more tool calls (or max_iterations reached)
        5. Return final text response
        """
        for _ in range(max_iterations):
            response = await self._llm_with_tools.ainvoke(messages)
            messages.append(response)

            # If no tool calls → we have the final answer
            if not getattr(response, "tool_calls", None):
                return _extract_text_content(response.content)

            # Execute all requested tool calls in parallel
            tool_results = await asyncio.gather(*[
                self._execute_tool(tc["name"], tc["args"])
                for tc in response.tool_calls
            ])
            for tc, tool_result in zip(response.tool_calls, tool_results):
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tc["id"],
                    )
                )

        # Fallback: force a final answer if max iterations reached
        final = await self._llm.ainvoke(messages)
        return _extract_text_content(final.content)

    async def _resolve_tool_calls(
        self, messages: list[BaseMessage], max_iterations: int = 6
    ) -> list[BaseMessage]:
        """
        Same as _run_agent_loop but only resolves tool calls;
        does NOT request a final text answer.
        Used by stream_chat so we can stream the last response.
        """
        for _ in range(max_iterations):
            response = await self._llm_with_tools.ainvoke(messages)
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                # Remove the AI response — caller will re-invoke for streaming
                messages.pop()
                break

            tool_results = await asyncio.gather(*[
                self._execute_tool(tc["name"], tc["args"])
                for tc in response.tool_calls
            ])
            for tc, tool_result in zip(response.tool_calls, tool_results):
                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tc["id"],
                    )
                )

        return messages

    async def _execute_tool(self, tool_name: str, tool_args: dict[str, Any]) -> str:
        """Execute a single tool by name, returning its string result."""
        tool = self._tool_map.get(tool_name)
        if tool is None:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            result = await tool.ainvoke(tool_args)
            return str(result)
        except Exception as e:
            log.error(f"Tool {tool_name} error: {e}")
            return json.dumps({"error": str(e), "tool": tool_name})


# ── Helper ────────────────────────────────────────────────────────────────────

def _extract_text_content(content: Any) -> str:
    """
    Claude sometimes returns content as a list of blocks
    (e.g. [{type: 'text', text: '...'}]).
    This helper normalises it to a plain string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


# ── Singleton ─────────────────────────────────────────────────────────────────

_agent: ForexMindAgent | None = None


def get_agent() -> ForexMindAgent:
    """Return the singleton agent instance (creates it on first call)."""
    global _agent
    if _agent is None:
        _agent = ForexMindAgent()
    return _agent
