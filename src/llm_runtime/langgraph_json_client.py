"""LangGraph-backed JSON decision client for policy routing."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, TypedDict


def load_env_file(env_path: str | Path = ".env") -> None:
    path = Path(env_path)
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        k = key.strip()
        v = value.strip().strip("\"'")
        if k and k not in os.environ:
            os.environ[k] = v


class _GraphState(TypedDict, total=False):
    system_prompt: str
    user_payload_json: str
    raw_content: str


@dataclass
class LangGraphResult:
    output: Dict[str, Any]
    raw_content: str


@dataclass
class LangGraphJSONClient:
    model: str
    provider: str = "openai"
    env_path: str = ".env"
    timeout_s: int = 90
    temperature: float = 0.1
    max_tokens: int | None = None

    def __post_init__(self) -> None:
        load_env_file(self.env_path)
        self.provider = str(self.provider).strip().lower() or "openai"
        self._ready = False
        self._init_error: str = ""
        self._graph = None
        self._build_graph()

    @property
    def ready(self) -> bool:
        return bool(self._ready)

    @property
    def init_error(self) -> str:
        return self._init_error

    def _build_graph(self) -> None:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            from langgraph.graph import END, StateGraph
        except Exception as exc:
            self._ready = False
            self._init_error = f"langgraph_import_error:{exc.__class__.__name__}"
            return

        try:
            llm = self._build_chat_model()
        except Exception as exc:
            self._ready = False
            self._init_error = f"chat_model_init_error:{exc.__class__.__name__}"
            return

        def _ask_model(state: _GraphState) -> _GraphState:
            messages = [
                SystemMessage(content=state.get("system_prompt", "")),
                HumanMessage(content=state.get("user_payload_json", "{}")),
            ]
            response = llm.invoke(messages)
            content = _message_text(response)
            return {"raw_content": content}

        builder: StateGraph = StateGraph(_GraphState)
        builder.add_node("ask_model", _ask_model)
        builder.set_entry_point("ask_model")
        builder.add_edge("ask_model", END)
        self._graph = builder.compile()
        self._ready = True
        self._init_error = ""

    def _build_chat_model(self) -> Any:
        kwargs: Dict[str, Any] = {
            "temperature": float(self.temperature),
            "timeout": int(self.timeout_s),
        }
        if self.max_tokens is not None:
            kwargs["max_tokens"] = int(self.max_tokens)

        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(model=self.model, **kwargs)

        # Generic path for alternate providers supported by LangChain.
        try:
            from langchain.chat_models import init_chat_model
        except Exception as exc:
            raise RuntimeError("langchain_init_chat_model_unavailable") from exc
        return init_chat_model(model=self.model, model_provider=self.provider, **kwargs)

    def invoke_json(self, system_prompt: str, user_payload: Dict[str, Any]) -> LangGraphResult:
        if not self.ready or self._graph is None:
            raise RuntimeError(f"langgraph_client_not_ready:{self._init_error or 'unknown'}")

        input_state: _GraphState = {
            "system_prompt": str(system_prompt),
            "user_payload_json": json.dumps(user_payload, separators=(",", ":"), ensure_ascii=True),
        }
        try:
            out = self._graph.invoke(input_state)
        except Exception as exc:
            raise RuntimeError(f"langgraph_invoke_error:{exc.__class__.__name__}") from exc
        if not isinstance(out, dict):
            raise RuntimeError("langgraph_invalid_state_output")

        raw_content = str(out.get("raw_content", "")).strip()
        if not raw_content:
            raise RuntimeError("langgraph_empty_model_response")
        try:
            parsed = _parse_json_object(raw_content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("langgraph_invalid_json_response") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("langgraph_non_object_json_response")

        return LangGraphResult(output=parsed, raw_content=raw_content)


def _parse_json_object(raw_content: str) -> Dict[str, Any]:
    text = str(raw_content).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    fenced = _strip_json_code_fence(text)
    if fenced is not None:
        parsed = json.loads(fenced)
        if isinstance(parsed, dict):
            return parsed

    extracted = _extract_first_json_object(text)
    if extracted is not None:
        parsed = json.loads(extracted)
        if isinstance(parsed, dict):
            return parsed

    raise json.JSONDecodeError("Unable to parse JSON object", text, 0)


def _strip_json_code_fence(text: str) -> str | None:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return None
    lines = stripped.splitlines()
    if len(lines) < 3:
        return None
    if not lines[0].startswith("```") or lines[-1].strip() != "```":
        return None
    body = "\n".join(lines[1:-1]).strip()
    if body.lower().startswith("json"):
        body = body[4:].lstrip()
    return body or None


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _message_text(msg: Any) -> str:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(content)
