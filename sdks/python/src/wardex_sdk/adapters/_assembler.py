"""Session assembler: correlates stream events and hook events into spans.

Sources: the transport tee (raw JSON lines -> native parser) and SDK hooks.
Rule (spec §6.2): hooks are the authority for lifecycle/attribution, the
stream is the authority for content; joined on tool_use_id. All timestamps
are host-arrival times (IPC level).
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any

from .. import _hub
from .._enums import (
    AgentType,
    CaptureSource,
    OperationName,
    ProviderName,
    SpanKind,
    StatusCode,
    ToolExecutionType,
)
from .._types import (
    AgentAttributes,
    CaptureIntegrity,
    ConversationContext,
    CorrelationInfo,
    GenAIAttributes,
    InternalSpan,
    SpanContext,
    SpanId,
    ToolAttributes,
    TraceId,
)
from ..protocol._claude_stream import AgentStreamEvent, parse_line

_BASE_LIMITATION = "transport_timing_unavailable_subprocess"
_MAX_OPEN_TOOLS = 256


def _safe_json_bytes(value: Any) -> bytes:
    """json.dumps a hook payload fragment; malformed/non-serializable input drops to b''."""
    try:
        return json.dumps(value).encode()
    except (TypeError, ValueError):
        return b""


@dataclass
class _OpenTool:
    tool_use_id: str | None
    name: str
    start_ns: int
    agent_id: str | None
    input_data: bytes
    from_hook: bool
    output_data: bytes = b""


@dataclass
class _Session:
    trace_id: TraceId
    root_ctx: SpanContext
    parent_span_id: SpanId | None  # user's ambient span at entry, if any
    start_ns: int
    session_id: str | None = None
    model: str | None = None  # from init -> request_model
    turn_start_ns: int = 0
    first_delta_ns: int = 0
    prompt: bytes = b""
    open_tools: dict[str, _OpenTool] = field(default_factory=dict)  # keyed by tool_use_id
    subagents: dict[str, tuple[SpanContext, str, int]] = field(default_factory=dict)
    # ^ agent_id -> (ctx, agent_type, start_ns)
    turn_index: int = 0
    stream_tool_meta: dict[str, tuple[str, bytes]] = field(default_factory=dict)
    # ^ tool_use_id -> (name, input_json) observed on the stream
    result: AgentStreamEvent | None = None
    error: str | None = None


class SessionAssembler:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._lock = threading.RLock()
        self._by_key: dict[int, _Session] = {}
        self._by_session_id: dict[str, _Session] = {}

    def open_session_count(self) -> int:
        with self._lock:
            return len(self._by_key)

    # --- ingestion ---

    def on_outbound(self, key: int, data: str) -> None:
        ev = parse_line(data.encode(), outbound=True)
        if ev is None:
            return
        now = time.time_ns()
        with self._lock:
            sess = self._ensure_session(key, now)
            sess.turn_start_ns = now
            sess.first_delta_ns = 0
            if ev.content_json and not sess.prompt:
                sess.prompt = ev.content_json

    def on_inbound(self, key: int, msg: dict) -> None:
        try:
            line = json.dumps(msg).encode()
        except (TypeError, ValueError):
            return
        ev = parse_line(line, outbound=False)
        if ev is None:
            return
        now = time.time_ns()
        with self._lock:
            sess = self._ensure_session(key, now)
            if ev.kind == "session_init":
                sess.session_id = ev.session_id
                sess.model = ev.model
                if ev.session_id:
                    self._by_session_id[ev.session_id] = sess
            elif ev.kind == "assistant_turn":
                self._emit_chat(sess, ev, now)
                for tu_id, tu_name, tu_input in ev.tool_uses:
                    sess.stream_tool_meta[tu_id] = (tu_name, tu_input)
            elif ev.kind == "tool_result":
                self._on_stream_tool_result(sess, ev, now)
            elif ev.kind == "stream_delta":
                if sess.first_delta_ns == 0:
                    sess.first_delta_ns = now
            elif ev.kind == "session_result":
                sess.result = ev

    def on_close(self, key: int, error: str | None) -> None:
        now = time.time_ns()
        with self._lock:
            sess = self._by_key.pop(key, None)
            if sess is None:
                return
            if sess.session_id:
                self._by_session_id.pop(sess.session_id, None)
            self._finalize(sess, error, now)

    def on_hook(self, event: str, payload: dict, tool_use_id: str | None) -> None:
        now = time.time_ns()
        with self._lock:
            sess = self._session_for_hook(payload)
            if sess is None:
                return
            if event == "PreToolUse":
                self._open_tool(sess, payload, tool_use_id, now)
            elif event in ("PostToolUse", "PostToolUseFailure"):
                self._close_tool(sess, payload, tool_use_id, now, failed=event.endswith("Failure"))
            elif event == "SubagentStart":
                agent_id = payload.get("agent_id")
                if agent_id:
                    ctx = SpanContext(trace_id=sess.trace_id, span_id=SpanId.generate())
                    sess.subagents[agent_id] = (ctx, payload.get("agent_type") or "sub_agent", now)
            elif event == "SubagentStop":
                self._emit_subagent(sess, payload.get("agent_id"), now)

    # --- emission helpers (all emit InternalSpan via self._client.capture_span) ---

    def _ensure_session(self, key: int, now: int) -> _Session:
        sess = self._by_key.get(key)
        if sess is not None:
            return sess
        active = _hub.get_current_scope().active_span_context
        if active is not None:
            trace_id = active.trace_id
            parent_span_id: SpanId | None = active.span_id
        else:
            trace_id = TraceId.generate()
            parent_span_id = None
        root_ctx = SpanContext(trace_id=trace_id, span_id=SpanId.generate())
        sess = _Session(
            trace_id=trace_id,
            root_ctx=root_ctx,
            parent_span_id=parent_span_id,
            start_ns=now,
        )
        self._by_key[key] = sess
        return sess

    def _session_for_hook(self, payload: dict) -> _Session | None:
        session_id = payload.get("session_id")
        if session_id is not None:
            sess = self._by_session_id.get(session_id)
            if sess is not None:
                return sess
        # Simple fallback: exactly one live session -> attribute the hook to it
        # (covers hooks arriving before the stream's session_init line lands).
        if len(self._by_key) == 1:
            return next(iter(self._by_key.values()))
        return None

    def _resolve_subagent_parent(self, sess: _Session, parent_tool_use_id: str | None) -> SpanId:
        """Best-effort join from a stream-side parent_tool_use_id to a subagent span.

        Direct match: parent_tool_use_id happens to be a known agent_id.
        Indirect match: parent_tool_use_id is a currently open tool that itself
        belongs to a subagent (nested activity inside a subagent's tool call).
        """
        if not parent_tool_use_id:
            return sess.root_ctx.span_id
        sub = sess.subagents.get(parent_tool_use_id)
        if sub is None:
            open_tool = sess.open_tools.get(parent_tool_use_id)
            if open_tool is not None and open_tool.agent_id is not None:
                sub = sess.subagents.get(open_tool.agent_id)
        return sub[0].span_id if sub is not None else sess.root_ctx.span_id

    def _emit_chat(self, sess: _Session, ev: AgentStreamEvent, now: int) -> None:
        parent_span_id = self._resolve_subagent_parent(sess, ev.parent_tool_use_id)
        start_ns = sess.turn_start_ns or now

        ttft: float | None = None
        if sess.first_delta_ns:
            ttft = (sess.first_delta_ns - sess.turn_start_ns) / 1e9
        limitations = (_BASE_LIMITATION,)
        if ttft is not None:
            limitations = limitations + ("ttft_ipc_approximation",)

        gen_ai = GenAIAttributes(
            operation=OperationName.CHAT,
            provider=ProviderName.ANTHROPIC,
            request_model=sess.model,
            response_model=ev.model,
            response_id=ev.message_id,
            input_tokens=ev.input_tokens,
            output_tokens=ev.output_tokens,
            cache_read_input_tokens=ev.cache_read_tokens,
            cache_creation_input_tokens=ev.cache_creation_tokens,
            finish_reasons=(ev.stop_reason,) if ev.stop_reason else None,
            time_to_first_chunk_s=ttft,
        )
        conversation = ConversationContext(
            conversation_id=sess.session_id or "",
            session_id=sess.session_id or "",
            turn_index=sess.turn_index,
        )
        is_first_turn = sess.turn_index == 0
        input_data = sess.prompt if is_first_turn else b""
        output_data = ev.content_json or b""

        span = InternalSpan(
            context=SpanContext(trace_id=sess.trace_id, span_id=SpanId.generate()),
            parent_span_id=parent_span_id,
            name=f"chat {ev.model}",
            kind=SpanKind.CLIENT,
            start_time_ns=start_ns,
            end_time_ns=now,
            status=StatusCode.OK,
            gen_ai=gen_ai,
            conversation=conversation,
            input_data=input_data,
            output_data=output_data,
            capture_sources=(CaptureSource.ADAPTER,),
            capture_integrity=CaptureIntegrity(
                request_body_captured=bool(input_data),
                response_body_captured=bool(output_data),
                limitations=limitations,
            ),
        )
        sess.turn_index += 1
        self._client.capture_span(span)

    def _open_tool(self, sess: _Session, payload: dict, tool_use_id: str | None, now: int) -> None:
        if tool_use_id is None:
            return
        if len(sess.open_tools) >= _MAX_OPEN_TOOLS:
            # Evict the oldest open entry (FIFO via dict insertion order) so the
            # session cannot accumulate unbounded open-tool state.
            oldest_id, oldest = next(iter(sess.open_tools.items()))
            del sess.open_tools[oldest_id]
            self._emit_tool(sess, oldest, now, markers=("tool_span_unclosed",))
        sess.open_tools[tool_use_id] = _OpenTool(
            tool_use_id=tool_use_id,
            name=payload.get("tool_name") or "unknown",
            start_ns=now,
            agent_id=payload.get("agent_id"),
            input_data=_safe_json_bytes(payload.get("tool_input", {})),
            from_hook=True,
        )

    def _close_tool(
        self, sess: _Session, payload: dict, tool_use_id: str | None, now: int, failed: bool
    ) -> None:
        if tool_use_id is None:
            return
        tool = sess.open_tools.pop(tool_use_id, None)
        if tool is None:
            tool = _OpenTool(
                tool_use_id=tool_use_id,
                name=payload.get("tool_name") or "unknown",
                start_ns=now,
                agent_id=payload.get("agent_id"),
                input_data=_safe_json_bytes(payload.get("tool_input", {})),
                from_hook=False,
            )
        meta = sess.stream_tool_meta.pop(tool_use_id, None)
        if meta is not None:
            stream_name, stream_input = meta
            if stream_name:
                tool.name = stream_name
            if not tool.input_data:
                tool.input_data = stream_input
        if "tool_response" in payload:
            tool.output_data = _safe_json_bytes(payload.get("tool_response"))
        self._emit_tool(sess, tool, now, failed=failed)

    def _emit_tool(
        self,
        sess: _Session,
        tool: _OpenTool,
        end_ns: int,
        failed: bool = False,
        markers: tuple[str, ...] = (),
    ) -> None:
        parent_span_id = sess.root_ctx.span_id
        if tool.agent_id is not None:
            sub = sess.subagents.get(tool.agent_id)
            if sub is not None:
                parent_span_id = sub[0].span_id

        span = InternalSpan(
            context=SpanContext(trace_id=sess.trace_id, span_id=SpanId.generate()),
            parent_span_id=parent_span_id,
            name=f"execute_tool {tool.name}",
            kind=SpanKind.INTERNAL,
            start_time_ns=tool.start_ns,
            end_time_ns=end_ns,
            status=StatusCode.ERROR if failed else StatusCode.OK,
            tool=ToolAttributes(
                name=tool.name,
                call_id=tool.tool_use_id,
                execution_type=ToolExecutionType.NETWORK,
            ),
            input_data=tool.input_data,
            output_data=tool.output_data,
            capture_sources=(CaptureSource.ADAPTER,),
            capture_integrity=CaptureIntegrity(
                request_body_captured=bool(tool.input_data),
                response_body_captured=bool(tool.output_data),
                limitations=(_BASE_LIMITATION, *markers),
            ),
            correlation=CorrelationInfo(
                request_id=tool.tool_use_id,
                confidence=1.0 if tool.from_hook else 0.7,
                strategy="adapter_hook" if tool.from_hook else "adapter_stream",
            ),
            extra=(("gen_ai.operation.name", OperationName.EXECUTE_TOOL.value),),
        )
        self._client.capture_span(span)

    def _on_stream_tool_result(self, sess: _Session, ev: AgentStreamEvent, now: int) -> None:
        tool_use_id = ev.parent_tool_use_id
        if tool_use_id is None:
            return
        if tool_use_id in sess.open_tools:
            # A PreToolUse hook already opened this call; the eventual
            # PostToolUse hook is the authority that will close it.
            return
        meta = sess.stream_tool_meta.pop(tool_use_id, None)
        if meta is None:
            # Already handled via the hook path (both open_tools and
            # stream_tool_meta are empty for this id) -> nothing to do.
            return
        name, input_json = meta
        tool = _OpenTool(
            tool_use_id=tool_use_id,
            name=name,
            start_ns=sess.turn_start_ns or now,
            agent_id=None,
            input_data=input_json,
            from_hook=False,
            output_data=ev.content_json or b"",
        )
        self._emit_tool(sess, tool, now)

    def _emit_subagent(self, sess: _Session, agent_id: str | None, now: int) -> None:
        if agent_id is None:
            return
        entry = sess.subagents.pop(agent_id, None)
        if entry is None:
            return
        ctx, agent_type, start_ns = entry
        span = InternalSpan(
            context=ctx,
            parent_span_id=sess.root_ctx.span_id,
            name=f"invoke_agent {agent_type}",
            kind=SpanKind.INTERNAL,
            start_time_ns=start_ns,
            end_time_ns=now,
            status=StatusCode.OK,
            agent=AgentAttributes(name=agent_type, id=agent_id, agent_type=AgentType.SUB_AGENT),
            capture_sources=(CaptureSource.ADAPTER,),
            capture_integrity=CaptureIntegrity(limitations=(_BASE_LIMITATION,)),
            extra=(("gen_ai.operation.name", OperationName.INVOKE_AGENT.value),),
        )
        self._client.capture_span(span)

    def _finalize(self, sess: _Session, error: str | None, now: int) -> None:
        # (1) Force-close any still-open tool spans — they never got a matching
        # PostToolUse/PostToolUseFailure hook before session teardown.
        for tool_use_id in list(sess.open_tools.keys()):
            tool = sess.open_tools.pop(tool_use_id)
            self._emit_tool(sess, tool, now, failed=True, markers=("tool_span_unclosed",))

        # (2) Emit any subagent spans that never received a SubagentStop.
        for agent_id in list(sess.subagents.keys()):
            self._emit_subagent(sess, agent_id, now)

        # (3) Root invoke_agent span.
        extra: list[tuple[str, str | int | float | bool]] = [
            ("gen_ai.operation.name", OperationName.INVOKE_AGENT.value)
        ]
        limitations = (_BASE_LIMITATION,)
        result = sess.result
        if result is not None:
            if result.num_turns is not None:
                extra.append(("wardex.agent.num_turns", result.num_turns))
            if result.total_cost_usd is not None:
                extra.append(("wardex.agent.cost_usd", result.total_cost_usd))
            if result.duration_api_ms is not None:
                extra.append(("wardex.agent.api_duration_ms", result.duration_api_ms))
            status = StatusCode.ERROR if result.is_error else StatusCode.OK
            if error is not None:
                status = StatusCode.ERROR
                limitations = limitations + ("session_aborted",)
        elif error is not None:
            status = StatusCode.ERROR
            limitations = limitations + ("session_aborted",)
        else:
            status = StatusCode.UNSET
            limitations = limitations + ("session_aborted",)

        span = InternalSpan(
            context=sess.root_ctx,
            parent_span_id=sess.parent_span_id,
            name="invoke_agent",
            kind=SpanKind.INTERNAL,
            start_time_ns=sess.start_ns,
            end_time_ns=now,
            status=status,
            agent=AgentAttributes(name=sess.model or "agent", agent_type=AgentType.PRIMARY),
            conversation=ConversationContext(
                conversation_id=sess.session_id or "",
                session_id=sess.session_id or "",
            ),
            capture_sources=(CaptureSource.ADAPTER,),
            capture_integrity=CaptureIntegrity(limitations=limitations),
            extra=tuple(extra),
        )
        self._client.capture_span(span)
