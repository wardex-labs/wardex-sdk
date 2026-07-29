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

from .. import _wardex_native
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
    ToolAttributes,
)
from ..assembly import (
    Evidence,
    Parentage,
    ParentSource,
    child_of,
    latch_ambient,
    resolve_parentage,
)
from ..protocol._claude_stream import AgentStreamEvent, parse_line

# Every span this assembler emits below the session root hangs off a context the
# parentage core produced and the session is holding — rule P2. Naming the
# evidence once here is what keeps the four emit paths from each inventing their
# own answer to "how did I know this was the parent". `UNIT_ACTIVE` is the
# forward-compatible spelling: step 6 turns the session into a real
# `assembly._units.Unit` that is activated around the framework call.
#
# NOTHING BELOW THE ROOT PUTS THIS ON THE WIRE, and that is deliberate. Only the
# session root's own edge — resolved from a real scope read in `_ensure_session`
# — reports a `CorrelationInfo` in step 1, which is exactly the delta design §11
# declares. The sub-root edges are still picked by heuristics this step does not
# own (`_resolve_subagent_anchor`'s fallback, `_session_for_hook`'s sole-session
# guess), so publishing `unit_active`/1.0 for them would assert certainty about
# a guess — I4's exact prohibition — and would ship a `strategy` value step 1
# never declared. Step 6 replaces those heuristics with `UnitRegistry.resolve()`,
# which returns evidence per edge; the correlation goes on the wire then, with
# the confidence and the marker the guess has earned.
_IN_SESSION = Evidence(ParentSource.UNIT_ACTIVE)

_BASE_LIMITATION = "transport_timing_unavailable_subprocess"


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
    parentage: Parentage  # the resolved edge from the user's scope to this session
    root_ctx: SpanContext  # the session root's own context — the anchor for P2
    start_ns: int
    session_id: str | None = None
    model: str | None = None  # from init -> request_model
    turn_start_ns: int = 0
    first_delta_ns: int = 0
    prompt: bytes = b""
    open_tools: dict[str, _OpenTool] = field(default_factory=dict)  # keyed by tool_use_id
    subagents: dict[str, tuple[SpanContext, str, int, Parentage]] = field(default_factory=dict)
    # ^ agent_id -> (ctx, agent_type, start_ns, parentage)
    turn_index: int = 0
    stream_tool_meta: dict[str, tuple[str, bytes]] = field(default_factory=dict)
    # ^ tool_use_id -> (name, input_json) observed on the stream
    result: AgentStreamEvent | None = None
    error: str | None = None


class SessionAssembler:
    def __init__(
        self,
        client: Any,
        skip_tool_names: set[str] | None = None,
        max_sessions: int | None = None,
        max_session_entries: int | None = None,
    ) -> None:
        self._client = client
        self._lock = threading.RLock()
        self._by_key: dict[int, _Session] = {}
        self._by_session_id: dict[str, _Session] = {}
        # None means "use the core default" — resolved here (rather than hardcoded)
        # so this can never silently drift from crates/wardex-limits.
        defaults = _wardex_native.limits_defaults()
        self._max_sessions = max_sessions if max_sessions is not None else defaults["max_sessions"]
        self._max_session_entries = (
            max_session_entries
            if max_session_entries is not None
            else defaults["max_session_entries"]
        )
        # Tool names handled by the adapter's in-process tool wrapper (execute_tool
        # spans opened directly around the handler call); hook-driven spans for
        # these names are skipped here to avoid emitting the call twice.
        self.skip_tool_names: set[str] = skip_tool_names if skip_tool_names is not None else set()

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
                    if len(sess.stream_tool_meta) < self._max_session_entries:
                        sess.stream_tool_meta[tu_id] = (tu_name, tu_input)
            elif ev.kind == "tool_result":
                self._on_stream_tool_result(sess, ev, now)
            elif ev.kind == "stream_delta":
                if sess.first_delta_ns == 0:
                    sess.first_delta_ns = now
            elif ev.kind == "session_result":
                sess.result = ev
            elif ev.kind == "task_lifecycle":
                # Deliberately unconsumed for now: subagent spans are built from
                # SubagentStart/Stop hooks; task usage enrichment is a deferred
                # follow-up. Parsed and exposed so the wire surface is stable.
                pass

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
                if agent_id and len(sess.subagents) < self._max_session_entries:
                    p = child_of(sess.root_ctx, _IN_SESSION)
                    sess.subagents[agent_id] = (
                        p.child_context(),
                        payload.get("agent_type") or "sub_agent",
                        now,
                        p,
                    )
            elif event == "SubagentStop":
                self._emit_subagent(sess, payload.get("agent_id"), now)

    # --- emission helpers (all emit InternalSpan via self._client.capture_span) ---

    def _ensure_session(self, key: int, now: int) -> _Session:
        sess = self._by_key.get(key)
        if sess is not None:
            return sess
        # The one scope read of the whole adapter. Everything below the root is
        # anchored to `root_ctx`, so `child_context()` is called exactly once
        # here: calling it again would anchor the children to a span id that is
        # never emitted (see `Parentage.child_context`).
        parentage = resolve_parentage(latch_ambient())
        sess = _Session(
            parentage=parentage,
            root_ctx=parentage.child_context(),
            start_ns=now,
        )
        self._new_session(key, sess)
        return sess

    def _new_session(self, key: int, sess: _Session) -> None:
        """Insert a session, evicting the oldest when the cap is reached.

        Sessions are removed on close, but a transport that never closes would
        otherwise accumulate them for the process lifetime.
        """
        if len(self._by_key) >= self._max_sessions:
            old_key = next(iter(self._by_key))
            old = self._by_key.pop(old_key)
            if old.session_id:
                self._by_session_id.pop(old.session_id, None)
        self._by_key[key] = sess

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

    def _resolve_subagent_anchor(
        self, sess: _Session, parent_tool_use_id: str | None
    ) -> SpanContext:
        """Best-effort join from a stream-side parent_tool_use_id to a subagent span.

        Direct match: parent_tool_use_id happens to be a known agent_id.
        Indirect match: parent_tool_use_id is a currently open tool that itself
        belongs to a subagent (nested activity inside a subagent's tool call).

        Returns the ANCHOR (a context this session already holds), not a span id:
        the edge itself is `child_of`'s to build. Selecting which anchor is still
        this method's job, and it is still a heuristic — a `parent_tool_use_id`
        that resolves to nothing (the hook has not landed yet, or the subagent
        was never recorded because `_max_session_entries` was reached) silently
        re-parents to the session root. That is unchanged from before step 1 and
        step 6 is where it gets fixed: `UnitRegistry.resolve()` returns the
        evidence alongside the unit, so the guess reports itself (design §3.4).
        Until it does, no span this method feeds may claim a confidence for its
        edge — see `_IN_SESSION`.
        """
        if not parent_tool_use_id:
            return sess.root_ctx
        sub = sess.subagents.get(parent_tool_use_id)
        if sub is None:
            open_tool = sess.open_tools.get(parent_tool_use_id)
            if open_tool is not None and open_tool.agent_id is not None:
                sub = sess.subagents.get(open_tool.agent_id)
        return sub[0] if sub is not None else sess.root_ctx

    def _emit_chat(self, sess: _Session, ev: AgentStreamEvent, now: int) -> None:
        p = child_of(self._resolve_subagent_anchor(sess, ev.parent_tool_use_id), _IN_SESSION)
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
            context=p.child_context(),
            parent_span_id=p.parent_span_id,
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
            # No `correlation=p.correlation`: the anchor above may be a fallback
            # (see `_resolve_subagent_anchor`), and `p.correlation` would report
            # it as `unit_active`/1.0 with no marker. Unchanged from before step
            # 1 — this span still makes no claim — because a claim it cannot
            # back is strictly worse than no claim (I4). Step 6 supplies the
            # evidence, and the field arrives with it.
        )
        sess.turn_index += 1
        self._client.capture_span(span)

    def _open_tool(self, sess: _Session, payload: dict, tool_use_id: str | None, now: int) -> None:
        if tool_use_id is None:
            return
        if payload.get("tool_name") in self.skip_tool_names:
            # In-process tool: the adapter's handler wrapper opens its own
            # execute_tool span; skip the hook-driven one to avoid double emission.
            return
        if len(sess.open_tools) >= self._max_session_entries:
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
        if payload.get("tool_name") in self.skip_tool_names:
            # Matches the _open_tool skip: nothing was opened for this call, and
            # the in-process handler wrapper owns its own span's lifecycle.
            sess.stream_tool_meta.pop(tool_use_id, None)
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
            # Content authority: the byte-exact stream input always wins over
            # the hook's re-serialized tool_input when the stream saw it.
            if stream_input:
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
        anchor = sess.root_ctx
        if tool.agent_id is not None:
            sub = sess.subagents.get(tool.agent_id)
            if sub is not None:
                anchor = sub[0]
        p = child_of(anchor, _IN_SESSION)

        span = InternalSpan(
            context=p.child_context(),
            parent_span_id=p.parent_span_id,
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
            # NOT `p.correlation`, deliberately. The edge above came from the
            # core; this field is the OTHER axis, and today it still mixes the
            # two: `adapter_hook`/`adapter_stream` answer "which source observed
            # the event", which design §4.1 moves to `capture_sources` and §11
            # schedules for step 3b. Overwriting it here would silently retire a
            # declared wire value one step early, so it stays until the step that
            # owns the vocabulary change replaces it with
            # `capture_sources=(ADAPTER,)` + `strategy=unit_active`.
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
        if name in self.skip_tool_names:
            # In-process tool without a matching hook observation (or one that
            # hasn't landed yet): the handler wrapper's own span is authoritative,
            # so skip the stream-only fallback path too.
            return
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
        ctx, agent_type, start_ns, p = entry
        span = InternalSpan(
            context=ctx,
            parent_span_id=p.parent_span_id,
            name=f"invoke_agent {agent_type}",
            kind=SpanKind.INTERNAL,
            start_time_ns=start_ns,
            end_time_ns=now,
            status=StatusCode.OK,
            agent=AgentAttributes(name=agent_type, id=agent_id, agent_type=AgentType.SUB_AGENT),
            capture_sources=(CaptureSource.ADAPTER,),
            capture_integrity=CaptureIntegrity(limitations=(_BASE_LIMITATION,)),
            # No `correlation=p.correlation`, for the reason at `_IN_SESSION`:
            # the edge to the session root is exact, but WHICH session this hook
            # belongs to is `_session_for_hook`'s sole-live-session guess, and
            # `p.correlation` reports the whole thing at confidence 1.0 with no
            # marker. Unchanged from before step 1.
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
            parent_span_id=sess.parentage.parent_span_id,
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
            correlation=sess.parentage.correlation,
            extra=tuple(extra),
        )
        self._client.capture_span(span)
