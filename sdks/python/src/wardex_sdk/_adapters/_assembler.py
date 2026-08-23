"""Session assembler: correlates stream events and hook events into spans.

Sources: the transport tee (raw JSON lines -> native parser) and SDK hooks.
Rule (spec §6.2): hooks are the authority for lifecycle/attribution — and for
the user-turn boundary — while the stream is the authority for content; joined
on tool_use_id. The hook's `UserPromptSubmit` prompt payload is the content
FALLBACK for exactly the turn whose outbound write the stream did not record.
All timestamps are host-arrival times (IPC level).

Two things this module used to hold now live next door, and the split is by how
long each one stays here. `_session_state.py` holds what the assembler
REMEMBERS between two events — records shaped by the Agent SDK's own vocabulary,
so adapter-specific by construction. `_sink.py` holds the one piece that is not
about Anthropic at all, and its destination is `_assembly/_emit.py`.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from typing import Any, TypeVar

from .._assembly import (
    Evidence,
    Limitation,
    ParentSource,
    SpanDraft,
    SpanIntent,
    Unit,
    UnitKey,
    UnitKind,
    UnitRegistry,
    child_of,
    counters,
    guard,
    latch_ambient,
)
from .._enums import (
    AgentType,
    CaptureSource,
    ProviderName,
    StatusCode,
    ToolExecutionType,
)
from .._limits import LimitsConfig, LimitsConsumer, limits_kwargs
from .._protocol._claude_stream import AgentStreamEvent, parse_line
from .._types import (
    AgentAttributes,
    ConversationContext,
    GenAIAttributes,
    ToolAttributes,
)
from ._anthropic_names import McpToolCatalog
from ._otel_merge import (
    OTEL_EXTRA_PREFIX,
    _ChatWindow,
    _OtelSpan,
    allowlisted_extras,
    classify,
    join_chats,
)
from ._session_state import (
    _BridgeBinding,
    _EvictedSubagent,
    _EvictedTool,
    _OpenSubagent,
    _OpenTool,
    _PendingSpan,
    _Session,
)
from ._sink import _ClientSink

# Every span this assembler emits below the session root hangs off a context the
# parentage core produced and the session is holding — rule P2. Naming the
# evidence once here is what keeps the four emit paths from each inventing their
# own answer to "how did I know this was the parent". The anchor is a real
# `assembly._units.Unit`, which is what makes `UNIT_ACTIVE` literally
# true: the session unit is pinned onto the SDK's reader task, so the hook
# callbacks and in-process tool handlers that arrive on tasks spawned from it
# inherit it by ordinary ContextVar copying.
#
# NOTHING BELOW THE ROOT PUTS THIS ON THE WIRE, and that is STILL deliberate.
# The session root's own edge is resolved from a real scope read, and an
# in-process tool call's edge is resolved by the registry — both publish a
# `CorrelationInfo` and both have earned it. The remaining sub-root edges here
# do not: which sub-agent a stream event belongs to is
# `_resolve_subagent_anchor`'s guess and which session a hook belongs to is
# `_session_for_hook`'s, and both silently fall back to the session root. Those
# two are the ingestion code design §3.4 moves into `_normalize.py`, where they
# produce a `UnitKey` for `UnitRegistry.resolve()` instead of choosing an anchor
# themselves; until then, publishing `unit_active`/1.0 for an edge picked that
# way would assert certainty about a guess, which is I4's exact prohibition.
_IN_SESSION = Evidence(ParentSource.UNIT_ACTIVE)

# Rides all four span classes THIS MODULE builds — the root `invoke_agent`, a
# sub-agent `invoke_agent`, `chat`, and the hook/stream-driven `execute_tool` —
# because every one of them is assembled out of the IPC stream and the hook
# payloads: the work happened inside a CLI subprocess and wardex observed only
# the pipe, so there is no transport timing at all — not zero timing, absent
# timing.
#
# NOT the whole adapter. The in-process `execute_tool` span that
# `_anthropic_agent_sdk.py::_run_tool` opens brackets a handler wardex wrapped
# in this process, so its duration is measured directly; attaching this marker
# there would claim the timing is absent when it is the one timing the adapter
# owns.
_BASE_LIMITATION = Limitation.TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS

#: Value type of a per-session table, so `_room_for` hands the caller back the
#: record it evicted rather than an `Any` the caller has to re-narrow.
_TableValue = TypeVar("_TableValue")

#: Whose runs these are. It must equal the adapter's `name()`, because that is
#: what the `AdapterContext` is built with and therefore what `sole_live(...,
#: owner=)` and `close_all(owner=)` filter on: a session opened here with a
#: different owner — or none — is one this adapter's own tool wrapper cannot
#: find, and `sole_live` treats an unowned unit as matching nothing rather than
#: everything, precisely so a guess cannot cross a framework boundary.
#: `test_agent_sdk_units.py` asserts the two spellings agree.
_OWNER = "anthropic_agent_sdk"


#: The two provenances a chat span's `input_data` can have, published as the
#: `wardex.agent.prompt_source` extra on every chat span that carries a prompt.
#: They double as the internal pending-source states on `_Session`. The two
#: SHAPES differ on the wire, and the extra is what lets a consumer parse
#: `input_data`: "stream" is the byte-exact message-object JSON slice from the
#: transport tee (content authority); "hook" is the CLI's re-decoded prompt
#: text from the `UserPromptSubmit` payload — the degraded fallback for a write
#: the stream did not record, published as the text wardex actually saw rather
#: than dressed up as a message object wardex never saw.
_PROMPT_STREAM = "stream"
_PROMPT_HOOK = "hook"


#: The rank the HOOK observer claims a tool call at. The in-process handler
#: wrapper claims the same key at 10, and the higher rank wins however late it
#: arrives — `PreToolUse` fires BEFORE the handler body, so first-come would hand
#: every in-process tool to the observer that did not wrap the execution.
HOOK_RANK = 0


def outranked(unit: Unit, key: UnitKey, rank: int) -> bool:
    """Has a HIGHER-ranked observer taken `key` since we claimed it?

    Not `claim() is False`. `claim()` refuses an EQUAL rank too, which is how it
    keeps one observer from silently replacing another of the same standing — but
    two hook observations of two concurrent `Bash` calls share one name key at
    one rank, and reading that refusal as "someone else owns this" would delete
    the second call's span. The question that decides ownership is strictly
    "does something outrank me", and this is it.
    """
    owner = unit.owner_rank(key)
    return owner is not None and owner > rank


def _safe_json_bytes(value: Any) -> bytes:
    """json.dumps a hook payload fragment; malformed/non-serializable input drops to b''."""
    try:
        return json.dumps(value).encode()
    except (TypeError, ValueError):
        return b""


class SessionAssembler:
    def __init__(
        self,
        client: Any,
        *,
        units: UnitRegistry | None = None,
        names: McpToolCatalog | None = None,
        max_sessions: int | None = None,
        max_session_entries: int | None = None,
        bridge: Any = None,
    ) -> None:
        self._client = client
        self._lock = threading.RLock()
        # The OTel bridge receiver (`_otel_receiver._OtelBridgeReceiver`), or
        # None — the default, which keeps every bridge branch below dead and
        # the off path line-for-line today's code. A session additionally
        # gates on its own `sess.bridge` binding, so even with a receiver
        # here, a session the injection never reached behaves as today.
        self._bridge = bridge
        self._by_key: dict[int, _Session] = {}
        self._by_session_id: dict[str, _Session] = {}
        # Bounds for the tables this assembler may have to BUILD, resolved from
        # the CLIENT's config and never from a bare `LimitsConfig()`. A caller
        # that hands us no registry still configured one, and process defaults
        # would silently replace the user's values — which is the exact half of
        # this class that was already doing so. `client=None` still yields the
        # core defaults, which is the honest answer for a caller with no config
        # at all.
        cfg = getattr(client, "config", None)
        lim = cfg.limits if cfg is not None else LimitsConfig()
        resolved = lim.resolved()
        self._max_sessions = max_sessions if max_sessions is not None else resolved["max_sessions"]
        self._max_session_entries = (
            max_session_entries
            if max_session_entries is not None
            else resolved["max_session_entries"]
        )
        # The unit registry is what makes a framework identifier a LOOKUP KEY and
        # nothing else: every parent this assembler hands out comes from a unit
        # whose own context `_assembly/_parentage.py` produced from a real scope
        # read. `skip_tool_names` — a set of raw strings, passed by reference and
        # compared against a name the CLI spells differently — is what it
        # replaces; `Unit.claim()` arbitrates on a normalized key instead.
        #
        # SUPPLIED, not built, when the adapter has one: the registry the
        # adapter's `AdapterContext` holds must be the SAME object, or `owner`
        # scoping is meaningless — `sole_live(owner=...)` and
        # `close_all(owner=...)` answer questions about one table, and two
        # tables give two adapters no way to be told apart inside either. Built
        # here only for a caller that has no context yet, which is every test
        # that drives this class directly — and one production path: an adapter
        # installed by hand, without going through `wardex.init()`. That path is
        # documented and warned about, not unsupported, so the registry it gets
        # owes the host every bound the host configured. It used to get two of
        # the four, forwarded from the adapter, and there was no third place a
        # bound could arrive from.
        self._units = (
            units
            if units is not None
            else UnitRegistry(
                sink=_ClientSink(client),
                debug=bool(getattr(cfg, "debug", False)),
                **limits_kwargs(LimitsConsumer.UNIT_REGISTRY, resolved),
            )
        )
        # The shared tool-name space (design §5.4). Empty when the adapter did not
        # supply one, which is the correct reading for an assembler with no
        # in-process servers registered: every hook name then resolves to its own
        # key and the hook observer owns every call.
        if names is not None:
            self._names = names
        else:
            self._names = McpToolCatalog()
            self._names.apply_bound(**limits_kwargs(LimitsConsumer.MCP_TOOL_CATALOG, resolved))

    @property
    def units(self) -> UnitRegistry:
        """The registry, for the adapter's in-process tool wrapper and the pin."""
        return self._units

    def open_session_count(self) -> int:
        with self._lock:
            return len(self._by_key)

    def unit_for(self, key: int) -> Unit | None:
        """The live session unit for a transport key, if there is one."""
        with self._lock:
            sess = self._by_key.get(key)
        return sess.unit if sess is not None and sess.unit.is_live else None

    def bridge_route(self, key: int) -> tuple[str | None, str | None] | None:
        """`(trace_id_hex, session_id)` for a live bridge-bound session, else None.

        The adapter's drain plan reads this before deciding whether to wait at
        all: None — no session, no binding, or a dead unit — means the close
        proceeds at full speed, which is the drain's own latency gate.
        """
        with self._lock:
            sess = self._by_key.get(key)
            if sess is None or sess.bridge is None or not sess.unit.is_live:
                return None
            return (sess.bridge.trace_id_hex, sess.session_id)

    def pin_reader(self, key: int, owner_task: object) -> bool:
        """Pin the session unit onto the task driving this transport's reader.

        The whole mechanism, in one call. An async generator body has no context
        of its own — its frames run in the context of the task that DRIVES it —
        so a `ContextVar.set()` performed inside the transport's message loop
        lands on the SDK's reader task and stays there. Every hook callback and
        every in-process MCP tool handler is dispatched from tasks spawned by
        that loop, so they inherit the session by ordinary context copying:
        confidence 1.0, zero framework identifiers.

        Returns whether the pin is installed. Refused pins are the registry's
        business (a pin declared for a task other than the caller is recorded as
        a `CORRELATION_CONFLICT` on the unit's own span); the caller retries on
        the next message, which is what covers the ordinary case of the reader
        being driven before the first outbound write created the session.
        """
        unit = self.unit_for(key)
        if unit is None:
            return False
        return self._units.pin_driver(unit, owner_task=owner_task).installed

    # --- ingestion ---

    def on_outbound(self, key: int, data: str, bridge: _BridgeBinding | None = None) -> None:
        """Host -> CLI write. A main-thread user message installs the pending prompt.

        The `parent_tool_use_id is None` gate is SYMMETRIC with consumption:
        `_build_chat` consumes the pending prompt only for a chat whose own
        `parent_tool_use_id` is None, i.e. a main-thread assistant turn. An
        outbound user line that carries one — a host-written message addressed
        into a sub-agent's thread — is not the next main-thread turn's prompt,
        so installing it here would overwrite a prompt the main thread has not
        consumed yet and charge the loss to a turn that never died.

        `bridge` is the adapter's injection-correlation verdict for this
        transport (the subprocess-env read-back), attached once, when the
        session it names first exists. It rides the outbound WRITE because
        that is the event that creates sessions — a binding cannot predate the
        thing it binds.
        """
        ev = parse_line(data.encode(), outbound=True)
        if ev is None:
            return
        now = time.time_ns()
        with self._lock:
            sess = self._ensure_session(key, now)
            if bridge is not None and sess.bridge is None and self._bridge is not None:
                sess.bridge = bridge
            sess.turn_start_ns = now
            sess.first_delta_ns = 0
            if ev.content_json and ev.parent_tool_use_id is None:
                if sess.pending_prompt_source is not None:
                    # A prompt was seen and no chat span ever consumed it. It
                    # is lost now, and the loss is counted rather than silent.
                    counters.bump("adapters.assembler.prompt_overwritten")
                sess.pending_prompt = ev.content_json
                sess.pending_prompt_source = _PROMPT_STREAM
                sess.pending_prompt_hook_seen = False

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
                if sess.session_id and ev.session_id and sess.session_id != ev.session_id:
                    # A SECOND `system/init`, naming a different run, on a
                    # session this table still holds live. One CLI subprocess
                    # emits that line once, so the transport key now names a
                    # different subprocess — the previous one went away without
                    # its close reaching us, and CPython handed its identity to
                    # the next object.
                    #
                    # MARKED, not split. Everything from here on is filed under
                    # the earlier run's root, so two agent runs share one trace
                    # and the tree says nothing about it; the marker is what
                    # makes that legible. Retiring the old session instead is
                    # the correct end state and is NOT done here: the same
                    # symptom would follow from a CLI that legitimately re-inits
                    # one transport, and splitting a real run into two traces to
                    # fix a merge is a wrong tree of the other shape. That call
                    # needs the lifecycle rework, and evidence from a live CLI.
                    counters.bump("adapters.assembler.session_key_recycled")
                    sess.unit.note(Limitation.CORRELATION_CONFLICT)
                sess.session_id = ev.session_id
                sess.model = ev.model
                if ev.session_id:
                    self._by_session_id[ev.session_id] = sess
                    # The CLI's own session id, recorded in the registry as a
                    # lookup ALIAS onto the session unit — the only shape I2 lets
                    # a framework id take, and the reason it goes here as well as
                    # into `_by_session_id` below.
                    #
                    # A RECORD, not a route. Nothing in this SDK calls
                    # `UnitRegistry.find()` or `resolve()`, so if the pin ever
                    # stopped holding, this alias would not be consulted and no
                    # `CORRELATION_CONFLICT` would come from it. What answers
                    # instead is one tier down and marks itself: on the hook path
                    # `_session_for_hook` falls from the scope to this adapter's
                    # own `_by_session_id` table and then to a counted sole-live
                    # inference, and the tool wrapper's declared
                    # `Fallback.SOLE_LIVE_RUN` falls to the one live run of this
                    # adapter's own, at 0.5 with `UNIT_INFERRED_SOLE`.
                    self._units.alias(
                        UnitKey("transport.id", str(key)),
                        UnitKey("claude.session_id", ev.session_id),
                    )
            elif ev.kind == "assistant_turn":
                self._emit_chat(sess, ev, now)
                for tu_id, tu_name, tu_input in ev.tool_uses:
                    # Refuse the NEWEST and keep the oldest, which is the
                    # opposite of the two tables above and deliberate. Nothing
                    # here owns a span, and both consumers
                    # (`_close_tool`, `_on_stream_tool_result`) pop by the id
                    # whose RESULT arrived — in a turn, results come back
                    # broadly in the order the uses were announced, so the
                    # oldest entry is the one most likely to be read next.
                    # Evicting it would discard exactly that.
                    if self._has_room(sess.stream_tool_meta, "stream_tool_meta"):
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
            # The liveness check and not a bare pop, because a transport can
            # close AFTER the registry evicted its root and nothing else arrived
            # in between — the one route to `_finalize` that no other event
            # guards. Finalizing a retired session would `_stamp_root` a draft
            # whose span shipped seconds ago (`finish()` neither freezes nor
            # refuses a second call) and then emit nothing, since
            # `UnitRegistry.close()` returns empty for a unit already closed. The
            # retirement path drains what the session still held open against the
            # span it actually hung off, and says so in a counter.
            sess = self._live_session(key, now)
            if sess is None:
                return
            self._by_key.pop(key, None)
            if sess.session_id:
                self._by_session_id.pop(sess.session_id, None)
            self._finalize(sess, error, now)

    def on_hook(self, event: str, payload: dict, tool_use_id: str | None) -> None:
        now = time.time_ns()
        with self._lock:
            sess = self._session_for_hook(payload, now)
            if sess is None:
                return
            if event == "PreToolUse":
                self._open_tool(sess, payload, tool_use_id, now)
            elif event in ("PostToolUse", "PostToolUseFailure"):
                self._close_tool(sess, payload, tool_use_id, now, failed=event.endswith("Failure"))
            elif event == "SubagentStart":
                agent_id = payload.get("agent_id")
                if agent_id:
                    evicted = self._room_for(sess.subagents, "subagent")
                    if evicted is not None:
                        # This table used to REFUSE the newest entry with no
                        # span and no counter — the same bound as the open-tool
                        # table saying nothing where its sibling said the wrong
                        # thing. Evicting the oldest and emitting it is what the
                        # bound owes an entry that owns a span.
                        evicted_id, entry = evicted
                        self._emit_subagent_entry(
                            sess,
                            entry,
                            evicted_id,
                            now,
                            markers=(Limitation.SESSION_ENTRY_TABLE_FULL,),
                            status=StatusCode.UNSET,
                        )
                        # And the ANCHOR outlives the span, because the three
                        # lookups that resolve a sub-agent do so at EMIT time.
                        # Without this, evicting the oldest — i.e. the
                        # longest-lived, outermost one — would re-parent every
                        # still-open tool and every later chat turn beneath it
                        # onto the session root and say nothing: one silent drop
                        # traded for a whole silently flattened subtree. A
                        # context stays a valid parent after its span ships.
                        self._room_for(sess.evicted_subagents, "evicted_subagent")
                        sess.evicted_subagents[evicted_id] = _EvictedSubagent(
                            context=entry.draft.context, agent_type=entry.agent_type
                        )
                    agent_type = payload.get("agent_type") or "sub_agent"
                    draft = SpanDraft(
                        child_of(sess.unit.context, _IN_SESSION),
                        intent=SpanIntent.INVOKE_AGENT,
                        subject=agent_type,
                        source=CaptureSource.ADAPTER,
                        start_ns=now,
                    )
                    draft.set_agent(
                        AgentAttributes(
                            name=agent_type, id=agent_id, agent_type=AgentType.SUB_AGENT
                        )
                    )
                    draft.add_limitation(_BASE_LIMITATION)
                    # See `_IN_SESSION`: which session this hook belongs to is
                    # `_session_for_hook`'s guess, so the span makes no claim.
                    draft.replace_correlation(None)
                    sess.subagents[agent_id] = _OpenSubagent(draft=draft, agent_type=agent_type)
            elif event == "SubagentStop":
                self._emit_subagent(sess, payload.get("agent_id"), now)
            elif event == "UserPromptSubmit":
                self._on_prompt_submit(sess, payload, now)

    def _on_prompt_submit(self, sess: _Session, payload: dict, now: int) -> None:
        """Consume a `UserPromptSubmit` hook: corroborate the stream, or fall back.

        The stream stays the CONTENT authority. A pending stream prompt not yet
        corroborated IS this prompt: the write that caused this hook passed
        through the tee before the CLI could act on it, and the CLI blocks on
        hook responses before inference, so the pending bytes and this payload
        describe one user turn — content stays byte-exact from the stream, and
        remembering the corroboration is what lets a SECOND submit while the
        same prompt still pends read as a new user turn whose write the stream
        missed, not as a duplicate of this one.

        Every other state means this hook is the only observation of its turn:
        no pending prompt (the stream missed the write), a pending hook prompt
        (two prompts with no assistant turn between), or an already-corroborated
        stream prompt (the previous turn died unconsumed AND this turn's write
        was missed). Then the hook's re-decoded text is captured as the degraded
        content fallback, and the hook arrival sets the turn boundary — and ONLY
        then: when the stream saw the write, hook arrival is polluted by user
        matchers that run before wardex's appended one, while the tee'd write is
        causally earlier and unpolluted.

        The fallback cannot create a session: a hook for a transport whose
        writes never parsed is dropped (counted) by `_session_for_hook` before
        this method runs, so it covers prompt-line misses within an observed
        session only — it is not stream-independence.
        """
        if sess.pending_prompt_source == _PROMPT_STREAM and not sess.pending_prompt_hook_seen:
            sess.pending_prompt_hook_seen = True
            return
        if sess.pending_prompt_source is not None:
            counters.bump("adapters.assembler.prompt_overwritten")
        prompt = payload.get("prompt")
        sess.pending_prompt = prompt.encode() if isinstance(prompt, str) else b""
        sess.pending_prompt_source = _PROMPT_HOOK
        sess.pending_prompt_hook_seen = True
        sess.turn_start_ns = now
        sess.first_delta_ns = 0

    # --- emission helpers (all build via SpanDraft, emit via capture_span) ---

    def _guard(self, where: str) -> guard:
        """The authorized swallow. `SpanDraft.finish()` raises `VocabularyError`
        on a vocabulary breach, and this code runs inside the host's own hook
        callbacks and transport tee — I6 forbids that reaching them."""
        config = getattr(self._client, "config", None)
        return guard(where, debug=bool(getattr(config, "debug", False)))

    def _capture(self, span: Any) -> None:
        """This class's ONE exit to the sink (C-S5): every finished span leaves
        through here, so a new emit path extends a list of callers rather than
        multiplying direct sink call sites."""
        self._client.capture_span(span)

    def _live_session(self, key: int, now: int) -> _Session | None:
        """The session filed under `key`, but only while its unit is still alive.

        Caller holds `self._lock`.

        Two tables hold the same objects: this assembler's `_by_key`, bounded by
        `max_sessions`, and `UnitRegistry._roots`, bounded by `max_units`. They
        are independent knobs sourced from different fields of the same core
        struct, and nothing relates them — so the REGISTRY can evict a session
        root while this side still believes the session is running. Nothing
        tells us. The eviction has already shipped that root as a semantic stub
        (`UNIT_EVICTED`, no model, no conversation), and every event after it
        would be built against a corpse: the pin can no longer be installed,
        `claim()` arbitrates on a unit the in-process handler path can no longer
        reach, and `UnitRegistry.close()` would emit nothing at all — silently
        discarding everything `_stamp_root` had just written onto the draft.

        So this asks, on every lookup. `unit_for` was already the one place that
        did; this is that check made unavoidable.

        ASKING, not being told, and the choice is forced. The registry evicts
        from inside `open()` while holding its own lock, and this assembler
        calls `open()` while holding `self._lock` — a callback would therefore
        take the two locks in opposite orders on two threads. One attribute read
        on a path we already walk has no ordering to get wrong.

        The retired session is NOT re-finalized. Its draft has already been
        emitted, and `SpanDraft.finish()` neither freezes the draft nor refuses a
        second call, so stamping it now would mutate a span that shipped seconds
        ago. What it still held open is closed out against the span it actually
        belongs to, and the run's IDENTITY is carried onto the replacement root
        by `_resume`.
        """
        sess = self._by_key.get(key)
        if sess is None:
            return None
        if sess.unit.is_live:
            return sess
        counters.bump("adapters.assembler.session_unit_evicted")
        self._by_key.pop(key, None)
        if sess.session_id:
            self._by_session_id.pop(sess.session_id, None)
        self._drain_children(sess, now)
        # The retirement half of the pending buffer's conservation rule: the
        # root already shipped, so there is no finalize-time merge left to
        # wait for — every held draft goes out NOW, unmerged, deferred markers
        # applied, and the receiver slot is unfiled so it cannot outlive the
        # session that reserved it.
        self._flush_pending(sess)
        self._retire_bridge(sess)
        return None

    def _retire_bridge(self, sess: _Session) -> None:
        """Unfile the receiver slot of a session retired without a merge.

        The retirement path runs mid-event with the lock held and a root that
        shipped seconds ago — there is nothing safe to merge INTO, so the slot
        is taken and dropped (counted). Its spans were the CLI's copy of work
        whose wardex spans just flushed unmerged; keeping the slot would only
        let it grow until the receiver's own bound evicted it.
        """
        if sess.bridge is None or self._bridge is None:
            return
        with self._guard("adapters.assembler.otel_bridge_retire"):
            self._bridge.take(sess.bridge.trace_id_hex, sess.session_id)
        counters.bump("adapters.assembler.otel_bridge_retired")

    def _resume(self, sess: _Session, previous: _Session) -> None:
        """Carry a retired session's identity onto the root that replaces it.

        A fresh unit issues a fresh `issued_conversation_id`, so without this the
        stub the eviction shipped and everything recorded after it would land in
        different conversation buckets — one run would read as two unrelated
        agents, and neither would say why. The identity carries; the in-flight
        state does not, because `_live_session` has already closed that out
        against the span it hung off. `UNIT_EVICTED` rides the NEW root as well,
        which is what turns "a second root appeared from nowhere" into "this run
        was truncated and resumes here".
        """
        sess.session_id = previous.session_id
        sess.issued_conversation_id = previous.issued_conversation_id
        sess.model = previous.model
        # Turn numbering is per CONVERSATION and the conversation continues.
        # Restarting at 0 would give the resumed turns the indices the retired
        # ones already used, under the same `conversation_id`.
        sess.turn_index = previous.turn_index
        sess.unit.note(Limitation.UNIT_EVICTED)
        if sess.session_id:
            self._by_session_id[sess.session_id] = sess

    def _ensure_session(self, key: int, now: int) -> _Session:
        # Read BEFORE the liveness check, because that check is what retires a
        # session whose root the registry evicted. `previous` is therefore the
        # session that was just retired, or None — never a live one, since a live
        # one returns below.
        previous = self._by_key.get(key)
        sess = self._live_session(key, now)
        if sess is not None:
            return sess
        # The one scope read of the whole adapter, and it happens HERE — on the
        # task that issued the work — because `latch_ambient()` on the response
        # path reads a scope that has already moved on. Everything below the root
        # is anchored to `sess.unit.context`, and that context exists exactly once
        # because the ROOT SPAN ITSELF holds it: the draft the registry opens here
        # is the span `_finalize` eventually emits. Allocating a bare context and
        # rebuilding the span from it later is what made it possible to anchor
        # children to a span id that is never emitted.
        self._make_room(now)
        unit = self._units.open(
            UnitKind.SESSION,
            UnitKey("transport.id", str(key)),
            ambient=latch_ambient(),
            intent=SpanIntent.INVOKE_AGENT,
            start_ns=now,
            owner=_OWNER,
        )
        # The required block AT OPEN, not only at finalize. A unit the registry
        # evicts is emitted immediately, and a draft missing `agent` is one
        # `finish()` refuses — so an eviction would leave a counter and no span,
        # which is the silent drop I10 exists to end. `_stamp_root` overwrites it
        # once the stream reports a model.
        unit.draft.set_agent(AgentAttributes(name="agent", agent_type=AgentType.PRIMARY))
        unit.draft.add_limitation(_BASE_LIMITATION)
        sess = _Session(unit=unit, start_ns=now, key=key)
        if previous is not None:
            self._resume(sess, previous)
        self._by_key[key] = sess
        return sess

    def _conversation(self, sess: _Session, *, turn_index: int = 0) -> ConversationContext:
        """The session's conversation identity, never the empty string (§6.3).

        `session_id` stays optional — it is the CLI's id and the CLI may not
        have reported one yet — but `conversation_id` is what a store keys on,
        so wardex issues a stable one per session rather than shipping "".
        """
        return ConversationContext(
            conversation_id=sess.session_id or sess.issued_conversation_id,
            session_id=sess.session_id,
            turn_index=turn_index,
        )

    def _make_room(self, now: int) -> None:
        """Evict the oldest session when the cap is reached, and EMIT it.

        Sessions are removed on close, but a transport that never closes would
        otherwise accumulate them for the process lifetime. The eviction itself
        is what changed: this used to drop the session AND its root span
        with no marker and no test, so a workload that crossed the cap simply
        stopped producing traces. Now the whole session is finalized and its root
        carries `UNIT_EVICTED` — a bound whose enforcement is invisible is worse
        than one nobody enforces (I10).
        """
        while len(self._by_key) >= self._max_sessions:
            old_key = next(iter(self._by_key))
            old = self._by_key.pop(old_key)
            if old.session_id:
                self._by_session_id.pop(old.session_id, None)
            old.unit.note(Limitation.UNIT_EVICTED)
            self._finalize(old, None, now)

    def _session_in_scope(self) -> _Session | None:
        """The session whose unit is ambient on the task this hook is running on.

        The mechanism the whole adapter is built on, finally read on the path
        that needs it most. `claude_agent_sdk` dispatches every hook callback
        from a task spawned inside the transport's message loop, and the tee
        pinned the session unit onto that loop — so the answer is already in this
        task's scope, obtained from a real scope read, needing no framework
        identifier at all.

        The walk to the SESSION unit is what makes it usable from anywhere: a
        hook fired from inside an in-process tool handler sees the CALL unit that
        handler activated, and the session is that unit's ancestor. A chain that
        reaches no session (a CALL opened with no parent when nothing was
        resolvable) yields None and the id tiers answer instead.

        A linear scan of `_by_key` rather than a unit-keyed index, deliberately:
        `_by_key` holds one entry per live CLI subprocess, and the split this
        adapter ALREADY carries — `_by_key` and `UnitRegistry._roots` holding the
        same sessions under two independent bounds, which `_live_session` has to
        reconcile on every single lookup — is the standing proof that a second
        table of the same objects is a reconciliation bug waiting to be written.
        """
        unit = self._units.current()
        while unit is not None and unit.kind is not UnitKind.SESSION:
            unit = unit.parent
        if unit is None:
            return None
        for sess in self._by_key.values():
            if sess.unit is unit:
                return sess
        return None

    def _session_for_hook(self, payload: dict, now: int) -> _Session | None:
        """Which session does this hook belong to? CONTEXT first, the id second.

        The order is the product's whole claim (I2). What this replaces asked
        `payload["session_id"]` FIRST and never consulted the scope at all, which
        made the hook path competitor-shaped in two measurable ways: a hook fired
        from session A's reader but carrying session B's id was parented under B
        at confidence 1.0 with no marker, and a hook whose id resolved to nothing
        while more than one session was live was DISCARDED — no span, no counter,
        no limitation. The second is the exact failure `sole_live`'s docstring
        says this design exists to end, one layer above the code that says it.

        So: the scope decides, and the id is a corroborating LOOKUP KEY. When the
        two disagree the scope wins and the disagreement is counted, because a
        hook callback cannot run on a task descended from a session it does not
        belong to, while an id is whatever the CLI wrote in the payload. When the
        scope is empty the id still answers — hooks legitimately arrive before
        the pin is installed. When neither answers and exactly one session is
        live, that is a marked inference rather than a guess. When nothing at all
        answers, the observation is dropped WITH A COUNTER: with two sessions
        live and an id naming neither, there is no defensible parent to prefer,
        and inventing one would be the wrong tree this method exists to avoid.
        """
        by_context = self._session_in_scope()

        by_id: _Session | None = None
        session_id = payload.get("session_id")
        if session_id is not None:
            sess = self._by_session_id.get(session_id)
            # `is sess`, not `is not None`: `_live_session` answers for a
            # TRANSPORT key, and `_by_key` may since have been re-filed under
            # that key by a resumed run. Anything but the same object means this
            # id no longer names a session we are driving.
            if sess is not None and self._live_session(sess.key, now) is sess:
                by_id = sess

        if by_context is not None:
            if by_id is not None and by_id is not by_context:
                # COUNTED, not marked, and the reason is that there is nothing
                # here to mark. This method chooses a SESSION; it builds no span,
                # and the only span in hand is that session's own root — whose
                # edge came from a real scope read and is not what disagreed, so
                # stamping `CORRELATION_CONFLICT` on it would charge one payload's
                # disagreement to the whole run. The spans the choice feeds make
                # no correlation claim at all (see `_IN_SESSION`), precisely
                # because which session a hook belongs to is this method's guess.
                # Design §3.4 is what ends the guess: the payload becomes a
                # `UnitKey` handed to `UnitRegistry.resolve()`, which already
                # emits that member when an id and the live context land in
                # different traces. Until then the counter is the record, the
                # same way it is for the unattributable hook below.
                counters.bump("adapters.assembler.hook_session_conflict")
            return by_context
        if by_id is not None:
            return by_id
        # Exactly one live session -> attribute the hook to it. Covers hooks
        # arriving before the pin is installed or before the stream's
        # session_init line lands, and it is an inference, so it is counted.
        if len(self._by_key) == 1:
            only = next(iter(self._by_key.values()))
            sole = self._live_session(only.key, now)
            if sole is not None:
                counters.bump("adapters.assembler.hook_session_inferred_sole")
                return sole
        # Nothing answered, so the observation is dropped — but never in silence.
        # `on_hook` returns on None, which leaves no span, no limitation and, if
        # this line were missing, nothing whatsoever to distinguish a hook that
        # could not be attributed from a hook that never fired. There is no span
        # to hang a limitation on, so the counter IS the record.
        counters.bump("adapters.assembler.hook_session_unresolved")
        return None

    def _resolve_subagent_anchor(self, sess: _Session, parent_tool_use_id: str | None) -> Any:
        """Best-effort join from a stream-side parent_tool_use_id to a subagent span.

        Direct match: parent_tool_use_id happens to be a known agent_id.
        Indirect match: parent_tool_use_id is a currently open tool that itself
        belongs to a subagent (nested activity inside a subagent's tool call).

        Returns the ANCHOR (a context this session already holds), not a span id:
        the edge itself is `child_of`'s to build. Selecting which anchor is still
        this method's job, and it is still a heuristic — a `parent_tool_use_id`
        that resolves to nothing silently re-parents to the session root. What
        reaches that fallback is narrower than it was: a sub-agent evicted by
        `_max_session_entries` is recorded and then evicted, and the
        `_EvictedSubagent` breadcrumb keeps its context, so only a hook that has
        not landed yet — or a breadcrumb that itself fell out of the same bound
        — gets the root. Making the session a unit and giving
        in-process tool calls a real edge did NOT change the fallback,
        deliberately:
        rewriting this method is the ingestion move design §3.4 schedules
        separately — it stops choosing an anchor and produces a `UnitKey` for
        `UnitRegistry.resolve()`, which returns the evidence with the unit so the
        guess reports itself. Until then, no span this method feeds may claim a
        confidence for its edge — see `_IN_SESSION`.
        """
        if not parent_tool_use_id:
            return sess.unit.context
        agent_id = parent_tool_use_id
        sub = sess.subagents.get(agent_id)
        if sub is None:
            open_tool = sess.open_tools.get(parent_tool_use_id)
            if open_tool is not None and open_tool.agent_id is not None:
                agent_id = open_tool.agent_id
                sub = sess.subagents.get(agent_id)
        if sub is not None:
            return sub.draft.context
        crumb = sess.evicted_subagents.get(agent_id)
        return crumb.context if crumb is not None else sess.unit.context

    def _emit_chat(self, sess: _Session, ev: AgentStreamEvent, now: int) -> None:
        if sess.bridge is not None:
            with self._guard("adapters.assembler.emit_chat"):
                self._pend_chat(sess, ev, now)
            sess.turn_index += 1
            return
        span = None
        with self._guard("adapters.assembler.emit_chat"):
            span = self._build_chat(sess, ev, now)
        sess.turn_index += 1
        if span is not None:
            self._capture(span)

    def _chat_draft(self, sess: _Session, ev: AgentStreamEvent, now: int) -> tuple:
        """One chat span, minus its END and its TIMING markers.

        Shared by the immediate path (`_build_chat`) and the bridge's pending
        path (`_pend_chat`) so the two cannot drift: the bridge's off state
        must reproduce today's span field for field. The timing markers are
        the callers' to attach because the merge is what can change their
        truth — the immediate path attaches them on the spot, the pending path
        defers them until the merge has answered.

        Returns `(draft, gen_ai, ttft, start_ns)`: the gen_ai block rides
        along for the merge's ttft rewrite, the ttft for the marker decision,
        and the start for the join window.
        """
        p = child_of(self._resolve_subagent_anchor(sess, ev.parent_tool_use_id), _IN_SESSION)
        start_ns = sess.turn_start_ns or now

        ttft: float | None = None
        if sess.first_delta_ns:
            ttft = (sess.first_delta_ns - sess.turn_start_ns) / 1e9

        # `subject=ev.model`, not an f-string. A turn whose stream never reported
        # a model used to produce the literal span name "chat None"; the grammar
        # yields the bare operation instead, and there is no interpolation left
        # at this site to get wrong.
        draft = SpanDraft(
            p,
            intent=SpanIntent.CHAT,
            subject=ev.model,
            source=CaptureSource.ADAPTER,
            start_ns=start_ns,
        )
        gen_ai = GenAIAttributes(
            operation=SpanIntent.CHAT.operation,
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
        draft.set_gen_ai(gen_ai)
        draft.set_conversation(self._conversation(sess, turn_index=sess.turn_index))
        draft.set_status(StatusCode.OK)

        # Consumption is gated exactly the way installation is (`on_outbound`):
        # only a MAIN-THREAD assistant turn consumes the pending prompt. A
        # subagent-attributed chat's input is the subagent's task, not the
        # user's session prompt, so it ships `input_attempted=False` and leaves
        # the pending prompt for the next main-thread turn. `attempted`, not
        # `bool(payload)`: an intermediate assistant turn of one agentic loop
        # HAS no user prompt, and saying so is different from reporting an
        # empty capture as a failed one.
        consume = ev.parent_tool_use_id is None and sess.pending_prompt_source is not None
        draft.set_io(
            input_data=sess.pending_prompt if consume else b"",
            output_data=ev.content_json or b"",
            input_attempted=consume,
        )
        if consume:
            # Which channel the prompt bytes came from — the two shapes differ
            # on the wire (see `_PROMPT_STREAM`/`_PROMPT_HOOK`).
            draft.set_extra("wardex.agent.prompt_source", sess.pending_prompt_source)
            sess.pending_prompt = b""
            sess.pending_prompt_source = None
            sess.pending_prompt_hook_seen = False
        # No correlation: the anchor above may be a fallback (see
        # `_resolve_subagent_anchor`), and the parentage's own record would
        # report it as `unit_active`/1.0 with no marker — a claim this span
        # cannot back (I4). The ingestion move that turns that fallback into a
        # `UnitKey` is what earns this field.
        draft.replace_correlation(None)
        return draft, gen_ai, ttft, start_ns

    def _build_chat(self, sess: _Session, ev: AgentStreamEvent, now: int) -> Any:
        draft, _gen_ai, ttft, _start_ns = self._chat_draft(sess, ev, now)
        draft.add_limitation(_BASE_LIMITATION)
        if ttft is not None:
            draft.add_limitation(Limitation.TTFT_IPC_APPROXIMATION)
        return draft.finish(now)

    def _pend_chat(self, sess: _Session, ev: AgentStreamEvent, now: int) -> None:
        """Hold this turn's chat draft for the finalize-time merge.

        The timing markers travel DEFERRED: the merge removes the whole
        rationale for both when a `claude_code.llm_request` claims this turn
        (CLI-measured interval + ttft), and an unmerged flush applies them
        unchanged — so the bridge failing produces exactly today's span.
        """
        draft, gen_ai, ttft, start_ns = self._chat_draft(sess, ev, now)
        draft.set_end_ns(now)
        deferred = (
            (_BASE_LIMITATION, Limitation.TTFT_IPC_APPROXIMATION)
            if ttft is not None
            else (_BASE_LIMITATION,)
        )
        self._pend(
            sess,
            _PendingSpan(
                draft=draft,
                kind="chat",
                deferred_markers=deferred,
                agent_id=self._chat_agent_id(sess, ev.parent_tool_use_id),
                window=(start_ns, now),
                gen_ai=gen_ai,
            ),
        )

    def _chat_agent_id(self, sess: _Session, parent_tool_use_id: str | None) -> str | None:
        """The subagent scope a chat is anchored to — the merge join's scope key.

        Mirrors `_resolve_subagent_anchor`'s two matches; None is the main
        thread. Kept separate rather than derived from the anchor because the
        anchor silently falls back to the session root, and a fallback must
        not masquerade as a main-thread scope claim in a JOIN — an unmatched
        scope fails honestly (no merge), a wrong scope merges wrongly.
        """
        if not parent_tool_use_id:
            return None
        if self._known_subagent(sess, parent_tool_use_id):
            return parent_tool_use_id
        open_tool = sess.open_tools.get(parent_tool_use_id)
        if open_tool is not None and self._known_subagent(sess, open_tool.agent_id):
            return open_tool.agent_id
        return None

    @staticmethod
    def _known_subagent(sess: _Session, agent_id: str | None) -> bool:
        """Live, or evicted-but-remembered. One question, three call sites.

        An evicted sub-agent is still a real scope: its context is a valid
        parent and its subtree keeps its shape, so a chat turn inside it is
        still that turn's scope key and not the main thread.
        """
        return agent_id is not None and (
            agent_id in sess.subagents or agent_id in sess.evicted_subagents
        )

    def _claim_key(self, sess: _Session, tool_name: str) -> UnitKey | None:
        """This hook observation's slot in the shared key space, or None.

        None is "do not claim and do not emit": the CLI reported a bare name two
        wrapped servers both export, so no key is right. Attributing it to one of
        them fails in both directions at once — a double emit for one server and
        a LOST handler span for the other. The handler wrapper owns the call and
        its span carries `TOOL_NAME_COLLISION`.
        """
        key = self._names.key_for_hook(tool_name)
        if key is None:
            counters.bump("adapters.assembler.tool_name_unattributable")
        return key

    def _room_for(
        self, table: dict[str, _TableValue], where: str
    ) -> tuple[str, _TableValue] | None:
        """FIFO room for one more entry in a per-session table. Caller holds `_lock`.

        The assembler's half of `Unit._evict_oldest`, written to the same shape
        on purpose: one bound, one policy, two containers with different
        owners. Returns the evicted `(key, value)` so the CALLER decides what an
        evicted entry owes the wire — a span (open tools, sub-agents) or a
        breadcrumb (the two memories that outlive them).

        Returns the pair rather than taking a callback so the non-evicting path,
        which is every path until the table is full, allocates nothing: a
        closure would be built on every insert to be discarded unused. That is
        the same reason `Unit._evict_oldest` has this shape.
        """
        if len(table) < self._max_session_entries:
            return None
        oldest = next(iter(table))
        value = table.pop(oldest)
        counters.bump(f"adapters.assembler.{where}_table_full")
        return oldest, value

    def _has_room(self, table: dict[str, Any], where: str) -> bool:
        """The refuse-the-newest half of the same bound, for a table whose
        entries own NO span and whose consumers read it in ARRIVAL order.

        Returns False and COUNTS the refusal — a bound that turns something away
        in silence is the defect these two helpers exist to end, and a table
        with no span to mark can still say so in a counter.
        """
        if len(table) < self._max_session_entries:
            return True
        counters.bump(f"adapters.assembler.{where}_table_full")
        return False

    def _open_tool(self, sess: _Session, payload: dict, tool_use_id: str | None, now: int) -> None:
        if tool_use_id is None:
            return
        key = self._claim_key(sess, payload.get("tool_name") or "unknown")
        if key is None:
            return
        # Claim at the hook's rank, then ask whether anything OUTRANKS it. The
        # claim is not a gate (two concurrent calls to one tool share a name key
        # at one rank and both must be observed); the rank comparison is.
        sess.unit.claim(key, rank=HOOK_RANK)
        if outranked(sess.unit, key, HOOK_RANK):
            # An in-process handler wrapper owns this call: it wrapped the real
            # execution, so §8.4 gives it the span. Nothing is opened here, which
            # is also what keeps `_finalize` from force-closing a phantom.
            return
        evicted = self._room_for(sess.open_tools, "open_tool")
        if evicted is not None:
            # The bound names ITSELF. `CHILD_SPAN_UNCLOSED` used to ride here and
            # says a parent's teardown closed the span — a teardown that never
            # happened — sending the reader to look for a close instead of to
            # `max_session_entries`. UNSET rather than OK for the same reason:
            # this span's outcome was never observed.
            oldest_id, oldest = evicted
            self._emit_tool(
                sess,
                oldest,
                now,
                status=StatusCode.UNSET,
                markers=(Limitation.SESSION_ENTRY_TABLE_FULL,),
            )
            # Under the SAME bound, so the memory of evictions cannot outgrow
            # what it remembers for. Overflowing it is itself counted: a
            # completion arriving after that is a call this session can no
            # longer recognize, and the counter is the only record of why.
            self._room_for(sess.evicted_tools, "evicted_tool")
            sess.evicted_tools[oldest_id] = _EvictedTool(
                start_ns=oldest.start_ns, name=oldest.name, agent_id=oldest.agent_id
            )
        sess.open_tools[tool_use_id] = _OpenTool(
            tool_use_id=tool_use_id,
            name=payload.get("tool_name") or "unknown",
            start_ns=now,
            agent_id=payload.get("agent_id"),
            input_data=_safe_json_bytes(payload.get("tool_input", {})),
            from_hook=True,
            claim_key=key,
        )

    def _close_tool(
        self, sess: _Session, payload: dict, tool_use_id: str | None, now: int, failed: bool
    ) -> None:
        if tool_use_id is None:
            return
        error_type: str | None = None
        if failed:
            # The verified payload contract (claude-agent-sdk 0.2.x,
            # `PostToolUseFailureHookInput`): `error: str`, `is_interrupt:
            # NotRequired[bool]`. The interrupt flag is the low-cardinality
            # half and ships as the type; the free-text `error` message is
            # deliberately NOT shipped — `Status.message` has no bound in the
            # core limits table and free text has no PII routing decision yet,
            # so the message is a follow-up while the type is this change.
            error_type = "tool_interrupted" if payload.get("is_interrupt") else "tool_error"
        tool = sess.open_tools.pop(tool_use_id, None)
        after_evict = False
        if tool is None:
            crumb = sess.evicted_tools.get(tool_use_id)
            if crumb is not None and crumb.completed:
                # The third observation of one call: hook and stream both
                # closed it. Two spans is the documented overlap; a third would
                # pollute the aggregates the overlap rule already asks readers
                # to correct for, and what is lost is one duplicate copy of a
                # response the other half already carries.
                counters.bump("adapters.assembler.tool_completion_after_evict_duplicate")
                sess.stream_tool_meta.pop(tool_use_id, None)
                return
            key = self._claim_key(sess, payload.get("tool_name") or "unknown")
            if key is None:
                # Unattributable name: the handler owns it. Drop the stream side
                # too, or the tool would resurface through the stream-only path.
                sess.stream_tool_meta.pop(tool_use_id, None)
                return
            if crumb is not None:
                # The COMPLETION half of an eviction: same call id, same start
                # instant, same parent, and the same marker reported from the
                # other end. Without the breadcrumb this is a brand-new record
                # starting `now`, i.e. a second tool call of zero duration under
                # whatever parent the payload happened to name.
                counters.bump("adapters.assembler.tool_completion_after_evict")
                crumb.completed = True
                after_evict = True
            else:
                # No open record and no breadcrumb: either the adapter was
                # installed mid-session or the breadcrumb table itself
                # overflowed. The span still ships, with `start_ns=now` and so a
                # duration of zero, and the counter is what says why.
                counters.bump("adapters.assembler.tool_close_without_open")
            tool = _OpenTool(
                tool_use_id=tool_use_id,
                name=crumb.name if crumb is not None else (payload.get("tool_name") or "unknown"),
                start_ns=crumb.start_ns if crumb is not None else now,
                agent_id=crumb.agent_id if crumb is not None else payload.get("agent_id"),
                input_data=_safe_json_bytes(payload.get("tool_input", {})),
                # TRUE, and it was wrong before the breadcrumb existed too.
                # Reaching here means no `PreToolUse` was SEEN, not that no hook
                # delivered this call — `from_hook` records who ANNOUNCED the
                # call, and a `PostToolUse` is a hook. False put `stdio` in
                # `capture_sources`, reporting a hook-observed call as
                # reconstructed from the CLI's stdout.
                from_hook=True,
                claim_key=key,
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
        self._emit_tool(
            sess,
            tool,
            now,
            status=StatusCode.ERROR if failed else StatusCode.OK,
            # Spelled at the call site rather than carried in a local named
            # `markers`: the vocabulary census reads any argument bound to a
            # marker-ish NAME as evidence that its callee is a marker sink, and
            # `_emit_tool` would then have every argument at every one of its
            # call sites read as a marker — including the `error.type` strings.
            markers=(Limitation.SESSION_ENTRY_TABLE_FULL,) if after_evict else (),
            error_type=error_type,
        )

    def _emit_tool(
        self,
        sess: _Session,
        tool: _OpenTool,
        end_ns: int,
        status: StatusCode = StatusCode.OK,
        markers: tuple[Limitation, ...] = (),
        error_type: str | None = None,
    ) -> None:
        """Emit (or pend) one tool span.

        `status` and not the `failed: bool` this used to take. A boolean encodes
        the status and the error type at once and has no room for the third
        outcome — UNSET, which is what a bound owes a call it stopped watching
        before the result. Naming the parameter after the field it sets also
        makes a missed call site a `TypeError` instead of a silently flipped
        status.
        """
        if tool.claim_key is not None and outranked(sess.unit, tool.claim_key, HOOK_RANK):
            # Re-checked HERE and not only at open, because the handler wrapper
            # claims the key while the tool body runs — i.e. AFTER `PreToolUse`
            # opened this entry and before `PostToolUse` closes it. That is the
            # ordinary order for every in-process SDK MCP tool, so without this
            # the call ships twice: once from the layer that wrapped the
            # execution and once from the hook that only watched it.
            counters.bump("adapters.assembler.tool_claim_lost")
            return
        if sess.bridge is not None:
            with self._guard("adapters.assembler.emit_tool"):
                self._pend_tool(sess, tool, end_ns, status, markers, error_type)
            return
        span = None
        with self._guard("adapters.assembler.emit_tool"):
            span = self._build_tool(sess, tool, end_ns, status, markers, error_type)
        if span is not None:
            self._capture(span)

    def _pend_tool(
        self,
        sess: _Session,
        tool: _OpenTool,
        end_ns: int,
        status: StatusCode,
        markers: tuple[Limitation, ...],
        error_type: str | None,
    ) -> None:
        """Hold a hook/stream tool draft for the finalize-time merge.

        The SAME draft `_build_tool` would finish, timing marker included: a
        merged tool span keeps its IPC times and its marker — the merge only
        ADDS the CLI-measured duration as an extra plus the source, because a
        rewritten time under a "timing unavailable" marker would lie and
        removing the marker is the merged-LLM deliverable, not this one.
        Deferral would buy nothing here, so nothing is deferred.
        """
        draft = self._tool_draft(sess, tool, status, markers, error_type)
        draft.set_end_ns(end_ns)
        self._pend(
            sess,
            _PendingSpan(
                draft=draft,
                kind="tool",
                tool_use_id=tool.tool_use_id,
                agent_id=tool.agent_id,
            ),
        )

    def _build_tool(
        self,
        sess: _Session,
        tool: _OpenTool,
        end_ns: int,
        status: StatusCode,
        markers: tuple[Limitation, ...],
        error_type: str | None,
    ) -> Any:
        return self._tool_draft(sess, tool, status, markers, error_type).finish(end_ns)

    def _tool_draft(
        self,
        sess: _Session,
        tool: _OpenTool,
        status: StatusCode,
        markers: tuple[Limitation, ...],
        error_type: str | None,
    ) -> SpanDraft:
        anchor = sess.unit.context
        if tool.agent_id is not None:
            sub = sess.subagents.get(tool.agent_id)
            if sub is not None:
                anchor = sub.draft.context
            else:
                # The sub-agent's own span may have shipped already — its bound
                # evicted it — and a shipped span's context is still a parent.
                crumb = sess.evicted_subagents.get(tool.agent_id)
                if crumb is not None:
                    anchor = crumb.context
        p = child_of(anchor, _IN_SESSION)

        draft = SpanDraft(
            p,
            intent=SpanIntent.EXECUTE_TOOL,
            subject=tool.name,
            source=CaptureSource.ADAPTER,
            start_ns=tool.start_ns,
        )
        draft.set_tool(
            ToolAttributes(
                name=tool.name,
                call_id=tool.tool_use_id,
                # UNKNOWN, not NETWORK. wardex never observes how a Claude Code
                # built-in (Bash, Read) executes, and asserting NETWORK for it
                # was a guess that happened to be wrong — §6.2 adds UNKNOWN
                # because an honest "not observed" beats a confident lie.
                execution_type=ToolExecutionType.UNKNOWN,
            )
        )
        draft.set_status(status)
        if status is StatusCode.ERROR:
            # `finish()` refuses ERROR without a type, which turns the
            # untyped-failure defect into a mechanism. The hook path refines
            # the type from the failure payload's `is_interrupt` flag before
            # it gets here (`_close_tool`); the stream result block carries no
            # interrupt signal, so stream-only failures stay coarse-but-true.
            draft.set_error(error_type or "tool_error")
        if not tool.from_hook:
            # Reconstructed from the CLI's stdout rather than announced by a
            # hook. The DIFFERENCE is real and worth publishing -- a stream-only
            # span has no hook payload behind it -- but it is a fact about who
            # observed the call, which is what `capture_sources` means.
            draft.add_source(CaptureSource.STDIO)
        draft.set_io(input_data=tool.input_data, output_data=tool.output_data)
        draft.add_limitation(_BASE_LIMITATION)
        for marker in markers:
            draft.add_limitation(marker)
        # NO CORRELATION, and `None` rather than leaving the base edge in place.
        #
        # What used to sit here was `CorrelationInfo(confidence=1.0 if from_hook
        # else 0.7, strategy=None)`. Both halves were wrong. `strategy=None`
        # encodes as `parent_source = UNSPECIFIED` on the wire, so the span
        # published a confidence with no answer to the question confidence
        # prices -- indistinguishable from a sender that never set the field.
        # And the number was never about the parent EDGE at all: it was the
        # observation channel, hook or stream, wearing a certainty's clothes.
        # That answer belongs to `capture_sources`, which now carries it.
        #
        # `None` and not a deletion: `Parentage.correlation` is never None, so
        # dropping the override republishes the base edge at `unit_active`/1.0 --
        # on an anchor that may have come from a silent fallback, which is the
        # claim this module's header forbids by name.
        draft.replace_correlation(None)
        return draft

    def _on_stream_tool_result(self, sess: _Session, ev: AgentStreamEvent, now: int) -> None:
        # `tool_result_id`, never `parent_tool_use_id`. The two answer different
        # questions about the same line -- which CALL this result belongs to, and
        # which SUB-AGENT produced the line -- and they differ exactly when a
        # sub-agent runs a tool, which is when getting it wrong costs the most:
        # the result was filed against the `Task` call, so `execute_tool Task`
        # shipped carrying the inner tool's output and the inner call shipped no
        # result at all. One span with the wrong bytes, one span missing, and no
        # counter anywhere.
        tool_use_id = ev.tool_result_id
        if tool_use_id is None:
            return
        if tool_use_id in sess.open_tools:
            # A PreToolUse hook already opened this call; the eventual
            # PostToolUse hook is the authority that will close it.
            return
        meta = sess.stream_tool_meta.pop(tool_use_id, None)
        # BEFORE the `meta is None` return and not after it. The state this
        # whole change is about — a full session table — is exactly the state in
        # which BOTH tables are full, so a completion whose metadata was refused
        # and whose open record was evicted is the common case, not the corner.
        # Looking the breadcrumb up after the early return would lose that
        # completion entirely: no span, no marker, no counter.
        crumb = sess.evicted_tools.get(tool_use_id)
        if crumb is not None and crumb.completed:
            counters.bump("adapters.assembler.tool_completion_after_evict_duplicate")
            return
        if meta is None and crumb is None:
            # Already handled via the hook path (open_tools, stream_tool_meta
            # and the breadcrumbs are all empty for this id) -> nothing to do.
            return
        name = (meta[0] if meta is not None else "") or (
            crumb.name if crumb is not None else "unknown"
        )
        input_json = meta[1] if meta is not None else b""
        key = self._claim_key(sess, name)
        if key is None:
            # In-process tool the hook cannot attribute to one server: the
            # handler wrapper's span is authoritative, so the stream-only
            # fallback stands down too.
            return
        after_evict = False
        start_ns = sess.turn_start_ns or now
        agent_id: str | None = None
        if crumb is not None:
            # The completion half again, from the stream side. `agent_id` comes
            # off the breadcrumb rather than the `None` this path used to
            # hardcode: without it the two halves of one call hang under two
            # different parents, the evicted half under its sub-agent and the
            # completion under the session root.
            counters.bump("adapters.assembler.tool_completion_after_evict")
            crumb.completed = True
            after_evict = True
            start_ns = crumb.start_ns
            agent_id = crumb.agent_id
        tool = _OpenTool(
            tool_use_id=tool_use_id,
            name=name,
            start_ns=start_ns,
            agent_id=agent_id,
            input_data=input_json,
            from_hook=False,
            output_data=ev.content_json or b"",
            claim_key=key,
        )
        # The result block says whether the call failed. Reading it here is what
        # keeps a reconstructed span honest: the arrival of a result was being
        # taken for the success of the call, so a tool that raised shipped `ok`
        # -- and status is the first field anyone filters an agent run by.
        self._emit_tool(
            sess,
            tool,
            now,
            status=StatusCode.ERROR if ev.is_error else StatusCode.OK,
            markers=(Limitation.SESSION_ENTRY_TABLE_FULL,) if after_evict else (),
            error_type="tool_error" if ev.is_error else None,
        )

    def _emit_subagent_entry(
        self,
        sess: _Session,
        entry: _OpenSubagent,
        agent_id: str,
        now: int,
        *,
        markers: tuple[Limitation, ...] = (),
        status: StatusCode = StatusCode.OK,
    ) -> None:
        """Finish `entry`'s draft and emit (or pend) it.

        Takes the RECORD, the way `_emit_tool` does, so the eviction site — which
        already holds the record `_room_for` handed back — has something to call.
        The pop-by-id shape is `_emit_subagent` below; a site that popped first
        and then called that one would emit nothing at all.

        Markers land BEFORE the pend, so an unmerged flush still carries them.
        """
        if sess.bridge is not None:
            # The subagent draft was built at SubagentStart with its timing
            # marker already attached, and it keeps it merged or not: a
            # `subagent.spawn` cross-check adds the source, never a time
            # rewrite. Nothing to defer — held only so the merge can find it.
            with self._guard("adapters.assembler.emit_subagent"):
                entry.draft.set_status(status)
                for marker in markers:
                    entry.draft.add_limitation(marker)
                entry.draft.set_end_ns(now)
                self._pend(
                    sess,
                    _PendingSpan(draft=entry.draft, kind="subagent", agent_id=agent_id),
                )
            return
        span = None
        with self._guard("adapters.assembler.emit_subagent"):
            entry.draft.set_status(status)
            for marker in markers:
                entry.draft.add_limitation(marker)
            span = entry.draft.finish(now)
        if span is not None:
            self._capture(span)

    def _emit_subagent(self, sess: _Session, agent_id: str | None, now: int) -> None:
        """Pop by id and delegate — the `SubagentStop` / `_drain_children` shape.

        A miss is no longer a silent return. After an eviction the entry is gone
        and this stop observation is the only source of the real end instant, so
        it is counted. No completion half is built: unlike a tool, a
        `SubagentStop` carries no bytes this SDK reads — the assembler takes one
        field off that payload, the `agent_id` used to look the entry up — so
        there is nothing for a second span to hold. The evicted half already
        says `[start, evicted]` with the marker and UNSET; what is lost is the
        true end instant, and the counter is where that loss is recorded.
        """
        if agent_id is None:
            return
        entry = sess.subagents.pop(agent_id, None)
        if entry is None:
            if agent_id in sess.evicted_subagents:
                counters.bump("adapters.assembler.subagent_stop_after_evict")
            return
        self._emit_subagent_entry(sess, entry, agent_id, now)

    # --- the bridge's pending buffer and finalize-time merge ---

    def _pend(self, sess: _Session, rec: _PendingSpan) -> None:
        """File a draft for the finalize-time merge, bounded by
        `max_session_entries`: overflow emits the OLDEST immediately, unmerged,
        deferred markers applied — the cap defers merging, never deletes (I10).
        """
        while len(sess.pending) >= self._max_session_entries:
            oldest = sess.pending.pop(0)
            counters.bump("adapters.assembler.otel_bridge_pending_overflow")
            self._emit_pending(oldest)
        sess.pending.append(rec)

    def _emit_pending(self, rec: _PendingSpan) -> None:
        """Finish and emit one held draft, exactly once (`claim_emit`).

        An UNMERGED record gets its deferred markers here — the bridge never
        answered, so the IPC-approximation facts stand exactly as they would
        have on the immediate path.
        """
        if not rec.draft.claim_emit():
            return
        span = None
        with self._guard("adapters.assembler.emit_pending"):
            if not rec.merged:
                for marker in rec.deferred_markers:
                    rec.draft.add_limitation(marker)
            span = rec.draft.finish()
        if span is not None:
            self._capture(span)

    def _flush_pending(self, sess: _Session) -> None:
        """Emit everything the session still holds pending, in pend order.

        EVERY retirement path calls this — `_finalize`, `close_all_sessions`,
        and the registry-eviction retirement in `_live_session` — because the
        pending buffer is a delay, not an ownership transfer: a span that
        entered it must leave it onto the wire (span conservation)."""
        pending, sess.pending = sess.pending, []
        for rec in pending:
            self._emit_pending(rec)

    def _merge_bridge(self, sess: _Session, now: int) -> None:
        """Join the session's CLI telemetry into its pended drafts — finalize time.

        Caller gates on `sess.bridge` and wraps this in a guard: the merge is
        fail-open by construction, because the pending flush that follows
        emits today's tree unchanged whenever this method did nothing.

        Authority split (spec rule): time and skeleton are the CLI's — it
        measured inside its own process; content and semantics stay the
        stream's. Merged chat spans get the CLI interval and lose the timing
        markers (the headline deliverable); merged tool spans keep IPC times
        AND the marker, gaining the CLI-measured duration as an additive
        extra — a rewritten time under a "timing unavailable" marker would
        lie, and removing the marker there would exceed what the merge can
        honestly claim. Fail-open verdicts land on the session ROOT:
        `otel_bridge_no_data` only when the CONFIRMED injection produced
        nothing, `otel_bridge_schema_unknown` when data arrived and
        classified as nothing.
        """
        binding = sess.bridge
        if binding is None or self._bridge is None:
            return
        slot = self._bridge.take(binding.trace_id_hex, sess.session_id)
        root = sess.unit.draft
        if slot is None or not slot.spans:
            if slot is not None and slot.schema_failed:
                root.add_limitation(Limitation.OTEL_BRIDGE_SCHEMA_UNKNOWN)
            elif binding.confirmed:
                root.add_limitation(Limitation.OTEL_BRIDGE_NO_DATA)
            else:
                # Injection was never read back from the subprocess env, so
                # "the CLI sent nothing" is a claim this session cannot back
                # (I4): a counter, not a marker.
                counters.bump("adapters.assembler.otel_bridge_unconfirmed_no_data")
            return
        view = classify(slot.spans)
        #: OTel span id -> the wardex context it merged into; increments walk
        #: their CLI parent chain to the nearest entry, root otherwise.
        anchors: dict[str, Any] = {}

        # (1) interaction — cross-check only: the root's boundaries came from
        # real transport events; what the CLI adds is that it saw the same
        # session, plus its own version for the schema-drift record.
        if view.interactions:
            root.add_source(CaptureSource.OTEL_BRIDGE)
            cli_version = slot.resource.get("service.version")
            if isinstance(cli_version, str) and cli_version:
                root.set_extra(OTEL_EXTRA_PREFIX + "cli_version", cli_version)
            for span in view.interactions:
                anchors[span.span_id] = sess.unit.context

        # (2) tools (exact join) and subagents (cross-check), both keyed.
        for rec in sess.pending:
            if rec.kind == "tool" and rec.tool_use_id:
                otel_tool = view.tools.pop(rec.tool_use_id, None)
                if otel_tool is None:
                    continue
                rec.draft.add_source(CaptureSource.OTEL_BRIDGE)
                duration_ms = otel_tool.duration_ms
                if duration_ms is not None:
                    rec.draft.set_extra(OTEL_EXTRA_PREFIX + "tool_duration_ms", duration_ms)
                rec.merged = True
                for span_id in otel_tool.span_ids:
                    anchors[span_id] = rec.draft.context
            elif rec.kind == "subagent" and rec.agent_id:
                spawn = view.spawns.get(rec.agent_id)
                if spawn is None:
                    continue
                rec.draft.add_source(CaptureSource.OTEL_BRIDGE)
                rec.merged = True
                anchors[spawn.span_id] = rec.draft.context

        # (3) chats — the unique-start-window join, scoped by agent. The
        # windows are SEQUENCED per scope before matching: every chat of one
        # agentic loop shares the same host write, so their recorded turn
        # starts collide — and colliding windows made every multi-turn
        # session degenerate to the ambiguity fallback (measured against a
        # live CLI). The request that produced chat N cannot have started
        # before chat N-1's message arrived, so N-1's arrival is N's floor.
        windows = []
        floor_by_scope: dict[str | None, int] = {}
        for i, rec in enumerate(sess.pending):
            if rec.kind != "chat" or rec.window is None:
                continue
            start_ns, end_ns = rec.window
            floor = floor_by_scope.get(rec.agent_id)
            if floor is not None and floor > start_ns:
                start_ns = floor
            floor_by_scope[rec.agent_id] = end_ns
            windows.append(
                _ChatWindow(key=i, start_ns=start_ns, end_ns=end_ns, agent_id=rec.agent_id)
            )
        outcome = join_chats(windows, view.llm)
        for key, llm in outcome.pairs:
            if not (0 < llm.span.start_ns < llm.span.end_ns):
                # No real interval means no time rewrite, and marker removal
                # without one would be a lie — the record ships unmerged.
                outcome.unjoined.append(llm)
                continue
            rec = sess.pending[key]
            rec.draft.set_start_ns(llm.span.start_ns)
            rec.draft.set_end_ns(llm.span.end_ns)
            if llm.ttft_ms is not None and rec.gen_ai is not None:
                rec.draft.set_gen_ai(
                    replace(rec.gen_ai, time_to_first_chunk_s=llm.ttft_ms / 1000.0)
                )
            elif rec.gen_ai is not None and rec.gen_ai.time_to_first_chunk_s is not None:
                # The CLI did not price the first chunk, so the stream's IPC
                # approximation stays on the span — and so must its marker.
                rec.draft.add_limitation(Limitation.TTFT_IPC_APPROXIMATION)
            rec.draft.add_source(CaptureSource.OTEL_BRIDGE)
            if llm.response_id:
                # The Anthropic request id (`req_...`) — the idempotency key a
                # backend can join on. RECORDED rather than used as the join
                # key, because the stream side carries `msg_...` ids only.
                rec.draft.set_extra(OTEL_EXTRA_PREFIX + "request_id", llm.response_id)
            rec.merged = True
            anchors[llm.span.span_id] = rec.draft.context

        # (4) pure increments — CLI work wardex never had a span for.
        for step_name, span in view.increments:
            self._emit_increment(sess, span, step_name, view, anchors, conflicted=False)
        # An llm_request the join could not place ships as a SIBLING step span
        # rather than merging into anyone: an id/time fact and the tree
        # disagree and nothing can arbitrate, which is CORRELATION_CONFLICT's
        # exact sentence. Never a guessed parent.
        for llm in outcome.unjoined:
            self._emit_increment(sess, llm.span, "llm_request", view, anchors, conflicted=True)

        # (5) fail-open verdicts on the root.
        if view.recognized == 0 or slot.schema_failed:
            root.add_limitation(Limitation.OTEL_BRIDGE_SCHEMA_UNKNOWN)

    def _emit_increment(
        self,
        sess: _Session,
        span: _OtelSpan,
        step_name: str,
        view: Any,
        anchors: dict,
        conflicted: bool,
    ) -> None:
        out = None
        with self._guard("adapters.assembler.emit_increment"):
            out = self._build_increment(sess, span, step_name, view, anchors, conflicted)
        if out is not None:
            self._capture(out)

    def _build_increment(
        self,
        sess: _Session,
        span: _OtelSpan,
        step_name: str,
        view: Any,
        anchors: dict,
        conflicted: bool,
    ) -> Any:
        """An EXECUTE_STEP span for CLI-internal work wardex could not see.

        The one honest fit in the closed intent grammar: internal work, not an
        agent, not a tool — EXECUTE_TOOL would double-count against the merged
        tool span. Times are the CLI's own measurements, so NO transport-timing
        marker; `capture_sources=(otel_bridge,)` ALONE, because no adapter-side
        channel observed this work and claiming ADAPTER would put an
        observation channel on the wire that never observed (the merged-span
        pair is for merged spans). The parent is the nearest CLI ancestor that
        merged into a wardex draft — the CLI's own parent chain, not a guess —
        with the session root as the resting place.
        """
        anchor = sess.unit.context
        seen: set[str] = set()
        parent = span.parent_hex
        while parent and parent not in seen:
            seen.add(parent)
            mapped = anchors.get(parent)
            if mapped is not None:
                anchor = mapped
                break
            ancestor = view.by_span_id.get(parent)
            if ancestor is None:
                break
            parent = ancestor.parent_hex
        p = child_of(anchor, _IN_SESSION)
        draft = SpanDraft(
            p,
            intent=SpanIntent.EXECUTE_STEP,
            subject=step_name,
            source=CaptureSource.OTEL_BRIDGE,
            start_ns=span.start_ns,
        )
        draft.set_extra("wardex.step.name", step_name)
        draft.set_extra(OTEL_EXTRA_PREFIX + "span", span.name)
        for key, value in allowlisted_extras(span.attrs):
            draft.set_extra(key, value)
        draft.set_conversation(self._conversation(sess))
        if span.status_code == 2:
            # OTel STATUS_CODE_ERROR — the CLI's own verdict about its own
            # work; the low-cardinality type says whose failure it was.
            draft.set_status(StatusCode.ERROR)
            draft.set_error("cli_error")
        elif span.status_code == 1:
            draft.set_status(StatusCode.OK)
        if conflicted:
            draft.add_limitation(Limitation.CORRELATION_CONFLICT)
        # Sub-root discipline unchanged (see `_IN_SESSION`): no correlation
        # claim until the §3.4 ingestion move.
        draft.replace_correlation(None)
        return draft.finish(span.end_ns)

    def close_all_sessions(self, *, marker: Limitation) -> None:
        """Finalize every live session, then close whatever the registry still holds.

        The shutdown counterpart of `_finalize`, and it deliberately reuses that
        path rather than reaching for `UnitRegistry.close_all` directly. The
        registry can close a root; only this class can SAY what the root was.
        Closing from below emits an `invoke_agent` still carrying the open-time
        placeholder name, no conversation id, no turn totals — and drops the
        session's open tool spans and unstopped subagents on the floor, because
        those live in `_Session`, not in the unit.

        The registry sweep still runs, last and outside the lock, because it
        answers a question this loop cannot: an in-process tool call opened when
        no session could be found is a registry root that no `_Session` owns.

        Declines when this thread already holds the lock, for the reason
        `UnitRegistry.close_all` gives — this is reachable from a signal
        handler, and a handler lands wherever the interpreter happened to be.
        A weakref finalizer is the same hazard with the main-thread restriction
        removed; the guard covers both because it asks the lock, not the caller.
        """
        if self._lock._is_owned():
            counters.bump("adapters.assembler.close_all_sessions_reentrant")
            return
        now = time.time_ns()
        with self._lock:
            for key in list(self._by_key):
                sess = self._by_key.pop(key)
                if sess.session_id:
                    self._by_session_id.pop(sess.session_id, None)
                # Popped BEFORE the drain, so a straggler that arrives mid-
                # teardown cannot be handed a session this loop has already
                # finalized — it opens a fresh one or is dropped, and either
                # way it does not resurrect a root that is on its way out.
                sess.unit.note(marker)
                # NEVER a drain, and not by discipline: no wait exists on this
                # path at all — the drain lives only in the adapter's async
                # transport-close patch, so atexit/signal/uninstall keep
                # their flush budgets structurally. The merge itself is free
                # and opportunistic: whatever the CLI already exported still
                # lands on the spans it belongs to.
                if sess.bridge is not None:
                    with self._guard("adapters.assembler.otel_bridge_merge"):
                        self._merge_bridge(sess, now)
                self._drain_children(sess, now)
                self._flush_pending(sess)
                status, error_type = StatusCode.UNSET, None
                with self._guard("adapters.assembler.close_all_sessions"):
                    status, error_type = self._stamp_root(sess, None)
                self._units.close(sess.unit, status=status, error_type=error_type, end_ns=now)
        self._units.close_all(reason=marker)

    def _drain_children(self, sess: _Session, now: int) -> None:
        """Force-close and EMIT everything the session still holds open.

        Two callers, and they are the two ways a session stops being driven: the
        transport closed (`_finalize`) or the registry evicted its root out from
        under us (`_live_session`). Both must drain, and the second is why this
        is a method rather than the first half of `_finalize`: a retired session
        cannot be finalized — its root already shipped — but the tool calls it
        was still holding are ordinary observations that belong on the wire with
        a marker, not dropped on the floor (I10).
        """
        # (1) Force-close any still-open tool spans — they never got a matching
        # PostToolUse/PostToolUseFailure hook before the session ended.
        for tool_use_id in list(sess.open_tools.keys()):
            tool = sess.open_tools.pop(tool_use_id)
            self._emit_tool(
                sess,
                tool,
                now,
                status=StatusCode.ERROR,
                markers=(Limitation.CHILD_SPAN_UNCLOSED,),
                error_type="tool_unclosed",
            )

        # (2) Emit any subagent spans that never received a SubagentStop.
        for agent_id in list(sess.subagents.keys()):
            self._emit_subagent(sess, agent_id, now)

    def _finalize(self, sess: _Session, error: str | None, now: int) -> None:
        # The merge runs FIRST, against the drafts pended so far; the drain
        # below then pends whatever was still open (those flush unmerged —
        # a tool that never got its close hook is a degraded record with or
        # without the bridge), and the flush ships everything in pend order.
        # By the time this runs, the adapter's async close patch has already
        # drained (or decided not to): no waiting happens here.
        if sess.bridge is not None:
            with self._guard("adapters.assembler.otel_bridge_merge"):
                self._merge_bridge(sess, now)
        self._drain_children(sess, now)
        self._flush_pending(sess)

        # (3) Root invoke_agent span — the unit opened in `_ensure_session`,
        # whose context every span above is anchored to. Closing the UNIT rather
        # than emitting its draft is what makes the two facts one action: the
        # registry drops it from the live tables, force-closes anything still
        # attached to it, and hands the span to the sink after releasing its lock
        # (I11). A pin left on the reader task also stops being ambient here,
        # because `current()` refuses a unit that is no longer live.
        status, error_type = StatusCode.UNSET, None
        with self._guard("adapters.assembler.finalize"):
            status, error_type = self._stamp_root(sess, error)
        self._units.close(sess.unit, status=status, error_type=error_type, end_ns=now)

    def _stamp_root(self, sess: _Session, error: str | None) -> tuple[StatusCode, str | None]:
        """Fill in the session root's own fields; the registry stamps the rest.

        Returns the verdict instead of setting it, because `UnitRegistry.close()`
        is what ends a unit's span — status, `error.type` and the end instant
        together, under the lock that also detaches it.
        """
        draft = sess.unit.draft
        # `agent.name` is still the model id here, which §6.3 calls out as wrong
        # — the model belongs in `gen_ai.request.model`. Correcting it changes a
        # field a dashboard groups by, and the extraction work that moved these
        # sites onto `_assembly/` deliberately kept every span field identical,
        # so it rides the adapter rewrite with the rest of the Anthropic
        # semantics.
        draft.set_agent(AgentAttributes(name=sess.model or "agent", agent_type=AgentType.PRIMARY))
        draft.set_conversation(self._conversation(sess))

        result = sess.result
        if result is not None:
            if result.num_turns is not None:
                draft.set_extra("wardex.agent.num_turns", result.num_turns)
            if result.total_cost_usd is not None:
                draft.set_extra("wardex.agent.cost_usd", result.total_cost_usd)
            if result.duration_api_ms is not None:
                draft.set_extra("wardex.agent.api_duration_ms", result.duration_api_ms)
            status = StatusCode.ERROR if result.is_error else StatusCode.OK
            if error is not None:
                status = StatusCode.ERROR
                draft.add_limitation(Limitation.SESSION_ABORTED)
        elif error is not None:
            status = StatusCode.ERROR
            draft.add_limitation(Limitation.SESSION_ABORTED)
        else:
            status = StatusCode.UNSET
            draft.add_limitation(Limitation.SESSION_ABORTED)

        if status is not StatusCode.ERROR:
            return status, None
        # ERROR requires a type. `error` is the transport-close reason and
        # `result.is_error` is the CLI's own verdict; naming which of the two
        # ended the session is the honest low-cardinality answer.
        return status, ("session_error" if error is not None else "agent_error")
