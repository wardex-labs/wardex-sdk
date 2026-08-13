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
from typing import Any

from .. import _wardex_native
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
from .._protocol._claude_stream import AgentStreamEvent, parse_line
from .._types import (
    AgentAttributes,
    ConversationContext,
    GenAIAttributes,
    ToolAttributes,
)
from ._anthropic_names import McpToolCatalog
from ._session_state import _OpenSubagent, _OpenTool, _Session
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
        max_units: int | None = None,
        max_entries_per_unit: int | None = None,
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
        # that drives this class directly.
        self._units = (
            units
            if units is not None
            else UnitRegistry(
                sink=_ClientSink(client),
                max_units=max_units,
                max_entries_per_unit=max_entries_per_unit,
                debug=bool(getattr(getattr(client, "config", None), "debug", False)),
            )
        )
        # The shared tool-name space (design §5.4). Empty when the adapter did not
        # supply one, which is the correct reading for an assembler with no
        # in-process servers registered: every hook name then resolves to its own
        # key and the hook observer owns every call.
        self._names = names if names is not None else McpToolCatalog()

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

    def on_outbound(self, key: int, data: str) -> None:
        """Host -> CLI write. A main-thread user message installs the pending prompt.

        The `parent_tool_use_id is None` gate is SYMMETRIC with consumption:
        `_build_chat` consumes the pending prompt only for a chat whose own
        `parent_tool_use_id` is None, i.e. a main-thread assistant turn. An
        outbound user line that carries one — a host-written message addressed
        into a sub-agent's thread — is not the next main-thread turn's prompt,
        so installing it here would overwrite a prompt the main thread has not
        consumed yet and charge the loss to a turn that never died.
        """
        ev = parse_line(data.encode(), outbound=True)
        if ev is None:
            return
        now = time.time_ns()
        with self._lock:
            sess = self._ensure_session(key, now)
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
                if agent_id and len(sess.subagents) < self._max_session_entries:
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
        return None

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
        that resolves to nothing (the hook has not landed yet, or the subagent
        was never recorded because `_max_session_entries` was reached) silently
        re-parents to the session root. Making the session a unit and giving
        in-process tool calls a real edge did NOT change that, deliberately:
        rewriting this method is the ingestion move design §3.4 schedules
        separately — it stops choosing an anchor and produces a `UnitKey` for
        `UnitRegistry.resolve()`, which returns the evidence with the unit so the
        guess reports itself. Until then, no span this method feeds may claim a
        confidence for its edge — see `_IN_SESSION`.
        """
        if not parent_tool_use_id:
            return sess.unit.context
        sub = sess.subagents.get(parent_tool_use_id)
        if sub is None:
            open_tool = sess.open_tools.get(parent_tool_use_id)
            if open_tool is not None and open_tool.agent_id is not None:
                sub = sess.subagents.get(open_tool.agent_id)
        return sub.draft.context if sub is not None else sess.unit.context

    def _emit_chat(self, sess: _Session, ev: AgentStreamEvent, now: int) -> None:
        span = None
        with self._guard("adapters.assembler.emit_chat"):
            span = self._build_chat(sess, ev, now)
        sess.turn_index += 1
        if span is not None:
            self._client.capture_span(span)

    def _build_chat(self, sess: _Session, ev: AgentStreamEvent, now: int) -> Any:
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
        draft.set_gen_ai(
            GenAIAttributes(
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
        )
        draft.set_conversation(self._conversation(sess, turn_index=sess.turn_index))
        draft.set_status(StatusCode.OK)
        draft.add_limitation(_BASE_LIMITATION)
        if ttft is not None:
            draft.add_limitation(Limitation.TTFT_IPC_APPROXIMATION)

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
        return draft.finish(now)

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
        if len(sess.open_tools) >= self._max_session_entries:
            # Evict the oldest open entry (FIFO via dict insertion order) so the
            # session cannot accumulate unbounded open-tool state.
            oldest_id, oldest = next(iter(sess.open_tools.items()))
            del sess.open_tools[oldest_id]
            self._emit_tool(
                sess,
                oldest,
                now,
                # Census rename (§6.5.1): `tool_span_unclosed` folded into the
                # declared member `CHILD_SPAN_UNCLOSED`. Nothing is lost — the
                # marker rides the tool span itself, where
                # `gen_ai.operation.name=execute_tool` already says the child
                # was a tool.
                markers=(Limitation.CHILD_SPAN_UNCLOSED,),
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
        if tool is None:
            key = self._claim_key(sess, payload.get("tool_name") or "unknown")
            if key is None:
                # Unattributable name: the handler owns it. Drop the stream side
                # too, or the tool would resurface through the stream-only path.
                sess.stream_tool_meta.pop(tool_use_id, None)
                return
            tool = _OpenTool(
                tool_use_id=tool_use_id,
                name=payload.get("tool_name") or "unknown",
                start_ns=now,
                agent_id=payload.get("agent_id"),
                input_data=_safe_json_bytes(payload.get("tool_input", {})),
                from_hook=False,
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
        self._emit_tool(sess, tool, now, failed=failed, error_type=error_type)

    def _emit_tool(
        self,
        sess: _Session,
        tool: _OpenTool,
        end_ns: int,
        failed: bool = False,
        markers: tuple[Limitation, ...] = (),
        error_type: str | None = None,
    ) -> None:
        if tool.claim_key is not None and outranked(sess.unit, tool.claim_key, HOOK_RANK):
            # Re-checked HERE and not only at open, because the handler wrapper
            # claims the key while the tool body runs — i.e. AFTER `PreToolUse`
            # opened this entry and before `PostToolUse` closes it. That is the
            # ordinary order for every in-process SDK MCP tool, so without this
            # the call ships twice: once from the layer that wrapped the
            # execution and once from the hook that only watched it.
            counters.bump("adapters.assembler.tool_claim_lost")
            return
        span = None
        with self._guard("adapters.assembler.emit_tool"):
            span = self._build_tool(sess, tool, end_ns, failed, markers, error_type)
        if span is not None:
            self._client.capture_span(span)

    def _build_tool(
        self,
        sess: _Session,
        tool: _OpenTool,
        end_ns: int,
        failed: bool,
        markers: tuple[Limitation, ...],
        error_type: str | None,
    ) -> Any:
        anchor = sess.unit.context
        if tool.agent_id is not None:
            sub = sess.subagents.get(tool.agent_id)
            if sub is not None:
                anchor = sub.draft.context
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
        draft.set_status(StatusCode.ERROR if failed else StatusCode.OK)
        if failed:
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
        return draft.finish(end_ns)

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
        if meta is None:
            # Already handled via the hook path (both open_tools and
            # stream_tool_meta are empty for this id) -> nothing to do.
            return
        name, input_json = meta
        key = self._claim_key(sess, name)
        if key is None:
            # In-process tool the hook cannot attribute to one server: the
            # handler wrapper's span is authoritative, so the stream-only
            # fallback stands down too.
            return
        tool = _OpenTool(
            tool_use_id=tool_use_id,
            name=name,
            start_ns=sess.turn_start_ns or now,
            agent_id=None,
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
            failed=ev.is_error,
            error_type="tool_error" if ev.is_error else None,
        )

    def _emit_subagent(self, sess: _Session, agent_id: str | None, now: int) -> None:
        if agent_id is None:
            return
        entry = sess.subagents.pop(agent_id, None)
        if entry is None:
            return
        span = None
        with self._guard("adapters.assembler.emit_subagent"):
            entry.draft.set_status(StatusCode.OK)
            span = entry.draft.finish(now)
        if span is not None:
            self._client.capture_span(span)

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
                self._drain_children(sess, now)
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
                failed=True,
                markers=(Limitation.CHILD_SPAN_UNCLOSED,),
                error_type="tool_unclosed",
            )

        # (2) Emit any subagent spans that never received a SubagentStop.
        for agent_id in list(sess.subagents.keys()):
            self._emit_subagent(sess, agent_id, now)

    def _finalize(self, sess: _Session, error: str | None, now: int) -> None:
        self._drain_children(sess, now)

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
