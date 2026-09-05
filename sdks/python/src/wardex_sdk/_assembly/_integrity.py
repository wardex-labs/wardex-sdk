"""Capture-integrity vocabulary — design §6.5 tier 3, §6.5.1, §I9.

A limitation marker says what wardex *failed* to capture, or captured only by
interpretation. It is the third and last tier of the "concept has no home in
the vocabulary" disposition table: tier 1 is a namespaced ``extra`` key, tier 2
is an ``InternalSpanEvent``, tier 3 is a marker here. Anything that would need
a new span kind is dropped.

The enum is CLOSED, and as of the 2026-07-29 census (design §6.5.1) it is also
**complete**: it is the full vocabulary, not a sample of it. Every limitation
string the SDK can attach to a span today — Python and Rust — has a member
below. Adding a member is a core change on purpose: the dashboard renders these
and they are a proto enum (``wardex.v1.Limitation``) whose value names lock the
moment they are declared (``buf`` ``ENUM_VALUE_SAME_NAME``), so a late
correction costs a second deliberate schema break. Emitters must never invent a
marker string inline.

**The census closed at 37 members = 15 originally declared + 21 from it + 1
from §5.4.** Six more landed since, each by its own deliberate core PR:
``INSTRUMENTATION_DEGRADED`` (38, wardex's own failure),
``OTLP_ATTRIBUTE_TRUNCATED`` (39, the OTLP export size guard),
``UNIT_TABLE_FULL`` (40, the registry's breadth bound), the Agent SDK
OTel bridge's fail-open pair ``OTEL_BRIDGE_NO_DATA`` (41, confirmed injection
and nothing arrived) / ``OTEL_BRIDGE_SCHEMA_UNKNOWN`` (42, data arrived and
classified as nothing), and ``SESSION_ENTRY_TABLE_FULL`` (43, the adapter's
per-session bound, which is the registry's breadth bound one layer out and a
different FIELD). Five more since that paragraph was last true:
``EXTRA_KEYS_DROPPED`` (44, the dynamic-key bound on the provider-usage
mirror), ``TRACKING_RESET_AT_FORK`` (45, the fork child's per-connection
tracking reset — the one member that names a process event and no knob), and
the deferred-parse pair ``PARSE_BACKLOG_FULL`` (46, the backlog's capacity
bound — a transaction shipped unparsed to admit a newer one) /
``PARSE_SKIPPED_AT_SHUTDOWN`` (47, the shutdown budget ran out first — same
unparsed shipment, different knob, so a different member by the census rule),
and ``WS_LLM_SEMANTICS_UNREAD`` (48, LLM calls crossed a WebSocket connection
wardex only counted — the transport, not a parser, is the gap).
``tests/test_limitation_census.py`` is the live count; this paragraph is its
history, not its source.
The census read every assignment and append site that reaches
``CaptureIntegrity.limitations`` and found 25 distinct strings (24 Python, 1
Rust). Four of those merged away — see the ``NOTE (census)`` comments on
``CONNECT_TIMING_UNAVAILABLE``, ``PAYLOAD_COMPRESSED``, ``FRAME_PARSE_FAILED``
and ``CHILD_SPAN_UNCLOSED`` — leaving 21 new members.

**Two vocabularies, not one — do not merge them.** A ``disabled_reason`` is why
a Rust parser latched *itself* off for a whole connection; no span exists to
carry it (nothing was ever parsed), so it stays a ``&'static str`` that
``init(debug=True)`` prints once to stderr. ``headers_exceeded``, ``not_http``
and ``chunk_size_exceeded`` are those, and they are deliberately absent below.
The trap is ``stream_buffer_exceeded``, which is **both**: the HTTP/1 latch
(``crates/wardex-protocol/src/http1.rs``) is a debug reason and stays out,
while the MCP-stdio JSON-RPC instance (``json_rpc.rs``) happens with a pending
request already open, so a span exists and design §4.6 site-3 makes it
reportable — that is why ``STREAM_BUFFER_EXCEEDED`` is a member. The two
vocabularies must stay in separate proto enums; they have different lifetimes
and different consumers (wire contract vs stderr).

**Read this before adding a member.** ``tests/test_limitation_census.py``
re-runs the census against the source on every test run and fails if any marker
string, in either language, has no member here. That test is the mechanism;
this docstring is only its description. If it just failed on you, the question
to answer is not "how do I make it pass" but "is my new string a wire-contract
marker (add a member) or a connection-level debug reason (add it to the
exclusion set instead)".

This module is the **python half** of the census, and completing it was the
hard prerequisite for routing all six span-emit sites through
``SpanDraft.finish()`` — ``finish()`` raises ``VocabularyError`` on a marker
that is not a member here, which the emit site's ``guard()`` swallows. Against
an incomplete enum that routing would have silently deleted every gRPC span,
every streaming chat span, every WS span and every adapter span. It is done:
**every Python emitter now names a member**, the seven pre-rename free strings
are gone from the tree, and ``tests/test_limitation_census.py`` asserts an
EMPTY string census as the standing rule. One producer is still textual —
``body_cap_exceeded``, built in ``crates/wardex-protocol`` and resolved once at
the PyO3 boundary by ``Limitation.from_wire``. Retyping the Rust side is the
remaining work, and it is what would make proto the single source of truth for
every language SDK (§6.6).
"""

from __future__ import annotations

from enum import Enum


class Limitation(Enum):
    """CLOSED and complete. The only legal values of a span's limitation markers.

    Values are the wire contract (``CaptureIntegrity.limitations``); they are
    lower_snake to match every other wardex enum and must not be renamed
    casually. Four values below were renamed by the census on the way in — that
    rename is itself a wire change, and it rode the same deliberate
    ``wardex.v1`` break that put this enum on the wire (§6.7). There is no
    second chance.

    Each member records the condition that emits it and the site that does so,
    as module-qualified functions rather than line numbers. The site named is
    the function that ATTACHES the marker, which on the byte seam and in the
    Agent SDK assembler is the ``_build_*`` half of a ``_build_*``/``_emit_*``
    pair — the ``_emit_*`` wrapper only guards the build and hands the finished
    span to the sink, so it holds no marker to grep for. A member marked
    "declared; no emitter" is vocabulary the design named and a later seam
    wires up — it is not dead code, and it is not evidence that the census
    missed something.
    """

    @classmethod
    def from_wire(cls, value: str) -> Limitation | None:
        """A marker that arrived as a STRING, resolved to its member, or None.

        Exactly one caller is legitimate and it is the PyO3 boundary: the Rust
        protocol parsers still build `Vec<&'static str>` (retyping them means
        touching `bindings/python/src/lib.rs` and the PII walk that runs
        regexes over the strings too), so a marker produced in Rust reaches Python
        as text and has to be resolved once, at the seam that folds it into a
        span. Python-side emitters name the member directly and must not come
        here — a string-to-member lookup used as a general entry point is the
        drift this enum exists to end.

        Returns `None` rather than raising for an unrecognized string, because
        the alternative on that path is deleting the span (`finish()` rejects a
        marker it cannot type) over a marker wardex itself produced. That is
        not a silent hole: `tests/test_limitation_census.py` scans `crates/`
        and `bindings/` on every run and fails the build if the Rust side
        starts emitting a string with no member here, so an unknown value is
        unreachable rather than merely tolerated.

        A dict lookup rather than `cls(value)` in a `try`: `_assembly/` is held
        to zero silent swallows (C-S4), and an `except ValueError: return None`
        here would be one — indistinguishable in a diff from a swallow that is
        hiding something.
        """
        return _BY_VALUE.get(value)

    # ------------------------------------------------------------------
    # Parentage: the edge exists but was not derived from a live scope
    # I4 — a guess reports itself. Each of these travels with a
    # CorrelationInfo whose confidence is below 1.0.
    # ------------------------------------------------------------------

    PARENT_UNRESOLVED = "parent_unresolved"
    """A parent was expected and no scope, unit or header supplied one.

Two emit sites, and the first is the mechanism the second restates.
    ``_assembly/_parentage.py``'s ``_MARKER`` table attaches it to every edge
    ``resolve_parentage`` builds for ``ParentSource.UNRESOLVED``, so no caller
    can name that source and omit the marker — I4 as a mechanism rather than as
    caller discipline. The second names the member at the slot where that
    source is CHOSEN, which is where a reader greps for it, and both are
    idempotent: ``Parentage.with_limitation`` and ``IntegrityBuilder.limitation``
    each no-op on a marker already present.

    * ``_assembly/_units.py::UnitRegistry._edge`` — ``resolve()``'s last tier,
      via ``Parentage.with_limitation``. No alias resolved to a unit, no live
      scope reached us and no single session was live, so the span roots its own
      trace and says the parent it expected was never found.
    The in-process tool span reaches it through that same table now, because
    ``_adapters/_context.py`` declares the site's placement and the registry
    decides. It used to be a third site here, stamping this by hand at the end
    of a tier ladder the adapter walked itself.

    The path that puts it on the most records is neither:
    ``__init__.py``'s ``capture_state_snapshot`` passes
    ``ParentSource.UNRESOLVED`` for a snapshot taken outside any span and names
    no member itself, leaving the ``_MARKER`` table to do it.

    A byte-seam span reaches it the same way, through
    ``resolve_observed``'s ``_ORPHANED_BY_WARDEX`` evidence, in three cases: the
    latched parent's unit had already closed, the request was issued inside a
    span wardex failed to open, and the latched parent was discarded by the h2
    stream latch's own cap. The last two arrive carrying
    ``INSTRUMENTATION_DEGRADED`` as well, which is what separates "wardex lost
    the parent" from "the parent was never there".
    """

    UNIT_INFERRED_SOLE = "unit_inferred_sole"
    """Exactly one logical unit was live, so it was taken as the parent.

    The same shape as ``PARENT_UNRESOLVED`` above: the ``_MARKER`` table in
    ``_assembly/_parentage.py`` attaches it to every edge built for
    ``ParentSource.UNIT_SOLE``, and ``_assembly/_units.py::UnitRegistry._edge``
    names it again at ``resolve()``'s sole-live tier — no alias and no live
    scope, but exactly one SESSION unit open, so it is taken as the parent.

    An ADAPTER reaches this by declaring ``Fallback.SOLE_LIVE_RUN`` at a site it
    expects to be reached through a carrier the framework may not have
    propagated to, and that declaration is the whole of what it may say: the
    candidate comes from the registry's own table, filtered to that adapter's
    own runs, and only when there is exactly one. What it replaces dropped an
    unattributable call outright, with no marker; a marked 0.5 edge beats
    unmarked data loss, and beats a call that becomes its own trace root.
    """

    CORRELATION_CONFLICT = "correlation_conflict"
    """A framework-id alias and the live context disagreed about the trace.

    Design §5.5. Five emit sites, four of them in ``_assembly/_units.py`` and all
    of them the same shape: two answers to one parent question, with the
    disagreement put on the wire rather than into a counter.

    * ``UnitRegistry._edge`` — ``resolve()``'s precedence ladder — attaches it
      via ``Parentage.with_limitation`` when an alias resolves into a DIFFERENT
      trace than the live context. The context wins and the disagreement ships.
    * ``UnitRegistry.pin_driver`` attaches it to the unit's own span when a pin
      is declared for a task other than the one calling — a self-consistent lie
      that no confidence value downstream could reveal.
    * ``UnitRegistry.open`` and ``UnitRegistry.resolve`` attach it when they
      REFUSE the scope a CLOSED pin left standing: the opened unit becomes a
      trace root, or the edge is rebuilt from the remaining tiers, instead of
      hanging later work off a finished unit at confidence 1.0 — unless the
      unit died by the registry's OWN eviction, in which case the refusal
      carries ``INSTRUMENTATION_DEGRADED`` instead
      (``UnitRegistry.refused_ambient_marker``): the two strands have
      different repairs, and this member's is pin and lifetime discipline,
      not a capacity knob.
    * ``_adapters/_context.py`` — the one emit path outside the registry —
      marks the span whose edge a declared fallback decided while a dead pin
      was standing. Taking a parent unit is precisely what stops ``open()``
      from seeing the poisoned ambient for itself, so without this the two
      reasons a guess happened are byte-identical: "nothing was pinned" and
      "what was pinned had died", the second being a call filed inside a run
      it has nothing to do with. The pinned session's own span is not
      reachable either way — it was materialized and shipped inside the very
      ``close()`` that made the pin stale, and ``Unit.note()`` on a closed
      unit is a no-op. The site spells no member of its own: it asks
      ``refused_ambient_marker`` for the word, so an evict-origin strand says
      ``INSTRUMENTATION_DEGRADED`` there exactly as it would had ``open()``
      seen the corpse itself.
    """

    # ------------------------------------------------------------------
    # Context propagation: the mechanism that produces parentage is weakened
    # or absent on this runtime/seam. Declared degradation, not a workaround
    # (design §5.3-(v-b), §9).
    # ------------------------------------------------------------------

    CONTEXT_PROPAGATION_DEGRADED = "context_propagation_degraded"
    """The seam propagates context, but not with full fidelity — a thread-pool
    tool handler that copies the context at submit time, say.

    Declared; no emitter yet. No seam in the SDK can currently tell a
    partially-copied context from a fully propagated one.
    """

    CONTEXT_PROPAGATION_UNAVAILABLE = "context_propagation_unavailable"
    """The seam cannot propagate context at all on this runtime, so every span
    below it is a root.

    Declared; no emitter yet. Every runtime the SDK supports today propagates
    a ContextVar across the seams it patches.
    """

    # ------------------------------------------------------------------
    # Unit lifecycle: a logical unit ended for a reason other than the
    # framework finishing it. I10 — eviction emits, it never drops silently.
    # ------------------------------------------------------------------

    UNIT_EVICTED = "unit_evicted"
    """A unit hit a capacity bound and was evicted before the framework closed
    it — ``max_units`` in the registry, ``max_sessions`` in the Anthropic
    adapter's own session table.

    Points at a UNIT bound. Deliberately NOT merged with ``CONNECTION_EVICTED``,
    which points at ``max_connections``: merging them would send a user to turn
    the wrong knob.

    Three emit sites. ``_assembly/_units.py::UnitRegistry._evict_root_locked``
    CLOSES the oldest root at ``max_units``, so its span is emitted carrying
    this marker, and ``_adapters/_assembler.py::SessionAssembler._make_room``
    does the same for ``max_sessions``. The third,
    ``_adapters/_assembler.py::SessionAssembler._resume``, is not an eviction: it
    puts the marker on the NEW root that continues a run whose predecessor was
    evicted, which is what turns "a second root appeared from nowhere" into
    "this run was truncated and resumes here".

    What both of the evictions replace dropped the unit and its root span
    outright, with no marker and no test, so a workload that crossed a cap
    simply stopped producing traces (I10).
    """

    UNIT_INTERRUPTED = "unit_interrupted"
    """The unit was torn down by cancellation or interpreter shutdown rather
    than by a normal end-of-run.

    Emitted by ``_runtime.py``'s signal handler, and by that one alone. It is
    reached only on the disposition where the app left the signal at its
    default: there the handler ends the process itself, so ``atexit`` never
    runs and the ordinary teardown never gets its turn. Under any other
    disposition the program either exits through the interpreter — where atexit
    reaches the adapter's ``uninstall`` and the span carries
    ``ADAPTER_UNINSTALLED`` instead — or carries on running, and closing a
    session that is still being driven would be the lie this marker exists to
    avoid telling.

    Reads as the more honest of the pair: it says a shutdown cut the run off,
    which is what a user wants to know, where its sibling says only that wardex
    stopped watching.
    """

    CHILD_SPAN_UNCLOSED = "child_span_unclosed"
    """A child span was force-closed by its parent's teardown or by an eviction
    instead of by its own completion event, so its ``end_time_ns`` is the
    teardown instant and its status is synthesized.

    Three emit sites across two modules, all of them the same rule: a bound or a
    teardown CLOSES what it stops tracking, it never drops it.

    In ``_assembly/_units.py`` — ``UnitRegistry._close_locked``, for both the
    surviving children and the still-open drafts of a unit being closed; a
    per-unit table crossing ``max_entries_per_unit`` used to land here too and
    now carries ``UNIT_TABLE_FULL``, because a bound and a teardown demand
    different next actions from the reader. The one
    thing evicted WITHOUT a marker is an arbitration loser, which is discarded
    exactly as ``close_span`` would discard it: a bound is a reason to stop
    tracking a draft, never a reason to promote one ``claim()`` already rejected.

    In ``_adapters/_assembler.py`` — ``::_drain_children``, when a session stops
    being driven with tools still open, which happens either because the
    transport closed or because the registry evicted the session's root out from
    under the assembler. It goes through ``_emit_tool(markers=...)``. Before the
    census rewired the assembler's sites it was the free string
    ``"tool_span_unclosed"``.

    ``SessionAssembler._open_tool``'s eviction was a second assembler site and
    is NOT one any more: a full ``open_tools`` table is a BOUND, and it names
    ``max_session_entries`` through ``SESSION_ENTRY_TABLE_FULL`` (43). This
    member kept it only until that site was decided on its own evidence, which
    is what the paragraph in ``UNIT_TABLE_FULL`` used to ask for.

    NOTE (census): the four-way merge that loses nothing. ``tool_span_unclosed``
    folded into this already-declared member because the marker rides the tool span
    itself, where ``gen_ai.operation.name=execute_tool`` already says the child
    was a tool. This is the single point where the two drifts overlapped —
    vocabulary without an emitter meeting an emitter without vocabulary — and
    without the census it would have frozen into the wire as two names for one
    fact.
    """

    UNIT_TABLE_FULL = "unit_table_full"
    """A per-unit table crossed ``max_entries_per_unit`` and its OLDEST entry
    was force-closed and emitted to admit the new one.

    Points at that one knob, and naming it is the whole reason this member
    exists (§6.5.1: if the user's next action differs, they are separate
    members). ``UNIT_EVICTED`` names ``max_units`` / ``max_sessions`` — whole
    roots crossing the unit-count cap. ``CHILD_SPAN_UNCLOSED`` names no knob
    at all: it says someone else's TEARDOWN closed the span. Before this
    member the breadth bound borrowed ``CHILD_SPAN_UNCLOSED``, so the
    canonical wide fan-out (300 simultaneously-live Send workers over a
    256-entry table) put a teardown marker on 44 healthy spans and sent the
    reader hunting for a close that never happened instead of to the knob
    that did it.

    Two emit sites, both in ``_assembly/_units.py`` and both the registry's
    own breadth bound: ``UnitRegistry.open`` closes the oldest CHILD unit of
    a full ``_children`` table, and ``Unit.open_span`` force-closes the
    oldest OPEN DRAFT of a full ``_open`` table (except an arbitration loser,
    which stays discarded — a bound is never a reason to promote a draft
    ``claim()`` already rejected). Only the entry that HIT the bound carries
    this marker: its descendants, closed by the same walk, keep
    ``CHILD_SPAN_UNCLOSED``, because they were closed by their parent's
    teardown — which is that member's exact sentence — and the table-full
    fact is not theirs to report.

    Deliberately NOT widened to ``max_session_entries``: that bound has its own
    member, ``SESSION_ENTRY_TABLE_FULL`` (43), decided on its own evidence as
    this paragraph used to ask for. ``crates/wardex-limits`` calls this knob a
    generalization of that one — same order, same semantics — and the decision
    turned on the census's operative test rather than on semantics: they are
    separate FIELDS, so a reader sent to the wrong one raises a number that
    changes nothing about the marker they are looking at. If the two fields are
    ever unified, 43 becomes an alias of this member.
    """

    SESSION_ENTRY_TABLE_FULL = "session_entry_table_full"
    """A per-SESSION table crossed ``max_session_entries`` and its OLDEST entry
    was force-closed and emitted to admit the new one.

    Points at that one knob, and naming it is the whole reason this member
    exists (§6.5.1: if the user's next action differs, they are separate
    members). ``UNIT_TABLE_FULL`` names ``max_entries_per_unit`` — the unit
    REGISTRY's per-unit tables — and although ``crates/wardex-limits`` calls
    that knob a generalization of this one, they are separate FIELDS: raising
    one leaves the other at its default, so a reader sent to the wrong one
    turns a knob that changes nothing. ``UNIT_EVICTED`` names the two COUNT
    caps (``max_units`` / ``max_sessions``); ``CHILD_SPAN_UNCLOSED`` names no
    knob at all — it says someone else's TEARDOWN closed the span, which is
    what these sites used to claim about a teardown that never happened.

    Three emit sites, all in ``_adapters/_assembler.py`` and all one bound.
    ``SessionAssembler._open_tool`` force-closes the oldest OPEN TOOL of a full
    ``open_tools`` table; the ``SubagentStart`` branch of ``::on_hook`` does the
    same for the oldest OPEN SUB-AGENT of a full ``subagents`` table (which
    used to be dropped with no span at all, and whose span CONTEXT is kept in
    an ``_EvictedSubagent`` breadcrumb so its children keep their parent); and
    ``::_close_tool`` / ``::_on_stream_tool_result`` put it on the COMPLETION
    half — the span built when a tool's own ``PostToolUse`` or stream
    ``tool_result`` arrives after wardex had already evicted its open record.
    That is the same bound reported from the other end, and the ``_EvictedTool``
    breadcrumb is what lets the completion say so instead of shipping as a
    second, zero-duration tool call under the wrong parent.

    ONE CALL, TWO OBSERVATIONS. An evicted call that later completes ships two
    spans with the same ``call_id``, both carrying this marker, and they
    OVERLAP: the ``StatusCode.UNSET`` one is ``[start, evicted]`` — the window
    wardex actually watched, holding the input bytes — and the other is
    ``[start, end]``, the whole call, holding the output bytes. A latency
    aggregate must exclude the spans that carry this marker AND ``UNSET`` or it
    counts the call twice.

    An evicted entry ships ``StatusCode.UNSET``: wardex stopped watching before
    the outcome, so OK would claim a success it never observed and ERROR would
    blame the agent for wardex's own full table.

    The bounded tables that hold NO span keep counters instead — a refused
    ``stream_tool_meta`` entry has no span to mark, exactly as an evicted alias
    does not (``adapters.assembler.stream_tool_meta_table_full``).
    """

    # ------------------------------------------------------------------
    # Adapter lifecycle (I7)
    # ------------------------------------------------------------------

    ADAPTER_UNINSTALLED = "adapter_uninstalled"
    """The span was closed by ``uninstall()`` rather than by the framework.

    Emitted from ``_adapters/_anthropic_agent_sdk.py::uninstall``, which is also
    the path an ordinary interpreter exit takes: ``atexit`` tears the adapter
    down, so this — not ``UNIT_INTERRUPTED`` — is what a Ctrl-C ultimately puts
    on the span. The two are not interchangeable. This one is mechanical and
    always true of an uninstall; its sibling additionally claims the process was
    cut off, which is only knowable in the signal handler.

    Deliberately NOT merged with ``WS_NO_CLOSE`` even though both of today's
    ``ws_no_close`` sites sit inside ``uninstall()``: the fact a user reads off
    ``WS_NO_CLOSE`` is capture completeness (no CLOSE frame, so close code and
    duration are untrustworthy), not lifecycle. The two are correct *together*.
    """

    PATCH_SUPERSEDED = "patch_superseded"
    """Something else re-patched a symbol wardex had already patched, so the
    interception wardex installed is no longer the one in effect.

    Emitted from ``_assembly/_patchset.py::PatchSet._restore``, when the attribute
    being restored no longer holds the exact wrapper this ``PatchSet`` installed
    — another library patched over wardex, or removed wardex's patch outright.
    That patch is then LEFT IN PLACE, which is the point: restoring over it would
    delete the other library's interception from a component that has just
    announced it is gone.

    Detectable only at restore time, when the component has stopped producing
    spans — so the live signal is the counter ``PatchSet`` bumps
    (``<owner>.patch_superseded``), and ``PatchSet.limitations()`` offers the
    member to any caller that does hold a span to hang it on.
    """

    INSTRUMENTATION_DEGRADED = "instrumentation_degraded"
    """wardex's own instrumentation failed at this site, so something that
    belongs on this span — or the whole span below it — is missing.

    Every other member of this vocabulary describes a limit of what could be
    OBSERVED: the framework did not say, the protocol does not carry it, a bound
    was reached. This one describes a limit of wardex. It exists because the two
    are indistinguishable downstream without it, and the wrong one gets blamed:
    a subtree missing because an adapter threw looks exactly like a subtree that
    never ran.

    Emitted from ``_adapters/_context.py``: on a unit whose open or description
    failed, on one whose activation failed, and on the ENCLOSING unit when the
    span itself will not ship. BEST EFFORT by contract, and the contract is what
    matters here — the span that would carry it is sometimes the very one that
    could not be built, so it lands on the enclosing unit instead, and where
    there is no enclosing unit it cannot land at all. A registry fault wide
    enough to reach the enclosing unit takes the marker with it. What always
    survives is the line the same failure prints to stderr, which touches
    nothing that can itself be broken.

    And from ``_assembly/_parentage.py::resolve_observed``, which is the same
    sentence said about an EDGE rather than about a unit: a byte-seam span whose
    parent wardex owed it and does not have. Two conditions reach that branch —
    a request issued inside a ``degraded_run`` (a span wardex failed to open, so
    nothing was ambient to latch) and one whose latched parent wardex itself
    discarded to stay inside a bound (``_interceptors/_trackers.py``, the h2
    stream latch at ``max_streams``). It travels with ``PARENT_UNRESOLVED``,
    which ``_MARKER`` attaches from the ``UNRESOLVED`` source; this one is what
    stops the pair reading as "the host has an untraced caller".

    And from ``_assembly/_units.py``, for the refusal of an EVICT-ORIGIN
    leftover scope: ``open()`` and ``resolve()`` refuse the standing fork of a
    unit the registry itself evicted (``Unit._evicted``), and the refused span
    carries this member rather than ``CORRELATION_CONFLICT`` — the strand is
    wardex's own bound at work, so the repair is ``max_units``, not the
    adapter's pinning or lifetime. The word is chosen in
    ``UnitRegistry.refused_ambient_marker``, which the adapter surface's
    declared fallback asks too, so the strand reads the same wherever the
    refusal happens. The same sentence ``resolve_observed`` already says for a
    byte-seam span whose latched parent wardex discarded to stay inside a
    bound.

    And from ``_interceptors/_seam.py``, on the deferred finalization's
    parse exception: ``_assemble`` runs ``parse_llm_semantics`` under the
    ``interceptors.seam.parse`` guard, and a parser that RAISES (as opposed
    to answering None — "not an LLM body") marks the span with this member
    and ships it. Before the deferred split that raise was swallowed
    uncounted, and under AGENT mode with no parent the span vanished
    entirely — a zero counter over deleted data, the exact I6 shape.

    And from ``_client.py``, on the deferred-parse path's spawn failure:
    ``capture_deferred`` could not bring the finalize worker up (a host at
    its thread ulimit is the measured shape), so the job is finalized
    inline, parse-less, carrying this member — wardex's own failure, and
    the span still ships rather than silently losing captured bodies.

    Deliberately not ``CONTEXT_PROPAGATION_DEGRADED``, which is declared as a
    property of the RUNTIME — work whose carrier legitimately could not inherit
    the context. Reusing it here would file a wardex bug under "the host's
    threading model", which is the attribution this member exists to correct.
    """

    TRACKING_RESET_AT_FORK = "tracking_reset_at_fork"
    """The connection this span rode crossed an ``os.fork()``: the child reset
    its per-connection tracking state, so parsing may have started mid-stream
    and the timing/stream fields' origin is younger than the connection.

    Emitted from ``_interceptors/_seam.py``. At fork the child's reset latches
    the ids of every connection the seam was tracking and clears the table
    (an inherited entry would otherwise hand a recycled ``id()`` a dead
    connection's tracker and its latched gate — the cross-process edition of
    the close-hook bug). When the host keeps using an INHERITED socket, the
    seam builds it a fresh state, finds its id in the latch, and the FIRST
    span assembled on it carries this marker, exactly once — the honest label
    for a body parsed from its middle and a ``connection_reused``/TTFB story
    whose epoch is the fork, not the connect. A brand-new connection opened in
    the child matches nothing and carries nothing.

    Not ``CONNECTION_EVICTED`` (24), by the census rule the caps band states:
    the reader's next action differs, because the KNOB differs. That member
    names ``limits.max_connections`` — raise it and the marker goes away —
    while this one names no knob at all: it reports a process event, and no
    limits field can prevent a host from forking. Merging them would send the
    reader to turn a knob that cannot help.

    Best-effort by one bounded false positive: the latch keys on ``id()``, so
    an inherited socket that is garbage-collected before the child touches it
    can donate its id to a NEW connection, which then wears the marker on one
    span. That is the pre-existing id-reuse class, bounded to one span per
    latched id, and accepted — the alternative (holding the socket objects)
    would pin the parent's file descriptors open for the child's lifetime.

    The DISCARDED side of the fork reset — the parent-owned buffer, sessions
    and units the child drops without emitting — carries no marker anywhere,
    deliberately: nothing was lost, it ships from the process that owns it.
    The fork event itself is a process fact, not a span fact, and lives in the
    ``_runtime.fork_child_reinit`` diagnostic counter (the ``disabled_reason``
    disposition).
    """

    # ------------------------------------------------------------------
    # Observation completeness (I8, §8.1 rule W)
    # ------------------------------------------------------------------

    NO_WIRE_EVIDENCE = "no_wire_evidence"
    """The span was assembled entirely from framework callbacks; no interceptor
    ever saw the bytes, so payloads are the framework's account of them.

    Declared; no emitter until the ownership table of §8.2.
    """

    POSSIBLE_DUPLICATE_CHAT = "possible_duplicate_chat"
    """An adapter emitted a ``chat`` span that an installed interceptor may have
    emitted as well. Dual-instrumentation policy, §8.1.

    Declared; no emitter until the ownership table of §8.2.
    """

    STREAM_BUFFER_EXCEEDED = "stream_buffer_exceeded"
    """A stream parser's reassembly buffer hit ``max_stream_buffer_bytes`` and
    the parser latched off mid-stream.

    Declared; no emitter yet. Reportable only on the MCP-stdio JSON-RPC path
    (``crates/wardex-protocol/src/json_rpc.rs``), where a pending request means
    a span already exists to carry it — design §4.6 site-3 wires that up. The
    identically-named HTTP/1 latch in ``http1.rs`` is NOT this: it is a
    connection-level ``disabled_reason``, it has no span, and it stays a debug
    string. Same string, two fates; see the module docstring.
    """

    TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS = "tool_call_id_unavailable_in_process"
    """The framework ran the tool in-process and never exposed a tool-call id,
    so the tool span cannot be joined to the assistant message that requested it.

    Emitted from ``_adapters/_anthropic_agent_sdk.py::_describe_tool_call``, on every
    span an in-process SDK MCP tool produces. The handler is dispatched with
    ``{name, arguments}`` and nothing else, so the id genuinely does not reach
    it; the alternative — matching name and arguments against the stream in
    arrival order — is the framework-identifier heuristic this design exists to
    remove, and it would be indistinguishable from a real join downstream.
    """

    SNAPSHOT_TYPE_UNKNOWN = "snapshot_type_unknown"
    """A snapshot arrived with a type outside ``SnapshotType``, so it was
    recorded as ``UNSPECIFIED``.

    Emitted from ``_assembly/_snapshot.py::SnapshotDraft.__init__``, when
    ``coerce_snapshot_type`` cannot resolve the value the caller handed
    ``capture_state_snapshot``. Closing ``SnapshotType`` is what built this
    emitter; before it, an unrecognized type was flattened to ``UNSPECIFIED``
    by ``codec.rs``'s ``map_snap`` with nothing recorded anywhere.
    """

    # ------------------------------------------------------------------
    # Identity ambiguity (§5.4)
    # ------------------------------------------------------------------

    TOOL_NAME_COLLISION = "tool_name_collision"
    """Two distinct MCP servers exposed tools that sanitize to the same
    CLI-visible name, so the name on this span cannot identify which server ran.

    Design §5.4's V3 correction. Both emitters are in
    ``_adapters/_anthropic_agent_sdk.py::_describe_tool_call``, and both
    mean "the two observers of this call may not have met on one key":

      * the server's CLI token was never resolved (nothing in any options'
        ``mcp_servers`` matched this server instance), so the handler keys on the
        server's own name while the hook keys on the dict key — a key SPLIT,
        which ``claim()`` cannot arbitrate, so the call is reported twice;
      * ``CLAUDE_AGENT_SDK_MCP_NO_PREFIX`` is set and two wrapped servers export
        the same bare name, so the hook cannot attribute its observation at all
        and stands down — this span is the only record of the call.
    """

    # ------------------------------------------------------------------
    # Transport timing (census) — a duration is missing, not zero
    # ------------------------------------------------------------------

    CONNECT_TIMING_UNAVAILABLE = "connect_timing_unavailable"
    """``tcp_connect_ms`` is 0 because it could not be measured, not because the
    connection was instant.

    Emitted from ``_interceptors/_socket.py::RawSocketInterceptor._resolve_timing``
    (always — the raw-socket seam has no connect-time store) and
    ``_interceptors/_ssl.py::SSLInterceptor._resolve_timing`` (sync path: the
    shared timing store had no record for this fileno; async path: no stamped
    ``_wardex_timing`` record at all).

    NOTE (census): absorbed the free string ``async_connect_unavailable``
    (``_ssl.py::_resolve_timing``, anyio/httpx path where TLS and TCP are
    separate layers so ``total_ms`` is 0 and connect cannot be derived). What is
    lost is the provenance — sync fileno miss vs anyio layer split. Merged
    anyway: both assert the same fact, ``tcp_connect_ms`` is unknown rather than
    zero, and the user action in both cases is the same (none).
    """

    TTFT_UNAVAILABLE_H2 = "ttft_unavailable_h2"
    """Time-to-first-token is absent on a streamed HTTP/2 response: the seam
    sees DATA frames, not SSE event boundaries.

    Emitted from ``_interceptors/_seam.py::ByteSeamInterceptor._build_span`` when
    a reassembled stream carries core semantics and ``txn.version == "2"``.
    """

    TTFT_IPC_APPROXIMATION = "ttft_ipc_approximation"
    """Time-to-first-token was measured at the IPC boundary, not at the wire, so
    it includes subprocess and pipe latency.

    Emitted from ``_adapters/_assembler.py::SessionAssembler._build_chat``
    whenever a first-delta timestamp exists for the turn.
    """

    TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS = "transport_timing_unavailable_subprocess"
    """No transport timing at all: the LLM call happened inside a CLI subprocess
    and wardex observed only the IPC stream.

    Emitted from ``_adapters/_assembler.py`` as the module constant
    ``_BASE_LIMITATION``, on the four span classes that module assembles out of
    the IPC stream and the hook payloads: the root ``invoke_agent``, a sub-agent
    ``invoke_agent``, ``chat``, and the hook/stream-driven ``execute_tool``.

    The adapter's FIFTH span class does not carry it, and must not. The
    in-process ``execute_tool`` span opened by
    ``_adapters/_anthropic_agent_sdk.py::_run_tool`` brackets a handler
    wardex wrapped in *this* process, so its duration is measured directly
    rather than inferred from an IPC stream — attaching the marker there would
    claim the timing is absent when it is the one timing the adapter owns.
    """

    # ------------------------------------------------------------------
    # Caps reached (census + one added since). These stay separate members
    # because they are separate FACTS about the capture, each with a different
    # replay consequence — not, as the first draft of this section claimed,
    # because each names a different `crates/wardex-limits` knob. Two of them
    # (BODY_CAP_EXCEEDED, GRPC_MESSAGE_TRUNCATED) are driven by the same
    # `max_body_bytes`, and another limit — `max_ws_frame_bytes` — surfaces
    # under FRAME_PARSE_FAILED rather than here. Each docstring below names the
    # knob that was MEASURED to drive it, because these values are what a
    # dashboard turns into "raise this setting" advice — a knob named from
    # memory sends the user to a tunable that changes nothing.
    # ------------------------------------------------------------------

    BODY_CAP_EXCEEDED = "body_cap_exceeded"
    """An HTTP body was stored up to ``max_body_bytes`` / ``max_opaque_body_bytes``
    and the rest was consumed without being kept.

    The only marker produced in Rust: ``crates/wardex-protocol/src/http1.rs``,
    ``append_capped``. It crosses the PyO3 boundary as
    ``Vec<&'static str>`` (``bindings/python/src/lib.rs``), is read back at
    ``_protocol/_http1.py``, and is folded into the span's markers by
    ``_interceptors/_seam.py::ByteSeamInterceptor._build_span``.
    """

    GRPC_MESSAGE_TRUNCATED = "grpc_message_truncated"
    """A gRPC message runs past the end of the captured body: the last
    length-prefixed frame is incomplete, so its payload is not the whole message.

    Emitted from ``_semantics/_grpc.py::build_grpc_fields`` when
    ``parse_grpc_frames`` (``crates/wardex-protocol/src/grpc.rs``) reports
    ``truncated`` — which it does when fewer than 5 bytes remain for a prefix or
    the declared length runs past the buffer. That parser takes no ``Limits`` at
    all, so the knob upstream of it is the HTTP body cap: ``application/grpc``
    is classified meaningful by ``cap_for_content_type``, so a gRPC body is
    stored up to ``max_body_bytes``, not the opaque cap. NOT
    ``max_decoded_bytes``, which bounds decompression in ``semantic.rs`` and
    never reaches gRPC framing — the semantic parser is not even called on the
    gRPC branch.

    Kept separate from ``BODY_CAP_EXCEEDED`` despite sharing that knob. "The
    body was stored short" and "a message was cut in half" are different facts
    with different replay consequences, and on HTTP/2 — where gRPC actually
    lives — they do not even co-occur: ``http2.rs`` records its truncation on
    the stream and pushes no marker, so ``body_cap_exceeded`` never appears and
    this is the only marker a user gets.
    """

    WS_PAYLOAD_TRUNCATED = "ws_payload_truncated"
    """The WebSocket content sample hit ``ws_sample_bytes`` in either direction,
    so the body bytes on this span are a prefix of the conversation.

    Emitted from ``_interceptors/_trackers.py::_WebSocketTracker._build_txn``,
    from the truncation flags ``_append_sample`` sets when the accumulated
    sample passes the cap — resolved from ``limits_defaults()["ws_sample_bytes"]``
    (and passed as ``sample_cap`` by ``_interceptors/_seam.py``). NOT
    ``max_ws_frame_bytes``: that one bounds a single frame's declared length in
    ``crates/wardex-protocol/src/websocket.rs`` and an oversize frame does not
    truncate anything — it disables the parser, which surfaces as
    ``FRAME_PARSE_FAILED``.
    """

    CONNECTION_EVICTED = "connection_evicted"
    """The connection table hit ``max_connections`` and this connection's
    tracker was flushed early, so its span ends at the eviction instant.

    Emitted from ``_interceptors/_seam.py::ByteSeamInterceptor._state`` via
    ``_WebSocketTracker.flush(marker)``. Before the census rewired that site it
    was the free string ``ws_evicted``.

    NOTE (census): renamed, NOT merged into ``UNIT_EVICTED``. Both say
    "something was evicted", but they name different tables and different
    tunables — ``max_connections`` here, ``max_units`` there — and a merged
    marker would send the user to the wrong knob. The rename drops the ``ws_``
    prefix because the connection table is not WebSocket-specific.
    """

    PARSE_SKIPPED_AT_SHUTDOWN = "parse_skipped_at_shutdown"
    """The process was ending and the shutdown budget ran out before this
    transaction's deferred LLM-semantic parse: it shipped unparsed rather than
    not at all — transport, timing and status are measured; ``gen_ai`` is
    absent because the parser never ran.

    Emitted from ``_finalize.py`` — the single site that names it, on the
    fallback ``drain_all`` assembles for every job still pending when the
    shutdown budget ends (``Leftover.FALLBACK``: close and the signal path,
    where a kept job would die with the process).

    The knob is a SHUTDOWN BUDGET, not a capacity cap — ``close(timeout)`` /
    ``batching.shutdown_timeout``, or wardex's own 2 s signal-flush budget —
    which is what keeps it apart from ``PARSE_BACKLOG_FULL`` (same unparsed
    shipment, capacity knob) under the census rule: the reader's next action
    differs. Not ``SEMANTIC_PARSE_FAILED`` (the parser ran and understood
    nothing) and not ``INSTRUMENTATION_DEGRADED`` (the parser raised): here
    the parser was never given the chance. Same body-withholding rule as
    ``PARSE_BACKLOG_FULL``: when wardex's own degradation is the only thing
    that admitted the span past the capture gate, the payload stays home.
    """

    OTLP_ATTRIBUTE_TRUNCATED = "otlp_attribute_truncated"
    """An attribute value hit ``max_otlp_attribute_bytes`` on the way out, so
    what a backend shows for it is a prefix of what wardex captured.

    Emitted from ``crates/wardex-codec/src/otlp/map.rs`` — ``cap_attribute_values``
    for a value over the bound, and ``drop_payload_attributes`` for a span whose
    encoded size alone exceeds ``max_otlp_request_bytes``, where the payload
    goes entirely so the span itself can still be exported. Both attach it by
    proto NUMBER rather than by a string literal, which is why
    ``test_limitation_census.py`` scans Rust for ``Limitation::`` references as
    well as for marker literals.

    The one marker in this enum that describes the EXPORT rather than the
    capture, and the reason it is not ``BODY_CAP_EXCEEDED``: that one names
    ``max_body_bytes``, a bound on the raw bytes a parser keeps, while this
    names a bound on what one attribute costs on a wire that measures it after
    the base64 rewrite. A payload inside the first and outside the second is
    the ordinary case rather than a corner, and merging the two would send a
    user to raise a capture cap that was never the constraint.
    """

    EXTRA_KEYS_DROPPED = "extra_keys_dropped"
    """An open dynamic key family crossed ``max_extra_keys`` and whole leaves
    were dropped; how many rides beside the marker as
    ``wardex.usage_leaves.dropped_count``.

    Emitted from ``_interceptors/_seam.py``, off the native parser's
    ``usage_dropped_count`` — today the bound has exactly one enforcement
    site, the provider-usage mirror (``wardex.usage.*``): every scalar leaf
    of the provider's usage object, spelling preserved, which is an OPEN
    family because providers add billing counters faster than any typed
    table follows (design §6.5 tier 1 is this exact case). Lowering the
    knob prunes usage leaves only until draft-level enforcement lands.

    Not ``OTLP_ATTRIBUTE_TRUNCATED`` (39), by the census rule: that one
    names ``max_otlp_attribute_bytes`` and cuts VALUES on the export
    surface; this one names ``max_extra_keys`` and drops whole KEYS at
    capture. An integer usage leaf can never carry 39 at all (numeric
    values are not truncatable), and a string leaf can carry both,
    independently. The count also includes two structural sanity bounds on
    the same family (path > 120 bytes, depth > 6) that no real provider
    usage object approaches — so every drop a real workload sees is
    ``max_extra_keys``'s, and the marker's knob is the user's next action.
    """

    PARSE_BACKLOG_FULL = "parse_backlog_full"
    """The deferred-parse queue was full, so the OLDEST captured-but-unparsed
    transaction shipped without its LLM-semantic parse — transport, timing and
    status are measured; ``gen_ai`` is absent because the parser never ran on
    this body.

    Emitted from ``_finalize.py`` — the single site that names it: on the
    eviction fallback when the backlog crosses ``max_parse_backlog`` /
    ``max_parse_backlog_bytes``, and on the clamp that keeps a single
    over-bound body out of the queue entirely (so the byte bound stays
    literal).

    Not ``BODY_CAP_EXCEEDED``, by the caps-band rule: that one caps what one
    body KEEPS in raw bytes; this one caps how many finished transactions may
    WAIT for the deferred parse. And not ``CONNECTION_EVICTED``: same
    eviction shape, different table — that one drops connection STATE before
    a transaction exists, while this one ships a finished transaction
    unparsed, never silently. Not ``SEMANTIC_PARSE_FAILED`` either: that
    marker means the parser RAN and understood nothing, this one means it
    never ran. Raw bodies ride along EXCEPT when wardex's own degradation is
    the only reason the span passed the capture gate — a body the user's mode
    excluded must not leave the process because wardex was overloaded.
    """

    # ------------------------------------------------------------------
    # Parsing / interpretation (census)
    # ------------------------------------------------------------------

    FRAME_PARSE_FAILED = "frame_parse_failed"
    """The framing layer failed, so the transport fields on this span are
    partial or synthesized.

    Emitted from ``_semantics/_grpc.py::build_grpc_fields`` (gRPC frame
    parse raised; the span falls back to plain h2 fields) and
    ``_interceptors/_trackers.py::_WebSocketTracker._build_txn`` (either
    direction's frame parser latched off). Before the census those two sites
    emitted the free strings ``grpc_parse_failed`` and ``ws_parse_failed``.

    This member carries a LIMIT as well as a bug, which its name does not say:
    a WebSocket frame whose declared payload exceeds ``max_ws_frame_bytes``
    makes ``crates/wardex-protocol/src/websocket.rs`` return ``ParseStep::Error``
    and latch ``disabled``, which the tracker reads back as ``ws_parse_failed``.
    So a user who sees this on a WebSocket span has two candidate causes — a
    desynced parser and a frame above the cap — and only the second has a knob.
    ``WS_PAYLOAD_TRUNCATED`` is NOT that knob; see its docstring.

    NOTE (census): merged, and deliberately NOT merged with
    ``SEMANTIC_PARSE_FAILED``. That one is a different layer — framing
    succeeded and the transport fields are valid — and it leads to a different
    follow-up (parser bug vs unsupported provider).
    """

    SEMANTIC_PARSE_FAILED = "semantic_parse_failed"
    """Framing succeeded but the LLM body parser produced no core semantics and
    no output messages, so the span has transport truth and no gen_ai truth.

    Emitted from ``_interceptors/_seam.py::ByteSeamInterceptor._build_span``.
    """

    PAYLOAD_COMPRESSED = "payload_compressed"
    """The payload was observed compressed and wardex did not decompress it, so
    body bytes on this span are not readable content.

    Emitted from ``_semantics/_grpc.py::build_grpc_fields`` (any request or
    response message had its compressed flag set) and
    ``_interceptors/_trackers.py::_WebSocketTracker._build_txn``
    (permessage-deflate negotiated). Before the census those two sites emitted
    the free strings ``grpc_compressed`` and ``ws_compressed``.

    NOTE (census): merged. What is lost is which protocol it was —
    ``TransportAttributes.protocol`` already carries that, and encoding a
    protocol into a marker duplicates a field.
    """

    TOOL_ARGS_UNPARSED = "tool_args_unparsed"
    """A tool call was found in the response but its arguments were not valid
    JSON, so they are carried as an opaque string.

    Emitted from ``_interceptors/_seam.py::ByteSeamInterceptor._build_span``.
    """

    OUTPUT_MESSAGES_UNMAPPED_PART = "output_messages_unmapped_part"
    """``gen_ai.output.messages`` was reconstructed with at least one part the
    mapper did not recognize, so the *response* replay is incomplete.

    Emitted from ``_interceptors/_seam.py::ByteSeamInterceptor._build_span``.
    Deliberately NOT merged with the input-side marker: in a forensic replay,
    an incomplete response and an incomplete prompt support different
    conclusions.
    """

    INPUT_MESSAGES_UNMAPPED_PART = "input_messages_unmapped_part"
    """``gen_ai.input.messages`` was reconstructed with at least one part the
    mapper did not recognize, so the *prompt* replay is incomplete.

    Emitted from ``_interceptors/_seam.py::ByteSeamInterceptor._build_span``.
    """

    # ------------------------------------------------------------------
    # Streaming (census)
    # ------------------------------------------------------------------

    REASSEMBLED_FROM_STREAM = "reassembled_from_stream"
    """The response body on this span is wardex's reassembly of a token stream,
    not a single response the server sent.

    Emitted from ``_interceptors/_seam.py::ByteSeamInterceptor._build_span`` when
    the semantic parser reports a stream and core semantics were recovered.
    """

    STREAM_USAGE_UNAVAILABLE = "stream_usage_unavailable"
    """A stream was reassembled but carried no usage block, so output token
    counts are absent rather than zero.

    Emitted from ``_interceptors/_seam.py::ByteSeamInterceptor._build_span``.
    """

    SSE_UNKNOWN_PROVIDER = "sse_unknown_provider"
    """An SSE stream was detected and reassembled but matched no known provider
    shape, so nothing was mapped out of it.

    Emitted from ``_interceptors/_seam.py::ByteSeamInterceptor._build_span``.
    """

    # ------------------------------------------------------------------
    # Protocol-specific (census)
    # ------------------------------------------------------------------

    GRPC_WEB_UNSUPPORTED = "grpc_web_unsupported"
    """The content type was ``application/grpc-web``, whose framing wardex does
    not parse; the span was left as plain HTTP/2.

    Emitted from ``_interceptors/_seam.py::ByteSeamInterceptor._build_span``.
    """

    GRPC_STATUS_UNAVAILABLE = "grpc_status_unavailable"
    """No ``grpc-status`` was observed — trailers-only response, or trailers the
    seam never saw — so the span's status is derived from HTTP alone.

    Emitted from ``_semantics/_grpc.py::build_grpc_fields``.
    """

    WS_NO_CLOSE = "ws_no_close"
    """The WebSocket span was emitted without ever seeing a CLOSE frame, so its
    close code and duration are not trustworthy.

    Emitted from ``_interceptors/_seam.py::ByteSeamInterceptor._retire``, via
    ``_WebSocketTracker.flush(marker)``, reached from ``uninstall()`` and from
    the shared socket-close hook (``_connection_closed``). The uninstall path
    travels alongside ``ADAPTER_UNINSTALLED``; see that member for why they
    stay two markers.
    """

    WS_LLM_SEMANTICS_UNREAD = "ws_llm_semantics_unread"
    """A WebSocket connection carried LLM calls wardex read none of: the
    upgrade path is a row the endpoint table marks WebSocket-capable (the
    OpenAI Responses API, ``wss://…/v1/responses``, the openai-agents SDK's
    opt-in transport), the host names that provider — or, on an unknown host,
    the first client message is a Responses ``response.create`` — and at least
    one client message crossed. Responses events inside WebSocket frames are
    not parsed, by decision, so the span carries transport counts and payload
    samples but no model, tokens or messages; ``ws.messages.sent``
    approximates the calls. Reader's next action: the framework's HTTP
    transport (openai-agents ``use_responses_websocket=False``, the default)
    yields gen_ai spans. Names NO knob.

    Not ``FRAME_PARSE_FAILED``: framing succeeded. Not
    ``SEMANTIC_PARSE_FAILED``: no parser ran; none exists for this transport.
    Not ``SSE_UNKNOWN_PROVIDER``: provider and endpoint are KNOWN; the
    transport is the gap. Not ``PARSE_BACKLOG_FULL`` /
    ``PARSE_SKIPPED_AT_SHUTDOWN``: those skip a parse for capacity and a knob
    restores it.

    Emitted from ``_interceptors/_trackers.py::_WebSocketTracker._build_txn``,
    confirmed on the first client message; the seam's gate reads the same fact
    (``_Txn.ws_llm_call``) as a capture claim.
    """

    # ------------------------------------------------------------------
    # Adapter session (census)
    # ------------------------------------------------------------------

    SESSION_ABORTED = "session_aborted"
    """The agent session ended without a clean result: an error was raised, or
    teardown arrived with no result message at all.

    Emitted from ``_adapters/_assembler.py::SessionAssembler._stamp_root``, on
    all three of its terminal branches (result present but errored, no result
    and an error, no result and no error).
    """

    # ------------------------------------------------------------------
    # Agent SDK OTel bridge, fail-open
    # ------------------------------------------------------------------

    OTEL_BRIDGE_NO_DATA = "otel_bridge_no_data"
    """The bridge injected telemetry env into this session's CLI — the
    subprocess-env read-back CONFIRMED the injection landed — and zero spans
    were routed to the session before it finalized. The tree below this root
    is exactly the bridge-off tree.

    Emitted from ``_adapters/_assembler.py::SessionAssembler._merge_bridge``
    at session finalize. A CONDITION marker that names no knob: the next
    action lives outside wardex (check the CLI version, or a machine policy
    that strips subprocess env or blocks loopback connections). Gated on the
    read-back-CONFIRMED binding, never on "bridge configured": an injection
    wardex cannot prove reached the subprocess env — a user transport, an SDK
    surface change — must not be reported as the CLI staying silent (I4), so
    unconfirmed sessions get the counter
    ``adapters.assembler.otel_bridge_unconfirmed_no_data`` instead.

    Not merged with ``OTEL_BRIDGE_SCHEMA_UNKNOWN``, by the census rule
    (§6.5.1): the reader's next action differs. Here nothing ARRIVED —
    a transport-level fact; there, data arrived and meant nothing to the
    bridge — a schema-level fact whose fix is a report or an SDK upgrade.
    """

    OTEL_BRIDGE_SCHEMA_UNKNOWN = "otel_bridge_schema_unknown"
    """Bridge telemetry arrived for this session and decoded to nothing the
    bridge recognizes: classification produced zero known ``claude_code.*``
    spans, or a POST under the bridge's token did not decode as OTLP at all.

    Emitted from ``_adapters/_assembler.py::SessionAssembler._merge_bridge``
    at finalize, on two routes: the session's routed spans classified to
    nothing known, or the receiver counted an undecodable POST attributable
    to the sole live bridge session (a counted inference — with several live
    sessions only the counter ``adapters.anthropic.otel_bridge.undecodable``
    speaks, because attributing a bodyless failure to one of N sessions would
    be a guess). The CLI's telemetry is beta —
    ``CLAUDE_CODE_ENHANCED_TELEMETRY_BETA`` disclaims stability by name — so
    this is the reachable fail-open the design requires: the tree stays
    exactly today's and the root says why the bridge added nothing. See
    ``OTEL_BRIDGE_NO_DATA`` for why the two stay separate members.
    """


_BY_VALUE: dict[str, Limitation] = {m.value: m for m in Limitation}
"""Wire value -> member, for `Limitation.from_wire`. See its docstring for why
this is a table rather than a `try: Limitation(value)`."""
