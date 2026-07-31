"""The six parentage sites share one core — `assembly.resolve_parentage` (I1).

Each site used to answer "who is my parent" on its own, and the six
answers had drifted: two of them dropped `trace_flags`, three produced
`correlation=None` when they started a new trace, one recorded a joined W3C
parent as if it had come from the ContextVar, and one dropped the call on the
floor. The point of routing them through `assembly.resolve_parentage()` is not
that the code is shorter — it is that the question below can be asked ONCE and
answered identically by every site.

So these tests are deliberately parametrized over the sites rather than written
per site. A site that drifts again fails as one row of a table, next to the five
that did not.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import CaptureMode
from wardex_sdk._tracing import span, trace
from wardex_sdk._types import ToolDefinitionSet
from wardex_sdk.adapters._assembler import SessionAssembler
from wardex_sdk.assembly import ParentSource
from wardex_sdk.interceptors._mcp_stdio import _ProcState
from wardex_sdk.interceptors._seam import ByteSeamInterceptor, _ConnectionState
from wardex_sdk.interceptors._trackers import _Txn

SAMPLED = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
UNSAMPLED = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-00"


class _FakeClient:
    def __init__(self, mode: CaptureMode = CaptureMode.ALL) -> None:
        self.config = WardexConfig(api_key="k", capture_mode=mode)
        self.spans: list = []
        self.snapshots: list = []

    def capture_span(self, span) -> None:
        self.spans.append(span)

    def capture_snapshot(self, snapshot) -> None:
        self.snapshots.append(snapshot)

    def close(self, timeout: float = 5.0) -> None:
        """conftest's autouse fixture closes whatever the hub holds."""


class _Seam(ByteSeamInterceptor):
    def _select_tracker(self, obj):  # pragma: no cover - not reached in these tests
        raise NotImplementedError

    def _resolve_timing(self, obj, st):
        return 0.0, 0.0, False, ()

    def name(self):
        return "test-seam"

    def install(self, client, ctx=None):
        self._client = client

    def uninstall(self):
        pass


def _client() -> _FakeClient:
    _hub.reset_for_test()
    client = _FakeClient()
    _hub.set_client(client)
    return client


def _latched_parent():
    """What the byte seam's tracker latches at REQUEST time, standing in for it.

    `_Txn.parent` is filled in by `_trackers.py` on the request path; these
    drivers run the emit path directly, so the test plays the tracker's part.
    """
    return _hub.get_current_scope().active_span_context


# --------------------------------------------------------------------------
# one driver per span-emitting site
# --------------------------------------------------------------------------


def _site_seam_http(client: _FakeClient):
    seam = _Seam()
    seam._client = client
    st = _ConnectionState(tracker=None, server_address="api.openai.com", server_port=443)
    txn = _Txn(
        method="POST",
        path="/v1/chat/completions",
        status=200,
        request_body=b"{}",
        response_body=b"{}",
        parent=_latched_parent(),
        start_ns=1,
        end_ns=2,
        ttfb_ms=0.0,
    )
    seam._emit_span(SimpleNamespace(server_hostname="api.openai.com"), st, txn)
    return client.spans[-1]


def _site_seam_ws(client: _FakeClient):
    seam = _Seam()
    seam._client = client
    st = _ConnectionState(tracker=None, server_address="ws.example.com", server_port=443)
    txn = _Txn(
        method="GET",
        path="/socket",
        status=101,
        request_body=b"",
        response_body=b"",
        parent=_latched_parent(),
        start_ns=1,
        end_ns=2,
        ttfb_ms=0.0,
        version="websocket",
        ws_close_code=1000,
    )
    seam._emit_ws(st, txn)
    return client.spans[-1]


def _site_mcp_stdio(client: _FakeClient):
    state = _ProcState()
    state.feed_request(b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"t"}}\n')
    spans = state.feed_response(b'{"jsonrpc":"2.0","id":1,"result":{"content":[]}}\n')
    return spans[0]


def _site_adapter_root(client: _FakeClient):
    asm = SessionAssembler(client)
    asm.on_outbound(
        1,
        json.dumps(
            {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": "hi"}}
        ),
    )
    asm.on_close(1, None)
    return next(s for s in client.spans if s.name == "invoke_agent")


def _site_manual_span(client: _FakeClient):
    with span("manual"):
        pass
    return client.spans[-1]


_SPAN_SITES = {
    "site-1 _seam._emit_span": _site_seam_http,
    "site-2 _seam._emit_ws": _site_seam_ws,
    "site-3 _mcp_stdio._build_mcp_span": _site_mcp_stdio,
    "site-4 _assembler session root": _site_adapter_root,
    "site-6 _tracing.span": _site_manual_span,
}


# --------------------------------------------------------------------------
# the convergence itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("site", sorted(_SPAN_SITES))
def test_every_site_reports_a_local_parent_identically(site):
    client = _client()

    with trace("root") as root:
        emitted = _SPAN_SITES[site](client)

    assert emitted.context.trace_id == root.context.trace_id
    assert emitted.parent_span_id == root.context.span_id
    assert emitted.correlation is not None
    assert emitted.correlation.strategy is ParentSource.CONTEXTVAR
    assert emitted.correlation.confidence == 1.0
    assert emitted.correlation.active_span_id_at_capture == root.context.span_id


@pytest.mark.parametrize("site", sorted(_SPAN_SITES))
def test_every_site_marks_its_own_root_identically(site):
    """I4: "I started a new trace" is a statement, not the absence of one.

    Three of these sites used to emit `correlation=None` here, which is
    indistinguishable from "a parent was expected and lost".
    """
    client = _client()

    emitted = _SPAN_SITES[site](client)

    assert emitted.parent_span_id is None
    assert emitted.correlation is not None
    assert emitted.correlation.strategy is ParentSource.TRACE_ROOT
    assert emitted.correlation.confidence == 1.0
    # V9: wardex does not head-sample, so a trace wardex ORIGINATES is sampled.
    assert emitted.context.trace_flags == 1


@pytest.mark.parametrize("site", sorted(_SPAN_SITES))
def test_every_site_calls_a_joined_parent_a_header_parent(site):
    """A remote parent is not a ContextVar parent, at any site.

    `header` was a declared `ParentSource` with zero producers until this
    convergence: every site labelled a joined W3C parent `contextvar`, so a
    trace that came in over the wire was indistinguishable from one started
    in-process.
    """
    client = _client()

    with wardex_sdk.continue_trace({"traceparent": SAMPLED}):
        emitted = _SPAN_SITES[site](client)

    assert emitted.correlation.strategy is ParentSource.HEADER
    assert emitted.correlation.confidence == 1.0
    assert emitted.context.trace_id.hex() == SAMPLED.split("-")[1]


@pytest.mark.parametrize("site", sorted(_SPAN_SITES))
@pytest.mark.parametrize(("header", "flags"), [(SAMPLED, 1), (UNSAMPLED, 0)])
def test_every_site_propagates_the_received_trace_flags(site, header, flags):
    """V9: received flags travel; only wardex-originated traces assert `01`.

    Every site dropped this before — a span context was built with the default
    `trace_flags=0` regardless of what the trace was joined with.
    """
    client = _client()

    with wardex_sdk.continue_trace({"traceparent": header}):
        emitted = _SPAN_SITES[site](client)

    assert emitted.context.trace_flags == flags


def test_no_site_disagrees_about_the_shape_of_one_run():
    """The five sites in one process, one ambient span: one trace, one parent."""
    client = _client()

    with trace("root") as root:
        emitted = [driver(client) for _, driver in sorted(_SPAN_SITES.items())]

    assert {s.context.trace_id for s in emitted} == {root.context.trace_id}
    assert {s.parent_span_id for s in emitted} == {root.context.span_id}
    assert {s.correlation.strategy for s in emitted} == {ParentSource.CONTEXTVAR}
    # every span got its OWN id: `child_context()` is a factory, not an accessor
    assert len({s.context.span_id for s in emitted}) == len(emitted)


# --------------------------------------------------------------------------
# site-4 — the edges BELOW the adapter root, which the parentage core does not
# resolve: the adapter still picks the anchor itself
# --------------------------------------------------------------------------


def _assembler_with_an_unresolvable_stream_parent(client: _FakeClient) -> SessionAssembler:
    """A `chat` event naming a subagent this session never recorded.

    Real and routine, not contrived: `SubagentStart` arrives on the hook channel
    and stream events on stdout, so a `parent_tool_use_id` can legitimately
    arrive first — and once a long session passes `max_session_entries` the
    subagent is never recorded at all.
    """
    asm = SessionAssembler(client)
    asm.on_outbound(
        1,
        json.dumps(
            {"type": "user", "session_id": "s-1", "message": {"role": "user", "content": "hi"}}
        ),
    )
    asm.on_inbound(
        1,
        {
            "type": "assistant",
            "session_id": "s-1",
            "parent_tool_use_id": "toolu_never_seen",
            "message": {
                "id": "msg_01",
                "model": "claude-x",
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "content": [{"type": "text", "text": "hi"}],
            },
        },
    )
    return asm


def test_a_guessed_adapter_edge_makes_no_confidence_claim():
    """I4, in its load-bearing direction: no full-confidence claim on a guess.

    `_resolve_subagent_anchor` falls back to the session root whenever the id
    resolves to nothing, so this `chat` span is re-parented and flattened. That
    is pre-existing, and the ingestion move design §3.4 schedules is what fixes
    it. What must NOT happen meanwhile is the
    span claiming `strategy="unit_active", confidence=1.0` with no marker —
    which would make the flattened subtree byte-indistinguishable from a correct
    one and invisible to the standard triage query (`confidence < 1.0` or a
    non-empty `limitations`). Until `UnitRegistry.resolve()` supplies the
    evidence, the honest report is no report.
    """
    client = _client()

    with trace("root"):
        _assembler_with_an_unresolvable_stream_parent(client)

    chat = next(s for s in client.spans if s.name.startswith("chat "))
    assert chat.correlation is None


def test_only_the_adapter_root_publishes_a_resolved_correlation():
    """The boundary of what the adapter may claim, asserted rather than described.

    Widening it later is a decision to make deliberately; drifting into it by
    passing `p.correlation` at one more emit site is not.
    """
    client = _client()

    with trace("root"):
        asm = _assembler_with_an_unresolvable_stream_parent(client)
        asm.on_hook("SubagentStart", {"session_id": "s-1", "agent_id": "a-1"}, None)
        asm.on_hook("SubagentStop", {"session_id": "s-1", "agent_id": "a-1"}, None)
        asm.on_close(1, None)

    with_corr = {s.name for s in client.spans if s.correlation is not None}
    assert with_corr == {"invoke_agent", "root"}, (
        "the adapter's session root is the only span with a resolved "
        f"`correlation`; got {sorted(with_corr)}"
    )
    root = next(s for s in client.spans if s.name == "invoke_agent")
    assert root.correlation.strategy is ParentSource.CONTEXTVAR


def test_no_undeclared_strategy_value_reaches_the_wire():
    """Every `strategy` reaching a span is a `ParentSource` member, and few are.

    `ParentSource` declares more members than the set below — `unit_active` and
    `unit_alias` among them — and they are reachable through the unit registry,
    which none of these scenarios goes through. So a value appearing here is a
    site that started answering "how was the parent derived" with something new.
    That is a decision to make deliberately, not one to discover on a user's
    wire, where a value the release notes do not name buckets as unknown in
    every consumer that maps strategies.
    """
    client = _client()

    with trace("root"):
        asm = _assembler_with_an_unresolvable_stream_parent(client)
        asm.on_hook("SubagentStart", {"session_id": "s-1", "agent_id": "a-1"}, None)
        asm.on_hook("SubagentStop", {"session_id": "s-1", "agent_id": "a-1"}, None)
        asm.on_close(1, None)
        for site in _SPAN_SITES.values():
            site(client)

    shipped = {s.correlation.strategy for s in client.spans if s.correlation is not None}
    # STRENGTHENED when `adapter_hook`/`adapter_stream` were retired: they
    # answered "which source observed this", not "how was the parent derived",
    # and a tool span now makes no parentage claim at all. The set is down to
    # three values plus `None` for the spans that deliberately claim nothing.
    assert shipped <= {
        ParentSource.CONTEXTVAR,
        ParentSource.HEADER,
        ParentSource.TRACE_ROOT,
        None,
    }, f"a new strategy value reached a span; the set is now: {sorted(map(str, shipped))}"
    # ...and every one of them is a MEMBER. A raw string here would satisfy the
    # subset check above only by accident of never being compared, which is how
    # `strategy=src.value` survived the retype unnoticed.
    assert all(s is None or isinstance(s, ParentSource) for s in shipped)


# --------------------------------------------------------------------------
# site-5 — capture_state_snapshot, which used to return silently
# --------------------------------------------------------------------------


def test_snapshot_still_attaches_to_the_active_span():
    """The unchanged half: a snapshot names an EXISTING span, not a new child."""
    client = _client()

    with trace("root") as root:
        wardex_sdk.capture_state_snapshot(conversation_state=b"{}")

    (snapshot,) = client.snapshots
    assert snapshot.trace_id == root.context.trace_id
    assert snapshot.span_id == root.context.span_id
    assert not [k for k, _ in snapshot.attributes if k == "wardex.limitations"]


def test_snapshot_without_an_active_span_is_emitted_and_says_so():
    """BEHAVIOUR CHANGE. This call used to produce nothing at all.

    It returned at the `active is None` check — no span, no snapshot, no marker,
    no counter — so a user whose snapshot never arrived had nothing to look at.
    It now emits, and the marker is what makes the emission honest: the span_id
    on it names no span, and `parent_unresolved` is the field that says that
    out loud (I4, design §4.5).
    """
    client = _client()

    wardex_sdk.capture_state_snapshot(
        conversation_state=b'{"messages":[]}',
        tool_definitions=ToolDefinitionSet(),
        attributes={"user.key": "kept"},
    )

    (snapshot,) = client.snapshots
    assert snapshot.conversation_state == b'{"messages":[]}'
    assert dict(snapshot.attributes)["wardex.limitations"] == "parent_unresolved"
    assert dict(snapshot.attributes)["user.key"] == "kept"  # caller's kv survives


def test_an_orphan_snapshot_names_no_span_in_the_wires_own_vocabulary():
    """The absence is the all-zero span id, not a random one.

    A minted id would name a span that is never emitted: same shape as a real
    reference, so an OTel backend joining logs to spans records a dangling
    pointer and the only thing saying otherwise is `wardex.limitations` — an
    opaque kv no standard consumer parses. The all-zero id is the invalid-span
    sentinel every consumer already knows.
    """
    client = _client()

    wardex_sdk.capture_state_snapshot(conversation_state=b"{}")
    wardex_sdk.capture_state_snapshot(conversation_state=b"{}")

    assert [s.span_id.value for s in client.snapshots] == [b"\x00" * 8] * 2


def test_the_snapshots_limitation_key_is_never_emitted_twice():
    """`attributes` is a repeated KeyValue and nothing de-duplicates it.

    A consumer folding it into a map keeps one entry per key; if it keeps the
    caller's, the marker that makes the orphan record honest is the half that
    disappears. So the SDK owns this key on this record.
    """
    client = _client()

    wardex_sdk.capture_state_snapshot(attributes={"wardex.limitations": "user_value"})

    keys = [k for k, _ in client.snapshots[-1].attributes]
    assert keys.count("wardex.limitations") == 1
    assert dict(client.snapshots[-1].attributes)["wardex.limitations"] == "parent_unresolved"


def test_snapshot_orphan_is_distinguishable_from_a_deliberate_root():
    """`unresolved` vs `trace_root` is the distinction I4 exists to preserve.

    A snapshot ALWAYS expects a span; a span may legitimately start a trace. If
    both came out as `trace_root` the two failures would look identical.
    """
    client = _client()

    wardex_sdk.capture_state_snapshot(conversation_state=b"{}")
    orphan = client.snapshots[-1]
    rooted = _site_manual_span(client)

    assert dict(orphan.attributes)["wardex.limitations"] == "parent_unresolved"
    assert rooted.correlation.strategy is ParentSource.TRACE_ROOT
    assert not rooted.capture_integrity or "parent_unresolved" not in (
        rooted.capture_integrity.limitations
    )


# --------------------------------------------------------------------------
# site-6 — _tracing, which had no correlation at all
# --------------------------------------------------------------------------


def test_manual_span_now_carries_correlation():
    """BEHAVIOUR CHANGE: manual spans had `correlation=None` on every path."""
    client = _client()

    with trace("root") as root:
        with span("inner"):
            pass

    inner = next(s for s in client.spans if s.name == "inner")
    assert inner.correlation is not None
    assert inner.correlation.strategy is ParentSource.CONTEXTVAR
    assert inner.correlation.active_span_id_at_capture == root.context.span_id


def test_manual_span_joined_from_a_header_is_labelled_header():
    client = _client()

    with wardex_sdk.continue_trace({"traceparent": SAMPLED}):
        with span("inner"):
            pass

    assert client.spans[-1].correlation.strategy is ParentSource.HEADER


def test_manual_root_still_emits_a_sampled_traceparent():
    """The regression canary V9 names, asserted from the manual-span side.

    `_w3c.format_traceparent` no longer hardcodes `-01`. What keeps a
    wardex-rooted trace sampled is `resolve_parentage`'s no-parent branch, and
    this is the assertion that fails if that `1` is ever "simplified" to 0.
    """
    _client()

    with trace("root"):
        assert wardex_sdk.get_traceparent().endswith("-01")


def test_an_upstream_unsampled_decision_is_no_longer_promoted():
    """The other half of V9, and a real behaviour change on the wire.

    wardex used to receive `-00` and re-emit `-01`, overriding the upstream
    sampler. Now the decision is honoured.
    """
    _client()

    with wardex_sdk.continue_trace({"traceparent": UNSAMPLED}):
        with span("inner"):
            assert wardex_sdk.get_traceparent().endswith("-00")
