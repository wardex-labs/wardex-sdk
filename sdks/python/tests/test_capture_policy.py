"""The capture gate has ONE implementation — `assembly.should_capture`.

The question used to have three answers. `ByteSeamInterceptor` asked the
design §5.1 policy. `RawSocketInterceptor` OVERRODE that method with a
different predicate of the same name. `_mcp_stdio` never asked. The copies had
drifted, and the drift was not theoretical — the override never learned about
`CaptureMode.ALL`, and it replaced the local-parent clause instead of composing
with it, so a plaintext request issued inside a live wardex span was dropped
while the byte-identical request over TLS was captured.

So, like `test_parentage_convergence.py` before it, this file is parametrized
over the SITES rather than written per site: the question is asked once and
every routed site must answer it identically. A site that re-grows its own
opinion fails as one row of a table, next to the ones that did not.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from wardex_sdk import _hub
from wardex_sdk._assembly import Prefilter, capture_mode_of, should_capture
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._enums import CaptureMode
from wardex_sdk._interceptors import _seam
from wardex_sdk._interceptors._conn_timing import (
    install_shared_timing,
    shared_timing_store,
    uninstall_shared_timing,
)
from wardex_sdk._interceptors._mcp_stdio import _ProcState
from wardex_sdk._interceptors._seam import ByteSeamInterceptor, _ConnectionState
from wardex_sdk._interceptors._socket import RawSocketInterceptor
from wardex_sdk._interceptors._trackers import _Txn
from wardex_sdk._types import SpanContext, SpanId, TraceId
from wardex_sdk.context._contextvar import fork_active_span


def _ctx(*, remote: bool) -> SpanContext:
    return SpanContext(trace_id=TraceId.generate(), span_id=SpanId.generate(), is_remote=remote)


LOCAL = _ctx(remote=False)
REMOTE = _ctx(remote=True)
PARENTS = {"no-parent": None, "local": LOCAL, "remote": REMOTE}
MODES = list(CaptureMode)


class _FakeClient:
    def __init__(self, mode: CaptureMode) -> None:
        self.config = WardexConfig(capture_mode=mode, backend=BackendConfig(api_key="k"))
        self.spans: list[Any] = []

    def capture_span(self, span: Any) -> None:
        self.spans.append(span)

    def close(self, timeout: float = 5.0) -> None:
        """conftest's autouse fixture closes whatever the hub holds."""


class _Sem:
    """A stand-in for the native `LlmSemantics` record.

    Only one of its fields is load-bearing here — `core`, which stands for
    "the parser found a model or a token count", i.e. the single bit the policy
    consumes as `agent_semantic`. Every other field answers None, which is what
    the real parser returns for something it did not find, so the emit path
    downstream of the gate runs for real rather than against a mock.
    """

    def __init__(self, *, core: bool) -> None:
        self.core = core

    def __getattr__(self, name: str) -> None:
        return None


LLM = _Sem(core=True)
GENERIC = _Sem(core=False)


@pytest.fixture(autouse=True)
def _llm_semantics_by_marker(monkeypatch):
    """Core semantics are whatever `_Sem.core` says.

    Same convention as `test_capture_mode.py`: the subject here is the policy,
    not the body parser, so the one input the parser contributes is stubbed and
    the seam is driven with it directly.
    """
    monkeypatch.setattr(_seam, "has_core_semantics", lambda sem: bool(getattr(sem, "core", False)))


@contextmanager
def _ambient(parent: SpanContext | None):
    """Run the block with `parent` as the active span context, or with none."""
    _hub.reset_for_test()
    if parent is None:
        yield
    else:
        with fork_active_span(parent):
            yield


class _TlsSeam(ByteSeamInterceptor):
    """The base policy and nothing else — a seam with no transport opinion."""

    sem: Any = None
    timing_calls = 0

    def _select_tracker(self, obj):  # pragma: no cover - not reached here
        raise NotImplementedError

    def _resolve_timing(self, obj, st):
        self.timing_calls += 1
        return 0.0, 0.0, False, ()

    def _parse_semantics(self, url_host, txn):
        return self.sem

    def name(self):
        return "test-tls-seam"

    def install(self, client, ctx=None):
        self._client = client

    def uninstall(self):
        pass


class _PlaintextSeam(RawSocketInterceptor):
    """The real plaintext seam, with only the body parser stubbed out.

    `_transport_prefilter`, `_resolve_timing` and the allowlist are the
    shipping ones: the prefilter composition is exactly what is under test.
    """

    sem: Any = None

    def _parse_semantics(self, url_host, txn):
        return self.sem


def _http_txn(parent: SpanContext | None, path: str = "/v1/chat/completions") -> _Txn:
    return _Txn(
        method="POST",
        path=path,
        status=200,
        request_body=b"{}",
        response_body=b"{}",
        parent=parent,
        start_ns=1,
        end_ns=2,
        ttfb_ms=0.0,
    )


def _ws_txn(parent: SpanContext | None) -> _Txn:
    return _Txn(
        method="GET",
        path="/socket",
        status=101,
        request_body=b"",
        response_body=b"",
        parent=parent,
        start_ns=1,
        end_ns=2,
        ttfb_ms=0.0,
        version="websocket",
        ws_close_code=1000,
    )


def _state(host: str = "api.example.com", port: int = 443) -> _ConnectionState:
    return _ConnectionState(tracker=None, server_address=host, server_port=port)


def _obj(fileno: int = 4242) -> Any:
    return SimpleNamespace(fileno=lambda: fileno)


# --------------------------------------------------------------------------
# one driver per gated site — each returns "was a span emitted?"
# --------------------------------------------------------------------------


def _drive_tls_http(mode: CaptureMode, parent: SpanContext | None, *, sem: Any) -> bool:
    seam = _TlsSeam()
    seam._client = _FakeClient(mode)
    seam.sem = sem
    seam._emit_span(_obj(), _state(), _http_txn(parent))
    return bool(seam._client.spans)


def _drive_tls_ws(mode: CaptureMode, parent: SpanContext | None) -> bool:
    seam = _TlsSeam()
    seam._client = _FakeClient(mode)
    seam._emit_ws(_state(), _ws_txn(parent))
    return bool(seam._client.spans)


def _drive_plaintext_http(mode: CaptureMode, parent: SpanContext | None, *, sem: Any) -> bool:
    seam = _PlaintextSeam()
    seam._client = _FakeClient(mode)
    seam.sem = sem
    seam._emit_span(_obj(), _state(), _http_txn(parent))
    return bool(seam._client.spans)


def _drive_mcp(mode: CaptureMode, parent: SpanContext | None) -> bool:
    state = _ProcState(mode=mode)
    with _ambient(parent):
        # The latch happens on the REQUEST path, so the ambient parent has to be
        # in place here and not on the response (design §4.1).
        state.feed_request(
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"t"}}\n'
        )
    spans = state.feed_response(b'{"jsonrpc":"2.0","id":1,"result":{"content":[]}}\n')
    return bool(spans)


# site -> (driver, what that site claims about `agent_semantic`)
_GATED_SITES: dict[str, tuple[Any, bool]] = {
    "seam-tls http (generic)": (lambda m, p: _drive_tls_http(m, p, sem=GENERIC), False),
    "seam-tls http (llm)": (lambda m, p: _drive_tls_http(m, p, sem=LLM), True),
    "seam-tls websocket": (_drive_tls_ws, False),
    "seam-plaintext http (generic)": (lambda m, p: _drive_plaintext_http(m, p, sem=GENERIC), False),
    "seam-plaintext http (llm)": (lambda m, p: _drive_plaintext_http(m, p, sem=LLM), True),
    "mcp_stdio tools/call": (_drive_mcp, True),
}


# --------------------------------------------------------------------------
# the convergence itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("site", sorted(_GATED_SITES))
@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("parent", sorted(PARENTS), ids=sorted(PARENTS))
def test_every_gated_site_answers_with_the_shared_predicate(site, mode, parent):
    """The whole point, as one table: emit ⇔ `assembly.should_capture` says so.

    Not "each site behaves sensibly" — each site's answer is asserted to be the
    SHARED predicate applied to that site's own declared inputs. A site that
    re-implements the rule fails here even if its re-implementation is correct
    today, which is the only way to keep it correct in a year.
    """
    drive, agent_semantic = _GATED_SITES[site]

    emitted = drive(mode, PARENTS[parent])

    assert emitted is should_capture(mode, parent=PARENTS[parent], agent_semantic=agent_semantic), (
        f"{site} disagrees with assembly.should_capture at mode={mode.value}, parent={parent}"
    )


def test_the_tls_and_plaintext_seams_no_longer_disagree_about_the_same_bytes():
    """BEHAVIOUR CHANGE. The asymmetry the override created, stated directly.

    Identical generic traffic, identical ambient span, two transports. The
    plaintext seam used to drop it because its `_should_capture` override
    replaced the local-parent clause rather than composing with it, so wrapping
    a call in `wardex.span()` bought you capture over TLS and nothing over
    plaintext (design §4.4).
    """
    assert _drive_tls_http(CaptureMode.AGENT, LOCAL, sem=GENERIC) is True
    assert _drive_plaintext_http(CaptureMode.AGENT, LOCAL, sem=GENERIC) is True


def test_capture_mode_all_now_reaches_the_plaintext_seam():
    """BEHAVIOUR CHANGE. `ALL` meant "everything except plaintext HTTP"."""
    assert _drive_plaintext_http(CaptureMode.ALL, None, sem=GENERIC) is True


# --------------------------------------------------------------------------
# the seam's own opinion — Prefilter, which is about the CONNECTION
# --------------------------------------------------------------------------


def test_the_base_seam_defers_because_it_knows_nothing_about_the_peer():
    seam = _TlsSeam()
    assert seam._transport_prefilter(_state()) is Prefilter.DEFER


@pytest.mark.parametrize(
    ("host", "allow", "expected"),
    [
        ("169.254.169.254", (), Prefilter.DENY),
        ("169.254.169.254", ("169.254.169.254",), Prefilter.DENY),  # DENY outranks ALLOW
        ("10.0.0.5", ("10.0.0.5",), Prefilter.ALLOW),
        ("10.0.0.5", ("10.0.0.5:443",), Prefilter.ALLOW),
        ("10.0.0.5", ("other.host",), Prefilter.DEFER),
        ("10.0.0.5", (), Prefilter.DEFER),
    ],
)
def test_the_plaintext_prefilter_table(host, allow, expected):
    seam = _PlaintextSeam(list(allow))
    assert seam._transport_prefilter(_state(host)) is expected


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("parent", sorted(PARENTS), ids=sorted(PARENTS))
def test_link_local_is_denied_whatever_the_policy_would_have_said(mode, parent):
    """The cloud metadata endpoint carries instance credentials. No mode opens it."""
    seam = _PlaintextSeam(["169.254.169.254"])  # allowlisted AND link-local
    seam._client = _FakeClient(mode)
    seam.sem = LLM  # and agent-semantic, which would otherwise pass every mode

    seam._emit_span(_obj(), _state("169.254.169.254"), _http_txn(PARENTS[parent]))

    assert seam._client.spans == []


@pytest.mark.parametrize("parent", sorted(PARENTS), ids=sorted(PARENTS))
def test_an_allowlisted_host_still_bypasses_the_mode(parent):
    """UNCHANGED, and deliberately so: naming a host by hand outranks the mode.

    Composing this one instead of bypassing it would silently drop traffic the
    user asked for by name, which is why `Prefilter` has an ALLOW at all.
    """
    seam = _PlaintextSeam(["10.0.0.5:8080"])
    seam._client = _FakeClient(CaptureMode.AGENT)

    seam._emit_span(_obj(), _state("10.0.0.5", 8080), _http_txn(PARENTS[parent]))

    assert len(seam._client.spans) == 1


# --------------------------------------------------------------------------
# gate ordering — what a DROP must NOT cost the spans after it
# --------------------------------------------------------------------------


def test_the_gate_does_not_change_which_request_owns_the_connect_cost():
    """UNCHANGED, and it has to stay that way — the gate moved, this did not.

    `_resolve_timing` is destructive: it sets `st.timing_consumed` and pops the
    connection's one record out of the shared store. The fact it encodes is
    "was this the FIRST transaction on this connection", which is exactly what
    `connection_reused` means. So it runs for every transaction that reaches
    `_emit_span`, captured or not.

    Put it below the gate and "first" quietly becomes "first CAPTURED": here
    the health check is dropped, and the LLM call that follows it on the SAME
    keep-alive socket would inherit `tcp_connect_ms=12.5` and
    `connection_reused=False` — a request that reused an established connection
    and paid no connect cost, asserting that it did both. The connect cost of a
    connection whose first request was filtered is simply not attributed to
    anyone, which is the honest answer: nothing is claimed rather than claimed
    of the wrong request.
    """
    install_shared_timing()
    try:
        shared_timing_store().set_connect(4242, 12.5)
        seam = _PlaintextSeam()
        seam._client = _FakeClient(CaptureMode.AGENT)
        st = _state()

        seam.sem = GENERIC  # generic, no ambient span -> dropped
        seam._emit_span(_obj(), st, _http_txn(None, path="/health"))
        assert seam._client.spans == []
        assert st.timing_consumed is True, (
            "a dropped transaction must still own its connection's timing record — "
            "otherwise the next captured span inherits a connect it never paid"
        )

        seam.sem = LLM  # the real call, on the same connection
        seam._emit_span(_obj(), st, _http_txn(None))

        (span,) = seam._client.spans
        assert span.transport.timing.tcp_connect_ms == 0.0
        assert span.transport.timing.tls_handshake_ms == 0.0
        assert span.transport.connection_reused is True
    finally:
        uninstall_shared_timing()


def test_the_first_captured_span_on_a_fresh_connection_still_gets_the_measurement():
    """The other half: nothing was traded away to keep the case above honest."""
    install_shared_timing()
    try:
        shared_timing_store().set_connect(4242, 12.5)
        seam = _PlaintextSeam()
        seam._client = _FakeClient(CaptureMode.AGENT)

        seam.sem = LLM
        seam._emit_span(_obj(), _state(), _http_txn(None))

        (span,) = seam._client.spans
        assert span.transport.timing.tcp_connect_ms == 12.5
        assert span.transport.connection_reused is False
    finally:
        uninstall_shared_timing()


def test_a_dropped_span_releases_its_slot_in_the_capped_timing_store():
    """`ConnTimingStore` is FIFO-capped, and only a `_resolve_timing` pops it.

    A dropped transaction that skipped the pop would leave its connection
    occupying a slot until eviction — and under the AGENT default most
    connections in a process are dropped-only, so the eviction pressure would
    land on the LLM connection whose record is the one that matters.
    """
    install_shared_timing()
    try:
        store = shared_timing_store()
        store.set_connect(4242, 12.5)
        seam = _PlaintextSeam()
        seam._client = _FakeClient(CaptureMode.AGENT)

        seam.sem = GENERIC  # dropped
        seam._emit_span(_obj(), _state(), _http_txn(None, path="/health"))

        assert seam._client.spans == []
        assert store.pop(4242) is None, "a dropped transaction left its slot in the store"
    finally:
        uninstall_shared_timing()


def test_a_captured_span_still_consumes_it_exactly_once():
    """The destructive read happens once per transaction, wherever the gate is."""
    seam = _TlsSeam()
    seam._client = _FakeClient(CaptureMode.ALL)

    seam._emit_span(_obj(), _state(), _http_txn(None))

    assert seam.timing_calls == 1


# --------------------------------------------------------------------------
# the policy itself
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
@pytest.mark.parametrize("parent", sorted(PARENTS), ids=sorted(PARENTS))
def test_agent_semantic_traffic_survives_every_mode(mode, parent):
    """Why `_mcp_stdio` passes a constant `True`, asserted rather than assumed.

    Today `CaptureMode` has two members and both keep agent traffic, so the
    gate at MCP stdio is a CLAIM the site makes about itself, not a filter that
    can drop anything. That is worth pinning: it is the fact that makes routing
    that site behaviour-neutral, and it is the fact that would change the day a
    third mode is added — at which point this test fails and names every site
    whose constant needs re-reading.
    """
    assert should_capture(mode, parent=PARENTS[parent], agent_semantic=True) is True


def test_a_remote_only_parent_does_not_open_the_agent_gate():
    """A service mesh puts `traceparent` on every request in the fleet.

    Treating that as evidence of agent activity would resurrect the firehose
    `AGENT` exists to suppress, so `is_remote` is load-bearing, not cosmetic.
    """
    assert should_capture(CaptureMode.AGENT, parent=REMOTE, agent_semantic=False) is False
    assert should_capture(CaptureMode.AGENT, parent=LOCAL, agent_semantic=False) is True


@pytest.mark.parametrize(
    "client",
    [None, object(), SimpleNamespace(config=None)],
    ids=["none", "not-a-client", "no-config"],
)
def test_a_client_with_no_config_at_all_filters_nothing(client):
    """The no-client case, decided once instead of once per caller.

    The seam spelled it `client is None -> True`; the socket override had no
    such branch at all; `_mcp_stdio` never looked. All three now get the same
    answer, and it is the one the SDK already behaves by: with no configuration
    in play there is no policy to apply, so nothing is filtered.
    """
    assert capture_mode_of(client) is CaptureMode.ALL


@pytest.mark.parametrize(
    "client",
    [
        SimpleNamespace(config=SimpleNamespace()),
        SimpleNamespace(config=SimpleNamespace(capture_mode=None)),
        SimpleNamespace(config=SimpleNamespace(capture_mode="agent")),  # a str, not the enum
        SimpleNamespace(config=SimpleNamespace(capture_mode="nonsense")),
    ],
    ids=["no-mode", "mode-is-none", "mode-is-a-str", "mode-is-garbage"],
)
def test_a_config_whose_mode_cannot_be_read_narrows_instead_of_widening(client):
    """An unreadable policy must not become the widest policy.

    `WardexConfig` does not validate this field, and the enum's values ARE the
    strings the README prose uses, so `capture_mode="agent"` is accepted in
    silence. Answering `ALL` there would hand a user who explicitly asked for
    filtering the pre-Phase-4 firehose — every intercepted request and response
    body on both byte seams, exported. §5.1's fail-open rule covers wardex's own
    failures; it does not license widening a policy the user set.

    This is also what the code `_policy` replaced already did: the seam asked
    `mode is CaptureMode.ALL`, which is False for a str or a None, so the AGENT
    clauses ran. Only "there is no config object at all" ever reached `True`.
    """
    assert capture_mode_of(client) is CaptureMode.AGENT


@pytest.mark.parametrize("mode", MODES, ids=lambda m: m.value)
def test_a_configured_client_reports_its_own_mode(mode):
    assert capture_mode_of(_FakeClient(mode)) is mode


# ==========================================================================
# A run wardex failed to open is not a run that never happened
# ==========================================================================
#
# The gate's premise is that an absent LOCAL parent means "this traffic is not
# agent work". When WARDEX is what lost the parent, that inference is wrong —
# and acting on it turns one bug at the top of a run into total silence
# underneath: every request and every tool call inside the host's block dropped
# at the seam, with no counter and no marker, so the run looks like it never
# happened rather than like one wardex could not follow.


def test_traffic_inside_a_run_wardex_failed_to_open_is_no_longer_dropped():
    """§5.1's rule is that noise is filterable and lost data is not. This is the
    one place it was being applied backwards — against the user, on wardex's own
    fault."""
    assert should_capture(CaptureMode.AGENT, parent=None, agent_semantic=False) is False
    assert (
        should_capture(CaptureMode.AGENT, parent=None, agent_semantic=False, degraded=True) is True
    )


def test_the_flag_is_a_declared_input_and_not_a_hidden_read():
    """`should_capture` stays a pure function of its arguments — testable by
    them, unable to raise on them. A `ContextVar` read inside it would make the
    gate's answer depend on a carrier its own tests cannot see, which is exactly
    the property this module's docstring is built on.
    """
    import inspect
    import pathlib

    from wardex_sdk._assembly import _parentage, _policy

    assert "degraded" in inspect.signature(should_capture).parameters

    # Read off the SHIPPED FILE, with the docstring cut away: a docstring that
    # names the flag is documentation, a body that reads it is the thing being
    # forbidden — and asking the live object would be asking whatever a test
    # happened to patch.
    src = pathlib.Path(_policy.__file__).read_text()
    body = src[src.index("def should_capture") :]
    body = body[body.index('"""', body.index('"""') + 3) :]
    assert "in_degraded_run" not in body
    assert "ContextVar" not in body
    # And the flag itself lives with parentage, because that is what it is about.
    assert hasattr(_parentage, "in_degraded_run")


def test_the_flag_widens_nothing_a_healthy_run_would_have_been_denied():
    """It answers ONE question — "is the missing parent wardex's doing" — and a
    healthy process never asks it. Every other row of the matrix is untouched.
    """
    for mode in (CaptureMode.AGENT, CaptureMode.ALL):
        for parent in (None, LOCAL, REMOTE):
            for sem in (True, False):
                plain = should_capture(mode, parent=parent, agent_semantic=sem)
                if plain:
                    assert should_capture(mode, parent=parent, agent_semantic=sem, degraded=True)


def test_the_flag_lasts_exactly_as_long_as_the_block_it_was_set_for():
    from wardex_sdk._assembly import degraded_run, in_degraded_run

    assert in_degraded_run() is False
    with degraded_run():
        assert in_degraded_run() is True
        with degraded_run():
            assert in_degraded_run() is True
        assert in_degraded_run() is True
    assert in_degraded_run() is False


def test_a_span_that_survived_a_degraded_run_does_not_ship_as_a_trace_root():
    """Capturing it is half the fix; the other half is that it must not pass for
    a legitimate root.

    A trace root at confidence 1.0 with no marker is the shattered-run shape —
    one run arriving as several, indistinguishable from genuine ones, silently
    inflating the trace count. Shipped this way it is one row a consumer can
    filter, count and file a bug about: the parent was expected
    (`parent_unresolved`) and wardex is why it is missing
    (`instrumentation_degraded`).
    """
    from wardex_sdk._assembly import (
        EMPTY_AMBIENT,
        Limitation,
        ParentSource,
        degraded_run,
        resolve_observed,
    )

    healthy = resolve_observed(EMPTY_AMBIENT)
    assert healthy.correlation.strategy is ParentSource.TRACE_ROOT
    assert healthy.correlation.confidence == 1.0
    assert healthy.limitations == ()

    with degraded_run():
        broken = resolve_observed(EMPTY_AMBIENT)
    assert broken.correlation.strategy is ParentSource.UNRESOLVED
    assert broken.correlation.confidence == 0.0
    assert set(broken.limitations) == {
        Limitation.PARENT_UNRESOLVED,
        Limitation.INSTRUMENTATION_DEGRADED,
    }


def test_a_real_parent_is_still_the_parent_inside_a_degraded_run():
    """The flag describes the ABSENCE of a parent, never a present one. A nested
    site that degraded while its session is still live has a real edge, and
    downgrading it would turn a wardex bug into a worse tree than the one it
    caused.
    """
    from wardex_sdk._assembly import Ambient, ParentSource, degraded_run, resolve_observed

    ambient = Ambient(span_context=LOCAL, conversation=None, tracestate=None)
    with degraded_run():
        p = resolve_observed(ambient)
    assert p.correlation.strategy is ParentSource.CONTEXTVAR
    assert p.correlation.confidence == 1.0
    assert p.limitations == ()
    assert p.parent_span_id == LOCAL.span_id


def test_the_edges_markers_reach_the_span_without_the_caller_copying_them():
    """`SpanDraft` is built FROM a parentage, so an interpreted edge arrives
    already knowing what it is. Leaving each of the six parentage sites to copy
    the markers across is how a span ships confidence below 1.0 with an EMPTY
    limitation list — half of I4 missing, and the half a dashboard renders.
    Two sites remembered; the rest did not.
    """
    from wardex_sdk._assembly import (
        EMPTY_AMBIENT,
        Limitation,
        SpanDraft,
        TransportLabel,
        degraded_run,
        resolve_observed,
    )
    from wardex_sdk._enums import CaptureSource, SpanKind

    with degraded_run():
        p = resolve_observed(EMPTY_AMBIENT)
    span = SpanDraft.transport(
        p,
        label=TransportLabel.HTTP,
        subject="/v1/messages",
        source=CaptureSource.SSL,
        start_ns=1,
        kind=SpanKind.CLIENT,
    ).finish(2)

    assert span.capture_integrity is not None, "the edge's markers never reached the span"
    assert set(span.capture_integrity.limitations) == {
        Limitation.PARENT_UNRESOLVED,
        Limitation.INSTRUMENTATION_DEGRADED,
    }


# --------------------------------------------------------------------------
# design §10.3(b) — a parent whose unit had already closed opens nothing
# --------------------------------------------------------------------------


@contextmanager
def _stranded_session():
    """A unit whose span has SHIPPED and whose `activate()` fork still stands.

    Closed from another thread, so the ContextVar Token cannot be reset — the
    `close_units()`-mid-stream / finalized-on-a-foreign-carrier shape. The
    activation CM is held for the duration: `activate_span` is generator-backed,
    so dropping the last reference runs its `finally` and takes the fork down,
    which would silently turn every test below into the clean case.
    """
    import threading

    from wardex_sdk._assembly import (
        EMPTY_AMBIENT,
        SpanIntent,
        UnitKey,
        UnitKind,
        UnitRegistry,
        counters,
    )
    from wardex_sdk._assembly._units import _ambient_unit
    from wardex_sdk._hub import reset_for_test
    from wardex_sdk._types import AgentAttributes

    class _Sink:
        def emit(self, draft: Any, *, agent_semantic: bool) -> bool:
            return True

    token = _ambient_unit.set(None)
    counters.reset()
    reg = UnitRegistry(sink=_Sink())
    unit = reg.open(
        UnitKind.SESSION,
        UnitKey("test.session", "s1"),
        ambient=EMPTY_AMBIENT,
        intent=SpanIntent.INVOKE_AGENT,
        subject="agent",
    )
    unit.draft.set_agent(AgentAttributes(name="agent", id="s1"))
    cm = unit.activate()
    cm.__enter__()
    closer = threading.Thread(target=lambda: reg.close(unit))
    closer.start()
    closer.join()
    try:
        yield reg, unit
    finally:
        del cm
        _ambient_unit.reset(token)
        reset_for_test()
        counters.reset()


def test_the_gate_closes_on_a_local_parent_whose_unit_already_closed():
    """AGENT's premise is that a local ambient span means agent work is in
    flight RIGHT NOW. A leftover activation fork breaks that premise without
    breaking the type — the span is local, is not remote, and is finished — so
    every later request on that carrier walked the gate on the strength of a
    span that was over. Closing it here loses nothing the mode wanted: it
    restores the answer the mode would have given had the fork come down.
    """
    assert should_capture(CaptureMode.AGENT, parent=LOCAL, agent_semantic=False) is True
    assert (
        should_capture(CaptureMode.AGENT, parent=LOCAL, agent_semantic=False, parent_closed=True)
        is False
    )
    # Every other row is untouched.
    assert (
        should_capture(CaptureMode.ALL, parent=LOCAL, agent_semantic=False, parent_closed=True)
        is True
    )
    assert (
        should_capture(
            CaptureMode.AGENT, parent=LOCAL, agent_semantic=False, parent_closed=True, degraded=True
        )
        is True
    )
    assert (
        should_capture(CaptureMode.AGENT, parent=None, agent_semantic=False, parent_closed=True)
        is False
    )


def test_agent_semantic_traffic_survives_a_closed_parent():
    """It loses a PARENT, not its reason to exist. `agent_semantic` is a claim
    the site makes about the BYTES — a parsed LLM response, a JSON-RPC tool call
    over a subprocess pipe — and a dead parent says nothing about those. The
    span still ships; `resolve_observed` is what makes its edge honest.
    """
    for mode in (CaptureMode.AGENT, CaptureMode.ALL):
        assert (
            should_capture(mode, parent=LOCAL, agent_semantic=True, parent_closed=True) is True
        ), mode


def test_the_closed_parent_flag_is_a_declared_input_and_not_a_hidden_read():
    """The twin of `test_the_flag_is_a_declared_input_and_not_a_hidden_read`.

    `should_capture` stays a pure function of its arguments. Reading the
    `_ambient_unit` carrier here would be worse than for `degraded`: the answer
    is not even knowable on this side, because the gate runs on the response
    path of a seam that shares neither the carrier nor the instant the work was
    issued on.
    """
    import inspect
    import pathlib

    from wardex_sdk._assembly import _policy, _units

    assert "parent_closed" in inspect.signature(should_capture).parameters

    src = pathlib.Path(_policy.__file__).read_text()
    body = src[src.index("def should_capture") :]
    body = body[body.index('"""', body.index('"""') + 3) :]
    assert "parent_is_closed_unit" not in body
    assert "ContextVar" not in body
    # And the fact itself lives with the units, because that is what it is about.
    assert hasattr(_units, "parent_is_closed_unit")


def test_an_observed_span_under_a_closed_parent_is_unresolved_not_a_child():
    """A shipped span adopting later traffic at 1.0 with no marker is the one
    shape no consumer can detect downstream. Refused, the edge is `UNRESOLVED`
    and not `TRACE_ROOT`: a parent was EXPECTED here — one was latched — so
    `PARENT_UNRESOLVED` is the literal truth. It is not
    `INSTRUMENTATION_DEGRADED` either; nothing failed to open.
    """
    from wardex_sdk._assembly import Ambient, Limitation, ParentSource, resolve_observed

    ambient = Ambient(span_context=LOCAL, conversation=None, tracestate=None)

    kept = resolve_observed(ambient)
    assert kept.parent_span_id == LOCAL.span_id
    assert kept.limitations == ()

    refused = resolve_observed(ambient, parent_closed=True)
    assert refused.parent_span_id is None
    assert refused.trace_id != LOCAL.trace_id
    assert refused.correlation.strategy is ParentSource.UNRESOLVED
    assert refused.correlation.confidence == 0.0
    assert Limitation.PARENT_UNRESOLVED in refused.limitations
    assert Limitation.INSTRUMENTATION_DEGRADED not in refused.limitations


def test_an_observed_span_whose_parent_wardex_evicted_says_whose_fault_it_is():
    """The sibling of the row above, one argument along.

    `parent_evicted` is the byte seam's own bound admitting it dropped a latched
    parent (`_interceptors/_trackers.py`, the h2 stream latch at `max_streams`).
    Shipped as a trace root that span is indistinguishable from one the host
    genuinely issued outside any agent work, so it takes `PARENT_UNRESOLVED` for
    the missing parent and `INSTRUMENTATION_DEGRADED` for whose doing that was —
    which is what separates it from the `parent_closed` row above.

    The ambient is refused in CODE and not by asking callers nicely: a non-empty
    one flowing through would take `resolve_parentage`'s join branch and ship a
    span that both adopts a parent and declares it has none.
    """
    from wardex_sdk._assembly import (
        EMPTY_AMBIENT,
        Ambient,
        Limitation,
        ParentSource,
        resolve_observed,
    )

    evicted = resolve_observed(EMPTY_AMBIENT, parent_evicted=True)
    assert evicted.parent_span_id is None
    assert evicted.correlation.strategy is ParentSource.UNRESOLVED
    assert evicted.correlation.confidence == 0.0
    assert Limitation.PARENT_UNRESOLVED in evicted.limitations
    assert Limitation.INSTRUMENTATION_DEGRADED in evicted.limitations

    stray = resolve_observed(
        Ambient(span_context=LOCAL, conversation=None, tracestate=None), parent_evicted=True
    )
    assert stray.parent_span_id is None, "an evicted edge must never adopt the ambient beside it"
    assert stray.correlation.strategy is ParentSource.UNRESOLVED


def test_an_evicted_parent_opens_the_agent_gate_the_way_a_degraded_run_does():
    """Both producers of `degraded` reach the same clause, and they must.

    The gate reads an absent parent as "not agent work". Under the declared
    default that drops the span before anything can say why the parent is
    missing — so the marker the row above attaches would be unreachable on
    exactly the traffic that earned it, and the bound would have introduced a
    silent drop instead of a labelled one.
    """
    assert should_capture(CaptureMode.AGENT, parent=None, agent_semantic=False) is False
    assert (
        should_capture(CaptureMode.AGENT, parent=None, agent_semantic=False, degraded=True) is True
    )


def test_the_seam_asks_about_unit_liveness_when_it_latches_not_when_it_emits():
    """The timing is the whole correctness argument, driven through the REAL
    `_Http1Tracker`.

    A request issued at t1 inside a live session whose unit closes at t2, before
    the reply lands, is a perfectly good child of that session's shipped span —
    the codebase already says so (`open()`'s `parent_closed` branch). An
    emit-time read would refuse it and invent a spurious second trace for work
    that genuinely happened inside the run: real data lost to escape a corpse
    that was not in front of the request when it left.
    """
    import threading

    from wardex_sdk._interceptors._trackers import _Http1Tracker

    with _stranded_session() as (reg, dead):
        after = _Http1Tracker()
        after.on_request_bytes(b"GET /after HTTP/1.1\r\nHost: h\r\n\r\n")
        (txn,) = after.on_response_bytes(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")

        assert txn.parent == dead.context, "the raw carrier still hands back the corpse"
        assert txn.parent_closed is True
        assert (
            should_capture(
                CaptureMode.AGENT,
                parent=txn.parent,
                agent_semantic=False,
                parent_closed=txn.parent_closed,
            )
            is False
        )

    # And the in-flight half: latched while LIVE, answered after the close.
    from wardex_sdk._assembly import (
        EMPTY_AMBIENT,
        SpanIntent,
        UnitKey,
        UnitKind,
        UnitRegistry,
    )
    from wardex_sdk._assembly._units import _ambient_unit
    from wardex_sdk._hub import reset_for_test
    from wardex_sdk._types import AgentAttributes

    class _Sink:
        def emit(self, draft: Any, *, agent_semantic: bool) -> bool:
            return True

    token = _ambient_unit.set(None)
    try:
        reg = UnitRegistry(sink=_Sink())
        unit = reg.open(
            UnitKind.SESSION,
            UnitKey("test.session", "s2"),
            ambient=EMPTY_AMBIENT,
            intent=SpanIntent.INVOKE_AGENT,
            subject="agent",
        )
        unit.draft.set_agent(AgentAttributes(name="agent", id="s2"))
        cm = unit.activate()
        cm.__enter__()
        inflight = _Http1Tracker()
        inflight.on_request_bytes(b"GET /inflight HTTP/1.1\r\nHost: h\r\n\r\n")
        closer = threading.Thread(target=lambda: reg.close(unit))
        closer.start()
        closer.join()
        (txn,) = inflight.on_response_bytes(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n")

        assert txn.parent == unit.context
        assert txn.parent_closed is False, "issued while live — it keeps its parent"
        assert (
            should_capture(
                CaptureMode.AGENT,
                parent=txn.parent,
                agent_semantic=False,
                parent_closed=txn.parent_closed,
            )
            is True
        )
        del cm
    finally:
        _ambient_unit.reset(token)
        reset_for_test()


def test_mcp_stdio_does_not_parent_a_tool_call_into_a_finished_run():
    """The route that cannot fall back on the gate: it claims
    `agent_semantic=True` unconditionally, so the gate is open by construction
    and the EDGE is the only thing left to get right. The span still ships — it
    just stops being a silent child of a session that had already ended.
    """
    from wardex_sdk._assembly import Limitation, ParentSource

    with _stranded_session() as (reg, dead):
        state = _ProcState(mode=CaptureMode.AGENT)
        state.feed_request(
            b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"t"}}\n'
        )
        (pending,) = list(state._latch.values())
        assert pending.ambient.span_context == dead.context
        assert pending.parent_closed is True

        spans = state.feed_response(b'{"jsonrpc":"2.0","id":1,"result":{"content":[]}}\n')

    assert len(spans) == 1, "agent_semantic=True keeps the span; only its PARENT changes"
    span = spans[0]
    assert span.parent_span_id is None
    assert span.context.trace_id != dead.context.trace_id
    assert span.correlation.strategy is ParentSource.UNRESOLVED
    assert Limitation.PARENT_UNRESOLVED in span.capture_integrity.limitations
