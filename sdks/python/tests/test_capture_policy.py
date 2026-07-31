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
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import CaptureMode
from wardex_sdk._types import SpanContext, SpanId, TraceId
from wardex_sdk.assembly import Prefilter, capture_mode_of, should_capture
from wardex_sdk.context._contextvar import fork_active_span
from wardex_sdk.interceptors import _seam
from wardex_sdk.interceptors._conn_timing import (
    install_shared_timing,
    shared_timing_store,
    uninstall_shared_timing,
)
from wardex_sdk.interceptors._mcp_stdio import _ProcState
from wardex_sdk.interceptors._seam import ByteSeamInterceptor, _ConnectionState
from wardex_sdk.interceptors._socket import RawSocketInterceptor
from wardex_sdk.interceptors._trackers import _Txn


def _ctx(*, remote: bool) -> SpanContext:
    return SpanContext(trace_id=TraceId.generate(), span_id=SpanId.generate(), is_remote=remote)


LOCAL = _ctx(remote=False)
REMOTE = _ctx(remote=True)
PARENTS = {"no-parent": None, "local": LOCAL, "remote": REMOTE}
MODES = list(CaptureMode)


class _FakeClient:
    def __init__(self, mode: CaptureMode) -> None:
        self.config = WardexConfig(api_key="k", capture_mode=mode)
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
