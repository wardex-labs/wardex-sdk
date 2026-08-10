"""Opt-in traceparent injection into HTTP client libraries."""

import http.server
import threading

import httpx
import pytest

import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, PropagationConfig, WardexConfig
from wardex_sdk._tracing import span, trace
from wardex_sdk._types import InternalEnvelope
from wardex_sdk.context._inject import (
    _build_inject_headers,
    install_propagation,
    uninstall_propagation,
)
from wardex_sdk.transport._base import Transport


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _setup(**cfg):
    _hub.reset_for_test()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k"), **cfg), _Recording()))


@pytest.fixture(autouse=True)
def _teardown_patches():
    yield
    uninstall_propagation()


def _capture_request(**client_kwargs) -> httpx.Request:
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["req"] = request
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(handler), **client_kwargs) as c:
        c.get("https://api.mycorp.com/x")
    return seen["req"]


def test_injects_traceparent_inside_span():
    _setup(propagation=PropagationConfig(enabled=True))
    install_propagation()
    with trace("root") as root:
        req = _capture_request()
    assert req.headers["traceparent"].split("-")[1] == root.context.trace_id.hex()


def test_no_injection_outside_any_context():
    _setup(propagation=PropagationConfig(enabled=True))
    install_propagation()
    req = _capture_request()
    assert "traceparent" not in req.headers


def test_no_injection_when_flag_off():
    _setup(propagation=PropagationConfig(enabled=False))
    install_propagation()  # wiring guards on the flag too, but the header path must also guard
    with trace("root"):
        req = _capture_request()
    assert "traceparent" not in req.headers


def test_user_header_never_overwritten():
    _setup(propagation=PropagationConfig(enabled=True))
    install_propagation()
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["tp"] = request.headers["traceparent"]
        return httpx.Response(200)

    with trace("root"):
        with httpx.Client(transport=httpx.MockTransport(handler)) as c:
            c.get("https://api.mycorp.com/x", headers={"traceparent": "user-set"})
    assert seen["tp"] == "user-set"


def test_targets_allowlist():
    _setup(propagation=PropagationConfig(enabled=True, targets=("*.mycorp.com",)))
    install_propagation()
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[request.url.host] = "traceparent" in request.headers
        return httpx.Response(200)

    with trace("root"):
        with httpx.Client(transport=httpx.MockTransport(handler)) as c:
            c.get("https://api.mycorp.com/x")
            c.get("https://api.openai.com/v1/chat")
    assert seen["api.mycorp.com"] is True
    assert seen["api.openai.com"] is False


def test_targets_match_case_insensitively():
    """Hostnames are case-insensitive; `fnmatch` is case-insensitive on Windows.

    Both halves of that sentence are the bug. Plain `fnmatch` defers to
    `os.path.normcase`, so the same allowlist admitted `API.MyCorp.com` on one
    developer's machine and refused it on the next — a propagation gap only one
    operating system can reproduce. Asserted at `_build_inject_headers` rather
    than through a client because httpx lowercases the host during URL
    normalization and would hide the host side of the fold entirely.

    BOTH folds are the injector's now: the pattern fold moved here from the
    config so that `targets` round-trips as written (see the test below).
    """
    _setup(propagation=PropagationConfig(enabled=True, targets=("*.MyCorp.com",)))
    with trace("root"):
        assert _build_inject_headers("api.mycorp.com") != {}  # pattern folded
        assert _build_inject_headers("API.MyCorp.COM") != {}  # host folded
        assert _build_inject_headers("api.othercorp.com") == {}  # still an allowlist


def test_targets_round_trip_verbatim_and_still_match_folded():
    """The fold's move out of the config, both halves at once: the capitals a user typed are the
    capitals they read back — the config never folds — AND the mixed-case
    pattern still admits a lowercase host, because the injector folds both
    sides of the match itself (once per configured allowlist, not per
    request). A fix that kept only the first half would be a propagation gap;
    only the second, a config that lies about itself.
    """
    _setup(propagation=PropagationConfig(enabled=True, targets=("*.MyCorp.com",)))
    client = _hub.get_client()
    assert client.config.propagation.targets == ("*.MyCorp.com",)  # verbatim readback
    with trace("root"):
        assert _build_inject_headers("api.mycorp.com") != {}  # and it still matches


def test_async_client_injects():
    import asyncio

    _setup(propagation=PropagationConfig(enabled=True))
    install_propagation()
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["has"] = "traceparent" in request.headers
        return httpx.Response(200)

    async def main():
        with trace("root"):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                await c.get("https://api.mycorp.com/x")

    asyncio.run(main())
    assert seen["has"] is True


def test_install_uninstall_idempotent():
    _setup(propagation=PropagationConfig(enabled=True))
    orig = httpx.Client.send
    install_propagation()
    install_propagation()
    uninstall_propagation()
    uninstall_propagation()
    assert httpx.Client.send is orig


class _HeaderEcho(http.server.BaseHTTPRequestHandler):
    """Records every header of every request, duplicates included.

    One recorder and not two: the previous pair kept `dict(self.headers)`
    alongside a separate `get_all("x-multi")` list, so the only header whose
    repetitions were visible was the one a test had thought to name in advance.
    Duplicate suppression is precisely what the injection rules below are
    about, so the raw pair list is what gets stored and `_sent()` answers both
    questions off it.
    """

    raw: list[list[tuple[str, str]]] = []

    def do_GET(self):
        _HeaderEcho.raw.append(list(self.headers.items()))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


def _sent(name: str) -> list[str]:
    """Every value of `name` on the last request received, in wire order."""
    lowered = name.lower()
    return [v for k, v in _HeaderEcho.raw[-1] if k.lower() == lowered]


@pytest.fixture()
def echo_server():
    """A loopback echo server that is fully gone when the test is.

    `shutdown()` alone stops the accept loop and leaves both the listening
    socket and the serving thread behind. Per test that is one descriptor and
    one thread, and the first symptom is never a failure here — it is an
    unrelated file, later in the session, that cannot open a socket. So the
    teardown is the full three: stop the loop, join the thread that ran it,
    then close the socket, and in a `finally` so a failing test tears down as
    completely as a passing one.
    """
    _HeaderEcho.raw = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _HeaderEcho)
    th = threading.Thread(target=srv.serve_forever, name="wardex-test-echo", daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}"
    finally:
        srv.shutdown()
        th.join(timeout=5)
        srv.server_close()
        _HeaderEcho.raw = []


#: The three patched libraries, driven through one signature so that every rule
#: below is asserted against all of them. The rules ARE the same rule — see
#: `_inject._headers_to_add` — and a per-library test that only exists for one
#: library is how they came to differ in the first place.
LIBRARIES = ("httpx", "requests", "aiohttp")


def _get(library: str, url: str, *, headers=None, session_headers=None) -> None:
    """One GET through `library`, with optional per-request and session headers."""
    if library == "httpx":
        with httpx.Client(headers=session_headers, timeout=5) as c:
            c.get(url, headers=headers)
    elif library == "requests":
        import requests

        with requests.Session() as s:
            if session_headers:
                s.headers.update(session_headers)
            s.get(url, headers=headers, timeout=5)
    else:
        import asyncio

        import aiohttp

        async def main():
            async with aiohttp.ClientSession(headers=session_headers) as s:
                async with s.get(url, headers=headers) as resp:
                    await resp.read()

        asyncio.run(main())


@pytest.mark.parametrize("library", LIBRARIES)
def test_injects_on_every_patched_library(library, echo_server):
    _setup(propagation=PropagationConfig(enabled=True))
    install_propagation()
    with trace("root") as root:
        _get(library, f"{echo_server}/x")
    assert _sent("traceparent") == [
        f"00-{root.context.trace_id.hex()}-{root.context.span_id.hex()}-01"
    ]


@pytest.mark.parametrize("library", LIBRARIES)
def test_tracestate_forwarded_verbatim(library, echo_server):
    tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    _setup(propagation=PropagationConfig(enabled=True))
    install_propagation()
    with wardex_sdk.continue_trace({"traceparent": tp, "tracestate": "dd=s:1"}):
        with trace("root"):
            _get(library, f"{echo_server}/x")
    assert _sent("tracestate") == ["dd=s:1"]


@pytest.mark.parametrize("library", LIBRARIES)
def test_a_non_ascii_inbound_tracestate_never_reaches_the_wire(library, echo_server):
    """A byte a remote peer chose must not raise into the host's own call.

    The inbound tracestate is re-emitted outbound, and `http.client` encodes
    header values as latin-1 — so a character above it came back as
    `UnicodeEncodeError` from `putheader`, outside the injector's guard and
    straight into the caller. Fail-silence is this module's whole contract, and
    a remote peer must not be able to spend it. Driven over a real socket
    rather than a mock transport, because the encode is what is on trial.
    """
    tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    _setup(propagation=PropagationConfig(enabled=True))
    install_propagation()
    with wardex_sdk.continue_trace({"traceparent": tp, "tracestate": "ja=安全"}):
        with trace("root"):
            _get(library, f"{echo_server}/x")
    assert _sent("tracestate") == []  # dropped at the edge, not forwarded
    assert len(_sent("traceparent")) == 1  # one bad header is not two


@pytest.mark.parametrize("library", LIBRARIES)
def test_a_session_default_traceparent_wins_over_injection(library, echo_server):
    """A header the host set on the SESSION is a header the host set.

    aiohttp is why this is parametrized. It merges session defaults with
    per-request headers AFTER the patch runs, and a per-request header wins
    there — so a predicate that consulted only the per-request mapping saw no
    traceparent, wrote one, and outranked the default the host had set once for
    every call on that session. httpx and requests merge before the patch and
    were always safe; nothing about that is visible from inside the injector,
    which is why all three are asked the same question here.
    """
    _setup(propagation=PropagationConfig(enabled=True))
    install_propagation()
    with trace("root"):
        _get(library, f"{echo_server}/x", session_headers={"traceparent": "session-default"})
    assert _sent("traceparent") == ["session-default"]


@pytest.mark.parametrize("library", LIBRARIES)
def test_a_per_request_traceparent_wins_over_injection(library, echo_server):
    _setup(propagation=PropagationConfig(enabled=True))
    install_propagation()
    with trace("root"):
        _get(library, f"{echo_server}/x", headers={"traceparent": "request-set"})
    assert _sent("traceparent") == ["request-set"]


@pytest.mark.parametrize("library", LIBRARIES)
def test_a_callers_tracestate_is_neither_replaced_nor_duplicated(library, echo_server):
    """The one rule, where the three libraries used to give three answers.

    A caller with a `tracestate` and no `traceparent` used to get: httpx and
    requests silently REPLACING the value (assignment into a case-insensitive
    mapping), and aiohttp sending BOTH on the wire (`extend` on a CIMultiDict)
    — one request with two tracestate headers, a shape the spec defines no
    reading for. Now all three add the traceparent the caller is missing and
    leave the caller's tracestate exactly as it was written.

    Adding the traceparent — rather than declining the whole injection the way
    a caller-set traceparent does — is the deliberate half. A tracestate with
    no traceparent is a header no conformant receiver can act on, so declining
    would break the trace link to protect something already inert. The rule and
    its reasoning are in `_headers_to_add`.
    """
    tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    _setup(propagation=PropagationConfig(enabled=True))
    install_propagation()
    with wardex_sdk.continue_trace({"traceparent": tp, "tracestate": "wardex=ours"}):
        with trace("root"):
            _get(library, f"{echo_server}/x", headers={"tracestate": "caller=theirs"})
    assert _sent("tracestate") == ["caller=theirs"]
    assert len(_sent("traceparent")) == 1


def test_aiohttp_preserves_duplicate_headers(echo_server):
    import asyncio

    import aiohttp

    _setup(propagation=PropagationConfig(enabled=True))
    install_propagation()

    async def main():
        with trace("root"):
            async with aiohttp.ClientSession() as s:
                headers = [("x-multi", "a"), ("x-multi", "b")]
                async with s.get(f"{echo_server}/x", headers=headers) as resp:
                    await resp.read()

    asyncio.run(main())
    assert _sent("x-multi") == ["a", "b"]
    assert len(_sent("traceparent")) == 1


def test_init_wires_propagation_and_close_unwires():
    import httpx

    _hub.reset_for_test()
    orig = httpx.Client.send
    wardex_sdk.init(intercept=False, propagation=PropagationConfig(enabled=True))
    assert httpx.Client.send is not orig
    wardex_sdk.close()
    assert httpx.Client.send is orig


def test_init_without_flag_does_not_patch():
    import httpx

    _hub.reset_for_test()
    orig = httpx.Client.send
    wardex_sdk.init(intercept=False)
    assert httpx.Client.send is orig
    wardex_sdk.close()


def _patched_attributes() -> dict[str, object]:
    """The four attributes this module patches, as they stand right now."""
    import aiohttp
    import requests

    return {
        "httpx.Client.send": httpx.Client.send,
        "httpx.AsyncClient.send": httpx.AsyncClient.send,
        "requests.Session.send": requests.Session.send,
        "aiohttp.ClientSession._request": aiohttp.ClientSession._request,
    }


def test_install_is_serialized_across_threads(monkeypatch):
    """The per-library idempotence check is only real under a lock.

    `"httpx" in _installed` and `_installed.add("httpx")` sit either side of
    three attribute swaps. Two threads that arrive between them both read
    False and both patch, and the second captures the FIRST one's wrapper as
    its original — a two-deep stack that one `uninstall_propagation()` cannot
    unwind, leaving the host patched by a wardex that believes it has left.

    Asserted by holding the install open from inside and watching a second
    thread block, rather than by racing and hoping: a timing test that passes
    when the bug is present is not a test.
    """
    from wardex_sdk.context import _inject

    _setup(propagation=PropagationConfig(enabled=True))
    inside = threading.Event()
    release = threading.Event()
    real_install_requests = _inject._install_requests

    def slow_install_requests(module):
        # Only the FIRST caller stalls. A stub that stalled every caller would
        # hold the second thread up by itself and report "serialized" whether
        # the lock existed or not.
        if not inside.is_set():
            inside.set()
            release.wait(timeout=5)
        real_install_requests(module)

    monkeypatch.setattr(_inject, "_install_requests", slow_install_requests)
    first = threading.Thread(target=install_propagation)
    second = threading.Thread(target=install_propagation)
    try:
        first.start()
        assert inside.wait(timeout=5)
        second.start()
        second.join(timeout=0.5)
        assert second.is_alive(), "a second install ran while the first held the lock"
    finally:
        release.set()
        first.join(timeout=5)
        second.join(timeout=5)
    assert not first.is_alive() and not second.is_alive()


def test_a_close_re_entering_an_install_neither_hangs_nor_outlives_itself(monkeypatch):
    """A signal handler landing on the thread that is inside the install.

    wardex chains to the application's previous SIGINT/SIGTERM handler, so an
    ordinary shutdown handler that calls `close()` runs `uninstall_propagation`
    on the thread already holding the install lock — the same-thread reentry
    every other lock on this SDK's teardown path is an RLock for. With a plain
    `Lock` this hangs the process at Ctrl-C, with no way out.

    Re-entered from inside an `_install_*` step rather than by raising a real
    signal: that is the same stack without asking the test runner for a SIGINT.
    The lock is swapped for a fresh instance OF THE MODULE'S OWN TYPE so that a
    regression fails this test instead of stranding the real lock and hanging
    every test after it.

    The second assertion is the half an RLock does not give on its own. The
    reentrant teardown is the LATER decision, so the install underneath it must
    unwind rather than carry on patching for a wardex that has already left.
    """
    from wardex_sdk.context import _inject

    _setup(propagation=PropagationConfig(enabled=True))
    before = _patched_attributes()
    monkeypatch.setattr(_inject, "_install_lock", type(_inject._install_lock)())
    real_install_requests = _inject._install_requests

    def closing_install_requests(module):
        real_install_requests(module)
        uninstall_propagation()  # the host's signal handler, on this thread

    monkeypatch.setattr(_inject, "_install_requests", closing_install_requests)
    done = threading.Event()

    def run():
        install_propagation()
        done.set()

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    assert done.wait(timeout=10), "install_propagation deadlocked on a re-entrant close"
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert _patched_attributes() == before
    assert _inject._installed == set()


def test_concurrent_installs_leave_exactly_one_layer():
    """Whatever the interleaving, ONE uninstall must put everything back."""
    _setup(propagation=PropagationConfig(enabled=True))
    before = _patched_attributes()
    barrier = threading.Barrier(8)

    def racer():
        barrier.wait(timeout=5)
        install_propagation()

    # A thread still running past its join is a patch installed after the
    # assertions below — the leak this whole file is about, on the thread side.
    threads = [threading.Thread(target=racer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive(), "an installing thread never finished"
    assert all(v is not before[k] for k, v in _patched_attributes().items())
    uninstall_propagation()
    assert _patched_attributes() == before


def test_concurrent_init_does_not_stack_propagation_patches():
    """The same guarantee through the public door, where hosts actually race.

    Two frameworks each calling `init()` on their own startup thread is not
    exotic, and the runtime serializes them — but the property being asserted
    belongs to the injector, so it is asserted against the injector's
    attributes rather than against the runtime's lock.

    Every join is checked, not just waited on. A straggler here is an `init()`
    that installs its patches AFTER the `close()` below and after the autouse
    teardown has uninstalled — three libraries left patched by a wardex the
    module believes has gone, for the rest of the session, with the failure
    surfacing in some unrelated test much later.
    """
    _hub.reset_for_test()
    before = _patched_attributes()
    barrier = threading.Barrier(4)

    def racer():
        barrier.wait(timeout=5)
        wardex_sdk.init(intercept=False, propagation=PropagationConfig(enabled=True))

    threads = [threading.Thread(target=racer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive(), "an init() thread never finished"
    assert all(v is not before[k] for k, v in _patched_attributes().items())
    wardex_sdk.close()
    assert _patched_attributes() == before


def test_a_request_in_flight_while_the_sdk_closes_neither_crashes_nor_leaks():
    """The drop cell: traffic is live at the instant wardex is torn down.

    `close()` runs from inside the transport, so the patch is removed while a
    request that already went through it is still on the way out. Two things
    have to hold: the host's call completes normally, and nothing of wardex is
    left in the call path afterwards.
    """
    _hub.reset_for_test()
    before = _patched_attributes()
    wardex_sdk.init(intercept=False, propagation=PropagationConfig(enabled=True))
    seen: list[str | None] = []

    def closing_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("traceparent"))
        wardex_sdk.close()  # the host tears wardex down mid-request
        return httpx.Response(200)

    with trace("root"):
        with httpx.Client(transport=httpx.MockTransport(closing_handler)) as c:
            response = c.get("https://api.mycorp.com/x")

    assert response.status_code == 200
    assert seen[0] is not None  # injected before the teardown reached it
    assert _patched_attributes() == before  # and no patch outlived it

    # and the very next request is plain traffic, not a half-removed patch
    def plain_handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("traceparent"))
        return httpx.Response(200)

    with httpx.Client(transport=httpx.MockTransport(plain_handler)) as c:
        c.get("https://api.mycorp.com/x")
    assert seen[1] is None


def test_close_under_live_traffic_raises_nothing_into_the_host():
    """The same cell with real concurrency: four threads issuing while we close.

    Fail-silence is the whole contract of this module — a failed patch or
    header computation must never break the user's HTTP call — and teardown is
    where it is hardest to keep, because the client, the config and the patch
    all disappear underneath a request that is already running.
    """
    _hub.reset_for_test()
    before = _patched_attributes()
    wardex_sdk.init(intercept=False, propagation=PropagationConfig(enabled=True))
    errors: list[BaseException] = []
    stop = threading.Event()
    started = threading.Barrier(5)

    def hammer():
        try:
            transport = httpx.MockTransport(lambda request: httpx.Response(200))
            with httpx.Client(transport=transport) as c:
                started.wait(timeout=5)
                while not stop.is_set():
                    with trace("root"):
                        c.get("https://api.mycorp.com/x")
        except BaseException as exc:  # noqa: BLE001 — the assertion is "none of these"
            errors.append(exc)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        started.wait(timeout=5)
        wardex_sdk.close()
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=10)
        stragglers = [t.name for t in threads if t.is_alive()]
    # Checked, not merely waited on — and outside the `finally`, so a straggler
    # reports itself rather than replacing whatever the body raised. A hammer
    # still running would keep issuing requests and appending to `errors` after
    # the next line has already read it.
    assert stragglers == []
    assert errors == []
    assert _patched_attributes() == before


def test_exporter_post_not_injected():
    """The OTLP exporter's own POST must never carry traceparent (self-exclusion)."""
    from wardex_sdk._suppress import suppress_capture
    from wardex_sdk.context._inject import _build_inject_headers

    _setup(propagation=PropagationConfig(enabled=True))
    with trace("root"):
        assert _build_inject_headers("collector.mycorp.com") != {}
        with suppress_capture():
            assert _build_inject_headers("collector.mycorp.com") == {}


def test_end_to_end_chain_one_trace():
    """Design §9(1) programmatically: join -> agent span -> gather tools ->
    outbound injection, all sharing the inbound trace_id."""
    import asyncio

    TP = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    inbound_tid = TP.split("-")[1]
    _setup(propagation=PropagationConfig(enabled=True))
    install_propagation()
    injected: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        injected.append(request.headers.get("traceparent", ""))
        return httpx.Response(200)

    async def tool(n: int):
        with span(f"tool-{n}"):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
                await c.get("https://billing.mycorp.com/x")

    async def main():
        with wardex_sdk.continue_trace({"traceparent": TP}):
            with trace("agent-turn"):
                await asyncio.gather(tool(1), tool(2))

    asyncio.run(main())
    # every outbound call carried the inbound trace id
    assert len(injected) == 2
    assert all(h.split("-")[1] == inbound_tid for h in injected)
    # and the two injected parent span ids differ (each tool's own span)
    assert injected[0].split("-")[2] != injected[1].split("-")[2]
