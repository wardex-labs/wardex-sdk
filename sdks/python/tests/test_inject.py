"""Opt-in traceparent injection into HTTP client libraries."""

import http.server
import threading

import httpx
import pytest

import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, PropagationPolicy, WardexConfig
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
    _setup(propagation=PropagationPolicy(enabled=True))
    install_propagation()
    with trace("root") as root:
        req = _capture_request()
    assert req.headers["traceparent"].split("-")[1] == root.context.trace_id.hex()


def test_no_injection_outside_any_context():
    _setup(propagation=PropagationPolicy(enabled=True))
    install_propagation()
    req = _capture_request()
    assert "traceparent" not in req.headers


def test_no_injection_when_flag_off():
    _setup(propagation=PropagationPolicy(enabled=False))
    install_propagation()  # wiring guards on the flag too, but the header path must also guard
    with trace("root"):
        req = _capture_request()
    assert "traceparent" not in req.headers


def test_user_header_never_overwritten():
    _setup(propagation=PropagationPolicy(enabled=True))
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
    _setup(propagation=PropagationPolicy(enabled=True, targets=("*.mycorp.com",)))
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
    """
    _setup(propagation=PropagationPolicy(enabled=True, targets=("*.MyCorp.com",)))
    with trace("root"):
        assert _build_inject_headers("api.mycorp.com") != {}  # pattern folded
        assert _build_inject_headers("API.MyCorp.COM") != {}  # host folded
        assert _build_inject_headers("api.othercorp.com") == {}  # still an allowlist


def test_async_client_injects():
    import asyncio

    _setup(propagation=PropagationPolicy(enabled=True))
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
    _setup(propagation=PropagationPolicy(enabled=True))
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
    _setup(propagation=PropagationPolicy(enabled=True))
    install_propagation()
    with trace("root") as root:
        _get(library, f"{echo_server}/x")
    assert _sent("traceparent") == [
        f"00-{root.context.trace_id.hex()}-{root.context.span_id.hex()}-01"
    ]


@pytest.mark.parametrize("library", LIBRARIES)
def test_tracestate_forwarded_verbatim(library, echo_server):
    tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    _setup(propagation=PropagationPolicy(enabled=True))
    install_propagation()
    with wardex_sdk.continue_trace({"traceparent": tp, "tracestate": "dd=s:1"}):
        with trace("root"):
            _get(library, f"{echo_server}/x")
    assert _sent("tracestate") == ["dd=s:1"]


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
    _setup(propagation=PropagationPolicy(enabled=True))
    install_propagation()
    with trace("root"):
        _get(library, f"{echo_server}/x", session_headers={"traceparent": "session-default"})
    assert _sent("traceparent") == ["session-default"]


@pytest.mark.parametrize("library", LIBRARIES)
def test_a_per_request_traceparent_wins_over_injection(library, echo_server):
    _setup(propagation=PropagationPolicy(enabled=True))
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
    """
    tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    _setup(propagation=PropagationPolicy(enabled=True))
    install_propagation()
    with wardex_sdk.continue_trace({"traceparent": tp, "tracestate": "wardex=ours"}):
        with trace("root"):
            _get(library, f"{echo_server}/x", headers={"tracestate": "caller=theirs"})
    assert _sent("tracestate") == ["caller=theirs"]
    assert len(_sent("traceparent")) == 1


def test_aiohttp_preserves_duplicate_headers(echo_server):
    import asyncio

    import aiohttp

    _setup(propagation=PropagationPolicy(enabled=True))
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
    wardex_sdk.init(propagation=PropagationPolicy(enabled=True))
    assert httpx.Client.send is not orig
    wardex_sdk.close()
    assert httpx.Client.send is orig


def test_init_without_flag_does_not_patch():
    import httpx

    _hub.reset_for_test()
    orig = httpx.Client.send
    wardex_sdk.init()
    assert httpx.Client.send is orig
    wardex_sdk.close()


def test_exporter_post_not_injected():
    """The OTLP exporter's own POST must never carry traceparent (self-exclusion)."""
    from wardex_sdk._suppress import suppress_capture
    from wardex_sdk.context._inject import _build_inject_headers

    _setup(propagation=PropagationPolicy(enabled=True))
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
    _setup(propagation=PropagationPolicy(enabled=True))
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
