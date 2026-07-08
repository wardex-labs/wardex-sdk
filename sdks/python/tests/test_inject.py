"""Opt-in traceparent injection into HTTP client libraries."""

import http.server
import threading

import httpx
import pytest

import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import WardexConfig
from wardex_sdk._tracing import trace
from wardex_sdk._types import InternalEnvelope
from wardex_sdk.context._inject import install_propagation, uninstall_propagation
from wardex_sdk.transport._base import Transport


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _setup(**cfg):
    _hub.reset_for_test()
    _hub.set_client(Client(WardexConfig(api_key="k", **cfg), _Recording()))


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
    _setup(propagate_trace=True)
    install_propagation()
    with trace("root") as root:
        req = _capture_request()
    assert req.headers["traceparent"].split("-")[1] == root.context.trace_id.hex()


def test_no_injection_outside_any_context():
    _setup(propagate_trace=True)
    install_propagation()
    req = _capture_request()
    assert "traceparent" not in req.headers


def test_no_injection_when_flag_off():
    _setup(propagate_trace=False)
    install_propagation()  # wiring guards on the flag too, but the header path must also guard
    with trace("root"):
        req = _capture_request()
    assert "traceparent" not in req.headers


def test_user_header_never_overwritten():
    _setup(propagate_trace=True)
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
    _setup(propagate_trace=True, propagate_targets=("*.mycorp.com",))
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


def test_async_client_injects():
    import asyncio

    _setup(propagate_trace=True)
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
    _setup(propagate_trace=True)
    orig = httpx.Client.send
    install_propagation()
    install_propagation()
    uninstall_propagation()
    uninstall_propagation()
    assert httpx.Client.send is orig


class _HeaderEcho(http.server.BaseHTTPRequestHandler):
    seen: list[dict] = []
    multi: list = []  # repeated x-multi values; dict(self.headers) collapses duplicates

    def do_GET(self):
        _HeaderEcho.seen.append(dict(self.headers))
        _HeaderEcho.multi.append(self.headers.get_all("x-multi"))
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *a):
        pass


@pytest.fixture()
def echo_server():
    _HeaderEcho.seen = []
    _HeaderEcho.multi = []
    srv = http.server.HTTPServer(("127.0.0.1", 0), _HeaderEcho)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_requests_injects(echo_server):
    import requests

    _setup(propagate_trace=True)
    install_propagation()
    with trace("root") as root:
        requests.get(f"{echo_server}/x", timeout=5)
    assert _HeaderEcho.seen[-1].get("traceparent", "").split("-")[1] == root.context.trace_id.hex()


def test_aiohttp_injects(echo_server):
    import asyncio

    import aiohttp

    _setup(propagate_trace=True)
    install_propagation()

    async def main():
        with trace("root") as root:
            async with aiohttp.ClientSession() as s:
                async with s.get(f"{echo_server}/x") as resp:
                    await resp.read()
            return root.context.trace_id.hex()

    tid = asyncio.run(main())
    assert _HeaderEcho.seen[-1].get("traceparent", "").split("-")[1] == tid


def test_aiohttp_preserves_duplicate_headers(echo_server):
    import asyncio

    import aiohttp

    _setup(propagate_trace=True)
    install_propagation()

    async def main():
        with trace("root"):
            async with aiohttp.ClientSession() as s:
                headers = [("x-multi", "a"), ("x-multi", "b")]
                async with s.get(f"{echo_server}/x", headers=headers) as resp:
                    await resp.read()

    asyncio.run(main())
    assert _HeaderEcho.multi[-1] == ["a", "b"]
    assert "traceparent" in _HeaderEcho.seen[-1]


def test_tracestate_forwarded_verbatim(echo_server):
    import requests

    tp = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    _setup(propagate_trace=True)
    install_propagation()
    with wardex_sdk.continue_trace({"traceparent": tp, "tracestate": "dd=s:1"}):
        with trace("root"):
            requests.get(f"{echo_server}/x", timeout=5)
    assert _HeaderEcho.seen[-1].get("tracestate") == "dd=s:1"
