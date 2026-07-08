"""ASGI/WSGI middleware — join via continue_trace, per-request isolation."""

import asyncio

import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import WardexConfig
from wardex_sdk._types import InternalEnvelope
from wardex_sdk.transport._base import Transport

TP = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _setup():
    _hub.reset_for_test()
    t = _Recording()
    _hub.set_client(Client(WardexConfig(api_key="k"), t))
    return t


def _http_scope(headers: list[tuple[bytes, bytes]]):
    return {"type": "http", "method": "GET", "path": "/", "headers": headers}


async def _noop_receive():
    return {"type": "http.request"}


async def _noop_send(message):
    return None


def test_asgi_joins_traceparent():
    _setup()
    seen = {}

    async def app(scope, receive, send):
        active = _hub.get_current_scope().active_span_context
        seen["trace"] = active.trace_id.hex() if active else None

    mw = wardex_sdk.WardexMiddleware(app)
    asyncio.run(mw(_http_scope([(b"traceparent", TP.encode())]), _noop_receive, _noop_send))
    assert seen["trace"] == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_asgi_requests_are_isolated():
    _setup()

    async def app(scope, receive, send):
        s = _hub.get_current_scope()
        assert "leak" not in s.tags
        s.set_tag("leak", "1")

    mw = wardex_sdk.WardexMiddleware(app)
    asyncio.run(mw(_http_scope([]), _noop_receive, _noop_send))
    asyncio.run(mw(_http_scope([]), _noop_receive, _noop_send))  # would fail on leak


def test_asgi_passes_through_lifespan():
    _setup()
    called = {}

    async def app(scope, receive, send):
        called["type"] = scope["type"]

    mw = wardex_sdk.WardexMiddleware(app)
    asyncio.run(mw({"type": "lifespan"}, _noop_receive, _noop_send))
    assert called["type"] == "lifespan"


def test_asgi_app_exceptions_propagate():
    _setup()

    async def app(scope, receive, send):
        raise RuntimeError("app error")

    mw = wardex_sdk.WardexMiddleware(app)
    try:
        asyncio.run(mw(_http_scope([]), _noop_receive, _noop_send))
        raise AssertionError("should have raised")
    except RuntimeError:
        pass
