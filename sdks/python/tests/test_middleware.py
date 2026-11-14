"""ASGI/WSGI middleware — join via continue_trace, per-request isolation."""

import asyncio

import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._types import Envelope
from wardex_sdk.transport._base import Transport

TP = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[Envelope] = []

    def export(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)


def _setup():
    _hub.reset_for_test()
    t = _Recording()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k")), t))
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

    mw = wardex_sdk.WardexAsgiMiddleware(app)
    asyncio.run(mw(_http_scope([(b"traceparent", TP.encode())]), _noop_receive, _noop_send))
    assert seen["trace"] == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_asgi_requests_are_isolated():
    _setup()

    async def app(scope, receive, send):
        s = _hub.get_current_scope()
        assert "leak" not in s.tags
        s.set_tag("leak", "1")

    mw = wardex_sdk.WardexAsgiMiddleware(app)
    asyncio.run(mw(_http_scope([]), _noop_receive, _noop_send))
    asyncio.run(mw(_http_scope([]), _noop_receive, _noop_send))  # would fail on leak


def test_asgi_passes_through_lifespan():
    _setup()
    called = {}

    async def app(scope, receive, send):
        called["type"] = scope["type"]

    mw = wardex_sdk.WardexAsgiMiddleware(app)
    asyncio.run(mw({"type": "lifespan"}, _noop_receive, _noop_send))
    assert called["type"] == "lifespan"


def test_asgi_app_exceptions_propagate():
    _setup()

    async def app(scope, receive, send):
        raise RuntimeError("app error")

    mw = wardex_sdk.WardexAsgiMiddleware(app)
    try:
        asyncio.run(mw(_http_scope([]), _noop_receive, _noop_send))
        raise AssertionError("should have raised")
    except RuntimeError:
        pass


def test_wsgi_joins_traceparent():
    _setup()
    seen = {}

    def app(environ, start_response):
        active = _hub.get_current_scope().active_span_context
        seen["trace"] = active.trace_id.hex() if active else None
        start_response("200 OK", [])
        return [b"ok"]

    mw = wardex_sdk.WardexWsgiMiddleware(app)
    environ = {"REQUEST_METHOD": "GET", "HTTP_TRACEPARENT": TP}
    body = mw(environ, lambda status, headers: None)
    assert list(body) == [b"ok"]
    assert seen["trace"] == "4bf92f3577b34da6a3ce929d0e0e4736"


def test_wsgi_no_header_starts_fresh():
    _setup()
    seen = {}

    def app(environ, start_response):
        seen["active"] = _hub.get_current_scope().active_span_context
        return [b"ok"]

    mw = wardex_sdk.WardexWsgiMiddleware(app)
    mw({"REQUEST_METHOD": "GET"}, lambda s, h: None)
    assert seen["active"] is None


def _serve(mw, environ):
    """What a thread-pool WSGI server does per request: call, iterate, close."""
    body = mw(environ, lambda status, headers: None)
    chunks = list(body)
    close = getattr(body, "close", None)
    if close is not None:
        close()
    return chunks


def _streaming_app(environ, start_response):
    if environ.get("HTTP_X_USER"):
        wardex_sdk.set_user(environ["HTTP_X_USER"])
    with wardex_sdk.span("view"):
        pass
    start_response("200 OK", [])

    def body():
        with wardex_sdk.span("stream"):
            yield b"chunk"
        if environ.get("HTTP_X_LATE_USER"):
            wardex_sdk.set_user(environ["HTTP_X_LATE_USER"])

    return body()


def _spans_by_request(t):
    spans = [s for e in t.envelopes for s in e.spans]
    return [(s.name, dict(s.extra or ()).get("user.id"), s.context.trace_id, s) for s in spans]


def test_wsgi_a_streamed_body_stays_in_its_request():
    """Work done while the server iterates the body is the request's: same
    trace, same user, and nothing it writes reaches the next request served
    on the same worker thread."""
    import concurrent.futures as cf

    t = _setup()
    mw = wardex_sdk.WardexWsgiMiddleware(_streaming_app)
    with cf.ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(_serve, mw, {"HTTP_X_USER": "alice", "HTTP_X_LATE_USER": "late"}).result()
        pool.submit(_serve, mw, {}).result()
        pool.submit(_serve, mw, {"HTTP_TRACEPARENT": TP, "HTTP_X_USER": "carol"}).result()
    _hub.get_client().flush()
    got = _spans_by_request(t)
    assert [(n, u) for n, u, _, _ in got] == [
        ("view", "alice"),
        ("stream", "alice"),
        ("view", None),
        ("stream", None),
        ("view", "carol"),
        ("stream", "carol"),
    ]
    # The streamed span continues the caller's trace like the view does.
    assert {got[4][2].hex(), got[5][2].hex()} == {"4bf92f3577b34da6a3ce929d0e0e4736"}


def test_wsgi_a_listed_body_and_an_app_error_pass_through_untouched():
    _setup()

    def listed(environ, start_response):
        start_response("200 OK", [])
        return [b"a", b"b"]

    body = wardex_sdk.WardexWsgiMiddleware(listed)({}, lambda s, h: None)
    assert body == [b"a", b"b"]

    def broken(environ, start_response):
        raise RuntimeError("app error")

    try:
        wardex_sdk.WardexWsgiMiddleware(broken)({}, lambda s, h: None)
        raise AssertionError("should have raised")
    except RuntimeError:
        pass
    assert _hub.get_current_scope().active_span_context is None
