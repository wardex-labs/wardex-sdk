"""What the pydantic-ai adapter's tests share: a fake model API, a model with per-request
names, the two instrumentation setups, and the wire/bridge split.

No test here; imported by the `test_pydantic_ai_*` files.

* `fake_openai()` — a loopback Chat Completions server on HTTP/1.0, so pydantic-ai's
  pooled client cannot carry a keep-alive connection from one test's event loop into
  the next (the openai-agents wire tests' reason). A plain request is answered with usage
  10/3; a streamed one with five chunks, the last carrying usage 11/4.
* `PerRequestModel` — a `FunctionModel` whose name is `f"{label}-{n}"` for its n-th
  request. Every framework `chat` span ships as a unit, and a conformance chain names
  each span exactly once, so two requests of one model need two names.
* `wardex_owned()` / `user_owned()` — the two ways instrumentation can stand when wardex
  installs: off (wardex turns it on, Mode W), or on with the host's own SDK provider
  (wardex attaches a processor, Mode U). Both put `Agent._instrument_default` back by
  identity afterwards; the conformance harness's `clean_state` does not touch it.
* `spans_by_source` — splits shipped spans into the bridge's units and everything else
  by `CaptureSource.OTEL_BRIDGE`, never by span kind: the bridge's chat unit and the wire
  CHAT are both CLIENT spans.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from opentelemetry.metrics import NoOpMeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.instrumented import InstrumentationSettings

from wardex_sdk._enums import CaptureSource

_COMPLETION = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello there"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
}

#: Usage the fake states, per request shape.
PLAIN_USAGE = (10, 3)
STREAM_USAGE = (11, 4)


def _sse() -> bytes:
    base = {
        "id": "chatcmpl-2",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
    }
    delta = [{"role": "assistant", "content": ""}, {"content": "hello"}, {"content": " there"}]
    chunks: list[dict[str, Any]] = [
        {**base, "choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in delta
    ]
    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
    chunks.append(
        {
            **base,
            "choices": [],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
        }
    )
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode()


@contextlib.contextmanager
def fake_openai(*, delay: float = 0.0) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """`(base_url, posts)`: a loopback Chat Completions API, and what it was sent. `delay`
    holds every answer that many seconds, for a request cancelled in flight."""
    posts: list[dict[str, Any]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            request = json.loads(raw or b"{}")
            posts.append({"path": self.path, "stream": bool(request.get("stream"))})
            if delay:
                time.sleep(delay)
            if request.get("stream"):
                body, kind = _sse(), "text/event-stream"
            else:
                body, kind = json.dumps(_COMPLETION).encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    host, port = httpd.socket.getsockname()[:2]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://{host}:{port}/v1", posts
    finally:
        httpd.shutdown()
        httpd.server_close()


def openai_model(base_url: str) -> Any:
    """`gpt-4o-mini` on the fake, through pydantic-ai's own OpenAI provider."""
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel("gpt-4o-mini", provider=OpenAIProvider(base_url=base_url, api_key="t"))


class PerRequestModel(FunctionModel):
    """A `FunctionModel` named `f"{label}-{n}"` for its n-th request.

    The name pydantic opens the `chat` span with is read before `request()` runs, and the
    response is stamped with it inside `request()`, so the counter moves only once the
    request has returned: both reads see the same `n`.
    """

    def __init__(self, label: str, function: Callable[..., Any]) -> None:
        super().__init__(function, model_name=f"{label}-0")
        self.stem = label
        self.n = 0

    async def request(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return await super().request(*args, **kwargs)
        finally:
            self.n += 1
            self._model_name = f"{self.stem}-{self.n}"


@dataclass
class Instrumentation:
    """How instrumentation stood when wardex installed, and what to compare against."""

    mode: str
    settings: Any = None
    exporter: InMemorySpanExporter | None = None
    provider: TracerProvider | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def processors(self) -> Any:
        """The host provider's processor tuple: the Mode U seam."""
        return self.settings.tracer.span_processor._span_processors


@contextlib.contextmanager
def _restoring_default() -> Iterator[None]:
    before = vars(Agent)["_instrument_default"]
    try:
        yield
    finally:
        Agent._instrument_default = before


@contextlib.contextmanager
def wardex_owned() -> Iterator[Instrumentation]:
    """Instrumentation off, as a host that never turned it on has it (Mode W)."""
    with _restoring_default():
        Agent._instrument_default = False
        yield Instrumentation("wardex_owned")


def host_provider() -> tuple[TracerProvider, InMemorySpanExporter]:
    """An SDK provider with one host processor already on it, so the processor tuple the
    seam check compares by identity is never CPython's empty-tuple singleton."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


@contextlib.contextmanager
def user_owned(*, version: int | None = None) -> Iterator[Instrumentation]:
    """Instrumentation on with the host's own SDK provider (Mode U)."""
    provider, exporter = host_provider()
    kwargs: dict[str, Any] = {} if version is None else {"version": version}
    settings = InstrumentationSettings(
        tracer_provider=provider, meter_provider=NoOpMeterProvider(), **kwargs
    )
    with _restoring_default():
        Agent.instrument_all(settings)
        try:
            yield Instrumentation("user_owned", settings, exporter, provider)
        finally:
            provider.shutdown()


@contextlib.contextmanager
def instrumentation_mode(mode: str) -> Iterator[Instrumentation]:
    """`wardex_owned()` or `user_owned()` by name, for a parametrized fixture."""
    factory = wardex_owned if mode == "wardex_owned" else user_owned
    with factory() as setup:
        yield setup


MODES = ("wardex_owned", "user_owned")


def is_bridge(span: Any) -> bool:
    return CaptureSource.OTEL_BRIDGE in span.capture_sources


def spans_by_source(spans: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """`(bridge units, everything else)`."""
    bridge = [s for s in spans if is_bridge(s)]
    other = [s for s in spans if not is_bridge(s)]
    return bridge, other


def named(spans: Sequence[Any], name: str) -> list[Any]:
    return [s for s in spans if s.name == name]


def one(spans: Sequence[Any], name: str) -> Any:
    found = named(spans, name)
    assert len(found) == 1, (name, sorted(s.name for s in spans))
    return found[0]


def extras(span: Any) -> dict[str, Any]:
    return dict(span.extra)


__all__ = [
    "MODES",
    "PLAIN_USAGE",
    "STREAM_USAGE",
    "Instrumentation",
    "PerRequestModel",
    "extras",
    "fake_openai",
    "host_provider",
    "instrumentation_mode",
    "is_bridge",
    "named",
    "one",
    "openai_model",
    "spans_by_source",
    "user_owned",
    "wardex_owned",
]
