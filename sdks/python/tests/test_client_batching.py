"""Client batching internals — thread safety, backpressure, drain error paths."""

import threading
import time

from wardex_sdk._client import Client
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._types import (
    InternalEnvelope,
    InternalSpan,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport._base import Transport


def _wait_for(predicate, timeout=5.0):
    """Poll a predicate with a generous upper bound — no bare-sleep asserts."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _span(name="s"):
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name=name,
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
    )


def test_concurrent_capture_and_drain_loses_nothing():
    """The core regression guard of this slice (spec §11): no loss, no dup."""
    t = _Recording()
    c = Client(WardexConfig(api_key="k", max_buffer_spans=100_000), t)
    n_threads, m_spans = 8, 500
    stop_draining = threading.Event()

    def producer():
        for _ in range(m_spans):
            c.capture_span(_span())

    def drainer():
        while not stop_draining.is_set():
            c.flush()

    producers = [threading.Thread(target=producer) for _ in range(n_threads)]
    d = threading.Thread(target=drainer)
    d.start()
    for th in producers:
        th.start()
    for th in producers:
        th.join()
    stop_draining.set()
    d.join()
    c.flush()  # drain whatever the racing drainer didn't take
    total_sent = sum(len(e.spans) for e in t.envelopes)
    assert total_sent == n_threads * m_spans
    c.close()


def test_backpressure_drops_oldest_keeps_newest():
    t = _Recording()
    c = Client(WardexConfig(api_key="k", max_buffer_spans=10), t)
    for i in range(15):
        c.capture_span(_span(name=f"s{i}"))
    c.flush()
    [env] = t.envelopes
    assert len(env.spans) == 10
    assert [s.name for s in env.spans] == [f"s{i}" for i in range(5, 15)]
    c.close()


def test_dropped_count_reported_once_in_debug(capsys):
    t = _Recording()
    c = Client(WardexConfig(api_key="k", max_buffer_spans=2, debug=True), t)
    for _ in range(5):
        c.capture_span(_span())
    c.flush()
    assert "[wardex] dropped 3 spans (buffer full)" in capsys.readouterr().err
    c.flush()  # counter was reset — no second report
    assert "dropped" not in capsys.readouterr().err
    c.close()


def test_before_send_exception_drops_envelope_and_does_not_propagate():
    def boom(envelope):
        raise RuntimeError("boom")

    t = _Recording()
    c = Client(WardexConfig(api_key="k", before_send=boom), t)
    c.capture_span(_span())
    c.flush()  # must not raise (fail-closed: drop, spec §10)
    assert t.envelopes == []
    c.close()


def test_export_exception_drops_envelope_and_does_not_propagate():
    class _Exploding(Transport):
        def export(self, envelope):
            raise ValueError("encode failed")

    c = Client(WardexConfig(api_key="k"), _Exploding())
    c.capture_span(_span())
    c.flush()  # must not raise
    c.close()


def test_capture_during_export_goes_to_fresh_buffer():
    """Capture must never block on (or leak into) an in-flight drain."""
    entered = threading.Event()
    release = threading.Event()

    class _Blocking(Transport):
        def __init__(self):
            self.envelopes = []

        def export(self, envelope):
            self.envelopes.append(envelope)
            entered.set()
            release.wait(timeout=5.0)

    t = _Blocking()
    c = Client(WardexConfig(api_key="k"), t)
    c.capture_span(_span(name="first"))
    flusher = threading.Thread(target=c.flush)
    flusher.start()
    assert entered.wait(timeout=5.0)
    c.capture_span(_span(name="second"))  # must return instantly, land in new buffer
    release.set()
    flusher.join(timeout=5.0)
    c.flush()
    assert [s.name for e in t.envelopes for s in e.spans] == ["first", "second"]
    c.close()
