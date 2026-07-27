from wardex_sdk._client import Client, build_sdk_info
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._limits import CaptureLimits
from wardex_sdk._types import (
    InternalEnvelope,
    InternalSpan,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport._base import Transport


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _span(output_data: bytes = b""):
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name="s",
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
        output_data=output_data,
    )


def test_build_sdk_info_has_runtime_meta():
    info = build_sdk_info()
    assert info.name == "wardex.python"
    assert info.python_version and info.os and info.arch


def test_flush_emits_buffered_spans_in_one_envelope():
    t = _Recording()
    c = Client(WardexConfig(api_key="k"), t)
    c.capture_span(_span())
    c.capture_span(_span())
    assert t.envelopes == []
    c.flush()
    assert len(t.envelopes) == 1
    assert len(t.envelopes[0].spans) == 2
    c.close()


def test_close_flushes_and_is_idempotent():
    t = _Recording()
    c = Client(WardexConfig(api_key="k"), t)
    c.capture_span(_span())
    c.close()
    c.close()
    assert len(t.envelopes) == 1


def test_flush_stamps_sent_at_ns():
    t = _Recording()
    c = Client(WardexConfig(api_key="k"), t)
    c.capture_span(_span())
    c.flush()
    assert t.envelopes[0].header.sent_at_ns > 0
    c.close()


def test_span_buffer_respects_the_byte_budget():
    """Count-based capping alone cannot bound memory: 2048 large spans is gigabytes."""
    t = _Recording()
    cfg = WardexConfig(
        api_key="k",
        limits=CaptureLimits(max_buffer_bytes=64 * 1024, max_buffer_spans=1000),
    )
    c = Client(cfg, t)
    try:
        for _ in range(50):
            c.capture_span(_span(output_data=b"x" * 8192))
        assert c._buffered_bytes <= 64 * 1024
        assert c._dropped > 0
        assert len(c._spans) < 50
    finally:
        c.close()


def test_byte_budget_leaves_small_spans_alone():
    t = _Recording()
    cfg = WardexConfig(api_key="k", limits=CaptureLimits(max_buffer_bytes=64 * 1024))
    c = Client(cfg, t)
    try:
        for _ in range(10):
            c.capture_span(_span(output_data=b"x" * 100))
        assert c._dropped == 0
        assert len(c._spans) == 10
    finally:
        c.close()
