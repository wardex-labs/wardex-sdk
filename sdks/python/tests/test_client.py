from wardex_sdk._client import Client, build_sdk_info
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


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _span():
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name="s",
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
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
