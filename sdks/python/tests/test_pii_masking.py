"""PII masking — native contract + end-to-end (design 2026-07-07-phase3-pii-masking)."""

from __future__ import annotations

import pytest

from wardex_sdk._enums import SpanKind, StatusCode
from wardex_sdk._types import (
    EnvelopeHeader,
    InternalEnvelope,
    InternalSpan,
    SdkInfo,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport import _codec

PII_INPUT = b"contact john.doe@acme.com card 4111-1111-1111-1111 from 10.0.0.5"


def _header() -> EnvelopeHeader:
    return EnvelopeHeader(
        event_id="evt-1",
        api_key="sk-live-aaaaaaaaaaaaaaaa1234",  # secret-shaped on purpose: must survive
        sdk=SdkInfo(
            name="wardex.python", version="0.1.0", python_version="3.12", os="mac", arch="arm64"
        ),
        sent_at_ns=42,
    )


def _span(**kw) -> InternalSpan:
    base = dict(
        context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
        parent_span_id=None,
        name="GET /v1/chat",
        kind=SpanKind.CLIENT,
        start_time_ns=1000,
        end_time_ns=2000,
        status=StatusCode.OK,
    )
    base.update(kw)
    return InternalSpan(**base)


def _env(span: InternalSpan) -> InternalEnvelope:
    return InternalEnvelope(header=_header(), spans=(span,))


class TestNativeContract:
    def test_mask_mode_masks_and_flags(self):
        env = _env(_span(input_data=PII_INPUT))
        out = _codec.decode(_codec.encode(env, pii_mode="mask", pii_disabled=()))
        span = out["items"][0]["span"]
        assert b"john.doe@acme.com" not in span["input_data"]
        assert b"[EMAIL]" in span["input_data"]
        assert b"****-****-****-1111" in span["input_data"]
        assert b"10.0.0.5" not in span["input_data"]
        assert span["capture_integrity"]["redacted"] is True

    def test_api_key_is_never_masked(self):
        env = _env(_span(input_data=PII_INPUT))
        out = _codec.decode(_codec.encode(env, pii_mode="mask", pii_disabled=()))
        assert out["header"]["api_key"] == "sk-live-aaaaaaaaaaaaaaaa1234"

    def test_off_mode_is_byte_identical_to_legacy(self):
        env = _env(_span(input_data=PII_INPUT))
        assert _codec.encode(env, pii_mode="off", pii_disabled=()) == _codec.encode(env)

    def test_disabled_category_passes_through(self):
        env = _env(_span(input_data=PII_INPUT))
        out = _codec.decode(_codec.encode(env, pii_mode="mask", pii_disabled=("ip_address",)))
        data = out["items"][0]["span"]["input_data"]
        assert b"10.0.0.5" in data
        assert b"[EMAIL]" in data

    def test_unknown_category_raises(self):
        with pytest.raises(ValueError, match="not_a_category"):
            _codec.encode(_env(_span()), pii_mode="mask", pii_disabled=("not_a_category",))

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="redact"):
            _codec.encode(_env(_span()), pii_mode="redact", pii_disabled=())


class TestTransportPolicy:
    def test_transport_defaults_are_secure(self):
        from wardex_sdk.transport._noop import NoOpTransport

        t = NoOpTransport()
        assert t._pii_mode == "mask"
        assert t._pii_disabled == ()

    def test_init_propagates_policy_to_transport(self):
        import wardex_sdk
        from wardex_sdk._enums import PIICategory, PIIMode
        from wardex_sdk.transport._noop import NoOpTransport

        t = NoOpTransport()
        wardex_sdk.init(
            transport=t,
            pii_mode=PIIMode.MASK,
            pii_disabled_categories=frozenset({PIICategory.IP_ADDRESS}),
        )
        assert t._pii_mode == "mask"
        assert t._pii_disabled == ("ip_address",)

    def test_otlp_wire_bytes_are_masked(self):
        from wardex_sdk import _wardex_native

        env = _env(_span(input_data=PII_INPUT))
        data = _wardex_native.codec.encode_otlp_traces(env, "mask", [])
        out = _wardex_native.codec.decode_otlp_traces(data)
        span = out["resource_spans"][0]["scope_spans"][0]["spans"][0]
        attrs = span["attributes"]
        assert b"john.doe@acme.com" not in attrs["wardex.input_data"]
        assert b"[EMAIL]" in attrs["wardex.input_data"]
        assert attrs["wardex.redacted"] is True

    def test_pii_category_is_public_api(self):
        import wardex_sdk

        assert wardex_sdk.PIICategory.EMAIL.value == "email"
