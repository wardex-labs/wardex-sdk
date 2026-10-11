"""PII masking — native contract + end-to-end (design 2026-07-07-phase3-pii-masking)."""

from __future__ import annotations

import pytest

from wardex_sdk._config import PIIConfig
from wardex_sdk._enums import SpanKind, StatusCode
from wardex_sdk._types import (
    Envelope,
    EnvelopeHeader,
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
        # Secret-shaped on purpose: the receiver-stamped id is whitelisted and
        # must survive masking untouched.
        project_id="sk-live-aaaaaaaaaaaaaaaa1234",
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


def _env(span: InternalSpan) -> Envelope:
    return Envelope(header=_header(), spans=(span,))


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

    def test_project_id_is_never_masked(self):
        env = _env(_span(input_data=PII_INPUT))
        out = _codec.decode(_codec.encode(env, pii_mode="mask", pii_disabled=()))
        assert out["header"]["project_id"] == "sk-live-aaaaaaaaaaaaaaaa1234"

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
            intercept=False,
            pii=PIIConfig(
                mode=PIIMode.MASK, disabled_categories=frozenset({PIICategory.IP_ADDRESS})
            ),
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
        # String, not bytes: the OTLP surface never emits bytes_value,
        # and masking runs first — on the raw bytes — so the mask still lands.
        assert "john.doe@acme.com" not in attrs["wardex.input_data"]
        assert "[EMAIL]" in attrs["wardex.input_data"]
        assert attrs["wardex.redacted"] is True

    def test_otlp_mask_mode_with_binary_payload_masks_its_text_and_goes_base64(self):
        """A payload that is not UTF-8 as a whole is masked stretch by stretch:
        the valid UTF-8 runs are masked, the bytes between them pass through,
        and the result still leaves the OTLP surface as a marked base64
        string, not bytes_value. Masking runs on the raw bytes BEFORE the
        base64 rewrite — which is why lifting the old all-or-nothing rule
        (design §4.4) landed the mask here with no further change."""
        import base64

        from wardex_sdk import _wardex_native

        payload = b"\xff\xfe" + b"contact: john.doe@acme.com;"
        env = _env(_span(input_data=payload))
        data = _wardex_native.codec.encode_otlp_traces(env, "mask", [])
        out = _wardex_native.codec.decode_otlp_traces(data)
        attrs = out["resource_spans"][0]["scope_spans"][0]["spans"][0]["attributes"]
        assert attrs["wardex.input_data.encoding"] == "base64"
        assert base64.b64decode(attrs["wardex.input_data"]) == b"\xff\xfe" + b"contact: [EMAIL];"

    def test_otlp_masking_runs_before_base64_never_on_it(self):
        """Pins mask-then-debyte ordering with an observable divergence: a
        binary payload whose BASE64 TEXT happens to contain a Luhn-valid card
        number. Masking the raw bytes finds nothing (non-UTF-8 passes through,
        §4.4); masking after the rewrite would find the '4111111111111111'
        inside the base64 string, replace it, and corrupt the payload beyond
        recovery. The exported string must round-trip to the exact bytes."""
        import base64

        from wardex_sdk import _wardex_native

        target_b64 = "//4111111111111111AA"
        payload = base64.b64decode(target_b64)
        # Preconditions that make the divergence observable:
        assert base64.b64encode(payload).decode() == target_b64
        with pytest.raises(UnicodeDecodeError):
            payload.decode()  # non-UTF-8 -> takes the base64 branch
        # Control: the engine really does mask this PAN in plain text, so a
        # disabled credit-card pattern can't silently vacuate this test.
        control = _env(_span(input_data=b"card: 4111111111111111"))
        cdata = _wardex_native.codec.encode_otlp_traces(control, "mask", [])
        cattrs = _wardex_native.codec.decode_otlp_traces(cdata)["resource_spans"][0]["scope_spans"][
            0
        ]["spans"][0]["attributes"]
        assert "4111111111111111" not in cattrs["wardex.input_data"]

        env = _env(_span(input_data=payload))
        data = _wardex_native.codec.encode_otlp_traces(env, "mask", [])
        attrs = _wardex_native.codec.decode_otlp_traces(data)["resource_spans"][0]["scope_spans"][
            0
        ]["spans"][0]["attributes"]
        assert attrs["wardex.input_data"] == target_b64
        assert base64.b64decode(attrs["wardex.input_data"]) == payload

    def test_pii_category_is_public_api(self):
        import wardex_sdk

        assert wardex_sdk.PIICategory.EMAIL.value == "email"

    def test_send_batch_passes_policy_to_native(self, monkeypatch):
        from wardex_sdk import _wardex_native
        from wardex_sdk._enums import PIICategory, PIIMode
        from wardex_sdk.transport._otlp_http import OtlpHttpTransport

        captured = {}

        def fake_encode(
            envelope, pii_mode, pii_disabled, limits, compress, *, pii_extra_names, pii_reveal_names
        ):
            captured["pii_mode"] = pii_mode
            captured["pii_disabled"] = pii_disabled
            captured["pii_extra_names"] = pii_extra_names
            captured["pii_reveal_names"] = pii_reveal_names
            # One empty body; the subsequent POST fails silently (the endpoint
            # is unreachable). Returning no bodies at all would let the
            # transport return before it ever built a request, which is a
            # different path from the one under test.
            return ([b""], 0, [], 0, 0, 0)

        monkeypatch.setattr(_wardex_native.codec, "encode_otlp_requests", fake_encode)
        t = OtlpHttpTransport(endpoint="http://localhost:1")
        t._set_pii_policy(
            PIIConfig(
                mode=PIIMode.MASK,
                disabled_categories={PIICategory.IP_ADDRESS},
                extra_secret_names={"x_corp_auth"},
                reveal_names={"page_token"},
            )
        )
        t._send_batch(_env(_span(input_data=PII_INPUT)))
        assert captured == {
            "pii_mode": "mask",
            "pii_disabled": ["ip_address"],
            "pii_extra_names": ["x_corp_auth"],
            "pii_reveal_names": ["page_token"],
        }


# --------------------------------------------------------------------------
# masking's own failure: fail-closed, marked, counted
# --------------------------------------------------------------------------

#: The masker's injection point: a span with this name, or a snapshot whose
#: conversation state is these bytes, panics the PII engine. It exists only in
#: a build with the `panic-injection` cargo feature, which the workspace
#: `pyproject.toml` turns on for `uv sync` and a release wheel never has.
_INJECTED = "__wardex_test_panic__"

_REBUILD = (
    "the native module was built without the `panic-injection` feature the "
    "workspace pyproject.toml asks for, most likely because it predates this "
    "source tree. Run `uv sync --reinstall-package wardex-sdk`."
)


def _report_lines(fn) -> list[str]:
    import logging

    from wardex_sdk._assembly._diag import reset_reports_for_test

    lines: list[str] = []

    class _Records(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    reset_reports_for_test()
    logger = logging.getLogger("wardex_sdk")
    handler = _Records(level=logging.DEBUG)
    logger.addHandler(handler)
    try:
        fn()
    finally:
        logger.removeHandler(handler)
        reset_reports_for_test()
    return lines


def _two_spans() -> Envelope:
    """A span masking succeeds on, and one it fails on, in ONE batch."""
    return Envelope(
        header=_header(),
        spans=(
            _span(name="kept", input_data=PII_INPUT),
            _span(
                name=_INJECTED,
                input_data=PII_INPUT,
                context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x03" * 8)),
            ),
        ),
    )


@pytest.mark.usefixtures("fresh_counters")
class TestMaskingFailure:
    """When the PII engine itself fails on a record, the record does not ship
    as captured: it leaves with its ids, kind and times only, named
    `[PII_FILTER_ERROR]`. That used to be all it said -- `redacted` is set on
    every span masking touched, so the name was the only tell, and nothing
    counted it. Now it carries `pii_filter_failed` in the closed vocabulary
    and each exporter counts it, on the build this suite runs against."""

    def test_the_development_build_carries_the_injection(self):
        from wardex_sdk import _wardex_native

        # The masker's injection and `_panic_for_test` are one cargo feature.
        assert hasattr(_wardex_native, "_panic_for_test"), _REBUILD

    def test_an_otlp_span_masking_failed_on_ships_emptied_marked_and_counted(self):
        from wardex_sdk import _wardex_native
        from wardex_sdk._assembly import counters
        from wardex_sdk.transport import Transport

        class _Encoding(Transport):
            def export(self, envelope, *, timeout=None):
                self.bodies = self.encode(envelope, compress=False)

        t = _Encoding()
        lines = _report_lines(lambda: t.export(_two_spans()))
        (body,) = t.bodies
        decoded = _wardex_native.codec.decode_otlp_traces(body)
        spans = {
            sp["name"]: sp
            for rs in decoded["resource_spans"]
            for ss in rs["scope_spans"]
            for sp in ss["spans"]
        }
        assert set(spans) == {"kept", "[PII_FILTER_ERROR]"}, _REBUILD
        scrubbed = spans["[PII_FILTER_ERROR]"]
        assert scrubbed["span_id"] == "03" * 8
        assert scrubbed["attributes"] == {
            "wardex.redacted": True,
            "wardex.limitations": ["pii_filter_failed"],
        }
        # The other span was masked as usual and carries no such marker.
        kept = spans["kept"]["attributes"]
        assert "[EMAIL]" in kept["wardex.input_data"]
        assert "wardex.limitations" not in kept
        assert counters.get("transport.otlp.pii_scrubbed") == 1
        (report,) = [line for line in lines if "transport.otlp.pii_scrubbed" in line]
        assert "john.doe" not in report

    def test_a_wardex_envelope_record_masking_failed_on_ships_emptied_marked_and_counted(self):
        from wardex_sdk import _wardex_native
        from wardex_sdk._assembly import counters
        from wardex_sdk._types import InternalStateSnapshot
        from wardex_sdk.transport import WardexTransport

        snap = InternalStateSnapshot(
            trace_id=TraceId(b"\x01" * 16),
            span_id=SpanId(b"\x02" * 8),
            timestamp_ns=1500,
            snapshot_type="turn_start",
            turn_index=1,
            conversation_state=_INJECTED.encode(),
            attributes=(("owner", "john.doe@acme.com"),),
        )
        env = _two_spans()
        env = Envelope(header=env.header, spans=env.spans, state_snapshots=(snap,))
        body, unmarshalled, scrubbed = _wardex_native.codec.encode_envelope_export(
            env, "mask", [], None
        )
        assert (unmarshalled, scrubbed) == ([], 2), _REBUILD
        items = _codec.decode(body)["items"]
        spans = {it["span"]["name"]: it["span"] for it in items if "span" in it}
        integrity = spans["[PII_FILTER_ERROR]"]["capture_integrity"]
        assert integrity["limitations"] == ["pii_filter_failed"]
        assert integrity["redacted"] is True
        assert spans["[PII_FILTER_ERROR]"]["input_data"] == b""
        assert spans["kept"]["capture_integrity"]["limitations"] == []
        (snapshot,) = [it["state_snapshot"] for it in items if "state_snapshot" in it]
        assert snapshot["conversation_state"] == b"[PII_FILTER_ERROR]"
        assert snapshot["attributes"] == [
            {"key": "wardex.limitations", "value": "pii_filter_failed"}
        ]

        # The exporter counts both, before it tries the (unreachable) POST.
        transport = WardexTransport("http://127.0.0.1:1", "test-key")
        lines = _report_lines(lambda: transport.export(env, timeout=1.0))
        assert counters.get("transport.wardex.pii_scrubbed") == 2
        assert any("transport.wardex.pii_scrubbed" in line for line in lines)
