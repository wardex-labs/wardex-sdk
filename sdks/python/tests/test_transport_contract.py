"""The Transport SPI's contract, held from the outside.

Three claims a third-party implementer relies on, each watched here because no
in-tree transport exercises it the way an out-of-tree one will:

  * `encode()` is the sanctioned path to wire bytes, and MASKING IS NOT
    SKIPPABLE THROUGH IT: a minimal subclass that implements only `export()`
    and reaches bytes through `self.encode()` ships masked data under
    `init()`'s defaults, without ever touching a PII knob.
  * the two transports that deliberately bypass `encode()` — ConsoleTransport
    prints, RecordingTransport records — SAY on their classes that what they
    hold is raw, pre-masking data. The claim is contract, so it is pinned.
  * `wardex_sdk.transport` is the ONE implementer home: everything the SPI
    names is importable from it.
"""

from __future__ import annotations

import wardex_sdk
import wardex_sdk.transport
from wardex_sdk import ConsoleTransport, Envelope, PIIConfig
from wardex_sdk.testing import RecordingTransport, exactly_one
from wardex_sdk.transport import Transport

RAW_EMAIL = b"contact john.doe@acme.com about the invoice"


class _MinimalThirdParty(Transport):
    """The smallest legal transport: only `export()`, bytes via `encode()`.

    `compress=False` so the assertions below can read the protobuf bodies as
    bytes; masking runs before compression, so the claim is the same either
    way.
    """

    def __init__(self) -> None:
        self.bodies: list[bytes] = []

    def export(self, envelope: Envelope, *, timeout: float | None = None) -> None:
        self.bodies.extend(self.encode(envelope, compress=False))


def test_masking_is_not_skippable_through_the_sanctioned_path():
    """A third-party subclass on pii DEFAULTS: `init()` installs the policy
    through the private plumbing, `encode()` hands it to the native encoder,
    and the raw email never reaches the bytes the transport ships."""
    t = _MinimalThirdParty()
    wardex_sdk.init(transport=t, intercept=False)  # pii defaults: MASK, no exemptions
    with wardex_sdk.span("carries-pii") as s:
        s.input_data = RAW_EMAIL
    wardex_sdk.flush()
    wardex_sdk.close()

    wire = b"".join(t.bodies)
    assert wire, "the span never reached encode(); the test measured nothing"
    assert b"john.doe@acme.com" not in wire, (
        "the raw email reached the wire bytes through the sanctioned path"
    )
    # The masked form the engine writes (see test_pii_masking): assertable, so
    # asserted — proof the mask LANDED rather than the payload being dropped.
    assert b"[EMAIL]" in wire


def test_encode_masks_even_on_a_hand_built_transport():
    """No init() at all: the class-level policy defaults are secure, so a
    transport driven by hand still cannot ship the raw email through
    `encode()`. The explicit `_set_pii_policy(PIIConfig())` is what init()
    installs — a no-op on the defaults, exercised so the plumbing shape is
    covered too."""
    from wardex_sdk._enums import SpanKind, StatusCode
    from wardex_sdk._types import (
        EnvelopeHeader,
        InternalSpan,
        SdkInfo,
        SpanContext,
        SpanId,
        TraceId,
    )

    span = InternalSpan(
        context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
        parent_span_id=None,
        name="hand-built",
        kind=SpanKind.CLIENT,
        start_time_ns=1000,
        end_time_ns=2000,
        status=StatusCode.OK,
        input_data=RAW_EMAIL,
    )
    envelope = Envelope(
        header=EnvelopeHeader(
            event_id="evt-1",
            api_key="k",
            sdk=SdkInfo(
                name="wardex.python",
                version="0.1.0",
                python_version="3.12",
                os="mac",
                arch="arm64",
            ),
            sent_at_ns=42,
        ),
        spans=(span,),
    )
    assert envelope.span_count == 1

    t = _MinimalThirdParty()
    t._set_pii_policy(PIIConfig())
    t.export(envelope)
    wire = b"".join(t.bodies)
    assert b"john.doe@acme.com" not in wire
    assert b"[EMAIL]" in wire


# -- the transports that bypass encode() say so on the class -----------------


def test_console_transport_says_it_prints_raw_pre_masking_data():
    doc = ConsoleTransport.__doc__ or ""
    assert "RAW" in doc.upper() and "PRE-MASKING" in doc.upper(), (
        "ConsoleTransport's docstring must state it prints the envelope raw, "
        "pre-masking — the claim is contract"
    )
    assert "debugging" in doc.lower(), "…and that it is a local debugging tool only"


def test_recording_transport_says_it_records_raw_pre_masking_data():
    doc = " ".join((RecordingTransport.__doc__ or "").split())
    assert "PRE-MASKING" in doc.upper(), (
        "RecordingTransport's docstring must state it records pre-masking, in-process data"
    )
    assert "test double" in doc and "not an export path" in doc


# -- testing.RecordingTransport is the user-facing double ---------------------


def test_recording_transport_records_manual_spans_with_parentage():
    t = RecordingTransport()
    wardex_sdk.init(transport=t, intercept=False)
    with wardex_sdk.conversation("session"):
        with wardex_sdk.span("inner"):
            pass
    wardex_sdk.flush()
    wardex_sdk.close()

    nodes = t.spans
    root = exactly_one(nodes, "session")
    inner = exactly_one(nodes, "inner")
    assert inner.parent_id == root.span_id, "the child's edge must point at the conversation"
    assert inner.trace_id == root.trace_id
    assert t.envelopes and t.envelopes[0].span_count == len(t.envelopes[0].spans)


# -- one implementer home ------------------------------------------------------


def test_the_transport_module_is_the_complete_implementer_home():
    assert wardex_sdk.transport.__all__ == [
        "Transport",
        "NoOpTransport",
        "ConsoleTransport",
        "OtlpHttpTransport",
        "Envelope",
        "UNDELIVERED",
        "Undelivered",
        "CallerBudget",
        "DEFAULT_TIMEOUT",
    ]
    for name in wardex_sdk.transport.__all__:
        assert hasattr(wardex_sdk.transport, name), f"{name} is named but not importable"
    assert isinstance(wardex_sdk.transport.UNDELIVERED, wardex_sdk.transport.Undelivered)
