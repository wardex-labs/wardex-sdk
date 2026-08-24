"""The two closed vocabularies on the wire — design §6.5.1, §6.6, §6.7.

`CaptureIntegrity.limitations` and `CorrelationInfo.strategy` were free-form
strings. They are now `repeated Limitation limitation_codes` and
`ParentSource parent_source`, two closed proto enums, and the old tags are
reserved.

What is asserted here is not "the code compiles". It is the three properties
that make a closed vocabulary worth the schema break:

  * **Agreement.** The Python enum and the proto enum are the same list, in both
    directions. A member on one side and not the other is a value that either
    cannot be sent or cannot be read, and either way it fails here rather than
    on a user's wire.
  * **No silent drop.** A marker the schema cannot name is REPORTED — as the
    meta value plus the original string — never dropped. A span that reads as
    fully captured because the reason it was not could not be spelled is the
    exact failure the vocabulary exists to prevent.
  * **No reuse of the retired tags.** Tag 8 and tag 6 stay reserved. Reusing
    either is not a style question: `repeated string` and packed enum share
    wire type 2, so old bytes decode `Ok` as one enum value per ASCII byte.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from wardex_sdk import _wardex_native
from wardex_sdk._assembly import Limitation
from wardex_sdk._assembly._parentage import ParentSource
from wardex_sdk._enums import SpanKind, StatusCode
from wardex_sdk._types import (
    CaptureIntegrity,
    CorrelationInfo,
    Envelope,
    EnvelopeHeader,
    InternalSpan,
    SdkInfo,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport import _codec

_REPO = Path(__file__).resolve().parents[3]
_SPAN_PROTO = _REPO / "proto" / "wardex" / "v1" / "span.proto"


@pytest.fixture(scope="module")
def tables() -> dict:
    return _wardex_native.codec.vocabulary_tables()


# ---------------------------------------------------------------- agreement


def test_every_python_limitation_is_declared_in_the_schema(tables: dict) -> None:
    """Python -> proto. A member with no proto value cannot reach the wire.

    It would not raise. `integrity_to_proto` would fall back to the meta value
    and the marker would arrive as `vocabulary_unmapped`, which is honest but
    useless: the dashboard cannot render "something was wrong, we forgot to
    declare what".
    """
    declared = tables["Limitation"]
    missing = sorted(m.value for m in Limitation if m.value not in declared)
    assert not missing, (
        f"declared in Python but not in common.proto: {missing}\n"
        "Add the value to `enum Limitation` in proto/wardex/v1/common.proto. "
        "Value names lock the moment they ship (buf ENUM_VALUE_SAME_NAME)."
    )


def test_every_schema_limitation_has_a_python_member(tables: dict) -> None:
    """proto -> Python. The other direction, and it is not symmetric.

    A proto value with no Python member is a value this SDK can DECODE from a
    newer sender and never produce — which is fine — but it is also how a
    vocabulary silently forks: the next member gets added to the schema, nobody
    notices Python lacks it, and the emitter that needed it invents a string.
    """
    python_values = {m.value for m in Limitation}
    extra = sorted(v for v in tables["Limitation"] if v not in python_values)
    assert not extra, f"declared in common.proto but not in assembly._integrity.Limitation: {extra}"


def test_the_vocabulary_is_the_same_size_on_both_sides(tables: dict) -> None:
    assert len(tables["Limitation"]) == len(list(Limitation)) == 43


def test_parent_source_agrees_in_both_directions(tables: dict) -> None:
    declared = tables["ParentSource"]
    python_values = {m.value for m in ParentSource}
    assert set(declared) == python_values
    assert len(declared) == 7


def test_the_retired_adapter_strategies_are_not_in_the_vocabulary(tables: dict) -> None:
    """`adapter_hook` / `adapter_stream` answered a different question.

    They said which SOURCE observed the event, in the field that means how the
    PARENT was derived. `capture_sources` carries the first; conflating the two
    is what let a reader mistake an observation channel for evidence about the
    edge. Neither is a `ParentSource` and neither may come back as one.
    """
    assert "adapter_hook" not in tables["ParentSource"]
    assert "adapter_stream" not in tables["ParentSource"]


# ------------------------------------------------------------- the meta value


def test_the_meta_value_is_not_part_of_the_vocabulary(tables: dict) -> None:
    """`vocabulary_unmapped` is a statement ABOUT the vocabulary, not in it.

    A consumer iterating "every limitation" — to build a filter menu, say —
    must not be offered it, which is why it sits in its own number band and its
    own table.
    """
    assert "vocabulary_unmapped" not in tables["Limitation"]
    assert tables["LimitationMeta"] == {"vocabulary_unmapped": 9001}
    assert all(n <= 43 for n in tables["Limitation"].values())


# ------------------------------------------------------------------ round trip


def _span(**kw) -> InternalSpan:
    base = dict(
        context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
        parent_span_id=None,
        name="chat gpt-4",
        kind=SpanKind.CLIENT,
        start_time_ns=1,
        end_time_ns=2,
        status=StatusCode.OK,
    )
    base.update(kw)
    return InternalSpan(**base)


def _roundtrip(span: InternalSpan) -> dict:
    """Through the real encoder and back. Not a mock of it.

    The point of these tests is that the two enums survive protobuf, so
    asserting against a hand-built dict would test nothing.
    """
    env = Envelope(
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
    return _codec.decode(_codec.encode(env))["items"][0]["span"]


def test_limitations_round_trip_as_named_values() -> None:
    span = _span(
        capture_integrity=CaptureIntegrity(
            limitations=(
                Limitation.BODY_CAP_EXCEEDED,
                Limitation.PARENT_UNRESOLVED,
                Limitation.SESSION_ABORTED,
            )
        )
    )
    got = _roundtrip(span)["capture_integrity"]["limitations"]
    assert got == ["body_cap_exceeded", "parent_unresolved", "session_aborted"]


def test_parent_source_round_trips() -> None:
    span = _span(correlation=CorrelationInfo(strategy=ParentSource.UNIT_SOLE, confidence=0.5))
    assert _roundtrip(span)["correlation"]["strategy"] == "unit_sole"


def test_no_parentage_claim_stays_no_claim() -> None:
    """`strategy=None` must not decode as a value.

    Every sub-root span the Anthropic adapter builds reports no parentage claim
    on purpose — its anchor can fall back to the session root — so "unset" has
    to survive the round trip as unset. Reading it back as `contextvar` (the
    first enum value) would turn a deliberate silence into the strongest claim
    in the vocabulary.
    """
    span = _span(correlation=CorrelationInfo(strategy=None, confidence=0.7, request_id="toolu_1"))
    assert _roundtrip(span)["correlation"]["strategy"] == ""


# --------------------------------------------------- the unmappable marker


class _ForeignIntegrity:
    """A `CaptureIntegrity` holding a marker this schema has no value for.

    Duck-typed on purpose: the real dataclass cannot hold one, which is the
    point of closing it. What is being tested is the ENCODER's behaviour when
    the two vocabularies have drifted — the situation a newer Python enum, or a
    hand-built object, actually produces.
    """

    request_headers_captured = False
    request_body_captured = True
    response_headers_captured = False
    response_body_captured = True
    redacted = False
    truncated = True
    dropped_chunk_count = 0
    limitations = ("a_marker_from_the_future",)


def test_an_unnameable_marker_is_reported_not_dropped() -> None:
    """The one behaviour that justifies the meta value existing.

    Dropping it would leave a span whose `truncated=True` has no stated reason
    — indistinguishable from a span that simply had none. The marker becomes
    `vocabulary_unmapped` AND the original string is preserved in `extra`, so a
    reader learns both that something was lost and what the sender called it.
    """
    got = _roundtrip(_span(capture_integrity=_ForeignIntegrity()))
    assert got["capture_integrity"]["limitations"] == ["vocabulary_unmapped"]
    assert {"key": "wardex.limitation.unmapped", "value": "a_marker_from_the_future"} in got[
        "extra"
    ]


def test_a_parent_source_the_schema_cannot_name_becomes_no_claim() -> None:
    """Different disposition from `Limitation`, and deliberately so.

    "wardex does not claim to know how this parent was derived" is a meaningful,
    honest statement, and `confidence` beside it already carries the trust
    level separately. There is nothing extra to preserve, so there is no meta
    value here.
    """

    class _Foreign:
        operation_id = None
        request_id = None
        attempt_id = None
        active_span_id_at_capture = None
        confidence = 0.5
        strategy = "adapter_hook"

    assert _roundtrip(_span(correlation=_Foreign()))["correlation"]["strategy"] == ""


# -------------------------------------------- the type must reach the PRODUCER


def test_the_parentage_core_hands_out_members_not_strings() -> None:
    """A retype that stops at the declaration closes nothing.

    This is the exact bug it was written for. `resolve_parentage` — the single
    producer of parentage in the SDK, and therefore the source of `strategy` on
    very nearly every span — built `CorrelationInfo(strategy=src.value, ...)`,
    the string. Nothing failed: the encoder accepts either form, and every
    assertion in the suite was written against the string. So
    `isinstance(strategy, ParentSource)` was False everywhere while the
    dataclass promised otherwise, and an assertion written in the member form
    would have failed where the old string form passed.
    """
    from wardex_sdk._assembly import Ambient, resolve_parentage

    rooted = resolve_parentage(Ambient(None, None, None))
    assert rooted.correlation is not None
    assert isinstance(rooted.correlation.strategy, ParentSource), (
        f"got {type(rooted.correlation.strategy).__name__}, not a ParentSource member"
    )
    assert rooted.correlation.strategy is ParentSource.TRACE_ROOT

    parent = SpanContext(trace_id=TraceId(b"\x07" * 16), span_id=SpanId(b"\x08" * 8))
    joined = resolve_parentage(Ambient(parent, None, None))
    assert joined.correlation is not None
    assert isinstance(joined.correlation.strategy, ParentSource)
    assert joined.correlation.strategy is ParentSource.CONTEXTVAR


def test_every_link_the_builder_makes_carries_a_member() -> None:
    """`SpanDraft.add_link` is the only production site that builds a link.

    It stored `reason.value`, so `link.reason is LinkReason.HANDOFF_FROM` — the
    check that tells a renderer to draw a SIBLING rather than nest — was always
    False. A five-hop handoff chain would have rendered as five levels of
    nesting, making every duration in the flame graph wrong, which is the whole
    reason the link carries a reason at all.
    """
    from wardex_sdk._assembly import Ambient, LinkReason, SpanDraft, resolve_parentage
    from wardex_sdk._enums import SpanKind

    draft = SpanDraft.manual(
        resolve_parentage(Ambient(None, None, None)),
        name="x",
        kind=SpanKind.INTERNAL,
        start_ns=1,
    )
    other = SpanContext(trace_id=TraceId(b"\x09" * 16), span_id=SpanId(b"\x0a" * 8))
    draft.add_link(other, LinkReason.HANDOFF_FROM)
    (link,) = draft.finish(2).links

    assert isinstance(link.reason, LinkReason)
    assert link.reason is LinkReason.HANDOFF_FROM


def test_a_span_off_the_real_path_carries_members_in_all_three_fields() -> None:
    """The three retyped fields, on one span, through the real constructor.

    Checked together because they failed independently: `limitations` landed on
    members in the same change that left `strategy` and `reason` as strings.
    Whatever is added next, this is the assertion that notices it did not.
    """
    from wardex_sdk._assembly import Ambient, LinkReason, SpanDraft, resolve_parentage
    from wardex_sdk._enums import SpanKind

    draft = SpanDraft.manual(
        resolve_parentage(Ambient(None, None, None)),
        name="x",
        kind=SpanKind.INTERNAL,
        start_ns=1,
    )
    draft.add_limitation(Limitation.NO_WIRE_EVIDENCE)
    draft.add_link(
        SpanContext(trace_id=TraceId(b"\x0b" * 16), span_id=SpanId(b"\x0c" * 8)),
        LinkReason.TRIGGERED_BY,
    )
    span = draft.finish(2)

    assert all(isinstance(m, Limitation) for m in span.capture_integrity.limitations)
    assert isinstance(span.correlation.strategy, ParentSource)
    assert all(isinstance(link.reason, LinkReason) for link in span.links)


# ------------------------------------------------------------ the reserved tags


def test_the_retired_tags_stay_reserved() -> None:
    """Tag 8 and tag 6 may never be reused, and this is why.

    `repeated string` -> packed enum shares wire type 2, so bytes written by an
    older SDK decode as `Ok` — one enum value per ASCII byte of the old string.
    That is data corruption with no error path, which is worse than the
    alternative (a scalar `string` -> enum is a hard DecodeError that prost
    propagates up the nesting and fails the entire envelope for one stale span).
    Neither is acceptable, so the tags are retired rather than retyped.
    """
    text = _SPAN_PROTO.read_text()

    integrity = re.search(r"message CaptureIntegrity \{(.*?)\n\}", text, re.S)
    assert integrity is not None
    body = integrity.group(1)
    assert "reserved 8;" in body
    assert 'reserved "limitations";' in body
    assert not re.search(r"^\s*\w[\w.<>, ]*\s+\w+\s*=\s*8\s*;", body, re.M), (
        "tag 8 was reused in CaptureIntegrity"
    )

    correlation = re.search(r"message CorrelationInfo \{(.*?)\n\}", text, re.S)
    assert correlation is not None
    body = correlation.group(1)
    assert "reserved 6;" in body
    assert 'reserved "strategy";' in body
    assert not re.search(r"^\s*\w[\w.<>, ]*\s+\w+\s*=\s*6\s*;", body, re.M), (
        "tag 6 was reused in CorrelationInfo"
    )
