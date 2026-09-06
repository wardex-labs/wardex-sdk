"""OTLP/HTTP traces marshaling.

Envelope -> encode_otlp_traces -> decode -> field preservation.
"""

from __future__ import annotations

from wardex_sdk import _wardex_native
from wardex_sdk._assembly import (
    AMBIENT,
    Ambient,
    SpanDraft,
    SpanIntent,
    resolve_parentage,
)
from wardex_sdk._enums import (
    CaptureSource,
    Direction,
    OperationName,
    Protocol,
    SpanKind,
    StatusCode,
)
from wardex_sdk._types import (
    Envelope,
    EnvelopeHeader,
    GenAIAttributes,
    HttpMeta,
    InputRef,
    InternalSpan,
    InternalStateSnapshot,
    SdkInfo,
    SpanContext,
    SpanId,
    ToolAttributes,
    TraceId,
    TransportAttributes,
    TransportTiming,
)


def _header() -> EnvelopeHeader:
    return EnvelopeHeader(
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
    )


def _span(**kw) -> InternalSpan:
    base = dict(
        context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
        parent_span_id=None,
        name="HTTP POST /v1/chat",
        kind=SpanKind.CLIENT,
        start_time_ns=1000,
        end_time_ns=2000,
        status=StatusCode.OK,
    )
    base.update(kw)
    return InternalSpan(**base)


def _envelope_with_span() -> Envelope:
    return Envelope(
        header=_header(),
        spans=(
            _span(
                server_address="api.openai.com",
                server_port=443,
                input_data=b"req-bytes",
                output_data=b"resp-bytes",
                transport=TransportAttributes(
                    protocol=Protocol.HTTP,
                    direction=Direction.OUTBOUND,
                    timing=TransportTiming(ttfb_ms=30.0),
                    http=HttpMeta(
                        method="POST",
                        url="https://api.openai.com/v1/chat",
                        status_code=200,
                    ),
                ),
            ),
        ),
    )


def _envelope_with_gen_ai(model: str, input_tokens: int) -> Envelope:
    return Envelope(
        header=_header(),
        spans=(
            _span(
                gen_ai=GenAIAttributes(
                    operation=OperationName.CHAT,
                    request_model=model,
                    input_tokens=input_tokens,
                    output_tokens=5,
                    temperature=0.7,
                    finish_reasons=("stop",),
                ),
            ),
        ),
    )


def _envelope_no_spans() -> Envelope:
    return Envelope(header=_header())


def _envelope_with_snapshot_only() -> Envelope:
    snap = InternalStateSnapshot(
        trace_id=TraceId(b"\x03" * 16),
        span_id=SpanId(b"\x04" * 8),
        timestamp_ns=999,
        snapshot_type="turn_start",
        turn_index=1,
        conversation_state=b"history",
        input_refs=(InputRef(key="doc.md", content_hash="sha256:abc"),),
    )
    return Envelope(header=_header(), state_snapshots=(snap,))


def _first_span(env: Envelope) -> dict:
    data = _wardex_native.codec.encode_otlp_traces(env)
    d = _wardex_native.codec.decode_otlp_traces(data)
    return d["resource_spans"][0]["scope_spans"][0]["spans"][0]


def test_encode_decode_roundtrip_core_fields():
    env = _envelope_with_span()
    data = _wardex_native.codec.encode_otlp_traces(env)
    assert isinstance(data, bytes) and len(data) > 0
    d = _wardex_native.codec.decode_otlp_traces(data)
    span = d["resource_spans"][0]["scope_spans"][0]["spans"][0]
    assert span["name"] == "HTTP POST /v1/chat"
    assert span["kind"] == 3  # OTLP CLIENT
    assert span["status"]["code"] == 1  # OTLP OK
    assert span["start_time_unix_nano"] == 1000
    assert span["end_time_unix_nano"] == 2000
    assert span["trace_id"] == "01" * 16
    assert span["span_id"] == "02" * 8
    attrs = span["attributes"]
    assert attrs["server.address"] == "api.openai.com"
    assert attrs["server.port"] == 443
    assert attrs["network.protocol.name"] == "http"
    assert attrs["http.request.method"] == "POST"
    assert attrs["http.response.status_code"] == 200
    assert attrs["url.full"] == "https://api.openai.com/v1/chat"
    # Raw I/O leaves as strings, never OTLP bytes_value: backends that
    # re-serialize attributes to JSON (Arize Phoenix) drop the whole span on a
    # bytes attribute — silently, with an HTTP 200.
    assert attrs["wardex.input_data"] == "req-bytes"
    assert attrs["wardex.output_data"] == "resp-bytes"
    assert "wardex.input_data.encoding" not in attrs
    assert "wardex.output_data.encoding" not in attrs


def test_non_utf8_payload_becomes_base64_with_encoding_marker():
    """Binary payloads (gRPC frames, compressed bodies) can't ship verbatim in
    a string attribute; they go base64 with a `.encoding` companion so a
    consumer can tell encoded binary from text that merely looks like base64."""
    import base64
    from dataclasses import replace

    binary = b"\x89PNG\xff\x00binary"
    env = _envelope_with_span()
    span = replace(env.spans[0], input_data=binary, output_data=b"plain text")
    attrs = _first_span(Envelope(header=env.header, spans=(span,)))["attributes"]
    assert attrs["wardex.input_data"] == base64.b64encode(binary).decode()
    assert attrs["wardex.input_data.encoding"] == "base64"
    assert attrs["wardex.output_data"] == "plain text"
    assert "wardex.output_data.encoding" not in attrs


def test_utf8_with_nul_is_binary_not_text():
    """U+0000 is valid UTF-8, but Postgres-backed ingests (Phoenix-on-Postgres,
    Langfuse) reject any string containing NUL — shipping it verbatim would
    reintroduce the exact silent span loss the string surface exists to prevent.
    Zero-value
    protobuf/gRPC payload bytes are the realistic producer."""
    import base64
    from dataclasses import replace

    payload = b"name: alice\x00\x00\x00\x00"  # valid UTF-8, contains NUL
    env = _envelope_with_span()
    span = replace(env.spans[0], input_data=payload)
    attrs = _first_span(Envelope(header=env.header, spans=(span,)))["attributes"]
    assert attrs["wardex.input_data"] == base64.b64encode(payload).decode()
    assert attrs["wardex.input_data.encoding"] == "base64"


def test_user_extra_cannot_collide_with_or_spoof_the_encoding_companion():
    """`<key>.encoding` is the debyte pass's namespace for keys it rewrote.
    A user extra sitting on that key would either duplicate the companion
    (duplicate OTLP keys — backend dedup order decides which wins) or claim an
    encoding the verbatim branch never applied. Both get dropped; a user
    `.encoding` suffix on a key that never carried bytes is left alone."""
    from dataclasses import replace

    env = _envelope_with_span()
    span = replace(
        env.spans[0],
        input_data=b"\xff\xfebinary",  # -> base64: companion must win
        output_data=b"plain text",  # -> verbatim: spoofed marker must vanish
        extra=(
            ("wardex.input_data.encoding", "gzip"),
            ("wardex.output_data.encoding", "base64"),
            ("myapp.blob.encoding", "hex"),  # not a rewritten key: untouched
        ),
    )
    data = _wardex_native.codec.encode_otlp_traces(Envelope(header=env.header, spans=(span,)))
    d = _wardex_native.codec.decode_otlp_traces(data)
    span_out = d["resource_spans"][0]["scope_spans"][0]["spans"][0]
    keys = [k for k in span_out["attributes"] if k.endswith(".encoding")]
    attrs = span_out["attributes"]
    assert attrs["wardex.input_data.encoding"] == "base64"
    assert "wardex.output_data.encoding" not in attrs
    assert attrs["myapp.blob.encoding"] == "hex"
    assert sorted(keys) == ["myapp.blob.encoding", "wardex.input_data.encoding"]


def test_gen_ai_flattened_to_attributes():
    env = _envelope_with_gen_ai(model="gpt-4o", input_tokens=10)
    attrs = _first_span(env)["attributes"]
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["gen_ai.usage.input_tokens"] == 10
    assert attrs["gen_ai.usage.output_tokens"] == 5
    assert attrs["gen_ai.request.temperature"] == 0.7
    assert attrs["gen_ai.operation.name"] == "chat"
    # semconv declares the list-valued keys as arrays; the CSV join this
    # replaces destroyed element boundaries on any value with a comma.
    assert attrs["gen_ai.response.finish_reasons"] == ["stop"]


def test_conversation_and_evaluation_blocks_reach_the_otlp_attributes():
    """Measured before this: a span built with `wardex.conversation()` or an
    adapter's `set_evaluation` left the process with NEITHER key — the two
    blocks were declared and never marshalled. The receiver is what the
    README promises `gen_ai.conversation.id` to, so this asserts there."""
    from wardex_sdk._types import ConversationContext, EvaluationAttributes

    env = Envelope(
        header=_header(),
        spans=(
            _span(
                conversation=ConversationContext(conversation_id="conv-123"),
                evaluation=EvaluationAttributes(name="block_input", score_label="tripwire"),
            ),
        ),
    )
    attrs = _first_span(env)["attributes"]
    assert attrs["gen_ai.conversation.id"] == "conv-123"
    assert attrs["gen_ai.evaluation.name"] == "block_input"
    assert attrs["gen_ai.evaluation.score.label"] == "tripwire"


def test_cache_and_reasoning_tokens_ship_under_the_semconv_dot_spellings():
    """The dataclass fields keep their snake_case names; only the wire key
    moved to semconv's dot spellings (defined since semconv 1.40.0)."""
    env = Envelope(
        header=_header(),
        spans=(
            _span(
                gen_ai=GenAIAttributes(
                    operation=OperationName.CHAT,
                    cache_read_input_tokens=3,
                    cache_creation_input_tokens=2,
                    reasoning_output_tokens=7,
                ),
            ),
        ),
    )
    attrs = _first_span(env)["attributes"]
    assert attrs["gen_ai.usage.cache_read.input_tokens"] == 3
    assert attrs["gen_ai.usage.cache_creation.input_tokens"] == 2
    assert attrs["gen_ai.usage.reasoning.output_tokens"] == 7
    # The old underscore spellings are gone, not doubled.
    assert "gen_ai.usage.cache_read_input_tokens" not in attrs
    assert "gen_ai.usage.cache_creation_input_tokens" not in attrs
    assert "gen_ai.usage.reasoning_output_tokens" not in attrs


def test_stop_sequences_and_encoding_formats_are_arrays_on_the_wire():
    env = Envelope(
        header=_header(),
        spans=(
            _span(
                gen_ai=GenAIAttributes(
                    operation=OperationName.EMBEDDINGS,
                    stop_sequences=("a,b", "c"),
                    encoding_formats=("float", "base64"),
                ),
            ),
        ),
    )
    attrs = _first_span(env)["attributes"]
    # "a,b" survives as ONE element — the case CSV could not carry.
    assert attrs["gen_ai.request.stop_sequences"] == ["a,b", "c"]
    assert attrs["gen_ai.request.encoding_formats"] == ["float", "base64"]


def test_an_llm_span_is_named_for_its_operation_and_model():
    """gen_ai semconv names an LLM span `{operation} {model}`.

    The transport name it replaces — `HTTP POST /v1/chat` — is the same string
    for every model, prompt and provider behind one endpoint, and span name is
    the axis every backend groups by. A latency or cost breakdown over an
    agent's LLM calls collapsed into a single bucket that answers no question
    anyone asks, while the two facts a reader groups by sat one level down in
    the attributes.
    """
    assert _first_span(_envelope_with_gen_ai(model="gpt-4o", input_tokens=10))["name"] == (
        "chat gpt-4o"
    )


def test_the_model_that_answered_names_the_span_when_the_request_recorded_none():
    """An assembled turn knows the response model before it knows the requested
    one, and the SDK has it one attribute away — so a span named for the model
    that answered beats one named for no model at all."""
    env = Envelope(
        header=_header(),
        spans=(
            _span(
                gen_ai=GenAIAttributes(
                    operation=OperationName.CHAT, response_model="claude-sonnet-5"
                )
            ),
        ),
    )
    assert _first_span(env)["name"] == "chat claude-sonnet-5"


def test_an_llm_span_with_no_model_at_all_keeps_its_own_name():
    """Not `embeddings` and not `embeddings unknown`: the first is strictly less
    than the name already there, the second invents a model of that name for a
    backend to aggregate."""
    env = Envelope(
        header=_header(),
        spans=(_span(gen_ai=GenAIAttributes(operation=OperationName.EMBEDDINGS)),),
    )
    assert _first_span(env)["name"] == "HTTP POST /v1/chat"


def test_a_span_without_llm_semantics_keeps_its_own_name():
    """The negative control for the rename above: it is a rename for calls to a
    model and a no-op for everything else, so a plain HTTP span still says what
    it was."""
    assert _first_span(_envelope_with_span())["name"] == "HTTP POST /v1/chat"


def test_a_tool_span_the_builder_produced_keeps_its_subject():
    """Built by `SpanDraft`, not by hand: EVERY vocabulary span carries
    `gen_ai.operation.name`, so reading that key as "this is a call to a model"
    renames the whole vocabulary. `execute_tool` requires a tool block and can
    carry no request model, so the collapse would be total — every tool call in
    every run in one bucket, which is the failure the rename exists to remove.
    """
    draft = SpanDraft(
        resolve_parentage(Ambient(None, None, None), AMBIENT),
        intent=SpanIntent.EXECUTE_TOOL,
        subject="Bash",
        source=CaptureSource.ADAPTER,
        start_ns=1,
    )
    draft.set_tool(ToolAttributes(name="Bash"))
    draft.set_status(StatusCode.OK)
    span = draft.finish(end_ns=2)
    assert span.name == "execute_tool Bash"
    env = Envelope(header=_header(), spans=(span,))
    assert _first_span(env)["name"] == "execute_tool Bash"


def test_a_decorator_named_span_keeps_the_name_the_host_chose():
    """`@wardex.tool(name="search_docs")` builds a MANUAL span with an operation
    LABEL and no tool block, because the decorator's `tool=` argument is
    optional. The name is the host's and it is a published API, so `search_docs`
    has to survive the one transport exported from the package root."""
    draft = SpanDraft.manual(
        resolve_parentage(Ambient(None, None, None), AMBIENT),
        name="search_docs",
        kind=SpanKind.INTERNAL,
        start_ns=1,
    )
    draft.set_operation_label(OperationName.EXECUTE_TOOL)
    draft.set_status(StatusCode.OK)
    span = draft.finish(end_ns=2)
    env = Envelope(header=_header(), spans=(span,))
    assert _first_span(env)["attributes"]["gen_ai.operation.name"] == "execute_tool"
    assert _first_span(env)["name"] == "search_docs"


def test_resource_identity_comes_from_the_configured_resource():
    """`service.name`/`service.version`/`deployment.environment.name` are the
    APP's, read off `EnvelopeHeader.resource` — never off SdkInfo, which
    describes the SDK doing the exporting."""
    from wardex_sdk._types import ResourceInfo

    env = Envelope(
        header=EnvelopeHeader(
            event_id="evt-1",
            api_key="k",
            sdk=_header().sdk,
            sent_at_ns=42,
            resource=ResourceInfo(
                service_name="checkout-api",
                release="1.2.3",
                environment="staging",
                process_pid=4242,
            ),
        ),
        spans=(_span(),),
    )
    d = _wardex_native.codec.decode_otlp_traces(_wardex_native.codec.encode_otlp_traces(env))
    res_attrs = d["resource_spans"][0]["resource"]["attributes"]
    assert res_attrs["service.name"] == "checkout-api"
    assert res_attrs["service.version"] == "1.2.3"
    assert res_attrs["deployment.environment.name"] == "staging"
    # The one per-PROCESS resource attribute: an int, exactly as stamped.
    assert res_attrs["process.pid"] == 4242
    assert res_attrs["telemetry.sdk.name"] == "wardex"
    assert res_attrs["telemetry.sdk.version"] == "0.1.0"
    assert res_attrs["telemetry.sdk.language"] == "python"


def test_an_unnamed_service_is_unknown_service_never_the_sdk_name():
    """The blocker this slice fixes: every app exported as
    `service.name = "wardex.python"`, making two services one service in every
    backend. Unconfigured, the fallback is semconv's own shape, and the
    unconfigured release/environment emit no key at all."""
    env = _envelope_with_span()  # header carries no ResourceInfo
    d = _wardex_native.codec.decode_otlp_traces(_wardex_native.codec.encode_otlp_traces(env))
    res_attrs = d["resource_spans"][0]["resource"]["attributes"]
    assert res_attrs["service.name"] == "unknown_service:python"
    # An unstamped pid (proto3 zero) emits no key: `process.pid = 0` would
    # claim the scheduler rather than say "not stamped".
    assert "process.pid" not in res_attrs
    assert res_attrs["telemetry.sdk.name"] == "wardex"
    assert "service.version" not in res_attrs
    assert "deployment.environment.name" not in res_attrs


def test_instrumentation_scope():
    env = _envelope_with_span()
    d = _wardex_native.codec.decode_otlp_traces(_wardex_native.codec.encode_otlp_traces(env))
    scope = d["resource_spans"][0]["scope_spans"][0]["scope"]
    assert scope["name"] == "wardex.python"
    assert scope["version"] == "0.1.0"


def test_determinism():
    env = _envelope_with_span()
    a = _wardex_native.codec.encode_otlp_traces(env)
    b = _wardex_native.codec.encode_otlp_traces(env)
    assert a == b


def test_empty_envelope_no_resource_spans():
    env = _envelope_no_spans()
    d = _wardex_native.codec.decode_otlp_traces(_wardex_native.codec.encode_otlp_traces(env))
    assert d["resource_spans"] == []


def test_state_snapshot_skipped():
    env = _envelope_with_snapshot_only()
    d = _wardex_native.codec.decode_otlp_traces(_wardex_native.codec.encode_otlp_traces(env))
    assert d["resource_spans"] == []


# ==========================================================================
# What wardex knows about its own uncertainty, on the path that leaves the process
# ==========================================================================


def _envelope_with_uncertainty() -> Envelope:
    from wardex_sdk._assembly import Limitation
    from wardex_sdk._assembly._parentage import ParentSource
    from wardex_sdk._types import CaptureIntegrity, CorrelationInfo

    return Envelope(
        header=_header(),
        spans=(
            _span(
                correlation=CorrelationInfo(
                    strategy=ParentSource.UNIT_SOLE,
                    confidence=0.5,
                    request_id="toolu_9",
                ),
                capture_integrity=CaptureIntegrity(
                    limitations=(
                        Limitation.UNIT_INFERRED_SOLE,
                        Limitation.BODY_CAP_EXCEEDED,
                    ),
                    request_body_captured=True,
                    response_body_captured=False,
                    truncated=True,
                    dropped_chunk_count=3,
                ),
            ),
        ),
    )


def test_a_guessed_edge_says_so_on_the_otlp_wire():
    """OTLP is the only transport exported from the package root, so a marker
    that reaches the wardex envelope and not this encoder reaches nobody.

    Every marker this SDK spends its design on says what could NOT be
    established. Stripping them here left a 0.5 guess and a 1.0 fact
    indistinguishable for every user on the documented path — the silent-loss
    shape `events_to_otlp` names for a different field in the same file.
    """
    attrs = _first_span(_envelope_with_uncertainty())["attributes"]

    assert attrs["wardex.parent_source"] == "unit_sole"
    assert attrs["wardex.parent_confidence"] == 0.5
    assert attrs["wardex.correlation.request_id"] == "toolu_9"


def test_every_limitation_a_span_carries_reaches_the_otlp_wire():
    attrs = _first_span(_envelope_with_uncertainty())["attributes"]

    assert attrs["wardex.limitations"] == ["unit_inferred_sole", "body_cap_exceeded"]


def test_what_was_and_was_not_captured_reaches_the_otlp_wire():
    """The four capture flags are always present because FALSE is their
    informative reading; truncation and drops are events, so they appear only
    when they happened."""
    attrs = _first_span(_envelope_with_uncertainty())["attributes"]

    assert attrs["wardex.capture.request_body"] is True
    assert attrs["wardex.capture.response_body"] is False
    assert attrs["wardex.capture.truncated"] is True
    assert attrs["wardex.capture.dropped_chunks"] == 3


def test_a_span_with_nothing_to_report_carries_no_uncertainty_attributes():
    """Absence has to stay legible: a span that reports no limitation must not
    be padded with keys that make it look examined and cleared."""
    attrs = _first_span(_envelope_with_span())["attributes"]

    assert "wardex.limitations" not in attrs
    assert "wardex.capture.truncated" not in attrs
    assert "wardex.parent_source" not in attrs


# ==========================================================================
# OTel wire alignment — error.type, tool payload keys, SSE, url.full, SpanKind
# ==========================================================================


def test_error_type_is_an_attribute_and_the_status_message_survives():
    """`error.type` is semconv's home for the exception class; it used to be
    substituted INTO `Status.message`, destroying the one field a backend
    renders as "what went wrong" to relabel it with a fact that now travels
    beside it."""
    env = Envelope(
        header=_header(),
        spans=(_span(status=StatusCode.ERROR, status_message="boom", error_type="TimeoutError"),),
    )
    span = _first_span(env)
    assert span["status"]["message"] == "boom"
    assert span["attributes"]["error.type"] == "TimeoutError"


def test_a_span_without_an_error_type_carries_no_error_type_key():
    env = Envelope(
        header=_header(),
        spans=(_span(status=StatusCode.ERROR, status_message="boom"),),
    )
    span = _first_span(env)
    assert span["status"]["message"] == "boom"
    assert "error.type" not in span["attributes"]


def test_an_execute_tool_payload_ships_under_the_semconv_tool_keys():
    """Same pipeline, same masking, same caps — only the key differs, and only
    for `execute_tool`: that operation's payload has a semconv home
    (`gen_ai.tool.call.arguments`/`result`)."""
    env = Envelope(
        header=_header(),
        spans=(
            _span(
                gen_ai=GenAIAttributes(operation=OperationName.EXECUTE_TOOL),
                tool=ToolAttributes(name="Bash"),
                input_data=b'{"command":"ls"}',
                output_data=b"README.md",
            ),
        ),
    )
    attrs = _first_span(env)["attributes"]
    assert attrs["gen_ai.tool.call.arguments"] == '{"command":"ls"}'
    assert attrs["gen_ai.tool.call.result"] == "README.md"
    assert "wardex.input_data" not in attrs
    assert "wardex.output_data" not in attrs


def test_every_other_operation_keeps_the_wardex_payload_keys():
    env = _envelope_with_gen_ai(model="gpt-4o", input_tokens=10)
    span = replace_first_span_payload(env, input_data=b"prompt", output_data=b"answer")
    attrs = _first_span(span)["attributes"]
    assert attrs["wardex.input_data"] == "prompt"
    assert attrs["wardex.output_data"] == "answer"
    assert "gen_ai.tool.call.arguments" not in attrs


def replace_first_span_payload(env: Envelope, **kw) -> Envelope:
    from dataclasses import replace

    return Envelope(header=env.header, spans=(replace(env.spans[0], **kw),))


def test_an_sse_span_is_http_on_the_wire_and_sse_under_the_wardex_key():
    """`network.protocol.name = "sse"` fails every backend's HTTP grouping —
    SSE is a framing over HTTP. The observed protocol survives under
    `wardex.transport.protocol`; every other protocol is unchanged."""
    env = Envelope(
        header=_header(),
        spans=(_span(transport=TransportAttributes(protocol=Protocol.SSE)),),
    )
    attrs = _first_span(env)["attributes"]
    assert attrs["network.protocol.name"] == "http"
    assert attrs["wardex.transport.protocol"] == "sse"

    env = Envelope(
        header=_header(),
        spans=(_span(transport=TransportAttributes(protocol=Protocol.GRPC)),),
    )
    attrs = _first_span(env)["attributes"]
    assert attrs["network.protocol.name"] == "grpc"
    assert "wardex.transport.protocol" not in attrs


def test_url_full_is_query_stripped_and_absent_when_no_url_was_captured():
    """Query strings are where credentials and PII ride (`?api_key=`); the
    exported URL is for grouping, not replay, so everything from `?` (and `#`)
    is dropped. No captured URL, no key."""

    def _with_url(url: str) -> Envelope:
        return Envelope(
            header=_header(),
            spans=(
                _span(
                    transport=TransportAttributes(
                        protocol=Protocol.HTTP,
                        http=HttpMeta(method="POST", url=url, status_code=200),
                    ),
                ),
            ),
        )

    attrs = _first_span(_with_url("https://api.example.com/v1/chat?api_key=sk-x#frag"))[
        "attributes"
    ]
    assert attrs["url.full"] == "https://api.example.com/v1/chat"
    assert "sk-x" not in str(attrs)

    attrs = _first_span(_with_url(""))["attributes"]
    assert "url.full" not in attrs


def test_producer_and_consumer_kinds_reach_the_otlp_wire():
    """The SDK pitches Celery/Kafka propagation and could not express the kinds
    those spans are. OTLP numbering: PRODUCER=4, CONSUMER=5."""
    assert (
        _first_span(Envelope(header=_header(), spans=(_span(kind=SpanKind.PRODUCER),)))["kind"] == 4
    )
    assert (
        _first_span(Envelope(header=_header(), spans=(_span(kind=SpanKind.CONSUMER),)))["kind"] == 5
    )


# --- G2: GenAIAttributes -> wire, exhaustively --------------------------------

#: Declared exceptions to "every field reaches the wire", with their MEASURED
#: reason — a wrong reason here turns a discovery into permanent furniture.
_UNFLATTENED = {
    "system_instructions": (
        "declared but never populated by any producer — the seam ships "
        "LlmSemantics.system_instructions directly as the "
        "gen_ai.system_instructions extra; disposition in follow-up"
    ),
    "tool_definitions_hash": (
        "declared but never populated by any producer; disposition in follow-up"
    ),
}


def test_every_gen_ai_field_reaches_the_wire():
    """Build a GenAIAttributes with a distinct sentinel per field, encode it,
    and require every sentinel to surface in the OTLP attributes. A field
    added to the dataclass without a `codec.rs` table row otherwise ships
    None forever with no failing test — the exact drift that made
    system_instructions and tool_definitions_hash dead fields.
    """
    import dataclasses

    sentinels: dict[str, object] = {}
    kwargs: dict[str, object] = {}
    for i, field in enumerate(dataclasses.fields(GenAIAttributes)):
        name = field.name
        if name in _UNFLATTENED:
            continue
        ann = str(field.type)
        if name == "operation":
            value = OperationName.CHAT
            probe = "chat"
        elif name == "provider":
            value = probe = "sentinel-provider"
        elif "tuple[str, ...]" in ann:
            value = (f"sentinel-{i}",)
            probe = [f"sentinel-{i}"]
        elif "int" in ann and "float" not in ann:
            value = probe = 1_000_000 + i
        elif "float" in ann:
            value = probe = float(f"0.{i + 1}")
        elif "bool" in ann:
            value = probe = True
        else:
            value = probe = f"sentinel-{i}"
        kwargs[name] = value
        sentinels[name] = probe

    env = Envelope(header=_header(), spans=(_span(gen_ai=GenAIAttributes(**kwargs)),))
    attrs = _first_span(env)["attributes"]
    values = list(attrs.values())
    missing = [name for name, probe in sentinels.items() if probe not in values]
    assert not missing, (
        f"GenAIAttributes fields that never reached the OTLP wire: {missing}.\n"
        "Add the field to the flatten table in bindings/python/src/codec.rs, "
        "or declare it in _UNFLATTENED with its measured reason."
    )


def test_reasoning_previous_response_and_status_ship_under_their_semconv_keys():
    env = Envelope(
        header=_header(),
        spans=(
            _span(
                gen_ai=GenAIAttributes(
                    operation=OperationName.CHAT,
                    reasoning_level="high",
                    previous_response_id="resp_prev",
                    response_status="completed",
                ),
            ),
        ),
    )
    attrs = _first_span(env)["attributes"]
    assert attrs["gen_ai.request.reasoning.level"] == "high"
    assert attrs["gen_ai.request.previous_response.id"] == "resp_prev"
    assert attrs["gen_ai.response.status"] == "completed"


def test_embeddings_dimension_count_reaches_the_wire():
    from wardex_sdk._types import EmbeddingsAttributes

    env = Envelope(
        header=_header(),
        spans=(
            _span(
                gen_ai=GenAIAttributes(operation=OperationName.EMBEDDINGS),
                embeddings=EmbeddingsAttributes(dimension_count=256),
            ),
        ),
    )
    attrs = _first_span(env)["attributes"]
    assert attrs["gen_ai.embeddings.dimension.count"] == 256


def test_wardex_usage_mirror_ints_ride_as_int_values():
    """T-C1 — the mirror's integers reach OTLP as IntValue attributes (the
    type that makes them structurally unmaskable), through the ordinary
    extra passthrough."""
    env = Envelope(
        header=_header(),
        spans=(
            _span(
                extra=(
                    ("wardex.usage.input_tokens", 52),
                    ("wardex.usage.cache_creation.ephemeral_1h_input_tokens", 64),
                    ("wardex.usage.service_tier", "standard"),
                    ("wardex.usage_leaves.dropped_count", 3),
                ),
            ),
        ),
    )
    attrs = _first_span(env)["attributes"]
    assert attrs["wardex.usage.input_tokens"] == 52
    assert attrs["wardex.usage.cache_creation.ephemeral_1h_input_tokens"] == 64
    assert attrs["wardex.usage.service_tier"] == "standard"
    assert attrs["wardex.usage_leaves.dropped_count"] == 3
