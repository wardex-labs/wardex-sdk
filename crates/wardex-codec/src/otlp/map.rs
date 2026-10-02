//! wardex envelope → OTLP traces: the semantic mapping.
//!
//! This is the half of the OTLP export that decides MEANING — what a span is
//! called, which keys carry the gen_ai / http / server semantics, what the
//! resource and scope around them say — as opposed to [`super::encode_traces`],
//! which only serializes what this module decided.
//!
//! It lived in the PyO3 binding until now, which made every one of those
//! decisions Python's private opinion. A Node or Java binding would have had to
//! re-derive span naming and each `wardex.*` key from prose, and the first
//! disagreement between two SDKs would have surfaced as two differently shaped
//! traces in one backend rather than as a failing build. Here it is one
//! implementation over the schema every binding already marshals into.
//!
//! ## The input is the wire schema, not a parallel model
//!
//! [`pb::Envelope`] is the input on purpose. Declaring a second Rust struct
//! tree to feed this mapping would be a third copy of a model the `.proto`
//! already owns (design §6.6) — the same duplication `vocab` exists to remove —
//! and a binding would then have to keep two marshaling paths in step.
//!
//! The cost is proto3's default semantics: a field a host language holds as
//! optional arrives as its zero value, so "absent" and "empty string" / "port
//! 0" / "no parent source" become the same input. Every site below says what it
//! does with the zero value, and none of the collapsed cases is reachable from
//! this SDK — `server.port` 0 is not a port, `error_type` is an exception class
//! name, and the closed vocabularies are asserted member-for-member against
//! their Python enums before anything ships.

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use wardex_limits::Limits;

use super::otlp_pb;
use crate::proto::wardex::v1 as pb;
use crate::vocab;

/// Who produced this export — the two facts the mapping cannot read off an
/// envelope because they describe the SDK doing the exporting, not the spans.
///
/// `scope_name` is deliberately not `SdkInfo.name`: `service.name` names the
/// user's application and a host is free to rename it, while
/// `InstrumentationScope.name` names the library that produced the spans and
/// must not move when it does.
pub struct Producer<'a> {
    /// `telemetry.sdk.language` — `"python"`, `"node"`, `"java"`.
    pub language: &'a str,
    /// `InstrumentationScope.name` — e.g. `"wardex.python"`.
    pub scope_name: &'a str,
}

// --- OTLP KeyValue builders ---

fn kv(key: &str, value: otlp_pb::common::any_value::Value) -> otlp_pb::common::KeyValue {
    otlp_pb::common::KeyValue {
        key: key.into(),
        value: Some(otlp_pb::common::AnyValue { value: Some(value) }),
    }
}

fn kv_str(key: &str, v: &str) -> otlp_pb::common::KeyValue {
    kv(
        key,
        otlp_pb::common::any_value::Value::StringValue(v.into()),
    )
}
fn kv_int(key: &str, v: i64) -> otlp_pb::common::KeyValue {
    kv(key, otlp_pb::common::any_value::Value::IntValue(v))
}
fn kv_bool(key: &str, v: bool) -> otlp_pb::common::KeyValue {
    kv(key, otlp_pb::common::any_value::Value::BoolValue(v))
}
fn kv_f64(key: &str, v: f64) -> otlp_pb::common::KeyValue {
    kv(key, otlp_pb::common::any_value::Value::DoubleValue(v))
}
fn kv_bytes(key: &str, v: Vec<u8>) -> otlp_pb::common::KeyValue {
    kv(key, otlp_pb::common::any_value::Value::BytesValue(v))
}
fn kv_strs(key: &str, vs: Vec<String>) -> otlp_pb::common::KeyValue {
    kv(
        key,
        otlp_pb::common::any_value::Value::ArrayValue(otlp_pb::common::ArrayValue {
            values: vs
                .into_iter()
                .map(|v| otlp_pb::common::AnyValue {
                    value: Some(otlp_pb::common::any_value::Value::StringValue(v)),
                })
                .collect(),
        }),
    )
}

/// wardex `AnyValue` → OTLP `AnyValue`. The two schemas declare the same
/// variants under different numbers, so this is a remap and not a conversion.
///
/// By value throughout this module: the envelope is built for this mapping and
/// dropped by it, so a borrow would buy nothing and cost a second copy of every
/// captured payload — alive at the same time as the first, on the background
/// export path, where `max_buffer_bytes` is sized against ONE.
fn any_value(v: pb::AnyValue) -> otlp_pb::common::AnyValue {
    use otlp_pb::common::any_value::Value as Otlp;
    use pb::any_value::Value as Wardex;
    let value = match v.value {
        Some(Wardex::StringValue(s)) => Some(Otlp::StringValue(s)),
        Some(Wardex::IntValue(i)) => Some(Otlp::IntValue(i)),
        Some(Wardex::DoubleValue(d)) => Some(Otlp::DoubleValue(d)),
        Some(Wardex::BoolValue(b)) => Some(Otlp::BoolValue(b)),
        // Bytes pass through here untouched: masking must still see the raw
        // payload. `strip_bytes_values` removes every bytes_value from the
        // request after masking, right before serialization.
        Some(Wardex::BytesValue(b)) => Some(Otlp::BytesValue(b)),
        // Containers remap recursively. Dropping them was survivable while
        // nothing put one in `extra`; the list-valued gen_ai keys
        // (stop_sequences, finish_reasons, encoding_formats) ship as
        // ArrayValue now, so a silent None here would delete exactly them.
        Some(Wardex::ArrayValue(arr)) => Some(Otlp::ArrayValue(otlp_pb::common::ArrayValue {
            values: arr.values.into_iter().map(any_value).collect(),
        })),
        Some(Wardex::KvlistValue(kvl)) => Some(Otlp::KvlistValue(otlp_pb::common::KeyValueList {
            values: kvl.values.into_iter().map(key_value).collect(),
        })),
        None => None,
    };
    otlp_pb::common::AnyValue { value }
}

fn key_value(kv: pb::KeyValue) -> otlp_pb::common::KeyValue {
    otlp_pb::common::KeyValue {
        key: kv.key,
        value: kv.value.map(any_value),
    }
}

// --- enums (the numbers differ between the two schemas) ---

/// wardex `SpanKind` → OTLP `SpanKind`. The names line up, the NUMBERS do not
/// (wardex CLIENT is 2, OTLP CLIENT is 3), so the pass has to be explicit.
/// Matching on the generated enum rather than on a string keeps the compiler in
/// the loop when the schema gains a kind.
fn span_kind(kind: i32) -> i32 {
    use otlp_pb::trace::span::SpanKind as Otlp;
    (match pb::SpanKind::try_from(kind) {
        Ok(pb::SpanKind::Internal) => Otlp::Internal,
        Ok(pb::SpanKind::Client) => Otlp::Client,
        Ok(pb::SpanKind::Server) => Otlp::Server,
        Ok(pb::SpanKind::Producer) => Otlp::Producer,
        Ok(pb::SpanKind::Consumer) => Otlp::Consumer,
        Ok(pb::SpanKind::Unspecified) | Err(_) => Otlp::Unspecified,
    }) as i32
}

/// wardex `StatusCode` → OTLP `StatusCode`.
fn status_code(code: i32) -> i32 {
    use otlp_pb::trace::status::StatusCode as Otlp;
    (match pb::StatusCode::try_from(code) {
        Ok(pb::StatusCode::Ok) => Otlp::Ok,
        Ok(pb::StatusCode::Error) => Otlp::Error,
        Ok(pb::StatusCode::Unset) | Err(_) => Otlp::Unset,
    }) as i32
}

/// `CorrelationInfo.confidence` is a proto `float`, and widening the raw bits
/// puts `0.8999999761581421` on a dashboard's confidence bar where the sender
/// wrote 0.9. Round-tripping through the shortest decimal that identifies the
/// same `f32` recovers the number the field's declared 0.0–1.0 domain is
/// written in, which is the number a reader is being asked to trust.
fn widen(v: f32) -> f64 {
    v.to_string().parse().unwrap_or(v as f64)
}

// --- span ---

/// The attribute every limitation marker rides on.
///
/// Named once because two different passes write it now: the mapping projects
/// `CaptureIntegrity.limitation_codes` onto it, and the size passes at the
/// bottom of this file append to whatever that produced. A second spelling here
/// would put a span's markers in two attributes, one of which no dashboard
/// reads.
const LIMITATIONS_KEY: &str = "wardex.limitations";

/// `CorrelationInfo` and `CaptureIntegrity` → OTLP span attributes.
///
/// OTLP has no native home for either, so they travel under `wardex.*` the same
/// way a link's `reason` does. Leaving them out is not a smaller version of the
/// same export — it is the one that cannot be audited. Every marker this SDK
/// spends its design on says what it could NOT establish, and OTLP is the only
/// transport exported from the package root: a user on the documented path
/// would receive spans stripped of every "this edge is a guess" and every "this
/// body was truncated", with nothing to distinguish them from spans that had
/// nothing to report.
fn uncertainty(sp: &pb::Span, attrs: &mut Vec<otlp_pb::common::KeyValue>) {
    // HOW each span was captured — seen by an adapter's hook, rebuilt from
    // wire bytes, merged from a bridge — is the same grade of fact as the
    // markers below: it says how far the span can be trusted. The envelope has
    // carried it in a typed field all along and this mapping left it out, so
    // an OTLP backend could not tell an observed tool call from a
    // reconstructed one. Zero is "unset" and names nothing; a number this
    // build does not know says so in its own name.
    let sources: Vec<String> = sp
        .capture_sources
        .iter()
        .map(|n| vocab::capture_source_name(*n))
        .filter(|name| !name.is_empty())
        .collect();
    if !sources.is_empty() {
        set_attr(attrs, kv_strs("wardex.capture_sources", sources));
    }
    if let Some(c) = &sp.correlation {
        // Zero is "no parent source recorded", not a source: naming it would
        // make a span whose parentage was never interpreted look interpreted.
        let source = vocab::parent_source_name(c.parent_source);
        if !source.is_empty() {
            attrs.push(kv_str("wardex.parent_source", &source));
        }
        attrs.push(kv_f64("wardex.parent_confidence", widen(c.confidence)));
        // The identifier that was CONSULTED to pick a parent. Present only when
        // one was, so a non-null value is actionable rather than decorative.
        for (value, key) in [
            (&c.request_id, "wardex.correlation.request_id"),
            (&c.operation_id, "wardex.correlation.operation_id"),
            (&c.attempt_id, "wardex.correlation.attempt_id"),
        ] {
            if !value.is_empty() {
                attrs.push(kv_str(key, value));
            }
        }
    }
    if let Some(i) = &sp.capture_integrity {
        // Numbers in, names out — a consumer reading `[3, 17]` would have to
        // hold its own copy of the vocabulary to know what was lost (§6.6), and
        // a number this build does not know says so in its own name rather than
        // passing for "nothing to report".
        let markers: Vec<String> = i
            .limitation_codes
            .iter()
            .map(|n| vocab::limitation_name(*n))
            .collect();
        if !markers.is_empty() {
            attrs.push(kv_strs(LIMITATIONS_KEY, markers));
        }
        for (value, key) in [
            (i.request_headers_captured, "wardex.capture.request_headers"),
            (i.request_body_captured, "wardex.capture.request_body"),
            (
                i.response_headers_captured,
                "wardex.capture.response_headers",
            ),
            (i.response_body_captured, "wardex.capture.response_body"),
        ] {
            attrs.push(kv_bool(key, value));
        }
        // Emitted only when true / non-zero: unlike the four above, whose FALSE
        // is the informative reading, these describe an event that either
        // happened or did not.
        if i.truncated {
            attrs.push(kv_bool("wardex.capture.truncated", true));
        }
        if i.redacted {
            attrs.push(kv_bool("wardex.capture.redacted", true));
        }
        // What an envelope masked before it reached this mapping. On the
        // export path the envelope arrives unmasked and these are empty; the
        // OTLP masker writes the same three keys for what it replaces.
        if i.redaction_count > 0 {
            attrs.push(kv_int(
                "wardex.redaction.count",
                i64::from(i.redaction_count),
            ));
            let rules: Vec<String> = i
                .redaction_rules
                .iter()
                .map(|n| vocab::redaction_rule_name(*n))
                .collect();
            if !rules.is_empty() {
                attrs.push(kv_strs("wardex.redaction.rules", rules));
            }
            if !i.redaction_names.is_empty() {
                attrs.push(kv_strs("wardex.redaction.names", i.redaction_names.clone()));
            }
        }
        if i.dropped_chunk_count > 0 {
            attrs.push(kv_int(
                "wardex.capture.dropped_chunks",
                i.dropped_chunk_count as i64,
            ));
        }
    }
}

/// Set `kv`, REPLACING a same-keyed attribute rather than sitting beside it.
///
/// The typed fields below are the sender's one home for their values, and a
/// receiver of the envelope reads them there. `extra` is the host's to write,
/// so a host can have spelled `gen_ai.conversation.id` into it by hand — and
/// then two rules collide. Left in place it would be a duplicated key, of
/// which an OTLP decoder keeps one, the backend's choice which. Allowed to win
/// it would make the two export surfaces disagree about one span: the envelope
/// says the typed value, OTLP says the host's. The typed value wins on both,
/// once.
fn set_attr(attrs: &mut Vec<otlp_pb::common::KeyValue>, kv: otlp_pb::common::KeyValue) {
    attrs.retain(|have| have.key != kv.key);
    attrs.push(kv);
}

/// `Span.conversation` and `Span.call_site` → OTLP span attributes.
///
/// Both are typed fields on the envelope and the only place the sender writes
/// them, so this projection is what an OTLP backend sees of either. The
/// conversation keys are the ones the sender used when it flattened the block
/// itself, and each is emitted when the typed field holds something: proto3
/// cannot tell an empty string or a zero from "unset", so an empty id, an
/// empty session id and a turn of 0 emit nothing.
///
/// The call site takes semconv's stable `code.*` names. `code.function.name`
/// is defined as FULLY QUALIFIED, with the namespace inside it rather than
/// beside it, so the module is composed in; a module with no function names no
/// function and is left out.
fn typed_blocks(sp: &pb::Span, attrs: &mut Vec<otlp_pb::common::KeyValue>) {
    if let Some(c) = &sp.conversation {
        if !c.conversation_id.is_empty() {
            set_attr(attrs, kv_str("gen_ai.conversation.id", &c.conversation_id));
        }
        if !c.session_id.is_empty() {
            set_attr(
                attrs,
                kv_str("wardex.conversation.session_id", &c.session_id),
            );
        }
        if c.turn_index != 0 {
            set_attr(
                attrs,
                kv_int("wardex.conversation.turn_index", c.turn_index as i64),
            );
        }
    }
    if let Some(c) = &sp.call_site {
        if !c.file.is_empty() {
            set_attr(attrs, kv_str("code.file.path", &c.file));
        }
        if c.line != 0 {
            set_attr(attrs, kv_int("code.line.number", c.line as i64));
        }
        if !c.function.is_empty() {
            let name = if c.module.is_empty() {
                c.function.clone()
            } else {
                format!("{}.{}", c.module, c.function)
            };
            set_attr(attrs, kv_str("code.function.name", &name));
        }
    }
}

// --- where every wire field goes ---

/// Where one wire field lands in the OTLP export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OtlpHome {
    /// A span attribute under this key, carrying the field's own value (an
    /// enum by its vocabulary name, a millisecond interval as a double).
    /// Whether an unset or empty value still emits the key is said where the
    /// field is mapped; a value the SDK did not observe never does.
    Attribute(&'static str),
    /// A field of the OTLP span itself, or a block projected onto several
    /// keys by its own function in this module and asserted by that
    /// function's tests. The text says where.
    Projected(&'static str),
    /// Deliberately not exported. The text is why.
    NotExported(&'static str),
}

/// The census of the wire fields this mapping is answerable for: every field
/// of `Span`, and every leaf under `Span.transport`. Each says where it lands
/// in OTLP or why it does not.
///
/// It exists because a field the mapping never read used to vanish from the
/// OTLP export with no error and every test green: `workflow_name` never
/// reached it, and of the sixteen transport values the SDK measures only four
/// did. Two checks hold this table to the code. `span` and `transport` below
/// destructure their input exhaustively, so a field added to the schema does
/// not compile until it is read or bound to `_`; and the tests compare this
/// table against the field list in `span.proto` and run a span carrying a
/// sentinel in every `Attribute` field through the mapping. The Python suite
/// asks the same table, through the binding, whether each value the SDK
/// produces survives the whole way.
///
/// A `NotExported` transport field is one nothing in the SDK fills: its
/// reason is "no producer", and the Python census fails the moment a
/// producer appears, which is when its OTLP name has to be decided.
pub const WIRE_FIELDS: &[(&str, OtlpHome)] = &[
    // -- Span
    ("Span.trace_id", OtlpHome::Projected("OTLP Span.trace_id")),
    ("Span.span_id", OtlpHome::Projected("OTLP Span.span_id")),
    (
        "Span.parent_span_id",
        OtlpHome::Projected("OTLP Span.parent_span_id"),
    ),
    (
        "Span.name",
        OtlpHome::Projected("OTLP Span.name (`span_name` renames a model call)"),
    ),
    (
        "Span.kind",
        OtlpHome::Projected("OTLP Span.kind (`span_kind` renumbers)"),
    ),
    (
        "Span.start_time_unix_nano",
        OtlpHome::Projected("OTLP Span.start_time_unix_nano"),
    ),
    (
        "Span.end_time_unix_nano",
        OtlpHome::Projected("OTLP Span.end_time_unix_nano"),
    ),
    (
        "Span.status",
        OtlpHome::Projected("OTLP Span.status (`status_code` renumbers)"),
    ),
    (
        "Span.extra",
        OtlpHome::Projected(
            "span attributes, key for key, except a reserved wardex.transport.* name (`span`)",
        ),
    ),
    (
        "Span.events",
        OtlpHome::Projected("OTLP Span.events (`event`)"),
    ),
    (
        "Span.links",
        OtlpHome::Projected("OTLP Span.links (`link`)"),
    ),
    (
        "Span.dropped_extra_count",
        OtlpHome::NotExported(NO_PRODUCER_DROPPED),
    ),
    (
        "Span.dropped_events_count",
        OtlpHome::NotExported(NO_PRODUCER_DROPPED),
    ),
    (
        "Span.dropped_links_count",
        OtlpHome::NotExported(NO_PRODUCER_DROPPED),
    ),
    (
        "Span.input_data",
        OtlpHome::Projected("wardex.input_data, or gen_ai.tool.call.arguments on execute_tool"),
    ),
    (
        "Span.output_data",
        OtlpHome::Projected("wardex.output_data, or gen_ai.tool.call.result on execute_tool"),
    ),
    (
        "Span.transport",
        OtlpHome::Projected("its leaves, each listed below"),
    ),
    ("Span.error_type", OtlpHome::Attribute("error.type")),
    ("Span.server_address", OtlpHome::Attribute("server.address")),
    ("Span.server_port", OtlpHome::Attribute("server.port")),
    ("Span.workflow_name", OtlpHome::Attribute(WORKFLOW_NAME)),
    (
        "Span.call_site",
        OtlpHome::Projected(
            "code.file.path, code.line.number, code.function.name (`typed_blocks`)",
        ),
    ),
    (
        "Span.conversation",
        OtlpHome::Projected("gen_ai.conversation.id, wardex.conversation.* (`typed_blocks`)"),
    ),
    (
        "Span.capture_sources",
        OtlpHome::Projected("wardex.capture_sources (`uncertainty`)"),
    ),
    (
        "Span.capture_integrity",
        OtlpHome::Projected(
            "wardex.limitations, wardex.capture.*, wardex.redaction.* (`uncertainty`)",
        ),
    ),
    (
        "Span.correlation",
        OtlpHome::Projected(
            "wardex.parent_source, wardex.parent_confidence, wardex.correlation.* (`uncertainty`)",
        ),
    ),
    // -- TransportAttributes
    (
        "TransportAttributes.connection_id",
        OtlpHome::Attribute(T_CONNECTION_ID),
    ),
    (
        "TransportAttributes.protocol",
        OtlpHome::Projected(
            "network.protocol.name; SSE goes out as http plus wardex.transport.protocol",
        ),
    ),
    (
        "TransportAttributes.direction",
        OtlpHome::Attribute(T_DIRECTION),
    ),
    (
        "TransportAttributes.timing",
        OtlpHome::Projected("its leaves, each listed below"),
    ),
    (
        "TransportAttributes.request_size",
        OtlpHome::Attribute(T_REQUEST_SIZE),
    ),
    (
        "TransportAttributes.response_size",
        OtlpHome::Attribute(T_RESPONSE_SIZE),
    ),
    (
        "TransportAttributes.http",
        OtlpHome::Projected("its leaves, each listed below"),
    ),
    (
        "TransportAttributes.grpc",
        OtlpHome::Projected("its leaves, each listed below"),
    ),
    (
        "TransportAttributes.websocket",
        OtlpHome::Projected("its leaves, each listed below"),
    ),
    (
        "TransportAttributes.mcp",
        OtlpHome::Projected("its leaves, each listed below"),
    ),
    (
        "TransportAttributes.sse",
        OtlpHome::Projected("its leaves, each listed below"),
    ),
    (
        "TransportAttributes.a2a",
        OtlpHome::Projected("its leaves, each listed below"),
    ),
    (
        "TransportAttributes.request_blob_ref",
        OtlpHome::NotExported(NO_PRODUCER),
    ),
    (
        "TransportAttributes.response_blob_ref",
        OtlpHome::NotExported(NO_PRODUCER),
    ),
    (
        "TransportAttributes.request_modality",
        OtlpHome::NotExported(NO_PRODUCER),
    ),
    (
        "TransportAttributes.response_modality",
        OtlpHome::NotExported(NO_PRODUCER),
    ),
    (
        "TransportAttributes.is_streaming",
        OtlpHome::Attribute(T_IS_STREAMING),
    ),
    (
        "TransportAttributes.connection_reused",
        OtlpHome::Attribute(T_CONNECTION_REUSED),
    ),
    (
        "TransportTiming.tcp_connect_ms",
        OtlpHome::Attribute(T_TCP_CONNECT_MS),
    ),
    (
        "TransportTiming.tls_handshake_ms",
        OtlpHome::Attribute(T_TLS_HANDSHAKE_MS),
    ),
    ("TransportTiming.ttfb_ms", OtlpHome::Attribute(T_TTFB_MS)),
    (
        "TransportTiming.transfer_ms",
        OtlpHome::Attribute(T_TRANSFER_MS),
    ),
    ("TransportTiming.ttft_ms", OtlpHome::Attribute(T_TTFT_MS)),
    (
        "HttpMeta.method",
        OtlpHome::Attribute("http.request.method"),
    ),
    (
        "HttpMeta.status_code",
        OtlpHome::Attribute("http.response.status_code"),
    ),
    ("HttpMeta.url", OtlpHome::Attribute("url.full")),
    ("GrpcMeta.service", OtlpHome::NotExported(NO_PRODUCER_GRPC)),
    ("GrpcMeta.method", OtlpHome::NotExported(NO_PRODUCER_GRPC)),
    (
        "GrpcMeta.stream_id",
        OtlpHome::NotExported(NO_PRODUCER_GRPC),
    ),
    (
        "GrpcMeta.status_code",
        OtlpHome::NotExported(NO_PRODUCER_GRPC),
    ),
    ("GrpcMeta.encoding", OtlpHome::NotExported(NO_PRODUCER_GRPC)),
    (
        "GrpcMeta.decoded_payload",
        OtlpHome::NotExported(NO_PRODUCER_GRPC),
    ),
    (
        "WebSocketMeta.opcode",
        OtlpHome::NotExported(NO_PRODUCER_WS),
    ),
    (
        "WebSocketMeta.direction",
        OtlpHome::NotExported(NO_PRODUCER_WS),
    ),
    ("McpMeta.rpc_method", OtlpHome::Attribute(MCP_METHOD_NAME)),
    ("McpMeta.rpc_id", OtlpHome::Attribute(JSONRPC_REQUEST_ID)),
    ("SseMeta.event_type", OtlpHome::NotExported(NO_PRODUCER)),
    ("A2aMeta.task_id", OtlpHome::NotExported(NO_PRODUCER)),
    ("A2aMeta.transport", OtlpHome::NotExported(NO_PRODUCER)),
];

const NO_PRODUCER: &str = "no producer: nothing in the SDK fills this field, so every span \
     carries it empty; its OTLP name is decided by the change that first fills it";
const NO_PRODUCER_GRPC: &str = "no producer: nothing in the SDK fills GrpcMeta; a gRPC span \
     carries the same facts as the semconv rpc.* keys in `extra`, which reach OTLP as they are";
const NO_PRODUCER_WS: &str = "no producer: nothing in the SDK fills WebSocketMeta; a WebSocket \
     session span carries its facts as ws.* keys in `extra`, which reach OTLP as they are";
const NO_PRODUCER_DROPPED: &str = "no producer: the SDK reports dropped keys as the \
     extra_keys_dropped limitation and the wardex.usage_leaves.dropped_count key instead";

/// `gen_ai.workflow.name` — GenAI semantic conventions (development).
const WORKFLOW_NAME: &str = "gen_ai.workflow.name";
/// `mcp.method.name` — GenAI semantic conventions, MCP (development).
const MCP_METHOD_NAME: &str = "mcp.method.name";
/// `jsonrpc.request.id` — semantic conventions registry (development).
const JSONRPC_REQUEST_ID: &str = "jsonrpc.request.id";
/// The OTLP attribute names reserved for the transport values the SDK
/// observed: what [`transport`] writes, and nothing else. A host attribute
/// under one of them would ship as if the SDK had measured it, on a span with
/// no transport at all or beside the SDK's own reading as a second copy of the
/// key, of which a backend keeps one. So the mapping leaves every such host
/// attribute out before it writes its own and reports how many it left
/// ([`Mapped::reserved_overwritten`]): the SDK's value replaces the host's,
/// and where the SDK has none the name does not go out.
pub const RESERVED_TRANSPORT_PREFIX: &str = "wardex.transport.";

fn is_reserved_transport_key(key: &str) -> bool {
    key.starts_with(RESERVED_TRANSPORT_PREFIX)
}

// No semantic convention defines these as span attributes, so they keep the
// field's own path under `wardex.transport.`, the rule `wardex.transport.protocol`
// already follows. The intervals stay in the schema's milliseconds.
/// `TransportAttributes.connection_id`, which only the SDK fills
/// (`str(id(socket))`). Public because the OTLP masker leaves it alone by this
/// name, and only the reservation above makes the name say who wrote it.
pub const T_CONNECTION_ID: &str = "wardex.transport.connection_id";
const T_DIRECTION: &str = "wardex.transport.direction";
const T_REQUEST_SIZE: &str = "wardex.transport.request_size";
const T_RESPONSE_SIZE: &str = "wardex.transport.response_size";
const T_IS_STREAMING: &str = "wardex.transport.is_streaming";
const T_CONNECTION_REUSED: &str = "wardex.transport.connection_reused";
const T_TCP_CONNECT_MS: &str = "wardex.transport.timing.tcp_connect_ms";
const T_TLS_HANDSHAKE_MS: &str = "wardex.transport.timing.tls_handshake_ms";
const T_TTFB_MS: &str = "wardex.transport.timing.ttfb_ms";
const T_TRANSFER_MS: &str = "wardex.transport.timing.transfer_ms";
const T_TTFT_MS: &str = "wardex.transport.timing.ttft_ms";

/// `Span.transport` → OTLP span attributes.
///
/// Only what was observed goes out. A field with presence emits its key when
/// the sender set it and no key otherwise; a string emits when non-empty; an
/// enum when it names a value. Both sizes have presence: a half whose body
/// went past its capture limit kept only a prefix, and a half the SDK lost
/// (an evicted HTTP/2 request) was never counted, so neither length is the
/// size of what crossed the wire. A zero that goes out is an empty body.
/// Exhaustive on its input, so a field added to the schema does not compile
/// here until it is mapped or bound to `_` with its reason in [`WIRE_FIELDS`].
///
/// Every key is written with [`set_attr`]: a `wardex.transport.*` name
/// reaches here already cleared of host values ([`RESERVED_TRANSPORT_PREFIX`]),
/// and a semantic-convention name a host spelled by hand gives way to the
/// observed value rather than shipping beside it.
fn transport(t: &pb::TransportAttributes, attrs: &mut Vec<otlp_pb::common::KeyValue>) {
    let pb::TransportAttributes {
        connection_id,
        protocol,
        direction,
        timing,
        request_size,
        response_size,
        http,
        grpc: _,
        websocket: _,
        mcp,
        sse: _,
        a2a: _,
        request_blob_ref: _,
        response_blob_ref: _,
        request_modality: _,
        response_modality: _,
        is_streaming,
        connection_reused,
    } = t;
    let protocol = vocab::protocol_name(*protocol);
    if protocol == "sse" {
        // SSE is not a network protocol, it is a framing over HTTP —
        // `network.protocol.name = "sse"` fails every backend's HTTP
        // grouping. The observed fact survives under a wardex key.
        set_attr(attrs, kv_str("network.protocol.name", "http"));
        set_attr(attrs, kv_str("wardex.transport.protocol", &protocol));
    } else {
        set_attr(attrs, kv_str("network.protocol.name", &protocol));
    }
    if !connection_id.is_empty() {
        set_attr(attrs, kv_str(T_CONNECTION_ID, connection_id));
    }
    let direction = vocab::direction_name(*direction);
    if !direction.is_empty() {
        set_attr(attrs, kv_str(T_DIRECTION, &direction));
    }
    if let Some(v) = request_size {
        set_attr(attrs, kv_int(T_REQUEST_SIZE, i64::from(*v)));
    }
    if let Some(v) = response_size {
        set_attr(attrs, kv_int(T_RESPONSE_SIZE, i64::from(*v)));
    }
    if let Some(v) = is_streaming {
        set_attr(attrs, kv_bool(T_IS_STREAMING, *v));
    }
    if let Some(v) = connection_reused {
        set_attr(attrs, kv_bool(T_CONNECTION_REUSED, *v));
    }
    if let Some(pb::TransportTiming {
        tcp_connect_ms,
        tls_handshake_ms,
        ttfb_ms,
        transfer_ms,
        ttft_ms,
    }) = timing
    {
        for (value, key) in [
            (tcp_connect_ms, T_TCP_CONNECT_MS),
            (tls_handshake_ms, T_TLS_HANDSHAKE_MS),
            (ttfb_ms, T_TTFB_MS),
            (transfer_ms, T_TRANSFER_MS),
            (ttft_ms, T_TTFT_MS),
        ] {
            if let Some(ms) = value {
                set_attr(attrs, kv_f64(key, widen(*ms)));
            }
        }
    }
    if let Some(pb::HttpMeta {
        method,
        status_code,
        url,
    }) = http
    {
        set_attr(attrs, kv_str("http.request.method", method));
        set_attr(
            attrs,
            kv_int("http.response.status_code", i64::from(*status_code)),
        );
        // The WHOLE URL, query included: `url.full` is semconv's one home
        // for a client span's URL, and the query is the call's arguments
        // (`?q=seoul&page=2`) — dropping it silently threw away what an
        // agent asked for while the same arguments sent in a POST body
        // shipped whole. Credentials in it are the masker's to replace,
        // by the same name rules that cover a body, before this request
        // leaves the process. An empty captured URL emits nothing.
        if !url.is_empty() {
            set_attr(attrs, kv_str("url.full", url));
        }
    }
    if let Some(pb::McpMeta { rpc_method, rpc_id }) = mcp {
        if !rpc_method.is_empty() {
            set_attr(attrs, kv_str(MCP_METHOD_NAME, rpc_method));
        }
        // semconv: a request without an id is a notification, and the
        // attribute is left out rather than written empty.
        if !rpc_id.is_empty() {
            set_attr(attrs, kv_str(JSONRPC_REQUEST_ID, rpc_id));
        }
    }
}

/// `Span.events` → OTLP `Span.events`.
///
/// OTLP is the surface that actually leaves the process, so filling
/// `Span.events`/`Span.links` on the wardex envelope and not here would leave
/// the two encoders disagreeing about the same span — and a user configured for
/// the OTLP exporter would lose design §6.3's whole graph model with no
/// counter, no `Limitation` marker and no failing test, indistinguishable from
/// "this agent has no graph edges". That is the silent-loss shape I4 forbids.
fn event(ev: pb::SpanEvent) -> otlp_pb::trace::span::Event {
    otlp_pb::trace::span::Event {
        time_unix_nano: ev.time_unix_nano,
        name: ev.name,
        attributes: ev.attributes.into_iter().map(key_value).collect(),
        ..Default::default()
    }
}

/// `Span.links` → OTLP `Span.links`.
///
/// `reason` has no OTLP-native home — `Link` carries `trace_state` and
/// attributes and nothing else — so it travels as the `wardex.link.reason`
/// attribute rather than being dropped, as the wardex value string so that it
/// is readable without a copy of the enum.
fn link(ln: pb::SpanLink) -> otlp_pb::trace::span::Link {
    let reason = vocab::link_reason_name(ln.reason);
    let mut attributes: Vec<otlp_pb::common::KeyValue> =
        ln.attributes.into_iter().map(key_value).collect();
    if !reason.is_empty() {
        attributes.push(kv_str("wardex.link.reason", &reason));
    }
    otlp_pb::trace::span::Link {
        trace_id: ln.trace_id,
        span_id: ln.span_id,
        attributes,
        ..Default::default()
    }
}

/// The non-empty string an `extra` key carries, if it carries one.
///
/// Empty reads as absent on purpose: proto3 cannot tell `""` from unset, and a
/// span named `"chat "` is worse than one named for the operation alone.
fn string_attr<'a>(extra: &'a [pb::KeyValue], key: &str) -> Option<&'a str> {
    extra.iter().find(|kv| kv.key == key).and_then(|kv| {
        match kv.value.as_ref()?.value.as_ref()? {
            pb::any_value::Value::StringValue(s) if !s.is_empty() => Some(s.as_str()),
            _ => None,
        }
    })
}

/// Does this operation name a call TO A MODEL — the one shape whose semconv
/// name is composed from a model id?
///
/// `gen_ai.operation.name` is not "this span is an LLM call": every span in the
/// closed vocabulary carries it, so `execute_tool`, `invoke_agent` and the rest
/// arrive here too. Their semconv subject is a tool or agent name, never a
/// model, and the sender already composed it into the name.
///
/// Exhaustive on the generated enum on purpose: a thirteenth operation has to
/// be classified here rather than inheriting whichever branch happened to be
/// the fallthrough. An unrecognized string is not a model call — a build that
/// does not know the operation cannot know how semconv names it.
fn is_model_operation(operation: &str) -> bool {
    use pb::OperationName as Op;
    let code = vocab::operation_name_to_proto(operation);
    match code.and_then(|n| Op::try_from(n).ok()) {
        Some(Op::Chat | Op::TextCompletion | Op::Embeddings | Op::GenerateContent) => true,
        Some(
            Op::ExecuteTool
            | Op::CreateAgent
            | Op::InvokeAgent
            | Op::InvokeWorkflow
            | Op::Retrieval
            | Op::ExecuteStep
            | Op::Handoff
            | Op::Evaluate
            | Op::Unspecified,
        )
        | None => false,
    }
}

/// The OTLP span name — `{gen_ai.operation.name} {model}` for a call to a
/// model, otherwise the name the sender gave it.
///
/// The name it replaces was the transport's: `HTTP POST /v1/chat/completions`,
/// which is the SAME STRING for every model, every prompt and every provider
/// behind one endpoint. Span name is the axis every backend groups by, so a
/// latency or cost breakdown over an agent's LLM calls collapsed into a single
/// bucket that answered no question anyone asks — while the two facts a reader
/// actually groups by sat one level down in the attributes.
///
/// Two things this deliberately does NOT do, because each would trade one
/// collapse for a wider one:
///
///   * It does not rename a span whose operation is not a model call. Those
///     names already carry a subject the sender composed — `execute_tool Bash`,
///     `invoke_agent researcher` — and rewriting them to the bare operation
///     would put every tool call in a run into one bucket, which is the very
///     failure this rename exists to remove.
///   * It does not fall back to the bare operation when no model was recorded.
///     `chat` is strictly less than the name that was already there, and
///     `chat unknown` is worse still: it invents a model of that name and a
///     backend aggregates it as one. An unnamed model leaves the name alone.
///
/// `gen_ai.response.model` is consulted when the request never recorded one.
/// That is not inventing a model — the SDK observed it, one attribute away, and
/// a span named for the model that answered beats one named for no model at
/// all.
fn span_name(sp: &pb::Span) -> String {
    let Some(operation) = string_attr(&sp.extra, "gen_ai.operation.name") else {
        return sp.name.clone();
    };
    if !is_model_operation(operation) {
        return sp.name.clone();
    }
    match string_attr(&sp.extra, "gen_ai.request.model")
        .or_else(|| string_attr(&sp.extra, "gen_ai.response.model"))
    {
        Some(model) => format!("{operation} {model}"),
        None => sp.name.clone(),
    }
}

/// One span, mapped. `overwritten` gains the host attributes under a reserved
/// transport name that this span does not ship ([`RESERVED_TRANSPORT_PREFIX`]).
fn span(mut sp: pb::Span, overwritten: &mut usize) -> otlp_pb::trace::Span {
    use std::mem::take;

    // Every field of the input, named once, read nowhere: a field added to
    // `Span` does not compile here until it has a line, and a line is only
    // honest once `WIRE_FIELDS` says where the value goes. The mapping below
    // reads them off `sp` as it always has.
    let pb::Span {
        trace_id: _,
        span_id: _,
        parent_span_id: _,
        name: _,
        kind: _,
        start_time_unix_nano: _,
        end_time_unix_nano: _,
        status: _,
        extra: _,
        events: _,
        links: _,
        dropped_extra_count: _,
        dropped_events_count: _,
        dropped_links_count: _,
        input_data: _,
        output_data: _,
        transport: _,
        error_type: _,
        server_address: _,
        server_port: _,
        workflow_name: _,
        call_site: _,
        conversation: _,
        capture_sources: _,
        capture_integrity: _,
        correlation: _,
    } = &sp;

    // Read before anything is moved out: naming consults `extra`, and
    // `uncertainty` wants the whole span.
    let name = span_name(&sp);
    let code = status_code(sp.status.as_ref().map(|s| s.code).unwrap_or_default());
    // A tool call's payload has a semconv home of its own; every other
    // operation keeps the wardex.* keys. Decided off the operation the sender
    // recorded — same pipeline, same masking, same caps, only the key differs
    // — and the operation name is read off the schema, never restated here.
    let execute_tool = vocab::operation_name_name(pb::OperationName::ExecuteTool as i32);
    let (input_key, output_key) =
        if string_attr(&sp.extra, "gen_ai.operation.name") == Some(execute_tool.as_str()) {
            ("gen_ai.tool.call.arguments", "gen_ai.tool.call.result")
        } else {
            ("wardex.input_data", "wardex.output_data")
        };

    // gen_ai / agent / tool attributes were flattened into `extra` when the
    // envelope was marshalled, so both export surfaces carry one flattening and
    // cannot drift.
    let mut attrs: Vec<otlp_pb::common::KeyValue> =
        take(&mut sp.extra).into_iter().map(key_value).collect();
    // The reserved transport names are the SDK's alone: a host's value under
    // one goes before `transport` writes the SDK's, whether or not the SDK has
    // one to write, and is counted.
    let host_attrs = attrs.len();
    attrs.retain(|kv| !is_reserved_transport_key(&kv.key));
    *overwritten += host_attrs - attrs.len();

    // server.* — the empty string and port 0 are proto3 "unset", and neither is
    // a value a host could have meant.
    if !sp.server_address.is_empty() {
        attrs.push(kv_str("server.address", &sp.server_address));
    }
    if sp.server_port != 0 {
        attrs.push(kv_int("server.port", sp.server_port as i64));
    }
    if let Some(t) = &sp.transport {
        transport(t, &mut attrs);
    }
    // The typed field is the sender's one home for the name; a host that also
    // spelled the key into `extra` does not get a second copy (`set_attr`).
    if !sp.workflow_name.is_empty() {
        set_attr(&mut attrs, kv_str(WORKFLOW_NAME, &sp.workflow_name));
    }
    // Raw I/O → payload attributes (omitted if empty). The key pair was chosen
    // above from the operation. Built as bytes so PII masking sees the raw
    // payload; `strip_bytes_values` converts to strings after masking, before
    // serialization.
    let input = take(&mut sp.input_data);
    if !input.is_empty() {
        attrs.push(kv_bytes(input_key, input));
    }
    let output = take(&mut sp.output_data);
    if !output.is_empty() {
        attrs.push(kv_bytes(output_key, output));
    }
    typed_blocks(&sp, &mut attrs);
    uncertainty(&sp, &mut attrs);

    // `error.type` is a span attribute — semconv's home for it — and the
    // status message stays the status message. It used to be SUBSTITUTED into
    // `Status.message`, which destroyed the one field a backend renders as
    // "what went wrong" to relabel it with a fact that now travels beside it.
    if !sp.error_type.is_empty() {
        attrs.push(kv_str("error.type", &sp.error_type));
    }
    let message = sp
        .status
        .as_mut()
        .map(|s| take(&mut s.message))
        .unwrap_or_default();

    otlp_pb::trace::Span {
        trace_id: take(&mut sp.trace_id),
        span_id: take(&mut sp.span_id),
        parent_span_id: take(&mut sp.parent_span_id),
        name,
        kind: span_kind(sp.kind),
        start_time_unix_nano: sp.start_time_unix_nano,
        end_time_unix_nano: sp.end_time_unix_nano,
        status: Some(otlp_pb::trace::Status { code, message }),
        attributes: attrs,
        events: take(&mut sp.events).into_iter().map(event).collect(),
        links: take(&mut sp.links).into_iter().map(link).collect(),
        // Spelled out rather than defaulted, so each zero below is a decision
        // a reader can see. The wire carries no W3C trace state or trace
        // flags for a span, and the `Span.dropped_*` counts have no producer
        // (see `WIRE_FIELDS`).
        trace_state: String::new(),
        flags: 0,
        dropped_attributes_count: 0,
        dropped_events_count: 0,
        dropped_links_count: 0,
    }
}

/// Envelope → `ExportTraceServiceRequest`.
///
/// Traces only: state snapshots have no OTLP trace form and are skipped rather
/// than flattened into one. An envelope with no spans produces no
/// `resource_spans`, so a caller can tell "nothing to send" from "a batch of
/// empty spans" without decoding.
///
/// BY VALUE, so the envelope's payloads are moved into the request rather than
/// copied beside it. The caller builds this envelope for this mapping and
/// discards it, and the export path runs on the background batch worker inside
/// the host's process: holding the envelope and the request alive together
/// would make one flush of a full batch cost an extra copy of every captured
/// request and response body, against a `max_buffer_bytes` backstop that
/// accounts for one.
pub fn envelope_to_traces(
    env: pb::Envelope,
    producer: Producer<'_>,
) -> otlp_pb::trace_service::ExportTraceServiceRequest {
    map_envelope(env, producer).request
}

/// An OTLP request, and what its mapping did that the request cannot say for
/// itself.
#[derive(Debug)]
pub struct Mapped {
    pub request: otlp_pb::trace_service::ExportTraceServiceRequest,
    /// Host attributes under a reserved transport name that did not ship as
    /// the host set them: each was replaced by the SDK's own value or, where
    /// the SDK had none, left out ([`RESERVED_TRANSPORT_PREFIX`]). The mapping
    /// has no channel to a user, so the binding that called it says so.
    pub reserved_overwritten: usize,
}

/// [`envelope_to_traces`], with the count the request does not carry. The
/// export path calls this one, so the count is of exactly what was mapped.
pub fn map_envelope(mut env: pb::Envelope, producer: Producer<'_>) -> Mapped {
    let mut reserved_overwritten = 0;
    let spans: Vec<otlp_pb::trace::Span> = std::mem::take(&mut env.items)
        .into_iter()
        .filter_map(|item| match item.payload {
            Some(pb::envelope_item::Payload::Span(sp)) => Some(span(sp, &mut reserved_overwritten)),
            _ => None,
        })
        .collect();
    let request = traces_request(&env, spans, producer);
    Mapped {
        request,
        reserved_overwritten,
    }
}

fn traces_request(
    env: &pb::Envelope,
    spans: Vec<otlp_pb::trace::Span>,
    producer: Producer<'_>,
) -> otlp_pb::trace_service::ExportTraceServiceRequest {
    if spans.is_empty() {
        return otlp_pb::trace_service::ExportTraceServiceRequest {
            resource_spans: vec![],
        };
    }
    let sdk = env.header.as_ref().and_then(|h| h.sdk.as_ref());
    let version = sdk.map(|s| s.version.as_str()).unwrap_or_default();
    // Resource identity is the APPLICATION's, read off `EnvelopeHeader.
    // resource`, never off SdkInfo: every app used to export as
    // `service.name = "wardex.python"`, which made two services one service
    // in any backend that groups by the resource — the axis they all group by.
    // An unnamed service gets semconv's own fallback shape, never the SDK's
    // name; `telemetry.sdk.name` is the constant `"wardex"` across languages
    // (the language already travels in `telemetry.sdk.language`).
    let resource = env.header.as_ref().and_then(|h| h.resource.as_ref());
    let service_name = resource
        .map(|r| r.service_name.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .unwrap_or_else(|| format!("unknown_service:{}", producer.language));
    let mut resource_attrs = vec![
        kv_str("service.name", &service_name),
        kv_str("telemetry.sdk.name", "wardex"),
        kv_str("telemetry.sdk.version", version),
        kv_str("telemetry.sdk.language", producer.language),
    ];
    if let Some(r) = resource {
        // Emitted iff configured: an empty `service.version = ""` is not a
        // smaller answer than no key, it is a version named "" for a backend
        // to group by.
        if !r.release.is_empty() {
            resource_attrs.push(kv_str("service.version", &r.release));
        }
        if !r.environment.is_empty() {
            resource_attrs.push(kv_str("deployment.environment.name", &r.environment));
        }
        // Same rule for the process identity: 0 is proto3's "not stamped",
        // and `process.pid = 0` would be a claim (the scheduler) rather than
        // an absence. The SDK stamps this live at drain time, so a fork
        // parent and its children arrive distinguishable.
        if r.process_pid > 0 {
            resource_attrs.push(kv_int("process.pid", r.process_pid as i64));
        }
    }
    otlp_pb::trace_service::ExportTraceServiceRequest {
        resource_spans: vec![otlp_pb::trace::ResourceSpans {
            resource: Some(otlp_pb::resource::Resource {
                attributes: resource_attrs,
                ..Default::default()
            }),
            scope_spans: vec![otlp_pb::trace::ScopeSpans {
                scope: Some(otlp_pb::common::InstrumentationScope {
                    name: producer.scope_name.into(),
                    version: version.into(),
                    ..Default::default()
                }),
                spans,
                ..Default::default()
            }],
            ..Default::default()
        }],
    }
}

// --- No bytes_value ever leaves on the OTLP surface ---
//
// OTLP `bytes_value` is legal per spec, but backends that re-serialize
// attributes to JSON can't represent it: Arize Phoenix (2026-08) drops the
// entire span at ingest — HTTP 200, no error — and the spans lost are exactly
// the LLM/tool ones that carry payloads. The wardex envelope keeps raw bytes;
// this surface degrades to strings: valid UTF-8 verbatim, anything else base64
// plus a `<key>.encoding = "base64"` companion so a consumer can tell encoded
// binary from text that merely looks like base64.

/// Strip every `bytes_value` from an OTLP export request, in place.
///
/// Run this AFTER masking, never before: the PII engine's byte-level patterns
/// match inside raw payloads, and a payload already rewritten to base64 would
/// hide them.
pub fn strip_bytes_values(req: &mut otlp_pb::trace_service::ExportTraceServiceRequest) {
    for rs in &mut req.resource_spans {
        if let Some(r) = rs.resource.as_mut() {
            strip_kvs(&mut r.attributes);
        }
        for ss in &mut rs.scope_spans {
            if let Some(sc) = ss.scope.as_mut() {
                strip_kvs(&mut sc.attributes);
            }
            for sp in &mut ss.spans {
                strip_kvs(&mut sp.attributes);
                for ev in &mut sp.events {
                    strip_kvs(&mut ev.attributes);
                }
                for link in &mut sp.links {
                    strip_kvs(&mut link.attributes);
                }
            }
        }
    }
}

fn strip_kvs(kvs: &mut Vec<otlp_pb::common::KeyValue>) {
    // (companion key, went_base64) for every attribute whose value WAS bytes.
    let mut rewritten: Vec<(String, bool)> = Vec::new();
    for kv in kvs.iter_mut() {
        if let Some(v) = kv.value.as_mut() {
            if let Some(went_base64) = strip_any(v) {
                rewritten.push((format!("{}.encoding", kv.key), went_base64));
            }
        }
    }
    if rewritten.is_empty() {
        return;
    }
    // For a key this pass rewrote, `<key>.encoding` is this pass's namespace.
    // A pre-existing attribute there (a user extra — any key passes through)
    // would either duplicate the companion (duplicate keys are undefined in
    // OTLP, backend dedup order decides which wins) or spoof an encoding the
    // verbatim branch never applied, making consumers base64-decode text that
    // shipped as-is. Drop it either way; user `.encoding` suffixes on keys that
    // never carried bytes are untouched.
    kvs.retain(|kv| !rewritten.iter().any(|(companion, _)| kv.key == *companion));
    for (companion, went_base64) in rewritten {
        if went_base64 {
            kvs.push(kv_str(&companion, "base64"));
        }
    }
}

/// Text safe for every real backend, or None → base64. Strict UTF-8 alone is
/// not enough: U+0000 is valid UTF-8, and Postgres-backed ingests reject any
/// string containing NUL — the same silent span loss this pass exists to
/// prevent, reintroduced for the NUL subset. Binary protobuf/gRPC payloads
/// are full of NULs and are exactly what must route to base64, so any C0
/// control byte other than \t \n \r means "not text".
fn as_text(b: &[u8]) -> Option<&str> {
    let s = std::str::from_utf8(b).ok()?;
    if b.iter()
        .any(|&c| c < 0x20 && c != b'\t' && c != b'\n' && c != b'\r')
    {
        return None;
    }
    Some(s)
}

/// `Some(went_base64)` when the value itself was a bytes_value — the caller
/// owns the attribute list and manages the `<key>.encoding` companion; a
/// bytes value nested in an ArrayValue has no key of its own, so there the
/// result has no receiver and non-text elements go base64 unmarked.
fn strip_any(v: &mut otlp_pb::common::AnyValue) -> Option<bool> {
    use otlp_pb::common::any_value::Value;
    match v.value.as_mut() {
        Some(Value::BytesValue(b)) => {
            let raw = std::mem::take(b);
            let (s, went_base64) = match as_text(&raw) {
                Some(s) => (s.to_owned(), false),
                None => (BASE64.encode(&raw), true),
            };
            v.value = Some(Value::StringValue(s));
            Some(went_base64)
        }
        Some(Value::ArrayValue(arr)) => {
            for item in &mut arr.values {
                strip_any(item);
            }
            None
        }
        Some(Value::KvlistValue(kvl)) => {
            strip_kvs(&mut kvl.values);
            None
        }
        _ => None,
    }
}

// --- Size caps at the OTLP surface ---
//
// Every cap the SDK enforced before this point measures a RAW size, taken
// before anything is encoded: `max_body_bytes` on what a parser keeps,
// `max_buffer_bytes` on what the client holds. None of them is the size a
// receiver measures. The rewrite above turns a binary payload into base64, so a
// span inside every raw cap can leave 4/3 larger than the largest number anyone
// configured — and an OTLP receiver rejects a request whole, so crossing its
// ceiling costs every span in the batch rather than the bytes that crossed it.
//
// So the last thing this surface does before serialization is measure itself in
// the units the receiver uses, and say so on the span when it has to cut.

/// Truncate every attribute value over `limits.max_otlp_attribute_bytes`,
/// marking each span whose content was cut.
///
/// Runs AFTER masking and BEFORE [`strip_bytes_values`], and neither half of
/// that is arbitrary. Before masking, a truncation could cut a PII match in two
/// and ship the surviving half — the engine would never see the pattern whole.
/// After the base64 rewrite, truncating would land mid-quantum and leave a
/// string that no consumer can decode; cutting the raw bytes first means the
/// rewrite encodes a shorter payload rather than a broken encoding of a longer
/// one.
///
/// Resource and scope attributes are deliberately not capped: this SDK writes
/// them itself, they are five short strings naming the service and the library,
/// and there is no span to carry a marker if one were cut.
pub fn cap_attribute_values(
    req: &mut otlp_pb::trace_service::ExportTraceServiceRequest,
    limits: Limits,
) {
    let cap = limits.max_otlp_attribute_bytes;
    for rs in &mut req.resource_spans {
        for ss in &mut rs.scope_spans {
            for sp in &mut ss.spans {
                let mut cut = cap_kvs(&mut sp.attributes, cap);
                for ev in &mut sp.events {
                    cut |= cap_kvs(&mut ev.attributes, cap);
                }
                for link in &mut sp.links {
                    cut |= cap_kvs(&mut link.attributes, cap);
                }
                // After the walk, so the marker itself is never a candidate for
                // truncation and an event's loss still reaches the span that
                // owns it — an event has no `wardex.limitations` of its own.
                if cut {
                    mark_truncated(&mut sp.attributes);
                }
            }
        }
    }
}

/// Remove a span's captured payload and mark it — the last thing tried for a
/// span whose encoded size ALONE exceeds the per-request cap.
///
/// The alternative at that point is dropping the span, and a span is worth more
/// than its payload: its identity, its parent edge, its timing and its
/// gen_ai semantics are what a trace is made of, and a hole in the tree is
/// read as "this call never happened" rather than as "this call was large".
pub fn drop_payload_attributes(sp: &mut otlp_pb::trace::Span) {
    // Both key pairs a payload can land under: the wardex.* defaults and the
    // semconv pair an `execute_tool` span uses. Whichever pair `span()` chose,
    // this pass has to find it — a payload it cannot name is one it cannot
    // drop, and the span dies whole instead.
    const PAYLOAD: [&str; 4] = [
        "wardex.input_data",
        "wardex.output_data",
        "gen_ai.tool.call.arguments",
        "gen_ai.tool.call.result",
    ];
    let before = sp.attributes.len();
    sp.attributes.retain(|kv| {
        // `<key>.encoding` is the companion `strip_bytes_values` writes; it
        // describes a value that is no longer here, and left behind it would
        // claim a base64 payload that a consumer cannot find.
        let base = kv.key.strip_suffix(".encoding").unwrap_or(&kv.key);
        !PAYLOAD.contains(&base)
    });
    if sp.attributes.len() != before {
        mark_truncated(&mut sp.attributes);
    }
}

fn mark_truncated(attrs: &mut Vec<otlp_pb::common::KeyValue>) {
    // By NUMBER off the schema, never as a literal: the vocabulary is declared
    // in `common.proto` and a string written here would be a second
    // declaration of it, free to drift (design §6.6).
    let name = vocab::limitation_name(pb::Limitation::OtlpAttributeTruncated as i32);
    push_limitation(attrs, name);
}

/// Append a marker to a span's `wardex.limitations`, creating the attribute
/// when the span had nothing to report. Idempotent — one truncated attribute
/// and twenty say the same thing about the span.
fn push_limitation(attrs: &mut Vec<otlp_pb::common::KeyValue>, name: String) {
    use otlp_pb::common::any_value::Value;

    let entry = otlp_pb::common::AnyValue {
        value: Some(Value::StringValue(name.clone())),
    };
    if let Some(kv) = attrs.iter_mut().find(|kv| kv.key == LIMITATIONS_KEY) {
        if let Some(Value::ArrayValue(arr)) = kv.value.as_mut().and_then(|v| v.value.as_mut()) {
            if !arr.values.contains(&entry) {
                arr.values.push(entry);
            }
        } else {
            // The key exists holding something that is not an array — nothing
            // this SDK produces, so a host `extra` that squatted on it.
            // Overwrite rather than add a second `wardex.limitations`:
            // duplicate keys are undefined in OTLP and which one a backend
            // keeps is its own business, which would make the marker a
            // coin flip.
            kv.value = Some(otlp_pb::common::AnyValue {
                value: Some(Value::ArrayValue(otlp_pb::common::ArrayValue {
                    values: vec![entry],
                })),
            });
        }
        return;
    }
    attrs.push(kv_strs(LIMITATIONS_KEY, vec![name]));
}

fn cap_kvs(kvs: &mut [otlp_pb::common::KeyValue], cap: usize) -> bool {
    let mut cut = false;
    for kv in kvs.iter_mut() {
        if let Some(v) = kv.value.as_mut() {
            cut |= cap_any(v, cap);
        }
    }
    cut
}

/// True when this value was cut. The bound is per VALUE, so an array is capped
/// element by element rather than in total: the elements are separate facts,
/// and a bound on their sum would silently delete whole entries from a list a
/// consumer reads positionally.
fn cap_any(v: &mut otlp_pb::common::AnyValue, cap: usize) -> bool {
    use otlp_pb::common::any_value::Value;
    match v.value.as_mut() {
        Some(Value::StringValue(s)) => {
            if s.len() <= cap {
                return false;
            }
            s.truncate(floor_char_boundary(s.as_bytes(), cap));
            true
        }
        Some(Value::BytesValue(b)) => {
            // At or under three quarters of the bound even the base64 form
            // fits, so the answer is known without asking which form it takes.
            // Worth an early return rather than tidiness: `as_text` scans the
            // whole payload, `strip_bytes_values` will scan it again a moment
            // later, and skipping this one keeps the cap free for every payload
            // that was never near the bound — which is all of them, normally.
            if b.len() <= cap / 4 * 3 {
                return false;
            }
            // Measured as it will LEAVE, not as it sits here: `as_text` is the
            // same test `strip_bytes_values` will apply, so this is that pass's
            // own arithmetic asked one step early.
            let text = as_text(b).is_some();
            let projected = if text { b.len() } else { base64_len(b.len()) };
            if projected <= cap {
                return false;
            }
            let keep = if text {
                floor_char_boundary(b, cap)
            } else {
                // Whole base64 quanta only. Cutting the raw bytes to a multiple
                // of three keeps the encoder from emitting a padded tail that
                // pushes the string back over the bound.
                cap / 4 * 3
            };
            b.truncate(keep);
            true
        }
        Some(Value::ArrayValue(arr)) => {
            let mut cut = false;
            for item in &mut arr.values {
                cut |= cap_any(item, cap);
            }
            cut
        }
        Some(Value::KvlistValue(kvl)) => cap_kvs(&mut kvl.values, cap),
        _ => false,
    }
}

/// Length of `n` bytes once base64-encoded with padding — what
/// `strip_bytes_values` will put on the wire for a payload that is not text.
fn base64_len(n: usize) -> usize {
    4 * n.div_ceil(3)
}

/// The largest index at or below `i` that starts a UTF-8 character.
///
/// `str::floor_char_boundary` is still unstable, and the naive `truncate(cap)`
/// it replaces panics on a multi-byte boundary — inside the export path, on the
/// background worker, for a payload whose only crime was being long.
fn floor_char_boundary(bytes: &[u8], i: usize) -> usize {
    if i >= bytes.len() {
        return bytes.len();
    }
    let mut i = i;
    while i > 0 && (bytes[i] & 0xC0) == 0x80 {
        i -= 1;
    }
    i
}

#[cfg(test)]
mod tests {
    use super::*;

    const PRODUCER: Producer<'static> = Producer {
        language: "python",
        scope_name: "wardex.python",
    };

    fn envelope(sp: pb::Span) -> pb::Envelope {
        pb::Envelope {
            header: Some(pb::EnvelopeHeader {
                sdk: Some(pb::SdkInfo {
                    name: "wardex.python".into(),
                    version: "0.1.0".into(),
                    ..Default::default()
                }),
                ..Default::default()
            }),
            items: vec![pb::EnvelopeItem {
                header: None,
                payload: Some(pb::envelope_item::Payload::Span(sp)),
            }],
        }
    }

    fn only_span(env: pb::Envelope) -> otlp_pb::trace::Span {
        let req = envelope_to_traces(env, PRODUCER);
        req.resource_spans[0].scope_spans[0].spans[0].clone()
    }

    fn attr<'a>(
        sp: &'a otlp_pb::trace::Span,
        key: &str,
    ) -> Option<&'a otlp_pb::common::any_value::Value> {
        sp.attributes
            .iter()
            .find(|kv| kv.key == key)
            .and_then(|kv| kv.value.as_ref())
            .and_then(|v| v.value.as_ref())
    }

    fn gen_ai_span(attrs: &[(&str, &str)]) -> pb::Span {
        pb::Span {
            name: "HTTP POST /v1/chat/completions".into(),
            extra: attrs.iter().map(|(k, v)| kv_wardex(k, v)).collect(),
            ..Default::default()
        }
    }

    fn kv_wardex(key: &str, v: &str) -> pb::KeyValue {
        pb::KeyValue {
            key: key.into(),
            value: Some(pb::AnyValue {
                value: Some(pb::any_value::Value::StringValue(v.into())),
            }),
        }
    }

    #[test]
    fn an_llm_span_is_named_for_its_operation_and_model() {
        let sp = gen_ai_span(&[
            ("gen_ai.operation.name", "chat"),
            ("gen_ai.request.model", "gpt-4.1-mini"),
        ]);
        assert_eq!(span_name(&sp), "chat gpt-4.1-mini");
    }

    #[test]
    fn the_model_that_answered_names_the_span_when_the_request_recorded_none() {
        // Not inventing a model: the SDK observed this one, one attribute away.
        // An assembled turn is exactly this shape — the response model is known
        // before the stream has reported what was requested.
        let sp = gen_ai_span(&[
            ("gen_ai.operation.name", "chat"),
            ("gen_ai.response.model", "claude-sonnet-5"),
        ]);
        assert_eq!(span_name(&sp), "chat claude-sonnet-5");
    }

    #[test]
    fn an_llm_span_with_no_model_at_all_keeps_the_name_it_was_given() {
        // Not "embeddings" and not "embeddings unknown": the first is strictly
        // less than the name already there, and the second invents a model of
        // that name for a backend to aggregate.
        let sp = gen_ai_span(&[("gen_ai.operation.name", "embeddings")]);
        assert_eq!(span_name(&sp), "HTTP POST /v1/chat/completions");
        let blank = gen_ai_span(&[
            ("gen_ai.operation.name", "embeddings"),
            ("gen_ai.request.model", ""),
        ]);
        assert_eq!(span_name(&blank), "HTTP POST /v1/chat/completions");
    }

    #[test]
    fn a_span_without_llm_semantics_keeps_the_name_it_was_given() {
        let sp = gen_ai_span(&[("http.request.method", "POST")]);
        assert_eq!(span_name(&sp), "HTTP POST /v1/chat/completions");
    }

    #[test]
    fn a_non_model_operation_keeps_the_subject_the_sender_composed() {
        // EVERY span in the closed vocabulary carries `gen_ai.operation.name`,
        // not only the LLM ones, and none of these can carry a request model —
        // `execute_tool` requires a tool block, `invoke_agent` an agent block.
        // Reading the key as "this is an LLM call" and dropping to the bare
        // operation would put every tool call in a run into one bucket, which
        // is the collapse this rename exists to remove.
        for (operation, subject) in [
            ("execute_tool", "execute_tool Bash"),
            ("invoke_agent", "invoke_agent researcher"),
            ("execute_step", "execute_step summarize"),
            ("invoke_workflow", "nightly reconciliation"),
            ("retrieval", "retrieval kb-docs"),
            ("handoff", "handoff reviewer"),
            ("evaluate", "evaluate toxicity"),
            ("create_agent", "create_agent planner"),
        ] {
            let sp = pb::Span {
                name: subject.into(),
                extra: vec![kv_wardex("gen_ai.operation.name", operation)],
                ..Default::default()
            };
            assert_eq!(span_name(&sp), subject);
        }
    }

    #[test]
    fn a_tool_span_is_not_renamed_by_a_model_that_wandered_onto_it() {
        // The gate is the OPERATION, not the presence of a model attribute: a
        // tool span that picked up a model from a parsed payload is still a
        // tool span, and `execute_tool gpt-4.1-mini` names the wrong thing.
        let sp = pb::Span {
            name: "execute_tool Bash".into(),
            extra: vec![
                kv_wardex("gen_ai.operation.name", "execute_tool"),
                kv_wardex("gen_ai.request.model", "gpt-4.1-mini"),
            ],
            ..Default::default()
        };
        assert_eq!(span_name(&sp), "execute_tool Bash");
    }

    #[test]
    fn an_operation_this_build_does_not_know_is_not_treated_as_a_model_call() {
        let sp = gen_ai_span(&[("gen_ai.operation.name", "banana")]);
        assert_eq!(span_name(&sp), "HTTP POST /v1/chat/completions");
    }

    #[test]
    fn the_name_is_the_only_thing_the_rename_touches() {
        // Attributes stay where a consumer expects them: the model reached the
        // name by being READ, not by being moved.
        let env = envelope(gen_ai_span(&[
            ("gen_ai.operation.name", "chat"),
            ("gen_ai.request.model", "gpt-4.1-mini"),
        ]));
        let sp = only_span(env);
        assert_eq!(sp.name, "chat gpt-4.1-mini");
        assert_eq!(
            attr(&sp, "gen_ai.request.model"),
            Some(&otlp_pb::common::any_value::Value::StringValue(
                "gpt-4.1-mini".into()
            ))
        );
    }

    #[test]
    fn span_kind_and_status_are_renumbered_not_copied() {
        // wardex CLIENT is 2 and OTLP CLIENT is 3: copying the number through
        // would silently relabel every LLM call as a SERVER span.
        assert_eq!(span_kind(pb::SpanKind::Client as i32), 3);
        assert_eq!(span_kind(pb::SpanKind::Internal as i32), 1);
        assert_eq!(span_kind(pb::SpanKind::Server as i32), 2);
        // PRODUCER/CONSUMER happen to share numbers with OTLP (both 4/5) —
        // asserted anyway, because "happens to line up" is exactly the state
        // a renumbering pass exists to stop anyone relying on.
        assert_eq!(span_kind(pb::SpanKind::Producer as i32), 4);
        assert_eq!(span_kind(pb::SpanKind::Consumer as i32), 5);
        assert_eq!(span_kind(404), 0);
        assert_eq!(status_code(pb::StatusCode::Error as i32), 2);
        assert_eq!(status_code(404), 0);
    }

    #[test]
    fn state_snapshots_are_not_traces() {
        let env = pb::Envelope {
            header: None,
            items: vec![pb::EnvelopeItem {
                header: None,
                payload: Some(pb::envelope_item::Payload::StateSnapshot(
                    pb::StateSnapshot::default(),
                )),
            }],
        };
        assert!(envelope_to_traces(env, PRODUCER).resource_spans.is_empty());
    }

    fn resource_attr<'a>(
        req: &'a otlp_pb::trace_service::ExportTraceServiceRequest,
        key: &str,
    ) -> Option<&'a otlp_pb::common::any_value::Value> {
        req.resource_spans[0]
            .resource
            .as_ref()
            .unwrap()
            .attributes
            .iter()
            .find(|kv| kv.key == key)
            .and_then(|kv| kv.value.as_ref())
            .and_then(|v| v.value.as_ref())
    }

    fn str_value(s: &str) -> otlp_pb::common::any_value::Value {
        otlp_pb::common::any_value::Value::StringValue(s.into())
    }

    #[test]
    fn the_service_is_named_by_the_resource_and_the_scope_by_the_library() {
        // The app's identity comes off `EnvelopeHeader.resource`; SdkInfo may
        // not supply it, and the instrumentation library's name moves for
        // neither of them.
        let mut env = envelope(pb::Span::default());
        env.header.as_mut().unwrap().resource = Some(pb::ResourceInfo {
            service_name: "checkout-api".into(),
            release: "1.2.3".into(),
            environment: "staging".into(),
            ..Default::default()
        });
        let req = envelope_to_traces(env, PRODUCER);
        assert_eq!(
            resource_attr(&req, "service.name"),
            Some(&str_value("checkout-api"))
        );
        assert_eq!(
            resource_attr(&req, "service.version"),
            Some(&str_value("1.2.3"))
        );
        assert_eq!(
            resource_attr(&req, "deployment.environment.name"),
            Some(&str_value("staging"))
        );
        let scope = req.resource_spans[0].scope_spans[0].scope.as_ref().unwrap();
        assert_eq!(scope.name, "wardex.python");
    }

    #[test]
    fn an_unnamed_service_exports_as_unknown_service_never_as_the_sdk() {
        // The regression this slice exists to fix: every app exported as
        // `service.name = "wardex.python"`, so two services were one service
        // in any backend. The fallback is semconv's own shape, and SdkInfo's
        // name may not leak into it.
        let req = envelope_to_traces(envelope(pb::Span::default()), PRODUCER);
        assert_eq!(
            resource_attr(&req, "service.name"),
            Some(&str_value("unknown_service:python"))
        );
        // Unconfigured release/environment emit NO key, not an empty one.
        assert!(resource_attr(&req, "service.version").is_none());
        assert!(resource_attr(&req, "deployment.environment.name").is_none());
    }

    #[test]
    fn a_stamped_pid_is_the_process_attribute_and_an_unstamped_one_is_no_key() {
        // `process.pid` is the one per-process resource attribute: the SDK
        // stamps it live at drain time, which is what lets a backend tell a
        // fork parent's spans from a child's. 0 is proto3's "not stamped" and
        // emits NO key — `process.pid = 0` would claim the scheduler.
        let mut env = envelope(pb::Span::default());
        env.header.as_mut().unwrap().resource = Some(pb::ResourceInfo {
            service_name: "checkout-api".into(),
            process_pid: 4242,
            ..Default::default()
        });
        let req = envelope_to_traces(env, PRODUCER);
        assert_eq!(
            resource_attr(&req, "process.pid"),
            Some(&otlp_pb::common::any_value::Value::IntValue(4242))
        );

        let mut unstamped = envelope(pb::Span::default());
        unstamped.header.as_mut().unwrap().resource = Some(pb::ResourceInfo {
            service_name: "checkout-api".into(),
            ..Default::default()
        });
        let req = envelope_to_traces(unstamped, PRODUCER);
        assert!(resource_attr(&req, "process.pid").is_none());
    }

    #[test]
    fn the_telemetry_sdk_is_wardex_in_every_language() {
        // `telemetry.sdk.name` is the cross-language constant; the language
        // travels in its own attribute and the version stays the SDK's.
        let req = envelope_to_traces(envelope(pb::Span::default()), PRODUCER);
        assert_eq!(
            resource_attr(&req, "telemetry.sdk.name"),
            Some(&str_value("wardex"))
        );
        assert_eq!(
            resource_attr(&req, "telemetry.sdk.version"),
            Some(&str_value("0.1.0"))
        );
        assert_eq!(
            resource_attr(&req, "telemetry.sdk.language"),
            Some(&str_value("python"))
        );
    }

    #[test]
    fn error_type_is_an_attribute_and_the_status_message_survives() {
        // The inverse of the substitution this mapping used to make: semconv's
        // home for the exception class is the `error.type` attribute, and
        // `Status.message` is the one field a backend renders as "what went
        // wrong" — overwriting it destroyed the message to relabel it with a
        // fact that now travels beside it.
        let env = envelope(pb::Span {
            error_type: "TimeoutError".into(),
            status: Some(pb::Status {
                code: pb::StatusCode::Error as i32,
                message: "boom".into(),
            }),
            ..Default::default()
        });
        let sp = only_span(env);
        assert_eq!(sp.status.as_ref().unwrap().message, "boom");
        assert_eq!(attr(&sp, "error.type"), Some(&str_value("TimeoutError")));
    }

    #[test]
    fn a_span_with_no_error_type_carries_no_error_type_key() {
        let sp = only_span(envelope(pb::Span {
            status: Some(pb::Status {
                code: pb::StatusCode::Error as i32,
                message: "boom".into(),
            }),
            ..Default::default()
        }));
        assert!(attr(&sp, "error.type").is_none());
        assert_eq!(sp.status.unwrap().message, "boom");
    }

    #[test]
    fn url_full_keeps_the_query_and_is_absent_when_nothing_was_captured() {
        let with_query = |url: &str| {
            envelope(pb::Span {
                transport: Some(pb::TransportAttributes {
                    protocol: pb::Protocol::Http as i32,
                    http: Some(pb::HttpMeta {
                        method: "POST".into(),
                        url: url.into(),
                        status_code: 200,
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            })
        };
        // The mapping carries the URL as captured; masking is not its job.
        let sp = only_span(with_query(
            "https://api.example.com/v1/chat?q=seoul&page=2#frag",
        ));
        assert_eq!(
            attr(&sp, "url.full"),
            Some(&str_value(
                "https://api.example.com/v1/chat?q=seoul&page=2#frag"
            ))
        );
        // An empty captured URL emits nothing at all.
        let sp = only_span(with_query(""));
        assert!(attr(&sp, "url.full").is_none());
    }

    #[test]
    fn an_sse_span_is_http_on_the_wire_and_sse_under_the_wardex_key() {
        // `network.protocol.name = "sse"` fails every backend's HTTP grouping;
        // SSE is a framing over HTTP and the observed fact keeps a key of its
        // own instead of being dropped.
        let env = envelope(pb::Span {
            transport: Some(pb::TransportAttributes {
                protocol: pb::Protocol::Sse as i32,
                ..Default::default()
            }),
            ..Default::default()
        });
        let sp = only_span(env);
        assert_eq!(attr(&sp, "network.protocol.name"), Some(&str_value("http")));
        assert_eq!(
            attr(&sp, "wardex.transport.protocol"),
            Some(&str_value("sse"))
        );
        // Every other protocol is unchanged and carries no wardex twin.
        let env = envelope(pb::Span {
            transport: Some(pb::TransportAttributes {
                protocol: pb::Protocol::Grpc as i32,
                ..Default::default()
            }),
            ..Default::default()
        });
        let sp = only_span(env);
        assert_eq!(attr(&sp, "network.protocol.name"), Some(&str_value("grpc")));
        assert!(attr(&sp, "wardex.transport.protocol").is_none());
    }

    #[test]
    fn a_tool_calls_payload_ships_under_the_semconv_tool_keys() {
        // Same pipeline, same masking, same caps — only the key differs, and
        // only for `execute_tool`: that operation's payload has a semconv home
        // (`gen_ai.tool.call.arguments`/`result`) and shipping it under
        // `wardex.*` hid it from every backend that renders the tool view.
        let env = envelope(pb::Span {
            name: "execute_tool Bash".into(),
            extra: vec![kv_wardex("gen_ai.operation.name", "execute_tool")],
            input_data: b"{\"command\":\"ls\"}".to_vec(),
            output_data: b"README.md".to_vec(),
            ..Default::default()
        });
        let mut req = envelope_to_traces(env, PRODUCER);
        strip_bytes_values(&mut req);
        let sp = &req.resource_spans[0].scope_spans[0].spans[0];
        assert_eq!(
            attr(sp, "gen_ai.tool.call.arguments"),
            Some(&str_value("{\"command\":\"ls\"}"))
        );
        assert_eq!(
            attr(sp, "gen_ai.tool.call.result"),
            Some(&str_value("README.md"))
        );
        assert!(attr(sp, "wardex.input_data").is_none());
        assert!(attr(sp, "wardex.output_data").is_none());
    }

    #[test]
    fn a_tool_payloads_companions_follow_the_key_it_landed_under() {
        // The base64 companion and the payload drop both key off whichever
        // pair the payload used — a companion under the OLD key would tell a
        // consumer to decode an attribute that is not there.
        let env = envelope(pb::Span {
            extra: vec![kv_wardex("gen_ai.operation.name", "execute_tool")],
            input_data: b"\x89PNG\xff\x00binary".to_vec(),
            ..Default::default()
        });
        let mut req = envelope_to_traces(env, PRODUCER);
        strip_bytes_values(&mut req);
        let sp = &mut req.resource_spans[0].scope_spans[0].spans[0];
        assert_eq!(
            attr(sp, "gen_ai.tool.call.arguments.encoding"),
            Some(&str_value("base64"))
        );
        drop_payload_attributes(sp);
        assert!(attr(sp, "gen_ai.tool.call.arguments").is_none());
        assert!(attr(sp, "gen_ai.tool.call.arguments.encoding").is_none());
        assert_eq!(markers(sp), vec!["otlp_attribute_truncated"]);
    }

    #[test]
    fn every_other_operation_keeps_the_wardex_payload_keys() {
        let env = envelope(pb::Span {
            extra: vec![kv_wardex("gen_ai.operation.name", "chat")],
            input_data: b"prompt".to_vec(),
            output_data: b"answer".to_vec(),
            ..Default::default()
        });
        let mut req = envelope_to_traces(env, PRODUCER);
        strip_bytes_values(&mut req);
        let sp = &req.resource_spans[0].scope_spans[0].spans[0];
        assert_eq!(attr(sp, "wardex.input_data"), Some(&str_value("prompt")));
        assert_eq!(attr(sp, "wardex.output_data"), Some(&str_value("answer")));
        assert!(attr(sp, "gen_ai.tool.call.arguments").is_none());
    }

    #[test]
    fn an_array_valued_extra_survives_onto_the_otlp_wire() {
        // The list-valued gen_ai keys (stop_sequences, finish_reasons,
        // encoding_formats) arrive in `extra` as wardex ArrayValues now; a
        // remap that dropped containers would delete exactly them.
        let env = envelope(pb::Span {
            extra: vec![pb::KeyValue {
                key: "gen_ai.response.finish_reasons".into(),
                value: Some(pb::AnyValue {
                    value: Some(pb::any_value::Value::ArrayValue(pb::ArrayValue {
                        values: vec![pb::AnyValue {
                            value: Some(pb::any_value::Value::StringValue("stop".into())),
                        }],
                    })),
                }),
            }],
            ..Default::default()
        });
        let sp = only_span(env);
        assert_eq!(
            attr(&sp, "gen_ai.response.finish_reasons"),
            Some(&otlp_pb::common::any_value::Value::ArrayValue(
                otlp_pb::common::ArrayValue {
                    values: vec![otlp_pb::common::AnyValue {
                        value: Some(str_value("stop")),
                    }],
                }
            ))
        );
    }

    #[test]
    fn confidence_reaches_the_wire_as_the_number_the_sender_wrote() {
        // 0.9 has no exact f32, so widening the bits would put
        // 0.8999999761581421 on a confidence bar.
        let env = envelope(pb::Span {
            correlation: Some(pb::CorrelationInfo {
                confidence: 0.9,
                parent_source: pb::ParentSource::UnitAlias as i32,
                ..Default::default()
            }),
            ..Default::default()
        });
        let sp = only_span(env);
        assert_eq!(
            attr(&sp, "wardex.parent_confidence"),
            Some(&otlp_pb::common::any_value::Value::DoubleValue(0.9))
        );
        assert_eq!(
            attr(&sp, "wardex.parent_source"),
            Some(&otlp_pb::common::any_value::Value::StringValue(
                "unit_alias".into()
            ))
        );
    }

    #[test]
    fn a_span_with_nothing_to_report_carries_no_uncertainty_attributes() {
        // Absence has to stay legible: a span that reports no limitation must
        // not be padded with keys that make it look examined and cleared.
        let sp = only_span(envelope(pb::Span::default()));
        assert!(attr(&sp, "wardex.limitations").is_none());
        assert!(attr(&sp, "wardex.parent_source").is_none());
        assert!(attr(&sp, "wardex.capture.truncated").is_none());
    }

    #[test]
    fn limitations_reach_the_wire_as_names() {
        let env = envelope(pb::Span {
            capture_integrity: Some(pb::CaptureIntegrity {
                limitation_codes: vec![pb::Limitation::BodyCapExceeded as i32],
                truncated: true,
                dropped_chunk_count: 3,
                ..Default::default()
            }),
            ..Default::default()
        });
        let sp = only_span(env);
        assert_eq!(
            attr(&sp, "wardex.limitations"),
            Some(&otlp_pb::common::any_value::Value::ArrayValue(
                otlp_pb::common::ArrayValue {
                    values: vec![otlp_pb::common::AnyValue {
                        value: Some(otlp_pb::common::any_value::Value::StringValue(
                            "body_cap_exceeded".into()
                        )),
                    }],
                }
            ))
        );
        assert_eq!(
            attr(&sp, "wardex.capture.dropped_chunks"),
            Some(&otlp_pb::common::any_value::Value::IntValue(3))
        );
    }

    #[test]
    fn text_payloads_ship_verbatim_and_binary_goes_base64() {
        let env = envelope(pb::Span {
            input_data: b"\x89PNG\xff\x00binary".to_vec(),
            output_data: b"plain text".to_vec(),
            ..Default::default()
        });
        let mut req = envelope_to_traces(env, PRODUCER);
        strip_bytes_values(&mut req);
        let sp = &req.resource_spans[0].scope_spans[0].spans[0];
        assert_eq!(
            attr(sp, "wardex.input_data"),
            Some(&otlp_pb::common::any_value::Value::StringValue(
                BASE64.encode(b"\x89PNG\xff\x00binary")
            ))
        );
        assert_eq!(
            attr(sp, "wardex.input_data.encoding"),
            Some(&otlp_pb::common::any_value::Value::StringValue(
                "base64".into()
            ))
        );
        assert_eq!(
            attr(sp, "wardex.output_data"),
            Some(&otlp_pb::common::any_value::Value::StringValue(
                "plain text".into()
            ))
        );
        assert!(attr(sp, "wardex.output_data.encoding").is_none());
    }

    // --- size caps ---

    fn limits_with_attribute_cap(cap: usize) -> Limits {
        Limits {
            max_otlp_attribute_bytes: cap,
            ..Limits::default()
        }
    }

    fn markers(sp: &otlp_pb::trace::Span) -> Vec<String> {
        match attr(sp, "wardex.limitations") {
            Some(otlp_pb::common::any_value::Value::ArrayValue(a)) => a
                .values
                .iter()
                .filter_map(|v| match v.value.as_ref() {
                    Some(otlp_pb::common::any_value::Value::StringValue(s)) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            _ => vec![],
        }
    }

    /// The whole pipeline in the order the binding runs it, so a test cannot
    /// pass under an ordering the export path does not use.
    fn capped(env: pb::Envelope, limits: Limits) -> otlp_pb::trace::Span {
        let mut req = envelope_to_traces(env, PRODUCER);
        cap_attribute_values(&mut req, limits);
        strip_bytes_values(&mut req);
        req.resource_spans[0].scope_spans[0].spans[0].clone()
    }

    #[test]
    fn a_binary_payload_is_capped_by_what_base64_will_cost_not_by_its_raw_size() {
        // 96 raw bytes is under a 100-byte bound and 128 base64 characters is
        // not. Measuring the raw size here is the bug the whole pass exists to
        // fix, so the payload is chosen to sit exactly in the gap.
        let env = envelope(pb::Span {
            input_data: vec![0u8; 96],
            ..Default::default()
        });
        let sp = capped(env, limits_with_attribute_cap(100));
        let Some(otlp_pb::common::any_value::Value::StringValue(s)) =
            attr(&sp, "wardex.input_data")
        else {
            panic!("payload attribute missing");
        };
        assert!(s.len() <= 100, "{} characters on the wire", s.len());
        assert_eq!(BASE64.decode(s).unwrap().len(), 100 / 4 * 3);
        assert_eq!(markers(&sp), vec!["otlp_attribute_truncated"]);
    }

    #[test]
    fn a_capped_binary_payload_is_still_decodable() {
        // Truncating the base64 STRING instead of the bytes behind it lands
        // mid-quantum and hands the consumer something that will not decode —
        // a payload lost in a way that looks like corruption rather than like
        // a cap.
        for raw in [3000usize, 3001, 3002, 3003] {
            let env = envelope(pb::Span {
                input_data: (0..raw).map(|i| (i % 251) as u8).collect(),
                ..Default::default()
            });
            let sp = capped(env, limits_with_attribute_cap(1000));
            let Some(otlp_pb::common::any_value::Value::StringValue(s)) =
                attr(&sp, "wardex.input_data")
            else {
                panic!("payload attribute missing");
            };
            assert!(BASE64.decode(s).is_ok(), "raw {raw} did not decode");
        }
    }

    #[test]
    fn a_text_payload_is_cut_on_a_character_boundary() {
        // `String::truncate` panics mid-character, and this runs on the export
        // worker inside the host's process.
        let env = envelope(pb::Span {
            output_data: "한글".repeat(200).into_bytes(),
            ..Default::default()
        });
        let sp = capped(env, limits_with_attribute_cap(100));
        let Some(otlp_pb::common::any_value::Value::StringValue(s)) =
            attr(&sp, "wardex.output_data")
        else {
            panic!("payload attribute missing");
        };
        assert!(s.len() <= 100);
        assert_eq!(s.len() % 3, 0, "cut between the bytes of a character");
        assert_eq!(markers(&sp), vec!["otlp_attribute_truncated"]);
    }

    #[test]
    fn the_cheap_early_return_agrees_with_the_measurement_it_skips() {
        // The early return answers "fits either way" by arithmetic instead of
        // by measuring, so it has to be exactly right at its own edge: one byte
        // too generous and an oversized payload ships unmarked.
        for cap in [16usize, 17, 18, 19, 100, 1000] {
            let edge = cap / 4 * 3;
            for raw in [edge, edge + 1] {
                let env = envelope(pb::Span {
                    // Not text: the base64 branch is the one the arithmetic is
                    // about, and the one with room to be wrong.
                    input_data: (0..raw).map(|i| 0x80 | (i % 64) as u8).collect(),
                    ..Default::default()
                });
                let sp = capped(env, limits_with_attribute_cap(cap));
                let Some(otlp_pb::common::any_value::Value::StringValue(s)) =
                    attr(&sp, "wardex.input_data")
                else {
                    panic!("payload attribute missing");
                };
                assert!(s.len() <= cap, "cap {cap}, raw {raw}: {} bytes", s.len());
                assert_eq!(
                    markers(&sp).is_empty(),
                    raw <= edge,
                    "cap {cap}, raw {raw}: marker disagrees with whether it was cut"
                );
            }
        }
    }

    #[test]
    fn a_payload_inside_the_bound_is_untouched_and_unmarked() {
        // Absence has to stay legible: a span padded with a truncation marker
        // it did not earn is indistinguishable from one that lost data.
        let env = envelope(pb::Span {
            output_data: b"plain text".to_vec(),
            ..Default::default()
        });
        let sp = capped(env, limits_with_attribute_cap(100));
        assert_eq!(
            attr(&sp, "wardex.output_data"),
            Some(&otlp_pb::common::any_value::Value::StringValue(
                "plain text".into()
            ))
        );
        assert!(attr(&sp, "wardex.limitations").is_none());
    }

    #[test]
    fn the_cap_reaches_event_attributes_and_marks_the_span_that_owns_them() {
        // An event has no `wardex.limitations` of its own, so a loss inside one
        // is reported on the span or not at all.
        let mut span = pb::Span::default();
        span.events.push(pb::SpanEvent {
            name: "gen_ai.content.prompt".into(),
            attributes: vec![pb::KeyValue {
                key: "content".into(),
                value: Some(pb::AnyValue {
                    value: Some(pb::any_value::Value::StringValue("x".repeat(5000))),
                }),
            }],
            ..Default::default()
        });
        let sp = capped(envelope(span), limits_with_attribute_cap(100));
        assert_eq!(
            sp.events[0].attributes[0].value.as_ref().unwrap().value,
            Some(otlp_pb::common::any_value::Value::StringValue(
                "x".repeat(100)
            ))
        );
        assert_eq!(markers(&sp), vec!["otlp_attribute_truncated"]);
    }

    #[test]
    fn a_truncation_marker_joins_the_markers_the_span_already_carried() {
        // The existing mechanism, not a parallel one: a span that was already
        // reporting a limitation must end up with both, in one attribute.
        let env = envelope(pb::Span {
            input_data: vec![0u8; 4096],
            capture_integrity: Some(pb::CaptureIntegrity {
                limitation_codes: vec![pb::Limitation::BodyCapExceeded as i32],
                ..Default::default()
            }),
            ..Default::default()
        });
        let sp = capped(env, limits_with_attribute_cap(100));
        assert_eq!(
            markers(&sp),
            vec!["body_cap_exceeded", "otlp_attribute_truncated"]
        );
    }

    #[test]
    fn two_truncated_attributes_report_one_marker() {
        let env = envelope(pb::Span {
            input_data: vec![0u8; 4096],
            output_data: vec![1u8; 4096],
            ..Default::default()
        });
        let sp = capped(env, limits_with_attribute_cap(100));
        assert_eq!(markers(&sp), vec!["otlp_attribute_truncated"]);
    }

    #[test]
    fn dropping_the_payload_takes_its_encoding_companion_with_it() {
        // A `.encoding = base64` left behind describes a value that is no
        // longer there, and tells a consumer to decode an attribute it cannot
        // find.
        let env = envelope(pb::Span {
            input_data: b"\x89PNG\xff\x00binary".to_vec(),
            extra: vec![kv_wardex("gen_ai.request.model", "gpt-4.1-mini")],
            ..Default::default()
        });
        let mut req = envelope_to_traces(env, PRODUCER);
        strip_bytes_values(&mut req);
        let sp = &mut req.resource_spans[0].scope_spans[0].spans[0];
        drop_payload_attributes(sp);
        assert!(attr(sp, "wardex.input_data").is_none());
        assert!(attr(sp, "wardex.input_data.encoding").is_none());
        // The semantics survive: the span keeps its place in the trace and
        // still says which model it called.
        assert_eq!(
            attr(sp, "gen_ai.request.model"),
            Some(&otlp_pb::common::any_value::Value::StringValue(
                "gpt-4.1-mini".into()
            ))
        );
        assert_eq!(markers(sp), vec!["otlp_attribute_truncated"]);
    }

    #[test]
    fn dropping_a_payload_that_was_never_there_marks_nothing() {
        let mut req = envelope_to_traces(envelope(pb::Span::default()), PRODUCER);
        let sp = &mut req.resource_spans[0].scope_spans[0].spans[0];
        drop_payload_attributes(sp);
        assert!(attr(sp, "wardex.limitations").is_none());
    }

    use otlp_pb::common::any_value::Value as V;

    fn keys(sp: &otlp_pb::trace::Span, key: &str) -> usize {
        sp.attributes.iter().filter(|kv| kv.key == key).count()
    }

    /// The typed conversation field is the sender's ONLY home for these
    /// values, so this projection is all an OTLP backend sees of them. The
    /// escape it closes: the sender wrote the keys into `extra` and left the
    /// typed field empty, which a receiver of the envelope — reading the typed
    /// field — stored as an empty conversation id on every span. Found by
    /// decoding a real encode and looking for the field, not by a failing run.
    #[test]
    fn the_typed_conversation_projects_onto_the_keys_a_backend_groups_by() {
        let sp = only_span(envelope(pb::Span {
            conversation: Some(pb::ConversationContext {
                conversation_id: "conv-1".into(),
                session_id: "sess-1".into(),
                turn_index: 2,
            }),
            ..Default::default()
        }));
        assert_eq!(
            attr(&sp, "gen_ai.conversation.id"),
            Some(&V::StringValue("conv-1".into()))
        );
        assert_eq!(
            attr(&sp, "wardex.conversation.session_id"),
            Some(&V::StringValue("sess-1".into()))
        );
        assert_eq!(
            attr(&sp, "wardex.conversation.turn_index"),
            Some(&V::IntValue(2))
        );
    }

    #[test]
    fn an_unset_session_and_turn_add_no_keys() {
        let sp = only_span(envelope(pb::Span {
            conversation: Some(pb::ConversationContext {
                conversation_id: "conv-1".into(),
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert!(attr(&sp, "gen_ai.conversation.id").is_some());
        assert!(attr(&sp, "wardex.conversation.session_id").is_none());
        assert!(attr(&sp, "wardex.conversation.turn_index").is_none());
    }

    #[test]
    fn a_span_with_no_typed_blocks_gains_no_keys() {
        let sp = only_span(envelope(pb::Span::default()));
        for key in [
            "gen_ai.conversation.id",
            "wardex.conversation.session_id",
            "wardex.conversation.turn_index",
            "wardex.capture_sources",
            "code.file.path",
            "code.line.number",
            "code.function.name",
        ] {
            assert!(attr(&sp, key).is_none(), "{key} on an empty span");
        }
    }

    /// A host that spelled a typed field's key into `extra` by hand gets the
    /// TYPED value, once: a duplicated key is the backend's coin toss, and the
    /// host's value winning would make OTLP disagree with the envelope a
    /// receiver reads. Found by review: `wardex.capture_sources` was pushed
    /// bare and shipped twice.
    #[test]
    fn a_typed_value_replaces_a_same_keyed_extra_and_appears_once() {
        let sp = only_span(envelope(pb::Span {
            extra: vec![
                kv_wardex("gen_ai.conversation.id", "host-said"),
                kv_wardex("wardex.capture_sources", "host-said"),
                kv_wardex("code.file.path", "host-said"),
            ],
            conversation: Some(pb::ConversationContext {
                conversation_id: "typed".into(),
                ..Default::default()
            }),
            call_site: Some(pb::CallSite {
                file: "/typed.py".into(),
                ..Default::default()
            }),
            capture_sources: vec![pb::CaptureSource::Adapter as i32],
            ..Default::default()
        }));
        for key in [
            "gen_ai.conversation.id",
            "wardex.capture_sources",
            "code.file.path",
        ] {
            assert_eq!(keys(&sp, key), 1, "{key}");
        }
        assert_eq!(
            attr(&sp, "gen_ai.conversation.id"),
            Some(&V::StringValue("typed".into()))
        );
        assert_eq!(
            attr(&sp, "code.file.path"),
            Some(&V::StringValue("/typed.py".into()))
        );
        assert!(matches!(
            attr(&sp, "wardex.capture_sources"),
            Some(V::ArrayValue(_))
        ));
    }

    #[test]
    fn the_call_site_takes_the_stable_code_names_with_the_module_in_the_function() {
        let sp = only_span(envelope(pb::Span {
            call_site: Some(pb::CallSite {
                file: "/app/booking.py".into(),
                line: 42,
                function: "reserve".into(),
                module: "app.booking".into(),
            }),
            ..Default::default()
        }));
        assert_eq!(
            attr(&sp, "code.file.path"),
            Some(&V::StringValue("/app/booking.py".into()))
        );
        assert_eq!(attr(&sp, "code.line.number"), Some(&V::IntValue(42)));
        assert_eq!(
            attr(&sp, "code.function.name"),
            Some(&V::StringValue("app.booking.reserve".into()))
        );
    }

    #[test]
    fn a_call_site_without_a_module_names_the_bare_function() {
        let sp = only_span(envelope(pb::Span {
            call_site: Some(pb::CallSite {
                function: "reserve".into(),
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert_eq!(
            attr(&sp, "code.function.name"),
            Some(&V::StringValue("reserve".into()))
        );
        assert!(attr(&sp, "code.file.path").is_none());
        assert!(attr(&sp, "code.line.number").is_none());
    }

    /// How a span was captured reaches an OTLP backend by name. Zero is
    /// "unset" and names nothing; a number this build does not know declares
    /// itself rather than vanishing.
    #[test]
    fn capture_sources_project_by_name() {
        let sp = only_span(envelope(pb::Span {
            capture_sources: vec![
                pb::CaptureSource::Adapter as i32,
                pb::CaptureSource::Unspecified as i32,
                pb::CaptureSource::Ssl as i32,
                9_999,
            ],
            ..Default::default()
        }));
        let Some(V::ArrayValue(arr)) = attr(&sp, "wardex.capture_sources") else {
            panic!("wardex.capture_sources is missing or not an array");
        };
        // Owned strings joined for the comparison, deliberately: a borrowed
        // string vector is the shape the limitation census reads as a marker
        // channel, and it registers the binding's NAME repository-wide.
        let got = arr
            .values
            .iter()
            .filter_map(|v| match v.value.as_ref() {
                Some(V::StringValue(s)) => Some(s.clone()),
                _ => None,
            })
            .collect::<Vec<String>>()
            .join(",");
        assert_eq!(got, "adapter,ssl,capture_source_unrecognized_9999");
    }

    // --- the wire field census ---

    /// `Msg.field` for every field of the messages `WIRE_FIELDS` answers for,
    /// read off the schema file itself rather than off a list kept beside it.
    fn schema_fields() -> Vec<String> {
        const MESSAGES: &[&str] = &[
            "Span",
            "TransportAttributes",
            "TransportTiming",
            "HttpMeta",
            "GrpcMeta",
            "WebSocketMeta",
            "McpMeta",
            "SseMeta",
            "A2aMeta",
        ];
        let schema = include_str!("../../../../proto/wardex/v1/span.proto");
        let mut out = Vec::new();
        let mut current: Option<&str> = None;
        for raw in schema.lines() {
            let line = raw.split("//").next().unwrap_or("").trim();
            if let Some(rest) = line.strip_prefix("message ") {
                current = rest.split_whitespace().next();
                continue;
            }
            if line == "}" {
                current = None;
                continue;
            }
            let Some(msg) = current.filter(|m| MESSAGES.contains(m)) else {
                continue;
            };
            if line.is_empty() || line.starts_with("reserved") || !line.contains('=') {
                continue;
            }
            let decl = line
                .trim_start_matches("optional ")
                .trim_start_matches("repeated ");
            let name = decl
                .split_whitespace()
                .nth(1)
                .expect("a field line is `type name = n;`");
            out.push(format!("{msg}.{name}"));
        }
        out
    }

    #[test]
    fn every_span_and_transport_field_has_a_stated_otlp_home() {
        let schema = schema_fields();
        assert!(
            schema.len() > 60,
            "the schema parse found too few fields: {schema:?}"
        );
        let census: Vec<&str> = WIRE_FIELDS.iter().map(|(path, _)| *path).collect();
        let mut seen = std::collections::HashSet::new();
        for path in &census {
            assert!(seen.insert(*path), "{path} is listed twice");
        }
        for field in &schema {
            assert!(
                seen.contains(field.as_str()),
                "{field} is in span.proto but WIRE_FIELDS does not say where it goes"
            );
        }
        for path in &census {
            assert!(
                schema.iter().any(|f| f == path),
                "{path} is in WIRE_FIELDS but not in span.proto"
            );
        }
        for (path, home) in WIRE_FIELDS {
            if let OtlpHome::NotExported(reason) = home {
                assert!(
                    reason.split_whitespace().count() >= 8,
                    "{path}: a reason a stranger could act on"
                );
            }
        }
    }

    fn sentinel_transport() -> pb::TransportAttributes {
        // Exhaustive literals, no `..Default::default()`: a field added to
        // one of these messages fails to compile here, next to the sentinel
        // it needs.
        pb::TransportAttributes {
            connection_id: "sentinel-conn".into(),
            protocol: pb::Protocol::Http as i32,
            direction: pb::Direction::Inbound as i32,
            timing: Some(pb::TransportTiming {
                tcp_connect_ms: Some(11.25),
                tls_handshake_ms: Some(22.5),
                ttfb_ms: Some(33.75),
                transfer_ms: Some(44.0),
                ttft_ms: Some(55.5),
            }),
            request_size: Some(4321),
            response_size: Some(8765),
            http: Some(pb::HttpMeta {
                method: "PATCH".into(),
                status_code: 418,
                url: "http://sentinel/x?q=1".into(),
            }),
            grpc: Some(pb::GrpcMeta {
                service: "sentinel.Svc".into(),
                method: "Sentinel".into(),
                stream_id: 77,
                status_code: 13,
                encoding: "sentinel-enc".into(),
                decoded_payload: "sentinel-payload".into(),
            }),
            websocket: Some(pb::WebSocketMeta {
                opcode: 9,
                direction: "sentinel-ws".into(),
            }),
            mcp: Some(pb::McpMeta {
                rpc_method: "tools/call".into(),
                rpc_id: "sentinel-rpc".into(),
            }),
            sse: Some(pb::SseMeta {
                event_type: "sentinel-event".into(),
            }),
            a2a: Some(pb::A2aMeta {
                task_id: "sentinel-task".into(),
                transport: "sentinel-a2a".into(),
            }),
            request_blob_ref: "sentinel-req-blob".into(),
            response_blob_ref: "sentinel-resp-blob".into(),
            request_modality: pb::Modality::Image as i32,
            response_modality: pb::Modality::Audio as i32,
            is_streaming: Some(true),
            connection_reused: Some(true),
        }
    }

    #[test]
    fn every_attribute_home_carries_the_value_the_sender_wrote() {
        let sp = only_span(envelope(pb::Span {
            error_type: "SentinelError".into(),
            server_address: "sentinel.host".into(),
            server_port: 4321,
            workflow_name: "sentinel-workflow".into(),
            transport: Some(sentinel_transport()),
            ..Default::default()
        }));
        let s = |v: &str| V::StringValue(v.into());
        let expected: &[(&str, V)] = &[
            ("Span.error_type", s("SentinelError")),
            ("Span.server_address", s("sentinel.host")),
            ("Span.server_port", V::IntValue(4321)),
            ("Span.workflow_name", s("sentinel-workflow")),
            ("TransportAttributes.connection_id", s("sentinel-conn")),
            ("TransportAttributes.direction", s("inbound")),
            ("TransportAttributes.request_size", V::IntValue(4321)),
            ("TransportAttributes.response_size", V::IntValue(8765)),
            ("TransportAttributes.is_streaming", V::BoolValue(true)),
            ("TransportAttributes.connection_reused", V::BoolValue(true)),
            ("TransportTiming.tcp_connect_ms", V::DoubleValue(11.25)),
            ("TransportTiming.tls_handshake_ms", V::DoubleValue(22.5)),
            ("TransportTiming.ttfb_ms", V::DoubleValue(33.75)),
            ("TransportTiming.transfer_ms", V::DoubleValue(44.0)),
            ("TransportTiming.ttft_ms", V::DoubleValue(55.5)),
            ("HttpMeta.method", s("PATCH")),
            ("HttpMeta.status_code", V::IntValue(418)),
            ("HttpMeta.url", s("http://sentinel/x?q=1")),
            ("McpMeta.rpc_method", s("tools/call")),
            ("McpMeta.rpc_id", s("sentinel-rpc")),
        ];
        let attribute_homes: Vec<(&str, &str)> = WIRE_FIELDS
            .iter()
            .filter_map(|(path, home)| match home {
                OtlpHome::Attribute(key) => Some((*path, *key)),
                _ => None,
            })
            .collect();
        assert_eq!(
            attribute_homes.len(),
            expected.len(),
            "every Attribute home needs a sentinel here, and nothing else does"
        );
        for (path, key) in attribute_homes {
            let (_, want) = expected
                .iter()
                .find(|(p, _)| *p == path)
                .unwrap_or_else(|| panic!("{path} has no sentinel in this test"));
            assert_eq!(attr(&sp, key), Some(want), "{path} under `{key}`");
        }
    }

    #[test]
    fn a_field_the_sdk_did_not_observe_emits_no_key() {
        let sp = only_span(envelope(pb::Span {
            transport: Some(pb::TransportAttributes {
                protocol: pb::Protocol::McpStdio as i32,
                timing: Some(pb::TransportTiming {
                    ttfb_ms: Some(3.5),
                    ..Default::default()
                }),
                mcp: Some(pb::McpMeta {
                    rpc_method: "notifications/initialized".into(),
                    rpc_id: String::new(),
                }),
                ..Default::default()
            }),
            ..Default::default()
        }));
        for key in [
            T_CONNECTION_ID,
            T_DIRECTION,
            T_REQUEST_SIZE,
            T_RESPONSE_SIZE,
            T_IS_STREAMING,
            T_CONNECTION_REUSED,
            T_TCP_CONNECT_MS,
            T_TLS_HANDSHAKE_MS,
            T_TRANSFER_MS,
            T_TTFT_MS,
            JSONRPC_REQUEST_ID,
            WORKFLOW_NAME,
        ] {
            assert!(
                attr(&sp, key).is_none(),
                "`{key}` was emitted with nothing observed"
            );
        }
        assert_eq!(attr(&sp, T_TTFB_MS), Some(&V::DoubleValue(3.5)));
        assert_eq!(
            attr(&sp, MCP_METHOD_NAME),
            Some(&str_value("notifications/initialized"))
        );
    }

    #[test]
    fn the_typed_workflow_name_replaces_a_same_keyed_extra_and_appears_once() {
        let sp = only_span(envelope(pb::Span {
            workflow_name: "typed".into(),
            extra: vec![kv_wardex(WORKFLOW_NAME, "host")],
            ..Default::default()
        }));
        assert_eq!(keys(&sp, WORKFLOW_NAME), 1);
        assert_eq!(attr(&sp, WORKFLOW_NAME), Some(&str_value("typed")));
    }

    fn kv_bool_extra(key: &str, v: bool) -> pb::KeyValue {
        pb::KeyValue {
            key: key.into(),
            value: Some(pb::AnyValue {
                value: Some(pb::any_value::Value::BoolValue(v)),
            }),
        }
    }

    fn mapped(sp: pb::Span) -> (otlp_pb::trace::Span, usize) {
        let Mapped {
            mut request,
            reserved_overwritten,
        } = map_envelope(envelope(sp), PRODUCER);
        let sp = request
            .resource_spans
            .remove(0)
            .scope_spans
            .remove(0)
            .spans
            .remove(0);
        (sp, reserved_overwritten)
    }

    fn reserved_keys(sp: &otlp_pb::trace::Span) -> Vec<&str> {
        sp.attributes
            .iter()
            .map(|kv| kv.key.as_str())
            .filter(|k| k.starts_with(RESERVED_TRANSPORT_PREFIX))
            .collect()
    }

    /// A span the SDK observed no transport for — a host's own span — ships
    /// no `wardex.transport.*` name, whatever the host set under one: the
    /// names say "the SDK measured this", and here it measured nothing.
    #[test]
    fn a_host_value_under_a_reserved_name_does_not_ship_as_a_measurement() {
        let (sp, overwritten) = mapped(pb::Span {
            name: "host-step".into(),
            extra: vec![
                kv_bool_extra(T_IS_STREAMING, true),
                kv_wardex(T_CONNECTION_ID, "host-conn"),
                kv_wardex("wardex.transport.anything_else", "host"),
                kv_wardex("wardex.transportation", "kept: not under the prefix"),
            ],
            ..Default::default()
        });
        assert_eq!(reserved_keys(&sp), Vec::<&str>::new());
        assert_eq!(overwritten, 3);
        assert_eq!(
            attr(&sp, "wardex.transportation"),
            Some(&str_value("kept: not under the prefix"))
        );
    }

    /// Where the SDK has a value, it replaces the host's under the same name,
    /// once: the duplicate this closes shipped the host's `true` beside the
    /// SDK's observed `false`, and a backend keeps one of the two.
    #[test]
    fn the_sdk_value_replaces_a_host_value_under_its_reserved_name_once() {
        let (sp, overwritten) = mapped(pb::Span {
            extra: vec![
                kv_bool_extra(T_IS_STREAMING, true),
                kv_wardex(T_CONNECTION_ID, "host-conn"),
                kv_wardex("wardex.transport.protocol", "host"),
                kv_wardex(T_TTFT_MS, "host"),
            ],
            transport: Some(pb::TransportAttributes {
                connection_id: "123456789012345".into(),
                protocol: pb::Protocol::Http as i32,
                is_streaming: Some(false),
                ..Default::default()
            }),
            ..Default::default()
        });
        assert_eq!(reserved_keys(&sp), vec![T_CONNECTION_ID, T_IS_STREAMING]);
        assert_eq!(attr(&sp, T_IS_STREAMING), Some(&V::BoolValue(false)));
        assert_eq!(
            attr(&sp, T_CONNECTION_ID),
            Some(&str_value("123456789012345"))
        );
        assert_eq!(overwritten, 4);
    }

    /// The count is of host attributes only: a span with none under a
    /// reserved name counts nothing, however many the SDK writes.
    #[test]
    fn the_sdks_own_transport_values_are_not_counted() {
        let (_, overwritten) = mapped(pb::Span {
            transport: Some(sentinel_transport()),
            ..Default::default()
        });
        assert_eq!(overwritten, 0);
    }

    /// A semantic-convention name the transport block projects onto gives way
    /// to the observed value too, without a count: those names are not
    /// reserved, and a host value under one ships where the SDK has none.
    #[test]
    fn an_observed_value_replaces_a_host_value_under_its_semconv_name() {
        let (sp, overwritten) = mapped(pb::Span {
            extra: vec![
                kv_wardex(MCP_METHOD_NAME, "host"),
                kv_wardex(JSONRPC_REQUEST_ID, "host-id"),
            ],
            transport: Some(pb::TransportAttributes {
                protocol: pb::Protocol::McpStdio as i32,
                mcp: Some(pb::McpMeta {
                    rpc_method: "tools/call".into(),
                    rpc_id: String::new(),
                }),
                ..Default::default()
            }),
            ..Default::default()
        });
        assert_eq!(keys(&sp, MCP_METHOD_NAME), 1);
        assert_eq!(attr(&sp, MCP_METHOD_NAME), Some(&str_value("tools/call")));
        assert_eq!(attr(&sp, JSONRPC_REQUEST_ID), Some(&str_value("host-id")));
        assert_eq!(overwritten, 0);
    }

    #[test]
    fn an_observed_false_or_zero_is_emitted_not_dropped() {
        // Presence is the whole point: `false` and `0.0` are readings when
        // the sender set them, and only an unset field is silent.
        let sp = only_span(envelope(pb::Span {
            transport: Some(pb::TransportAttributes {
                protocol: pb::Protocol::Http as i32,
                is_streaming: Some(false),
                connection_reused: Some(false),
                timing: Some(pb::TransportTiming {
                    tcp_connect_ms: Some(0.0),
                    ..Default::default()
                }),
                request_size: Some(0),
                response_size: Some(0),
                ..Default::default()
            }),
            ..Default::default()
        }));
        assert_eq!(attr(&sp, T_IS_STREAMING), Some(&V::BoolValue(false)));
        assert_eq!(attr(&sp, T_CONNECTION_REUSED), Some(&V::BoolValue(false)));
        assert_eq!(attr(&sp, T_TCP_CONNECT_MS), Some(&V::DoubleValue(0.0)));
        assert_eq!(attr(&sp, T_REQUEST_SIZE), Some(&V::IntValue(0)));
        assert_eq!(attr(&sp, T_RESPONSE_SIZE), Some(&V::IntValue(0)));
    }
}
