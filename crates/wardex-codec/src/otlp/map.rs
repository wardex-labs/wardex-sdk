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
        _ => None,
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
        if i.dropped_chunk_count > 0 {
            attrs.push(kv_int(
                "wardex.capture.dropped_chunks",
                i.dropped_chunk_count as i64,
            ));
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

fn span(mut sp: pb::Span) -> otlp_pb::trace::Span {
    use std::mem::take;

    // Read before anything is moved out: naming consults `extra`, and
    // `uncertainty` wants the whole span.
    let name = span_name(&sp);
    let code = status_code(sp.status.as_ref().map(|s| s.code).unwrap_or_default());

    // gen_ai / agent / tool attributes were flattened into `extra` when the
    // envelope was marshalled, so both export surfaces carry one flattening and
    // cannot drift.
    let mut attrs: Vec<otlp_pb::common::KeyValue> =
        take(&mut sp.extra).into_iter().map(key_value).collect();

    // server.* — the empty string and port 0 are proto3 "unset", and neither is
    // a value a host could have meant.
    if !sp.server_address.is_empty() {
        attrs.push(kv_str("server.address", &sp.server_address));
    }
    if sp.server_port != 0 {
        attrs.push(kv_int("server.port", sp.server_port as i64));
    }
    if let Some(t) = &sp.transport {
        attrs.push(kv_str(
            "network.protocol.name",
            &vocab::protocol_name(t.protocol),
        ));
        if let Some(h) = &t.http {
            attrs.push(kv_str("http.request.method", &h.method));
            attrs.push(kv_int("http.response.status_code", h.status_code as i64));
        }
    }
    // Raw I/O → wardex.input_data / wardex.output_data (omitted if empty).
    // Built as bytes so PII masking sees the raw payload; `strip_bytes_values`
    // converts to strings after masking, before serialization.
    let input = take(&mut sp.input_data);
    if !input.is_empty() {
        attrs.push(kv_bytes("wardex.input_data", input));
    }
    let output = take(&mut sp.output_data);
    if !output.is_empty() {
        attrs.push(kv_bytes("wardex.output_data", output));
    }
    uncertainty(&sp, &mut attrs);

    // `error.type` takes priority over the status message: on a failure it is
    // the more specific of the two, and it is empty exactly when there is no
    // error type to report.
    let message = if sp.error_type.is_empty() {
        sp.status
            .as_mut()
            .map(|s| take(&mut s.message))
            .unwrap_or_default()
    } else {
        take(&mut sp.error_type)
    };

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
        ..Default::default()
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
    mut env: pb::Envelope,
    producer: Producer<'_>,
) -> otlp_pb::trace_service::ExportTraceServiceRequest {
    let spans: Vec<otlp_pb::trace::Span> = std::mem::take(&mut env.items)
        .into_iter()
        .filter_map(|item| match item.payload {
            Some(pb::envelope_item::Payload::Span(sp)) => Some(span(sp)),
            _ => None,
        })
        .collect();
    if spans.is_empty() {
        return otlp_pb::trace_service::ExportTraceServiceRequest {
            resource_spans: vec![],
        };
    }
    let sdk = env.header.as_ref().and_then(|h| h.sdk.as_ref());
    let name = sdk.map(|s| s.name.as_str()).unwrap_or_default();
    let version = sdk.map(|s| s.version.as_str()).unwrap_or_default();
    otlp_pb::trace_service::ExportTraceServiceRequest {
        resource_spans: vec![otlp_pb::trace::ResourceSpans {
            resource: Some(otlp_pb::resource::Resource {
                attributes: vec![
                    kv_str("service.name", name),
                    kv_str("service.version", version),
                    kv_str("telemetry.sdk.name", name),
                    kv_str("telemetry.sdk.version", version),
                    kv_str("telemetry.sdk.language", producer.language),
                ],
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
    const PAYLOAD: [&str; 2] = ["wardex.input_data", "wardex.output_data"];
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

    #[test]
    fn the_scope_names_the_library_not_the_service() {
        // A host that renames its service must not rename the instrumentation
        // library that produced its spans.
        let mut env = envelope(pb::Span::default());
        env.header.as_mut().unwrap().sdk.as_mut().unwrap().name = "checkout-api".into();
        let req = envelope_to_traces(env, PRODUCER);
        let resource = req.resource_spans[0].resource.as_ref().unwrap();
        let service = resource
            .attributes
            .iter()
            .find(|kv| kv.key == "service.name")
            .unwrap();
        assert_eq!(
            service.value,
            Some(otlp_pb::common::AnyValue {
                value: Some(otlp_pb::common::any_value::Value::StringValue(
                    "checkout-api".into()
                )),
            })
        );
        let scope = req.resource_spans[0].scope_spans[0].scope.as_ref().unwrap();
        assert_eq!(scope.name, "wardex.python");
    }

    #[test]
    fn error_type_wins_over_the_status_message() {
        let env = envelope(pb::Span {
            error_type: "TimeoutError".into(),
            status: Some(pb::Status {
                code: pb::StatusCode::Error as i32,
                message: "boom".into(),
            }),
            ..Default::default()
        });
        assert_eq!(only_span(env).status.unwrap().message, "TimeoutError");
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
}
