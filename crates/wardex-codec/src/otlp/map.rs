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
fn any_value(v: &pb::AnyValue) -> otlp_pb::common::AnyValue {
    use otlp_pb::common::any_value::Value as Otlp;
    use pb::any_value::Value as Wardex;
    let value = match &v.value {
        Some(Wardex::StringValue(s)) => Some(Otlp::StringValue(s.clone())),
        Some(Wardex::IntValue(i)) => Some(Otlp::IntValue(*i)),
        Some(Wardex::DoubleValue(d)) => Some(Otlp::DoubleValue(*d)),
        Some(Wardex::BoolValue(b)) => Some(Otlp::BoolValue(*b)),
        // Bytes pass through here untouched: masking must still see the raw
        // payload. `strip_bytes_values` removes every bytes_value from the
        // request after masking, right before serialization.
        Some(Wardex::BytesValue(b)) => Some(Otlp::BytesValue(b.clone())),
        _ => None,
    };
    otlp_pb::common::AnyValue { value }
}

fn key_value(kv: &pb::KeyValue) -> otlp_pb::common::KeyValue {
    otlp_pb::common::KeyValue {
        key: kv.key.clone(),
        value: kv.value.as_ref().map(any_value),
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
            attrs.push(kv_strs("wardex.limitations", markers));
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
fn event(ev: &pb::SpanEvent) -> otlp_pb::trace::span::Event {
    otlp_pb::trace::span::Event {
        time_unix_nano: ev.time_unix_nano,
        name: ev.name.clone(),
        attributes: ev.attributes.iter().map(key_value).collect(),
        ..Default::default()
    }
}

/// `Span.links` → OTLP `Span.links`.
///
/// `reason` has no OTLP-native home — `Link` carries `trace_state` and
/// attributes and nothing else — so it travels as the `wardex.link.reason`
/// attribute rather than being dropped, as the wardex value string so that it
/// is readable without a copy of the enum.
fn link(ln: &pb::SpanLink) -> otlp_pb::trace::span::Link {
    let mut attributes: Vec<otlp_pb::common::KeyValue> =
        ln.attributes.iter().map(key_value).collect();
    let reason = vocab::link_reason_name(ln.reason);
    if !reason.is_empty() {
        attributes.push(kv_str("wardex.link.reason", &reason));
    }
    otlp_pb::trace::span::Link {
        trace_id: ln.trace_id.clone(),
        span_id: ln.span_id.clone(),
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

fn span(sp: &pb::Span) -> otlp_pb::trace::Span {
    // gen_ai / agent / tool attributes were flattened into `extra` when the
    // envelope was marshalled, so both export surfaces carry one flattening and
    // cannot drift.
    let mut attrs: Vec<otlp_pb::common::KeyValue> = sp.extra.iter().map(key_value).collect();

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
    if !sp.input_data.is_empty() {
        attrs.push(kv_bytes("wardex.input_data", sp.input_data.clone()));
    }
    if !sp.output_data.is_empty() {
        attrs.push(kv_bytes("wardex.output_data", sp.output_data.clone()));
    }
    uncertainty(sp, &mut attrs);

    // `error.type` takes priority over the status message: on a failure it is
    // the more specific of the two, and it is empty exactly when there is no
    // error type to report.
    let message = if sp.error_type.is_empty() {
        sp.status
            .as_ref()
            .map(|s| s.message.clone())
            .unwrap_or_default()
    } else {
        sp.error_type.clone()
    };
    let code = status_code(sp.status.as_ref().map(|s| s.code).unwrap_or_default());

    otlp_pb::trace::Span {
        trace_id: sp.trace_id.clone(),
        span_id: sp.span_id.clone(),
        parent_span_id: sp.parent_span_id.clone(),
        name: span_name(sp),
        kind: span_kind(sp.kind),
        start_time_unix_nano: sp.start_time_unix_nano,
        end_time_unix_nano: sp.end_time_unix_nano,
        status: Some(otlp_pb::trace::Status { code, message }),
        attributes: attrs,
        events: sp.events.iter().map(event).collect(),
        links: sp.links.iter().map(link).collect(),
        ..Default::default()
    }
}

/// Envelope → `ExportTraceServiceRequest`.
///
/// Traces only: state snapshots have no OTLP trace form and are skipped rather
/// than flattened into one. An envelope with no spans produces no
/// `resource_spans`, so a caller can tell "nothing to send" from "a batch of
/// empty spans" without decoding.
pub fn envelope_to_traces(
    env: &pb::Envelope,
    producer: Producer<'_>,
) -> otlp_pb::trace_service::ExportTraceServiceRequest {
    let spans: Vec<otlp_pb::trace::Span> = env
        .items
        .iter()
        .filter_map(|item| match &item.payload {
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

    fn only_span(env: &pb::Envelope) -> otlp_pb::trace::Span {
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
        let sp = only_span(&env);
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
        assert!(envelope_to_traces(&env, PRODUCER).resource_spans.is_empty());
    }

    #[test]
    fn the_scope_names_the_library_not_the_service() {
        // A host that renames its service must not rename the instrumentation
        // library that produced its spans.
        let mut env = envelope(pb::Span::default());
        env.header.as_mut().unwrap().sdk.as_mut().unwrap().name = "checkout-api".into();
        let req = envelope_to_traces(&env, PRODUCER);
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
        assert_eq!(only_span(&env).status.unwrap().message, "TimeoutError");
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
        let sp = only_span(&env);
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
        let sp = only_span(&envelope(pb::Span::default()));
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
        let sp = only_span(&env);
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
        let mut req = envelope_to_traces(&env, PRODUCER);
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
}
