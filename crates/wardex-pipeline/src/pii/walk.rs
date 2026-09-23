//! Proto-tree masking walks — design §4.
//!
//! Every message is destructured WITHOUT `..` so that adding a proto field
//! breaks this build until someone decides whether the new field is masked
//! (design §4.3). Non-text fields are bound to `_`.

use std::panic::{catch_unwind, AssertUnwindSafe};

use wardex_codec::otlp::otlp_pb;
use wardex_codec::proto::wardex::v1 as pb;

use super::{PiiEngine, Report, Rule, Shape, PII_FILTER_ERROR, PLACEHOLDER};

/// The engine plus what it has replaced so far in ONE record. A span gets a
/// fresh one, so its report says what was masked in that span and nowhere
/// else; the envelope header and snapshots use a scratch one.
struct Masker<'e> {
    engine: &'e PiiEngine,
    report: Report,
}

impl<'e> Masker<'e> {
    fn new(engine: &'e PiiEngine) -> Masker<'e> {
        Masker {
            engine,
            report: Report::default(),
        }
    }

    /// The rule a structured attribute's KEY puts its value under. A key is a
    /// name in the JSON sense — `span.set_attribute("api_key", v)` is the
    /// same argument as `{"api_key": v}` — so the `name=value`-only names do
    /// not apply.
    fn key_rule(&self, key: &str) -> Option<Rule> {
        self.engine.name_rules()?.judge(key, Shape::Json)
    }

    /// The report with its names passed through the value rules: a key that
    /// is itself an e-mail address must not leave the process in the list of
    /// names that were masked.
    fn finish(self) -> Report {
        let Masker { engine, mut report } = self;
        for n in &mut report.names {
            if let Some(masked) = engine.mask_text(n) {
                *n = masked;
            }
        }
        report
    }
}

/// Test-only panic injection for the fail-closed path.
#[cfg(test)]
pub(super) const TEST_PANIC_SPAN_NAME: &str = "__wardex_test_panic__";

/// Mask every outbound text surface of a wardex envelope (design §3).
/// A span/snapshot that panics during masking is scrubbed whole (fail-closed §8);
/// other items keep processing.
pub fn mask_envelope(engine: &PiiEngine, env: &mut pb::Envelope) {
    let pb::Envelope { header, items } = env;
    if let Some(h) = header {
        mask_header(&mut Masker::new(engine), h);
    }
    for item in items {
        let pb::EnvelopeItem { header, payload } = item;
        if let Some(pb::EnvelopeItemHeader { r#type, length: _ }) = header {
            // Item-level metadata, masked at the same tier as the envelope
            // header — it does not feed any per-span redacted flag.
            mask_string(&mut Masker::new(engine), r#type);
        }
        match payload {
            Some(pb::envelope_item::Payload::Span(span)) => {
                let masked = catch_unwind(AssertUnwindSafe(|| mask_span(engine, span)));
                if masked.is_err() {
                    scrub_span_fail_closed(span);
                }
            }
            Some(pb::envelope_item::Payload::StateSnapshot(snap)) => {
                let masked = catch_unwind(AssertUnwindSafe(|| {
                    mask_snapshot(&mut Masker::new(engine), snap)
                }));
                if masked.is_err() {
                    scrub_snapshot_fail_closed(snap);
                }
            }
            Some(pb::envelope_item::Payload::ClientReport(report)) => {
                let pb::ClientReport {
                    timestamp_ns: _,
                    // Map keys are our own internal event-type tags (same
                    // tier as KeyValue.key, deliberately unmasked); values
                    // are numeric counters.
                    discarded_events: _,
                    failed_sends: _,
                    queue_depth: _,
                    uptime_ms: _,
                } = report;
            }
            None => {}
        }
    }
}

// --- primitives ---

fn mask_string(m: &mut Masker<'_>, s: &mut String) -> bool {
    match m.engine.mask_text_into(s, &mut m.report) {
        Some(masked) => {
            *s = masked;
            true
        }
        None => false,
    }
}

/// Bytes are masked as text, run by run: every stretch that is valid UTF-8
/// is masked, and the bytes between stretches pass through untouched. One
/// Latin-1 `é` in a form body used to switch masking off for the whole body;
/// raw binary still passes through (documented limitation, design §4.4).
fn mask_bytes(m: &mut Masker<'_>, b: &mut Vec<u8>) -> bool {
    if let Ok(text) = std::str::from_utf8(b) {
        return match m.engine.mask_text_into(text, &mut m.report) {
            Some(masked) => {
                *b = masked.into_bytes();
                true
            }
            None => false,
        };
    }
    let mut out = Vec::with_capacity(b.len());
    let mut hit = false;
    for chunk in b.utf8_chunks() {
        match m.engine.mask_text_into(chunk.valid(), &mut m.report) {
            Some(masked) => {
                out.extend_from_slice(masked.as_bytes());
                hit = true;
            }
            None => out.extend_from_slice(chunk.valid().as_bytes()),
        }
        out.extend_from_slice(chunk.invalid());
    }
    if hit {
        *b = out;
    }
    hit
}

/// Keys are attribute NAMES: never rewritten, but judged by the name rules —
/// a host's `set_attribute("db.password", v)` is a secret argument like any
/// other.
fn mask_kvs(m: &mut Masker<'_>, kvs: &mut [pb::KeyValue]) -> bool {
    let mut hit = false;
    for kv in kvs.iter_mut() {
        let pb::KeyValue { key, value } = kv;
        if let Some(v) = value {
            hit |= match m.key_rule(key) {
                Some(rule) => replace_any(m, v, rule, key),
                None => mask_any(m, v),
            };
        }
    }
    hit
}

/// Replace a value that sits under a secret name. Scalars become the
/// placeholder (a number becomes a string, as it does in JSON); a list has
/// each element replaced; a nested map is walked, its own keys judged in turn.
fn replace_any(m: &mut Masker<'_>, v: &mut pb::AnyValue, rule: Rule, key: &str) -> bool {
    use pb::any_value::Value;
    let pb::AnyValue { value } = v;
    let replaced = match value {
        Some(Value::StringValue(s)) if !s.is_empty() && s != PLACEHOLDER => {
            *s = PLACEHOLDER.into();
            true
        }
        Some(Value::BytesValue(b)) if !b.is_empty() && b != PLACEHOLDER.as_bytes() => {
            *b = PLACEHOLDER.as_bytes().to_vec();
            true
        }
        Some(Value::IntValue(_)) | Some(Value::DoubleValue(_)) => {
            *value = Some(Value::StringValue(PLACEHOLDER.into()));
            true
        }
        Some(Value::ArrayValue(arr)) => {
            let pb::ArrayValue { values } = arr;
            let mut hit = false;
            for item in values {
                hit |= replace_any(m, item, rule, key);
            }
            return hit;
        }
        Some(Value::KvlistValue(kvl)) => {
            let pb::KeyValueList { values } = kvl;
            return mask_kvs(m, values);
        }
        _ => false,
    };
    if replaced {
        m.report.record(rule, Some(key));
    }
    replaced
}

fn mask_any(m: &mut Masker<'_>, v: &mut pb::AnyValue) -> bool {
    use pb::any_value::Value;
    let pb::AnyValue { value } = v;
    match value {
        Some(Value::StringValue(s)) => mask_string(m, s),
        Some(Value::BytesValue(b)) => mask_bytes(m, b),
        Some(Value::ArrayValue(arr)) => {
            let pb::ArrayValue { values } = arr;
            let mut hit = false;
            for item in values {
                hit |= mask_any(m, item);
            }
            hit
        }
        Some(Value::KvlistValue(kvl)) => {
            let pb::KeyValueList { values } = kvl;
            mask_kvs(m, values)
        }
        Some(Value::BoolValue(_))
        | Some(Value::IntValue(_))
        | Some(Value::DoubleValue(_))
        | None => false,
    }
}

// --- message walks ---

fn mask_header(m: &mut Masker<'_>, h: &mut pb::EnvelopeHeader) {
    let pb::EnvelopeHeader {
        event_id,
        sdk,
        sent_at_unix_nano: _,
        session_status: _,
        retention_class: _,
        resource,
        // WHITELIST: stamped by the receiver from the authenticated API key,
        // never host free text — the SDK sends it empty. Rewriting it would
        // detach a stored batch from its project.
        project_id: _,
    } = h;
    mask_string(m, event_id);
    if let Some(r) = resource {
        // The app's identity strings are host-supplied free text, so they get
        // the same treatment as SdkInfo's strings below.
        let pb::ResourceInfo {
            service_name,
            release,
            environment,
            // WHITELIST: an OS-assigned integer, not host free text — there is
            // nothing to mask, and rewriting it would destroy the per-process
            // attribution (fork parent vs child) the field exists to carry.
            process_pid: _,
        } = r;
        mask_string(m, service_name);
        mask_string(m, release);
        mask_string(m, environment);
    }
    if let Some(s) = sdk {
        let pb::SdkInfo {
            name,
            version,
            python_version,
            os,
            arch,
            adapters,
            interceptors,
            otel_semconv_version,
            shell,
        } = s;
        mask_string(m, name);
        mask_string(m, version);
        mask_string(m, python_version);
        mask_string(m, os);
        mask_string(m, arch);
        for a in adapters {
            mask_string(m, a);
        }
        for i in interceptors {
            mask_string(m, i);
        }
        mask_string(m, otel_semconv_version);
        mask_string(m, shell);
    }
}

fn mask_span(engine: &PiiEngine, span: &mut pb::Span) {
    let m = &mut Masker::new(engine);
    #[cfg(test)]
    if span.name == TEST_PANIC_SPAN_NAME {
        panic!("injected test panic");
    }
    let pb::Span {
        trace_id: _,       // binary id
        span_id: _,        // binary id
        parent_span_id: _, // binary id
        name,
        kind: _,
        start_time_unix_nano: _,
        end_time_unix_nano: _,
        status,
        extra,
        events,
        links,
        dropped_extra_count: _,
        dropped_events_count: _,
        dropped_links_count: _,
        input_data,
        output_data,
        transport,
        error_type,
        server_address,
        server_port: _,
        workflow_name,
        call_site,
        conversation,
        capture_sources: _,
        capture_integrity,
        correlation,
    } = span;
    let mut hit = false;
    hit |= mask_string(m, name);
    if let Some(pb::Status { code: _, message }) = status {
        hit |= mask_string(m, message);
    }
    hit |= mask_kvs(m, extra);
    for ev in events {
        let pb::SpanEvent {
            name,
            time_unix_nano: _,
            attributes,
        } = ev;
        hit |= mask_string(m, name);
        hit |= mask_kvs(m, attributes);
    }
    for link in links {
        let pb::SpanLink {
            trace_id: _,
            span_id: _,
            attributes,
            // A closed enum cannot carry PII by construction, and there is no
            // regex to run over an i32. This is the same disposition
            // `CaptureIntegrity.limitations` gets now that it is a repeated
            // closed enum rather than `repeated string` — see design §6.5.1's
            // "what Rust must receive".
            reason: _,
        } = link;
        hit |= mask_kvs(m, attributes);
    }
    hit |= mask_bytes(m, input_data);
    hit |= mask_bytes(m, output_data);
    if let Some(t) = transport {
        hit |= mask_transport(m, t);
    }
    hit |= mask_string(m, error_type);
    hit |= mask_string(m, server_address);
    hit |= mask_string(m, workflow_name);
    if let Some(pb::CallSite {
        file,
        line: _,
        function,
        module,
    }) = call_site
    {
        hit |= mask_string(m, file);
        hit |= mask_string(m, function);
        hit |= mask_string(m, module);
    }
    if let Some(pb::ConversationContext {
        conversation_id,
        session_id,
        turn_index: _,
    }) = conversation
    {
        hit |= mask_string(m, conversation_id);
        hit |= mask_string(m, session_id);
    }
    if let Some(pb::CorrelationInfo {
        operation_id,
        request_id,
        attempt_id,
        active_span_id_at_capture: _, // binary id
        confidence: _,
        // Was `strategy: String`, and the regexes ran over it. Now a closed
        // enum, so there is nothing to scan and nothing that could match: the
        // set of values is fixed by the schema and contains no user data.
        parent_source: _,
    }) = correlation
    {
        hit |= mask_string(m, operation_id);
        hit |= mask_string(m, request_id);
        hit |= mask_string(m, attempt_id);
    }
    // `CaptureIntegrity` is deliberately NOT destructured any more. Every one
    // of its fields is a bool, an i32 or a repeated closed enum; none can carry
    // PII, so masking it was work with no possible effect.
    // Note the ordering that survives: `redacted` and the report are written
    // below, and they have to stay after every other field has been scanned.
    let report = std::mem::replace(m, Masker::new(engine)).finish();
    if hit {
        let ci = capture_integrity.get_or_insert_with(Default::default);
        ci.redacted = true;
        ci.redaction_count = i32::try_from(report.count).unwrap_or(i32::MAX);
        ci.redaction_rules = report.rules.iter().map(|r| *r as i32).collect();
        ci.redaction_names = report.names;
    }
}

fn mask_transport(m: &mut Masker<'_>, t: &mut pb::TransportAttributes) -> bool {
    let pb::TransportAttributes {
        connection_id,
        protocol: _,
        direction: _,
        timing,
        request_size: _,
        response_size: _,
        http,
        grpc,
        websocket,
        mcp,
        sse,
        a2a,
        request_blob_ref,
        response_blob_ref,
        request_modality: _,
        response_modality: _,
        is_streaming: _,
        chunk_index: _,
        is_final_chunk: _,
        connection_reused: _,
    } = t;
    let mut hit = false;
    hit |= mask_string(m, connection_id);
    // All numeric today; the exhaustive destructure is the §4.3 compile-time
    // tripwire for future text fields.
    if let Some(pb::TransportTiming {
        tcp_connect_ms: _,
        tls_handshake_ms: _,
        ttfb_ms: _,
        transfer_ms: _,
        ttft_ms: _,
    }) = timing
    {}
    if let Some(pb::HttpMeta {
        method,
        status_code: _,
        url,
    }) = http
    {
        hit |= mask_string(m, method);
        hit |= mask_string(m, url);
    }
    if let Some(pb::GrpcMeta {
        service,
        method,
        stream_id: _,
        status_code: _,
        encoding,
        decoded_payload,
    }) = grpc
    {
        hit |= mask_string(m, service);
        hit |= mask_string(m, method);
        hit |= mask_string(m, encoding);
        hit |= mask_string(m, decoded_payload);
    }
    if let Some(pb::WebSocketMeta {
        opcode: _,
        direction,
    }) = websocket
    {
        hit |= mask_string(m, direction);
    }
    if let Some(pb::McpMeta { rpc_method, rpc_id }) = mcp {
        hit |= mask_string(m, rpc_method);
        hit |= mask_string(m, rpc_id);
    }
    if let Some(pb::SseMeta { event_type }) = sse {
        hit |= mask_string(m, event_type);
    }
    if let Some(pb::A2aMeta { task_id, transport }) = a2a {
        hit |= mask_string(m, task_id);
        hit |= mask_string(m, transport);
    }
    hit |= mask_string(m, request_blob_ref);
    hit |= mask_string(m, response_blob_ref);
    hit
}

fn mask_snapshot(m: &mut Masker<'_>, snap: &mut pb::StateSnapshot) {
    let pb::StateSnapshot {
        trace_id: _,
        span_id: _,
        timestamp_ns: _,
        snapshot_type: _,
        turn_index: _,
        attributes,
        conversation_state,
        input_refs,
        tool_definitions,
    } = snap;
    mask_kvs(m, attributes);
    mask_bytes(m, conversation_state);
    for r in input_refs {
        let pb::InputRef {
            key,
            content_hash,
            blob_ref,
        } = r;
        mask_string(m, key);
        mask_string(m, content_hash);
        mask_string(m, blob_ref);
    }
    if let Some(pb::ToolDefinitionSet { tools, set_hash }) = tool_definitions {
        for tool in tools {
            let pb::ToolDefinition {
                name,
                description,
                parameters_schema,
                version,
                r#type,
                hash,
            } = tool;
            mask_string(m, name);
            mask_string(m, description);
            mask_bytes(m, parameters_schema);
            mask_string(m, version);
            mask_string(m, r#type);
            mask_string(m, hash);
        }
        mask_string(m, set_hash);
    }
}

// --- OTLP masking (design §4.2, OTLP path) ---

/// Mask an OTLP export request (design §4.2, OTLP path). A span that panics
/// is scrubbed whole; a span with replacements gains `wardex.redacted=true`
/// (OTLP spans have no capture_integrity — this is the OTLP equivalent).
pub fn mask_otlp(engine: &PiiEngine, req: &mut otlp_pb::trace_service::ExportTraceServiceRequest) {
    let otlp_pb::trace_service::ExportTraceServiceRequest { resource_spans } = req;
    for rs in resource_spans {
        // Destructure exhaustively; if the vendored OTLP proto carries extra
        // fields the compiler will list them — bind non-text ones to `_`.
        let otlp_pb::trace::ResourceSpans {
            resource,
            scope_spans,
            schema_url,
        } = rs;
        if let Some(otlp_pb::resource::Resource {
            attributes,
            dropped_attributes_count: _,
        }) = resource
        {
            mask_otlp_kvs(&mut Masker::new(engine), attributes);
        }
        mask_string(&mut Masker::new(engine), schema_url);
        for ss in scope_spans {
            let otlp_pb::trace::ScopeSpans {
                scope,
                spans,
                schema_url,
            } = ss;
            if let Some(otlp_pb::common::InstrumentationScope {
                name,
                version,
                attributes,
                dropped_attributes_count: _,
            }) = scope
            {
                let m = &mut Masker::new(engine);
                mask_string(m, name);
                mask_string(m, version);
                mask_otlp_kvs(m, attributes);
            }
            mask_string(&mut Masker::new(engine), schema_url);
            for span in spans {
                match catch_unwind(AssertUnwindSafe(|| mask_otlp_span(engine, span))) {
                    Err(_) => scrub_otlp_span_fail_closed(span),
                    Ok(Some(report)) => push_otlp_report(&mut span.attributes, report),
                    Ok(None) => {}
                }
            }
        }
    }
}

/// `Some(report)` when anything in the span was replaced.
fn mask_otlp_span(engine: &PiiEngine, span: &mut otlp_pb::trace::Span) -> Option<Report> {
    let m = &mut Masker::new(engine);
    #[cfg(test)]
    if span.name == TEST_PANIC_SPAN_NAME {
        panic!("injected test panic");
    }
    let otlp_pb::trace::Span {
        trace_id: _,
        span_id: _,
        trace_state,
        parent_span_id: _,
        flags: _,
        name,
        kind: _,
        start_time_unix_nano: _,
        end_time_unix_nano: _,
        attributes,
        dropped_attributes_count: _,
        events,
        dropped_events_count: _,
        links,
        dropped_links_count: _,
        status,
    } = span;
    let mut hit = false;
    hit |= mask_string(m, trace_state);
    hit |= mask_string(m, name);
    hit |= mask_otlp_kvs(m, attributes);
    for ev in events {
        let otlp_pb::trace::span::Event {
            time_unix_nano: _,
            name,
            attributes,
            dropped_attributes_count: _,
        } = ev;
        hit |= mask_string(m, name);
        hit |= mask_otlp_kvs(m, attributes);
    }
    for link in links {
        let otlp_pb::trace::span::Link {
            trace_id: _,
            span_id: _,
            trace_state,
            attributes,
            dropped_attributes_count: _,
            flags: _,
        } = link;
        hit |= mask_string(m, trace_state);
        hit |= mask_otlp_kvs(m, attributes);
    }
    if let Some(otlp_pb::trace::Status { message, code: _ }) = status {
        hit |= mask_string(m, message);
    }
    hit.then(|| std::mem::replace(m, Masker::new(engine)).finish())
}

/// `wardex.redacted` plus what was replaced, merged into whatever the mapping
/// already carried from the envelope's own report — one set of keys per span,
/// however many passes wrote to it.
fn push_otlp_report(attrs: &mut Vec<otlp_pb::common::KeyValue>, report: Report) {
    use otlp_pb::common::any_value::Value;
    let take = |attrs: &mut Vec<otlp_pb::common::KeyValue>, key: &str| {
        attrs
            .iter()
            .position(|kv| kv.key == key)
            .and_then(|i| attrs.remove(i).value)
            .and_then(|v| v.value)
    };
    let mut count = i64::from(report.count);
    if let Some(Value::IntValue(prior)) = take(attrs, "wardex.redaction.count") {
        count = count.saturating_add(prior);
    }
    let strings = |v: Option<Value>| -> Vec<String> {
        match v {
            Some(Value::ArrayValue(a)) => a
                .values
                .into_iter()
                .filter_map(|x| match x.value {
                    Some(Value::StringValue(s)) => Some(s),
                    _ => None,
                })
                .collect(),
            _ => Vec::new(),
        }
    };
    let mut rules = strings(take(attrs, "wardex.redaction.rules"));
    for r in &report.rules {
        let name = wardex_codec::vocab::redaction_rule_name(*r as i32);
        if !rules.contains(&name) {
            rules.push(name);
        }
    }
    let mut names = strings(take(attrs, "wardex.redaction.names"));
    for n in report.names {
        if names.len() < super::MAX_REPORTED_NAMES && !names.contains(&n) {
            names.push(n);
        }
    }
    if !attrs.iter().any(|kv| kv.key == "wardex.redacted") {
        attrs.push(otlp_kv_bool("wardex.redacted", true));
    }
    attrs.push(otlp_pb::common::KeyValue {
        key: "wardex.redaction.count".into(),
        value: Some(otlp_pb::common::AnyValue {
            value: Some(Value::IntValue(count)),
        }),
    });
    for (key, list) in [
        ("wardex.redaction.rules", rules),
        ("wardex.redaction.names", names),
    ] {
        if list.is_empty() {
            continue;
        }
        attrs.push(otlp_pb::common::KeyValue {
            key: key.into(),
            value: Some(otlp_pb::common::AnyValue {
                value: Some(Value::ArrayValue(otlp_pb::common::ArrayValue {
                    values: list
                        .into_iter()
                        .map(|s| otlp_pb::common::AnyValue {
                            value: Some(Value::StringValue(s)),
                        })
                        .collect(),
                })),
            }),
        });
    }
}

/// The OTLP twin of `mask_kvs`: keys judged, never rewritten.
fn mask_otlp_kvs(m: &mut Masker<'_>, kvs: &mut [otlp_pb::common::KeyValue]) -> bool {
    let mut hit = false;
    for kv in kvs.iter_mut() {
        let otlp_pb::common::KeyValue { key, value } = kv;
        if let Some(v) = value {
            hit |= match m.key_rule(key) {
                Some(rule) => replace_otlp_any(m, v, rule, key),
                None => mask_otlp_any(m, v),
            };
        }
    }
    hit
}

/// The OTLP twin of `replace_any`.
fn replace_otlp_any(
    m: &mut Masker<'_>,
    v: &mut otlp_pb::common::AnyValue,
    rule: Rule,
    key: &str,
) -> bool {
    use otlp_pb::common::any_value::Value;
    let otlp_pb::common::AnyValue { value } = v;
    let replaced = match value {
        Some(Value::StringValue(s)) if !s.is_empty() && s != PLACEHOLDER => {
            *s = PLACEHOLDER.into();
            true
        }
        Some(Value::BytesValue(b)) if !b.is_empty() && b != PLACEHOLDER.as_bytes() => {
            *b = PLACEHOLDER.as_bytes().to_vec();
            true
        }
        Some(Value::IntValue(_)) | Some(Value::DoubleValue(_)) => {
            *value = Some(Value::StringValue(PLACEHOLDER.into()));
            true
        }
        Some(Value::ArrayValue(arr)) => {
            let otlp_pb::common::ArrayValue { values } = arr;
            let mut hit = false;
            for item in values {
                hit |= replace_otlp_any(m, item, rule, key);
            }
            return hit;
        }
        Some(Value::KvlistValue(kvl)) => {
            let otlp_pb::common::KeyValueList { values } = kvl;
            return mask_otlp_kvs(m, values);
        }
        _ => false,
    };
    if replaced {
        m.report.record(rule, Some(key));
    }
    replaced
}

fn mask_otlp_any(m: &mut Masker<'_>, v: &mut otlp_pb::common::AnyValue) -> bool {
    use otlp_pb::common::any_value::Value;
    let otlp_pb::common::AnyValue { value } = v;
    match value {
        Some(Value::StringValue(s)) => mask_string(m, s),
        Some(Value::BytesValue(b)) => mask_bytes(m, b),
        Some(Value::ArrayValue(arr)) => {
            let otlp_pb::common::ArrayValue { values } = arr;
            let mut hit = false;
            for item in values {
                hit |= mask_otlp_any(m, item);
            }
            hit
        }
        Some(Value::KvlistValue(kvl)) => {
            let otlp_pb::common::KeyValueList { values } = kvl;
            mask_otlp_kvs(m, values)
        }
        Some(Value::BoolValue(_))
        | Some(Value::IntValue(_))
        | Some(Value::DoubleValue(_))
        | None => false,
    }
}

fn otlp_kv_bool(key: &str, v: bool) -> otlp_pb::common::KeyValue {
    otlp_pb::common::KeyValue {
        key: key.into(),
        value: Some(otlp_pb::common::AnyValue {
            value: Some(otlp_pb::common::any_value::Value::BoolValue(v)),
        }),
    }
}

fn scrub_otlp_span_fail_closed(span: &mut otlp_pb::trace::Span) {
    *span = otlp_pb::trace::Span {
        trace_id: span.trace_id.clone(),
        span_id: span.span_id.clone(),
        parent_span_id: span.parent_span_id.clone(),
        kind: span.kind,
        start_time_unix_nano: span.start_time_unix_nano,
        end_time_unix_nano: span.end_time_unix_nano,
        name: PII_FILTER_ERROR.into(),
        attributes: vec![otlp_kv_bool("wardex.redacted", true)],
        ..Default::default()
    };
}

// --- fail-closed scrubs (§8): identity/timing survive, no text does ---

fn scrub_span_fail_closed(span: &mut pb::Span) {
    *span = pb::Span {
        trace_id: span.trace_id.clone(),
        span_id: span.span_id.clone(),
        parent_span_id: span.parent_span_id.clone(),
        kind: span.kind,
        start_time_unix_nano: span.start_time_unix_nano,
        end_time_unix_nano: span.end_time_unix_nano,
        name: PII_FILTER_ERROR.into(),
        capture_integrity: Some(pb::CaptureIntegrity {
            redacted: true,
            ..Default::default()
        }),
        ..Default::default()
    };
}

fn scrub_snapshot_fail_closed(snap: &mut pb::StateSnapshot) {
    *snap = pb::StateSnapshot {
        trace_id: snap.trace_id.clone(),
        span_id: snap.span_id.clone(),
        timestamp_ns: snap.timestamp_ns,
        snapshot_type: snap.snapshot_type,
        turn_index: snap.turn_index,
        conversation_state: PII_FILTER_ERROR.as_bytes().to_vec(),
        ..Default::default()
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pii::{PiiEngine, Rule};

    fn engine() -> PiiEngine {
        PiiEngine::new(&[]).unwrap()
    }

    fn pii_span() -> pb::Span {
        pb::Span {
            name: "call john.doe@acme.com".into(),
            input_data: b"card 4111-1111-1111-1111".to_vec(),
            output_data: vec![0xff, 0xfe, 0x00], // not UTF-8 -> must pass through
            status: Some(pb::Status {
                code: 2,
                message: "failed for jane@x.io".into(),
            }),
            transport: Some(pb::TransportAttributes {
                http: Some(pb::HttpMeta {
                    method: "POST".into(),
                    status_code: 200,
                    url: "https://api.x.com?key=sk-abcdefghijklmnop1234".into(),
                }),
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn env_with(span: pb::Span) -> pb::Envelope {
        pb::Envelope {
            header: Some(pb::EnvelopeHeader {
                event_id: "evt".into(),
                // Secret-shaped on purpose: the whitelist must hold even for a
                // value the secret pattern would otherwise catch.
                project_id: "sk-live-aaaaaaaaaaaaaaaa1234".into(),
                ..Default::default()
            }),
            items: vec![pb::EnvelopeItem {
                header: Some(pb::EnvelopeItemHeader {
                    r#type: "span for john.doe@acme.com".into(),
                    length: 0,
                }),
                payload: Some(pb::envelope_item::Payload::Span(span)),
            }],
        }
    }

    fn span_of(env: &pb::Envelope) -> &pb::Span {
        match &env.items[0].payload {
            Some(pb::envelope_item::Payload::Span(s)) => s,
            _ => panic!("expected span"),
        }
    }

    #[test]
    fn masks_every_text_surface_of_a_span() {
        let mut env = env_with(pii_span());
        mask_envelope(&engine(), &mut env);
        let span = span_of(&env);
        assert_eq!(span.name, "call [EMAIL]");
        assert_eq!(span.input_data, b"card ****-****-****-1111".to_vec());
        assert_eq!(span.output_data, vec![0xff, 0xfe, 0x00]); // binary untouched
        assert_eq!(span.status.as_ref().unwrap().message, "failed for [EMAIL]");
        let url = &span.transport.as_ref().unwrap().http.as_ref().unwrap().url;
        assert_eq!(url, "https://api.x.com?key=[SECRET]");
        assert_eq!(
            env.items[0].header.as_ref().unwrap().r#type,
            "span for [EMAIL]"
        );
    }

    #[test]
    fn project_id_is_whitelisted() {
        let mut env = env_with(pii_span());
        mask_envelope(&engine(), &mut env);
        assert_eq!(
            env.header.as_ref().unwrap().project_id,
            "sk-live-aaaaaaaaaaaaaaaa1234"
        );
    }

    #[test]
    fn redacted_flag_set_only_when_replacements_happened() {
        let mut dirty = env_with(pii_span());
        mask_envelope(&engine(), &mut dirty);
        assert!(span_of(&dirty).capture_integrity.as_ref().unwrap().redacted);

        let mut clean = env_with(pb::Span {
            name: "plain".into(),
            ..Default::default()
        });
        mask_envelope(&engine(), &mut clean);
        let ci = &span_of(&clean).capture_integrity;
        assert!(ci.is_none() || !ci.as_ref().unwrap().redacted);
    }

    #[test]
    fn snapshot_payloads_are_masked() {
        let mut env = pb::Envelope {
            header: None,
            items: vec![pb::EnvelopeItem {
                header: None,
                payload: Some(pb::envelope_item::Payload::StateSnapshot(
                    pb::StateSnapshot {
                        conversation_state: b"user john@x.io asked".to_vec(),
                        tool_definitions: Some(pb::ToolDefinitionSet {
                            tools: vec![pb::ToolDefinition {
                                name: "lookup".into(),
                                description: "reach admin@corp.com".into(),
                                ..Default::default()
                            }],
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                )),
            }],
        };
        mask_envelope(&engine(), &mut env);
        match &env.items[0].payload {
            Some(pb::envelope_item::Payload::StateSnapshot(s)) => {
                assert_eq!(s.conversation_state, b"user [EMAIL] asked".to_vec());
                assert_eq!(
                    s.tool_definitions.as_ref().unwrap().tools[0].description,
                    "reach [EMAIL]"
                );
            }
            _ => panic!("expected snapshot"),
        }
    }

    #[test]
    fn panicking_span_is_scrubbed_fail_closed() {
        let mut span = pii_span();
        span.name = TEST_PANIC_SPAN_NAME.into(); // test-only injection point
        let mut env = env_with(span);
        mask_envelope(&engine(), &mut env);
        let s = span_of(&env);
        assert_eq!(s.name, crate::pii::PII_FILTER_ERROR);
        assert!(s.input_data.is_empty());
        assert!(s.status.is_none());
        assert!(s.capture_integrity.as_ref().unwrap().redacted);
    }

    // --- OTLP tests ---

    use wardex_codec::otlp::otlp_pb;

    #[test]
    fn a_masked_span_reports_count_rules_and_names() {
        let mut env = env_with(pii_span());
        mask_envelope(&engine(), &mut env);
        let ci = span_of(&env).capture_integrity.clone().unwrap();
        assert!(ci.redacted);
        // name + status (two e-mails), the card, the `sk-` key in the URL.
        assert_eq!(ci.redaction_count, 4);
        let rules: Vec<Rule> = ci
            .redaction_rules
            .iter()
            .map(|n| Rule::try_from(*n).unwrap())
            .collect();
        assert_eq!(
            rules,
            vec![Rule::Email, Rule::CreditCard, Rule::SecretValue]
        );
        // `?key=` is a name-rule hit too, but the `sk-` value hit starts at the
        // same byte and wins the tie, so the rule that is reported is the
        // value's and no name is recorded.
        assert!(ci.redaction_names.is_empty(), "{:?}", ci.redaction_names);
    }

    #[test]
    fn a_clean_span_reports_nothing() {
        let mut env = env_with(pb::Span {
            name: "HTTP GET /v1/models".into(),
            ..Default::default()
        });
        mask_envelope(&engine(), &mut env);
        assert!(span_of(&env).capture_integrity.is_none());
    }

    #[test]
    fn one_byte_that_is_not_utf8_no_longer_switches_a_body_off() {
        let mut span = pb::Span {
            name: "HTTP POST /login".into(),
            input_data: b"password=LATIN1&name=caf\xe9".to_vec(),
            ..Default::default()
        };
        let mut env = env_with(std::mem::take(&mut span));
        mask_envelope(&engine(), &mut env);
        assert_eq!(
            span_of(&env).input_data,
            b"password=[SECRET]&name=caf\xe9".to_vec()
        );
    }

    fn text_kv(key: &str, v: &str) -> pb::KeyValue {
        pb::KeyValue {
            key: key.into(),
            value: Some(pb::AnyValue {
                value: Some(pb::any_value::Value::StringValue(v.into())),
            }),
        }
    }

    fn int_kv(key: &str, v: i64) -> pb::KeyValue {
        pb::KeyValue {
            key: key.into(),
            value: Some(pb::AnyValue {
                value: Some(pb::any_value::Value::IntValue(v)),
            }),
        }
    }

    #[test]
    fn a_host_attribute_under_a_secret_name_is_replaced_whole() {
        let mut span = pb::Span {
            name: "tool".into(),
            extra: vec![
                text_kv("db.password", "hunter2"),
                int_kv("pin_token", 1234),
                text_kv("session_id", "conv-1"),
                text_kv("code", "E42"),
                pb::KeyValue {
                    key: "request".into(),
                    value: Some(pb::AnyValue {
                        value: Some(pb::any_value::Value::KvlistValue(pb::KeyValueList {
                            values: vec![text_kv("api_key", "abc"), text_kv("q", "seoul")],
                        })),
                    }),
                },
            ],
            ..Default::default()
        };
        let mut env = env_with(std::mem::take(&mut span));
        mask_envelope(&engine(), &mut env);
        let sp = span_of(&env);
        let get = |k: &str| {
            sp.extra
                .iter()
                .find(|kv| kv.key == k)
                .and_then(|kv| kv.value.clone())
                .and_then(|v| v.value)
        };
        use pb::any_value::Value;
        assert_eq!(
            get("db.password"),
            Some(Value::StringValue("[SECRET]".into()))
        );
        assert_eq!(
            get("pin_token"),
            Some(Value::StringValue("[SECRET]".into()))
        );
        // Not secret names: an agent conversation id, and a JSON-shaped `code`.
        assert_eq!(get("session_id"), Some(Value::StringValue("conv-1".into())));
        assert_eq!(get("code"), Some(Value::StringValue("E42".into())));
        match get("request") {
            Some(Value::KvlistValue(kvl)) => {
                assert_eq!(
                    kvl.values[0].value.clone().unwrap().value,
                    Some(Value::StringValue("[SECRET]".into()))
                );
                assert_eq!(
                    kvl.values[1].value.clone().unwrap().value,
                    Some(Value::StringValue("seoul".into()))
                );
            }
            other => panic!("expected a kvlist, got {other:?}"),
        }
        let ci = sp.capture_integrity.clone().unwrap();
        assert_eq!(ci.redaction_count, 3);
        assert_eq!(
            ci.redaction_names,
            vec!["db.password", "pin_token", "api_key"]
        );
    }

    #[test]
    fn otlp_reports_merge_into_one_set_of_keys() {
        let mut req = otlp_req_with_pii();
        mask_otlp(&engine(), &mut req);
        let span = &req.resource_spans[0].scope_spans[0].spans[0];
        let count = |k: &str| span.attributes.iter().filter(|kv| kv.key == k).count();
        assert_eq!(count("wardex.redacted"), 1);
        assert_eq!(count("wardex.redaction.count"), 1);
        assert_eq!(count("wardex.redaction.rules"), 1);
        match otlp_attr(span, "wardex.redaction.count").unwrap() {
            otlp_pb::common::any_value::Value::IntValue(n) => assert_eq!(*n, 3),
            other => panic!("{other:?}"),
        }
        // Masking the already-masked request again replaces nothing and
        // leaves the report as it was.
        let before = req.clone();
        mask_otlp(&engine(), &mut req);
        assert_eq!(req, before);
    }

    fn otlp_kv_str(key: &str, v: &str) -> otlp_pb::common::KeyValue {
        otlp_pb::common::KeyValue {
            key: key.into(),
            value: Some(otlp_pb::common::AnyValue {
                value: Some(otlp_pb::common::any_value::Value::StringValue(v.into())),
            }),
        }
    }

    fn otlp_req_with_pii() -> otlp_pb::trace_service::ExportTraceServiceRequest {
        otlp_pb::trace_service::ExportTraceServiceRequest {
            resource_spans: vec![otlp_pb::trace::ResourceSpans {
                resource: None,
                scope_spans: vec![otlp_pb::trace::ScopeSpans {
                    scope: None,
                    spans: vec![otlp_pb::trace::Span {
                        name: "chat for john@x.io".into(),
                        attributes: vec![
                            otlp_kv_str("gen_ai.output.messages", "reply to jane@y.io"),
                            otlp_pb::common::KeyValue {
                                key: "wardex.input_data".into(),
                                value: Some(otlp_pb::common::AnyValue {
                                    value: Some(otlp_pb::common::any_value::Value::BytesValue(
                                        b"card 4111-1111-1111-1111".to_vec(),
                                    )),
                                }),
                            },
                        ],
                        ..Default::default()
                    }],
                    ..Default::default()
                }],
                ..Default::default()
            }],
        }
    }

    fn otlp_attr<'a>(
        span: &'a otlp_pb::trace::Span,
        key: &str,
    ) -> Option<&'a otlp_pb::common::any_value::Value> {
        span.attributes
            .iter()
            .find(|kv| kv.key == key)?
            .value
            .as_ref()?
            .value
            .as_ref()
    }

    #[test]
    fn otlp_spans_are_masked_and_flagged() {
        let mut req = otlp_req_with_pii();
        mask_otlp(&engine(), &mut req);
        let span = &req.resource_spans[0].scope_spans[0].spans[0];
        assert_eq!(span.name, "chat for [EMAIL]");
        match otlp_attr(span, "gen_ai.output.messages").unwrap() {
            otlp_pb::common::any_value::Value::StringValue(s) => {
                assert_eq!(s, "reply to [EMAIL]");
            }
            other => panic!("unexpected value: {other:?}"),
        }
        match otlp_attr(span, "wardex.input_data").unwrap() {
            otlp_pb::common::any_value::Value::BytesValue(b) => {
                assert_eq!(b, &b"card ****-****-****-1111".to_vec());
            }
            other => panic!("unexpected value: {other:?}"),
        }
        match otlp_attr(span, "wardex.redacted").unwrap() {
            otlp_pb::common::any_value::Value::BoolValue(v) => assert!(*v),
            other => panic!("unexpected value: {other:?}"),
        }
    }

    #[test]
    fn otlp_panicking_span_is_scrubbed_fail_closed() {
        let mut req = otlp_req_with_pii();
        let span = &mut req.resource_spans[0].scope_spans[0].spans[0];
        span.name = TEST_PANIC_SPAN_NAME.into();
        span.trace_id = vec![0xAB; 16];
        span.span_id = vec![0xCD; 8];
        span.start_time_unix_nano = 1000;
        span.end_time_unix_nano = 2000;
        mask_otlp(&engine(), &mut req);
        let span = &req.resource_spans[0].scope_spans[0].spans[0];
        assert_eq!(span.name, crate::pii::PII_FILTER_ERROR);
        assert_eq!(span.trace_id, vec![0xAB; 16]);
        assert_eq!(span.span_id, vec![0xCD; 8]);
        assert_eq!(span.start_time_unix_nano, 1000);
        assert_eq!(span.end_time_unix_nano, 2000);
        assert_eq!(
            span.attributes,
            vec![otlp_kv_bool("wardex.redacted", true)],
            "attributes reduced to wardex.redacted=true only"
        );
    }

    #[test]
    fn otlp_clean_span_gets_no_redacted_attr() {
        let mut req = otlp_req_with_pii();
        req.resource_spans[0].scope_spans[0].spans[0] = otlp_pb::trace::Span {
            name: "plain".into(),
            ..Default::default()
        };
        mask_otlp(&engine(), &mut req);
        let span = &req.resource_spans[0].scope_spans[0].spans[0];
        assert!(otlp_attr(span, "wardex.redacted").is_none());
    }
}

#[cfg(test)]
mod usage_pins {
    //! Structural pins for the usage attributes: numeric usage values are
    //! untouchable BY TYPE (the `mask_any`/`mask_otlp_any` match arms answer
    //! `false` for Int/Double/Bool before any engine runs), and model-id
    //! strings pass the pattern table unchanged. Pinned here because the
    //! `wardex.usage.*` mirror multiplies the numeric attributes per span,
    //! and a masking regression on them would corrupt billing math silently.
    use super::*;
    use crate::pii::PiiEngine;

    fn kv_int(key: &str, v: i64) -> pb::KeyValue {
        pb::KeyValue {
            key: key.into(),
            value: Some(pb::AnyValue {
                value: Some(pb::any_value::Value::IntValue(v)),
            }),
        }
    }

    fn kv_double(key: &str, v: f64) -> pb::KeyValue {
        pb::KeyValue {
            key: key.into(),
            value: Some(pb::AnyValue {
                value: Some(pb::any_value::Value::DoubleValue(v)),
            }),
        }
    }

    fn kv_str(key: &str, v: &str) -> pb::KeyValue {
        pb::KeyValue {
            key: key.into(),
            value: Some(pb::AnyValue {
                value: Some(pb::any_value::Value::StringValue(v.into())),
            }),
        }
    }

    /// T-R11 — integer/double usage attributes are byte-identical after a
    /// full masking walk, for both key families, structurally (no exemption
    /// list involved — there is none to get wrong).
    #[test]
    fn integer_usage_attributes_are_untouched_by_masking() {
        let engine = PiiEngine::new(&[]).unwrap();
        let mut span = pb::Span {
            extra: vec![
                kv_int("gen_ai.usage.input_tokens", 4111111111111111),
                kv_int(
                    "wardex.usage.cache_creation.ephemeral_1h_input_tokens",
                    1234,
                ),
                kv_double("wardex.usage.score", 4111.1111),
            ],
            ..Default::default()
        };
        let before = span.clone();
        let mut env = pb::Envelope {
            items: vec![pb::EnvelopeItem {
                header: None,
                payload: Some(pb::envelope_item::Payload::Span(span)),
            }],
            ..Default::default()
        };
        mask_envelope(&engine, &mut env);
        span = match env.items.remove(0).payload {
            Some(pb::envelope_item::Payload::Span(s)) => s,
            other => panic!("expected span, got {other:?}"),
        };
        assert_eq!(span.extra, before.extra);
    }

    /// T-R11 — model ids and plain tier strings survive the shipped pattern
    /// table. Deliberately NOT a key-based exemption: these go through the
    /// engine like every other string and come out unchanged because no
    /// pattern matches them.
    #[test]
    fn model_strings_that_look_like_model_ids_are_not_masked() {
        let engine = PiiEngine::new(&[]).unwrap();
        for value in [
            "gpt-4o-2024-08-06",
            "claude-opus-4-8",
            "text-embedding-3-small",
            "default",
            "standard",
            "flex",
        ] {
            let mut env = pb::Envelope {
                items: vec![pb::EnvelopeItem {
                    header: None,
                    payload: Some(pb::envelope_item::Payload::Span(pb::Span {
                        extra: vec![
                            kv_str("gen_ai.response.model", value),
                            kv_str("wardex.usage.service_tier", value),
                        ],
                        ..Default::default()
                    })),
                }],
                ..Default::default()
            };
            mask_envelope(&engine, &mut env);
            let span = match &env.items[0].payload {
                Some(pb::envelope_item::Payload::Span(s)) => s,
                other => panic!("expected span, got {other:?}"),
            };
            for kv in &span.extra {
                let got = match kv.value.as_ref().unwrap().value.as_ref().unwrap() {
                    pb::any_value::Value::StringValue(s) => s.as_str(),
                    other => panic!("unexpected value {other:?}"),
                };
                assert_eq!(got, value, "{} was rewritten", kv.key);
            }
        }
    }
}
