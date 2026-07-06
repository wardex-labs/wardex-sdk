//! Proto-tree masking walks — design §4.
//!
//! Every message is destructured WITHOUT `..` so that adding a proto field
//! breaks this build until someone decides whether the new field is masked
//! (design §4.3). Non-text fields are bound to `_`.

use std::panic::{catch_unwind, AssertUnwindSafe};

use wardex_codec::proto::wardex::v1 as pb;

use super::{PiiEngine, PII_FILTER_ERROR};

/// Test-only panic injection for the fail-closed path.
#[cfg(test)]
pub(super) const TEST_PANIC_SPAN_NAME: &str = "__wardex_test_panic__";

/// Mask every outbound text surface of a wardex envelope (design §3).
/// A span/snapshot that panics during masking is scrubbed whole (fail-closed §8);
/// other items keep processing.
pub fn mask_envelope(engine: &PiiEngine, env: &mut pb::Envelope) {
    let pb::Envelope { header, items } = env;
    if let Some(h) = header {
        mask_header(engine, h);
    }
    for item in items {
        let pb::EnvelopeItem { header, payload } = item;
        if let Some(pb::EnvelopeItemHeader { r#type, length: _ }) = header {
            // Item-level metadata, masked at the same tier as the envelope
            // header — it does not feed any per-span redacted flag.
            mask_string(engine, r#type);
        }
        match payload {
            Some(pb::envelope_item::Payload::Span(span)) => {
                let masked = catch_unwind(AssertUnwindSafe(|| mask_span(engine, span)));
                if masked.is_err() {
                    scrub_span_fail_closed(span);
                }
            }
            Some(pb::envelope_item::Payload::StateSnapshot(snap)) => {
                let masked = catch_unwind(AssertUnwindSafe(|| mask_snapshot(engine, snap)));
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

fn mask_string(engine: &PiiEngine, s: &mut String) -> bool {
    match engine.mask_text(s) {
        Some(masked) => {
            *s = masked;
            true
        }
        None => false,
    }
}

/// bytes are masked only when they decode as UTF-8; raw binary passes through
/// (documented limitation, design §4.4).
fn mask_bytes(engine: &PiiEngine, b: &mut Vec<u8>) -> bool {
    let Ok(text) = std::str::from_utf8(b) else {
        return false;
    };
    match engine.mask_text(text) {
        Some(masked) => {
            *b = masked.into_bytes();
            true
        }
        None => false,
    }
}

fn mask_kvs(engine: &PiiEngine, kvs: &mut [pb::KeyValue]) -> bool {
    let mut hit = false;
    for kv in kvs.iter_mut() {
        let pb::KeyValue { key: _, value } = kv; // keys are our own attribute names
        if let Some(v) = value {
            hit |= mask_any(engine, v);
        }
    }
    hit
}

fn mask_any(engine: &PiiEngine, v: &mut pb::AnyValue) -> bool {
    use pb::any_value::Value;
    let pb::AnyValue { value } = v;
    match value {
        Some(Value::StringValue(s)) => mask_string(engine, s),
        Some(Value::BytesValue(b)) => mask_bytes(engine, b),
        Some(Value::ArrayValue(arr)) => {
            let pb::ArrayValue { values } = arr;
            let mut hit = false;
            for item in values {
                hit |= mask_any(engine, item);
            }
            hit
        }
        Some(Value::KvlistValue(kvl)) => {
            let pb::KeyValueList { values } = kvl;
            mask_kvs(engine, values)
        }
        Some(Value::BoolValue(_))
        | Some(Value::IntValue(_))
        | Some(Value::DoubleValue(_))
        | None => false,
    }
}

// --- message walks ---

fn mask_header(engine: &PiiEngine, h: &mut pb::EnvelopeHeader) {
    let pb::EnvelopeHeader {
        event_id,
        api_key: _, // WHITELIST: our own backend credential — masking breaks auth (§4.4)
        sdk,
        sent_at_unix_nano: _,
        session_status: _,
        retention_class: _,
    } = h;
    mask_string(engine, event_id);
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
        mask_string(engine, name);
        mask_string(engine, version);
        mask_string(engine, python_version);
        mask_string(engine, os);
        mask_string(engine, arch);
        for a in adapters {
            mask_string(engine, a);
        }
        for i in interceptors {
            mask_string(engine, i);
        }
        mask_string(engine, otel_semconv_version);
        mask_string(engine, shell);
    }
}

fn mask_span(engine: &PiiEngine, span: &mut pb::Span) {
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
    hit |= mask_string(engine, name);
    if let Some(pb::Status { code: _, message }) = status {
        hit |= mask_string(engine, message);
    }
    hit |= mask_kvs(engine, extra);
    for ev in events {
        let pb::SpanEvent {
            name,
            time_unix_nano: _,
            attributes,
        } = ev;
        hit |= mask_string(engine, name);
        hit |= mask_kvs(engine, attributes);
    }
    for link in links {
        let pb::SpanLink {
            trace_id: _,
            span_id: _,
            attributes,
        } = link;
        hit |= mask_kvs(engine, attributes);
    }
    hit |= mask_bytes(engine, input_data);
    hit |= mask_bytes(engine, output_data);
    if let Some(t) = transport {
        hit |= mask_transport(engine, t);
    }
    hit |= mask_string(engine, error_type);
    hit |= mask_string(engine, server_address);
    hit |= mask_string(engine, workflow_name);
    if let Some(pb::CallSite {
        file,
        line: _,
        function,
        module,
    }) = call_site
    {
        hit |= mask_string(engine, file);
        hit |= mask_string(engine, function);
        hit |= mask_string(engine, module);
    }
    if let Some(pb::ConversationContext {
        conversation_id,
        session_id,
        turn_index: _,
    }) = conversation
    {
        hit |= mask_string(engine, conversation_id);
        hit |= mask_string(engine, session_id);
    }
    if let Some(pb::CorrelationInfo {
        operation_id,
        request_id,
        attempt_id,
        active_span_id_at_capture: _, // binary id
        confidence: _,
        strategy,
    }) = correlation
    {
        hit |= mask_string(engine, operation_id);
        hit |= mask_string(engine, request_id);
        hit |= mask_string(engine, attempt_id);
        hit |= mask_string(engine, strategy);
    }
    if let Some(pb::CaptureIntegrity {
        request_headers_captured: _,
        request_body_captured: _,
        response_headers_captured: _,
        response_body_captured: _,
        redacted: _, // set below from `hit`
        truncated: _,
        dropped_chunk_count: _,
        limitations,
    }) = capture_integrity
    {
        for l in limitations {
            hit |= mask_string(engine, l);
        }
    }
    if hit {
        capture_integrity
            .get_or_insert_with(Default::default)
            .redacted = true;
    }
}

fn mask_transport(engine: &PiiEngine, t: &mut pb::TransportAttributes) -> bool {
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
    hit |= mask_string(engine, connection_id);
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
        hit |= mask_string(engine, method);
        hit |= mask_string(engine, url);
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
        hit |= mask_string(engine, service);
        hit |= mask_string(engine, method);
        hit |= mask_string(engine, encoding);
        hit |= mask_string(engine, decoded_payload);
    }
    if let Some(pb::WebSocketMeta {
        opcode: _,
        direction,
    }) = websocket
    {
        hit |= mask_string(engine, direction);
    }
    if let Some(pb::McpMeta { rpc_method, rpc_id }) = mcp {
        hit |= mask_string(engine, rpc_method);
        hit |= mask_string(engine, rpc_id);
    }
    if let Some(pb::SseMeta { event_type }) = sse {
        hit |= mask_string(engine, event_type);
    }
    if let Some(pb::A2aMeta { task_id, transport }) = a2a {
        hit |= mask_string(engine, task_id);
        hit |= mask_string(engine, transport);
    }
    hit |= mask_string(engine, request_blob_ref);
    hit |= mask_string(engine, response_blob_ref);
    hit
}

fn mask_snapshot(engine: &PiiEngine, snap: &mut pb::StateSnapshot) {
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
    mask_kvs(engine, attributes);
    mask_bytes(engine, conversation_state);
    for r in input_refs {
        let pb::InputRef {
            key,
            content_hash,
            blob_ref,
        } = r;
        mask_string(engine, key);
        mask_string(engine, content_hash);
        mask_string(engine, blob_ref);
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
            mask_string(engine, name);
            mask_string(engine, description);
            mask_bytes(engine, parameters_schema);
            mask_string(engine, version);
            mask_string(engine, r#type);
            mask_string(engine, hash);
        }
        mask_string(engine, set_hash);
    }
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
    use crate::pii::PiiEngine;

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
                api_key: "sk-live-aaaaaaaaaaaaaaaa1234".into(), // matches the secret pattern on purpose
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
    fn api_key_is_whitelisted() {
        let mut env = env_with(pii_span());
        mask_envelope(&engine(), &mut env);
        assert_eq!(
            env.header.as_ref().unwrap().api_key,
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
}
