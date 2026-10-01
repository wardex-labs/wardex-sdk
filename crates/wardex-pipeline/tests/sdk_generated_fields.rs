//! The census of text fields the SDK fills by itself.
//!
//! Masking exists for what the host application and its traffic put on a
//! span. A value the SDK minted is neither, and the rules cannot tell the
//! two apart: `str(id(socket))` is fifteen digits on 64-bit Linux, about
//! one in ten of those passes the card checksum, and the card rule then
//! rewrote the id and wrote `credit_card` into the span's record of what
//! it had masked. These tests pin which fields are the SDK's own, prove a
//! value every rule fires on passes through each of them untouched and
//! unrecorded, and prove every other text field still loses it.
use std::collections::{BTreeMap, BTreeSet};

use wardex_codec::otlp::map::{envelope_to_traces, Producer};
use wardex_codec::otlp::otlp_pb;
use wardex_codec::proto::wardex::v1 as pb;
use wardex_pipeline::pii::{mask_envelope, mask_otlp, PiiEngine, Rule};

/// Envelope fields only the SDK writes, each beside the code that fills
/// it. A field with no producer yet (`SdkInfo.adapters`, `.interceptors`,
/// `.shell`, the blob refs, `WebSocketMeta.direction`) is not on this
/// list: until something fills it and is classified, it is masked like
/// host text.
const SDK_GENERATED_ENVELOPE: &[&str] = &[
    // The Python client's drain: `str(uuid.uuid4())`, once per batch.
    "EnvelopeHeader.event_id",
    // The binding's marshaller: the literal "span" or "state_snapshot".
    "EnvelopeItemHeader.type",
    // `build_sdk_info()`: the literal "wardex.python", the package
    // version, `platform.python_version()`, `sys.platform` and
    // `platform.machine()`.
    "SdkInfo.name",
    "SdkInfo.version",
    "SdkInfo.python_version",
    "SdkInfo.os",
    "SdkInfo.arch",
    // The `SdkInfo` default, `OTEL_SEMCONV_VERSION`.
    "SdkInfo.otel_semconv_version",
    // The socket seam: `str(id(socket))`, a process object id.
    "TransportAttributes.connection_id",
];

/// OTLP fields only the mapping writes, from the producer the binding
/// passes and from `SdkInfo.version`. `service.name` is not one: its
/// fallback `unknown_service:<language>` is the mapping's, but the key is
/// the host's, and the one language a binding passes today matches no
/// rule.
const SDK_GENERATED_OTLP: &[&str] = &[
    "InstrumentationScope.name",
    "InstrumentationScope.version",
    "Resource.attributes[telemetry.sdk.name]",
    "Resource.attributes[telemetry.sdk.version]",
    "Resource.attributes[telemetry.sdk.language]",
];

/// Left alone for its own reason: the receiver stamps it from the API key
/// it authenticated, and the SDK sends it empty.
const RECEIVER_STAMPED: &[&str] = &["EnvelopeHeader.project_id"];

/// One fragment per rule, and the part of it that rule must remove.
const FRAGMENTS: &[(Rule, &str, &str)] = &[
    (Rule::Email, "mail john.doe@acme.com", "john.doe@acme.com"),
    (Rule::PhoneNumber, "phone (555) 123-4567", "123-4567"),
    (
        Rule::CreditCard,
        "card 4111-1111-1111-1111",
        "4111-1111-1111-1111",
    ),
    (Rule::UsSsn, "ssn 123-45-6789", "123-45-6789"),
    (Rule::IpAddress, "host 10.0.0.5", "10.0.0.5"),
    (Rule::UsBankRouting, "aba 021000021", "021000021"),
    (
        Rule::Iban,
        "iban GB82WEST12345698765432",
        "GB82WEST12345698765432",
    ),
    (
        Rule::SecretValue,
        "key sk-abcdefghijklmnop1234",
        "sk-abcdefghijklmnop1234",
    ),
    (Rule::SecretWord, "password=hunter2", "hunter2"),
    (Rule::SecretLastWord, "api_key=abc123", "abc123"),
    (Rule::SecretExactName, "appid=owm999", "owm999"),
    (Rule::SecretUserName, "x_corp_widget=w1dget", "w1dget"),
    (
        Rule::UrlUserinfo,
        "https://jdoe:pw0rd@example.com/x",
        "jdoe:pw0rd",
    ),
];

/// Every fragment in one value, so one field holding it is judged by
/// every rule there is.
fn every_rule() -> String {
    FRAGMENTS
        .iter()
        .map(|(_, fragment, _)| *fragment)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Every rule the schema names, in enum order — the order a report
/// lists them in. A rule added later is in here without an edit, and
/// `FRAGMENTS` then fails to reach it until it gains a row.
fn all_rules() -> Vec<Rule> {
    (1..)
        .map_while(|n| Rule::try_from(n).ok())
        .collect::<Vec<_>>()
}

/// The default rules plus one name the application listed, so the
/// application's own rule is exercised too.
fn engine() -> PiiEngine {
    PiiEngine::with_names(&[], &["x_corp_widget".into()], &[]).unwrap()
}

enum Text<'a> {
    Str(&'a mut String),
    Bytes(&'a mut Vec<u8>),
}

type Visit<'f> = dyn FnMut(&'static str, Text<'_>) + 'f;

fn each_any(path: &'static str, v: &mut pb::AnyValue, f: &mut Visit<'_>) {
    use pb::any_value::Value;
    match &mut v.value {
        Some(Value::StringValue(s)) => f(path, Text::Str(s)),
        Some(Value::BytesValue(b)) => f(path, Text::Bytes(b)),
        Some(Value::ArrayValue(a)) => {
            for x in &mut a.values {
                each_any(path, x, f);
            }
        }
        Some(Value::KvlistValue(l)) => each_kv(path, &mut l.values, f),
        Some(Value::BoolValue(_) | Value::IntValue(_) | Value::DoubleValue(_)) | None => {}
    }
}

/// Attribute VALUES. A key is a name: the name rules judge it and the
/// walk never rewrites it, so it is not a field either side of the census.
fn each_kv(path: &'static str, kvs: &mut [pb::KeyValue], f: &mut Visit<'_>) {
    for pb::KeyValue { key: _, value } in kvs {
        if let Some(v) = value {
            each_any(path, v, f);
        }
    }
}

/// Every text field of an envelope, by path. Destructured without `..`,
/// like the walks, so a field added to the schema fails this build too
/// until someone decides which side of the census it is on.
fn each_text(env: &mut pb::Envelope, f: &mut Visit<'_>) {
    let pb::Envelope { header, items } = env;
    if let Some(pb::EnvelopeHeader {
        event_id,
        sdk,
        sent_at_unix_nano: _,
        session_status: _,
        retention_class: _,
        resource,
        project_id,
    }) = header
    {
        f("EnvelopeHeader.event_id", Text::Str(event_id));
        f("EnvelopeHeader.project_id", Text::Str(project_id));
        if let Some(pb::ResourceInfo {
            service_name,
            release,
            environment,
            process_pid: _,
        }) = resource
        {
            f("ResourceInfo.service_name", Text::Str(service_name));
            f("ResourceInfo.release", Text::Str(release));
            f("ResourceInfo.environment", Text::Str(environment));
        }
        if let Some(pb::SdkInfo {
            name,
            version,
            python_version,
            os,
            arch,
            adapters,
            interceptors,
            otel_semconv_version,
            shell,
        }) = sdk
        {
            f("SdkInfo.name", Text::Str(name));
            f("SdkInfo.version", Text::Str(version));
            f("SdkInfo.python_version", Text::Str(python_version));
            f("SdkInfo.os", Text::Str(os));
            f("SdkInfo.arch", Text::Str(arch));
            for a in adapters {
                f("SdkInfo.adapters[]", Text::Str(a));
            }
            for i in interceptors {
                f("SdkInfo.interceptors[]", Text::Str(i));
            }
            f(
                "SdkInfo.otel_semconv_version",
                Text::Str(otel_semconv_version),
            );
            f("SdkInfo.shell", Text::Str(shell));
        }
    }
    for pb::EnvelopeItem { header, payload } in items {
        if let Some(pb::EnvelopeItemHeader { r#type, length: _ }) = header {
            f("EnvelopeItemHeader.type", Text::Str(r#type));
        }
        match payload {
            Some(pb::envelope_item::Payload::Span(span)) => each_span_text(span, f),
            Some(pb::envelope_item::Payload::StateSnapshot(snap)) => each_snapshot_text(snap, f),
            // Map keys are internal event-type tags, values are counters.
            Some(pb::envelope_item::Payload::ClientReport(pb::ClientReport {
                timestamp_ns: _,
                discarded_events: _,
                failed_sends: _,
                queue_depth: _,
                uptime_ms: _,
            }))
            | None => {}
        }
    }
}

fn each_span_text(span: &mut pb::Span, f: &mut Visit<'_>) {
    let pb::Span {
        // Binary ids: never text, never scanned.
        trace_id: _,
        span_id: _,
        parent_span_id: _,
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
        // The masker's own output, written after every field is scanned —
        // not an input it judges.
        capture_integrity: _,
        correlation,
    } = span;
    f("Span.name", Text::Str(name));
    if let Some(pb::Status { code: _, message }) = status {
        f("Status.message", Text::Str(message));
    }
    each_kv("Span.extra[]", extra, f);
    for pb::SpanEvent {
        name,
        time_unix_nano: _,
        attributes,
    } in events
    {
        f("SpanEvent.name", Text::Str(name));
        each_kv("SpanEvent.attributes[]", attributes, f);
    }
    for pb::SpanLink {
        trace_id: _,
        span_id: _,
        attributes,
        reason: _,
    } in links
    {
        each_kv("SpanLink.attributes[]", attributes, f);
    }
    f("Span.input_data", Text::Bytes(input_data));
    f("Span.output_data", Text::Bytes(output_data));
    if let Some(t) = transport {
        each_transport_text(t, f);
    }
    f("Span.error_type", Text::Str(error_type));
    f("Span.server_address", Text::Str(server_address));
    f("Span.workflow_name", Text::Str(workflow_name));
    if let Some(pb::CallSite {
        file,
        line: _,
        function,
        module,
    }) = call_site
    {
        f("CallSite.file", Text::Str(file));
        f("CallSite.function", Text::Str(function));
        f("CallSite.module", Text::Str(module));
    }
    if let Some(pb::ConversationContext {
        conversation_id,
        session_id,
        turn_index: _,
    }) = conversation
    {
        f(
            "ConversationContext.conversation_id",
            Text::Str(conversation_id),
        );
        f("ConversationContext.session_id", Text::Str(session_id));
    }
    if let Some(pb::CorrelationInfo {
        operation_id,
        request_id,
        attempt_id,
        active_span_id_at_capture: _,
        confidence: _,
        parent_source: _,
    }) = correlation
    {
        f("CorrelationInfo.operation_id", Text::Str(operation_id));
        f("CorrelationInfo.request_id", Text::Str(request_id));
        f("CorrelationInfo.attempt_id", Text::Str(attempt_id));
    }
}

fn each_transport_text(t: &mut pb::TransportAttributes, f: &mut Visit<'_>) {
    let pb::TransportAttributes {
        connection_id,
        protocol: _,
        direction: _,
        timing: _,
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
    f(
        "TransportAttributes.connection_id",
        Text::Str(connection_id),
    );
    if let Some(pb::HttpMeta {
        method,
        status_code: _,
        url,
    }) = http
    {
        f("HttpMeta.method", Text::Str(method));
        f("HttpMeta.url", Text::Str(url));
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
        f("GrpcMeta.service", Text::Str(service));
        f("GrpcMeta.method", Text::Str(method));
        f("GrpcMeta.encoding", Text::Str(encoding));
        f("GrpcMeta.decoded_payload", Text::Str(decoded_payload));
    }
    if let Some(pb::WebSocketMeta {
        opcode: _,
        direction,
    }) = websocket
    {
        f("WebSocketMeta.direction", Text::Str(direction));
    }
    if let Some(pb::McpMeta { rpc_method, rpc_id }) = mcp {
        f("McpMeta.rpc_method", Text::Str(rpc_method));
        f("McpMeta.rpc_id", Text::Str(rpc_id));
    }
    if let Some(pb::SseMeta { event_type }) = sse {
        f("SseMeta.event_type", Text::Str(event_type));
    }
    if let Some(pb::A2aMeta { task_id, transport }) = a2a {
        f("A2aMeta.task_id", Text::Str(task_id));
        f("A2aMeta.transport", Text::Str(transport));
    }
    f(
        "TransportAttributes.request_blob_ref",
        Text::Str(request_blob_ref),
    );
    f(
        "TransportAttributes.response_blob_ref",
        Text::Str(response_blob_ref),
    );
}

fn each_snapshot_text(snap: &mut pb::StateSnapshot, f: &mut Visit<'_>) {
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
    each_kv("StateSnapshot.attributes[]", attributes, f);
    f(
        "StateSnapshot.conversation_state",
        Text::Bytes(conversation_state),
    );
    for pb::InputRef {
        key,
        content_hash,
        blob_ref,
    } in input_refs
    {
        f("InputRef.key", Text::Str(key));
        f("InputRef.content_hash", Text::Str(content_hash));
        f("InputRef.blob_ref", Text::Str(blob_ref));
    }
    if let Some(pb::ToolDefinitionSet { tools, set_hash }) = tool_definitions {
        for pb::ToolDefinition {
            name,
            description,
            parameters_schema,
            version,
            r#type,
            hash,
        } in tools
        {
            f("ToolDefinition.name", Text::Str(name));
            f("ToolDefinition.description", Text::Str(description));
            f(
                "ToolDefinition.parameters_schema",
                Text::Bytes(parameters_schema),
            );
            f("ToolDefinition.version", Text::Str(version));
            f("ToolDefinition.type", Text::Str(r#type));
            f("ToolDefinition.hash", Text::Str(hash));
        }
        f("ToolDefinitionSet.set_hash", Text::Str(set_hash));
    }
}

/// An envelope with every text field present — one span carrying every
/// sub-message, one snapshot carrying every list — and `value` written
/// into the fields `pick` chooses.
fn envelope_with(value: &str, pick: impl Fn(&str) -> bool) -> pb::Envelope {
    let one_kv = || {
        vec![pb::KeyValue {
            key: "note".into(),
            value: Some(pb::AnyValue {
                value: Some(pb::any_value::Value::StringValue(String::new())),
            }),
        }]
    };
    let mut env = pb::Envelope {
        header: Some(pb::EnvelopeHeader {
            resource: Some(Default::default()),
            sdk: Some(pb::SdkInfo {
                adapters: vec![String::new()],
                interceptors: vec![String::new()],
                ..Default::default()
            }),
            ..Default::default()
        }),
        items: vec![
            pb::EnvelopeItem {
                header: Some(Default::default()),
                payload: Some(pb::envelope_item::Payload::Span(pb::Span {
                    status: Some(Default::default()),
                    extra: one_kv(),
                    events: vec![pb::SpanEvent {
                        attributes: one_kv(),
                        ..Default::default()
                    }],
                    links: vec![pb::SpanLink {
                        attributes: one_kv(),
                        ..Default::default()
                    }],
                    transport: Some(pb::TransportAttributes {
                        http: Some(Default::default()),
                        grpc: Some(Default::default()),
                        websocket: Some(Default::default()),
                        mcp: Some(Default::default()),
                        sse: Some(Default::default()),
                        a2a: Some(Default::default()),
                        ..Default::default()
                    }),
                    call_site: Some(Default::default()),
                    conversation: Some(Default::default()),
                    correlation: Some(Default::default()),
                    ..Default::default()
                })),
            },
            pb::EnvelopeItem {
                header: Some(Default::default()),
                payload: Some(pb::envelope_item::Payload::StateSnapshot(
                    pb::StateSnapshot {
                        attributes: one_kv(),
                        input_refs: vec![Default::default()],
                        tool_definitions: Some(pb::ToolDefinitionSet {
                            tools: vec![Default::default()],
                            ..Default::default()
                        }),
                        ..Default::default()
                    },
                )),
            },
        ],
    };
    each_text(&mut env, &mut |path, text| {
        if pick(path) {
            match text {
                Text::Str(s) => *s = value.into(),
                Text::Bytes(b) => *b = value.as_bytes().to_vec(),
            }
        }
    });
    env
}

fn texts(env: &mut pb::Envelope) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    each_text(env, &mut |path, text| {
        out.push((
            path,
            match text {
                Text::Str(s) => s.clone(),
                Text::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
            },
        ));
    });
    out
}

fn the_span(env: &pb::Envelope) -> &pb::Span {
    env.items
        .iter()
        .find_map(|it| match &it.payload {
            Some(pb::envelope_item::Payload::Span(s)) => Some(s),
            _ => None,
        })
        .unwrap()
}

#[test]
fn the_census_value_fires_every_rule_in_a_customer_field() {
    // A tool call's arguments and result: the payloads of an
    // `execute_tool` span.
    let value = every_rule();
    let mut env = pb::Envelope {
        header: None,
        items: vec![pb::EnvelopeItem {
            header: None,
            payload: Some(pb::envelope_item::Payload::Span(pb::Span {
                name: "execute_tool lookup".into(),
                extra: vec![pb::KeyValue {
                    key: "gen_ai.operation.name".into(),
                    value: Some(pb::AnyValue {
                        value: Some(pb::any_value::Value::StringValue("execute_tool".into())),
                    }),
                }],
                input_data: value.clone().into_bytes(),
                output_data: value.clone().into_bytes(),
                ..Default::default()
            })),
        }],
    };
    mask_envelope(&engine(), &mut env);
    let span = the_span(&env);
    for payload in [&span.input_data, &span.output_data] {
        let text = String::from_utf8_lossy(payload);
        for (rule, _, secret) in FRAGMENTS {
            assert!(!text.contains(secret), "{rule:?} left {secret:?} in {text}");
        }
    }
    let rules: Vec<Rule> = span
        .capture_integrity
        .as_ref()
        .unwrap()
        .redaction_rules
        .iter()
        .map(|n| Rule::try_from(*n).unwrap())
        .collect();
    assert_eq!(rules, all_rules());
}

#[test]
fn an_sdk_field_holding_it_is_shipped_as_written_and_unrecorded() {
    let value = every_rule();
    let mut env = envelope_with(&value, |path| SDK_GENERATED_ENVELOPE.contains(&path));
    let before = env.clone();
    mask_envelope(&engine(), &mut env);
    // Byte-identical, which includes "no CaptureIntegrity appeared": a
    // span whose only masked-looking value is the SDK's own carries no
    // record of a masking that did not happen.
    assert_eq!(env, before);
    assert!(the_span(&env).capture_integrity.is_none());
}

#[test]
fn every_other_envelope_text_field_still_loses_it() {
    let value = every_rule();
    let mut env = envelope_with(&value, |_| true);
    mask_envelope(&engine(), &mut env);
    let mut seen = BTreeSet::new();
    for (path, text) in texts(&mut env) {
        seen.insert(path);
        if SDK_GENERATED_ENVELOPE.contains(&path) || RECEIVER_STAMPED.contains(&path) {
            assert_eq!(text, value, "{path} is the SDK's own and was rewritten");
        } else {
            for (rule, _, secret) in FRAGMENTS {
                assert!(
                    !text.contains(secret),
                    "{path} carries host data and kept {secret:?} ({rule:?})"
                );
            }
        }
    }
    for path in SDK_GENERATED_ENVELOPE.iter().chain(RECEIVER_STAMPED) {
        assert!(seen.contains(path), "the census names {path}, no field");
    }
    // The span's record says what its host fields lost, and only that.
    let ci = the_span(&env).capture_integrity.clone().unwrap();
    assert!(ci.redacted && ci.redaction_count > 0);
}

/// Every text value of an OTLP request, by path. Attribute values are
/// keyed by their attribute's key, and repeated paths are refused so a
/// lookup cannot pick the wrong one of two.
fn otlp_texts(req: &otlp_pb::trace_service::ExportTraceServiceRequest) -> BTreeMap<String, String> {
    use otlp_pb::common::any_value::Value;
    fn any(
        out: &mut BTreeMap<String, String>,
        path: String,
        v: &Option<otlp_pb::common::AnyValue>,
    ) {
        let text = match v.as_ref().and_then(|v| v.value.as_ref()) {
            Some(Value::StringValue(s)) => s.clone(),
            Some(Value::BytesValue(b)) => String::from_utf8_lossy(b).into_owned(),
            _ => return,
        };
        assert!(out.insert(path.clone(), text).is_none(), "{path} twice");
    }
    fn kvs(out: &mut BTreeMap<String, String>, at: &str, kvs: &[otlp_pb::common::KeyValue]) {
        for kv in kvs {
            any(out, format!("{at}.attributes[{}]", kv.key), &kv.value);
        }
    }
    let mut out = BTreeMap::new();
    let mut put = |path: String, s: &str| {
        assert!(
            out.insert(path.clone(), s.to_owned()).is_none(),
            "{path} twice"
        );
    };
    let mut attrs = Vec::new();
    for rs in &req.resource_spans {
        put("ResourceSpans.schema_url".into(), &rs.schema_url);
        if let Some(r) = &rs.resource {
            attrs.push(("Resource".to_string(), r.attributes.clone()));
        }
        for ss in &rs.scope_spans {
            put("ScopeSpans.schema_url".into(), &ss.schema_url);
            if let Some(sc) = &ss.scope {
                put("InstrumentationScope.name".into(), &sc.name);
                put("InstrumentationScope.version".into(), &sc.version);
                attrs.push(("InstrumentationScope".into(), sc.attributes.clone()));
            }
            for (i, sp) in ss.spans.iter().enumerate() {
                put(format!("Span[{i}].name"), &sp.name);
                put(format!("Span[{i}].trace_state"), &sp.trace_state);
                attrs.push((format!("Span[{i}]"), sp.attributes.clone()));
                for (j, ev) in sp.events.iter().enumerate() {
                    put(format!("Span[{i}].events[{j}].name"), &ev.name);
                    attrs.push((format!("Span[{i}].events[{j}]"), ev.attributes.clone()));
                }
                for (j, ln) in sp.links.iter().enumerate() {
                    put(format!("Span[{i}].links[{j}].trace_state"), &ln.trace_state);
                    attrs.push((format!("Span[{i}].links[{j}]"), ln.attributes.clone()));
                }
                if let Some(st) = &sp.status {
                    put(format!("Span[{i}].status.message"), &st.message);
                }
            }
        }
    }
    for (at, list) in &attrs {
        kvs(&mut out, at, list);
    }
    out
}

#[test]
fn the_otlp_mapping_of_the_sdk_fields_is_shipped_as_written() {
    let value = every_rule();
    let mut env = envelope_with(&value, |path| SDK_GENERATED_ENVELOPE.contains(&path));
    // A configured service name, so `service.name` is the host's and not
    // the mapping's fallback, which composes the producer's language in.
    if let Some(r) = env.header.as_mut().unwrap().resource.as_mut() {
        r.service_name = "checkout".into();
    }
    let producer = Producer {
        language: &value,
        scope_name: &value,
    };
    let mut req = envelope_to_traces(env, producer);
    let before = req.clone();
    mask_otlp(&engine(), &mut req);
    // Byte-identical: no value rewritten and no `wardex.redacted` added.
    assert_eq!(req, before);
    // And each field did hold the value — except the one the mapping
    // writes from no input at all.
    let texts = otlp_texts(&req);
    for path in SDK_GENERATED_OTLP {
        let want = match *path {
            "Resource.attributes[telemetry.sdk.name]" => "wardex",
            _ => value.as_str(),
        };
        assert_eq!(texts.get(*path).map(String::as_str), Some(want), "{path}");
    }
}

#[test]
fn every_other_otlp_text_field_still_loses_it() {
    let value = every_rule();
    let env = envelope_with(&value, |_| true);
    let producer = Producer {
        language: &value,
        scope_name: &value,
    };
    let mut req = envelope_to_traces(env, producer);
    let before = otlp_texts(&req);
    mask_otlp(&engine(), &mut req);
    let after = otlp_texts(&req);
    let mut carried = 0;
    for (path, text) in &before {
        if !text.contains("john.doe@acme.com") {
            continue; // a value the mapping wrote from nothing of ours
        }
        carried += 1;
        let now = &after[path];
        if SDK_GENERATED_OTLP.contains(&path.as_str()) {
            assert_eq!(now, text, "{path} is the SDK's own and was rewritten");
        } else {
            for (rule, _, secret) in FRAGMENTS {
                assert!(
                    !now.contains(secret),
                    "{path} carries host data and kept {secret:?} ({rule:?})"
                );
            }
        }
    }
    assert!(carried > SDK_GENERATED_OTLP.len(), "only {carried} paths");
    for path in SDK_GENERATED_OTLP {
        assert!(
            before.contains_key(*path),
            "the census names {path}, no field"
        );
    }
}

/// The closed vocabularies are SDK-generated text too, wherever the OTLP
/// mapping spells one into an attribute (`wardex.limitations`,
/// `wardex.parent_source`, `network.protocol.name`, ...). They are not exempted
/// by key — a key in a span's attributes can be the host's as well — and need
/// not be: no name in any of them, nor the name an unknown number is given, is
/// something a rule fires on.
#[test]
fn no_closed_vocabulary_name_is_something_a_rule_fires_on() {
    use wardex_codec::vocab;
    let engine = engine();
    let vocabularies: [fn(i32) -> String; 13] = [
        vocab::span_kind_name,
        vocab::status_code_name,
        vocab::capture_source_name,
        vocab::protocol_name,
        vocab::direction_name,
        vocab::modality_name,
        vocab::snapshot_type_name,
        vocab::operation_name_name,
        vocab::tool_execution_type_name,
        vocab::link_reason_name,
        vocab::limitation_name,
        vocab::parent_source_name,
        vocab::redaction_rule_name,
    ];
    let mut known = 0;
    for name in vocabularies {
        for n in (0..=10_000).chain([i32::MIN, i32::MAX]) {
            let text = name(n);
            if text.is_empty() || (text.contains("_unrecognized_") && (0..=10_000).contains(&n)) {
                continue;
            }
            known += usize::from(!text.contains("_unrecognized_"));
            assert_eq!(engine.mask_text(&text), None, "{text}");
        }
    }
    assert!(known > 100, "only {known} names");
}
