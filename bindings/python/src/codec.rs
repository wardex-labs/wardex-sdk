//! PyO3 core marshaling — `InternalEnvelope`(Python) ↔ `pb::Envelope`(prost).
//!
//! Option 1 (Rust getattr traversal): walks the Python object directly to populate the proto struct.
//! Lossless round-trip including the transport tree, capture_integrity, correlation, and state_snapshots.

// In the trampoline code generated when the pyo3 #[pyfunction] macro wraps a function
// returning `PyResult<T>`, clippy mistakes the `?`'s `From<PyErr> for PyErr` (identity)
// conversion for a useless conversion
// (a pre-existing pyo3 0.22 issue; a function-level #[allow] can't cover macro-generated sibling items).
#![allow(clippy::useless_conversion)]

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList};

use wardex_core::codec::otlp::{self, otlp_pb};
use wardex_core::codec::proto::wardex::v1 as pb;
use wardex_core::codec::{decode_envelope, encode_envelope};
use wardex_core::pipeline::pii;

// --- getattr helpers ---

/// None if the attribute is None, otherwise Some(Bound).
fn opt<'py>(obj: &Bound<'py, PyAny>, name: &str) -> PyResult<Option<Bound<'py, PyAny>>> {
    let v = obj.getattr(name)?;
    Ok(if v.is_none() { None } else { Some(v) })
}

/// TraceId/SpanId wrapper (.value: bytes) → Vec<u8>.
fn id_bytes(obj: &Bound<PyAny>) -> PyResult<Vec<u8>> {
    obj.getattr("value")?.extract()
}

/// Python enum → .value string.
fn enum_str(obj: &Bound<PyAny>) -> PyResult<String> {
    obj.getattr("value")?.extract()
}

// --- value types ---

/// str|int|float|bool Python scalar → proto AnyValue. (bool is a subtype of int, so check it first)
fn any_value(v: &Bound<PyAny>) -> PyResult<pb::AnyValue> {
    use pb::any_value::Value;
    let value = if v.is_instance_of::<PyBool>() {
        Value::BoolValue(v.extract()?)
    } else if let Ok(i) = v.extract::<i64>() {
        Value::IntValue(i)
    } else if let Ok(f) = v.extract::<f64>() {
        Value::DoubleValue(f)
    } else {
        Value::StringValue(v.extract()?)
    };
    Ok(pb::AnyValue { value: Some(value) })
}

fn kv_str(key: &str, v: String) -> pb::KeyValue {
    pb::KeyValue {
        key: key.into(),
        value: Some(pb::AnyValue {
            value: Some(pb::any_value::Value::StringValue(v)),
        }),
    }
}
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
fn kv_bool(key: &str, v: bool) -> pb::KeyValue {
    pb::KeyValue {
        key: key.into(),
        value: Some(pb::AnyValue {
            value: Some(pb::any_value::Value::BoolValue(v)),
        }),
    }
}

/// Python tuple[(key, scalar)] → Vec<KeyValue>.
fn kv_list(extra: &Bound<PyAny>, out: &mut Vec<pb::KeyValue>) -> PyResult<()> {
    for pair in extra.iter()? {
        let pair = pair?;
        let key: String = pair.get_item(0)?.extract()?;
        out.push(pb::KeyValue {
            key,
            value: Some(any_value(&pair.get_item(1)?)?),
        });
    }
    Ok(())
}

// --- enum mapping (unmapped → *_UNSPECIFIED) ---

fn map_span_kind(s: &str) -> i32 {
    (match s {
        "internal" => pb::SpanKind::Internal,
        "client" => pb::SpanKind::Client,
        "server" => pb::SpanKind::Server,
        _ => pb::SpanKind::Unspecified,
    }) as i32
}
fn map_status_code(s: &str) -> i32 {
    (match s {
        "ok" => pb::StatusCode::Ok,
        "error" => pb::StatusCode::Error,
        _ => pb::StatusCode::Unset,
    }) as i32
}
fn map_capture_source(s: &str) -> i32 {
    (match s {
        "adapter" => pb::CaptureSource::Adapter,
        "ssl" => pb::CaptureSource::Ssl,
        "socket" => pb::CaptureSource::Socket,
        "stdio" => pb::CaptureSource::Stdio,
        "grpc" => pb::CaptureSource::Grpc,
        "websocket" => pb::CaptureSource::Websocket,
        "manual" => pb::CaptureSource::Manual,
        _ => pb::CaptureSource::Unspecified,
    }) as i32
}

// --- gen_ai flattening ---

/// GenAIAttributes → extra KeyValue (OTel keys). Only populated fields.
fn flatten_gen_ai(g: &Bound<PyAny>, out: &mut Vec<pb::KeyValue>) -> PyResult<()> {
    // operation: OperationName|str (required). If enum, use .value; if str, use as-is.
    let op = g.getattr("operation")?;
    let op_s: String = if op.hasattr("value")? {
        enum_str(&op)?
    } else {
        op.extract()?
    };
    out.push(kv_str("gen_ai.operation.name", op_s));
    if let Some(p) = opt(g, "provider")? {
        let p_s: String = if p.hasattr("value")? {
            enum_str(&p)?
        } else {
            p.extract()?
        };
        out.push(kv_str("gen_ai.provider.name", p_s));
    }
    // string fields
    for (attr, key) in [
        ("request_model", "gen_ai.request.model"),
        ("response_model", "gen_ai.response.model"),
        ("response_id", "gen_ai.response.id"),
        ("prompt_name", "gen_ai.prompt.name"),
    ] {
        if let Some(v) = opt(g, attr)? {
            out.push(kv_str(key, v.extract()?));
        }
    }
    // integer fields
    for (attr, key) in [
        ("input_tokens", "gen_ai.usage.input_tokens"),
        ("output_tokens", "gen_ai.usage.output_tokens"),
        (
            "cache_read_input_tokens",
            "gen_ai.usage.cache_read_input_tokens",
        ),
        (
            "cache_creation_input_tokens",
            "gen_ai.usage.cache_creation_input_tokens",
        ),
        (
            "reasoning_output_tokens",
            "gen_ai.usage.reasoning_output_tokens",
        ),
        ("max_tokens", "gen_ai.request.max_tokens"),
        ("seed", "gen_ai.request.seed"),
        ("choice_count", "gen_ai.request.choice.count"),
    ] {
        if let Some(v) = opt(g, attr)? {
            out.push(kv_int(key, v.extract()?));
        }
    }
    // float fields
    for (attr, key) in [
        ("temperature", "gen_ai.request.temperature"),
        ("top_p", "gen_ai.request.top_p"),
        ("top_k", "gen_ai.request.top_k"),
        ("frequency_penalty", "gen_ai.request.frequency_penalty"),
        ("presence_penalty", "gen_ai.request.presence_penalty"),
        (
            "time_to_first_chunk_s",
            "gen_ai.response.time_to_first_chunk",
        ),
    ] {
        if let Some(v) = opt(g, attr)? {
            out.push(kv_double(key, v.extract()?));
        }
    }
    // bool
    if let Some(v) = opt(g, "stream")? {
        out.push(kv_bool("gen_ai.request.stream", v.extract()?));
    }
    // tuple[str] → CSV
    for (attr, key) in [
        ("stop_sequences", "gen_ai.request.stop_sequences"),
        ("finish_reasons", "gen_ai.response.finish_reasons"),
        ("encoding_formats", "gen_ai.request.encoding_formats"),
    ] {
        if let Some(v) = opt(g, attr)? {
            let items: Vec<String> = v.extract()?;
            out.push(kv_str(key, items.join(",")));
        }
    }
    // output_type: enum|str
    if let Some(v) = opt(g, "output_type")? {
        let s: String = if v.hasattr("value")? {
            enum_str(&v)?
        } else {
            v.extract()?
        };
        out.push(kv_str("gen_ai.output.type", s));
    }
    Ok(())
}

/// AgentAttributes → extra KeyValue. Only populated fields.
fn flatten_agent(a: &Bound<PyAny>, out: &mut Vec<pb::KeyValue>) -> PyResult<()> {
    out.push(kv_str("gen_ai.agent.name", a.getattr("name")?.extract()?));
    for (attr, key) in [
        ("id", "gen_ai.agent.id"),
        ("description", "gen_ai.agent.description"),
        ("version", "wardex.agent.version"),
        ("parent_agent", "wardex.agent.parent"),
    ] {
        if let Some(v) = opt(a, attr)? {
            out.push(kv_str(key, v.extract()?));
        }
    }
    if let Some(t) = opt(a, "agent_type")? {
        out.push(kv_str("wardex.agent.type", enum_str(&t)?));
    }
    Ok(())
}

/// ToolAttributes → extra KeyValue. Only populated fields.
fn flatten_tool(t: &Bound<PyAny>, out: &mut Vec<pb::KeyValue>) -> PyResult<()> {
    out.push(kv_str("gen_ai.tool.name", t.getattr("name")?.extract()?));
    for (attr, key) in [
        ("call_id", "gen_ai.tool.call.id"),
        ("description", "gen_ai.tool.description"),
    ] {
        if let Some(v) = opt(t, attr)? {
            out.push(kv_str(key, v.extract()?));
        }
    }
    if let Some(ty) = opt(t, "type")? {
        let ty_s: String = if ty.hasattr("value")? {
            enum_str(&ty)?
        } else {
            ty.extract()?
        };
        out.push(kv_str("gen_ai.tool.type", ty_s));
    }
    if let Some(ex) = opt(t, "execution_type")? {
        out.push(kv_str("wardex.tool.execution_type", enum_str(&ex)?));
    }
    Ok(())
}

// --- enum mapping (transport/state) ---

fn map_protocol(s: &str) -> i32 {
    (match s {
        "http" => pb::Protocol::Http,
        "grpc" => pb::Protocol::Grpc,
        "websocket" => pb::Protocol::Websocket,
        "mcp_stdio" => pb::Protocol::McpStdio,
        "sse" => pb::Protocol::Sse,
        _ => pb::Protocol::Unspecified,
    }) as i32
}
fn map_direction(s: &str) -> i32 {
    (match s {
        "outbound" => pb::Direction::Outbound,
        "inbound" => pb::Direction::Inbound,
        _ => pb::Direction::Unspecified,
    }) as i32
}
fn map_modality(s: &str) -> i32 {
    (match s {
        "text" => pb::Modality::Text,
        "image" => pb::Modality::Image,
        "audio" => pb::Modality::Audio,
        "video" => pb::Modality::Video,
        "embedding" => pb::Modality::Embedding,
        _ => pb::Modality::Unspecified,
    }) as i32
}

// --- transport / capture_integrity / correlation / state ---

fn transport_to_proto(t: &Bound<PyAny>) -> PyResult<pb::TransportAttributes> {
    let tm = t.getattr("timing")?;
    let mut tr = pb::TransportAttributes {
        connection_id: t.getattr("connection_id")?.extract()?,
        protocol: map_protocol(&enum_str(&t.getattr("protocol")?)?),
        direction: map_direction(&enum_str(&t.getattr("direction")?)?),
        timing: Some(pb::TransportTiming {
            tcp_connect_ms: tm.getattr("tcp_connect_ms")?.extract()?,
            tls_handshake_ms: tm.getattr("tls_handshake_ms")?.extract()?,
            ttfb_ms: tm.getattr("ttfb_ms")?.extract()?,
            transfer_ms: tm.getattr("transfer_ms")?.extract()?,
            ttft_ms: tm.getattr("ttft_ms")?.extract()?,
        }),
        request_size: t.getattr("request_size")?.extract()?,
        response_size: t.getattr("response_size")?.extract()?,
        request_modality: map_modality(&enum_str(&t.getattr("request_modality")?)?),
        response_modality: map_modality(&enum_str(&t.getattr("response_modality")?)?),
        is_streaming: t.getattr("is_streaming")?.extract()?,
        chunk_index: t.getattr("chunk_index")?.extract()?,
        is_final_chunk: t.getattr("is_final_chunk")?.extract()?,
        connection_reused: t.getattr("connection_reused")?.extract()?,
        ..Default::default()
    };
    if let Some(h) = opt(t, "http")? {
        tr.http = Some(pb::HttpMeta {
            method: h.getattr("method")?.extract()?,
            url: h.getattr("url")?.extract()?,
            status_code: h.getattr("status_code")?.extract()?,
        });
    }
    if let Some(g) = opt(t, "grpc")? {
        let mut m = pb::GrpcMeta {
            service: g.getattr("service")?.extract()?,
            method: g.getattr("method")?.extract()?,
            ..Default::default()
        };
        if let Some(v) = opt(&g, "stream_id")? {
            m.stream_id = v.extract()?;
        }
        if let Some(v) = opt(&g, "status_code")? {
            m.status_code = v.extract()?;
        }
        if let Some(v) = opt(&g, "encoding")? {
            m.encoding = v.extract()?;
        }
        tr.grpc = Some(m);
    }
    if let Some(w) = opt(t, "websocket")? {
        tr.websocket = Some(pb::WebSocketMeta {
            opcode: w.getattr("opcode")?.extract()?,
            direction: w.getattr("direction")?.extract()?,
        });
    }
    if let Some(mc) = opt(t, "mcp")? {
        let mut m = pb::McpMeta {
            rpc_method: mc.getattr("rpc_method")?.extract()?,
            ..Default::default()
        };
        if let Some(v) = opt(&mc, "rpc_id")? {
            m.rpc_id = v.extract()?;
        }
        tr.mcp = Some(m);
    }
    if let Some(se) = opt(t, "sse")? {
        let mut m = pb::SseMeta::default();
        if let Some(v) = opt(&se, "event_type")? {
            m.event_type = v.extract()?;
        }
        tr.sse = Some(m);
    }
    if let Some(a) = opt(t, "a2a")? {
        tr.a2a = Some(pb::A2aMeta {
            task_id: a.getattr("task_id")?.extract()?,
            transport: a.getattr("transport")?.extract()?,
        });
    }
    if let Some(v) = opt(t, "request_blob_ref")? {
        tr.request_blob_ref = v.extract()?;
    }
    if let Some(v) = opt(t, "response_blob_ref")? {
        tr.response_blob_ref = v.extract()?;
    }
    Ok(tr)
}

fn integrity_to_proto(c: &Bound<PyAny>) -> PyResult<pb::CaptureIntegrity> {
    Ok(pb::CaptureIntegrity {
        request_headers_captured: c.getattr("request_headers_captured")?.extract()?,
        request_body_captured: c.getattr("request_body_captured")?.extract()?,
        response_headers_captured: c.getattr("response_headers_captured")?.extract()?,
        response_body_captured: c.getattr("response_body_captured")?.extract()?,
        redacted: c.getattr("redacted")?.extract()?,
        truncated: c.getattr("truncated")?.extract()?,
        dropped_chunk_count: c.getattr("dropped_chunk_count")?.extract()?,
        limitations: c.getattr("limitations")?.extract()?,
    })
}

fn correlation_to_proto(c: &Bound<PyAny>) -> PyResult<pb::CorrelationInfo> {
    let mut corr = pb::CorrelationInfo {
        confidence: c.getattr("confidence")?.extract()?,
        ..Default::default()
    };
    if let Some(v) = opt(c, "operation_id")? {
        corr.operation_id = v.extract()?;
    }
    if let Some(v) = opt(c, "request_id")? {
        corr.request_id = v.extract()?;
    }
    if let Some(v) = opt(c, "attempt_id")? {
        corr.attempt_id = v.extract()?;
    }
    if let Some(v) = opt(c, "strategy")? {
        corr.strategy = v.extract()?;
    }
    if let Some(v) = opt(c, "active_span_id_at_capture")? {
        corr.active_span_id_at_capture = id_bytes(&v)?;
    }
    Ok(corr)
}

fn state_to_proto(s: &Bound<PyAny>) -> PyResult<pb::StateSnapshot> {
    let map_snap = |v: &str| -> i32 {
        (match v {
            "span_start" => pb::SnapshotType::SpanStart,
            "span_end" => pb::SnapshotType::SpanEnd,
            "turn_start" => pb::SnapshotType::TurnStart,
            _ => pb::SnapshotType::Unspecified,
        }) as i32
    };
    let mut ss = pb::StateSnapshot {
        trace_id: id_bytes(&s.getattr("trace_id")?)?,
        span_id: id_bytes(&s.getattr("span_id")?)?,
        timestamp_ns: s.getattr("timestamp_ns")?.extract()?,
        snapshot_type: map_snap(&s.getattr("snapshot_type")?.extract::<String>()?),
        turn_index: s.getattr("turn_index")?.extract()?,
        conversation_state: s.getattr("conversation_state")?.extract()?,
        ..Default::default()
    };
    kv_list(&s.getattr("attributes")?, &mut ss.attributes)?;
    for r in s.getattr("input_refs")?.iter()? {
        let r = r?;
        let mut ir = pb::InputRef {
            key: r.getattr("key")?.extract()?,
            content_hash: r.getattr("content_hash")?.extract()?,
            ..Default::default()
        };
        if let Some(v) = opt(&r, "blob_ref")? {
            ir.blob_ref = v.extract()?;
        }
        ss.input_refs.push(ir);
    }
    if let Some(td_set) = opt(s, "tool_definitions")? {
        let mut set = pb::ToolDefinitionSet {
            set_hash: td_set.getattr("set_hash")?.extract()?,
            ..Default::default()
        };
        for td in td_set.getattr("tools")?.iter()? {
            let td = td?;
            let mut def = pb::ToolDefinition {
                name: td.getattr("name")?.extract()?,
                parameters_schema: td.getattr("parameters_schema")?.extract()?,
                ..Default::default()
            };
            if let Some(v) = opt(&td, "description")? {
                def.description = v.extract()?;
            }
            if let Some(v) = opt(&td, "version")? {
                def.version = v.extract()?;
            }
            if let Some(v) = opt(&td, "hash")? {
                def.hash = v.extract()?;
            }
            if let Some(v) = opt(&td, "type")? {
                def.r#type = if v.hasattr("value")? {
                    enum_str(&v)?
                } else {
                    v.extract()?
                };
            }
            set.tools.push(def);
        }
        ss.tool_definitions = Some(set);
    }
    Ok(ss)
}

// --- span ---

fn span_to_proto(sp: &Bound<PyAny>) -> PyResult<pb::Span> {
    let ctx = sp.getattr("context")?;
    let mut span = pb::Span {
        trace_id: id_bytes(&ctx.getattr("trace_id")?)?,
        span_id: id_bytes(&ctx.getattr("span_id")?)?,
        name: sp.getattr("name")?.extract()?,
        kind: map_span_kind(&enum_str(&sp.getattr("kind")?)?),
        start_time_unix_nano: sp.getattr("start_time_ns")?.extract()?,
        end_time_unix_nano: sp.getattr("end_time_ns")?.extract()?,
        status: Some(pb::Status {
            code: map_status_code(&enum_str(&sp.getattr("status")?)?),
            message: sp.getattr("status_message")?.extract()?,
        }),
        input_data: sp.getattr("input_data")?.extract()?,
        output_data: sp.getattr("output_data")?.extract()?,
        ..Default::default()
    };
    if let Some(pid) = opt(sp, "parent_span_id")? {
        span.parent_span_id = id_bytes(&pid)?;
    }
    if let Some(v) = opt(sp, "error_type")? {
        span.error_type = v.extract()?;
    }
    if let Some(v) = opt(sp, "server_address")? {
        span.server_address = v.extract()?;
    }
    if let Some(v) = opt(sp, "server_port")? {
        span.server_port = v.extract()?;
    }
    if let Some(v) = opt(sp, "workflow_name")? {
        span.workflow_name = v.extract()?;
    }
    // capture_sources
    for src in sp.getattr("capture_sources")?.iter()? {
        span.capture_sources
            .push(map_capture_source(&enum_str(&src?)?));
    }
    // extra passthrough
    kv_list(&sp.getattr("extra")?, &mut span.extra)?;
    // gen_ai flattening → extra
    if let Some(g) = opt(sp, "gen_ai")? {
        flatten_gen_ai(&g, &mut span.extra)?;
    }
    if let Some(a) = opt(sp, "agent")? {
        flatten_agent(&a, &mut span.extra)?;
    }
    if let Some(t) = opt(sp, "tool")? {
        flatten_tool(&t, &mut span.extra)?;
    }
    if let Some(t) = opt(sp, "transport")? {
        span.transport = Some(transport_to_proto(&t)?);
    }
    if let Some(c) = opt(sp, "capture_integrity")? {
        span.capture_integrity = Some(integrity_to_proto(&c)?);
    }
    if let Some(c) = opt(sp, "correlation")? {
        span.correlation = Some(correlation_to_proto(&c)?);
    }
    Ok(span)
}

// --- envelope ---

fn header_to_proto(h: &Bound<PyAny>) -> PyResult<pb::EnvelopeHeader> {
    let sdk = h.getattr("sdk")?;
    Ok(pb::EnvelopeHeader {
        event_id: h.getattr("event_id")?.extract()?,
        api_key: h.getattr("api_key")?.extract()?,
        sent_at_unix_nano: h.getattr("sent_at_ns")?.extract()?,
        sdk: Some(pb::SdkInfo {
            name: sdk.getattr("name")?.extract()?,
            version: sdk.getattr("version")?.extract()?,
            python_version: sdk.getattr("python_version")?.extract()?,
            os: sdk.getattr("os")?.extract()?,
            arch: sdk.getattr("arch")?.extract()?,
            adapters: sdk.getattr("adapters")?.extract()?,
            interceptors: sdk.getattr("interceptors")?.extract()?,
            otel_semconv_version: sdk.getattr("otel_semconv_version")?.extract()?,
            shell: sdk.getattr("shell")?.extract()?,
        }),
        ..Default::default()
    })
}

fn envelope_to_proto(env: &Bound<PyAny>) -> PyResult<pb::Envelope> {
    let mut items = Vec::new();
    for sp in env.getattr("spans")?.iter()? {
        items.push(pb::EnvelopeItem {
            header: Some(pb::EnvelopeItemHeader {
                r#type: "span".into(),
                length: 0,
            }),
            payload: Some(pb::envelope_item::Payload::Span(span_to_proto(&sp?)?)),
        });
    }
    for s in env.getattr("state_snapshots")?.iter()? {
        items.push(pb::EnvelopeItem {
            header: Some(pb::EnvelopeItemHeader {
                r#type: "state_snapshot".into(),
                length: 0,
            }),
            payload: Some(pb::envelope_item::Payload::StateSnapshot(state_to_proto(
                &s?,
            )?)),
        });
    }
    Ok(pb::Envelope {
        header: Some(header_to_proto(&env.getattr("header")?)?),
        items,
    })
}

// --- decode → dict ---

fn any_to_py(py: Python<'_>, v: &pb::AnyValue) -> PyObject {
    use pb::any_value::Value;
    match &v.value {
        Some(Value::StringValue(s)) => s.into_py(py),
        Some(Value::IntValue(i)) => i.into_py(py),
        Some(Value::DoubleValue(d)) => d.into_py(py),
        Some(Value::BoolValue(b)) => b.into_py(py),
        Some(Value::BytesValue(b)) => PyBytes::new_bound(py, b).into_py(py),
        _ => py.None(),
    }
}

fn kv_to_py(py: Python<'_>, kvs: &[pb::KeyValue]) -> PyResult<PyObject> {
    let list = PyList::empty_bound(py);
    for kv in kvs {
        let d = PyDict::new_bound(py);
        d.set_item("key", &kv.key)?;
        d.set_item(
            "value",
            kv.value
                .as_ref()
                .map(|v| any_to_py(py, v))
                .unwrap_or_else(|| py.None()),
        )?;
        list.append(d)?;
    }
    Ok(list.into_py(py))
}

fn span_to_dict(py: Python<'_>, sp: &pb::Span) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    d.set_item("trace_id", PyBytes::new_bound(py, &sp.trace_id))?;
    d.set_item("span_id", PyBytes::new_bound(py, &sp.span_id))?;
    d.set_item("parent_span_id", PyBytes::new_bound(py, &sp.parent_span_id))?;
    d.set_item("name", &sp.name)?;
    d.set_item("kind", sp.kind)?;
    d.set_item("start_time_unix_nano", sp.start_time_unix_nano)?;
    d.set_item("end_time_unix_nano", sp.end_time_unix_nano)?;
    let status = PyDict::new_bound(py);
    if let Some(s) = &sp.status {
        status.set_item("code", s.code)?;
        status.set_item("message", &s.message)?;
    }
    d.set_item("status", status)?;
    d.set_item("error_type", &sp.error_type)?;
    d.set_item("server_address", &sp.server_address)?;
    d.set_item("server_port", sp.server_port)?;
    d.set_item("workflow_name", &sp.workflow_name)?;
    d.set_item("input_data", PyBytes::new_bound(py, &sp.input_data))?;
    d.set_item("output_data", PyBytes::new_bound(py, &sp.output_data))?;
    d.set_item("capture_sources", sp.capture_sources.clone())?;
    d.set_item("extra", kv_to_py(py, &sp.extra)?)?;
    if let Some(t) = &sp.transport {
        let td = PyDict::new_bound(py);
        td.set_item("protocol", t.protocol)?;
        td.set_item("direction", t.direction)?;
        td.set_item("request_size", t.request_size)?;
        td.set_item("response_size", t.response_size)?;
        td.set_item("is_streaming", t.is_streaming)?;
        td.set_item("connection_reused", t.connection_reused)?;
        if let Some(tm) = &t.timing {
            let d2 = PyDict::new_bound(py);
            d2.set_item("tcp_connect_ms", tm.tcp_connect_ms)?;
            d2.set_item("tls_handshake_ms", tm.tls_handshake_ms)?;
            d2.set_item("ttfb_ms", tm.ttfb_ms)?;
            d2.set_item("transfer_ms", tm.transfer_ms)?;
            d2.set_item("ttft_ms", tm.ttft_ms)?;
            td.set_item("timing", d2)?;
        }
        if let Some(h) = &t.http {
            let d2 = PyDict::new_bound(py);
            d2.set_item("method", &h.method)?;
            d2.set_item("url", &h.url)?;
            d2.set_item("status_code", h.status_code)?;
            td.set_item("http", d2)?;
        }
        if let Some(a) = &t.a2a {
            let d2 = PyDict::new_bound(py);
            d2.set_item("task_id", &a.task_id)?;
            d2.set_item("transport", &a.transport)?;
            td.set_item("a2a", d2)?;
        }
        td.set_item("request_blob_ref", &t.request_blob_ref)?;
        td.set_item("response_blob_ref", &t.response_blob_ref)?;
        d.set_item("transport", td)?;
    }
    if let Some(c) = &sp.capture_integrity {
        let cd = PyDict::new_bound(py);
        cd.set_item("request_body_captured", c.request_body_captured)?;
        cd.set_item("response_body_captured", c.response_body_captured)?;
        cd.set_item("truncated", c.truncated)?;
        cd.set_item("redacted", c.redacted)?;
        cd.set_item("limitations", c.limitations.clone())?;
        d.set_item("capture_integrity", cd)?;
    }
    if let Some(c) = &sp.correlation {
        let cd = PyDict::new_bound(py);
        cd.set_item("strategy", &c.strategy)?;
        cd.set_item("confidence", c.confidence)?;
        d.set_item("correlation", cd)?;
    }
    Ok(d.into_py(py))
}

fn envelope_to_dict(py: Python<'_>, env: &pb::Envelope) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    let h = PyDict::new_bound(py);
    if let Some(header) = &env.header {
        h.set_item("event_id", &header.event_id)?;
        h.set_item("api_key", &header.api_key)?;
        h.set_item("sent_at_unix_nano", header.sent_at_unix_nano)?;
        let sdk = PyDict::new_bound(py);
        if let Some(s) = &header.sdk {
            sdk.set_item("name", &s.name)?;
            sdk.set_item("version", &s.version)?;
        }
        h.set_item("sdk", sdk)?;
    }
    d.set_item("header", h)?;
    let items = PyList::empty_bound(py);
    for item in &env.items {
        let id = PyDict::new_bound(py);
        if let Some(pb::envelope_item::Payload::Span(sp)) = &item.payload {
            id.set_item("span", span_to_dict(py, sp)?)?;
        }
        if let Some(pb::envelope_item::Payload::StateSnapshot(ss)) = &item.payload {
            let sd = PyDict::new_bound(py);
            sd.set_item("trace_id", PyBytes::new_bound(py, &ss.trace_id))?;
            sd.set_item("turn_index", ss.turn_index)?;
            sd.set_item(
                "conversation_state",
                PyBytes::new_bound(py, &ss.conversation_state),
            )?;
            let refs = PyList::empty_bound(py);
            for r in &ss.input_refs {
                let rd = PyDict::new_bound(py);
                rd.set_item("key", &r.key)?;
                rd.set_item("content_hash", &r.content_hash)?;
                refs.append(rd)?;
            }
            sd.set_item("input_refs", refs)?;
            if let Some(td_set) = &ss.tool_definitions {
                let tdd = PyDict::new_bound(py);
                tdd.set_item("set_hash", &td_set.set_hash)?;
                let names = PyList::empty_bound(py);
                for td in &td_set.tools {
                    names.append(&td.name)?;
                }
                tdd.set_item("tool_names", names)?;
                sd.set_item("tool_definitions", tdd)?;
            }
            id.set_item("state_snapshot", sd)?;
        }
        items.append(id)?;
    }
    d.set_item("items", items)?;
    Ok(d.into_py(py))
}

// --- OTLP mapping (traces-only; ExportTraceServiceRequest) ---

// OTLP enum values (different from wardex values): SpanKind CLIENT=3, StatusCode OK=1/ERROR=2.
fn map_span_kind_otlp(s: &str) -> i32 {
    use otlp_pb::trace::span::SpanKind;
    (match s {
        "internal" => SpanKind::Internal,
        "client" => SpanKind::Client,
        "server" => SpanKind::Server,
        "producer" => SpanKind::Producer,
        "consumer" => SpanKind::Consumer,
        _ => SpanKind::Unspecified,
    }) as i32
}
fn map_status_code_otlp(s: &str) -> i32 {
    use otlp_pb::trace::status::StatusCode;
    (match s {
        "ok" => StatusCode::Ok,
        "error" => StatusCode::Error,
        _ => StatusCode::Unset,
    }) as i32
}

/// wardex AnyValue → OTLP AnyValue (variant remapping; avoids large-scale logic duplication).
fn wardex_any_to_otlp(v: &pb::AnyValue) -> otlp_pb::common::AnyValue {
    use otlp_pb::common::any_value::Value as OV;
    use pb::any_value::Value as WV;
    let value = match &v.value {
        Some(WV::StringValue(s)) => Some(OV::StringValue(s.clone())),
        Some(WV::IntValue(i)) => Some(OV::IntValue(*i)),
        Some(WV::DoubleValue(d)) => Some(OV::DoubleValue(*d)),
        Some(WV::BoolValue(b)) => Some(OV::BoolValue(*b)),
        Some(WV::BytesValue(b)) => Some(OV::BytesValue(b.clone())),
        _ => None,
    };
    otlp_pb::common::AnyValue { value }
}
/// wardex KeyValue → OTLP KeyValue.
fn wardex_kv_to_otlp(kv: &pb::KeyValue) -> otlp_pb::common::KeyValue {
    otlp_pb::common::KeyValue {
        key: kv.key.clone(),
        value: kv.value.as_ref().map(wardex_any_to_otlp),
    }
}

// Small builders for native OTLP KeyValue.
fn otlp_kv_str(key: &str, v: &str) -> otlp_pb::common::KeyValue {
    otlp_pb::common::KeyValue {
        key: key.into(),
        value: Some(otlp_pb::common::AnyValue {
            value: Some(otlp_pb::common::any_value::Value::StringValue(v.into())),
        }),
    }
}
fn otlp_kv_int(key: &str, v: i64) -> otlp_pb::common::KeyValue {
    otlp_pb::common::KeyValue {
        key: key.into(),
        value: Some(otlp_pb::common::AnyValue {
            value: Some(otlp_pb::common::any_value::Value::IntValue(v)),
        }),
    }
}
fn otlp_kv_bytes(key: &str, v: Vec<u8>) -> otlp_pb::common::KeyValue {
    otlp_pb::common::KeyValue {
        key: key.into(),
        value: Some(otlp_pb::common::AnyValue {
            value: Some(otlp_pb::common::any_value::Value::BytesValue(v)),
        }),
    }
}

/// InternalSpan(Python) → OTLP Span. Mirrors `span_to_proto` + OTLP enum/types.
fn span_to_otlp(sp: &Bound<PyAny>) -> PyResult<otlp_pb::trace::Span> {
    let ctx = sp.getattr("context")?;
    let mut span = otlp_pb::trace::Span {
        trace_id: id_bytes(&ctx.getattr("trace_id")?)?,
        span_id: id_bytes(&ctx.getattr("span_id")?)?,
        name: sp.getattr("name")?.extract()?,
        kind: map_span_kind_otlp(&enum_str(&sp.getattr("kind")?)?),
        start_time_unix_nano: sp.getattr("start_time_ns")?.extract()?,
        end_time_unix_nano: sp.getattr("end_time_ns")?.extract()?,
        ..Default::default()
    };
    if let Some(pid) = opt(sp, "parent_span_id")? {
        span.parent_span_id = id_bytes(&pid)?;
    }
    // status: code + message (error_type takes priority, falls back to status_message)
    let code = map_status_code_otlp(&enum_str(&sp.getattr("status")?)?);
    let message: String = match opt(sp, "error_type")? {
        Some(v) => v.extract()?,
        None => sp.getattr("status_message")?.extract()?,
    };
    span.status = Some(otlp_pb::trace::Status { code, message });

    // attributes: extra passthrough + gen_ai flattening (reuses wardex KV) → OTLP conversion
    let mut wkv: Vec<pb::KeyValue> = Vec::new();
    kv_list(&sp.getattr("extra")?, &mut wkv)?;
    if let Some(g) = opt(sp, "gen_ai")? {
        flatten_gen_ai(&g, &mut wkv)?;
    }
    if let Some(a) = opt(sp, "agent")? {
        flatten_agent(&a, &mut wkv)?;
    }
    if let Some(t) = opt(sp, "tool")? {
        flatten_tool(&t, &mut wkv)?;
    }
    let mut attrs: Vec<otlp_pb::common::KeyValue> = wkv.iter().map(wardex_kv_to_otlp).collect();

    // server.* (span-level)
    if let Some(v) = opt(sp, "server_address")? {
        attrs.push(otlp_kv_str("server.address", &v.extract::<String>()?));
    }
    if let Some(v) = opt(sp, "server_port")? {
        attrs.push(otlp_kv_int("server.port", v.extract()?));
    }
    // transport-derived: network.protocol.name + http.request.method/http.response.status_code
    if let Some(t) = opt(sp, "transport")? {
        let proto = enum_str(&t.getattr("protocol")?)?;
        attrs.push(otlp_kv_str("network.protocol.name", &proto));
        if let Some(h) = opt(&t, "http")? {
            attrs.push(otlp_kv_str(
                "http.request.method",
                &h.getattr("method")?.extract::<String>()?,
            ));
            attrs.push(otlp_kv_int(
                "http.response.status_code",
                h.getattr("status_code")?.extract()?,
            ));
        }
    }
    // raw I/O → wardex.input_data / wardex.output_data (omitted if empty)
    let input: Vec<u8> = sp.getattr("input_data")?.extract()?;
    if !input.is_empty() {
        attrs.push(otlp_kv_bytes("wardex.input_data", input));
    }
    let output: Vec<u8> = sp.getattr("output_data")?.extract()?;
    if !output.is_empty() {
        attrs.push(otlp_kv_bytes("wardex.output_data", output));
    }
    span.attributes = attrs;
    Ok(span)
}

/// InternalEnvelope → ExportTraceServiceRequest. state_snapshots are skipped (traces-only).
/// If spans is empty, resource_spans is an empty vector.
fn envelope_to_otlp(
    env: &Bound<PyAny>,
) -> PyResult<otlp_pb::trace_service::ExportTraceServiceRequest> {
    let mut spans = Vec::new();
    for sp in env.getattr("spans")?.iter()? {
        spans.push(span_to_otlp(&sp?)?);
    }
    if spans.is_empty() {
        return Ok(otlp_pb::trace_service::ExportTraceServiceRequest {
            resource_spans: vec![],
        });
    }
    let sdk = env.getattr("header")?.getattr("sdk")?;
    let name: String = sdk.getattr("name")?.extract()?;
    let version: String = sdk.getattr("version")?.extract()?;
    let resource = otlp_pb::resource::Resource {
        attributes: vec![
            otlp_kv_str("service.name", &name),
            otlp_kv_str("service.version", &version),
            otlp_kv_str("telemetry.sdk.name", &name),
            otlp_kv_str("telemetry.sdk.version", &version),
            otlp_kv_str("telemetry.sdk.language", "python"),
        ],
        ..Default::default()
    };
    let scope = otlp_pb::common::InstrumentationScope {
        name: "wardex.python".into(),
        version,
        ..Default::default()
    };
    Ok(otlp_pb::trace_service::ExportTraceServiceRequest {
        resource_spans: vec![otlp_pb::trace::ResourceSpans {
            resource: Some(resource),
            scope_spans: vec![otlp_pb::trace::ScopeSpans {
                scope: Some(scope),
                spans,
                ..Default::default()
            }],
            ..Default::default()
        }],
    })
}

// --- OTLP decode → dict (for tests/debugging) ---

fn to_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        let _ = write!(s, "{:02x}", b);
    }
    s
}

fn otlp_any_to_py(py: Python<'_>, v: &otlp_pb::common::AnyValue) -> PyObject {
    use otlp_pb::common::any_value::Value;
    match &v.value {
        Some(Value::StringValue(s)) => s.into_py(py),
        Some(Value::IntValue(i)) => i.into_py(py),
        Some(Value::DoubleValue(d)) => d.into_py(py),
        Some(Value::BoolValue(b)) => b.into_py(py),
        Some(Value::BytesValue(b)) => PyBytes::new_bound(py, b).into_py(py),
        _ => py.None(),
    }
}

/// OTLP KeyValue list → key→value dict.
fn otlp_attrs_to_py(py: Python<'_>, kvs: &[otlp_pb::common::KeyValue]) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    for kv in kvs {
        let val = kv
            .value
            .as_ref()
            .map(|v| otlp_any_to_py(py, v))
            .unwrap_or_else(|| py.None());
        d.set_item(&kv.key, val)?;
    }
    Ok(d.into_py(py))
}

fn otlp_traces_to_dict(
    py: Python<'_>,
    req: &otlp_pb::trace_service::ExportTraceServiceRequest,
) -> PyResult<PyObject> {
    let d = PyDict::new_bound(py);
    let rs_list = PyList::empty_bound(py);
    for rs in &req.resource_spans {
        let rsd = PyDict::new_bound(py);
        let resd = PyDict::new_bound(py);
        if let Some(r) = &rs.resource {
            resd.set_item("attributes", otlp_attrs_to_py(py, &r.attributes)?)?;
        }
        rsd.set_item("resource", resd)?;
        let ss_list = PyList::empty_bound(py);
        for ss in &rs.scope_spans {
            let ssd = PyDict::new_bound(py);
            let scoped = PyDict::new_bound(py);
            if let Some(sc) = &ss.scope {
                scoped.set_item("name", &sc.name)?;
                scoped.set_item("version", &sc.version)?;
            }
            ssd.set_item("scope", scoped)?;
            let spans_list = PyList::empty_bound(py);
            for sp in &ss.spans {
                let spd = PyDict::new_bound(py);
                spd.set_item("name", &sp.name)?;
                spd.set_item("trace_id", to_hex(&sp.trace_id))?;
                spd.set_item("span_id", to_hex(&sp.span_id))?;
                spd.set_item("parent_span_id", to_hex(&sp.parent_span_id))?;
                spd.set_item("kind", sp.kind)?;
                spd.set_item("start_time_unix_nano", sp.start_time_unix_nano)?;
                spd.set_item("end_time_unix_nano", sp.end_time_unix_nano)?;
                let status = PyDict::new_bound(py);
                if let Some(s) = &sp.status {
                    status.set_item("code", s.code)?;
                    status.set_item("message", &s.message)?;
                }
                spd.set_item("status", status)?;
                spd.set_item("attributes", otlp_attrs_to_py(py, &sp.attributes)?)?;
                spans_list.append(spd)?;
            }
            ssd.set_item("spans", spans_list)?;
            ss_list.append(ssd)?;
        }
        rsd.set_item("scope_spans", ss_list)?;
        rs_list.append(rsd)?;
    }
    d.set_item("resource_spans", rs_list)?;
    Ok(d.into_py(py))
}

// --- pyfunctions + submodule ---

/// Apply the PII policy to a marshalled wardex envelope (design §4.2).
/// pii_mode contract: "mask" | "off" — anything else is a hard error
/// (REDACT/HASH are rejected earlier by Python init; defense in depth here).
fn pii_apply_envelope(
    proto: &mut pb::Envelope,
    pii_mode: &str,
    pii_disabled: &[String],
) -> PyResult<()> {
    match pii_mode {
        "off" => Ok(()),
        "mask" => {
            let engine = pii::engine_for(pii_disabled)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            pii::mask_envelope(&engine, proto);
            Ok(())
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported pii_mode {other:?} (expected \"mask\" or \"off\")"
        ))),
    }
}

/// Apply the PII policy to a marshalled OTLP export request (design §4.2).
/// Same "mask" | "off" contract as `pii_apply_envelope`.
fn pii_apply_otlp(
    req: &mut otlp_pb::trace_service::ExportTraceServiceRequest,
    pii_mode: &str,
    pii_disabled: &[String],
) -> PyResult<()> {
    match pii_mode {
        "off" => Ok(()),
        "mask" => {
            let engine = pii::engine_for(pii_disabled)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            pii::mask_otlp(&engine, req);
            Ok(())
        }
        other => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unsupported pii_mode {other:?} (expected \"mask\" or \"off\")"
        ))),
    }
}

#[pyfunction]
#[pyo3(signature = (envelope, pii_mode = "off", pii_disabled = Vec::new()))]
fn encode_envelope_py(
    py: Python<'_>,
    envelope: &Bound<'_, PyAny>,
    pii_mode: &str,
    pii_disabled: Vec<String>,
) -> PyResult<Py<PyBytes>> {
    // Marshalling walks Python objects — the only part that needs the GIL.
    let mut proto = envelope_to_proto(envelope)?;
    // Masking + protobuf + zstd are pure Rust: release the GIL so app threads
    // keep running while the batch worker encodes (design §9).
    let bytes = py.allow_threads(|| -> PyResult<Vec<u8>> {
        pii_apply_envelope(&mut proto, pii_mode, &pii_disabled)?;
        encode_envelope(&proto).map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    })?;
    Ok(PyBytes::new_bound(py, &bytes).unbind())
}

#[pyfunction]
fn decode_envelope_py(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    let env = decode_envelope(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    envelope_to_dict(py, &env)
}

#[pyfunction]
#[pyo3(signature = (envelope, pii_mode = "off", pii_disabled = Vec::new()))]
fn encode_otlp_traces(
    py: Python<'_>,
    envelope: &Bound<'_, PyAny>,
    pii_mode: &str,
    pii_disabled: Vec<String>,
) -> PyResult<Py<PyBytes>> {
    // Marshalling walks Python objects — the only part that needs the GIL.
    let mut req = envelope_to_otlp(envelope)?;
    // Masking + protobuf + zstd are pure Rust: release the GIL so app threads
    // keep running while the batch worker encodes (design §9).
    let bytes = py.allow_threads(|| -> PyResult<Vec<u8>> {
        pii_apply_otlp(&mut req, pii_mode, &pii_disabled)?;
        otlp::encode_traces(&req)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    })?;
    Ok(PyBytes::new_bound(py, &bytes).unbind())
}

#[pyfunction]
fn decode_otlp_traces(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    let req = otlp::decode_traces(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    otlp_traces_to_dict(py, &req)
}

/// Registers the `_wardex_native.codec` submodule.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "codec")?;
    m.add_function(wrap_pyfunction!(encode_envelope_py, &m)?)?;
    m.add_function(wrap_pyfunction!(decode_envelope_py, &m)?)?;
    m.add_function(wrap_pyfunction!(encode_otlp_traces, &m)?)?;
    m.add_function(wrap_pyfunction!(decode_otlp_traces, &m)?)?;
    // Exposed in Python as codec.encode_envelope / codec.decode_envelope
    m.add("encode_envelope", m.getattr("encode_envelope_py")?)?;
    m.add("decode_envelope", m.getattr("decode_envelope_py")?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
