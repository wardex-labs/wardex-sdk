//! PyO3 core marshaling — `InternalEnvelope`(Python) ↔ `pb::Envelope`(prost).
//!
//! Option 1 (Rust getattr traversal): walks the Python object directly to populate the proto struct.
//! Lossless round-trip including the transport tree, capture_integrity, correlation, and state_snapshots.

// In the trampoline code generated when the pyo3 #[pyfunction] macro wraps a function
// returning `PyResult<T>`, clippy mistakes the `?`'s `From<PyErr> for PyErr` (identity)
// conversion for a useless conversion
// (a pre-existing pyo3 0.22 issue; a function-level #[allow] can't cover macro-generated sibling items).
#![allow(clippy::useless_conversion)]

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList};

use wardex_core::codec::otlp::{self, otlp_pb};
use wardex_core::codec::proto::wardex::v1 as pb;
use wardex_core::codec::{decode_envelope, encode_envelope};
use wardex_core::pipeline::pii;

use crate::limits::PyLimits;

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
//
// Every one of these was a hand-written `match` from the Python enum value to
// the proto number — a second declaration of a list the `.proto` already owns,
// with nothing to make the compiler compare the two. `wardex_core::codec::vocab`
// derives the mapping from the schema instead (design §6.6), so the tables here
// are only about what to do when the schema has no such value.
//
// That decision is per-vocabulary and stays visible at the call site rather
// than buried in the mapper:
//
//   * `status_code` falls back to UNSET, which is a real "no status" value;
//   * everything else falls back to UNSPECIFIED, and for those the caller side
//     already refuses to produce an unlisted value (`SpanDraft` takes enums,
//     not strings), so the fallback is a backstop rather than a path;
//   * `limitation` alone has somewhere honest to put the fact — see
//     `integrity_to_proto`.

fn map_span_kind(s: &str) -> i32 {
    vocab::span_kind_to_proto(s).unwrap_or(pb::SpanKind::Unspecified as i32)
}
fn map_status_code(s: &str) -> i32 {
    vocab::status_code_to_proto(s).unwrap_or(pb::StatusCode::Unset as i32)
}
fn map_capture_source(s: &str) -> i32 {
    vocab::capture_source_to_proto(s).unwrap_or(pb::CaptureSource::Unspecified as i32)
}

// --- span vocabulary registry (design §6.2, §6.3, §6.6) ---
//
// proto is the single source of truth for the span vocabulary, and these
// functions are the mapping from the Python enum VALUE (`"execute_tool"`,
// `"ipc"`, `"triggered_by"`) to the declared proto number. Node and Java get
// the same table by generating from the same `.proto`, which is the whole point
// of declaring the enums: a second SDK must not re-derive the vocabulary from
// prose.
//
// `map_operation_name` and `map_tool_execution_type` intentionally fill no
// `Span` field in this schema version — those two values ride as `extra` keys
// (`gen_ai.operation.name`, `wardex.tool.execution_type`), and a typed field
// alongside would double-carry them and force a precedence rule. They are
// reachable, exercised and pinned through `vocabulary_tables()` below, which is
// also what lets a Python test assert the two vocabularies agree value by value
// instead of trusting that they do.

fn map_operation_name(s: &str) -> i32 {
    vocab::operation_name_to_proto(s).unwrap_or(pb::OperationName::Unspecified as i32)
}

fn map_tool_execution_type(s: &str) -> i32 {
    vocab::tool_execution_type_to_proto(s).unwrap_or(pb::ToolExecutionType::Unspecified as i32)
}

fn map_link_reason(s: &str) -> i32 {
    vocab::link_reason_to_proto(s).unwrap_or(pb::LinkReason::Unspecified as i32)
}

use wardex_core::codec::vocab::{self, link_reason_name};

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
    vocab::protocol_to_proto(s).unwrap_or(pb::Protocol::Unspecified as i32)
}
fn map_direction(s: &str) -> i32 {
    vocab::direction_to_proto(s).unwrap_or(pb::Direction::Unspecified as i32)
}
fn map_modality(s: &str) -> i32 {
    vocab::modality_to_proto(s).unwrap_or(pb::Modality::Unspecified as i32)
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

/// The key that carries a marker the schema could not name. Read the comment on
/// `UNMAPPED_LIMITATION_KEY`'s use in `integrity_to_proto` before changing it —
/// it is the only thing standing between a vocabulary gap and a silent drop.
const UNMAPPED_LIMITATION_KEY: &str = "wardex.limitation.unmapped";

/// `CaptureIntegrity` → proto, returning any marker the schema has no value for.
///
/// The second half of that return type is the point. `limitations` used to be
/// `repeated string`, so anything the Python side held reached the wire
/// verbatim; now it is a closed enum and a value with no proto counterpart has
/// nowhere to go. Dropping it would produce exactly the failure the vocabulary
/// exists to prevent — a span that reads as fully captured because the reason it
/// was not could not be spelled.
///
/// So an unmapped marker becomes `LIMITATION_VOCABULARY_UNMAPPED` (a meta value,
/// deliberately outside the vocabulary's number band) and the caller writes the
/// original string into `Span.extra`. The two together say "something was
/// wrong, here is what the sender called it" without pretending the schema knew.
///
/// This should be unreachable: `test_vocabulary.py` asserts the Python enum and
/// the proto enum agree member for member, so a gap fails CI before it ships.
/// It exists because "unreachable" is a property of today's build, and the
/// decode side of this same file has to survive an envelope written by a NEWER
/// SDK — where the gap is not a bug but the normal case.
fn integrity_to_proto(c: &Bound<PyAny>) -> PyResult<(pb::CaptureIntegrity, Vec<String>)> {
    let mut codes: Vec<i32> = Vec::new();
    let mut unmapped: Vec<String> = Vec::new();
    for m in c.getattr("limitations")?.iter()? {
        let m = m?;
        // A `Limitation` member normally; a bare string is still accepted so
        // this cannot become the reason a span dies at the boundary.
        let value: String = if m.hasattr("value")? {
            enum_str(&m)?
        } else {
            m.extract()?
        };
        match vocab::limitation_to_proto(&value) {
            Some(n) => codes.push(n),
            None => {
                codes.push(pb::Limitation::VocabularyUnmapped as i32);
                unmapped.push(value);
            }
        }
    }
    Ok((
        pb::CaptureIntegrity {
            request_headers_captured: c.getattr("request_headers_captured")?.extract()?,
            request_body_captured: c.getattr("request_body_captured")?.extract()?,
            response_headers_captured: c.getattr("response_headers_captured")?.extract()?,
            response_body_captured: c.getattr("response_body_captured")?.extract()?,
            redacted: c.getattr("redacted")?.extract()?,
            truncated: c.getattr("truncated")?.extract()?,
            dropped_chunk_count: c.getattr("dropped_chunk_count")?.extract()?,
            limitation_codes: codes,
        },
        unmapped,
    ))
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
        // `ParentSource` member or its value string. An unlisted value maps to
        // UNSPECIFIED, which is honest here in a way it would not be for
        // `Limitation`: "wardex does not claim to know how this parent was
        // derived" is a meaningful statement, and the confidence field beside it
        // already carries the trust level separately.
        let value: String = if v.hasattr("value")? {
            enum_str(&v)?
        } else {
            v.extract()?
        };
        corr.parent_source =
            vocab::parent_source_to_proto(&value).unwrap_or(pb::ParentSource::Unspecified as i32);
    }
    if let Some(v) = opt(c, "active_span_id_at_capture")? {
        corr.active_span_id_at_capture = id_bytes(&v)?;
    }
    Ok(corr)
}

/// Lifted out of `state_to_proto` so `vocabulary_tables()` can pin it beside
/// the other three registry enums. An unrecognized value still becomes
/// UNSPECIFIED here, but it can no longer arrive unnoticed: `SnapshotType` is
/// closed on the Python side too, and `SnapshotDraft` attaches
/// `Limitation.SNAPSHOT_TYPE_UNKNOWN` at the point of coercion instead of
/// letting this function flatten it in silence.
fn map_snapshot_type(v: &str) -> i32 {
    vocab::snapshot_type_to_proto(v).unwrap_or(pb::SnapshotType::Unspecified as i32)
}

fn state_to_proto(s: &Bound<PyAny>) -> PyResult<pb::StateSnapshot> {
    let mut ss = pb::StateSnapshot {
        trace_id: id_bytes(&s.getattr("trace_id")?)?,
        span_id: id_bytes(&s.getattr("span_id")?)?,
        timestamp_ns: s.getattr("timestamp_ns")?.extract()?,
        snapshot_type: map_snapshot_type(&s.getattr("snapshot_type")?.extract::<String>()?),
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
        let (integrity, unmapped) = integrity_to_proto(&c)?;
        span.capture_integrity = Some(integrity);
        // After the `extra` passthrough above, deliberately: a marker the schema
        // could not name is wardex's own note about this encode, not something
        // the caller supplied, and it must not be overwritten by a same-keyed
        // caller value. Repeats are allowed — `extra` is a repeated KeyValue,
        // and two unnameable markers are two facts.
        for value in unmapped {
            span.extra.push(pb::KeyValue {
                key: UNMAPPED_LIMITATION_KEY.to_owned(),
                value: Some(pb::AnyValue {
                    value: Some(pb::any_value::Value::StringValue(value)),
                }),
            });
        }
    }
    if let Some(c) = opt(sp, "correlation")? {
        span.correlation = Some(correlation_to_proto(&c)?);
    }
    events_to_proto(sp, &mut span.events)?;
    links_to_proto(sp, &mut span.links)?;
    Ok(span)
}

/// `InternalSpan.events` -> `Span.events` (tag 10).
///
/// This encoder did not exist. `Span.events` and `Span.links` were declared in
/// `span.proto` and never filled, so `_types.py`'s `events`/`links` tuples were
/// dropped whole at encode — and with them `InternalSpanLink.reason`, which
/// made `LinkReason` dead vocabulary and design §6.3's entire graph model
/// (TRIGGERED_BY edges, HANDOFF_FROM siblings, RESUMED_FROM across a
/// checkpoint) impossible to transmit at all. Declaring the enum without this
/// would have been theatre.
fn events_to_proto(sp: &Bound<PyAny>, out: &mut Vec<pb::SpanEvent>) -> PyResult<()> {
    for ev in sp.getattr("events")?.iter()? {
        let ev = ev?;
        let mut event = pb::SpanEvent {
            name: ev.getattr("name")?.extract()?,
            time_unix_nano: ev.getattr("timestamp_ns")?.extract()?,
            ..Default::default()
        };
        kv_list(&ev.getattr("attributes")?, &mut event.attributes)?;
        out.push(event);
    }
    Ok(())
}

/// `InternalSpan.links` -> `Span.links` (tag 11), including `reason` (tag 4).
///
/// `reason` is a closed `LinkReason` on the wire, and this boundary accepts it
/// either as that member or as its bare value string — the same duck-typed
/// shape `CaptureIntegrity.limitations` crosses on, so a host that hand-builds
/// an `InternalSpanLink` is not forced to import the enum.
/// The mapping is one-way lossy by construction — an unrecognized reason
/// becomes UNSPECIFIED — and that is bounded by `LinkReason` being closed and
/// `SpanDraft.add_link` taking the enum, so no caller inside the SDK can
/// produce a string that is not a member.
fn links_to_proto(sp: &Bound<PyAny>, out: &mut Vec<pb::SpanLink>) -> PyResult<()> {
    for ln in sp.getattr("links")?.iter()? {
        let ln = ln?;
        let mut link = pb::SpanLink {
            trace_id: id_bytes(&ln.getattr("trace_id")?)?,
            span_id: id_bytes(&ln.getattr("span_id")?)?,
            ..Default::default()
        };
        if let Some(r) = opt(&ln, "reason")? {
            let reason: String = if r.hasattr("value")? {
                enum_str(&r)?
            } else {
                r.extract()?
            };
            link.reason = map_link_reason(&reason);
        }
        out.push(link);
    }
    Ok(())
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
        // Numbers in, names out. Handing a consumer a raw `i32` would make it
        // re-derive the vocabulary from the schema by hand — the exact drift
        // §6.6 exists to stop — and an unrecognized number says so in its own
        // name rather than passing for "unset".
        cd.set_item(
            "limitations",
            c.limitation_codes
                .iter()
                .map(|n| vocab::limitation_name(*n))
                .collect::<Vec<_>>(),
        )?;
        d.set_item("capture_integrity", cd)?;
    }
    if let Some(c) = &sp.correlation {
        let cd = PyDict::new_bound(py);
        cd.set_item("strategy", vocab::parent_source_name(c.parent_source))?;
        cd.set_item("confidence", c.confidence)?;
        d.set_item("correlation", cd)?;
    }
    // The decode half of the events/links encoder above. Both directions land in
    // the same commit so the round-trip is testable: an encoder nobody can
    // decode is indistinguishable from no encoder at all.
    let events = PyList::empty_bound(py);
    for ev in &sp.events {
        let ed = PyDict::new_bound(py);
        ed.set_item("name", &ev.name)?;
        ed.set_item("timestamp_ns", ev.time_unix_nano)?;
        ed.set_item("attributes", kv_to_py(py, &ev.attributes)?)?;
        events.append(ed)?;
    }
    d.set_item("events", events)?;
    let links = PyList::empty_bound(py);
    for ln in &sp.links {
        let ld = PyDict::new_bound(py);
        ld.set_item("trace_id", PyBytes::new_bound(py, &ln.trace_id))?;
        ld.set_item("span_id", PyBytes::new_bound(py, &ln.span_id))?;
        // The wardex value string, not the raw i32 the rest of this function
        // still hands back for enums. A number is not a vocabulary: a consumer
        // reading `2` has to hold a copy of the enum to know what it means,
        // which is the drift §6.6 exists to prevent. (The other enums here
        // still hand back numbers; converting them is separate work.)
        ld.set_item("reason", link_reason_name(ln.reason))?;
        ld.set_item("attributes", kv_to_py(py, &ln.attributes)?)?;
        links.append(ld)?;
    }
    d.set_item("links", links)?;
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
        // Bytes pass through here untouched: masking must still see the raw
        // payload. `debyte_otlp` strips every bytes_value from the request
        // after masking, right before serialization (WAR-77).
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
fn otlp_kv_bool(key: &str, v: bool) -> otlp_pb::common::KeyValue {
    otlp_pb::common::KeyValue {
        key: key.into(),
        value: Some(otlp_pb::common::AnyValue {
            value: Some(otlp_pb::common::any_value::Value::BoolValue(v)),
        }),
    }
}
fn otlp_kv_f64(key: &str, v: f64) -> otlp_pb::common::KeyValue {
    otlp_pb::common::KeyValue {
        key: key.into(),
        value: Some(otlp_pb::common::AnyValue {
            value: Some(otlp_pb::common::any_value::Value::DoubleValue(v)),
        }),
    }
}
fn otlp_kv_strs(key: &str, vs: Vec<String>) -> otlp_pb::common::KeyValue {
    otlp_pb::common::KeyValue {
        key: key.into(),
        value: Some(otlp_pb::common::AnyValue {
            value: Some(otlp_pb::common::any_value::Value::ArrayValue(
                otlp_pb::common::ArrayValue {
                    values: vs
                        .into_iter()
                        .map(|v| otlp_pb::common::AnyValue {
                            value: Some(otlp_pb::common::any_value::Value::StringValue(v)),
                        })
                        .collect(),
                },
            )),
        }),
    }
}

/// `CorrelationInfo` and `CaptureIntegrity` -> OTLP span attributes.
///
/// OTLP has no native home for either, so they travel under `wardex.*` the same
/// way a link's `reason` does. Leaving them out is not a smaller version of the
/// same export -- it is the one that cannot be audited. Every marker this SDK
/// spends its design on says what it could NOT establish, and `OtlpHttpTransport`
/// is the only transport exported from the package root: a user on the
/// documented path was receiving spans stripped of every "this edge is a guess"
/// and every "this body was truncated", with nothing to distinguish them from
/// spans that had nothing to report. `events_to_otlp` names this exact failure
/// for a different field one screen below.
fn integrity_to_otlp(
    sp: &Bound<PyAny>,
    attrs: &mut Vec<otlp_pb::common::KeyValue>,
) -> PyResult<()> {
    if let Some(c) = opt(sp, "correlation")? {
        if let Some(src) = opt(&c, "strategy")? {
            attrs.push(otlp_kv_str("wardex.parent_source", &enum_str(&src)?));
        }
        attrs.push(otlp_kv_f64(
            "wardex.parent_confidence",
            c.getattr("confidence")?.extract()?,
        ));
        // The identifier that was CONSULTED to pick a parent. Present only when
        // one was, so a non-null value is actionable rather than decorative.
        for (field, key) in [
            ("request_id", "wardex.correlation.request_id"),
            ("operation_id", "wardex.correlation.operation_id"),
            ("attempt_id", "wardex.correlation.attempt_id"),
        ] {
            if let Some(v) = opt(&c, field)? {
                attrs.push(otlp_kv_str(key, &v.extract::<String>()?));
            }
        }
    }
    if let Some(i) = opt(sp, "capture_integrity")? {
        let mut markers: Vec<String> = Vec::new();
        for m in i.getattr("limitations")?.iter()? {
            markers.push(enum_str(&m?)?);
        }
        if !markers.is_empty() {
            attrs.push(otlp_kv_strs("wardex.limitations", markers));
        }
        for (field, key) in [
            ("request_headers_captured", "wardex.capture.request_headers"),
            ("request_body_captured", "wardex.capture.request_body"),
            (
                "response_headers_captured",
                "wardex.capture.response_headers",
            ),
            ("response_body_captured", "wardex.capture.response_body"),
        ] {
            attrs.push(otlp_kv_bool(key, i.getattr(field)?.extract()?));
        }
        // Emitted only when true / non-zero: unlike the four above, whose FALSE
        // is the informative reading, these describe an event that either
        // happened or did not.
        if i.getattr("truncated")?.extract()? {
            attrs.push(otlp_kv_bool("wardex.capture.truncated", true));
        }
        if i.getattr("redacted")?.extract()? {
            attrs.push(otlp_kv_bool("wardex.capture.redacted", true));
        }
        let dropped: i64 = i.getattr("dropped_chunk_count")?.extract()?;
        if dropped > 0 {
            attrs.push(otlp_kv_int("wardex.capture.dropped_chunks", dropped));
        }
    }
    Ok(())
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
    // raw I/O → wardex.input_data / wardex.output_data (omitted if empty).
    // Built as bytes so PII masking sees the raw payload; `debyte_otlp`
    // converts to strings after masking, before serialization (WAR-77).
    let input: Vec<u8> = sp.getattr("input_data")?.extract()?;
    if !input.is_empty() {
        attrs.push(otlp_kv_bytes("wardex.input_data", input));
    }
    let output: Vec<u8> = sp.getattr("output_data")?.extract()?;
    if !output.is_empty() {
        attrs.push(otlp_kv_bytes("wardex.output_data", output));
    }
    integrity_to_otlp(sp, &mut attrs)?;
    span.attributes = attrs;
    events_to_otlp(sp, &mut span.events)?;
    links_to_otlp(sp, &mut span.links)?;
    Ok(span)
}

/// `InternalSpan.events` -> OTLP `Span.events` (tag 11).
///
/// The envelope encoder is not enough on its own. OTLP is the surface that
/// actually leaves the process today (`transport/_otlp_http.py`), so filling
/// `Span.events`/`Span.links` on the wardex envelope and not here would leave
/// the two encoders disagreeing about the same span — and a user configured for
/// the OTLP exporter would lose §6.3's whole graph model with no counter, no
/// `Limitation` marker and no failing test, indistinguishable from "this agent
/// has no graph edges". That is the silent-loss shape I4 forbids.
fn events_to_otlp(sp: &Bound<PyAny>, out: &mut Vec<otlp_pb::trace::span::Event>) -> PyResult<()> {
    for ev in sp.getattr("events")?.iter()? {
        let ev = ev?;
        let mut wkv: Vec<pb::KeyValue> = Vec::new();
        kv_list(&ev.getattr("attributes")?, &mut wkv)?;
        out.push(otlp_pb::trace::span::Event {
            time_unix_nano: ev.getattr("timestamp_ns")?.extract()?,
            name: ev.getattr("name")?.extract()?,
            attributes: wkv.iter().map(wardex_kv_to_otlp).collect(),
            ..Default::default()
        });
    }
    Ok(())
}

/// `InternalSpan.links` -> OTLP `Span.links` (tag 13).
///
/// `reason` has no OTLP-native home — `Link` carries `trace_state` and
/// attributes and nothing else — so it travels as the `wardex.link.reason`
/// attribute rather than being dropped. Emitting the wardex value string keeps
/// it readable without a copy of the enum, which is the same argument §6.6
/// makes for not shipping raw `i32`s.
///
/// `InternalSpanLink` carries no attributes of its own (`_types.py:235`), so
/// `reason` is the only one there is to carry.
fn links_to_otlp(sp: &Bound<PyAny>, out: &mut Vec<otlp_pb::trace::span::Link>) -> PyResult<()> {
    for ln in sp.getattr("links")?.iter()? {
        let ln = ln?;
        let mut attributes: Vec<otlp_pb::common::KeyValue> = Vec::new();
        if let Some(r) = opt(&ln, "reason")? {
            let reason: String = if r.hasattr("value")? {
                enum_str(&r)?
            } else {
                r.extract()?
            };
            if !reason.is_empty() {
                attributes.push(otlp_kv_str("wardex.link.reason", &reason));
            }
        }
        out.push(otlp_pb::trace::span::Link {
            trace_id: id_bytes(&ln.getattr("trace_id")?)?,
            span_id: id_bytes(&ln.getattr("span_id")?)?,
            attributes,
            ..Default::default()
        });
    }
    Ok(())
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
        // A list attribute decoded as `None` is not a smaller answer, it is a
        // wrong one: the encoder wrote the values and the reader reports the
        // key as unset. `limitations` is the first array this SDK sends and it
        // is exactly the field a consumer checks to decide whether to trust a
        // span, so silence here would be indistinguishable from "nothing to
        // report".
        Some(Value::ArrayValue(a)) => {
            PyList::new_bound(py, a.values.iter().map(|e| otlp_any_to_py(py, e))).into_py(py)
        }
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
                // The decode half of `events_to_otlp`/`links_to_otlp`. An
                // encoder nobody can decode is indistinguishable from no
                // encoder, which is how these two fields went missing on this
                // surface in the first place.
                let events = PyList::empty_bound(py);
                for ev in &sp.events {
                    let ed = PyDict::new_bound(py);
                    ed.set_item("name", &ev.name)?;
                    ed.set_item("time_unix_nano", ev.time_unix_nano)?;
                    ed.set_item("attributes", otlp_attrs_to_py(py, &ev.attributes)?)?;
                    events.append(ed)?;
                }
                spd.set_item("events", events)?;
                let links = PyList::empty_bound(py);
                for ln in &sp.links {
                    let ld = PyDict::new_bound(py);
                    ld.set_item("trace_id", to_hex(&ln.trace_id))?;
                    ld.set_item("span_id", to_hex(&ln.span_id))?;
                    ld.set_item("attributes", otlp_attrs_to_py(py, &ln.attributes)?)?;
                    links.append(ld)?;
                }
                spd.set_item("links", links)?;
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
#[pyo3(signature = (envelope, pii_mode = "off", pii_disabled = Vec::new(), limits = None))]
fn encode_envelope_py(
    py: Python<'_>,
    envelope: &Bound<'_, PyAny>,
    pii_mode: &str,
    pii_disabled: Vec<String>,
    limits: Option<PyLimits>,
) -> PyResult<Py<PyBytes>> {
    // `zstd_level` is the only limit the codec reads, and it must come from
    // the caller's resolved limits: hardcoding the default here would let a
    // configured level be validated and then silently discarded.
    let limits = limits.map(|p| p.inner).unwrap_or_default();
    // Marshalling walks Python objects — the only part that needs the GIL.
    let mut proto = envelope_to_proto(envelope)?;
    // Masking + protobuf + zstd are pure Rust: release the GIL so app threads
    // keep running while the batch worker encodes (design §9).
    let bytes = py.allow_threads(|| -> PyResult<Vec<u8>> {
        pii_apply_envelope(&mut proto, pii_mode, &pii_disabled)?;
        encode_envelope(&proto, limits)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
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
        // After masking, never before: the PII engine's byte-level patterns
        // match inside raw payloads, and a payload already rewritten to
        // base64 would hide them (WAR-77).
        debyte_otlp(&mut req);
        otlp::encode_traces(&req)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    })?;
    Ok(PyBytes::new_bound(py, &bytes).unbind())
}

// --- WAR-77: no bytes_value ever leaves on the OTLP surface ---
//
// OTLP `bytes_value` is legal per spec, but backends that re-serialize
// attributes to JSON can't represent it: Arize Phoenix (2026-08) drops the
// entire span at ingest — HTTP 200, no error — and the spans lost are exactly
// the LLM/tool ones that carry payloads. The wardex envelope keeps raw bytes;
// this surface degrades to strings: valid UTF-8 verbatim, anything else base64
// plus a `<key>.encoding = "base64"` companion so a consumer can tell encoded
// binary from text that merely looks like base64.

/// Strip every `bytes_value` from an OTLP export request, in place.
fn debyte_otlp(req: &mut otlp_pb::trace_service::ExportTraceServiceRequest) {
    for rs in &mut req.resource_spans {
        if let Some(r) = rs.resource.as_mut() {
            debyte_otlp_kvs(&mut r.attributes);
        }
        for ss in &mut rs.scope_spans {
            if let Some(sc) = ss.scope.as_mut() {
                debyte_otlp_kvs(&mut sc.attributes);
            }
            for sp in &mut ss.spans {
                debyte_otlp_kvs(&mut sp.attributes);
                for ev in &mut sp.events {
                    debyte_otlp_kvs(&mut ev.attributes);
                }
                for link in &mut sp.links {
                    debyte_otlp_kvs(&mut link.attributes);
                }
            }
        }
    }
}

fn debyte_otlp_kvs(kvs: &mut Vec<otlp_pb::common::KeyValue>) {
    // (companion key, went_base64) for every attribute whose value WAS bytes.
    let mut rewritten: Vec<(String, bool)> = Vec::new();
    for kv in kvs.iter_mut() {
        if let Some(v) = kv.value.as_mut() {
            if let Some(went_base64) = debyte_otlp_any(v) {
                rewritten.push((format!("{}.encoding", kv.key), went_base64));
            }
        }
    }
    if rewritten.is_empty() {
        return;
    }
    // For a key this pass rewrote, `<key>.encoding` is this pass's namespace.
    // A pre-existing attribute there (a user extra — kv_list passes any key
    // through) would either duplicate the companion (duplicate keys are
    // undefined in OTLP, backend dedup order decides which wins) or spoof an
    // encoding the verbatim branch never applied, making consumers
    // base64-decode text that shipped as-is. Drop it either way; user
    // `.encoding` suffixes on keys that never carried bytes are untouched.
    kvs.retain(|kv| !rewritten.iter().any(|(companion, _)| kv.key == *companion));
    for (companion, went_base64) in rewritten {
        if went_base64 {
            kvs.push(otlp_kv_str(&companion, "base64"));
        }
    }
}

/// Text safe for every real backend, or None → base64. Strict UTF-8 alone is
/// not enough: U+0000 is valid UTF-8, and Postgres-backed ingests reject any
/// string containing NUL — the same silent span loss this pass exists to
/// prevent, reintroduced for the NUL subset. Binary protobuf/gRPC payloads
/// are full of NULs and are exactly what must route to base64, so any C0
/// control byte other than \t \n \r means "not text".
fn otlp_text(b: &[u8]) -> Option<&str> {
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
fn debyte_otlp_any(v: &mut otlp_pb::common::AnyValue) -> Option<bool> {
    use otlp_pb::common::any_value::Value;
    match v.value.as_mut() {
        Some(Value::BytesValue(b)) => {
            let raw = std::mem::take(b);
            let (s, went_base64) = match otlp_text(&raw) {
                Some(s) => (s.to_owned(), false),
                None => (BASE64.encode(&raw), true),
            };
            v.value = Some(Value::StringValue(s));
            Some(went_base64)
        }
        Some(Value::ArrayValue(arr)) => {
            for item in &mut arr.values {
                debyte_otlp_any(item);
            }
            None
        }
        Some(Value::KvlistValue(kvl)) => {
            debyte_otlp_kvs(&mut kvl.values);
            None
        }
        _ => None,
    }
}

#[pyfunction]
fn decode_otlp_traces(py: Python<'_>, data: &[u8]) -> PyResult<PyObject> {
    let req = otlp::decode_traces(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    otlp_traces_to_dict(py, &req)
}

/// The declared span vocabulary, as `{enum name: {wardex value: proto number}}`.
///
/// This is what makes "proto is the single source of truth" (§6.6) checkable
/// rather than aspirational. `OperationName` and `ToolExecutionType` are
/// declared in `common.proto` and referenced by no message field in this schema
/// version — deliberately, because both already travel as `Span.extra` keys and
/// a typed field would double-carry them. Without an export like this the two
/// registry enums would be unreachable from any test, `map_operation_name` and
/// `map_tool_execution_type` would be dead code, and the Python enums could
/// drift from the proto ones with nothing to notice.
///
/// The tables are keyed by the WARDEX value (`"execute_tool"`), not the proto
/// value name, because the wardex value is what actually reaches the wire today
/// in `extra` — so a mismatch here is a mismatch a consumer would see.
#[pyfunction]
fn vocabulary_tables(py: Python<'_>) -> PyResult<PyObject> {
    let out = PyDict::new_bound(py);

    let ops = PyDict::new_bound(py);
    for name in [
        "chat",
        "text_completion",
        "embeddings",
        "execute_tool",
        "create_agent",
        "invoke_agent",
        "invoke_workflow",
        "generate_content",
        "retrieval",
        "execute_step",
        "handoff",
        "evaluate",
    ] {
        ops.set_item(name, map_operation_name(name))?;
    }
    out.set_item("OperationName", ops)?;

    let tools = PyDict::new_bound(py);
    for name in ["network", "in_process", "ipc", "unknown"] {
        tools.set_item(name, map_tool_execution_type(name))?;
    }
    out.set_item("ToolExecutionType", tools)?;

    let links = PyDict::new_bound(py);
    for name in [
        "triggered_by",
        "handoff_from",
        "resumed_from",
        "retried_from",
        "cache_source",
    ] {
        links.set_item(name, map_link_reason(name))?;
    }
    out.set_item("LinkReason", links)?;

    let snaps = PyDict::new_bound(py);
    for name in ["span_start", "span_end", "turn_start"] {
        snaps.set_item(name, map_snapshot_type(name))?;
    }
    out.set_item("SnapshotType", snaps)?;

    // The two closed vocabularies that live on the wire, exposed the OTHER way
    // round — number → name, walking the schema rather than a list written
    // here. A table keyed by hand would only prove that this file agrees with
    // itself; walking the numbers lets a Python test compare the SCHEMA against
    // `assembly._integrity.Limitation` and `assembly._parentage.ParentSource`
    // member for member, in both directions, which is what makes a missing
    // member a CI failure instead of a silently unnameable span.
    let limits_tbl = PyDict::new_bound(py);
    for n in 1..=200 {
        let name = vocab::limitation_name(n);
        if !name.contains("unrecognized") {
            limits_tbl.set_item(name, n)?;
        }
    }
    out.set_item("Limitation", limits_tbl)?;

    let sources = PyDict::new_bound(py);
    for n in 1..=200 {
        let name = vocab::parent_source_name(n);
        if !name.contains("unrecognized") {
            sources.set_item(name, n)?;
        }
    }
    out.set_item("ParentSource", sources)?;

    // The meta value, kept OUT of the vocabulary table above on purpose — a
    // consumer iterating "the vocabulary" must not find it there — but exposed
    // so a test can pin both its number and the fact that it is not vocabulary.
    let meta = PyDict::new_bound(py);
    let unmapped = pb::Limitation::VocabularyUnmapped as i32;
    meta.set_item(vocab::limitation_name(unmapped), unmapped)?;
    out.set_item("LimitationMeta", meta)?;

    Ok(out.into_py(py))
}

/// Registers the `_wardex_native.codec` submodule.
pub fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "codec")?;
    m.add_function(wrap_pyfunction!(encode_envelope_py, &m)?)?;
    m.add_function(wrap_pyfunction!(decode_envelope_py, &m)?)?;
    m.add_function(wrap_pyfunction!(encode_otlp_traces, &m)?)?;
    m.add_function(wrap_pyfunction!(decode_otlp_traces, &m)?)?;
    m.add_function(wrap_pyfunction!(vocabulary_tables, &m)?)?;
    // Exposed in Python as codec.encode_envelope / codec.decode_envelope
    m.add("encode_envelope", m.getattr("encode_envelope_py")?)?;
    m.add("decode_envelope", m.getattr("decode_envelope_py")?)?;
    parent.add_submodule(&m)?;
    Ok(())
}
