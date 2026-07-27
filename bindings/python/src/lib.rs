//! PyO3 entry point for the `_wardex_native` extension module.

mod codec;
mod limits;

use limits::PyLimits;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use pyo3::wrap_pyfunction;
use wardex_core::protocol::claude_stream_json as ccs;
use wardex_core::protocol::grpc::{
    grpc_status_name as core_grpc_status_name, parse_grpc_frames as core_parse_grpc_frames,
    GrpcFrames as CoreGrpcFrames, GrpcMessage as CoreGrpcMessage,
};
use wardex_core::protocol::http1::{Http1Stream, ParsedHttp};
use wardex_core::protocol::http2::{Http2Connection, Http2Transaction as CoreHttp2Txn};
use wardex_core::protocol::json_rpc::{JsonRpcKind, JsonRpcMessage as CoreJsonRpc, JsonRpcStream};
use wardex_core::protocol::semantic::{parse_llm, LlmSemantics as CoreLlmSemantics};
use wardex_core::protocol::websocket::{
    WsFeedResult as CoreWsFeedResult, WsFrame as CoreWsFrame, WsParser as CoreWsParser,
};

/// A single parsed HTTP message (for Python exposure).
#[pyclass]
struct RawHttpMessage {
    inner: ParsedHttp,
}

#[pymethods]
impl RawHttpMessage {
    #[getter]
    fn is_request(&self) -> bool {
        self.inner.is_request
    }
    #[getter]
    fn method(&self) -> Option<String> {
        self.inner.method.clone()
    }
    #[getter]
    fn path(&self) -> Option<String> {
        self.inner.path.clone()
    }
    #[getter]
    fn version(&self) -> u8 {
        self.inner.version
    }
    #[getter]
    fn status(&self) -> Option<u16> {
        self.inner.status
    }
    #[getter]
    fn headers(&self) -> Vec<(String, String)> {
        self.inner.headers.clone()
    }
    #[getter]
    fn body<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.body)
    }
    #[getter]
    fn truncated(&self) -> bool {
        self.inner.truncated
    }
    #[getter]
    fn header_len(&self) -> usize {
        self.inner.header_len
    }
    /// Capture-limitation markers the parser attached to this message (e.g.
    /// `body_cap_exceeded`). Without this the host SDK cannot report that a
    /// body was capped, and HTTP/1 would truncate silently while HTTP/2 does
    /// not.
    #[getter]
    fn limitations(&self) -> Vec<&'static str> {
        self.inner.limitations.clone()
    }
}

/// A completed HTTP/2 transaction (for Python exposure).
#[pyclass]
struct Http2Transaction {
    inner: CoreHttp2Txn,
}

#[pymethods]
impl Http2Transaction {
    #[getter]
    fn stream_id(&self) -> u32 {
        self.inner.stream_id
    }
    #[getter]
    fn method(&self) -> String {
        self.inner.method.clone()
    }
    #[getter]
    fn path(&self) -> String {
        self.inner.path.clone()
    }
    #[getter]
    fn status(&self) -> u16 {
        self.inner.status
    }
    #[getter]
    fn request_body<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.request_body)
    }
    #[getter]
    fn response_body<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new_bound(py, &self.inner.response_body)
    }
    #[getter]
    fn truncated(&self) -> bool {
        self.inner.truncated
    }
    #[getter]
    fn content_type(&self) -> Option<String> {
        self.inner.content_type.clone()
    }
    #[getter]
    fn grpc_status(&self) -> Option<i32> {
        self.inner.grpc_status
    }
    #[getter]
    fn grpc_message(&self) -> Option<String> {
        self.inner.grpc_message.clone()
    }
}

/// Incremental parser for a single h2 connection.
#[pyclass]
struct Http2Parser {
    inner: Http2Connection,
}

#[pymethods]
impl Http2Parser {
    #[new]
    #[pyo3(signature = (limits=None))]
    fn new(limits: Option<PyLimits>) -> Self {
        let l = limits.map(|p| p.inner).unwrap_or_default();
        Self {
            inner: Http2Connection::new(l),
        }
    }

    fn feed(&mut self, from_client: bool, data: &[u8]) -> (Vec<u32>, Vec<Http2Transaction>) {
        let r = self.inner.feed(from_client, data);
        let txns = r
            .transactions
            .into_iter()
            .map(|inner| Http2Transaction { inner })
            .collect();
        (r.opened_request_streams, txns)
    }
}

/// Incremental HTTP/1.x parser for one direction of a connection.
#[pyclass]
struct Http1Parser {
    inner: Http1Stream,
}

#[pymethods]
impl Http1Parser {
    #[new]
    #[pyo3(signature = (is_request, limits=None))]
    fn new(is_request: bool, limits: Option<PyLimits>) -> Self {
        let l = limits.map(|p| p.inner).unwrap_or_default();
        Self {
            inner: Http1Stream::new(is_request, l),
        }
    }

    fn feed(&mut self, data: &[u8]) -> Vec<RawHttpMessage> {
        self.inner
            .feed(data)
            .into_iter()
            .map(|inner| RawHttpMessage { inner })
            .collect()
    }

    fn flush_truncated(&mut self) -> Option<RawHttpMessage> {
        self.inner
            .flush_truncated()
            .map(|inner| RawHttpMessage { inner })
    }

    /// Why the parser latched off, if it did. `None` while the stream is
    /// still parsing normally.
    fn disabled_reason(&self) -> Option<&'static str> {
        self.inner.disabled_reason()
    }
}

/// LLM body semantic extraction result (wrapper around core LlmSemantics).
#[pyclass]
struct LlmSemantics {
    inner: CoreLlmSemantics,
}

#[pymethods]
impl LlmSemantics {
    #[getter]
    fn provider(&self) -> String {
        self.inner.provider.clone()
    }
    #[getter]
    fn operation(&self) -> String {
        self.inner.operation.clone()
    }
    #[getter]
    fn request_model(&self) -> Option<String> {
        self.inner.request_model.clone()
    }
    #[getter]
    fn response_model(&self) -> Option<String> {
        self.inner.response_model.clone()
    }
    #[getter]
    fn response_id(&self) -> Option<String> {
        self.inner.response_id.clone()
    }
    #[getter]
    fn input_tokens(&self) -> Option<i64> {
        self.inner.input_tokens
    }
    #[getter]
    fn output_tokens(&self) -> Option<i64> {
        self.inner.output_tokens
    }
    #[getter]
    fn cache_read_input_tokens(&self) -> Option<i64> {
        self.inner.cache_read_input_tokens
    }
    #[getter]
    fn cache_creation_input_tokens(&self) -> Option<i64> {
        self.inner.cache_creation_input_tokens
    }
    #[getter]
    fn reasoning_output_tokens(&self) -> Option<i64> {
        self.inner.reasoning_output_tokens
    }
    #[getter]
    fn temperature(&self) -> Option<f64> {
        self.inner.temperature
    }
    #[getter]
    fn top_p(&self) -> Option<f64> {
        self.inner.top_p
    }
    #[getter]
    fn top_k(&self) -> Option<f64> {
        self.inner.top_k
    }
    #[getter]
    fn frequency_penalty(&self) -> Option<f64> {
        self.inner.frequency_penalty
    }
    #[getter]
    fn presence_penalty(&self) -> Option<f64> {
        self.inner.presence_penalty
    }
    #[getter]
    fn max_tokens(&self) -> Option<i64> {
        self.inner.max_tokens
    }
    #[getter]
    fn seed(&self) -> Option<i64> {
        self.inner.seed
    }
    #[getter]
    fn choice_count(&self) -> Option<i64> {
        self.inner.choice_count
    }
    #[getter]
    fn stop_sequences(&self) -> Option<Vec<String>> {
        self.inner.stop_sequences.clone()
    }
    #[getter]
    fn stream(&self) -> Option<bool> {
        self.inner.stream
    }
    #[getter]
    fn finish_reasons(&self) -> Option<Vec<String>> {
        self.inner.finish_reasons.clone()
    }
    #[getter]
    fn output_type(&self) -> Option<String> {
        self.inner.output_type.clone()
    }
    #[getter]
    fn decoded_response<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .decoded_response
            .as_ref()
            .map(|b| PyBytes::new_bound(py, b))
    }
    #[getter]
    fn reassembled_from_stream(&self) -> bool {
        self.inner.reassembled_from_stream
    }
    #[getter]
    fn output_messages(&self) -> Option<String> {
        self.inner.output_messages.clone()
    }
    #[getter]
    fn tool_args_unparsed(&self) -> bool {
        self.inner.tool_args_unparsed
    }
    #[getter]
    fn output_messages_has_unmapped(&self) -> bool {
        self.inner.output_messages_has_unmapped
    }
    #[getter]
    fn input_messages(&self) -> Option<String> {
        self.inner.input_messages.clone()
    }
    #[getter]
    fn system_instructions(&self) -> Option<String> {
        self.inner.system_instructions.clone()
    }
    #[getter]
    fn input_messages_has_unmapped(&self) -> bool {
        self.inner.input_messages_has_unmapped
    }
}

/// A single parsed JSON-RPC message (for Python exposure).
#[pyclass]
struct JsonRpcMessage {
    inner: CoreJsonRpc,
}

#[pymethods]
impl JsonRpcMessage {
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            JsonRpcKind::Request => "request",
            JsonRpcKind::Response => "response",
            JsonRpcKind::Notification => "notification",
        }
    }
    #[getter]
    fn id(&self) -> Option<String> {
        self.inner.id.clone()
    }
    #[getter]
    fn method(&self) -> Option<String> {
        self.inner.method.clone()
    }
    #[getter]
    fn params<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .params
            .as_ref()
            .map(|b| PyBytes::new_bound(py, b))
    }
    #[getter]
    fn result<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .result
            .as_ref()
            .map(|b| PyBytes::new_bound(py, b))
    }
    #[getter]
    fn error<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner.error.as_ref().map(|b| PyBytes::new_bound(py, b))
    }
}

/// Incremental JSON-RPC parser for a one-directional byte stream.
#[pyclass]
struct JsonRpcParser {
    inner: JsonRpcStream,
}

#[pymethods]
impl JsonRpcParser {
    #[new]
    #[pyo3(signature = (limits=None))]
    fn new(limits: Option<PyLimits>) -> Self {
        let l = limits.map(|p| p.inner).unwrap_or_default();
        Self {
            inner: JsonRpcStream::new(l),
        }
    }

    fn feed(&mut self, data: &[u8]) -> Vec<JsonRpcMessage> {
        self.inner
            .feed(data)
            .into_iter()
            .map(|inner| JsonRpcMessage { inner })
            .collect()
    }

    /// Why the parser latched off, if it did. `None` while the stream is
    /// still parsing normally.
    fn disabled_reason(&self) -> Option<&'static str> {
        self.inner.disabled_reason()
    }

    /// Bytes currently held awaiting a newline.
    fn buffered_len(&self) -> usize {
        self.inner.buffered_len()
    }
}

/// Metadata for a single gRPC message (for Python exposure).
#[pyclass]
struct GrpcMessage {
    inner: CoreGrpcMessage,
}

#[pymethods]
impl GrpcMessage {
    #[getter]
    fn compressed(&self) -> bool {
        self.inner.compressed
    }
    #[getter]
    fn length(&self) -> u32 {
        self.inner.length
    }
}

/// gRPC framing result for one direction's body (for Python exposure).
#[pyclass]
struct GrpcFrames {
    inner: CoreGrpcFrames,
}

#[pymethods]
impl GrpcFrames {
    #[getter]
    fn messages(&self) -> Vec<GrpcMessage> {
        self.inner
            .messages
            .iter()
            .map(|m| GrpcMessage { inner: m.clone() })
            .collect()
    }
    #[getter]
    fn truncated(&self) -> bool {
        self.inner.truncated
    }
}

/// A single WS frame (for Python exposure).
#[pyclass]
struct WsFrame {
    inner: CoreWsFrame,
}

#[pymethods]
impl WsFrame {
    #[getter]
    fn fin(&self) -> bool {
        self.inner.fin
    }
    #[getter]
    fn opcode(&self) -> &'static str {
        self.inner.opcode.as_str()
    }
    #[getter]
    fn masked(&self) -> bool {
        self.inner.masked
    }
    #[getter]
    fn payload_len(&self) -> u64 {
        self.inner.payload_len
    }
    #[getter]
    fn close_code(&self) -> Option<u16> {
        self.inner.close_code
    }
}

/// Result of a single feed call (for Python exposure).
#[pyclass]
struct WsFeedResult {
    inner: CoreWsFeedResult,
}

#[pymethods]
impl WsFeedResult {
    #[getter]
    fn frames(&self) -> Vec<WsFrame> {
        self.inner
            .frames
            .iter()
            .map(|f| WsFrame { inner: f.clone() })
            .collect()
    }
    #[getter]
    fn messages<'py>(&self, py: Python<'py>) -> Vec<Bound<'py, PyBytes>> {
        self.inner
            .messages
            .iter()
            .map(|m| PyBytes::new_bound(py, m))
            .collect()
    }
}

/// Incremental WS stream parser for one direction (for Python exposure).
#[pyclass]
struct WsParser {
    inner: CoreWsParser,
}

#[pymethods]
impl WsParser {
    #[new]
    #[pyo3(signature = (limits=None))]
    fn new(limits: Option<PyLimits>) -> Self {
        let l = limits.map(|p| p.inner).unwrap_or_default();
        Self {
            inner: CoreWsParser::new(l),
        }
    }
    fn feed(&mut self, data: &[u8]) -> WsFeedResult {
        WsFeedResult {
            inner: self.inner.feed(data),
        }
    }
    fn is_disabled(&self) -> bool {
        self.inner.is_disabled()
    }
}

/// One parsed claude stream-json event (flat; kind discriminates).
#[pyclass]
struct ClaudeStreamEvent {
    inner: ccs::ClaudeStreamEvent,
}

#[pymethods]
impl ClaudeStreamEvent {
    #[getter]
    fn kind(&self) -> &'static str {
        match self.inner.kind {
            ccs::EventKind::SessionInit => "session_init",
            ccs::EventKind::UserPrompt => "user_prompt",
            ccs::EventKind::AssistantTurn => "assistant_turn",
            ccs::EventKind::ToolResult => "tool_result",
            ccs::EventKind::StreamDelta => "stream_delta",
            ccs::EventKind::TaskLifecycle => "task_lifecycle",
            ccs::EventKind::SessionResult => "session_result",
        }
    }
    #[getter]
    fn session_id(&self) -> Option<String> {
        self.inner.session_id.clone()
    }
    #[getter]
    fn model(&self) -> Option<String> {
        self.inner.model.clone()
    }
    #[getter]
    fn message_id(&self) -> Option<String> {
        self.inner.message_id.clone()
    }
    #[getter]
    fn stop_reason(&self) -> Option<String> {
        self.inner.stop_reason.clone()
    }
    #[getter]
    fn parent_tool_use_id(&self) -> Option<String> {
        self.inner.parent_tool_use_id.clone()
    }
    #[getter]
    fn subtype(&self) -> Option<String> {
        self.inner.subtype.clone()
    }
    #[getter]
    fn task_id(&self) -> Option<String> {
        self.inner.task_id.clone()
    }
    #[getter]
    fn task_status(&self) -> Option<String> {
        self.inner.task_status.clone()
    }
    #[getter]
    fn task_tool_use_id(&self) -> Option<String> {
        self.inner.task_tool_use_id.clone()
    }
    #[getter]
    fn content_json<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyBytes>> {
        self.inner
            .content_json
            .as_ref()
            .map(|b| PyBytes::new_bound(py, b))
    }
    #[getter]
    fn tool_uses(&self) -> Vec<(String, String, Vec<u8>)> {
        self.inner
            .tool_uses
            .iter()
            .map(|t| (t.id.clone(), t.name.clone(), t.input_json.clone()))
            .collect()
    }
    #[getter]
    fn input_tokens(&self) -> Option<i64> {
        self.inner.usage.as_ref().and_then(|u| u.input_tokens)
    }
    #[getter]
    fn output_tokens(&self) -> Option<i64> {
        self.inner.usage.as_ref().and_then(|u| u.output_tokens)
    }
    #[getter]
    fn cache_read_tokens(&self) -> Option<i64> {
        self.inner
            .usage
            .as_ref()
            .and_then(|u| u.cache_read_input_tokens)
    }
    #[getter]
    fn cache_creation_tokens(&self) -> Option<i64> {
        self.inner
            .usage
            .as_ref()
            .and_then(|u| u.cache_creation_input_tokens)
    }
    #[getter]
    fn num_turns(&self) -> Option<i64> {
        self.inner.num_turns
    }
    #[getter]
    fn total_cost_usd(&self) -> Option<f64> {
        self.inner.total_cost_usd
    }
    #[getter]
    fn duration_ms(&self) -> Option<i64> {
        self.inner.duration_ms
    }
    #[getter]
    fn duration_api_ms(&self) -> Option<i64> {
        self.inner.duration_api_ms
    }
    #[getter]
    fn is_error(&self) -> bool {
        self.inner.is_error
    }
}

/// Parse one stream-json line. Returns None for unknown/non-semantic lines.
#[pyfunction]
fn parse_claude_stream_line(data: &[u8], outbound: bool) -> Option<ClaudeStreamEvent> {
    ccs::parse_stream_line(data, outbound).map(|inner| ClaudeStreamEvent { inner })
}

#[pyfunction]
fn parse_grpc_frames(body: &[u8]) -> GrpcFrames {
    GrpcFrames {
        inner: core_parse_grpc_frames(body),
    }
}

#[pyfunction]
fn grpc_status_name(code: i32) -> &'static str {
    core_grpc_status_name(code)
}

#[pyfunction]
#[pyo3(signature = (host, path, req, resp, limits=None))]
fn parse_llm_semantics(
    host: &str,
    path: &str,
    req: &[u8],
    resp: &[u8],
    limits: Option<PyLimits>,
) -> Option<LlmSemantics> {
    let l = limits.map(|p| p.inner).unwrap_or_default();
    parse_llm(host, path, req, resp, l).map(|inner| LlmSemantics { inner })
}

#[pymodule]
fn _wardex_native(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    limits::register(m)?;

    let protocol = PyModule::new_bound(py, "protocol")?;
    protocol.add_class::<Http1Parser>()?;
    protocol.add_class::<RawHttpMessage>()?;
    protocol.add_class::<Http2Parser>()?;
    protocol.add_class::<Http2Transaction>()?;
    protocol.add_class::<LlmSemantics>()?;
    protocol.add_class::<JsonRpcParser>()?;
    protocol.add_class::<JsonRpcMessage>()?;
    protocol.add_function(wrap_pyfunction!(parse_llm_semantics, &protocol)?)?;
    protocol.add_class::<WsParser>()?;
    protocol.add_class::<WsFeedResult>()?;
    protocol.add_class::<WsFrame>()?;
    protocol.add_class::<GrpcMessage>()?;
    protocol.add_class::<GrpcFrames>()?;
    protocol.add_function(wrap_pyfunction!(parse_grpc_frames, &protocol)?)?;
    protocol.add_function(wrap_pyfunction!(grpc_status_name, &protocol)?)?;
    protocol.add_class::<ClaudeStreamEvent>()?;
    protocol.add_function(wrap_pyfunction!(parse_claude_stream_line, &protocol)?)?;
    m.add_submodule(&protocol)?;
    // Register in sys.modules so that `import wardex_sdk._wardex_native.protocol` works
    py.import_bound("sys")?
        .getattr("modules")?
        .set_item("wardex_sdk._wardex_native.protocol", &protocol)?;

    codec::register(m)?;

    Ok(())
}
