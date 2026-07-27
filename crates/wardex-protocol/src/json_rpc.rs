//! Generic JSON-RPC 2.0 parser — stdio newline-delimited framing.
//! MCP-agnostic, pyo3-free. params/result/error are preserved as raw JSON bytes (semantic extraction happens upstream).

use serde_json::Value;
use wardex_limits::Limits;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JsonRpcKind {
    Request,
    Response,
    Notification,
}

/// A single parsed JSON-RPC message.
#[derive(Debug, Clone)]
pub struct JsonRpcMessage {
    pub kind: JsonRpcKind,
    pub id: Option<String>, // numeric/string id normalized to a string (correlation key)
    pub method: Option<String>, // request/notification
    pub params: Option<Vec<u8>>, // request/notification raw JSON
    pub result: Option<Vec<u8>>, // response success raw JSON
    pub error: Option<Vec<u8>>, // response error raw JSON
}

/// Incremental parser for one direction (stdin or stdout) of a byte stream. Carves out messages at newline boundaries.
#[derive(Default)]
pub struct JsonRpcStream {
    buf: Vec<u8>,
    limits: Limits,
    /// Why the parser latched off, if it did. Mirrors `http1::Http1Stream`'s
    /// disable latch: a line that never terminates (the stdio equivalent of
    /// non-HTTP TLS traffic reaching the HTTP/1 parser) must not grow the
    /// buffer without bound.
    disabled_reason: Option<&'static str>,
}

impl JsonRpcStream {
    pub fn new(limits: Limits) -> Self {
        Self {
            buf: Vec::new(),
            limits,
            disabled_reason: None,
        }
    }

    /// Bytes currently held awaiting a newline.
    pub fn buffered_len(&self) -> usize {
        self.buf.len()
    }

    /// Why the parser latched off, if it did. Surfaced for debug logging: no
    /// span exists to carry it, because no message was ever parsed.
    pub fn disabled_reason(&self) -> Option<&'static str> {
        self.disabled_reason
    }

    /// Accumulates bytes and returns zero or more completed (newline-terminated) JSON-RPC messages. Non-JSON-RPC lines are skipped.
    pub fn feed(&mut self, data: &[u8]) -> Vec<JsonRpcMessage> {
        if self.disabled_reason.is_some() {
            return Vec::new();
        }
        self.buf.extend_from_slice(data);
        if self.buf.len() > self.limits.max_stream_buffer_bytes {
            // A line that never terminates: bytes keep arriving but no message
            // ever completes. Latch off rather than growing without bound.
            // Checked immediately after the append and before any parse
            // attempt, so the buffer is released as soon as the ceiling is
            // crossed rather than after one more scan.
            self.disabled_reason = Some("stream_buffer_exceeded");
            self.buf = Vec::new();
            return Vec::new();
        }
        let mut out = Vec::new();
        while let Some(nl) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=nl).collect(); // consume including '\n'
            let mut end = line.len() - 1; // exclude '\n'
            if end > 0 && line[end - 1] == b'\r' {
                end -= 1; // handle CRLF
            }
            let line = &line[..end];
            if line.is_empty() {
                continue;
            }
            if let Some(msg) = parse_line(line) {
                out.push(msg);
            }
        }
        out
    }
}

fn parse_line(line: &[u8]) -> Option<JsonRpcMessage> {
    let v: Value = serde_json::from_slice(line).ok()?;
    let obj = v.as_object()?;
    let id = obj.get("id").and_then(normalize_id);
    let method = obj
        .get("method")
        .and_then(|m| m.as_str())
        .map(str::to_string);
    let params = obj
        .get("params")
        .map(|p| serde_json::to_vec(p).unwrap_or_default());
    let result = obj
        .get("result")
        .map(|r| serde_json::to_vec(r).unwrap_or_default());
    let error = obj
        .get("error")
        .map(|e| serde_json::to_vec(e).unwrap_or_default());
    let kind = if method.is_some() {
        if id.is_some() {
            JsonRpcKind::Request
        } else {
            JsonRpcKind::Notification
        }
    } else if result.is_some() || error.is_some() {
        JsonRpcKind::Response
    } else {
        return None; // not a JSON-RPC shape
    };
    Some(JsonRpcMessage {
        kind,
        id,
        method,
        params,
        result,
        error,
    })
}

/// Normalizes the id to a string. Only numbers/strings are allowed (null etc. become None → excluded from correlation).
fn normalize_id(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(line: &str) -> JsonRpcMessage {
        let mut s = JsonRpcStream::new(Limits::default());
        let mut v = s.feed(line.as_bytes());
        assert_eq!(v.len(), 1, "exactly 1 message");
        v.pop().unwrap()
    }

    #[test]
    fn parses_request_with_numeric_id() {
        let m = one("{\"jsonrpc\":\"2.0\",\"id\":7,\"method\":\"tools/call\",\"params\":{\"name\":\"x\"}}\n");
        assert_eq!(m.kind, JsonRpcKind::Request);
        assert_eq!(m.id.as_deref(), Some("7"));
        assert_eq!(m.method.as_deref(), Some("tools/call"));
        assert!(m.params.is_some());
        assert!(m.result.is_none());
    }

    #[test]
    fn parses_request_with_string_id() {
        let m = one("{\"jsonrpc\":\"2.0\",\"id\":\"abc\",\"method\":\"ping\"}\n");
        assert_eq!(m.kind, JsonRpcKind::Request);
        assert_eq!(m.id.as_deref(), Some("abc"));
    }

    #[test]
    fn parses_response_result() {
        let m = one("{\"jsonrpc\":\"2.0\",\"id\":7,\"result\":{\"ok\":true}}\n");
        assert_eq!(m.kind, JsonRpcKind::Response);
        assert_eq!(m.id.as_deref(), Some("7"));
        assert!(m.result.is_some());
        assert!(m.error.is_none());
    }

    #[test]
    fn parses_response_error() {
        let m = one(
            "{\"jsonrpc\":\"2.0\",\"id\":7,\"error\":{\"code\":-32601,\"message\":\"nope\"}}\n",
        );
        assert_eq!(m.kind, JsonRpcKind::Response);
        assert!(m.error.is_some());
    }

    #[test]
    fn parses_notification_without_id() {
        let m = one("{\"jsonrpc\":\"2.0\",\"method\":\"notifications/cancelled\"}\n");
        assert_eq!(m.kind, JsonRpcKind::Notification);
        assert!(m.id.is_none());
        assert_eq!(m.method.as_deref(), Some("notifications/cancelled"));
    }

    #[test]
    fn handles_split_arrival() {
        let mut s = JsonRpcStream::new(Limits::default());
        assert_eq!(s.feed(b"{\"jsonrpc\":\"2.0\",\"id\":1,").len(), 0);
        let v = s.feed(b"\"method\":\"a\"}\n");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].method.as_deref(), Some("a"));
    }

    #[test]
    fn handles_multiple_and_crlf() {
        let mut s = JsonRpcStream::new(Limits::default());
        let v = s.feed(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"a\"}\r\n{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"b\"}\n");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].id.as_deref(), Some("1"));
        assert_eq!(v[1].id.as_deref(), Some("2"));
    }

    #[test]
    fn unbounded_line_latches_off() {
        let limits = Limits {
            max_stream_buffer_bytes: 512,
            ..Default::default()
        };
        let mut s = JsonRpcStream::new(limits);
        for _ in 0..100 {
            s.feed(&[b'x'; 64]); // never a newline
        }
        assert_eq!(s.disabled_reason(), Some("stream_buffer_exceeded"));
        assert!(s.buffered_len() <= 512);
    }

    #[test]
    fn latched_stream_discards_further_bytes() {
        let limits = Limits {
            max_stream_buffer_bytes: 64,
            ..Default::default()
        };
        let mut s = JsonRpcStream::new(limits);
        for _ in 0..10 {
            s.feed(&[b'x'; 64]);
        }
        assert_eq!(s.disabled_reason(), Some("stream_buffer_exceeded"));
        // Even a well-formed line is ignored once latched.
        let v = s.feed(b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"a\"}\n");
        assert_eq!(v.len(), 0);
        assert_eq!(s.buffered_len(), 0);
    }

    #[test]
    fn skips_malformed_and_non_jsonrpc_lines() {
        let mut s = JsonRpcStream::new(Limits::default());
        // non-JSON line + not a jsonrpc shape (an object but missing method/result/error) → skipped, no panic
        let v = s.feed(
            b"not json at all\n{\"hello\":1}\n{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"c\"}\n",
        );
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].id.as_deref(), Some("3"));
    }
}
