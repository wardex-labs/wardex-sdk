//! Generic JSON-RPC 2.0 parser — stdio newline-delimited framing.
//! MCP-agnostic, pyo3-free. params/result/error are kept as JSON bytes re-encoded compact by
//! serde_json, not as the bytes on the pipe: a sender's whitespace and needless `\u` escapes are
//! dropped (semantic extraction happens upstream).
//!
//! A line holding a JSON array is a batch (JSON-RPC 2.0 section 6): each element is read as one
//! message, in order, and an element that is not a JSON-RPC object is skipped exactly as a
//! non-JSON-RPC line is. Older MCP revisions allowed batches, so a server written against one
//! can answer a batch with a batch.
//!
//! No line longer than `max_stream_buffer_bytes` is ever parsed, however it arrives, but what
//! happens to the stream depends on how it arrives. Only a line that lands WHOLE in one read is
//! skipped on its own and counted (`lines_over_cap`), with the stream kept: its newline is in
//! hand, so the framing around it is intact. A line that arrives across reads latches the whole
//! stream off (`disabled_reason() == "stream_buffer_exceeded"`) as soon as the unterminated part
//! passes the bound, because the parser cannot yet find where it ends. The second is the usual
//! case: a pipe write over 16 MiB reaches the reader in pieces far smaller than that.

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
    pub params: Option<Vec<u8>>, // request/notification JSON, re-encoded compact
    pub result: Option<Vec<u8>>, // response success JSON, re-encoded compact
    pub error: Option<Vec<u8>>, // response error JSON, re-encoded compact
}

/// Incremental parser for one direction (stdin or stdout) of a byte stream. Carves out messages at newline boundaries.
#[derive(Default)]
pub struct JsonRpcStream {
    buf: Vec<u8>,
    /// Read cursor into `buf`: bytes before it belong to lines already
    /// carved. Dropped once per `feed`, so a read of many short lines costs
    /// one pass rather than one memmove of the rest of the read per line.
    pos: usize,
    /// Where the next newline search starts. Everything between `pos` and
    /// here is already known to hold no newline, so a long line arriving in
    /// many reads is searched once rather than from its start on every read.
    scan: usize,
    moved: u64,
    scanned: u64,
    lines_over_cap: u64,
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
            limits,
            ..Default::default()
        }
    }

    /// Bytes currently held awaiting a newline.
    pub fn buffered_len(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Why the parser latched off, if it did. Surfaced for debug logging: no
    /// span exists to carry it, because no message was ever parsed.
    pub fn disabled_reason(&self) -> Option<&'static str> {
        self.disabled_reason
    }

    /// Complete lines longer than `max_stream_buffer_bytes` that were skipped
    /// unparsed, counted since construction. Each one is a message lost
    /// whole: a request whose call is never seen, or a response whose call
    /// never completes.
    pub fn lines_over_cap(&self) -> u64 {
        self.lines_over_cap
    }

    /// Bytes copied into the buffer or relocated inside it since
    /// construction: each fed byte once on arrival, plus the residue a
    /// compaction moves to the front. The residue is the start of one line
    /// still unterminated, and no compaction runs again until that line is
    /// carved, so each byte moves at most once and the count stays under
    /// twice the input.
    pub fn bytes_moved(&self) -> u64 {
        self.moved
    }

    /// Bytes the newline search examined since construction. Each byte is
    /// examined once, so the count never exceeds the input.
    pub fn bytes_scanned(&self) -> u64 {
        self.scanned
    }

    /// Accumulates bytes and returns zero or more completed (newline-terminated) JSON-RPC messages. Non-JSON-RPC lines are skipped.
    pub fn feed(&mut self, data: &[u8]) -> Vec<JsonRpcMessage> {
        if self.disabled_reason.is_some() {
            return Vec::new();
        }
        self.buf.extend_from_slice(data);
        self.moved += data.len() as u64;
        let mut out = Vec::new();
        loop {
            let from = self.scan.max(self.pos);
            let Some(offset) = self.buf[from..].iter().position(|&b| b == b'\n') else {
                self.scanned += (self.buf.len() - from) as u64;
                self.scan = self.buf.len();
                break;
            };
            self.scanned += (offset + 1) as u64;
            let nl = from + offset;
            let start = self.pos;
            self.pos = nl + 1; // consume including '\n'
            self.scan = self.pos;
            let mut end = nl; // exclude '\n'
            if end > start && self.buf[end - 1] == b'\r' {
                end -= 1; // handle CRLF
            }
            let line = &self.buf[start..end];
            if line.is_empty() {
                continue;
            }
            if line.len() > self.limits.max_stream_buffer_bytes {
                self.lines_over_cap += 1;
                continue;
            }
            parse_line(line, &mut out);
        }
        self.compact();
        // What is left after every complete line has been carved is a line
        // that never terminates: bytes keep arriving but no message ever
        // completes. Latch off rather than growing without bound.
        //
        // Measured after the pass, not on the raw appended buffer: a single
        // write carrying many complete lines is fully consumed here, and
        // disabling the stream over it would cost every later message on it.
        // Peak memory is identical either way — `data` is appended before
        // either check.
        if self.buffered_len() > self.limits.max_stream_buffer_bytes {
            self.disabled_reason = Some("stream_buffer_exceeded");
            self.buf = Vec::new();
            self.pos = 0;
            self.scan = 0;
        }
        out
    }

    /// Drops the consumed prefix, once per `feed`.
    fn compact(&mut self) {
        if self.pos == 0 {
            return;
        }
        if self.pos >= self.buf.len() {
            self.buf.clear();
        } else {
            self.moved += (self.buf.len() - self.pos) as u64;
            self.buf.drain(..self.pos);
        }
        // `scan` never trails `pos`: it is at or past the start of the residue.
        self.scan -= self.pos;
        self.pos = 0;
    }
}

/// One line: a single message, or a batch of them.
fn parse_line(line: &[u8], out: &mut Vec<JsonRpcMessage>) {
    let Ok(v) = serde_json::from_slice::<Value>(line) else {
        return;
    };
    match &v {
        Value::Array(items) => out.extend(items.iter().filter_map(parse_message)),
        single => out.extend(parse_message(single)),
    }
}

fn parse_message(v: &Value) -> Option<JsonRpcMessage> {
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
    fn params_and_result_are_re_encoded_compact_not_kept_as_sent() {
        // Spaced and `\u`-escaped on the pipe, as Python's stdlib json writes it.
        let sent = "{\"jsonrpc\": \"2.0\", \"id\": 7, \"result\": {\"text\": \"\\uc548\\ub155\", \"n\": [1, 2]}}\n";
        let kept = one(sent).result.unwrap();
        let compact = "{\"text\":\"안녕\",\"n\":[1,2]}";
        assert_eq!(kept.len(), compact.len());
        assert!(!kept.contains(&b' '));
        assert!(!kept.windows(2).any(|w| w == b"\\u"));
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
    fn a_single_write_of_complete_lines_larger_than_the_ceiling_is_not_latched() {
        // The ceiling bounds an unterminated line, not a large write that the
        // drain consumes entirely. A writer that flushes many messages at once
        // must not cost the stream every message that follows.
        let limits = Limits {
            max_stream_buffer_bytes: 512,
            ..Default::default()
        };
        let mut s = JsonRpcStream::new(limits);
        let mut data = Vec::new();
        for i in 0..50 {
            data.extend_from_slice(
                format!("{{\"jsonrpc\":\"2.0\",\"id\":{i},\"method\":\"tools/call\"}}\n")
                    .as_bytes(),
            );
        }
        assert!(data.len() > 512, "the write must exceed the ceiling");

        let v = s.feed(&data);
        assert_eq!(v.len(), 50);
        assert_eq!(s.disabled_reason(), None);

        // The stream keeps working afterwards.
        let v = s.feed(b"{\"jsonrpc\":\"2.0\",\"id\":99,\"method\":\"ping\"}\n");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].id.as_deref(), Some("99"));
    }

    #[test]
    fn a_complete_line_longer_than_the_ceiling_is_skipped_and_counted() {
        // One read carries a short line, a line over the ceiling and another
        // short line. The long line is never parsed, whatever its size; the
        // framing around it is intact, so the lines either side still are.
        let limits = Limits {
            max_stream_buffer_bytes: 64,
            ..Default::default()
        };
        let mut s = JsonRpcStream::new(limits);
        let long = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"tools/call\",\"params\":{{\"x\":\"{}\"}}}}\n",
            "y".repeat(100)
        );
        let mut data = b"{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"a\"}\n".to_vec();
        data.extend_from_slice(long.as_bytes());
        data.extend_from_slice(b"{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"c\"}\n");
        let v = s.feed(&data);
        let ids: Vec<_> = v.iter().map(|m| m.id.as_deref().unwrap()).collect();
        assert_eq!(ids, ["1", "3"]);
        assert_eq!(s.lines_over_cap(), 1);
        assert_eq!(s.disabled_reason(), None);
        // A line exactly at the ceiling is still parsed.
        let at = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":4,\"method\":\"{}\"}}",
            "m".repeat(28)
        );
        assert_eq!(at.len(), 64);
        assert_eq!(s.feed(format!("{at}\n").as_bytes()).len(), 1);
        assert_eq!(s.lines_over_cap(), 1);
    }

    #[test]
    fn a_batch_line_yields_every_message_in_it() {
        // 24 calls batched on one line, answered by a batch of 24 responses:
        // each element is one message, in order.
        let mut s = JsonRpcStream::new(Limits::default());
        let calls: Vec<String> = (0..24)
            .map(|i| format!("{{\"jsonrpc\":\"2.0\",\"id\":{i},\"method\":\"tools/call\"}}"))
            .collect();
        let v = s.feed(format!("[{}]\n", calls.join(",")).as_bytes());
        assert_eq!(v.len(), 24);
        assert!(v.iter().all(|m| m.kind == JsonRpcKind::Request));
        assert_eq!(v[23].id.as_deref(), Some("23"));

        let answers: Vec<String> = (0..24)
            .map(|i| format!("{{\"jsonrpc\":\"2.0\",\"id\":{i},\"result\":{{}}}}"))
            .collect();
        let v = s.feed(format!("[{}]\n", answers.join(",")).as_bytes());
        assert_eq!(v.len(), 24);
        assert!(v.iter().all(|m| m.kind == JsonRpcKind::Response));

        // Elements that are not JSON-RPC objects are skipped like such lines.
        let v = s.feed(b"[1,\"x\",{\"hello\":1},{\"jsonrpc\":\"2.0\",\"method\":\"n\"}]\n[]\n");
        assert_eq!(v.len(), 1);
        assert_eq!(v[0].kind, JsonRpcKind::Notification);
    }

    #[test]
    fn a_read_of_many_short_lines_is_linear_not_quadratic() {
        // 256 KiB of empty lines in ONE read: removing each line from the
        // front of the buffer moved the rest of the read once per line, about
        // 34 GB of memmove for this input.
        let raw = vec![b'\n'; 256 * 1024];
        let mut s = JsonRpcStream::new(Limits::default());
        assert!(s.feed(&raw).is_empty());
        assert!(s.bytes_moved() <= 2 * raw.len() as u64);
        assert!(s.bytes_scanned() <= raw.len() as u64);
    }

    #[test]
    fn a_long_line_arriving_in_many_reads_is_searched_once() {
        // A 1 MiB line in 1 KiB reads: searching the whole held line for a
        // newline on every read examined ~512 MB. The search resumes where the
        // last one stopped.
        let mut line = format!(
            "{{\"jsonrpc\":\"2.0\",\"id\":1,\"result\":\"{}\"}}",
            "z".repeat(1 << 20)
        )
        .into_bytes();
        line.push(b'\n');
        let mut s = JsonRpcStream::new(Limits::default());
        let mut got = Vec::new();
        for part in line.chunks(1024) {
            got.extend(s.feed(part));
        }
        assert_eq!(got.len(), 1);
        assert!(
            s.bytes_scanned() <= line.len() as u64,
            "scanned {} bytes for a {}-byte line",
            s.bytes_scanned(),
            line.len()
        );
        assert!(s.bytes_moved() <= 2 * line.len() as u64);
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
