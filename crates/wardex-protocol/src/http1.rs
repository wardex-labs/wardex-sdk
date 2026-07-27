//! Incremental HTTP/1.x parser — the destination the SSL interceptor streams plaintext bytes into.
//! Parses headers with httparse and handles Content-Length / chunked body boundaries directly.
//!
//! Parse state survives across `feed` calls, so every byte is examined once and
//! decoded body bytes move straight into the message. The buffer therefore holds
//! only the tail that could not yet be consumed — roughly one read, never one body.

use wardex_limits::Limits;

/// A single parsed HTTP message (request or response).
#[derive(Debug, Clone)]
pub struct ParsedHttp {
    pub is_request: bool,
    pub method: Option<String>,
    pub path: Option<String>,
    pub version: u8,
    pub status: Option<u16>,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub truncated: bool,
    pub header_len: usize,
    /// Capture-limitation markers surfaced on the span.
    pub limitations: Vec<&'static str>,
}

/// Incremental parser for one direction of a connection (request or response). Accumulates bytes and carves off completed messages.
pub struct Http1Stream {
    buf: Vec<u8>,
    /// Read cursor into `buf`. Bytes before it are consumed; the prefix is
    /// dropped once per `feed` so advancing over a token stays O(1).
    pos: usize,
    is_request: bool,
    limits: Limits,
    state: State,
    bytes_scanned: u64,
}

/// Where the parser is within the message it is currently assembling.
enum State {
    /// Accumulating a header block.
    Headers,
    /// Reading a body of known length. `remaining` is `usize::MAX` for an
    /// EOF-terminated response, which only `flush_truncated` can end.
    Body {
        msg: ParsedHttp,
        remaining: usize,
        cap: usize,
    },
    /// Reading a chunked body.
    Chunked {
        msg: ParsedHttp,
        dec: ChunkState,
        cap: usize,
    },
    /// The peer is not speaking HTTP; the parser has latched off.
    Disabled,
}

/// Where the chunked decoder is within the current chunk.
#[derive(Clone, Copy)]
enum ChunkState {
    /// Reading the `1a2b\r\n` size line.
    Size,
    /// Reading chunk payload.
    Data { remaining: usize },
    /// Consuming the CRLF that follows a chunk payload.
    DataCrlf,
    /// Consuming trailer lines until the blank line.
    Trailer,
}

/// Outcome of one step of the state machine.
enum Step {
    /// A message completed; it is ready to emit.
    Done(ParsedHttp),
    /// More bytes are required.
    NeedMore,
    /// The stream is not HTTP; the parser latches off.
    Fail,
}

enum Framing {
    Length(usize),
    Chunked,
    None,
}

impl Http1Stream {
    pub fn new(is_request: bool, limits: Limits) -> Self {
        Self {
            buf: Vec::new(),
            pos: 0,
            is_request,
            limits,
            state: State::Headers,
            bytes_scanned: 0,
        }
    }

    /// Total bytes examined since construction. A single-pass parser stays close
    /// to the stream length; a re-parsing one grows quadratically.
    pub fn bytes_scanned(&self) -> u64 {
        self.bytes_scanned
    }

    /// Bytes currently held awaiting more input.
    pub fn buffered_len(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Accumulates bytes and returns zero or more completed messages (handles keep-alive).
    pub fn feed(&mut self, data: &[u8]) -> Vec<ParsedHttp> {
        if matches!(self.state, State::Disabled) {
            return Vec::new();
        }
        self.buf.extend_from_slice(data);
        let mut out = Vec::new();
        loop {
            match self.step() {
                Step::Done(msg) => {
                    out.push(msg);
                    self.state = State::Headers;
                    if self.avail().is_empty() {
                        break;
                    }
                }
                Step::NeedMore => break,
                Step::Fail => {
                    self.state = State::Disabled;
                    self.buf = Vec::new();
                    self.pos = 0;
                    break;
                }
            }
        }
        self.compact();
        out
    }

    /// Called on connection close — carves off the message in flight, whose body
    /// had no framing (or never finished arriving).
    pub fn flush_truncated(&mut self) -> Option<ParsedHttp> {
        if self.is_request {
            return None;
        }
        match std::mem::replace(&mut self.state, State::Headers) {
            State::Body { mut msg, .. } | State::Chunked { mut msg, .. } => {
                msg.truncated = true;
                self.buf = Vec::new();
                self.pos = 0;
                Some(msg)
            }
            other => {
                self.state = other;
                None
            }
        }
    }

    /// Bytes fed but not yet consumed.
    fn avail(&self) -> &[u8] {
        &self.buf[self.pos..]
    }

    fn consume(&mut self, n: usize) {
        self.pos += n;
    }

    /// Drops the consumed prefix. Called once per `feed` rather than per token,
    /// so a body arriving as many small chunks does not memmove the rest of the
    /// read on every chunk boundary.
    fn compact(&mut self) {
        if self.pos == 0 {
            return;
        }
        if self.pos >= self.buf.len() {
            self.buf.clear();
        } else {
            self.buf.drain(..self.pos);
        }
        self.pos = 0;
    }

    /// Advances the machine as far as the buffered bytes allow.
    fn step(&mut self) -> Step {
        match std::mem::replace(&mut self.state, State::Headers) {
            State::Headers => self.step_headers(),
            State::Body {
                msg,
                remaining,
                cap,
            } => self.step_body(msg, remaining, cap),
            State::Chunked { msg, dec, cap } => self.step_chunked(msg, dec, cap),
            State::Disabled => {
                self.state = State::Disabled;
                Step::NeedMore
            }
        }
    }

    fn step_headers(&mut self) -> Step {
        if self.avail().is_empty() {
            return Step::NeedMore;
        }
        let mut headers = vec![httparse::EMPTY_HEADER; self.limits.max_headers];
        self.bytes_scanned += self.avail().len() as u64;

        let (header_len, mut msg) = if self.is_request {
            let mut req = httparse::Request::new(&mut headers);
            match req.parse(self.avail()) {
                Ok(httparse::Status::Complete(n)) => (n, request_to_parsed(&req)),
                Ok(httparse::Status::Partial) => return Step::NeedMore,
                Err(_) => return Step::Fail,
            }
        } else {
            let mut resp = httparse::Response::new(&mut headers);
            match resp.parse(self.avail()) {
                Ok(httparse::Status::Complete(n)) => (n, response_to_parsed(&resp)),
                Ok(httparse::Status::Partial) => return Step::NeedMore,
                Err(_) => return Step::Fail,
            }
        };

        msg.header_len = header_len;
        self.consume(header_len);

        // RFC 7230 §3.3.3: 1xx/204/304 responses have no body → complete immediately from headers alone.
        // (Without this, a 101 upgrade response with no Content-Length would be treated as EOF-terminated and never emitted.)
        if !msg.is_request {
            if let Some(code) = msg.status {
                if (100..200).contains(&code) || code == 204 || code == 304 {
                    return Step::Done(msg);
                }
            }
        }

        let cap = body_cap(&msg.headers, &self.limits);
        match body_framing(&msg.headers) {
            Framing::Length(n) => {
                self.state = State::Body {
                    msg,
                    remaining: n,
                    cap,
                };
                self.step()
            }
            Framing::Chunked => {
                self.state = State::Chunked {
                    msg,
                    dec: ChunkState::Size,
                    cap,
                };
                self.step()
            }
            Framing::None => {
                if msg.is_request {
                    // a request with no CL/TE has no body (GET etc.) → complete
                    Step::Done(msg)
                } else {
                    // a response body is EOF-terminated → handled by flush_truncated on close
                    self.state = State::Body {
                        msg,
                        remaining: usize::MAX,
                        cap,
                    };
                    self.step()
                }
            }
        }
    }

    fn step_body(&mut self, mut msg: ParsedHttp, remaining: usize, cap: usize) -> Step {
        let take = remaining.min(self.avail().len());
        if take > 0 {
            append_capped(&mut msg, &self.buf[self.pos..self.pos + take], cap);
            self.bytes_scanned += take as u64;
            self.consume(take);
        }
        let left = remaining - take;
        if left == 0 {
            Step::Done(msg)
        } else {
            self.state = State::Body {
                msg,
                remaining: left,
                cap,
            };
            Step::NeedMore
        }
    }

    fn step_chunked(&mut self, mut msg: ParsedHttp, mut dec: ChunkState, cap: usize) -> Step {
        loop {
            match dec {
                ChunkState::Size => {
                    let Some(idx) = find_crlf(self.avail()) else {
                        self.bytes_scanned += self.avail().len() as u64;
                        self.state = State::Chunked { msg, dec, cap };
                        return Step::NeedMore;
                    };
                    self.bytes_scanned += (idx + 2) as u64;
                    let Ok(line) = std::str::from_utf8(&self.avail()[..idx]) else {
                        return Step::Fail;
                    };
                    let Ok(size) =
                        usize::from_str_radix(line.split(';').next().unwrap_or("").trim(), 16)
                    else {
                        return Step::Fail;
                    };
                    self.consume(idx + 2);
                    dec = if size == 0 {
                        ChunkState::Trailer
                    } else {
                        ChunkState::Data { remaining: size }
                    };
                }
                ChunkState::Data { remaining } => {
                    if self.avail().is_empty() {
                        self.state = State::Chunked { msg, dec, cap };
                        return Step::NeedMore;
                    }
                    let take = remaining.min(self.avail().len());
                    append_capped(&mut msg, &self.buf[self.pos..self.pos + take], cap);
                    self.bytes_scanned += take as u64;
                    self.consume(take);
                    let left = remaining - take;
                    dec = if left == 0 {
                        ChunkState::DataCrlf
                    } else {
                        ChunkState::Data { remaining: left }
                    };
                }
                ChunkState::DataCrlf => {
                    if self.avail().len() < 2 {
                        self.state = State::Chunked { msg, dec, cap };
                        return Step::NeedMore;
                    }
                    self.bytes_scanned += 2;
                    self.consume(2);
                    dec = ChunkState::Size;
                }
                ChunkState::Trailer => {
                    // Trailer lines are consumed, not captured; a blank line ends the body.
                    let Some(idx) = find_crlf(self.avail()) else {
                        self.bytes_scanned += self.avail().len() as u64;
                        self.state = State::Chunked { msg, dec, cap };
                        return Step::NeedMore;
                    };
                    self.bytes_scanned += (idx + 2) as u64;
                    self.consume(idx + 2);
                    if idx == 0 {
                        return Step::Done(msg);
                    }
                }
            }
        }
    }
}

/// Appends `data` to the message body, stopping at `cap`. Bytes past the cap are
/// consumed but not stored: framing must keep advancing so the next keep-alive
/// message on this connection still parses.
fn append_capped(msg: &mut ParsedHttp, data: &[u8], cap: usize) {
    let room = cap.saturating_sub(msg.body.len());
    let take = room.min(data.len());
    msg.body.extend_from_slice(&data[..take]);
    if take < data.len() && !msg.truncated {
        msg.truncated = true;
        msg.limitations.push("body_cap_exceeded");
    }
}

/// Selects the body cap for a message. A later slice makes this content-type aware.
fn body_cap(_headers: &[(String, String)], limits: &Limits) -> usize {
    limits.max_body_bytes
}

fn request_to_parsed(req: &httparse::Request) -> ParsedHttp {
    ParsedHttp {
        is_request: true,
        method: req.method.map(str::to_string),
        path: req.path.map(str::to_string),
        version: req.version.unwrap_or(1),
        status: None,
        headers: collect_headers(req.headers),
        body: Vec::new(),
        truncated: false,
        header_len: 0,
        limitations: Vec::new(),
    }
}

fn response_to_parsed(resp: &httparse::Response) -> ParsedHttp {
    ParsedHttp {
        is_request: false,
        method: None,
        path: None,
        version: resp.version.unwrap_or(1),
        status: resp.code,
        headers: collect_headers(resp.headers),
        body: Vec::new(),
        truncated: false,
        header_len: 0,
        limitations: Vec::new(),
    }
}

fn collect_headers(headers: &[httparse::Header]) -> Vec<(String, String)> {
    headers
        .iter()
        .filter(|h| !h.name.is_empty())
        .map(|h| {
            (
                h.name.to_string(),
                String::from_utf8_lossy(h.value).into_owned(),
            )
        })
        .collect()
}

fn body_framing(headers: &[(String, String)]) -> Framing {
    for (k, v) in headers {
        if k.eq_ignore_ascii_case("transfer-encoding") && v.to_ascii_lowercase().contains("chunked")
        {
            return Framing::Chunked;
        }
    }
    for (k, v) in headers {
        if k.eq_ignore_ascii_case("content-length") {
            if let Ok(n) = v.trim().parse::<usize>() {
                return Framing::Length(n);
            }
        }
    }
    Framing::None
}

fn find_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(2).position(|w| w == b"\r\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_complete_request_with_body() {
        let raw = b"POST /v1/messages HTTP/1.1\r\nHost: api.x\r\nContent-Length: 7\r\n\r\nhello!!";
        let mut s = Http1Stream::new(true, Limits::default());
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].method.as_deref(), Some("POST"));
        assert_eq!(msgs[0].path.as_deref(), Some("/v1/messages"));
        assert_eq!(msgs[0].body, b"hello!!");
    }

    #[test]
    fn parses_complete_response_with_content_length() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi";
        let mut s = Http1Stream::new(false, Limits::default());
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].status, Some(200));
        assert_eq!(msgs[0].body, b"hi");
    }

    #[test]
    fn parses_chunked_response() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let mut s = Http1Stream::new(false, Limits::default());
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body, b"hello world");
    }

    #[test]
    fn handles_split_arrival() {
        let mut s = Http1Stream::new(false, Limits::default());
        assert_eq!(s.feed(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n").len(), 0);
        assert_eq!(s.feed(b"\r\nhel").len(), 0);
        let msgs = s.feed(b"lo");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body, b"hello");
    }

    #[test]
    fn handles_keep_alive_two_messages() {
        let raw = b"GET /a HTTP/1.1\r\nContent-Length: 0\r\n\r\nGET /b HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
        let mut s = Http1Stream::new(true, Limits::default());
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].path.as_deref(), Some("/a"));
        assert_eq!(msgs[1].path.as_deref(), Some("/b"));
    }

    #[test]
    fn malformed_yields_no_messages_without_panic() {
        let mut s = Http1Stream::new(false, Limits::default());
        // not valid HTTP, so zero completed messages (without panicking)
        assert_eq!(s.feed(b"this is not http \x00\x01\x02").len(), 0);
    }

    #[test]
    fn chunked_response_with_trailer_fields() {
        // chunked response with trailer headers — only the body should be decoded, and the trailer should be consumed
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\nX-Trace: abc\r\n\r\n";
        let mut s = Http1Stream::new(false, Limits::default());
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body, b"hello");
    }

    #[test]
    fn chunked_with_trailer_then_keep_alive_next_message() {
        // after a chunked response with a trailer, the next keep-alive message should parse correctly
        let mut buf = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\nX-Trace: abc\r\n\r\n".to_vec();
        buf.extend_from_slice(b"HTTP/1.1 201 Created\r\nContent-Length: 2\r\n\r\nok");
        let mut s = Http1Stream::new(false, Limits::default());
        let msgs = s.feed(&buf);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].body, b"hello");
        assert_eq!(msgs[1].status, Some(201));
        assert_eq!(msgs[1].body, b"ok");
    }

    #[test]
    fn flush_truncated_returns_partial_body() {
        // a response with neither Content-Length nor chunked → flush on close
        let mut s = Http1Stream::new(false, Limits::default());
        assert_eq!(
            s.feed(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\npartial")
                .len(),
            0
        );
        let m = s.flush_truncated().expect("truncated message");
        assert_eq!(m.status, Some(200));
        assert_eq!(m.body, b"partial");
        assert!(m.truncated);
    }

    #[test]
    fn exposes_header_len_for_response() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi";
        let mut s = Http1Stream::new(false, Limits::default());
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 1);
        // end of headers (right after the blank line \r\n\r\n) = the offset where "hi" starts
        let header_len = raw.len() - 2; // exclude the 2 bytes of "hi"
        assert_eq!(msgs[0].header_len, header_len);
        assert_eq!(msgs[0].body, b"hi");
    }

    #[test]
    fn emits_101_switching_protocols_without_content_length() {
        // RFC 7230 §3.3.3: 101 is a bodyless 1xx → must be emitted immediately even without Content-Length.
        // The following WS frame bytes must not be consumed.
        let ws_frame = b"\x81\x02hi";
        let mut raw = b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n".to_vec();
        let header_only_len = raw.len();
        raw.extend_from_slice(ws_frame);

        let mut s = Http1Stream::new(false, Limits::default());
        let msgs = s.feed(&raw);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].status, Some(101));
        assert_eq!(msgs[0].body, b"");
        // header_len must point to only the header block and must not include the WS frame.
        assert_eq!(msgs[0].header_len, header_only_len);
    }

    #[test]
    fn emits_204_no_content_without_content_length() {
        // RFC 7230 §3.3.3: 204 has no body → must be emitted immediately even without Content-Length.
        let raw = b"HTTP/1.1 204 No Content\r\nX-Request-Id: abc\r\n\r\n";
        let mut s = Http1Stream::new(false, Limits::default());
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].status, Some(204));
        assert_eq!(msgs[0].body, b"");
        assert_eq!(msgs[0].header_len, raw.len());
    }

    #[test]
    fn header_count_comes_from_limits() {
        let limits = Limits {
            max_headers: 1,
            ..Default::default()
        };
        let mut s = Http1Stream::new(false, limits);
        // Two headers exceeds the cap of one.
        let raw = b"HTTP/1.1 200 OK\r\nA: 1\r\nB: 2\r\nContent-Length: 0\r\n\r\n";
        assert_eq!(s.feed(raw).len(), 0);
    }

    /// Feeds `data` to `s` in fixed-size chunks.
    fn feed_chunked(s: &mut Http1Stream, data: &[u8], chunk: usize) -> Vec<ParsedHttp> {
        let mut out = Vec::new();
        for part in data.chunks(chunk) {
            out.extend(s.feed(part));
        }
        out
    }

    /// Builds a chunked response whose decoded body is `size` bytes.
    fn chunked_response(size: usize) -> Vec<u8> {
        let mut v = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n".to_vec();
        let payload = vec![b'x'; size];
        for c in payload.chunks(8192) {
            v.extend_from_slice(format!("{:x}\r\n", c.len()).as_bytes());
            v.extend_from_slice(c);
            v.extend_from_slice(b"\r\n");
        }
        v.extend_from_slice(b"0\r\n\r\n");
        v
    }

    /// Streams whose framing exercises a different corner of the machine:
    /// Content-Length, multi-chunk, a trailer section, a keep-alive pair, and
    /// the bodyless-response path.
    fn split_invariance_cases() -> Vec<Vec<u8>> {
        let mut keep_alive =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\nX-Trace: abc\r\n\r\n"
                .to_vec();
        keep_alive.extend_from_slice(b"HTTP/1.1 201 Created\r\nContent-Length: 2\r\n\r\nok");
        vec![
            b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello".to_vec(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n".to_vec(),
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\nX-Trace: abc\r\n\r\n".to_vec(),
            keep_alive,
            b"HTTP/1.1 204 No Content\r\nX-Request-Id: abc\r\n\r\n".to_vec(),
        ]
    }

    fn assert_same_messages(got: &[ParsedHttp], expected: &[ParsedHttp], ctx: &str) {
        assert_eq!(got.len(), expected.len(), "message count differs: {ctx}");
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_eq!(a.body, b.body, "body differs: {ctx}");
            assert_eq!(a.status, b.status, "status differs: {ctx}");
            assert_eq!(a.header_len, b.header_len, "header_len differs: {ctx}");
        }
    }

    #[test]
    fn split_invariance_one_byte_at_a_time() {
        // The worst possible split: every state transition is interrupted.
        for raw in split_invariance_cases() {
            let mut whole = Http1Stream::new(false, Limits::default());
            let expected = whole.feed(&raw);

            let mut split = Http1Stream::new(false, Limits::default());
            let got = feed_chunked(&mut split, &raw, 1);

            assert_same_messages(&got, &expected, &format!("{raw:?}"));
        }
    }

    #[test]
    fn split_invariance_every_two_way_split() {
        for raw in split_invariance_cases() {
            let mut whole = Http1Stream::new(false, Limits::default());
            let expected = whole.feed(&raw);

            for split_at in 1..raw.len() {
                let mut s = Http1Stream::new(false, Limits::default());
                let mut got = s.feed(&raw[..split_at]);
                got.extend(s.feed(&raw[split_at..]));
                assert_same_messages(&got, &expected, &format!("split at {split_at}"));
            }
        }
    }

    #[test]
    fn chunked_decode_is_linear_not_quadratic() {
        // Ten megabytes arriving in 64 KiB reads. Re-decoding the accumulated body
        // on every read would scan roughly 800 MB; a single pass scans ~10 MB.
        let raw = chunked_response(10 * 1024 * 1024);
        let mut s = Http1Stream::new(false, Limits::default());
        let msgs = feed_chunked(&mut s, &raw, 64 * 1024);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body.len(), 10 * 1024 * 1024);
        assert!(
            s.bytes_scanned() < 2 * raw.len() as u64,
            "scanned {} bytes for a {}-byte stream",
            s.bytes_scanned(),
            raw.len()
        );
    }

    #[test]
    fn buffer_stays_small_during_a_large_body() {
        let raw = chunked_response(4 * 1024 * 1024);
        let mut s = Http1Stream::new(false, Limits::default());
        for part in raw.chunks(64 * 1024) {
            s.feed(part);
            assert!(
                s.buffered_len() < 256 * 1024,
                "buffer grew to {}",
                s.buffered_len()
            );
        }
    }
}
