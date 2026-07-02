//! Incremental HTTP/1.x parser — the destination the SSL interceptor streams plaintext bytes into.
//! Parses headers with httparse and handles Content-Length / chunked body boundaries directly.

const MAX_HEADERS: usize = 96;

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
}

/// Incremental parser for one direction of a connection (request or response). Accumulates bytes and carves off completed messages.
pub struct Http1Stream {
    buf: Vec<u8>,
    is_request: bool,
}

enum Framing {
    Length(usize),
    Chunked,
    None,
}

impl Http1Stream {
    pub fn new(is_request: bool) -> Self {
        Self {
            buf: Vec::new(),
            is_request,
        }
    }

    /// Accumulates bytes and returns zero or more completed messages (handles keep-alive).
    pub fn feed(&mut self, data: &[u8]) -> Vec<ParsedHttp> {
        self.buf.extend_from_slice(data);
        let mut out = Vec::new();
        while let Some((msg, consumed)) = self.try_parse_one() {
            self.buf.drain(..consumed);
            out.push(msg);
            if self.buf.is_empty() {
                break;
            }
        }
        out
    }

    /// Called on connection close — carves off a truncated response whose headers arrived but had no body boundary.
    pub fn flush_truncated(&mut self) -> Option<ParsedHttp> {
        let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
        if self.is_request {
            return None;
        }
        let mut resp = httparse::Response::new(&mut headers);
        match resp.parse(&self.buf) {
            Ok(httparse::Status::Complete(n)) => {
                let mut msg = response_to_parsed(&resp);
                msg.body = self.buf[n..].to_vec();
                msg.truncated = true;
                msg.header_len = n;
                self.buf.clear();
                Some(msg)
            }
            _ => None,
        }
    }

    fn try_parse_one(&self) -> Option<(ParsedHttp, usize)> {
        let mut headers = [httparse::EMPTY_HEADER; MAX_HEADERS];
        let (header_len, mut msg) = if self.is_request {
            let mut req = httparse::Request::new(&mut headers);
            match req.parse(&self.buf) {
                Ok(httparse::Status::Complete(n)) => (n, request_to_parsed(&req)),
                _ => return None,
            }
        } else {
            let mut resp = httparse::Response::new(&mut headers);
            match resp.parse(&self.buf) {
                Ok(httparse::Status::Complete(n)) => (n, response_to_parsed(&resp)),
                _ => return None,
            }
        };

        msg.header_len = header_len;
        // RFC 7230 §3.3.3: 1xx/204/304 responses have no body → complete immediately from headers alone.
        // (Without this, a 101 upgrade response with no Content-Length would be treated as EOF-terminated and never emitted.)
        if !msg.is_request {
            if let Some(code) = msg.status {
                if (100..200).contains(&code) || code == 204 || code == 304 {
                    return Some((msg, header_len));
                }
            }
        }
        let rest = &self.buf[header_len..];
        match body_framing(&msg.headers) {
            Framing::Length(n) => {
                if rest.len() < n {
                    return None;
                }
                msg.body = rest[..n].to_vec();
                Some((msg, header_len + n))
            }
            Framing::Chunked => {
                let (body, used) = decode_chunked(rest)?;
                msg.body = body;
                Some((msg, header_len + used))
            }
            Framing::None => {
                if msg.is_request {
                    // a request with no CL/TE has no body (GET etc.) → complete
                    Some((msg, header_len))
                } else {
                    // a response body is EOF-terminated → handled by flush_truncated on close
                    None
                }
            }
        }
    }
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

/// Decodes a chunked body. Returns (decoded body, bytes consumed) if complete, None if incomplete.
fn decode_chunked(buf: &[u8]) -> Option<(Vec<u8>, usize)> {
    let mut body = Vec::new();
    let mut pos = 0usize;
    loop {
        let line_end = find_crlf(&buf[pos..])? + pos;
        let size_str = std::str::from_utf8(&buf[pos..line_end]).ok()?;
        let size_str = size_str.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_str, 16).ok()?;
        let data_start = line_end + 2;
        if size == 0 {
            // consume trailer fields: skip header lines until a blank line is found
            let mut pos = data_start;
            loop {
                let crlf = find_crlf(&buf[pos..])?; // None means wait for more bytes
                if crlf == 0 {
                    // blank line → end of trailer section
                    return Some((body, pos + 2));
                }
                // skip one trailer line
                pos += crlf + 2;
            }
        }
        let data_end = data_start + size;
        if buf.len() < data_end + 2 {
            return None;
        }
        body.extend_from_slice(&buf[data_start..data_end]);
        pos = data_end + 2; // skip past data + CRLF
    }
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
        let mut s = Http1Stream::new(true);
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].method.as_deref(), Some("POST"));
        assert_eq!(msgs[0].path.as_deref(), Some("/v1/messages"));
        assert_eq!(msgs[0].body, b"hello!!");
    }

    #[test]
    fn parses_complete_response_with_content_length() {
        let raw = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi";
        let mut s = Http1Stream::new(false);
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].status, Some(200));
        assert_eq!(msgs[0].body, b"hi");
    }

    #[test]
    fn parses_chunked_response() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let mut s = Http1Stream::new(false);
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body, b"hello world");
    }

    #[test]
    fn handles_split_arrival() {
        let mut s = Http1Stream::new(false);
        assert_eq!(s.feed(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n").len(), 0);
        assert_eq!(s.feed(b"\r\nhel").len(), 0);
        let msgs = s.feed(b"lo");
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body, b"hello");
    }

    #[test]
    fn handles_keep_alive_two_messages() {
        let raw = b"GET /a HTTP/1.1\r\nContent-Length: 0\r\n\r\nGET /b HTTP/1.1\r\nContent-Length: 0\r\n\r\n";
        let mut s = Http1Stream::new(true);
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].path.as_deref(), Some("/a"));
        assert_eq!(msgs[1].path.as_deref(), Some("/b"));
    }

    #[test]
    fn malformed_yields_no_messages_without_panic() {
        let mut s = Http1Stream::new(false);
        // not valid HTTP, so zero completed messages (without panicking)
        assert_eq!(s.feed(b"this is not http \x00\x01\x02").len(), 0);
    }

    #[test]
    fn chunked_response_with_trailer_fields() {
        // chunked response with trailer headers — only the body should be decoded, and the trailer should be consumed
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\nX-Trace: abc\r\n\r\n";
        let mut s = Http1Stream::new(false);
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].body, b"hello");
    }

    #[test]
    fn chunked_with_trailer_then_keep_alive_next_message() {
        // after a chunked response with a trailer, the next keep-alive message should parse correctly
        let mut buf = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhello\r\n0\r\nX-Trace: abc\r\n\r\n".to_vec();
        buf.extend_from_slice(b"HTTP/1.1 201 Created\r\nContent-Length: 2\r\n\r\nok");
        let mut s = Http1Stream::new(false);
        let msgs = s.feed(&buf);
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].body, b"hello");
        assert_eq!(msgs[1].status, Some(201));
        assert_eq!(msgs[1].body, b"ok");
    }

    #[test]
    fn flush_truncated_returns_partial_body() {
        // a response with neither Content-Length nor chunked → flush on close
        let mut s = Http1Stream::new(false);
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
        let mut s = Http1Stream::new(false);
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

        let mut s = Http1Stream::new(false);
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
        let mut s = Http1Stream::new(false);
        let msgs = s.feed(raw);
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].status, Some(204));
        assert_eq!(msgs[0].body, b"");
        assert_eq!(msgs[0].header_len, raw.len());
    }
}
