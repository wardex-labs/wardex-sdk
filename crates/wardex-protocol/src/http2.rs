//! Incremental HTTP/2 parser — frame carving + HPACK decode + stream reassembly.
//! The frame decoder carves out the 9-byte header + payload; Http2Connection then assigns meaning.

/// A carved raw HTTP/2 frame.
#[derive(Debug, Clone)]
pub struct Http2Frame {
    pub frame_type: u8,
    pub flags: u8,
    pub stream_id: u32,
    pub payload: Vec<u8>,
}

/// Incremental decoder that accumulates bytes for one direction and carves out completed frames.
/// (The connection preface is stripped beforehand by Http2Connection.)
pub struct Http2FrameDecoder {
    buf: Vec<u8>,
}

impl Default for Http2FrameDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl Http2FrameDecoder {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    pub fn feed(&mut self, data: &[u8]) -> Vec<Http2Frame> {
        self.buf.extend_from_slice(data);
        let mut out = Vec::new();
        loop {
            if self.buf.len() < 9 {
                break;
            }
            let length = ((self.buf[0] as usize) << 16)
                | ((self.buf[1] as usize) << 8)
                | (self.buf[2] as usize);
            let total = 9 + length;
            if self.buf.len() < total {
                break;
            }
            let frame_type = self.buf[3];
            let flags = self.buf[4];
            let stream_id = (((self.buf[5] & 0x7f) as u32) << 24)
                | ((self.buf[6] as u32) << 16)
                | ((self.buf[7] as u32) << 8)
                | (self.buf[8] as u32);
            let payload = self.buf[9..total].to_vec();
            out.push(Http2Frame {
                frame_type,
                flags,
                stream_id,
                payload,
            });
            self.buf.drain(..total);
        }
        out
    }
}

use fluke_hpack::Decoder;
use std::collections::HashMap;

const FRAME_DATA: u8 = 0x0;
const FRAME_HEADERS: u8 = 0x1;
const FRAME_RST_STREAM: u8 = 0x3;
const FRAME_PUSH_PROMISE: u8 = 0x5;
const FRAME_CONTINUATION: u8 = 0x9;

const FLAG_END_STREAM: u8 = 0x1;
const FLAG_END_HEADERS: u8 = 0x4;
const FLAG_PADDED: u8 = 0x8;
const FLAG_PRIORITY: u8 = 0x20;

const PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";
const MAX_BODY: usize = 8 * 1024 * 1024;
const MAX_STREAMS: usize = 1024;

/// A completed request/response transaction for one stream.
pub struct Http2Transaction {
    pub stream_id: u32,
    pub method: String,
    pub path: String,
    pub status: u16,
    pub content_type: Option<String>,
    pub grpc_status: Option<i32>,
    pub grpc_message: Option<String>,
    pub request_body: Vec<u8>,
    pub response_body: Vec<u8>,
    pub truncated: bool,
}

/// Result of a single feed call: streams whose request just ended + transactions whose response also ended.
#[derive(Default)]
pub struct Http2FeedResult {
    pub opened_request_streams: Vec<u32>,
    pub transactions: Vec<Http2Transaction>,
}

#[derive(Default)]
struct StreamState {
    method: Option<String>,
    path: Option<String>,
    status: Option<u16>,
    content_type: Option<String>,
    grpc_status: Option<i32>,
    grpc_message: Option<String>,
    req_body: Vec<u8>,
    resp_body: Vec<u8>,
    req_ended: bool,
    resp_ended: bool,
    truncated: bool,
}

struct Pending {
    from_client: bool,
    end_stream: bool,
    buf: Vec<u8>,
}

/// Parser for one h2 connection (bidirectional). 2x frame decoders + 2x HPACK decoders + stream map.
pub struct Http2Connection {
    client_frames: Http2FrameDecoder,
    server_frames: Http2FrameDecoder,
    req_decoder: Decoder<'static>,
    resp_decoder: Decoder<'static>,
    streams: HashMap<u32, StreamState>,
    pending: HashMap<u32, Pending>,
    preface_buf: Vec<u8>,
    preface_seen: bool,
    disabled: bool,
}

impl Default for Http2Connection {
    fn default() -> Self {
        Self::new()
    }
}

impl Http2Connection {
    pub fn new() -> Self {
        Self {
            client_frames: Http2FrameDecoder::new(),
            server_frames: Http2FrameDecoder::new(),
            req_decoder: Decoder::new(),
            resp_decoder: Decoder::new(),
            streams: HashMap::new(),
            pending: HashMap::new(),
            preface_buf: Vec::new(),
            preface_seen: false,
            disabled: false,
        }
    }

    pub fn feed(&mut self, from_client: bool, data: &[u8]) -> Http2FeedResult {
        let mut result = Http2FeedResult::default();
        if self.disabled {
            return result;
        }
        let frames = if from_client {
            match self.strip_preface(data) {
                Some(rest) => self.client_frames.feed(&rest),
                None => return result,
            }
        } else {
            self.server_frames.feed(data)
        };
        for frame in frames {
            if self.handle_frame(from_client, frame, &mut result).is_err() {
                self.disabled = true;
                break;
            }
        }
        result
    }

    /// Strips the connection preface (24 bytes) from the client's first bytes. Handles split arrival.
    /// If the bytes don't match the actual preface, pass through immediately (middlebox/test compatibility).
    fn strip_preface(&mut self, data: &[u8]) -> Option<Vec<u8>> {
        if self.preface_seen {
            return Some(data.to_vec());
        }
        self.preface_buf.extend_from_slice(data);
        let n = self.preface_buf.len().min(PREFACE.len());
        if self.preface_buf[..n] != PREFACE[..n] {
            // Not a preface — return the whole buffer immediately
            self.preface_seen = true;
            return Some(std::mem::take(&mut self.preface_buf));
        }
        if self.preface_buf.len() >= PREFACE.len() {
            // Preface fully matched → strip it and return the rest
            self.preface_seen = true;
            let rest = self.preface_buf.split_off(PREFACE.len());
            self.preface_buf.clear();
            return Some(rest);
        }
        // Matches the preface prefix but still under 24 bytes → wait for more
        None
    }

    fn handle_frame(
        &mut self,
        from_client: bool,
        frame: Http2Frame,
        result: &mut Http2FeedResult,
    ) -> Result<(), ()> {
        match frame.frame_type {
            FRAME_HEADERS => self.on_headers(from_client, frame, result),
            FRAME_CONTINUATION => self.on_continuation(frame, result),
            FRAME_DATA => {
                self.on_data(from_client, frame, result);
                Ok(())
            }
            FRAME_PUSH_PROMISE => self.on_push_promise(frame),
            FRAME_RST_STREAM => {
                self.streams.remove(&frame.stream_id);
                self.pending.remove(&frame.stream_id);
                Ok(())
            }
            _ => Ok(()), // Skip SETTINGS/PRIORITY/WINDOW_UPDATE/PING/GOAWAY, etc.
        }
    }

    fn on_headers(
        &mut self,
        from_client: bool,
        frame: Http2Frame,
        result: &mut Http2FeedResult,
    ) -> Result<(), ()> {
        let mut payload = &frame.payload[..];
        let mut pad_len = 0usize;
        if frame.flags & FLAG_PADDED != 0 {
            if payload.is_empty() {
                return Err(());
            }
            pad_len = payload[0] as usize;
            payload = &payload[1..];
        }
        if frame.flags & FLAG_PRIORITY != 0 {
            if payload.len() < 5 {
                return Err(());
            }
            payload = &payload[5..];
        }
        if payload.len() < pad_len {
            return Err(());
        }
        let block = &payload[..payload.len() - pad_len];
        let end_stream = frame.flags & FLAG_END_STREAM != 0;

        if frame.flags & FLAG_END_HEADERS != 0 {
            self.decode_block(from_client, frame.stream_id, block, end_stream, result)
        } else {
            self.pending.insert(
                frame.stream_id,
                Pending {
                    from_client,
                    end_stream,
                    buf: block.to_vec(),
                },
            );
            Ok(())
        }
    }

    fn on_continuation(
        &mut self,
        frame: Http2Frame,
        result: &mut Http2FeedResult,
    ) -> Result<(), ()> {
        let mut p = match self.pending.remove(&frame.stream_id) {
            Some(p) => p,
            None => return Err(()), // CONTINUATION without HEADERS → desync
        };
        p.buf.extend_from_slice(&frame.payload);
        if frame.flags & FLAG_END_HEADERS != 0 {
            let block = std::mem::take(&mut p.buf);
            self.decode_block(p.from_client, frame.stream_id, &block, p.end_stream, result)
        } else {
            self.pending.insert(frame.stream_id, p);
            Ok(())
        }
    }

    fn decode_block(
        &mut self,
        from_client: bool,
        stream_id: u32,
        block: &[u8],
        end_stream: bool,
        result: &mut Http2FeedResult,
    ) -> Result<(), ()> {
        // HPACK decode — scope the decoder's mutable borrow to this block, then access streams
        let headers = {
            let decoder = if from_client {
                &mut self.req_decoder
            } else {
                &mut self.resp_decoder
            };
            decoder.decode(block).map_err(|_| ())?
        };
        if self.streams.len() > MAX_STREAMS {
            if let Some(&k) = self.streams.keys().next() {
                self.streams.remove(&k);
            }
        }
        let st = self.streams.entry(stream_id).or_default();
        for (name, value) in headers {
            match name.as_slice() {
                b":method" => st.method = Some(String::from_utf8_lossy(&value).into_owned()),
                b":path" => st.path = Some(String::from_utf8_lossy(&value).into_owned()),
                b":status" => {
                    st.status = String::from_utf8_lossy(&value).trim().parse().ok();
                }
                b"content-type" => {
                    st.content_type = Some(String::from_utf8_lossy(&value).into_owned());
                }
                b"grpc-status" => {
                    st.grpc_status = String::from_utf8_lossy(&value).trim().parse().ok();
                }
                b"grpc-message" => {
                    st.grpc_message = Some(String::from_utf8_lossy(&value).into_owned());
                }
                _ => {}
            }
        }
        if end_stream {
            self.mark_ended(from_client, stream_id, result);
        }
        Ok(())
    }

    fn on_data(&mut self, from_client: bool, frame: Http2Frame, result: &mut Http2FeedResult) {
        let mut payload = &frame.payload[..];
        let mut pad_len = 0usize;
        if frame.flags & FLAG_PADDED != 0 {
            if payload.is_empty() {
                return;
            }
            pad_len = payload[0] as usize;
            payload = &payload[1..];
        }
        if payload.len() < pad_len {
            return;
        }
        let body = &payload[..payload.len() - pad_len];
        {
            let st = self.streams.entry(frame.stream_id).or_default();
            let target = if from_client {
                &mut st.req_body
            } else {
                &mut st.resp_body
            };
            let room = MAX_BODY.saturating_sub(target.len());
            if body.len() <= room {
                target.extend_from_slice(body);
            } else {
                target.extend_from_slice(&body[..room]);
                st.truncated = true;
            }
        }
        if frame.flags & FLAG_END_STREAM != 0 {
            self.mark_ended(from_client, frame.stream_id, result);
        }
    }

    fn on_push_promise(&mut self, frame: Http2Frame) -> Result<(), ()> {
        // Feed the header block into the response decoder to keep the table in sync. Push streams aren't emitted.
        let mut payload = &frame.payload[..];
        let mut pad_len = 0usize;
        if frame.flags & FLAG_PADDED != 0 {
            if payload.is_empty() {
                return Err(());
            }
            pad_len = payload[0] as usize;
            payload = &payload[1..];
        }
        if payload.len() < 4 {
            return Err(());
        }
        payload = &payload[4..]; // skip promised stream id
        if payload.len() < pad_len {
            return Err(());
        }
        let block = &payload[..payload.len() - pad_len];
        if frame.flags & FLAG_END_HEADERS != 0 {
            self.resp_decoder.decode(block).map_err(|_| ())?;
            Ok(())
        } else {
            // Push promises continued by CONTINUATION are rare — sync can't be guaranteed, so disable safely
            Err(())
        }
    }

    fn mark_ended(&mut self, from_client: bool, stream_id: u32, result: &mut Http2FeedResult) {
        let st = self.streams.entry(stream_id).or_default();
        if from_client {
            if !st.req_ended {
                st.req_ended = true;
                result.opened_request_streams.push(stream_id);
            }
        } else {
            st.resp_ended = true;
        }
        let done = self
            .streams
            .get(&stream_id)
            .map(|s| s.resp_ended)
            .unwrap_or(false);
        if done {
            if let Some(s) = self.streams.remove(&stream_id) {
                result.transactions.push(Http2Transaction {
                    stream_id,
                    method: s.method.unwrap_or_default(),
                    path: s.path.unwrap_or_default(),
                    status: s.status.unwrap_or(0),
                    content_type: s.content_type,
                    grpc_status: s.grpc_status,
                    grpc_message: s.grpc_message,
                    request_body: s.req_body,
                    response_body: s.resp_body,
                    truncated: s.truncated,
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(ftype: u8, flags: u8, stream_id: u32, payload: &[u8]) -> Vec<u8> {
        let len = payload.len();
        let mut f = vec![
            (len >> 16) as u8,
            (len >> 8) as u8,
            len as u8,
            ftype,
            flags,
            (stream_id >> 24) as u8,
            (stream_id >> 16) as u8,
            (stream_id >> 8) as u8,
            stream_id as u8,
        ];
        f.extend_from_slice(payload);
        f
    }

    #[test]
    fn carves_single_frame() {
        let mut d = Http2FrameDecoder::new();
        let frames = d.feed(&frame(0x1, 0x4, 1, b"hello"));
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].frame_type, 0x1);
        assert_eq!(frames[0].flags, 0x4);
        assert_eq!(frames[0].stream_id, 1);
        assert_eq!(frames[0].payload, b"hello");
    }

    #[test]
    fn masks_reserved_bit_in_stream_id() {
        let mut d = Http2FrameDecoder::new();
        // Even if the top (R) bit of the stream_id byte is set to 1, it must be masked off
        let mut raw = frame(0x0, 0x0, 1, b"x");
        raw[5] |= 0x80;
        let frames = d.feed(&raw);
        assert_eq!(frames[0].stream_id, 1);
    }

    #[test]
    fn handles_split_arrival() {
        let mut d = Http2FrameDecoder::new();
        let raw = frame(0x0, 0x1, 3, b"abcdef");
        assert_eq!(d.feed(&raw[..4]).len(), 0); // not even the header arrives
        assert_eq!(d.feed(&raw[4..10]).len(), 0); // partial payload
        let frames = d.feed(&raw[10..]);
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].payload, b"abcdef");
    }

    #[test]
    fn carves_multiple_frames_in_one_feed() {
        let mut d = Http2FrameDecoder::new();
        let mut buf = frame(0x1, 0x4, 1, b"aa");
        buf.extend_from_slice(&frame(0x0, 0x1, 1, b"bbbb"));
        let frames = d.feed(&buf);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].payload, b"aa");
        assert_eq!(frames[1].payload, b"bbbb");
    }

    #[test]
    fn empty_payload_ok() {
        let mut d = Http2FrameDecoder::new();
        let frames = d.feed(&frame(0x4, 0x0, 0, b""));
        assert_eq!(frames.len(), 1);
        assert!(frames[0].payload.is_empty());
    }

    #[test]
    fn captures_grpc_content_type_and_trailers() {
        let mut c = Http2Connection::new();

        // client: HEADERS(:method POST, :path /echo.Echo/Say, content-type application/grpc)
        //       + DATA(1 gRPC message, END_STREAM)
        let req_block = hpack(&[
            (b":method", b"POST"),
            (b":path", b"/echo.Echo/Say"),
            (b"content-type", b"application/grpc"),
        ]);
        let mut req = frame(0x1, FH, 1, &req_block);
        // gRPC message: [0][len=3][b"abc"]
        req.extend_from_slice(&frame(0x0, FS, 1, b"\x00\x00\x00\x00\x03abc"));
        let r = c.feed(true, &req);
        assert_eq!(r.opened_request_streams, vec![1]);

        // server: HEADERS(:status 200, content-type) + DATA(message) + trailers HEADERS(grpc-status 0, END_STREAM)
        let resp_hdr = hpack(&[(b":status", b"200"), (b"content-type", b"application/grpc")]);
        let trailers = hpack(&[(b"grpc-status", b"0")]);
        let mut resp = frame(0x1, FH, 1, &resp_hdr);
        resp.extend_from_slice(&frame(0x0, 0x0, 1, b"\x00\x00\x00\x00\x02xy"));
        resp.extend_from_slice(&frame(0x1, FH | FS, 1, &trailers));
        let r = c.feed(false, &resp);

        assert_eq!(r.transactions.len(), 1);
        let t = &r.transactions[0];
        assert_eq!(t.content_type.as_deref(), Some("application/grpc"));
        assert_eq!(t.grpc_status, Some(0));
        assert_eq!(t.path, "/echo.Echo/Say");
        assert_eq!(t.status, 200); // HTTP status is still 200
    }

    #[test]
    // trailers-only: must correctly parse the gRPC error response pattern where the server
    // carries grpc-status on the first HEADERS frame without a DATA frame and closes with END_STREAM.
    fn captures_trailers_only_grpc_error() {
        let mut c = Http2Connection::new();

        // client: HEADERS(POST /pkg.Svc/M, content-type application/grpc, END_HEADERS|END_STREAM)
        // empty-body request — the client side can be simple since we're only reproducing a trailers-only error
        let req_block = hpack(&[
            (b":method", b"POST"),
            (b":path", b"/pkg.Svc/M"),
            (b"content-type", b"application/grpc"),
        ]);
        let r = c.feed(true, &frame(0x1, FH | FS, 1, &req_block));
        assert_eq!(r.opened_request_streams, vec![1]);

        // server: single HEADERS frame with :status 200 + grpc-status 5 + END_STREAM (trailers-only)
        let resp_block = hpack(&[
            (b":status", b"200"),
            (b"content-type", b"application/grpc"),
            (b"grpc-status", b"5"),
        ]);
        let r = c.feed(false, &frame(0x1, FH | FS, 1, &resp_block));

        assert_eq!(r.transactions.len(), 1);
        let t = &r.transactions[0];
        assert_eq!(t.grpc_status, Some(5));
        assert_eq!(t.content_type.as_deref(), Some("application/grpc"));
        assert_eq!(t.status, 200);
    }

    // Build a header block with the HPACK encoder (test-only — round-trips with real encoding)
    fn hpack(headers: &[(&[u8], &[u8])]) -> Vec<u8> {
        let mut enc = fluke_hpack::Encoder::new();
        enc.encode(headers.iter().map(|(n, v)| (*n, *v)))
    }

    const FH: u8 = 0x4; // END_HEADERS
    const FS: u8 = 0x1; // END_STREAM

    #[test]
    fn reassembles_simple_request_response() {
        let mut c = Http2Connection::new();

        // client: HEADERS(:method GET, :path /v1/x, END_HEADERS+END_STREAM)
        let req_block = hpack(&[(b":method", b"GET"), (b":path", b"/v1/x")]);
        let r = c.feed(true, &frame(0x1, FH | FS, 1, &req_block));
        assert_eq!(r.opened_request_streams, vec![1]);
        assert!(r.transactions.is_empty());

        // server: HEADERS(:status 200, END_HEADERS) + DATA("pong", END_STREAM)
        let resp_block = hpack(&[(b":status", b"200")]);
        let mut buf = frame(0x1, FH, 1, &resp_block);
        buf.extend_from_slice(&frame(0x0, FS, 1, b"pong"));
        let r = c.feed(false, &buf);
        assert_eq!(r.transactions.len(), 1);
        let t = &r.transactions[0];
        assert_eq!(t.stream_id, 1);
        assert_eq!(t.method, "GET");
        assert_eq!(t.path, "/v1/x");
        assert_eq!(t.status, 200);
        assert_eq!(t.response_body, b"pong");
    }

    #[test]
    fn request_with_body_emits_after_response() {
        let mut c = Http2Connection::new();
        let req_block = hpack(&[(b":method", b"POST"), (b":path", b"/p")]);
        // HEADERS(END_HEADERS, no END_STREAM) + DATA(body, END_STREAM)
        let mut req = frame(0x1, FH, 1, &req_block);
        req.extend_from_slice(&frame(0x0, FS, 1, b"{\"a\":1}"));
        let r = c.feed(true, &req);
        assert_eq!(r.opened_request_streams, vec![1]);

        let resp_block = hpack(&[(b":status", b"201")]);
        let mut resp = frame(0x1, FH, 1, &resp_block);
        resp.extend_from_slice(&frame(0x0, FS, 1, b"ok"));
        let r = c.feed(false, &resp);
        assert_eq!(r.transactions.len(), 1);
        assert_eq!(r.transactions[0].request_body, b"{\"a\":1}");
        assert_eq!(r.transactions[0].response_body, b"ok");
        assert_eq!(r.transactions[0].status, 201);
    }

    #[test]
    fn multiplexed_streams_correlate_independently() {
        let mut c = Http2Connection::new();
        let b1 = hpack(&[(b":method", b"GET"), (b":path", b"/a")]);
        let b3 = hpack(&[(b":method", b"GET"), (b":path", b"/b")]);
        // interleave requests on stream 1, 3
        let mut req = frame(0x1, FH | FS, 1, &b1);
        req.extend_from_slice(&frame(0x1, FH | FS, 3, &b3));
        let r = c.feed(true, &req);
        assert_eq!(r.opened_request_streams, vec![1, 3]);

        // respond on 3 first, then 1
        let s3 = hpack(&[(b":status", b"404")]);
        let s1 = hpack(&[(b":status", b"200")]);
        let mut resp = frame(0x1, FH | FS, 3, &s3);
        resp.extend_from_slice(&frame(0x1, FH | FS, 1, &s1));
        let r = c.feed(false, &resp);
        assert_eq!(r.transactions.len(), 2);
        let by_id: std::collections::HashMap<u32, u16> = r
            .transactions
            .iter()
            .map(|t| (t.stream_id, t.status))
            .collect();
        assert_eq!(by_id[&3], 404);
        assert_eq!(by_id[&1], 200);
    }

    #[test]
    fn continuation_frames_join_header_block() {
        let mut c = Http2Connection::new();
        let block = hpack(&[(b":method", b"GET"), (b":path", b"/long")]);
        let mid = block.len() / 2;
        // HEADERS(no END_HEADERS) + CONTINUATION(END_HEADERS), END_STREAM on HEADERS
        let mut req = frame(0x1, FS, 1, &block[..mid]);
        req.extend_from_slice(&frame(0x9, FH, 1, &block[mid..]));
        let r = c.feed(true, &req);
        assert_eq!(r.opened_request_streams, vec![1]);

        let s = hpack(&[(b":status", b"200")]);
        let r = c.feed(false, &frame(0x1, FH | FS, 1, &s));
        assert_eq!(r.transactions[0].path, "/long");
    }

    #[test]
    fn skips_unknown_frames_and_preface() {
        let mut c = Http2Connection::new();
        // client preface + SETTINGS + WINDOW_UPDATE + HEADERS
        let mut buf = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n".to_vec();
        buf.extend_from_slice(&frame(0x4, 0x0, 0, b"")); // SETTINGS
        buf.extend_from_slice(&frame(0x8, 0x0, 0, b"\x00\x00\x00\x01")); // WINDOW_UPDATE
        let block = hpack(&[(b":method", b"GET"), (b":path", b"/")]);
        buf.extend_from_slice(&frame(0x1, FH | FS, 1, &block));
        let r = c.feed(true, &buf);
        assert_eq!(r.opened_request_streams, vec![1]);
    }

    #[test]
    fn padded_data_strips_padding() {
        let mut c = Http2Connection::new();
        let req_block = hpack(&[(b":method", b"GET"), (b":path", b"/")]);
        c.feed(true, &frame(0x1, FH | FS, 1, &req_block));
        let s = hpack(&[(b":status", b"200")]);
        // PADDED DATA: [pad_len=3]["body"][3 pad bytes], END_STREAM+PADDED
        let mut payload = vec![3u8];
        payload.extend_from_slice(b"body");
        payload.extend_from_slice(&[0, 0, 0]);
        let mut resp = frame(0x1, FH, 1, &s);
        resp.extend_from_slice(&frame(0x0, FS | 0x8, 1, &payload));
        let r = c.feed(false, &resp);
        assert_eq!(r.transactions[0].response_body, b"body");
    }

    #[test]
    fn desync_disables_capture_without_panic() {
        let mut c = Http2Connection::new();
        // send a bad HPACK block with END_HEADERS → decode fails → disabled
        let r = c.feed(true, &frame(0x1, FH | FS, 1, &[0xff, 0xff, 0xff, 0xff]));
        // passes through without panicking + subsequent normal traffic also isn't captured
        let block = hpack(&[(b":method", b"GET"), (b":path", b"/")]);
        let r2 = c.feed(true, &frame(0x1, FH | FS, 3, &block));
        assert!(r.transactions.is_empty());
        assert!(r2.opened_request_streams.is_empty());
    }
}
