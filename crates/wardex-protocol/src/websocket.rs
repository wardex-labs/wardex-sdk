//! WebSocket (RFC 6455) incremental frame parser (library/transport-layer agnostic).
//! For one direction (client or server) of a stream. Handles split arrival, multiple frames, and fragment reassembly.
//! Content samples are kept only up to the configured sample cap per message (the per-direction cumulative cap is owned by the caller).

use wardex_limits::Limits;

/// A frame's 4-bit opcode (RFC 6455 section 5.2).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsOpcode {
    /// `0x0`: a later fragment of the data message the last Text or Binary
    /// frame started.
    Continuation,
    /// `0x1`: the first frame of a UTF-8 text message.
    Text,
    /// `0x2`: the first frame of a binary message.
    Binary,
    /// `0x8`: the sender is closing; the first two payload bytes, when
    /// present, are the close code.
    Close,
    /// `0x9`: a keepalive the peer answers with a Pong.
    Ping,
    /// `0xA`: the answer to a Ping, or an unsolicited heartbeat.
    Pong,
    /// An opcode RFC 6455 leaves unassigned, carrying its 4-bit value. The
    /// frame is still parsed and counted but never starts a message.
    Reserved(u8),
}

impl WsOpcode {
    fn from_u8(v: u8) -> Self {
        match v {
            0x0 => WsOpcode::Continuation,
            0x1 => WsOpcode::Text,
            0x2 => WsOpcode::Binary,
            0x8 => WsOpcode::Close,
            0x9 => WsOpcode::Ping,
            0xA => WsOpcode::Pong,
            other => WsOpcode::Reserved(other),
        }
    }
    /// String exposed to Python.
    pub fn as_str(&self) -> &'static str {
        match self {
            WsOpcode::Continuation => "continuation",
            WsOpcode::Text => "text",
            WsOpcode::Binary => "binary",
            WsOpcode::Close => "close",
            WsOpcode::Ping => "ping",
            WsOpcode::Pong => "pong",
            WsOpcode::Reserved(_) => "reserved",
        }
    }
}

/// Header facts of one complete frame. The payload itself is not kept here;
/// data frames contribute to [`WsFeedResult::messages`] instead.
#[derive(Debug, Clone)]
pub struct WsFrame {
    /// The FIN bit: this frame ends its message.
    pub fin: bool,
    /// The frame type, including control frames and reserved opcodes, which
    /// are reported here rather than dropped.
    pub opcode: WsOpcode,
    /// The MASK bit. RFC 6455 requires it on every client-to-server frame and
    /// forbids it on server-to-client frames; the parser records it without
    /// enforcing either rule.
    pub masked: bool,
    /// Payload length in bytes as declared on the wire, before any sample
    /// cap. Never above `max_ws_frame_bytes`: a larger frame disables the
    /// parser instead of being reported.
    pub payload_len: u64,
    /// The close code of a Close frame. `None` for every other opcode, and for
    /// a Close frame whose payload is shorter than two bytes or whose sample
    /// cap is below two bytes.
    pub close_code: Option<u16>,
}

/// Everything one [`WsParser::feed`] call completed.
#[derive(Default)]
pub struct WsFeedResult {
    /// Every frame completed by this call, in wire order, control frames
    /// included.
    pub frames: Vec<WsFrame>,
    /// Unmasked payloads of every data message completed by this call, in
    /// order. Each holds at most `ws_sample_bytes` bytes: the start of the
    /// message, cut at the cap. A message whose final frame has not arrived
    /// yet appears in a later call.
    pub messages: Vec<Vec<u8>>,
}

impl std::fmt::Debug for WsFeedResult {
    /// Prints message sizes, never their bytes: messages are captured traffic.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message_lens: Vec<usize> = self.messages.iter().map(Vec::len).collect();
        f.debug_struct("WsFeedResult")
            .field("frames", &self.frames)
            .field("message_lens", &message_lens)
            .finish()
    }
}

enum ParseStep {
    Parsed,
    NeedMore,
    Error,
}

/// Incremental frame parser for one direction of a WebSocket connection.
///
/// A frame whose declared length exceeds `max_ws_frame_bytes` disables the
/// parser for good (see [`WsParser::is_disabled`]): the framing after it can
/// no longer be trusted, and buffering it would be unbounded.
pub struct WsParser {
    buf: Vec<u8>,
    /// Read cursor into `buf`: bytes before it belong to frames already
    /// parsed. The prefix is dropped once per `feed`, not once per frame, so a
    /// read holding many small frames costs one pass over it rather than one
    /// memmove of the rest of the read per frame.
    pos: usize,
    /// Bytes copied into `buf` or relocated inside it — see `bytes_moved`.
    moved: u64,
    frag_opcode: Option<WsOpcode>,
    frag_payload: Vec<u8>,
    disabled: bool,
    limits: Limits,
}

impl std::fmt::Debug for WsParser {
    /// Prints sizes and state only: the buffers hold captured traffic.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WsParser")
            .field("buffered_len", &self.avail().len())
            .field("bytes_moved", &self.moved)
            .field("frag_opcode", &self.frag_opcode)
            .field("frag_payload_len", &self.frag_payload.len())
            .field("disabled", &self.disabled)
            .finish_non_exhaustive()
    }
}

impl Default for WsParser {
    fn default() -> Self {
        Self::new(Limits::default())
    }
}

impl WsParser {
    /// A parser with nothing buffered. `limits.max_ws_frame_bytes` is the
    /// largest frame accepted and `limits.ws_sample_bytes` the most payload
    /// kept per message.
    pub fn new(limits: Limits) -> Self {
        Self {
            buf: Vec::new(),
            pos: 0,
            moved: 0,
            frag_opcode: None,
            frag_payload: Vec::new(),
            disabled: false,
            limits,
        }
    }

    /// Appends `data` and parses every frame it completes. Bytes of an
    /// incomplete frame stay buffered for the next call. Once disabled, every
    /// call returns an empty result and buffers nothing.
    #[must_use = "the returned frames and messages are the only record of the bytes this call consumed"]
    pub fn feed(&mut self, data: &[u8]) -> WsFeedResult {
        let mut result = WsFeedResult::default();
        if self.disabled {
            return result;
        }
        self.buf.extend_from_slice(data);
        self.moved += data.len() as u64;
        loop {
            match self.try_parse_one(&mut result) {
                ParseStep::Parsed => {}
                ParseStep::NeedMore => break,
                ParseStep::Error => {
                    self.disabled = true;
                    self.buf = Vec::new();
                    self.pos = 0;
                    break;
                }
            }
        }
        self.compact();
        result
    }

    /// Bytes copied into the parser's buffer or relocated inside it, counted
    /// since construction: every fed byte once on arrival, plus the residue a
    /// compaction moves to the front. Each byte is moved at most once, because
    /// what a compaction moves is the start of a frame still incomplete, and
    /// the next compaction that runs is after that frame completes — so the
    /// count stays under twice the input. A parser that drops each parsed
    /// frame from the front of the buffer moves the rest of the read once per
    /// frame, and this count is what grows quadratically then.
    pub fn bytes_moved(&self) -> u64 {
        self.moved
    }

    /// Bytes fed but not yet consumed by a parsed frame.
    fn avail(&self) -> &[u8] {
        &self.buf[self.pos..]
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
        self.pos = 0;
    }

    /// Whether disabled (desync/oversize) — used by the tracker to decide markers.
    pub fn is_disabled(&self) -> bool {
        self.disabled
    }

    fn try_parse_one(&mut self, result: &mut WsFeedResult) -> ParseStep {
        let buf = self.avail();
        if buf.len() < 2 {
            return ParseStep::NeedMore;
        }
        let b0 = buf[0];
        let b1 = buf[1];
        let fin = b0 & 0x80 != 0;
        let opcode = WsOpcode::from_u8(b0 & 0x0f);
        let masked = b1 & 0x80 != 0;
        let len7 = u64::from(b1 & 0x7f);

        let mut offset = 2usize;
        let payload_len: u64 = if len7 < 126 {
            len7
        } else if len7 == 126 {
            if buf.len() < offset + 2 {
                return ParseStep::NeedMore;
            }
            let l = u64::from(u16::from_be_bytes([buf[offset], buf[offset + 1]]));
            offset += 2;
            l
        } else {
            if buf.len() < offset + 8 {
                return ParseStep::NeedMore;
            }
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&buf[offset..offset + 8]);
            offset += 8;
            u64::from_be_bytes(arr)
        };

        // Oversized frame defense — blocks unbounded buffering. A length the
        // address space cannot hold is oversized by definition.
        let Ok(plen) = usize::try_from(payload_len) else {
            return ParseStep::Error;
        };
        if plen > self.limits.max_ws_frame_bytes {
            return ParseStep::Error;
        }

        let mask_key: Option<[u8; 4]> = if masked {
            if buf.len() < offset + 4 {
                return ParseStep::NeedMore;
            }
            let mut k = [0u8; 4];
            k.copy_from_slice(&buf[offset..offset + 4]);
            offset += 4;
            Some(k)
        } else {
            None
        };

        if buf.len() < offset + plen {
            return ParseStep::NeedMore;
        }

        // Unmask — copy separately only up to the sample cap (wire length is preserved in payload_len).
        let raw = &buf[offset..offset + plen];
        let take = plen.min(self.limits.ws_sample_bytes);
        let mut payload = Vec::with_capacity(take);
        if let Some(k) = mask_key {
            for (i, &byte) in raw[..take].iter().enumerate() {
                payload.push(byte ^ k[i & 3]);
            }
        } else {
            payload.extend_from_slice(&raw[..take]);
        }

        let close_code = if matches!(opcode, WsOpcode::Close) && payload.len() >= 2 {
            Some(u16::from_be_bytes([payload[0], payload[1]]))
        } else {
            None
        };

        // Message reassembly — data frames only. Control frames (ping/pong/close) are not messages.
        match opcode {
            WsOpcode::Text | WsOpcode::Binary => {
                self.frag_opcode = Some(opcode.clone());
                self.frag_payload.clear();
                self.append_frag(&payload);
                if fin {
                    result.messages.push(std::mem::take(&mut self.frag_payload));
                    self.frag_opcode = None;
                }
            }
            WsOpcode::Continuation if self.frag_opcode.is_some() => {
                self.append_frag(&payload);
                if fin {
                    result.messages.push(std::mem::take(&mut self.frag_payload));
                    self.frag_opcode = None;
                }
            }
            _ => {}
        }

        result.frames.push(WsFrame {
            fin,
            opcode,
            masked,
            payload_len,
            close_code,
        });
        self.pos += offset + plen;
        ParseStep::Parsed
    }

    fn append_frag(&mut self, payload: &[u8]) {
        let room = self
            .limits
            .ws_sample_bytes
            .saturating_sub(self.frag_payload.len());
        let take = payload.len().min(room);
        self.frag_payload.extend_from_slice(&payload[..take]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only frame serialization. Masks the payload if mask_key is Some.
    fn frame(fin: bool, opcode: u8, mask_key: Option<[u8; 4]>, payload: &[u8]) -> Vec<u8> {
        let mut v = Vec::new();
        v.push(if fin { 0x80 } else { 0x00 } | opcode);
        let masked_bit = if mask_key.is_some() { 0x80 } else { 0x00 };
        let n = payload.len();
        if let Ok(short @ 0..=125) = u8::try_from(n) {
            v.push(masked_bit | short);
        } else if let Ok(medium) = u16::try_from(n) {
            v.push(masked_bit | 126);
            v.extend_from_slice(&medium.to_be_bytes());
        } else {
            v.push(masked_bit | 127);
            v.extend_from_slice(&(n as u64).to_be_bytes());
        }
        if let Some(k) = mask_key {
            v.extend_from_slice(&k);
            for (i, &b) in payload.iter().enumerate() {
                v.push(b ^ k[i & 3]);
            }
        } else {
            v.extend_from_slice(payload);
        }
        v
    }

    #[test]
    fn parses_unmasked_text() {
        let mut p = WsParser::new(Limits::default());
        let r = p.feed(&frame(true, 0x1, None, b"hello"));
        assert_eq!(r.frames.len(), 1);
        assert_eq!(r.frames[0].opcode, WsOpcode::Text);
        assert!(r.frames[0].fin);
        assert_eq!(r.frames[0].payload_len, 5);
        assert_eq!(r.messages, vec![b"hello".to_vec()]);
    }

    #[test]
    fn unmasks_client_frame() {
        let mut p = WsParser::new(Limits::default());
        let r = p.feed(&frame(true, 0x1, Some([0x01, 0x02, 0x03, 0x04]), b"hello"));
        assert_eq!(r.messages, vec![b"hello".to_vec()]);
        assert!(r.frames[0].masked);
    }

    #[test]
    fn reassembles_fragmented_message() {
        let mut p = WsParser::new(Limits::default());
        let mut buf = frame(false, 0x1, None, b"he"); // text, FIN=0
        buf.extend(frame(false, 0x0, None, b"ll")); // continuation, FIN=0
        buf.extend(frame(true, 0x0, None, b"o")); // continuation, FIN=1
        let r = p.feed(&buf);
        assert_eq!(r.messages, vec![b"hello".to_vec()]);
    }

    #[test]
    fn handles_split_arrival() {
        let mut p = WsParser::new(Limits::default());
        let raw = frame(true, 0x2, None, b"abcdef"); // binary
        assert!(p.feed(&raw[..3]).frames.is_empty());
        let r = p.feed(&raw[3..]);
        assert_eq!(r.frames.len(), 1);
        assert_eq!(r.messages, vec![b"abcdef".to_vec()]);
    }

    #[test]
    fn extracts_close_code() {
        let mut p = WsParser::new(Limits::default());
        // close payload: 2-byte code (1000=normal) + reason
        let mut payload = 1000u16.to_be_bytes().to_vec();
        payload.extend_from_slice(b"bye");
        let r = p.feed(&frame(true, 0x8, None, &payload));
        assert_eq!(r.frames[0].opcode, WsOpcode::Close);
        assert_eq!(r.frames[0].close_code, Some(1000));
        // close is not a message, so messages should be empty
        assert!(r.messages.is_empty());
    }

    #[test]
    fn ping_pong_are_not_messages() {
        let mut p = WsParser::new(Limits::default());
        let mut buf = frame(true, 0x9, None, b"p"); // ping
        buf.extend(frame(true, 0xA, None, b"p")); // pong
        let r = p.feed(&buf);
        assert_eq!(r.frames.len(), 2);
        assert!(r.messages.is_empty());
        assert_eq!(r.frames[0].opcode, WsOpcode::Ping);
        assert_eq!(r.frames[1].opcode, WsOpcode::Pong);
    }

    #[test]
    fn oversize_frame_disables() {
        let mut p = WsParser::new(Limits::default());
        // claim a 64-bit length that exceeds the configured max WS frame size
        let mut hdr = vec![0x82u8, 127];
        hdr.extend_from_slice(&(2 * 1024 * 1024u64).to_be_bytes());
        let r = p.feed(&hdr);
        assert!(r.frames.is_empty());
        assert!(p.is_disabled());
    }

    #[test]
    fn sample_capped_but_length_counted() {
        let mut p = WsParser::new(Limits::default());
        let big = vec![b'x'; Limits::default().ws_sample_bytes + 100];
        let r = p.feed(&frame(true, 0x2, None, &big));
        assert_eq!(
            r.frames[0].payload_len,
            (Limits::default().ws_sample_bytes + 100) as u64
        ); // full wire length
        assert_eq!(r.messages[0].len(), Limits::default().ws_sample_bytes); // sample only up to the cap
    }

    #[test]
    fn frame_cap_comes_from_limits() {
        let limits = Limits {
            max_ws_frame_bytes: 2,
            ..Default::default()
        };
        let mut s = WsParser::new(limits);
        // 5-byte payload text frame, unmasked
        let r = s.feed(b"\x81\x05hello");
        assert!(r.frames.is_empty(), "frame above the cap must be rejected");
    }

    #[test]
    fn a_read_of_many_small_frames_is_linear_not_quadratic() {
        // 256 KiB of 2-byte frames in a single read: dropping each frame from
        // the front of the buffer would move the rest of the read 131072 times, about
        // 17 GB of memmove. Moving each byte at most once after it arrives
        // keeps the count under twice the input.
        let raw = frame(true, 0x9, None, b"").repeat(128 * 1024);
        let mut p = WsParser::new(Limits::default());
        let r = p.feed(&raw);
        assert_eq!(r.frames.len(), 128 * 1024);
        assert!(
            p.bytes_moved() <= 2 * raw.len() as u64,
            "moved {} bytes for a {}-byte read",
            p.bytes_moved(),
            raw.len()
        );
    }

    #[test]
    fn residue_split_across_reads_is_still_linear_and_intact() {
        // The same stream in 1000-byte reads leaves a torn frame at the end of
        // nearly every read; that residue is moved once and then completed.
        let one = frame(true, 0x1, Some([1, 2, 3, 4]), b"hello world, 25 bytes...");
        let raw = one.repeat(4096);
        let mut p = WsParser::new(Limits::default());
        let mut messages = 0;
        for part in raw.chunks(1000) {
            messages += p.feed(part).messages.len();
        }
        assert_eq!(messages, 4096);
        assert!(p.bytes_moved() <= 2 * raw.len() as u64);
    }

    #[test]
    fn debug_output_shows_sizes_not_payload_bytes() {
        let secret = b"planted-secret-7f3a";
        let mut raw = frame(true, 0x1, Some([9, 8, 7, 6]), secret);
        // A second, incomplete frame leaves captured bytes in the buffer and
        // in the fragment being reassembled.
        raw.extend_from_slice(&frame(false, 0x1, None, secret));
        raw.extend_from_slice(&frame(true, 0x0, None, secret)[..6]);
        let mut p = WsParser::new(Limits::default());
        let r = p.feed(&raw);
        assert_eq!(r.messages, vec![secret.to_vec()]);
        let shown = format!("{r:?} {p:?}");
        assert!(!shown.contains("planted-secret-7f3a"), "{shown}");
        assert!(!shown.contains("112, 108, 97"), "{shown}");
        assert!(shown.contains("message_lens"), "{shown}");
    }
}
