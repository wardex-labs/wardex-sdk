//! WebSocket (RFC 6455) incremental frame parser (library/transport-layer agnostic).
//! For one direction (client or server) of a stream. Handles split arrival, multiple frames, and fragment reassembly.
//! Content samples are kept only up to the configured sample cap per message (the per-direction cumulative cap is owned by the caller).

use wardex_limits::Limits;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsOpcode {
    Continuation,
    Text,
    Binary,
    Close,
    Ping,
    Pong,
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

#[derive(Debug, Clone)]
pub struct WsFrame {
    pub fin: bool,
    pub opcode: WsOpcode,
    pub masked: bool,
    pub payload_len: u64,        // full wire payload length (for aggregation)
    pub close_code: Option<u16>, // set when opcode==Close
}

#[derive(Default)]
pub struct WsFeedResult {
    pub frames: Vec<WsFrame>,
    pub messages: Vec<Vec<u8>>, // reassembled messages (single message ≤ the configured sample cap)
}

enum ParseStep {
    Parsed,
    NeedMore,
    Error,
}

pub struct WsParser {
    buf: Vec<u8>,
    frag_opcode: Option<WsOpcode>,
    frag_payload: Vec<u8>,
    disabled: bool,
    limits: Limits,
}

impl Default for WsParser {
    fn default() -> Self {
        Self::new(Limits::default())
    }
}

impl WsParser {
    pub fn new(limits: Limits) -> Self {
        Self {
            buf: Vec::new(),
            frag_opcode: None,
            frag_payload: Vec::new(),
            disabled: false,
            limits,
        }
    }

    pub fn feed(&mut self, data: &[u8]) -> WsFeedResult {
        let mut result = WsFeedResult::default();
        if self.disabled {
            return result;
        }
        self.buf.extend_from_slice(data);
        loop {
            match self.try_parse_one(&mut result) {
                ParseStep::Parsed => continue,
                ParseStep::NeedMore => break,
                ParseStep::Error => {
                    self.disabled = true;
                    self.buf.clear();
                    break;
                }
            }
        }
        result
    }

    /// Whether disabled (desync/oversize) — used by the tracker to decide markers.
    pub fn is_disabled(&self) -> bool {
        self.disabled
    }

    fn try_parse_one(&mut self, result: &mut WsFeedResult) -> ParseStep {
        if self.buf.len() < 2 {
            return ParseStep::NeedMore;
        }
        let b0 = self.buf[0];
        let b1 = self.buf[1];
        let fin = b0 & 0x80 != 0;
        let opcode = WsOpcode::from_u8(b0 & 0x0f);
        let masked = b1 & 0x80 != 0;
        let len7 = (b1 & 0x7f) as u64;

        let mut offset = 2usize;
        let payload_len: u64 = if len7 < 126 {
            len7
        } else if len7 == 126 {
            if self.buf.len() < offset + 2 {
                return ParseStep::NeedMore;
            }
            let l = u16::from_be_bytes([self.buf[offset], self.buf[offset + 1]]) as u64;
            offset += 2;
            l
        } else {
            if self.buf.len() < offset + 8 {
                return ParseStep::NeedMore;
            }
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&self.buf[offset..offset + 8]);
            offset += 8;
            u64::from_be_bytes(arr)
        };

        // Oversized frame defense — blocks unbounded buffering.
        if payload_len > self.limits.max_ws_frame_bytes as u64 {
            return ParseStep::Error;
        }

        let mask_key: Option<[u8; 4]> = if masked {
            if self.buf.len() < offset + 4 {
                return ParseStep::NeedMore;
            }
            let mut k = [0u8; 4];
            k.copy_from_slice(&self.buf[offset..offset + 4]);
            offset += 4;
            Some(k)
        } else {
            None
        };

        let plen = payload_len as usize;
        if self.buf.len() < offset + plen {
            return ParseStep::NeedMore;
        }

        // Unmask — copy separately only up to the sample cap (wire length is preserved in payload_len).
        let raw = &self.buf[offset..offset + plen];
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
        self.buf.drain(..offset + plen);
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
        if n < 126 {
            v.push(masked_bit | n as u8);
        } else if n <= 0xffff {
            v.push(masked_bit | 126);
            v.extend_from_slice(&(n as u16).to_be_bytes());
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
}
