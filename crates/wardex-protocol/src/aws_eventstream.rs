//! AWS event stream (`application/vnd.amazon.eventstream`) framing decoder.
//! Provider-agnostic: it splits a completed body into messages and their
//! headers and says nothing about what a payload means.
//!
//! One message on the wire:
//!
//! ```text
//! total_length   u32 BE   the whole message, these 4 bytes and the CRC included
//! headers_length u32 BE
//! prelude_crc    u32 BE   CRC-32 of the 8 bytes above
//! headers        headers_length bytes
//! payload        total_length - headers_length - 16 bytes
//! message_crc    u32 BE   CRC-32 of everything before it
//! ```
//!
//! A header is `name_len u8, name, value_type u8, value`, where the value's
//! size follows from its type (strings and byte arrays carry a `u16` length).
//!
//! Fail-safe like every parser in this crate: a body that stops making sense
//! stops the decode, and what was not read is counted in
//! [`Decoded::unparsed`] rather than dropped silently.

/// The CRC-32 the framing uses: IEEE 802.3, reflected, polynomial
/// `0xEDB88320`, initial value and final XOR `0xFFFFFFFF` (what zlib's
/// `crc32` computes). A bitwise loop: the bodies are small and the crate
/// takes no dependency for it.
pub fn crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

/// The prelude (12 bytes) plus the trailing message CRC (4 bytes).
const FRAME_OVERHEAD: usize = 16;
const PRELUDE_LEN: usize = 12;

/// One header value, in the type the header declared.
#[derive(Debug, Clone, PartialEq)]
pub enum HeaderValue<'a> {
    /// Header types 0 and 1: the value is the type itself, with no payload.
    Bool(bool),
    /// Header type 2: one signed byte.
    Byte(i8),
    /// Header type 3: a big-endian signed 16-bit integer.
    Short(i16),
    /// Header type 4: a big-endian signed 32-bit integer.
    Int(i32),
    /// Header type 5: a big-endian signed 64-bit integer.
    Long(i64),
    /// Header type 6: a length-prefixed byte string, borrowed from the body.
    Bytes(&'a [u8]),
    /// Header type 7: a length-prefixed UTF-8 string, borrowed from the body.
    /// A value that is not UTF-8 fails the header parse instead.
    String(&'a str),
    /// Header type 8: milliseconds since the Unix epoch.
    Timestamp(i64),
    /// Header type 9: a 16-byte UUID, as laid out on the wire.
    Uuid([u8; 16]),
}

/// One decoded message: its headers in wire order and its raw payload.
#[derive(Debug, Clone, PartialEq)]
pub struct Message<'a> {
    /// Every header in wire order, as `(name, value)`; duplicates kept.
    pub headers: Vec<(&'a str, HeaderValue<'a>)>,
    /// The bytes between the header block and the message CRC, untouched:
    /// for Bedrock an event's JSON, for an exception its error document.
    pub payload: &'a [u8],
}

impl<'a> Message<'a> {
    /// The first string-typed header with this name.
    pub fn header_str(&self, name: &str) -> Option<&'a str> {
        self.headers.iter().find_map(|(n, v)| match v {
            HeaderValue::String(s) if *n == name => Some(*s),
            _ => None,
        })
    }
}

/// A decoded body: the messages that framed and checked out, and how many
/// could not be read.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Decoded<'a> {
    /// Every message whose preludes and CRCs checked out, in wire order.
    pub messages: Vec<Message<'a>>,
    /// Messages skipped, plus one for an unreadable remainder. A message
    /// whose prelude checks out but whose message CRC or headers do not is
    /// skipped and counted, and the decode continues at the next message (its
    /// length was covered by the prelude CRC, so the boundary is trusted). A
    /// remainder that cannot be framed at all (a prelude CRC mismatch, a
    /// length that runs past the body, a tail cut inside a prelude) ends the
    /// decode and counts once: how many messages it held is unknowable.
    pub unparsed: u32,
}

/// Whether `body` begins with one message whose prelude CRC checks out and
/// whose length fits the body: the sniff for a stream sent without the
/// path or content type that would say so. Twelve bytes whose last four are
/// the CRC-32 of the first eight do not happen by accident in a JSON or SSE
/// body.
pub fn starts_with_message(body: &[u8]) -> bool {
    matches!(prelude(body), Some((total, _)) if total <= body.len())
}

/// `(total_length, headers_length)` when the first 12 bytes are a prelude
/// whose CRC checks out and whose lengths are self-consistent.
fn prelude(bytes: &[u8]) -> Option<(usize, usize)> {
    if bytes.len() < PRELUDE_LEN {
        return None;
    }
    let total = u32_at(bytes, 0) as usize;
    let headers = u32_at(bytes, 4) as usize;
    if crc32(&bytes[..8]) != u32_at(bytes, 8) {
        return None;
    }
    if total < FRAME_OVERHEAD || headers > total - FRAME_OVERHEAD {
        return None;
    }
    Some((total, headers))
}

fn u32_at(bytes: &[u8], at: usize) -> u32 {
    u32::from_be_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

/// Splits a completed body into messages. Never panics, whatever the bytes.
pub fn decode(body: &[u8]) -> Decoded<'_> {
    let mut out = Decoded::default();
    let mut rest = body;
    while !rest.is_empty() {
        let Some((total, headers_len)) = prelude(rest) else {
            // Cut inside a prelude, or a prelude that does not check out:
            // the framing is lost from here on.
            out.unparsed = out.unparsed.saturating_add(1);
            break;
        };
        if total > rest.len() {
            // A tail cut short (a capped or interrupted body).
            out.unparsed = out.unparsed.saturating_add(1);
            break;
        }
        let frame = &rest[..total];
        rest = &rest[total..];
        if crc32(&frame[..total - 4]) != u32_at(frame, total - 4) {
            out.unparsed = out.unparsed.saturating_add(1);
            continue;
        }
        let header_bytes = &frame[PRELUDE_LEN..PRELUDE_LEN + headers_len];
        let payload = &frame[PRELUDE_LEN + headers_len..total - 4];
        match parse_headers(header_bytes) {
            Some(headers) => out.messages.push(Message { headers, payload }),
            None => out.unparsed = out.unparsed.saturating_add(1),
        }
    }
    out
}

/// The header block, or None when a header runs past the block, names an
/// unknown value type, or carries a name or string that is not UTF-8.
fn parse_headers(mut b: &[u8]) -> Option<Vec<(&str, HeaderValue<'_>)>> {
    fn take<'a>(b: &mut &'a [u8], n: usize) -> Option<&'a [u8]> {
        if b.len() < n {
            return None;
        }
        let (head, tail) = b.split_at(n);
        *b = tail;
        Some(head)
    }
    fn be<const N: usize>(b: &mut &[u8]) -> Option<[u8; N]> {
        take(b, N)?.try_into().ok()
    }
    let mut headers = Vec::new();
    while !b.is_empty() {
        let name_len = usize::from(take(&mut b, 1)?[0]);
        let name = std::str::from_utf8(take(&mut b, name_len)?).ok()?;
        let value = match take(&mut b, 1)?[0] {
            0 => HeaderValue::Bool(true),
            1 => HeaderValue::Bool(false),
            2 => HeaderValue::Byte(i8::from_be_bytes(be::<1>(&mut b)?)),
            3 => HeaderValue::Short(i16::from_be_bytes(be::<2>(&mut b)?)),
            4 => HeaderValue::Int(i32::from_be_bytes(be::<4>(&mut b)?)),
            5 => HeaderValue::Long(i64::from_be_bytes(be::<8>(&mut b)?)),
            6 => {
                let len = usize::from(u16::from_be_bytes(be::<2>(&mut b)?));
                HeaderValue::Bytes(take(&mut b, len)?)
            }
            7 => {
                let len = usize::from(u16::from_be_bytes(be::<2>(&mut b)?));
                HeaderValue::String(std::str::from_utf8(take(&mut b, len)?).ok()?)
            }
            8 => HeaderValue::Timestamp(i64::from_be_bytes(be::<8>(&mut b)?)),
            9 => HeaderValue::Uuid(be::<16>(&mut b)?),
            _ => return None,
        };
        headers.push((name, value));
    }
    Some(headers)
}

/// Test-only encoder: string headers and a payload, framed with correct CRCs.
#[cfg(test)]
pub(crate) fn encode_message(headers: &[(&str, &str)], payload: &[u8]) -> Vec<u8> {
    let mut hb = Vec::new();
    for (name, value) in headers {
        hb.push(u8::try_from(name.len()).expect("test header names are under 256 bytes"));
        hb.extend_from_slice(name.as_bytes());
        hb.push(7);
        let value_len = u16::try_from(value.len()).expect("test header values are under 64 KiB");
        hb.extend_from_slice(&value_len.to_be_bytes());
        hb.extend_from_slice(value.as_bytes());
    }
    let total = u32::try_from(FRAME_OVERHEAD + hb.len() + payload.len())
        .expect("test messages are far below 4 GiB");
    let headers_len = u32::try_from(hb.len()).expect("test header blocks are far below 4 GiB");
    let mut msg = Vec::new();
    msg.extend_from_slice(&total.to_be_bytes());
    msg.extend_from_slice(&headers_len.to_be_bytes());
    let prelude_crc = crc32(&msg);
    msg.extend_from_slice(&prelude_crc.to_be_bytes());
    msg.extend_from_slice(&hb);
    msg.extend_from_slice(payload);
    let message_crc = crc32(&msg);
    msg.extend_from_slice(&message_crc.to_be_bytes());
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(name: &str, payload: &str) -> Vec<u8> {
        encode_message(
            &[
                (":message-type", "event"),
                (":event-type", name),
                (":content-type", "application/json"),
            ],
            payload.as_bytes(),
        )
    }

    /// The published check value of CRC-32/ISO-HDLC, plus the empty input.
    #[test]
    fn crc32_matches_the_ieee_check_value() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32(b""), 0);
    }

    /// A message from the AWS event stream specification's test suite
    /// (`int32_header`): a hand-written encoder could agree with this decoder
    /// on a wrong layout, an outside vector cannot.
    #[test]
    fn decodes_the_specification_vector() {
        let bytes: [u8; 45] = [
            0x00, 0x00, 0x00, 0x2d, 0x00, 0x00, 0x00, 0x10, 0x41, 0xc4, 0x24, 0xb8, 0x0a, 0x65,
            0x76, 0x65, 0x6e, 0x74, 0x2d, 0x74, 0x79, 0x70, 0x65, 0x04, 0x00, 0x00, 0xa0, 0x0c,
            0x7b, 0x27, 0x66, 0x6f, 0x6f, 0x27, 0x3a, 0x27, 0x62, 0x61, 0x72, 0x27, 0x7d, 0x36,
            0xf4, 0x80, 0xa0,
        ];
        let d = decode(&bytes);
        assert_eq!(d.unparsed, 0);
        assert_eq!(d.messages.len(), 1);
        assert_eq!(
            d.messages[0].headers,
            vec![("event-type", HeaderValue::Int(40972))]
        );
        assert_eq!(d.messages[0].payload, b"{'foo':'bar'}");
        assert!(starts_with_message(&bytes));
    }

    #[test]
    fn decodes_consecutive_messages_and_their_string_headers() {
        let mut body = event("messageStart", r#"{"role":"assistant"}"#);
        body.extend(event("messageStop", r#"{"stopReason":"end_turn"}"#));
        let d = decode(&body);
        assert_eq!(d.unparsed, 0);
        assert_eq!(d.messages.len(), 2);
        assert_eq!(
            d.messages[0].header_str(":event-type"),
            Some("messageStart")
        );
        assert_eq!(d.messages[1].header_str(":message-type"), Some("event"));
        assert_eq!(d.messages[1].payload, br#"{"stopReason":"end_turn"}"#);
        assert_eq!(d.messages[1].header_str(":missing"), None);
    }

    /// A body cut inside the last message keeps every whole message before
    /// it and counts the cut one.
    #[test]
    fn a_truncated_tail_is_counted_once() {
        let first = event("messageStart", r#"{"role":"assistant"}"#);
        let second = event("contentBlockDelta", r#"{"delta":{"text":"hi"}}"#);
        for cut in [1, 5, PRELUDE_LEN, second.len() - 1] {
            let mut body = first.clone();
            body.extend_from_slice(&second[..cut]);
            let d = decode(&body);
            assert_eq!(d.messages.len(), 1, "cut at {cut}");
            assert_eq!(d.unparsed, 1, "cut at {cut}");
        }
    }

    /// A corrupted payload fails the message CRC: that message is skipped
    /// and counted, and the next one (whose boundary the prelude CRC vouched
    /// for) still decodes.
    #[test]
    fn a_bad_message_crc_skips_that_message_only() {
        let mut bad = event("contentBlockDelta", r#"{"delta":{"text":"hi"}}"#);
        let at = bad.len() - 6;
        bad[at] ^= 0x01;
        let mut body = bad;
        body.extend(event("messageStop", r#"{"stopReason":"end_turn"}"#));
        let d = decode(&body);
        assert_eq!(d.unparsed, 1);
        assert_eq!(d.messages.len(), 1);
        assert_eq!(d.messages[0].header_str(":event-type"), Some("messageStop"));
    }

    /// A corrupted prelude loses the framing: the decode stops there and
    /// counts the remainder once.
    #[test]
    fn a_bad_prelude_crc_stops_the_decode() {
        let mut body = event("messageStart", r#"{"role":"assistant"}"#);
        let mut bad = event("messageStop", r#"{"stopReason":"end_turn"}"#);
        bad[2] ^= 0x01;
        body.extend(bad);
        body.extend(event("metadata", r#"{"usage":{}}"#));
        let d = decode(&body);
        assert_eq!(d.messages.len(), 1);
        assert_eq!(d.unparsed, 1);
    }

    #[test]
    fn non_stream_bodies_are_not_sniffed_as_a_stream() {
        for body in [
            &b"{\"output\":{\"message\":{}}}"[..],
            b"data: {\"a\":1}\n\n",
            b"",
            b"\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00",
        ] {
            assert!(!starts_with_message(body), "{body:?}");
        }
    }

    #[test]
    fn every_header_type_decodes() {
        let mut hb = Vec::new();
        let mut put = |name: &str, ty: u8, value: &[u8]| {
            hb.push(u8::try_from(name.len()).expect("test header names are under 256 bytes"));
            hb.extend_from_slice(name.as_bytes());
            hb.push(ty);
            hb.extend_from_slice(value);
        };
        put("t", 0, &[]);
        put("f", 1, &[]);
        put("b", 2, &[0xff]);
        put("s", 3, &(-2i16).to_be_bytes());
        put("i", 4, &7i32.to_be_bytes());
        put("l", 5, &9i64.to_be_bytes());
        put("y", 6, &[0, 2, 0xAB, 0xCD]);
        put("z", 7, &[0, 1, b'x']);
        put("ts", 8, &1i64.to_be_bytes());
        put("u", 9, &[3u8; 16]);
        let headers = parse_headers(&hb).expect("well formed");
        let values: Vec<HeaderValue<'_>> = headers.into_iter().map(|(_, v)| v).collect();
        assert_eq!(
            values,
            vec![
                HeaderValue::Bool(true),
                HeaderValue::Bool(false),
                HeaderValue::Byte(-1),
                HeaderValue::Short(-2),
                HeaderValue::Int(7),
                HeaderValue::Long(9),
                HeaderValue::Bytes(&[0xAB, 0xCD]),
                HeaderValue::String("x"),
                HeaderValue::Timestamp(1),
                HeaderValue::Uuid([3u8; 16]),
            ]
        );
        // An unknown value type, or a header running past the block, is a
        // malformed block, not a guess.
        assert_eq!(parse_headers(&[1, b'a', 10]), None);
        assert_eq!(parse_headers(&[1, b'a', 7, 0, 5, b'x']), None);
    }
}
