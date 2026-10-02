//! Server-Sent Events (text/event-stream) framing parser. Provider-agnostic.
//! WHATWG SSE line convention: blank line = event boundary, starts with ':' = comment (ignored),
//! "field: value" (one space after the colon is stripped), data spans multiple lines joined by '\n'.

/// A single SSE event.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
    pub id: Option<String>,
}

/// Parses a completed SSE body into a list of events. Events with empty data are not emitted.
pub fn parse(body: &[u8]) -> Vec<SseEvent> {
    let text = match std::str::from_utf8(body) {
        Ok(t) => t,
        Err(_) => return Vec::new(),
    };
    let mut events = Vec::new();
    let mut cur = SseEvent::default();
    let mut data_lines: Vec<String> = Vec::new();
    for raw_line in text.split('\n') {
        let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
        if line.is_empty() {
            if !data_lines.is_empty() {
                cur.data = data_lines.join("\n");
                events.push(std::mem::take(&mut cur));
            } else {
                cur = SseEvent::default();
            }
            data_lines.clear();
            continue;
        }
        if line.starts_with(':') {
            continue; // comment
        }
        let (field, value) = match line.find(':') {
            Some(idx) => {
                let f = &line[..idx];
                let mut v = &line[idx + 1..];
                if let Some(stripped) = v.strip_prefix(' ') {
                    v = stripped;
                }
                (f, v)
            }
            None => (line, ""),
        };
        match field {
            "data" => data_lines.push(value.to_string()),
            "event" => cur.event = Some(value.to_string()),
            "id" => cur.id = Some(value.to_string()),
            _ => {}
        }
    }
    // handle the trailing event when there's no blank line at the end
    if !data_lines.is_empty() {
        cur.data = data_lines.join("\n");
        events.push(cur);
    }
    events
}

/// Sniffs whether the body looks like SSE. If the first non-whitespace character after leading
/// whitespace is '{'/'[', it's JSON (false); otherwise true if it contains a "data:"/"event:" line.
pub fn looks_like_sse(body: &[u8]) -> bool {
    sniff(body) == Some(true)
}

/// The same sniff, saying when it could not read the body at all. `Some(true)`: an event stream.
/// `Some(false)`: something else was read — JSON (first non-blank byte `{`/`[`), an empty body, or
/// text with no "data:"/"event:" line. `None`: the body is not UTF-8 text (binary, compressed in a
/// coding nothing inflated, or cut inside a character), so "not SSE" was never observed.
///
/// The whole body must be text before any answer, the JSON one included: compressed bytes can
/// begin with `{` or `[` (a brotli stream of 64 KiB to 1 MiB at the default window starts with
/// `[`, and a raw deflate block can start with either), and one byte is not a body read as JSON.
pub fn sniff(body: &[u8]) -> Option<bool> {
    let text = std::str::from_utf8(body).ok()?;
    let first = text
        .bytes()
        .find(|&b| b != b' ' && b != b'\t' && b != b'\r' && b != b'\n');
    match first {
        None => Some(false),
        Some(b'{' | b'[') => Some(false),
        Some(_) => Some(
            text.lines()
                .any(|l| l.starts_with("data:") || l.starts_with("event:")),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_two_events_with_comment() {
        let body = b": ping\n\ndata: {\"a\":1}\n\ndata: {\"b\":2}\n\n";
        let evs = parse(body);
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].data, "{\"a\":1}");
        assert_eq!(evs[1].data, "{\"b\":2}");
    }

    #[test]
    fn joins_multiline_data_with_newline() {
        let body = b"data: line1\ndata: line2\n\n";
        let evs = parse(body);
        assert_eq!(evs.len(), 1);
        assert_eq!(evs[0].data, "line1\nline2");
    }

    #[test]
    fn captures_event_and_id_fields() {
        let body = b"event: message_start\nid: 42\ndata: {}\n\n";
        let evs = parse(body);
        assert_eq!(evs[0].event.as_deref(), Some("message_start"));
        assert_eq!(evs[0].id.as_deref(), Some("42"));
        assert_eq!(evs[0].data, "{}");
    }

    #[test]
    fn handles_crlf_and_trailing_event_without_blank_line() {
        let body = b"data: x\r\n\r\ndata: y\r\n";
        let evs = parse(body);
        assert_eq!(evs.len(), 2);
        assert_eq!(evs[0].data, "x");
        assert_eq!(evs[1].data, "y");
    }

    #[test]
    fn looks_like_sse_true_for_event_stream() {
        assert!(looks_like_sse(b"data: {\"a\":1}\n\n"));
        assert!(looks_like_sse(b"event: message_start\ndata: {}\n\n"));
    }

    #[test]
    fn looks_like_sse_false_for_json() {
        assert!(!looks_like_sse(b"{\"choices\":[]}"));
        assert!(!looks_like_sse(b"  [1,2,3]"));
        assert!(!looks_like_sse(b""));
    }

    #[test]
    fn sniff_reads_text_and_json_and_says_so() {
        assert_eq!(sniff("data: {\"t\":\"안녕\"}\n\n".as_bytes()), Some(true));
        assert_eq!(sniff(b"{\"choices\":[]}"), Some(false));
        assert_eq!(sniff(b"plain text, no event lines"), Some(false));
        assert_eq!(sniff(b""), Some(false));
    }

    #[test]
    fn sniff_cannot_read_a_body_that_is_not_text() {
        // An SSE body cut inside a multi-byte character: no longer UTF-8.
        let whole = "data: {\"t\":\"안녕\"}\n\n".as_bytes();
        let cut = &whole[..whole.iter().position(|&b| b >= 0x80).unwrap() + 1];
        assert_eq!(sniff(cut), None);
        assert!(!looks_like_sse(cut));
        // Compressed bytes nothing inflated.
        assert_eq!(sniff(&[0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x07, 0x00]), None);
    }

    #[test]
    fn sniff_does_not_call_compressed_bytes_json_by_their_first_byte() {
        // The head of a real brotli stream (`brotli -w 22`, a 109 KB SSE body): its first byte is
        // `[` because of the window and length bits, not because the body is a JSON array.
        let brotli = [
            0x5b, 0x35, 0xab, 0x01, 0xc4, 0xaa, 0xc0, 0x6e, 0x33, 0xfd, 0xa4, 0x82,
        ];
        assert_eq!(sniff(&brotli), None);
        assert!(!looks_like_sse(&brotli));
        assert_eq!(sniff(&[b' ', b'{', 0xff, 0x00, 0x9c]), None);
        // Text that is JSON still reads as JSON.
        assert_eq!(sniff(b" [1, 2, \"\xec\x95\x88\"]"), Some(false));
    }
}
