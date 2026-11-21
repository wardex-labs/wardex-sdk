//! Server-Sent Events (text/event-stream) framing parser. Provider-agnostic.
//! WHATWG SSE line convention: blank line = event boundary, starts with ':' = comment (ignored),
//! "field: value" (one space after the colon is stripped), data spans multiple lines joined by '\n'.

/// A single SSE event.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct SseEvent {
    /// The last `event:` line's value. `None` when the event had no such
    /// line, which the WHATWG convention reads as type `message`.
    pub event: Option<String>,
    /// Every `data:` line's value, joined by `\n`. Empty only when every
    /// `data:` line of the event was empty; an event with no `data:` line at
    /// all is not emitted.
    pub data: String,
    /// The last `id:` line's value. `None` when the event had no such line.
    pub id: Option<String>,
}

/// Parses a completed SSE body into a list of events. An event with no
/// `data:` line is not emitted; a body that is not UTF-8 yields no events.
pub fn parse(body: &[u8]) -> Vec<SseEvent> {
    let Ok(text) = std::str::from_utf8(body) else {
        return Vec::new();
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
        let (field, value) = match line.split_once(':') {
            Some((f, v)) => (f, v.strip_prefix(' ').unwrap_or(v)),
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

/// How many leading bytes of a body `opens_event_stream` reads.
pub const OPENING_BYTES: usize = 1024;

/// Do a body's first bytes open an event stream? `Some(true)` once a line within the first
/// `OPENING_BYTES` begins with `data:` or `event:`; `Some(false)` once they cannot (the first
/// non-blank byte opens JSON, a byte that is not UTF-8, or all `OPENING_BYTES` read without such a
/// line); `None` while the bytes so far decide nothing.
///
/// For a body still arriving, where `sniff` needs the whole body. A longer prefix of the same body
/// never changes an answer, so a stream parser deciding on the first bytes and a caller asking
/// again over the stored body agree. A character the end of `body` cuts is not a byte that is not
/// UTF-8: the rest of it may still arrive.
pub fn opens_event_stream(body: &[u8]) -> Option<bool> {
    let window = body.get(..OPENING_BYTES).unwrap_or(body);
    let (text, not_text) = match std::str::from_utf8(window) {
        Ok(text) => (text, false),
        Err(err) => (
            window
                .get(..err.valid_up_to())
                .and_then(|valid| std::str::from_utf8(valid).ok())
                .unwrap_or(""),
            err.error_len().is_some(),
        ),
    };
    let first = text
        .bytes()
        .find(|&b| b != b' ' && b != b'\t' && b != b'\r' && b != b'\n');
    if matches!(first, Some(b'{' | b'[')) {
        return Some(false);
    }
    if text
        .lines()
        .any(|l| l.starts_with("data:") || l.starts_with("event:"))
    {
        return Some(true);
    }
    (not_text || window.len() == OPENING_BYTES).then_some(false)
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

    #[test]
    fn the_opening_bytes_decide_once_and_a_longer_prefix_never_changes_it() {
        let stream = b": keep-alive\n\nevent: delta\ndata: {\"t\":\"\xec\x95\x88\"}\n\n".as_slice();
        let answers: Vec<_> = (0..=stream.len())
            .map(|n| opens_event_stream(&stream[..n]))
            .collect();
        let first = answers.iter().position(Option::is_some).unwrap();
        assert!(answers[..first].iter().all(Option::is_none));
        assert!(answers[first..].iter().all(|a| *a == Some(true)));
        assert_eq!(opens_event_stream(b"da"), None, "a line still arriving");
        assert_eq!(opens_event_stream(b"  {\"model\""), Some(false));
        assert_eq!(opens_event_stream(&[0x1f, 0x8b, 0x08]), Some(false));
        assert_eq!(
            opens_event_stream("data: \u{c548}".as_bytes().split_last().unwrap().1),
            Some(true),
            "a character cut at the end is still arriving"
        );
        let prose = vec![b'x'; OPENING_BYTES + 10];
        assert_eq!(opens_event_stream(&prose[..OPENING_BYTES - 1]), None);
        assert_eq!(opens_event_stream(&prose), Some(false));
    }
}
