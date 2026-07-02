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
    let first = body
        .iter()
        .find(|&&b| b != b' ' && b != b'\t' && b != b'\r' && b != b'\n');
    match first {
        None => return false,
        Some(&b) if b == b'{' || b == b'[' => return false,
        _ => {}
    }
    let text = match std::str::from_utf8(body) {
        Ok(t) => t,
        Err(_) => return false,
    };
    text.lines()
        .any(|l| l.starts_with("data:") || l.starts_with("event:"))
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
}
