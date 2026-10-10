//! One SSE event's `data`, read as JSON, even when it holds a number outside
//! `f64`'s range.
//!
//! `serde_json` refuses a literal such as `1e400` ("number out of range"),
//! and every reassembler used to skip the event that carried one: its text,
//! its finish reason, its error, all for one number nothing reads (a token's
//! `logprob`, a stray field). Python's `json`, which the provider SDKs read
//! the same stream with, takes the literal as an infinity, and a
//! `serde_json::Value` cannot hold one. So each such literal is read as the
//! JSON string of its own spelling (`-1e400` becomes `"-1e400"`): the
//! provider's exact characters rather than a number nobody sent (saturating to
//! `f64::MAX` would be one), and a truthy value wherever truthiness is the
//! test, so `"error": 1e400` still declares a failure.
//!
//! The strict parse runs first and the rewrite only on an event it refused,
//! so an ordinary event costs nothing more. An event that is still not JSON
//! (cut short, or not JSON at all) is the caller's to count.

use serde_json::Value;

/// `data` as JSON, or None when it is not JSON even with its out-of-range
/// numbers read as strings.
pub(super) fn parse_event(data: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str(data) {
        return Some(v);
    }
    serde_json::from_str(&quote_out_of_range_numbers(data)?).ok()
}

/// The OpenAI stream's end marker. It is not JSON and carries nothing, so an
/// event holding only it was never content a reassembler skipped.
pub(super) fn is_done_marker(data: &str) -> bool {
    data.trim() == "[DONE]"
}

/// `text` with every JSON number literal outside `f64`'s range wrapped in
/// quotes, or None when it holds none (there is nothing to retry). A literal
/// inside a string is left alone, as is anything that is not a well-formed
/// JSON number: the retry must not accept text the strict parse rightly
/// refused for another reason.
fn quote_out_of_range_numbers(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let mut out = String::new();
    let mut copied = 0; // bytes of `text` already in `out`
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'"' => i = string_end(bytes, i),
            b'-' | b'0'..=b'9' => {
                let start = i;
                while i < bytes.len()
                    && matches!(bytes[i], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
                {
                    i += 1;
                }
                // ASCII bounds on both sides, so slicing cannot split a char.
                let literal = &text[start..i];
                if is_json_number(literal.as_bytes())
                    && literal.parse::<f64>().is_ok_and(f64::is_infinite)
                {
                    out.push_str(&text[copied..start]);
                    out.push('"');
                    out.push_str(literal);
                    out.push('"');
                    copied = i;
                }
            }
            _ => i += 1,
        }
    }
    if copied == 0 {
        return None;
    }
    out.push_str(&text[copied..]);
    Some(out)
}

/// The index just past the string that opens at `open` (or the end of the
/// text, when it never closes).
fn string_end(bytes: &[u8], open: usize) -> usize {
    let mut i = open + 1;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return i + 1,
            _ => i += 1,
        }
    }
    bytes.len()
}

/// RFC 8259 `number`: `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`.
fn is_json_number(t: &[u8]) -> bool {
    let digits = |mut i: usize| {
        let start = i;
        while t.get(i).is_some_and(u8::is_ascii_digit) {
            i += 1;
        }
        (i > start).then_some(i)
    };
    let mut i = usize::from(t.first() == Some(&b'-'));
    i = match t.get(i) {
        Some(b'0') => i + 1,
        Some(b'1'..=b'9') => match digits(i) {
            Some(end) => end,
            None => return false,
        },
        _ => return false,
    };
    if t.get(i) == Some(&b'.') {
        match digits(i + 1) {
            Some(end) => i = end,
            None => return false,
        }
    }
    if matches!(t.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(t.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        match digits(i) {
            Some(end) => i = end,
            None => return false,
        }
    }
    i == t.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The number that broke the event is kept as its own spelling, and
    /// everything around it is read as before.
    #[test]
    fn an_out_of_range_number_is_read_as_its_own_spelling() {
        for (data, expected) in [
            (
                r#"{"logprob":-1e400,"content":" world"}"#,
                json!({"logprob": "-1e400", "content": " world"}),
            ),
            (r#"{"error":1e400}"#, json!({"error": "1e400"})),
            (r#"[1E+309, 2.5e999, 0]"#, json!(["1E+309", "2.5e999", 0])),
            (
                r#"{"a":{"b":[-1.0e400]}}"#,
                json!({"a": {"b": ["-1.0e400"]}}),
            ),
        ] {
            assert_eq!(parse_event(data), Some(expected), "{data}");
        }
        // A 400-digit integer overflows `f64` just the same.
        let big = format!("{{\"n\":{}}}", "9".repeat(400));
        let v = parse_event(&big).expect("read");
        assert_eq!(v["n"], Value::from("9".repeat(400)));
    }

    /// Only out-of-range literals change: an in-range number beside one keeps
    /// its type, and an underflow (`1e-400`) is the zero the strict parser
    /// already reads, not a string.
    #[test]
    fn numbers_in_range_are_left_as_numbers() {
        let v = parse_event(r#"{"big":1e400,"small":-1.5,"tiny":1e-400,"int":7}"#).unwrap();
        assert_eq!(v["big"], "1e400");
        assert_eq!(v["small"], -1.5);
        assert_eq!(v["tiny"], 0.0);
        assert_eq!(v["int"], 7);
        assert_eq!(
            parse_event(r#"{"tiny":1e-400}"#),
            Some(json!({"tiny": 0.0}))
        );
    }

    /// A literal inside a string is text, escaped quotes included, and is
    /// never rewritten.
    #[test]
    fn a_number_inside_a_string_is_left_alone() {
        let v = parse_event(r#"{"t":"x \" 1e400 \\","n":1e400}"#).unwrap();
        assert_eq!(v["t"], "x \" 1e400 \\");
        assert_eq!(v["n"], "1e400");
        assert_eq!(quote_out_of_range_numbers(r#"{"t":"1e400"}"#), None);
    }

    /// The retry accepts nothing the strict parser refused for any other
    /// reason: not JSON, cut short, or a malformed number that merely
    /// overflows.
    #[test]
    fn what_is_not_json_stays_unread() {
        for data in [
            "[DONE]",
            "",
            "not json",
            r#"{"id":"c1","choices":[{"delta":{"content":" wor"#,
            r#"{"n":1e400,"cut":"#,
            r#"{"n":01e400}"#,
            r#"{"n":1.e400}"#,
            r#"{"n":--1e400}"#,
            r#"{"n":1e400 1}"#,
        ] {
            assert_eq!(parse_event(data), None, "{data}");
        }
    }

    #[test]
    fn json_number_grammar() {
        for ok in [
            "0", "-0", "1", "-12", "1.5", "1e5", "1E+5", "1e-5", "-0.0e0", "10",
        ] {
            assert!(is_json_number(ok.as_bytes()), "{ok}");
        }
        for bad in [
            "", "-", "01", "1.", ".5", "1e", "1e+", "+1", "1-", "1.2.3", "--1",
        ] {
            assert!(!is_json_number(bad.as_bytes()), "{bad}");
        }
    }

    #[test]
    fn the_done_marker_is_recognized_with_its_whitespace() {
        assert!(is_done_marker("[DONE]"));
        assert!(is_done_marker(" [DONE]\r"));
        assert!(!is_done_marker("[DONE] x"));
        assert!(!is_done_marker("{}"));
    }
}

/// The reassemblers and the stream-shape dispatch, end to end through
/// `parse_llm`: an event holding an out-of-range number keeps everything it
/// carried, and an event that is still not JSON is counted, never silent.
#[cfg(test)]
mod reassembled {
    use super::super::{parse_llm, LlmSemantics};
    use wardex_limits::Limits;

    fn parse(host: &str, path: &str, req: &str, sse: &str) -> LlmSemantics {
        parse_llm(
            host,
            path,
            req.as_bytes(),
            sse.as_bytes(),
            Limits::default(),
        )
        .expect("an LLM endpoint always parses")
    }

    fn texts(s: &LlmSemantics) -> Vec<String> {
        let msgs: serde_json::Value =
            serde_json::from_str(s.output_messages.as_deref().unwrap_or("[]")).unwrap();
        msgs.as_array()
            .into_iter()
            .flatten()
            .flat_map(|m| m["parts"].as_array().cloned().unwrap_or_default())
            .filter(|p| p["type"] == "text")
            .filter_map(|p| p["content"].as_str().map(str::to_string))
            .collect()
    }

    const CHAT_REQ: &str = r#"{"model":"gpt-4o","stream":true,"messages":[]}"#;
    const C1: &str =
        r#"{"id":"c1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":"Hello"}}]}"#;
    const C3: &str =
        r#"{"id":"c1","model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#;

    fn chat(middle: &str) -> LlmSemantics {
        parse(
            "api.openai.com",
            "/v1/chat/completions",
            CHAT_REQ,
            &format!("data: {C1}\n\ndata: {middle}\n\ndata: {C3}\n\ndata: [DONE]\n\n"),
        )
    }

    #[test]
    fn a_chat_chunk_with_an_out_of_range_logprob_keeps_its_text() {
        let s = chat(
            r#"{"id":"c1","model":"gpt-4o","choices":[{"index":0,"delta":{"content":" world"},"logprobs":{"content":[{"token":" world","logprob":-1e400}]}}]}"#,
        );
        assert_eq!(texts(&s), ["Hello world"]);
        assert_eq!(s.finish_reasons, Some(vec!["stop".to_string()]));
        assert_eq!(s.stream_events_unparsed, 0);
    }

    /// The error chunk itself holding the number: the failure is still
    /// declared, classified by its code — or, when the number IS the error,
    /// declared and unclassified, as Python's truthy infinity is.
    #[test]
    fn a_chat_error_chunk_with_an_out_of_range_number_still_declares_the_failure() {
        for (error, expected) in [
            (
                r#"{"code":"server_error","retry_after":1e400}"#,
                "server_error",
            ),
            ("1e400", ""),
        ] {
            let s = parse(
                "api.openai.com",
                "/v1/chat/completions",
                CHAT_REQ,
                &format!("data: {C1}\n\ndata: {{\"id\":\"c1\",\"error\":{error}}}\n\n"),
            );
            assert_eq!(s.error_type.as_deref(), Some(expected), "{error}");
            assert_eq!(s.finish_reasons, Some(vec!["error".to_string()]), "{error}");
            assert_eq!(s.stream_terminated, Some(true), "{error}");
        }
    }

    #[test]
    fn an_anthropic_delta_with_an_out_of_range_number_keeps_its_text() {
        let s = parse(
            "api.anthropic.com",
            "/v1/messages",
            r#"{"model":"claude-sonnet-4-6","max_tokens":64,"stream":true,"messages":[]}"#,
            "event: message_start\n\
             data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-6\",\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n\
             event: content_block_delta\n\
             data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello\"}}\n\n\
             event: content_block_delta\n\
             data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"},\"x\":1e400}\n\n\
             event: message_delta\n\
             data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n\
             event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        );
        assert_eq!(texts(&s), ["Hello world"]);
        assert_eq!(s.finish_reasons, Some(vec!["stop".to_string()]));
        assert_eq!(s.stream_events_unparsed, 0);
    }

    /// The Responses terminal snapshot is the whole body; one number in it
    /// used to cost the whole body. It is kept, and the number reads as the
    /// spelling the provider sent.
    #[test]
    fn a_responses_snapshot_with_an_out_of_range_number_is_the_body() {
        let s = parse(
            "api.openai.com",
            "/v1/responses",
            r#"{"model":"gpt-4.1","stream":true,"input":"hi"}"#,
            "event: response.completed\n\
             data: {\"type\":\"response.completed\",\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\
             \"status\":\"completed\",\"model\":\"gpt-4.1\",\"output\":[{\"id\":\"msg_1\",\"type\":\"message\",\
             \"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"Hello world\",\
             \"annotations\":[],\"logprobs\":[{\"token\":\"x\",\"logprob\":-1e400}]}]}],\
             \"usage\":{\"input_tokens\":2,\"output_tokens\":2}}}\n\n",
        );
        assert_eq!(texts(&s), ["Hello world"]);
        assert_eq!(s.finish_reasons, Some(vec!["stop".to_string()]));
        assert_eq!(s.usage.output_tokens(), Some(2));
        let body: serde_json::Value =
            serde_json::from_slice(s.decoded_response.as_deref().unwrap()).unwrap();
        assert_eq!(
            body["output"][0]["content"][0]["logprobs"][0]["logprob"],
            "-1e400"
        );
        assert_eq!(s.stream_events_unparsed, 0);
    }

    /// A gateway path that names no endpoint: the stream's shape picks the
    /// reassembler, and a chunk holding the number still shows that shape.
    #[test]
    fn the_stream_shape_is_read_through_an_out_of_range_number() {
        let s = parse(
            "gateway.example.test",
            "/proxy/stream",
            CHAT_REQ,
            "data: {\"id\":\"c1\",\"model\":\"gpt-4o\",\"choices\":[{\"index\":0,\
             \"delta\":{\"content\":\"Hi\"},\"finish_reason\":\"stop\",\"logprobs\":{\"content\":\
             [{\"token\":\"Hi\",\"logprob\":-1e400}]}}]}\n\n",
        );
        assert_eq!(s.api_type, Some("chat_completions"));
        assert_eq!(texts(&s), ["Hi"]);
    }

    /// What is still not JSON is counted, per stream, by every reassembler;
    /// `[DONE]` never is (it ends the stream and carries nothing), and a body
    /// that is not a stream counts nothing.
    #[test]
    fn an_event_still_not_json_is_counted_and_done_is_not() {
        let cut = r#"{"id":"c1","choices":[{"delta":{"content":" wor"#;
        assert_eq!(chat(cut).stream_events_unparsed, 1);
        assert_eq!(texts(&chat(cut)), ["Hello"]);
        assert_eq!(chat(C1).stream_events_unparsed, 0);

        let anthropic = |tail: &str| {
            parse(
                "api.anthropic.com",
                "/v1/messages",
                r#"{"model":"claude-sonnet-4-6","max_tokens":64,"messages":[]}"#,
                &format!(
                    "event: message_start\ndata: {{\"type\":\"message_start\",\"message\":\
                     {{\"id\":\"msg_1\",\"model\":\"claude-sonnet-4-6\"}}}}\n\n{tail}\
                     event: message_stop\ndata: {{\"type\":\"message_stop\"}}\n\n"
                ),
            )
            .stream_events_unparsed
        };
        assert_eq!(anthropic(""), 0);
        assert_eq!(anthropic("data: [DONE]\n\n"), 0);
        assert_eq!(
            anthropic("event: content_block_delta\ndata: {\"type\":\n\n"),
            1
        );
        assert_eq!(anthropic("data: not json\n\ndata: {\n\n"), 2);

        let responses = |tail: &str| {
            parse(
                "api.openai.com",
                "/v1/responses",
                r#"{"model":"gpt-4.1","stream":true,"input":"hi"}"#,
                &format!(
                    "event: response.created\ndata: {{\"type\":\"response.created\",\"response\":\
                     {{\"id\":\"resp_1\",\"status\":\"in_progress\",\"output\":[]}}}}\n\n{tail}"
                ),
            )
            .stream_events_unparsed
        };
        assert_eq!(responses(""), 0);
        assert_eq!(responses("data: [DONE]\n\n"), 0);
        assert_eq!(
            responses("event: response.output_text.delta\ndata: {\"delta\":\"x\n\n"),
            1
        );

        let body = parse(
            "api.openai.com",
            "/v1/chat/completions",
            CHAT_REQ,
            r#"{"id":"c1","model":"gpt-4o","choices":[{"message":{"content":"x"},"finish_reason":"stop"}]}"#,
        );
        assert_eq!(body.stream_events_unparsed, 0);
    }
}
