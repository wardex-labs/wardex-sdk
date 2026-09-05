//! The original `semantic.rs` test suite, moved whole in the module split
//! (names and count unchanged). New modules test beside their own code.

use super::parts::{normalize_finish_reason, FINISH_REASONS};
use super::*;

const OPENAI_CHAT: &[u8] = br#"{
        "id":"chatcmpl-abc","model":"gpt-4o-mini-2024-07-18",
        "choices":[{"finish_reason":"stop"}],
        "usage":{"prompt_tokens":12,"completion_tokens":3,
                 "completion_tokens_details":{"reasoning_tokens":1}}
    }"#;
const OPENAI_REQ: &[u8] = br#"{"model":"gpt-4o-mini","temperature":0.5,"max_tokens":64,"top_p":1.0,"n":2,"stream":false,"stop":"END"}"#;

#[test]
fn openai_chat_response_extracts_tokens_and_model() {
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        OPENAI_REQ,
        OPENAI_CHAT,
        Limits::default(),
    )
    .expect("supported host");
    assert_eq!(s.provider, "openai");
    assert_eq!(s.operation, "chat");
    assert_eq!(s.response_model.as_deref(), Some("gpt-4o-mini-2024-07-18"));
    assert_eq!(s.response_id.as_deref(), Some("chatcmpl-abc"));
    assert_eq!(s.usage.input_tokens(), Some(12));
    assert_eq!(s.usage.output_tokens(), Some(3));
    assert_eq!(s.usage.reasoning_output_tokens(), Some(1));
    assert_eq!(s.finish_reasons.as_deref(), Some(&["stop".to_string()][..]));
    // request params
    assert_eq!(s.request_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(s.temperature, Some(0.5));
    assert_eq!(s.max_tokens, Some(64));
    assert_eq!(s.choice_count, Some(2));
    assert_eq!(s.stop_sequences.as_deref(), Some(&["END".to_string()][..]));
    // this body is not compressed, so `decode_body` passes the original through
    assert_eq!(s.decoded_response.as_deref(), Some(OPENAI_CHAT));
}

#[test]
fn decoded_cap_comes_from_limits() {
    // The cap only bites on the decompression path (plain bodies pass through uncapped),
    // so gzip-compress the body to exercise it — mirrors `gzip_response_is_decompressed_and_parsed`.
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let limits = Limits {
        max_decoded_bytes: 1,
        ..Default::default()
    };
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(OPENAI_CHAT).unwrap();
    let gz = enc.finish().unwrap();

    let sem = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        OPENAI_REQ,
        &gz,
        limits,
    );
    // The decompressed response exceeds the tiny decode cap, so decompression is
    // rejected and no semantics can be extracted from the (still-compressed) bytes.
    assert!(sem.is_none() || sem.unwrap().response_model.is_none());
}

#[test]
fn unsupported_host_returns_none() {
    assert!(parse_llm(
        "example.com",
        "/v1/foo",
        b"{}",
        b"{\"ok\":true}",
        Limits::default()
    )
    .is_none());
}

const ANTHROPIC_MSG: &[u8] = br#"{
        "id":"msg_1","model":"claude-opus-4-8","stop_reason":"end_turn",
        "usage":{"input_tokens":12,"output_tokens":3,"cache_read_input_tokens":4}
    }"#;
const OPENAI_EMB: &[u8] =
    br#"{"model":"text-embedding-3-small","data":[{}],"usage":{"prompt_tokens":8}}"#;

#[test]
fn anthropic_messages_extracts_tokens() {
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        b"{}",
        ANTHROPIC_MSG,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.provider, "anthropic");
    assert_eq!(s.operation, "chat");
    assert_eq!(s.response_model.as_deref(), Some("claude-opus-4-8"));
    // Inclusive total: raw 12 + cache_read 4. That this asserted 12 —
    // and passed — was the defect's evidence (R2).
    assert_eq!(s.usage.input_tokens(), Some(16));
    assert_eq!(s.usage.output_tokens(), Some(3));
    assert_eq!(s.usage.cache_read_input_tokens(), Some(4));
    // Normalized spelling (`end_turn` -> `stop`): one producer for every
    // endpoint, so dashboards group one fact under one string.
    assert_eq!(s.finish_reasons.as_deref(), Some(&["stop".to_string()][..]));
}

#[test]
fn openai_embeddings_extracts_input_tokens() {
    let s = parse_llm(
        "api.openai.com",
        "/v1/embeddings",
        b"{}",
        OPENAI_EMB,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.operation, "embeddings");
    assert_eq!(s.usage.input_tokens(), Some(8));
    assert_eq!(s.usage.output_tokens(), None);
}

#[test]
fn body_shape_fallback_detects_provider_on_localhost() {
    // even when the host is unsupported (127.0.0.1), infer the provider from the response shape
    let oa = parse_llm(
        "127.0.0.1",
        "/v1/chat/completions",
        OPENAI_REQ,
        OPENAI_CHAT,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(oa.provider, "openai");
    assert_eq!(oa.usage.input_tokens(), Some(12));
    let an = parse_llm(
        "127.0.0.1",
        "/v1/messages",
        b"{}",
        ANTHROPIC_MSG,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(an.provider, "anthropic");
    // ANTHROPIC_MSG reports raw 12 + cache_read 4 — inclusive total 16.
    assert_eq!(an.usage.input_tokens(), Some(16));
}

#[test]
fn non_llm_json_unknown_host_returns_none() {
    assert!(parse_llm(
        "127.0.0.1",
        "/v1/chat/completions",
        b"{}",
        br#"{"ok":true}"#,
        Limits::default(),
    )
    .is_none());
}

#[test]
fn gzip_response_is_decompressed_and_parsed() {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(OPENAI_CHAT).unwrap();
    let gz = enc.finish().unwrap();
    assert_eq!(&gz[..2], &[0x1f, 0x8b]); // gzip magic

    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        OPENAI_REQ,
        &gz,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.usage.input_tokens(), Some(12)); // parsed after decompression
    assert_eq!(s.decoded_response.as_deref(), Some(OPENAI_CHAT)); // decoded body stored
}

#[test]
fn decode_body_caps_decompression_bomb() {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let big = vec![0u8; 9 * 1024 * 1024]; // 9 MiB > 8 MiB cap
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(&big).unwrap();
    let gz = enc.finish().unwrap();
    assert!(gz.len() < big.len()); // compressed
                                   // exceeds cap → not decompressed, original (gz) returned
    assert_eq!(decode_body(&gz, Limits::default()), gz);
}

#[test]
fn decode_body_under_cap_decompresses() {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let small = b"hello world";
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(small).unwrap();
    let gz = enc.finish().unwrap();
    assert_eq!(decode_body(&gz, Limits::default()), small);
}

#[test]
fn anthropic_cache_creation_input_tokens_extracted() {
    let resp = br#"{
            "id":"msg_2","model":"claude-opus-4-8","stop_reason":"end_turn",
            "usage":{"input_tokens":5,"output_tokens":2,"cache_creation_input_tokens":7}
        }"#;
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        b"{}",
        resp,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.usage.cache_creation_input_tokens(), Some(7));
    // Inclusive total: raw 5 + cache_creation 7 (R2).
    assert_eq!(s.usage.input_tokens(), Some(12));
    assert_eq!(s.usage.output_tokens(), Some(2));
}

#[test]
fn openai_stop_array_maps_to_string_or_vec_many() {
    // request where stop is an array (["A","B"]) → verifies the Many path into stop_sequences
    let req = br#"{"model":"gpt-4o","stop":["A","B"]}"#;
    let resp = br#"{
            "id":"chatcmpl-y","model":"gpt-4o",
            "choices":[{"finish_reason":"stop"}],
            "usage":{"prompt_tokens":1,"completion_tokens":1}
        }"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        resp,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(
        s.stop_sequences.as_deref(),
        Some(&["A".to_string(), "B".to_string()][..])
    );
}

const OPENAI_SSE: &[u8] = b"data: {\"id\":\"chatcmpl-s\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"content\":\"Hi\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"chatcmpl-s\",\"model\":\"gpt-4o-mini\",\"choices\":[{\"delta\":{\"content\":\"!\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";

#[test]
fn openai_sse_stream_reassembles_text_and_finish_no_usage() {
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        b"{}",
        OPENAI_SSE,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.provider, "openai");
    assert!(s.reassembled_from_stream);
    assert_eq!(s.response_model.as_deref(), Some("gpt-4o-mini"));
    assert_eq!(s.response_id.as_deref(), Some("chatcmpl-s"));
    assert_eq!(s.finish_reasons.as_deref(), Some(&["stop".to_string()][..]));
    assert_eq!(s.usage.output_tokens(), None); // no usage
    let body = String::from_utf8(s.decoded_response.clone().unwrap()).unwrap();
    assert!(body.contains("\"content\":\"Hi!\"")); // text reassembled into the synthetic JSON
}

#[test]
fn openai_sse_with_usage_chunk_extracts_tokens() {
    let sse = b"data: {\"id\":\"c\",\"model\":\"gpt-4o\",\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":11,\"completion_tokens\":2}}\n\ndata: [DONE]\n\n";
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        b"{}",
        sse,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.usage.input_tokens(), Some(11));
    assert_eq!(s.usage.output_tokens(), Some(2));
}

#[test]
fn anthropic_sse_stream_extracts_tokens_text_stop() {
    let sse = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"model\":\"claude-opus-4-8\",\"usage\":{\"input_tokens\":9,\"output_tokens\":1}}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":5}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        b"{}",
        sse,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.provider, "anthropic");
    assert!(s.reassembled_from_stream);
    assert_eq!(s.response_model.as_deref(), Some("claude-opus-4-8"));
    assert_eq!(s.usage.input_tokens(), Some(9));
    assert_eq!(s.usage.output_tokens(), Some(5)); // message_delta provides the final output_tokens
    assert_eq!(s.finish_reasons.as_deref(), Some(&["stop".to_string()][..]));
    let body = String::from_utf8(s.decoded_response.clone().unwrap()).unwrap();
    assert!(body.contains("\"text\":\"Hello\""));
}

/// P2 double-add guard: `reassemble_anthropic` keeps the RAW usage in its
/// synthetic JSON and `fill_anthropic` does the one and only inclusive
/// sum. If the reassembler ever pre-adds the cache tiers, this total
/// doubles and the test names the reason.
#[test]
fn test_reassembled_stream_is_not_double_added() {
    let sse = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_s\",\"model\":\"claude-sonnet-4-6\",\"usage\":{\"input_tokens\":1000,\"output_tokens\":1,\"cache_read_input_tokens\":8000,\"cache_creation_input_tokens\":2000}}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\nevent: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":500}}\n\nevent: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        b"{}",
        sse,
        Limits::default(),
    )
    .unwrap();
    assert!(s.reassembled_from_stream);
    // Exactly once: 1000 + 8000 + 2000, not 11000 + 10000.
    assert_eq!(s.usage.input_tokens(), Some(11000));
    assert_eq!(s.usage.cache_read_input_tokens(), Some(8000));
    assert_eq!(s.usage.cache_creation_input_tokens(), Some(2000));
    assert_eq!(s.usage.output_tokens(), Some(500));
    // The synthetic body still carries Anthropic's RAW value — the wire
    // truth is preserved and the sum happens in exactly one place.
    let body = String::from_utf8(s.decoded_response.clone().unwrap()).unwrap();
    assert!(body.contains("\"input_tokens\":1000"));
}

#[test]
fn unknown_provider_sse_returns_raw_concat_no_semantics() {
    let sse = b"data: {\"foo\":1}\n\ndata: {\"bar\":2}\n\n";
    let s = parse_llm("127.0.0.1", "/v1/stream", b"{}", sse, Limits::default()).unwrap();
    assert!(s.reassembled_from_stream);
    assert_eq!(s.provider, ""); // unidentified
    assert_eq!(s.response_model, None);
    let body = String::from_utf8(s.decoded_response.clone().unwrap()).unwrap();
    assert_eq!(body, "{\"foo\":1}\n{\"bar\":2}");
}

#[test]
fn non_sse_json_still_uses_existing_path() {
    // non-SSE keeps reassembled_from_stream=false (regression guard)
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        OPENAI_REQ,
        OPENAI_CHAT,
        Limits::default(),
    )
    .unwrap();
    assert!(!s.reassembled_from_stream);
    assert_eq!(s.usage.input_tokens(), Some(12));
}

#[test]
fn gzip_sse_is_decompressed_then_reassembled() {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    let mut enc = GzEncoder::new(Vec::new(), Compression::default());
    enc.write_all(OPENAI_SSE).unwrap();
    let gz = enc.finish().unwrap();
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        b"{}",
        &gz,
        Limits::default(),
    )
    .unwrap();
    assert!(s.reassembled_from_stream);
    assert_eq!(s.response_model.as_deref(), Some("gpt-4o-mini"));
}

#[test]
fn openai_chat_extracts_tool_call() {
    let req = br#"{"model":"gpt-4o-mini"}"#;
    let resp = br#"{"id":"chatcmpl-x","model":"gpt-4o-mini","choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"add","arguments":"{\"a\":17,\"b\":25}"}}]},"finish_reason":"tool_calls"}]}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        resp,
        Limits::default(),
    )
    .expect("some");
    let om = s.output_messages.expect("output_messages");
    let v: serde_json::Value = serde_json::from_str(&om).unwrap();
    let part = &v[0]["parts"][0];
    assert_eq!(part["type"], "tool_call");
    assert_eq!(part["name"], "add");
    assert_eq!(part["id"], "call_1");
    assert_eq!(part["arguments"]["a"], 17);
    assert_eq!(part["arguments"]["b"], 25);
    assert!(!s.tool_args_unparsed);
}

#[test]
fn openai_chat_multiple_parallel_tool_calls() {
    let req = br#"{"model":"gpt-4o-mini"}"#;
    let resp = br#"{"model":"gpt-4o-mini","choices":[{"message":{"tool_calls":[{"id":"c1","function":{"name":"a","arguments":"{}"}},{"id":"c2","function":{"name":"b","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    let parts = v[0]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["name"], "a");
    assert_eq!(parts[1]["name"], "b");
}

#[test]
fn text_only_response_emits_text_part() {
    let req = br#"{"model":"gpt-4o-mini"}"#;
    let resp = br#"{"model":"gpt-4o-mini","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    assert_eq!(v[0]["parts"][0]["type"], "text");
    assert_eq!(v[0]["parts"][0]["content"], "hi");
    assert_eq!(v[0]["finish_reason"], "stop");
}

#[test]
fn openai_tool_args_unparsable_keeps_raw_and_marks() {
    let req = br#"{"model":"gpt-4o-mini"}"#;
    let resp = br#"{"model":"gpt-4o-mini","choices":[{"message":{"tool_calls":[{"id":"c1","function":{"name":"a","arguments":"{not json"}}]},"finish_reason":"tool_calls"}]}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    assert_eq!(v[0]["parts"][0]["arguments"], "{not json");
    assert!(s.tool_args_unparsed);
}

#[test]
fn anthropic_extracts_tool_use() {
    let req = br#"{"model":"claude-3"}"#;
    let resp = br#"{"id":"msg_1","model":"claude-3","stop_reason":"tool_use","content":[{"type":"text","text":"let me calculate"},{"type":"tool_use","id":"toolu_1","name":"add","input":{"a":17,"b":25}}],"usage":{"input_tokens":5,"output_tokens":2}}"#;
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        req,
        resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    let parts = v[0]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "tool_call");
    assert_eq!(parts[1]["name"], "add");
    assert_eq!(parts[1]["id"], "toolu_1");
    assert_eq!(parts[1]["arguments"]["a"], 17);
    assert_eq!(v[0]["finish_reason"], "tool_call");
}

#[test]
fn anthropic_thinking_text_tooluse_ordered() {
    let resp = r#"{"model":"c","stop_reason":"tool_use","content":[{"type":"thinking","thinking":"hmm"},{"type":"text","text":"answer"},{"type":"tool_use","id":"t1","name":"add","input":{}}]}"#.as_bytes();
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        br#"{"model":"c"}"#,
        resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    let parts = v[0]["parts"].as_array().unwrap();
    assert_eq!(parts[0]["type"], "reasoning");
    assert_eq!(parts[0]["content"], "hmm");
    assert_eq!(parts[1]["type"], "text");
    assert_eq!(parts[1]["content"], "answer");
    assert_eq!(parts[2]["type"], "tool_call");
    assert!(!s.output_messages_has_unmapped);
}

#[test]
fn anthropic_redacted_thinking_to_generic_no_flag() {
    let resp = br#"{"model":"c","stop_reason":"end_turn","content":[{"type":"redacted_thinking","data":"enc"}]}"#;
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        br#"{"model":"c"}"#,
        resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    assert_eq!(v[0]["parts"][0]["type"], "reasoning");
    assert_eq!(v[0]["parts"][0].as_object().unwrap().len(), 1); // type only
    assert!(!s.output_messages_has_unmapped);
}

#[test]
fn anthropic_server_tool_use_and_result() {
    let resp = br#"{"model":"c","stop_reason":"end_turn","content":[{"type":"server_tool_use","id":"s1","name":"web_search","input":{"query":"x"}},{"type":"web_search_tool_result","tool_use_id":"s1","content":[{"title":"r"}]}]}"#;
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        br#"{"model":"c"}"#,
        resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    let parts = v[0]["parts"].as_array().unwrap();
    assert_eq!(parts[0]["type"], "server_tool_call");
    assert_eq!(parts[0]["name"], "web_search");
    assert_eq!(parts[0]["id"], "s1");
    assert_eq!(parts[0]["server_tool_call"]["query"], "x");
    assert_eq!(parts[1]["type"], "server_tool_call_response");
    assert_eq!(parts[1]["id"], "s1");
    assert_eq!(parts[1]["server_tool_call_response"][0]["title"], "r");
}

#[test]
fn anthropic_unknown_block_to_generic_sets_flag() {
    let resp = br#"{"model":"c","stop_reason":"end_turn","content":[{"type":"some_future_block","foo":1}]}"#;
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        br#"{"model":"c"}"#,
        resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    assert_eq!(v[0]["parts"][0]["type"], "some_future_block");
    assert!(s.output_messages_has_unmapped);
}

#[test]
fn anthropic_finish_reason_normalized() {
    let resp =
        br#"{"model":"c","stop_reason":"max_tokens","content":[{"type":"text","text":"x"}]}"#;
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        br#"{"model":"c"}"#,
        resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    assert_eq!(v[0]["finish_reason"], "length");
}

#[test]
fn openai_and_anthropic_tool_call_same_shape() {
    let oa = parse_llm("api.openai.com", "/v1/chat/completions",
            br#"{"model":"m"}"#,
            br#"{"model":"m","choices":[{"message":{"tool_calls":[{"id":"x","function":{"name":"add","arguments":"{\"a\":1}"}}]}}]}"#, Limits::default()).unwrap();
    let an = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        br#"{"model":"m"}"#,
        br#"{"model":"m","content":[{"type":"tool_use","id":"x","name":"add","input":{"a":1}}]}"#,
        Limits::default(),
    )
    .unwrap();
    let ov: serde_json::Value = serde_json::from_str(&oa.output_messages.unwrap()).unwrap();
    let av: serde_json::Value = serde_json::from_str(&an.output_messages.unwrap()).unwrap();
    assert_eq!(ov[0]["parts"][0], av[0]["parts"][0]);
}

// --- SSE tool_call delta accumulation ---

fn sse_bytes(lines: &[&str]) -> Vec<u8> {
    lines
        .iter()
        .map(|l| format!("data: {}\n\n", l))
        .collect::<String>()
        .into_bytes()
}

#[test]
fn openai_sse_reassembles_tool_call_deltas() {
    let resp = sse_bytes(&[
        r#"{"id":"chatcmpl-s","model":"gpt-4o-mini","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"add","arguments":"{\"a\":1"}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"7}"}}]},"finish_reason":"tool_calls"}]}"#,
        "[DONE]",
    ]);
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        br#"{"model":"gpt-4o-mini","stream":true}"#,
        &resp,
        Limits::default(),
    )
    .unwrap();
    assert!(s.reassembled_from_stream);
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    assert_eq!(v[0]["parts"][0]["name"], "add");
    assert_eq!(v[0]["parts"][0]["arguments"]["a"], 17);
}

#[test]
fn openai_sse_parallel_tool_calls_by_index() {
    let resp = sse_bytes(&[
        r#"{"model":"m","choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","function":{"name":"a","arguments":"{}"}}]}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":1,"id":"c1","function":{"name":"b","arguments":"{}"}}]}}]}"#,
        "[DONE]",
    ]);
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        br#"{"stream":true}"#,
        &resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    let parts = v[0]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["name"], "a");
    assert_eq!(parts[1]["name"], "b");
}

#[test]
fn sse_text_plus_tool_call() {
    let resp = sse_bytes(&[
        r#"{"model":"m","choices":[{"delta":{"content":"wait"}}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c0","function":{"name":"a","arguments":"{}"}}]}}]}"#,
        "[DONE]",
    ]);
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        br#"{"stream":true}"#,
        &resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    let parts = v[0]["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 2);
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["content"], "wait");
    assert_eq!(parts[1]["name"], "a");
}

#[test]
fn finish_reason_normalization_is_total_and_closed_over_known_values() {
    // (a) Every KNOWN provider raw value maps into the closed set — the full
    // table, per provider, so a new endpoint cannot leak a provider spelling
    // for a value this function already knows.
    let known: &[(&str, &str, &str)] = &[
        ("openai", "stop", "stop"),
        ("openai", "length", "length"),
        ("openai", "tool_calls", "tool_call"),
        ("openai", "function_call", "tool_call"),
        ("openai", "content_filter", "content_filter"),
        // Responses: `incomplete_details.reason` and terminal statuses.
        ("openai", "max_output_tokens", "length"),
        ("openai", "failed", "error"),
        ("openai", "cancelled", "error"),
        ("anthropic", "end_turn", "stop"),
        ("anthropic", "stop_sequence", "stop"),
        ("anthropic", "max_tokens", "length"),
        ("anthropic", "tool_use", "tool_call"),
        ("anthropic", "refusal", "content_filter"),
    ];
    for (provider, raw, want) in known {
        let got = normalize_finish_reason(provider, raw);
        assert_eq!(&got, want, "({provider}, {raw})");
        assert!(FINISH_REASONS.contains(&got.as_str()));
    }
    // (b) Total: an UNKNOWN raw value passes through in the provider's own
    // spelling — a new finish reason surfaces under its own name instead of
    // silently vanishing (the old mapper returned None and the field was
    // omitted).
    assert_eq!(normalize_finish_reason("openai", "banana"), "banana");
    assert_eq!(
        normalize_finish_reason("anthropic", "pause_turn"),
        "pause_turn"
    );
    // (c) Members of the closed set are fixed points.
    for member in FINISH_REASONS {
        assert_eq!(&normalize_finish_reason("openai", member), member);
    }
}

#[test]
fn openai_text_then_tool_call_order() {
    let req = br#"{"model":"m"}"#;
    let resp = br#"{"model":"m","choices":[{"message":{"content":"calc","tool_calls":[{"id":"c1","function":{"name":"add","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    let parts = v[0]["parts"].as_array().unwrap();
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "tool_call");
    assert_eq!(v[0]["finish_reason"], "tool_call");
}

#[test]
fn openai_multiple_choices_emit_multiple_messages() {
    let req = br#"{"model":"m"}"#;
    let resp = br#"{"model":"m","choices":[{"message":{"content":"a"},"finish_reason":"stop"},{"message":{"content":"b"},"finish_reason":"tool_calls"}]}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        resp,
        Limits::default(),
    )
    .unwrap();
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    assert_eq!(v.as_array().unwrap().len(), 2);
    assert_eq!(v[0]["parts"][0]["content"], "a");
    assert_eq!(v[1]["finish_reason"], "tool_call");
}

// --- Anthropic SSE tool_use reassembly ---

#[test]
fn anthropic_sse_reassembles_tool_use() {
    let raw = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3\",\"usage\":{\"input_tokens\":5,\"output_tokens\":1}}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"toolu_1\",\"name\":\"add\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"7}\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":3}}\n\n",
        );
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        br#"{"model":"claude-3","stream":true}"#,
        raw.as_bytes(),
        Limits::default(),
    )
    .unwrap();
    assert!(s.reassembled_from_stream);
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    assert_eq!(v[0]["parts"][0]["name"], "add");
    assert_eq!(v[0]["parts"][0]["id"], "toolu_1");
    assert_eq!(v[0]["parts"][0]["arguments"]["a"], 17);
}

/// thinking_delta events accumulate to assemble a ReasoningPart, and
/// if text is also present, both parts are returned in the correct order.
#[test]
fn anthropic_sse_thinking_reassembled() {
    let raw = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"c\"}}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"thi\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"nking\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"answer\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        );
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        br#"{"model":"c","stream":true}"#,
        raw.as_bytes(),
        Limits::default(),
    )
    .unwrap();
    assert!(s.reassembled_from_stream);
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    let parts = v[0]["parts"].as_array().unwrap();
    assert_eq!(parts[0]["type"], "reasoning");
    assert_eq!(parts[0]["content"], "thinking");
    assert_eq!(parts[1]["type"], "text");
    assert_eq!(parts[1]["content"], "answer");
}

// --- OpenAI input message parsing ---

const OPENAI_REQ_MESSAGES: &[u8] = r#"{
        "model":"gpt-4o",
        "messages":[
            {"role":"system","content":"You are a weather assistant."},
            {"role":"user","content":"Weather in Seoul?"},
            {"role":"assistant","content":null,
             "tool_calls":[{"id":"call_1","type":"function",
                "function":{"name":"get_weather","arguments":"{\"city\":\"Seoul\"}"}}]},
            {"role":"tool","tool_call_id":"call_1","content":"18 degrees, clear"}
        ]
    }"#
.as_bytes();
const OPENAI_RESP_MIN: &[u8] =
    br#"{"id":"c1","model":"gpt-4o","choices":[{"finish_reason":"stop"}]}"#;

#[test]
fn openai_system_message_to_system_instructions() {
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        OPENAI_REQ_MESSAGES,
        OPENAI_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    let si: serde_json::Value = serde_json::from_str(
        s.system_instructions
            .as_deref()
            .expect("system_instructions"),
    )
    .unwrap();
    assert_eq!(si.as_array().unwrap().len(), 1);
    assert_eq!(si[0]["type"], "text");
    assert_eq!(si[0]["content"], "You are a weather assistant.");
}

#[test]
fn openai_input_messages_roles_and_tool_result() {
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        OPENAI_REQ_MESSAGES,
        OPENAI_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    let im: serde_json::Value =
        serde_json::from_str(s.input_messages.as_deref().expect("input_messages")).unwrap();
    let arr = im.as_array().unwrap();
    // system is not in input.messages → 3 entries for user/assistant/tool
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0]["role"], "user");
    assert_eq!(arr[0]["parts"][0]["type"], "text");
    assert_eq!(arr[0]["parts"][0]["content"], "Weather in Seoul?");
    // assistant past tool_calls → ToolCallRequestPart
    assert_eq!(arr[1]["role"], "assistant");
    assert_eq!(arr[1]["parts"][0]["type"], "tool_call");
    assert_eq!(arr[1]["parts"][0]["id"], "call_1");
    assert_eq!(arr[1]["parts"][0]["name"], "get_weather");
    assert_eq!(arr[1]["parts"][0]["arguments"]["city"], "Seoul");
    // tool role → ToolCallResponsePart (A-3)
    assert_eq!(arr[2]["role"], "tool");
    assert_eq!(arr[2]["parts"][0]["type"], "tool_call_response");
    assert_eq!(arr[2]["parts"][0]["id"], "call_1");
    assert_eq!(arr[2]["parts"][0]["response"], "18 degrees, clear");
}

#[test]
fn openai_developer_role_to_system_instructions() {
    let req = r#"{"model":"gpt-4o","messages":[{"role":"developer","content":"instruction"},{"role":"user","content":"hi"}]}"#.as_bytes();
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        OPENAI_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    let si: serde_json::Value = serde_json::from_str(
        s.system_instructions
            .as_deref()
            .expect("system_instructions"),
    )
    .unwrap();
    assert_eq!(si[0]["content"], "instruction");
}

#[test]
fn openai_unknown_content_block_sets_unmapped_flag() {
    let req = br#"{"model":"gpt-4o","messages":[{"role":"user","content":[{"type":"future_thing","x":1}]}]}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        OPENAI_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    assert!(s.input_messages_has_unmapped);
    let im: serde_json::Value = serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
    assert_eq!(im[0]["parts"][0]["type"], "future_thing");
}

#[test]
fn openai_input_messages_none_when_absent() {
    // request with no messages field → both input_messages/system_instructions are None
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        OPENAI_REQ,
        OPENAI_CHAT,
        Limits::default(),
    )
    .expect("supported");
    assert!(s.input_messages.is_none());
    assert!(s.system_instructions.is_none());
    assert!(!s.input_messages_has_unmapped);
}

// --- OpenAI input media parsing ---

#[test]
fn openai_image_url_data_uri_to_blob_part() {
    let req = r#"{"model":"gpt-4o","messages":[{"role":"user","content":[
            {"type":"text","text":"desc"},
            {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo="}}
        ]}]}"#
        .as_bytes();
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        OPENAI_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    let im: serde_json::Value = serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
    let parts = im[0]["parts"].as_array().unwrap();
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "blob");
    assert_eq!(parts[1]["modality"], "image");
    assert_eq!(parts[1]["mime_type"], "image/png");
    assert_eq!(parts[1]["content"], "iVBORw0KGgo=");
    assert!(!s.input_messages_has_unmapped);
}

#[test]
fn openai_image_url_http_to_uri_part() {
    let req = br#"{"model":"gpt-4o","messages":[{"role":"user","content":[
            {"type":"image_url","image_url":{"url":"https://example.com/a.png"}}
        ]}]}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        OPENAI_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    let im: serde_json::Value = serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
    let p = &im[0]["parts"][0];
    assert_eq!(p["type"], "uri");
    assert_eq!(p["modality"], "image");
    assert_eq!(p["uri"], "https://example.com/a.png");
}

#[test]
fn openai_input_audio_to_blob_part() {
    let req = br#"{"model":"gpt-4o-audio","messages":[{"role":"user","content":[
            {"type":"input_audio","input_audio":{"data":"UklGRg==","format":"wav"}}
        ]}]}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        OPENAI_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    let im: serde_json::Value = serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
    let p = &im[0]["parts"][0];
    assert_eq!(p["type"], "blob");
    assert_eq!(p["modality"], "audio");
    assert_eq!(p["mime_type"], "audio/wav");
    assert_eq!(p["content"], "UklGRg==");
}

#[test]
fn openai_file_id_to_file_part() {
    let req = br#"{"model":"gpt-4o","messages":[{"role":"user","content":[
            {"type":"file","file":{"file_id":"file-abc"}}
        ]}]}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        OPENAI_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    let im: serde_json::Value = serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
    let p = &im[0]["parts"][0];
    assert_eq!(p["type"], "file");
    assert_eq!(p["modality"], "document");
    assert_eq!(p["file_id"], "file-abc");
}

// --- Anthropic input message parsing ---

const ANTHROPIC_RESP_MIN: &[u8] =
    br#"{"id":"msg_1","model":"claude-3-5-sonnet","stop_reason":"end_turn","content":[]}"#;

#[test]
fn anthropic_top_level_system_string_to_system_instructions() {
    let req = r#"{"model":"claude-3-5-sonnet","system":"You are an assistant.","messages":[{"role":"user","content":"hi"}]}"#.as_bytes();
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        req,
        ANTHROPIC_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    let si: serde_json::Value =
        serde_json::from_str(s.system_instructions.as_deref().unwrap()).unwrap();
    assert_eq!(si[0]["type"], "text");
    assert_eq!(si[0]["content"], "You are an assistant.");
    // system is not in input.messages
    let im: serde_json::Value = serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
    assert_eq!(im.as_array().unwrap().len(), 1);
    assert_eq!(im[0]["role"], "user");
}

#[test]
fn anthropic_system_array_to_system_instructions() {
    let req = br#"{"model":"claude-3-5-sonnet","system":[{"type":"text","text":"A"},{"type":"text","text":"B"}],"messages":[{"role":"user","content":"hi"}]}"#;
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        req,
        ANTHROPIC_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    let si: serde_json::Value =
        serde_json::from_str(s.system_instructions.as_deref().unwrap()).unwrap();
    assert_eq!(si.as_array().unwrap().len(), 2);
    assert_eq!(si[1]["content"], "B");
}

#[test]
fn anthropic_tool_result_block_to_tool_call_response_part() {
    let req = r#"{"model":"claude-3-5-sonnet","messages":[
            {"role":"assistant","content":[{"type":"tool_use","id":"tu_1","name":"get_weather","input":{"city":"Seoul"}}]},
            {"role":"user","content":[{"type":"tool_result","tool_use_id":"tu_1","content":"18 degrees"}]}
        ]}"#.as_bytes();
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        req,
        ANTHROPIC_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    let im: serde_json::Value = serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
    assert_eq!(im[0]["parts"][0]["type"], "tool_call");
    assert_eq!(im[0]["parts"][0]["id"], "tu_1");
    assert_eq!(im[1]["role"], "user");
    assert_eq!(im[1]["parts"][0]["type"], "tool_call_response");
    assert_eq!(im[1]["parts"][0]["id"], "tu_1");
    assert_eq!(im[1]["parts"][0]["response"], "18 degrees");
}

#[test]
fn anthropic_image_base64_source_to_blob_part() {
    let req = br#"{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":[
            {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":"/9j/4AAQ="}}
        ]}]}"#;
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        req,
        ANTHROPIC_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    let im: serde_json::Value = serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
    let p = &im[0]["parts"][0];
    assert_eq!(p["type"], "blob");
    assert_eq!(p["modality"], "image");
    assert_eq!(p["mime_type"], "image/jpeg");
    assert_eq!(p["content"], "/9j/4AAQ=");
}

#[test]
fn anthropic_image_url_source_to_uri_part() {
    let req = br#"{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":[
            {"type":"image","source":{"type":"url","url":"https://example.com/a.jpg"}}
        ]}]}"#;
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        req,
        ANTHROPIC_RESP_MIN,
        Limits::default(),
    )
    .expect("supported");
    let im: serde_json::Value = serde_json::from_str(s.input_messages.as_deref().unwrap()).unwrap();
    let p = &im[0]["parts"][0];
    assert_eq!(p["type"], "uri");
    assert_eq!(p["uri"], "https://example.com/a.jpg");
}

// --- OpenAI output audio BlobPart ---

#[test]
fn openai_output_audio_to_blob_part() {
    let resp = r#"{"id":"c1","model":"gpt-4o-audio","choices":[{"finish_reason":"stop",
            "message":{"role":"assistant","content":"hi",
                "audio":{"id":"a1","data":"UklGRg==","transcript":"hi"}}}]}"#
        .as_bytes();
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        OPENAI_REQ,
        resp,
        Limits::default(),
    )
    .expect("supported");
    let om: serde_json::Value =
        serde_json::from_str(s.output_messages.as_deref().unwrap()).unwrap();
    let parts = om[0]["parts"].as_array().unwrap();
    // text + blob coexist
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[1]["type"], "blob");
    assert_eq!(parts[1]["modality"], "audio");
    assert_eq!(parts[1]["content"], "UklGRg==");
}

/// Even if an orphan partial_json delta arrives without a content_block_start, no ghost tool call must be created.
/// The Anthropic API guarantees content_block_start arrives before its deltas, so
/// when a start event is missing (e.g. due to stream truncation) the corresponding delta must be silently ignored.
#[test]
fn anthropic_sse_orphan_partial_json_no_ghost_tool_call() {
    let raw = concat!(
            "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m\",\"model\":\"c\"}}\n\n",
            // partial_json delta arrives directly without content_block_start(tool_use) → orphan delta
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"a\\\":1}\"}}\n\n",
            "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        );
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        br#"{"model":"c","stream":true}"#,
        raw.as_bytes(),
        Limits::default(),
    )
    .unwrap();
    // An orphan partial_json must not create a tool call (parts must be empty).
    // Since stop_reason="end_turn" is present, output_messages must be Some.
    // Wrapping with if let Some would silently let a None regression pass, so a hard unwrap is used.
    let om = s
        .output_messages
        .as_ref()
        .expect("even the orphan case should be Some when finish_reason is present");
    let v: serde_json::Value = serde_json::from_str(om).unwrap();
    let parts = v[0]["parts"].as_array().unwrap();
    assert!(
        parts.is_empty(),
        "an orphan partial_json must not create a ghost tool call: {:?}",
        om
    );
}

// --- Anthropic server_tool SSE reassembly ---

// b"..." byte literals are ASCII-only → since this contains Korean text (weather), str::as_bytes() is used instead.
const ANTHROPIC_SSE_SERVER_TOOL: &[u8] = concat!(
        "event: message_start\n",
        "data: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"model\":\"claude-3-5-sonnet\",\"content\":[]}}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"server_tool_use\",\"id\":\"srv_1\",\"name\":\"web_search\",\"input\":{}}}\n\n",
        "event: content_block_delta\n",
        "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"query\\\":\\\"weather\\\"}\"}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
        "event: content_block_start\n",
        "data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"web_search_tool_result\",\"tool_use_id\":\"srv_1\",\"content\":[{\"type\":\"web_search_result\",\"title\":\"T\"}]}}\n\n",
        "event: content_block_stop\n",
        "data: {\"type\":\"content_block_stop\",\"index\":1}\n\n",
        "event: message_delta\n",
        "data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"}}\n\n",
        "event: message_stop\n",
        "data: {\"type\":\"message_stop\"}\n\n",
    ).as_bytes();

#[test]
fn anthropic_sse_server_tool_use_reassembled() {
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        b"{}",
        ANTHROPIC_SSE_SERVER_TOOL,
        Limits::default(),
    )
    .expect("supported");
    assert!(s.reassembled_from_stream);
    let om: serde_json::Value =
        serde_json::from_str(s.output_messages.as_deref().unwrap()).unwrap();
    let parts = om[0]["parts"].as_array().unwrap();
    let stc = parts
        .iter()
        .find(|p| p["type"] == "server_tool_call")
        .expect("server_tool_call");
    assert_eq!(stc["name"], "web_search");
    assert_eq!(stc["server_tool_call"]["query"], "weather");
}

#[test]
fn anthropic_sse_web_search_result_reassembled() {
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        b"{}",
        ANTHROPIC_SSE_SERVER_TOOL,
        Limits::default(),
    )
    .expect("supported");
    let om: serde_json::Value =
        serde_json::from_str(s.output_messages.as_deref().unwrap()).unwrap();
    let parts = om[0]["parts"].as_array().unwrap();
    let res = parts
        .iter()
        .find(|p| p["type"] == "server_tool_call_response")
        .expect("response part");
    assert_eq!(res["server_tool_call_response"][0]["title"], "T");
}

// --- the open usage model (fixture-backed) --------------------------------

/// Loads one fixture file at compile time.
macro_rules! fixture {
    ($case:literal, $file:literal) => {
        include_bytes!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/llm/",
            $case,
            "/",
            $file
        )) as &[u8]
    };
}

/// T-R6 / U4 — every path in a provider's `NORMALIZED_USAGE_PATHS` table is
/// really extracted from that provider's fixture, and the normalized getter
/// agrees with the mirror leaf at the same path. For the one ExcludesCache
/// provider the input getter equals leaf(input) + leaf(cache_read) +
/// leaf(cache_creation) — the S-3 rescope: `gen_ai.usage.*` is the semconv
/// norm (inclusive), `wardex.usage.*` is the provider's own text (raw).
#[test]
fn every_normalized_usage_path_is_extracted_from_its_fixture() {
    struct Case {
        host: &'static str,
        path: &'static str,
        req: &'static [u8],
        resp: &'static [u8],
        table: &'static [(&'static str, &'static str)],
        input_excludes_cache: bool,
    }
    let cases = [
        Case {
            host: "api.openai.com",
            path: "/v1/chat/completions",
            req: fixture!("openai_chat", "request.json"),
            resp: fixture!("openai_chat", "response.json"),
            table: super::openai_chat::NORMALIZED_USAGE_PATHS,
            input_excludes_cache: false,
        },
        Case {
            host: "api.anthropic.com",
            path: "/v1/messages",
            req: fixture!("anthropic_messages", "request.json"),
            resp: fixture!("anthropic_messages", "response.json"),
            table: super::anthropic::NORMALIZED_USAGE_PATHS,
            input_excludes_cache: true,
        },
        Case {
            host: "api.openai.com",
            path: "/v1/responses",
            req: fixture!("openai_responses", "request.json"),
            resp: fixture!("openai_responses", "response.json"),
            table: super::openai_responses::NORMALIZED_USAGE_PATHS,
            input_excludes_cache: false,
        },
        Case {
            host: "api.openai.com",
            path: "/v1/embeddings",
            req: fixture!("openai_embeddings", "request.json"),
            resp: fixture!("openai_embeddings", "response.json"),
            table: super::openai_embeddings::NORMALIZED_USAGE_PATHS,
            input_excludes_cache: false,
        },
    ];
    for case in &cases {
        let s = parse_llm(case.host, case.path, case.req, case.resp, Limits::default())
            .expect("fixture parses");
        let leaf = |path: &str| -> i64 {
            match s.usage_leaves.iter().find(|(p, _)| p == path) {
                Some((_, UsageLeaf::Int(v))) => *v,
                other => panic!("{}: leaf {path} missing or non-int: {other:?}", case.path),
            }
        };
        for (field, path) in case.table {
            let mirrored = leaf(path);
            let normalized = match *field {
                "input_tokens" => s.usage.input_tokens(),
                "output_tokens" => s.usage.output_tokens(),
                "cache_read_input_tokens" => s.usage.cache_read_input_tokens(),
                "cache_creation_input_tokens" => s.usage.cache_creation_input_tokens(),
                "reasoning_output_tokens" => s.usage.reasoning_output_tokens(),
                other => panic!("table names an unknown field {other}"),
            }
            .unwrap_or_else(|| panic!("{}: {field} not extracted", case.path));
            if *field == "input_tokens" && case.input_excludes_cache {
                let tiers = leaf("cache_read_input_tokens") + leaf("cache_creation_input_tokens");
                assert_eq!(
                    normalized,
                    mirrored + tiers,
                    "{}: inclusive input != raw leaf + cache tiers",
                    case.path
                );
            } else {
                assert_eq!(
                    normalized, mirrored,
                    "{}: {field} != leaf {path}",
                    case.path
                );
            }
        }
    }
}

/// T-R8 — the three Anthropic additions, each red before this commit.
#[test]
fn anthropic_thinking_tokens_map_to_reasoning_output_tokens() {
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        fixture!("anthropic_messages", "request.json"),
        fixture!("anthropic_messages", "response.json"),
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.usage.reasoning_output_tokens(), Some(120));
}

#[test]
fn anthropic_sse_message_delta_usage_is_merged_deep() {
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        fixture!("anthropic_messages_sse", "request.json"),
        fixture!("anthropic_messages_sse", "stream.sse"),
        Limits::default(),
    )
    .unwrap();
    let leaves: std::collections::BTreeMap<&str, &UsageLeaf> = s
        .usage_leaves
        .iter()
        .map(|(p, v)| (p.as_str(), v))
        .collect();
    // From the start event, kept through the merge:
    assert_eq!(
        leaves.get("cache_creation.ephemeral_5m_input_tokens"),
        Some(&&UsageLeaf::Int(2000))
    );
    // From the delta, overwriting the start's 0 (deep, key-wise):
    assert_eq!(
        leaves.get("cache_creation.ephemeral_1h_input_tokens"),
        Some(&&UsageLeaf::Int(64))
    );
    // Only in the delta — a subtree the start never had:
    assert_eq!(
        leaves.get("server_tool_use.web_search_requests"),
        Some(&&UsageLeaf::Int(2))
    );
    assert_eq!(s.usage.output_tokens(), Some(500));
    assert_eq!(s.usage.input_tokens(), Some(11000), "inclusive total");
    assert_eq!(s.stream_terminated, Some(true));
}

#[test]
fn anthropic_output_config_effort_is_reasoning_level() {
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        fixture!("anthropic_messages", "request.json"),
        fixture!("anthropic_messages", "response.json"),
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.reasoning_level.as_deref(), Some("medium"));
}

/// T-R9 — Chat Completions additions.
#[test]
fn openai_chat_service_tier_fingerprint_reasoning_effort_extracted() {
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        fixture!("openai_chat", "request.json"),
        fixture!("openai_chat", "response.json"),
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.api_type, Some("chat_completions"));
    assert_eq!(s.request_service_tier.as_deref(), Some("auto"));
    assert_eq!(s.response_service_tier.as_deref(), Some("default"));
    assert_eq!(s.system_fingerprint.as_deref(), Some("fp_fx1"));
    assert_eq!(s.reasoning_level.as_deref(), Some("low"));
}

#[test]
fn chat_response_format_json_sets_output_type_json() {
    let req =
        br#"{"model":"m","response_format":{"type":"json_schema","json_schema":{"name":"x"}}}"#;
    let resp = br#"{"model":"m","choices":[{"message":{"content":"{}"},"finish_reason":"stop"}]}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        req,
        resp,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.output_type.as_deref(), Some("json"));
    // and the default stays what it always was:
    let s2 = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        br#"{"model":"m"}"#,
        resp,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s2.output_type.as_deref(), Some("text"));
}

/// T-R10 — embeddings request semantics, red before this commit.
#[test]
fn embeddings_request_model_formats_dimensions_and_no_output_type() {
    let s = parse_llm(
        "api.openai.com",
        "/v1/embeddings",
        fixture!("openai_embeddings", "request.json"),
        fixture!("openai_embeddings", "response.json"),
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.request_model.as_deref(), Some("text-embedding-3-small"));
    assert_eq!(
        s.encoding_formats.as_deref(),
        Some(&["float".to_string()][..])
    );
    assert_eq!(s.embedding_dimensions, Some(256));
    assert_eq!(
        s.output_type, None,
        "an embeddings response is vectors, not text"
    );
}

/// T-R14 — terminal detection is about the provider's grammar, not `[DONE]`
/// alone (compatible gateways omit it), and absence is reported, not guessed.
#[test]
fn chat_stream_without_done_but_with_finish_reason_is_terminated() {
    let sse = b"data: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":null}]}\n\ndata: {\"id\":\"c\",\"model\":\"m\",\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n";
    let s = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        br#"{"model":"m"}"#,
        sse,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.stream_terminated, Some(true));
}

#[test]
fn anthropic_stream_without_message_stop_is_unterminated() {
    let sse = b"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"m1\",\"model\":\"c\",\"usage\":{\"input_tokens\":9,\"output_tokens\":1}}}\n\nevent: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n";
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        br#"{"model":"c"}"#,
        sse,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.stream_terminated, Some(false));
    // non-SSE stays None — the field is about streams only
    let plain = parse_llm(
        "api.anthropic.com",
        "/v1/messages",
        br#"{"model":"c"}"#,
        br#"{"model":"c","content":[{"type":"text","text":"x"}]}"#,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(plain.stream_terminated, None);
}

/// G5 shape at the Rust boundary: the cap is the limit field, observed.
#[test]
fn usage_leaves_are_capped_by_max_extra_keys() {
    let resp = br#"{"model":"m","choices":[{"message":{"content":"x"},"finish_reason":"stop"}],
        "usage":{"a":1,"b":2,"c":3,"d":4,"e":5,"prompt_tokens":6,"completion_tokens":7}}"#;
    let limited = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        br#"{"model":"m"}"#,
        resp,
        Limits {
            max_extra_keys: 3,
            ..Limits::default()
        },
    )
    .unwrap();
    assert_eq!(limited.usage_leaves.len(), 3);
    assert_eq!(limited.usage_dropped_count, 4);
    // U2: the normalized fields are extracted separately and never capped.
    assert_eq!(limited.usage.input_tokens(), Some(6));
    assert_eq!(limited.usage.output_tokens(), Some(7));
    let unlimited = parse_llm(
        "api.openai.com",
        "/v1/chat/completions",
        br#"{"model":"m"}"#,
        resp,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(unlimited.usage_dropped_count, 0);
    assert_eq!(unlimited.usage_leaves.len(), 7);
}

/// T-R13 — the regression pin for the fabricated body: a Responses SSE
/// stream on the OpenAI host must never come back as a chat-shaped JSON the
/// Chat reassembler invented (empty content, null id/model). The dispatch
/// goes by Api now, so the Chat reassembler cannot receive the stream.
#[test]
fn responses_sse_on_openai_host_never_yields_a_fabricated_chat_body() {
    let sse = b"event: response.created\ndata: {\"type\":\"response.created\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"object\":\"response\",\"status\":\"in_progress\",\"model\":\"gpt-4.1\",\"output\":[]}}\n\n";
    let s = parse_llm(
        "api.openai.com",
        "/v1/responses",
        br#"{"model":"gpt-4.1","stream":true}"#,
        sse,
        Limits::default(),
    )
    .unwrap();
    let body = String::from_utf8(s.decoded_response.unwrap()).unwrap();
    assert!(
        !body.contains("\"choices\""),
        "a Responses stream must not be re-shaped as a chat completion: {body}"
    );
}

/// The substring false positives, fixed: token counting and batch management
/// under /v1/messages are NOT chat calls, and their spans must not carry a
/// false `semantic_parse_failed`.
#[test]
fn count_tokens_and_batches_are_not_chat_calls() {
    for path in ["/v1/messages/count_tokens", "/v1/messages/batches"] {
        let s = parse_llm(
            "api.anthropic.com",
            path,
            br#"{"model":"claude-sonnet-4-6","messages":[]}"#,
            br#"{"input_tokens": 14}"#,
            Limits::default(),
        );
        assert!(s.is_none(), "{path} parsed as an LLM call");
    }
}

// --- OpenAI Responses API (fixture-backed; typed-model-validated) ----------

fn parse_responses_fixture(req: &'static [u8], resp: &'static [u8]) -> LlmSemantics {
    parse_llm(
        "api.openai.com",
        "/v1/responses",
        req,
        resp,
        Limits::default(),
    )
    .expect("responses fixture parses")
}

/// T-R1 — the openai-agents default path, visible: a non-streaming
/// /v1/responses call yields chat semantics with usage (it yielded None
/// before this parser existed).
#[test]
fn responses_non_streaming_is_parsed_as_chat_with_usage() {
    let s = parse_responses_fixture(
        fixture!("openai_responses", "request.json"),
        fixture!("openai_responses", "response.json"),
    );
    assert_eq!(s.provider, "openai");
    assert_eq!(s.operation, "chat");
    assert_eq!(s.api_type, Some("responses"));
    assert_eq!(s.request_model.as_deref(), Some("gpt-4.1"));
    assert_eq!(s.response_model.as_deref(), Some("gpt-4.1-2025-04-14"));
    assert_eq!(s.response_id.as_deref(), Some("resp_fx1"));
    assert_eq!(s.usage.input_tokens(), Some(52));
    assert_eq!(s.usage.output_tokens(), Some(17));
    assert_eq!(s.usage.cache_read_input_tokens(), Some(16));
    assert_eq!(s.usage.reasoning_output_tokens(), Some(4));
    assert_eq!(s.finish_reasons.as_deref(), Some(&["stop".to_string()][..]));
    assert_eq!(s.response_status.as_deref(), Some("completed"));
    assert_eq!(s.reasoning_level.as_deref(), Some("low"));
    assert_eq!(s.request_service_tier.as_deref(), Some("auto"));
    assert_eq!(s.response_service_tier.as_deref(), Some("default"));
    assert_eq!(s.max_tokens, Some(256));
    // The system prompt and the string input both landed:
    assert!(s.system_instructions.is_some());
    assert!(s.input_messages.unwrap().contains("Say hello."));
    // The open mirror carries the leaf the CLOSED table era would have
    // dropped: openai 3.3.1's typed usage REQUIRES cache_write_tokens.
    assert!(s
        .usage_leaves
        .iter()
        .any(|(p, _)| p == "input_tokens_details.cache_write_tokens"));
}

/// T-R2 — the correlation key: `call_id`, not the item id.
#[test]
fn responses_function_call_becomes_tool_call_part_with_call_id() {
    let s = parse_responses_fixture(
        fixture!("openai_responses_tools", "request.json"),
        fixture!("openai_responses_tools", "response.json"),
    );
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    let part = &v[0]["parts"][0];
    assert_eq!(part["type"], "tool_call");
    assert_eq!(part["id"], "call_weather_2");
    assert_eq!(part["name"], "get_weather");
    assert_eq!(part["arguments"]["city"], "Busan");
    assert_eq!(
        s.finish_reasons.as_deref(),
        Some(&["tool_call".to_string()][..])
    );
    // The request half: agents-SDK history shape (captured off a live
    // openai-agents run against a mock) — the prior call and its output.
    let inputs: serde_json::Value = serde_json::from_str(&s.input_messages.unwrap()).unwrap();
    let all = inputs.to_string();
    assert!(all.contains("\"tool_call\""));
    assert!(all.contains("\"tool_call_response\""));
    assert!(all.contains("call_weather_1"));
    assert_eq!(s.previous_response_id.as_deref(), Some("resp_fx_prev"));
}

/// T-R3 — reasoning items and status mapping.
#[test]
fn responses_reasoning_item_becomes_reasoning_part() {
    let s = parse_responses_fixture(
        fixture!("openai_responses_reasoning", "request.json"),
        fixture!("openai_responses_reasoning", "response.json"),
    );
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    assert_eq!(v[0]["parts"][0]["type"], "reasoning");
    assert_eq!(
        v[0]["parts"][0]["content"],
        "First, consider.\nThen, conclude."
    );
    assert_eq!(v[0]["parts"][1]["type"], "text");
    assert_eq!(s.usage.reasoning_output_tokens(), Some(64));
    assert_eq!(s.reasoning_level.as_deref(), Some("high"));
}

#[test]
fn responses_incomplete_maps_reason_to_length() {
    let s = parse_responses_fixture(
        fixture!("openai_responses_incomplete", "request.json"),
        fixture!("openai_responses_incomplete", "response.json"),
    );
    assert_eq!(s.response_status.as_deref(), Some("incomplete"));
    assert_eq!(
        s.finish_reasons.as_deref(),
        Some(&["length".to_string()][..])
    );
}

#[test]
fn responses_http_error_envelope_claims_nothing_about_the_response_half() {
    // The bare error envelope every 4xx/5xx carries is NOT a Response
    // object: no status, no finish reason, no output message may be derived
    // from it. Same contract `test_llm_error_spans.py` pins for the sibling
    // `/v1/messages` endpoint; the request identity still comes through.
    let resp = br#"{"error":{"type":"rate_limit_exceeded","message":"slow down"}}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/responses",
        br#"{"model":"gpt-4.1"}"#,
        resp,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.request_model.as_deref(), Some("gpt-4.1"));
    assert_eq!(s.response_status, None);
    assert_eq!(s.finish_reasons, None);
    assert_eq!(s.output_messages, None);
}

/// Anthropic's documented OpenAI-SDK compatibility endpoint: an
/// anthropic-named HOST serving the Chat Completions SHAPE. The API shape
/// picks the parser and the host picks the provider label — requiring the
/// two to agree dropped this endpoint's non-streaming capture entirely
/// (empty semantics, so no span under AGENT mode) while the SSE half kept
/// capturing it. Both forms are pinned so the pair cannot drift apart.
#[test]
fn anthropic_openai_compat_endpoint_parses_chat_with_the_host_label() {
    let req = br#"{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hi"}]}"#;
    let resp = br#"{"id":"chatcmpl-compat","object":"chat.completion","model":"claude-sonnet-4-6","choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}"#;
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/chat/completions",
        req,
        resp,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.provider, "anthropic");
    assert_eq!(s.request_model.as_deref(), Some("claude-sonnet-4-6"));
    assert_eq!(s.response_id.as_deref(), Some("chatcmpl-compat"));
    assert_eq!(s.usage.input_tokens(), Some(3));
    assert_eq!(s.usage.output_tokens(), Some(2));
    assert_eq!(s.finish_reasons.as_deref(), Some(&["stop".to_string()][..]));
}

#[test]
fn anthropic_openai_compat_endpoint_sse_keeps_the_host_label() {
    let s = parse_llm(
        "api.anthropic.com",
        "/v1/chat/completions",
        b"{}",
        OPENAI_SSE,
        Limits::default(),
    )
    .unwrap();
    assert!(s.reassembled_from_stream);
    assert_eq!(
        s.provider, "anthropic",
        "the host names the provider; the Api names the parser"
    );
    assert_eq!(s.response_id.as_deref(), Some("chatcmpl-s"));
    assert_eq!(s.finish_reasons.as_deref(), Some(&["stop".to_string()][..]));
}

#[test]
fn responses_failed_maps_to_error() {
    let resp = br#"{"id":"resp_f","object":"response","status":"failed","model":"gpt-4.1",
        "error":{"code":"server_error","message":"boom"},"output":[]}"#;
    let s = parse_llm(
        "api.openai.com",
        "/v1/responses",
        br#"{"model":"gpt-4.1"}"#,
        resp,
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.response_status.as_deref(), Some("failed"));
    assert_eq!(
        s.finish_reasons.as_deref(),
        Some(&["error".to_string()][..])
    );
}

/// T-R4 — the terminal snapshot is the truth source: the fixture's deltas
/// and even its output_item.done deliberately DISAGREE with the snapshot,
/// and the snapshot's values are the ones extracted.
#[test]
fn responses_sse_takes_completed_snapshot_not_deltas() {
    let s = parse_responses_fixture(
        fixture!("openai_responses_sse", "request.json"),
        fixture!("openai_responses_sse", "stream.sse"),
    );
    assert!(s.reassembled_from_stream);
    assert_eq!(s.stream_terminated, Some(true));
    assert_eq!(s.usage.input_tokens(), Some(9));
    assert_eq!(s.usage.output_tokens(), Some(6));
    assert_eq!(s.usage.cache_read_input_tokens(), Some(2));
    let body = String::from_utf8(s.decoded_response.clone().unwrap()).unwrap();
    assert!(body.contains("Hello from the snapshot!"));
    assert!(!body.contains("DELTAS"), "deltas must not win: {body}");
    // and the decoded body is Response-shaped, not chat-shaped:
    assert!(body.contains("\"output\""));
    assert!(!body.contains("\"choices\""));
    assert_eq!(s.response_status.as_deref(), Some("completed"));
}

/// T-R5 — no terminal event: items reconstructed (done whole, added+deltas
/// merged, orphan deltas ignored), usage honestly absent, unterminated.
#[test]
fn responses_sse_without_terminal_falls_back_to_items_and_marks_unterminated() {
    let s = parse_responses_fixture(
        fixture!("openai_responses_sse_unterminated", "request.json"),
        fixture!("openai_responses_sse_unterminated", "stream.sse"),
    );
    assert_eq!(s.stream_terminated, Some(false));
    assert_eq!(s.usage.output_tokens(), None);
    assert_eq!(s.response_status.as_deref(), Some("in_progress"));
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    let parts = v[0]["parts"].as_array().unwrap();
    // done item came through whole; the added-only function call was
    // restored from its argument deltas; the ghost item was ignored.
    assert_eq!(parts[0]["type"], "text");
    assert_eq!(parts[0]["content"], "Partial answer.");
    assert_eq!(parts[1]["type"], "tool_call");
    assert_eq!(parts[1]["id"], "call_sse2");
    assert_eq!(parts[1]["arguments"]["city"], "Seoul");
    assert_eq!(parts.len(), 2, "the orphan delta must not become a part");
}

/// T-R5b — the in-stream `error` event is the third terminal form: its
/// payload (the only bytes naming the failure) is preserved into the
/// synthetic body, and status/finish are the provider's own declaration.
#[test]
fn responses_sse_error_event_preserves_error_and_maps_status() {
    let s = parse_responses_fixture(
        fixture!("openai_responses_sse_error", "request.json"),
        fixture!("openai_responses_sse_error", "stream.sse"),
    );
    assert_eq!(s.stream_terminated, Some(true));
    assert_eq!(s.response_status.as_deref(), Some("failed"));
    assert_eq!(
        s.finish_reasons.as_deref(),
        Some(&["error".to_string()][..])
    );
    assert_eq!(
        s.usage.output_tokens(),
        None,
        "no usage arrived; none is invented"
    );
    let body: serde_json::Value =
        serde_json::from_slice(&s.decoded_response.clone().unwrap()).unwrap();
    assert_eq!(body["error"]["code"], "server_error");
    assert_eq!(
        body["error"]["message"],
        "The model failed to generate a response."
    );
}

/// T-R5c — background mode: the create answers 200/queued with usage null.
/// A tokenless, finish-less, marker-less identified span is the DESIGN here
/// (the usage lands only on the deferred fetch_response path), pinned so the
/// shape is a decision rather than an accident.
#[test]
fn responses_background_queued_create_is_identified_but_tokenless() {
    let s = parse_responses_fixture(
        fixture!("openai_responses_background_queued", "request.json"),
        fixture!("openai_responses_background_queued", "response.json"),
    );
    assert_eq!(s.response_model.as_deref(), Some("gpt-4.1-2025-04-14"));
    assert_eq!(s.response_status.as_deref(), Some("queued"));
    assert_eq!(s.usage.input_tokens(), None);
    assert_eq!(s.usage.output_tokens(), None);
    assert_eq!(s.finish_reasons, None);
    assert!(s.usage_leaves.is_empty());
    assert_eq!(s.usage_dropped_count, 0);
}

/// T-R12's parse-level half: a Responses body on localhost resolves through
/// the ordered body-shape check, not to Anthropic (Responses bodies carry
/// `usage.input_tokens` too).
#[test]
fn responses_body_on_localhost_is_not_mistaken_for_anthropic() {
    let s = parse_llm(
        "127.0.0.1",
        "/v1/responses",
        fixture!("openai_responses", "request.json"),
        fixture!("openai_responses", "response.json"),
        Limits::default(),
    )
    .unwrap();
    assert_eq!(s.provider, "openai");
    assert_eq!(s.api_type, Some("responses"));
    assert_eq!(s.usage.input_tokens(), Some(52));
}

/// A Responses output `message` item that carries a role other than
/// `assistant` (a compaction echoes the caller's own messages) keeps that
/// role in `gen_ai.output.messages` instead of being folded into the
/// assistant's message as words the model never said.
#[test]
fn user_role_output_items_keep_their_role() {
    let resp = br#"{"id":"resp_r","object":"response","status":"completed","output":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hello"}]},{"type":"message","role":"assistant","content":[{"type":"output_text","text":"hi"}]}]}"#;
    let s = parse_responses_fixture(br#"{"model":"gpt-5.1","input":"x"}"#, resp);
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    let msgs = v.as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["parts"][0]["type"], "text");
    assert_eq!(msgs[0]["parts"][0]["content"], "hello");
    assert!(msgs[0].get("finish_reason").is_none());
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(msgs[1]["parts"][0]["content"], "hi");
    assert_eq!(msgs[1]["finish_reason"], "stop");
    for m in msgs.iter().filter(|m| m["role"] == "assistant") {
        for part in m["parts"].as_array().unwrap() {
            assert_ne!(
                part["content"], "hello",
                "user text leaked into the assistant"
            );
        }
    }
}

/// `POST /v1/responses/compact` is a billable LLM call: model + input in,
/// usage + output items out. It has no assistant turn (no `status` on the
/// wire), so finish_reasons/response_status stay empty; its echoed user
/// message keeps its role and the compaction item ships as an unmapped
/// generic part rather than as anything the model said.
#[test]
fn responses_compact_is_a_billable_llm_call() {
    let s = parse_llm(
        "api.openai.com",
        "/v1/responses/compact",
        fixture!("openai_responses_compact", "request.json"),
        fixture!("openai_responses_compact", "response.json"),
        Limits::default(),
    )
    .expect("compact fixture parses");
    assert_eq!(s.operation, "chat");
    assert_eq!(s.api_type, Some("responses"));
    assert_eq!(s.request_model.as_deref(), Some("gpt-5.1"));
    assert_eq!(s.response_id.as_deref(), Some("resp_cmp1"));
    assert_eq!(s.usage.input_tokens(), Some(1200));
    assert_eq!(s.usage.output_tokens(), Some(300));
    assert!(s.finish_reasons.is_none());
    assert!(s.response_status.is_none());
    assert!(s.output_messages_has_unmapped);
    let v: serde_json::Value = serde_json::from_str(&s.output_messages.unwrap()).unwrap();
    let msgs = v.as_array().unwrap();
    assert_eq!(msgs.len(), 2);
    assert_eq!(msgs[0]["role"], "user");
    assert_eq!(msgs[0]["parts"][0]["content"], "hello");
    assert_eq!(msgs[1]["role"], "assistant");
    assert_eq!(
        msgs[1]["parts"],
        serde_json::json!([{"type": "compaction"}]),
        "the compaction item is the assistant's only part"
    );
    for part in msgs[1]["parts"].as_array().unwrap() {
        assert_ne!(part.get("content"), Some(&serde_json::json!("hello")));
    }
}
