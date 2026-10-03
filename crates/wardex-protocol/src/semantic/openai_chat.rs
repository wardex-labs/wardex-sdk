//! OpenAI Chat Completions: request/response fill and SSE reassembly.

use super::parts::*;
use super::usage::{flatten_usage, UsageBounds, UsageView};
use super::{LlmSemantics, StringOrVec};
use crate::sse::SseEvent;
use crate::usage::{InputConvention, TokenUsage};
use serde::Deserialize;

// Normalization table: LlmSemantics usage field <- dotted path in the raw
// usage Value. The extraction below reads THROUGH these constants, so the
// table a test walks and the path the code reads are one string (U4).
const P_INPUT: &str = "prompt_tokens";
const P_OUTPUT: &str = "completion_tokens";
const P_CACHE_READ: &str = "prompt_tokens_details.cached_tokens";
const P_REASONING: &str = "completion_tokens_details.reasoning_tokens";
// Read by the fixture test (`every_normalized_usage_path_is_extracted_
// from_its_fixture`), not by the runtime path — the runtime reads the
// P_* consts the table is built from, which makes the two one string.
#[cfg_attr(not(test), allow(dead_code))]
pub(super) const NORMALIZED_USAGE_PATHS: &[(&str, &str)] = &[
    ("input_tokens", P_INPUT),
    ("output_tokens", P_OUTPUT),
    ("cache_read_input_tokens", P_CACHE_READ),
    ("reasoning_output_tokens", P_REASONING),
];

// --- OpenAI structs ---

#[derive(Deserialize)]
struct OAFunction {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}
#[derive(Deserialize)]
struct OAToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<OAFunction>,
}
#[derive(Deserialize, Default)]
struct OAMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    refusal: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<OAToolCall>>,
    #[serde(default)]
    audio: Option<serde_json::Value>,
}

#[derive(Deserialize)]
struct OAChoice {
    #[serde(default)]
    finish_reason: Option<String>,
    #[serde(default)]
    message: Option<OAMessage>,
}
#[derive(Deserialize)]
struct OpenAIChatResponse {
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    choices: Vec<OAChoice>,
    #[serde(default)]
    usage: Option<serde_json::Value>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    system_fingerprint: Option<String>,
    /// Present on a bare HTTP error envelope, and on the synthetic body of a
    /// stream that carried an in-stream `error` chunk (`reassemble_openai`).
    #[serde(default)]
    error: Option<serde_json::Value>,
}
#[derive(Deserialize)]
struct OpenAIChatRequest {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    max_tokens: Option<i64>,
    #[serde(default)]
    max_completion_tokens: Option<i64>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    frequency_penalty: Option<f64>,
    #[serde(default)]
    presence_penalty: Option<f64>,
    #[serde(default)]
    seed: Option<i64>,
    #[serde(default)]
    n: Option<i64>,
    #[serde(default)]
    stream: Option<bool>,
    #[serde(default)]
    stop: Option<StringOrVec>,
    #[serde(default)]
    messages: Option<Vec<serde_json::Value>>,
    #[serde(default)]
    service_tier: Option<String>,
    #[serde(default)]
    reasoning_effort: Option<String>,
    #[serde(default)]
    response_format: Option<serde_json::Value>,
}

/// OpenAI chat request messages[] → system_instructions + input.messages.
fn fill_openai_input(out: &mut LlmSemantics, messages: &[serde_json::Value]) {
    let mut sys_parts: Vec<serde_json::Value> = Vec::new();
    let mut msgs: Vec<InMsg> = Vec::new();
    for m in messages {
        let role = m.get("role").and_then(|x| x.as_str()).unwrap_or("");
        let content = m.get("content");
        match role {
            "system" | "developer" => {
                if let Some(c) = content.and_then(|x| x.as_str()) {
                    sys_parts.push(text_part(c.to_string()));
                }
            }
            "tool" => {
                let id = m
                    .get("tool_call_id")
                    .and_then(|x| x.as_str())
                    .map(str::to_string);
                let response = content.cloned().unwrap_or(serde_json::Value::Null);
                msgs.push(InMsg {
                    role: "tool".to_string(),
                    parts: vec![tool_call_response_part(id, response)],
                });
            }
            "user" | "assistant" => {
                let mut parts = openai_content_to_parts(content, out);
                // assistant past tool_calls → ToolCallRequestPart
                if let Some(tcs) = m.get("tool_calls").and_then(|x| x.as_array()) {
                    for tc in tcs {
                        let id = tc.get("id").and_then(|x| x.as_str()).map(str::to_string);
                        let f = tc.get("function");
                        let name = f
                            .and_then(|x| x.get("name"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("")
                            .to_string();
                        let raw = f
                            .and_then(|x| x.get("arguments"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        let arguments = serde_json::from_str::<serde_json::Value>(raw)
                            .unwrap_or_else(|_| serde_json::Value::from(raw));
                        parts.push(tool_call_part(id, name, arguments));
                    }
                }
                msgs.push(InMsg {
                    role: role.to_string(),
                    parts,
                });
            }
            _ => {}
        }
    }
    out.system_instructions = build_system_instructions(sys_parts);
    out.input_messages = build_input_messages(msgs);
}

/// OpenAI message content (string | array) → part list. Handles the media blocks
/// alongside text: `image_url`, `input_audio` and `file`.
fn openai_content_to_parts(
    content: Option<&serde_json::Value>,
    out: &mut LlmSemantics,
) -> Vec<serde_json::Value> {
    let mut parts: Vec<serde_json::Value> = Vec::new();
    match content {
        Some(serde_json::Value::String(s)) if !s.is_empty() => {
            parts.push(text_part(s.clone()));
        }
        Some(serde_json::Value::Array(blocks)) => {
            for b in blocks {
                let bt = b.get("type").and_then(|x| x.as_str()).unwrap_or("");
                match bt {
                    "text" => {
                        let t = b.get("text").and_then(|x| x.as_str()).unwrap_or("");
                        parts.push(text_part(t.to_string()));
                    }
                    "image_url" => {
                        let url = b
                            .get("image_url")
                            .and_then(|x| x.get("url"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        if let Some((mime, b64)) = parse_data_url(url) {
                            parts.push(blob_part(modality_from_mime(mime), Some(mime), b64));
                        } else if !url.is_empty() {
                            parts.push(uri_part("image", None, url));
                        }
                    }
                    "input_audio" => {
                        let audio = b.get("input_audio");
                        let data = audio
                            .and_then(|x| x.get("data"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        let fmt = audio
                            .and_then(|x| x.get("format"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        let mime = format!("audio/{fmt}");
                        parts.push(blob_part("audio", Some(&mime), data));
                    }
                    "file" => {
                        let file_id = b
                            .get("file")
                            .and_then(|x| x.get("file_id"))
                            .and_then(|x| x.as_str())
                            .unwrap_or("");
                        parts.push(file_part("document", None, file_id));
                    }
                    other => {
                        parts.push(generic_part(other));
                        out.input_messages_has_unmapped = true;
                    }
                }
            }
        }
        _ => {}
    }
    parts
}

pub(super) fn fill_openai_chat(
    out: &mut LlmSemantics,
    req: &[u8],
    resp: &[u8],
    bounds: UsageBounds,
) {
    out.api_type = Some("chat_completions");
    // The pre-split dispatcher stamped `output_type="text"` on every parse;
    // the request half below refines it from `response_format.type`.
    out.output_type = Some("text".to_string());
    if let Ok(r) = serde_json::from_slice::<OpenAIChatResponse>(resp) {
        // An error object INSIDE a response object is the provider's own
        // failure declaration — the in-stream `error` chunk, which the
        // reassembler keeps in the synthetic body — and it wins over whatever
        // finish the choices carried. A bare HTTP error envelope
        // (`{"error":{...}}`, no id, no choices) is not a response object:
        // it keeps an EMPTY response half, as before, and its span takes the
        // error from the HTTP status.
        let is_response_object = r.id.is_some() || !r.choices.is_empty();
        let declared = r
            .error
            .as_ref()
            .filter(|e| is_response_object && !e.is_null())
            .map(declared_error_type);
        out.response_id = r.id;
        out.response_model = r.model;
        let mut fr: Vec<String> = Vec::new();
        let mut msgs: Vec<OutMsg> = Vec::new();
        for c in r.choices {
            // One producer for both carriers: the normalized value goes into
            // `finish_reasons` AND onto the message (unknown raw passes
            // through as itself — total function, never a silent drop).
            let finish_reason = if declared.is_some() {
                Some(FINISH_ERROR.to_string())
            } else {
                c.finish_reason
                    .as_deref()
                    .map(|f| normalize_finish_reason("openai", f))
            };
            if let Some(f) = &finish_reason {
                fr.push(f.clone());
            }
            let mut parts: Vec<serde_json::Value> = Vec::new();
            if let Some(msg) = c.message {
                if let Some(content) = msg.content {
                    if !content.is_empty() {
                        parts.push(text_part(content));
                    }
                }
                if let Some(refusal) = msg.refusal {
                    if !refusal.is_empty() {
                        parts.push(text_part(refusal));
                    }
                }
                if let Some(tcs) = msg.tool_calls {
                    for tc in tcs {
                        if let Some(func) = tc.function {
                            let name = func.name.unwrap_or_default();
                            let raw = func.arguments.unwrap_or_default();
                            let arguments = serde_json::from_str::<serde_json::Value>(&raw)
                                .unwrap_or_else(|_| {
                                    out.tool_args_unparsed = true;
                                    serde_json::Value::from(raw)
                                });
                            parts.push(tool_call_part(tc.id, name, arguments));
                        }
                    }
                }
                if let Some(audio) = msg.audio {
                    if let Some(data) = audio.get("data").and_then(|x| x.as_str()) {
                        parts.push(blob_part("audio", None, data));
                    }
                }
            }
            msgs.push(OutMsg {
                role: None,
                parts,
                finish_reason,
            });
        }
        if !fr.is_empty() {
            out.finish_reasons = Some(fr);
        }
        out.error_type = declared;
        if out.output_messages.is_none() {
            out.output_messages = build_output_messages(msgs);
        }
        if let Some(u) = r.usage {
            // OpenAI `prompt_tokens` already contains `cached_tokens`, and
            // `completion_tokens` already contains the reasoning tokens.
            let view = UsageView(&u);
            out.usage = TokenUsage::new(
                InputConvention::Inclusive,
                view.i64_at(P_INPUT),
                view.i64_at(P_OUTPUT),
                view.i64_at(P_CACHE_READ),
                None,
                view.i64_at(P_REASONING),
            );
            let flat = flatten_usage(&u, bounds);
            out.usage_leaves = flat.leaves;
            out.usage_dropped_count = flat.dropped;
        }
        if let Some(v) = r.service_tier {
            out.response_service_tier = Some(v);
        }
        if let Some(v) = r.system_fingerprint {
            out.system_fingerprint = Some(v);
        }
    }
    if let Ok(q) = serde_json::from_slice::<OpenAIChatRequest>(req) {
        out.request_model = q.model;
        out.temperature = q.temperature;
        out.max_tokens = q.max_tokens.or(q.max_completion_tokens);
        out.top_p = q.top_p;
        out.frequency_penalty = q.frequency_penalty;
        out.presence_penalty = q.presence_penalty;
        out.seed = q.seed;
        out.choice_count = q.n;
        out.stream = q.stream;
        out.stop_sequences = q.stop.map(|s| s.into_vec());
        if let Some(v) = q.service_tier {
            out.request_service_tier = Some(v);
        }
        if let Some(v) = q.reasoning_effort {
            out.reasoning_level = Some(v);
        }
        // `response_format.type`: json_schema/json_object -> "json", anything
        // else (or absent) -> "text" — the same rule the Responses parser
        // applies to `text.format.type`.
        out.output_type = Some(output_type_from_format(q.response_format.as_ref()).to_string());
        if let Some(messages) = q.messages {
            fill_openai_input(out, &messages);
        }
    }
}

/// `response_format.type` / `text.format.type` -> `gen_ai.output.type`.
pub(super) fn output_type_from_format(format: Option<&serde_json::Value>) -> &'static str {
    let ty = format
        .and_then(|f| f.get("type"))
        .and_then(|t| t.as_str())
        .unwrap_or("text");
    match ty {
        "json_schema" | "json_object" => "json",
        _ => "text",
    }
}

/// OpenAI chat stream chunks → synthetic non-streaming-shaped JSON.
/// Accumulates tool_call deltas into a BTreeMap keyed by index, concatenating the arguments string,
/// then emits them as the choices[0].message.tool_calls array → fill_openai_chat reuses it for extraction.
pub(super) fn reassemble_openai(events: &[SseEvent]) -> Reassembled {
    use std::collections::BTreeMap;
    let mut id: Option<String> = None;
    let mut model: Option<String> = None;
    let mut content = String::new();
    let mut finish: Option<String> = None;
    let mut usage: Option<serde_json::Value> = None;
    // index -> (id, name, arguments(accumulated))
    let mut tool_calls: BTreeMap<i64, (Option<String>, Option<String>, String)> = BTreeMap::new();
    // Terminal = `[DONE]` OR any chunk with a non-null finish_reason: an
    // OpenAI-compatible gateway that omits `[DONE]` must not read as an
    // unterminated stream when its last chunk said why it stopped.
    let mut terminated = false;
    // The in-stream failure: a chunk whose top-level `error` is the
    // provider's error object (the shape the OpenAI SDK raises `APIError`
    // from mid-stream). The only bytes that name the failure, so they are
    // kept for the synthetic body, and a terminal form: the provider said
    // why the stream ends. Such a chunk carries no choice, so only a chunk
    // without one is asked — the ordinary chunk pays nothing for this.
    let mut error: Option<serde_json::Value> = None;
    for ev in events {
        if ev.data.trim() == "[DONE]" {
            terminated = true;
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(&ev.data) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if id.is_none() {
            id = v.get("id").and_then(|x| x.as_str()).map(str::to_string);
        }
        if model.is_none() {
            model = v.get("model").and_then(|x| x.as_str()).map(str::to_string);
        }
        if let Some(c0) = v
            .get("choices")
            .and_then(|c| c.as_array())
            .and_then(|a| a.first())
        {
            if let Some(t) = c0
                .get("delta")
                .and_then(|d| d.get("content"))
                .and_then(|x| x.as_str())
            {
                content.push_str(t);
            }
            if let Some(tcs) = c0
                .get("delta")
                .and_then(|d| d.get("tool_calls"))
                .and_then(|x| x.as_array())
            {
                for tc in tcs {
                    let idx = tc.get("index").and_then(|x| x.as_i64()).unwrap_or(0);
                    let entry = tool_calls.entry(idx).or_default();
                    if entry.0.is_none() {
                        if let Some(i) = tc.get("id").and_then(|x| x.as_str()) {
                            entry.0 = Some(i.to_string());
                        }
                    }
                    if let Some(f) = tc.get("function") {
                        if entry.1.is_none() {
                            if let Some(n) = f.get("name").and_then(|x| x.as_str()) {
                                entry.1 = Some(n.to_string());
                            }
                        }
                        if let Some(a) = f.get("arguments").and_then(|x| x.as_str()) {
                            entry.2.push_str(a);
                        }
                    }
                }
            }
            if let Some(fr) = c0.get("finish_reason").and_then(|x| x.as_str()) {
                finish = Some(fr.to_string());
                terminated = true;
            }
        } else if let Some(e) = v.get("error").filter(|e| !e.is_null()) {
            error = Some(e.clone());
            terminated = true;
        }
        if let Some(u) = v.get("usage") {
            if !u.is_null() {
                usage = Some(u.clone());
            }
        }
    }
    let mut message = serde_json::json!({"role": "assistant", "content": content});
    if !tool_calls.is_empty() {
        let arr: Vec<serde_json::Value> = tool_calls
            .into_iter()
            .map(|(_, (tc_id, name, args))| {
                serde_json::json!({
                    "id": tc_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                })
            })
            .collect();
        message["tool_calls"] = serde_json::Value::Array(arr);
    }
    let mut obj = serde_json::json!({
        "id": id,
        "model": model,
        "choices": [{
            "message": message,
            "finish_reason": finish,
        }],
    });
    if let Some(u) = usage {
        obj["usage"] = u;
    }
    if let Some(e) = error {
        // Preserved, not dropped: the fill maps it to `finish_reasons=
        // ["error"]` and `error_type`, the provider's own declaration.
        obj["error"] = e;
    }
    Reassembled {
        body: serde_json::to_vec(&obj).unwrap_or_default(),
        terminated,
    }
}

#[cfg(test)]
mod tests {
    use super::super::{parse_llm, LlmSemantics};
    use wardex_limits::Limits;

    const ERROR_REQUEST: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llm/openai_chat_sse_error/request.json"
    ));
    const ERROR_STREAM: &[u8] = include_bytes!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/llm/openai_chat_sse_error/stream.sse"
    ));

    fn chat(req: &[u8], resp: &[u8]) -> LlmSemantics {
        parse_llm(
            "api.openai.com",
            "/v1/chat/completions",
            req,
            resp,
            Limits::default(),
        )
        .expect("a chat endpoint always parses")
    }

    fn json(bytes: Option<&[u8]>) -> serde_json::Value {
        serde_json::from_slice(bytes.unwrap_or_default()).unwrap_or_default()
    }

    /// A stream that ends in a top-level `error` chunk: the error object (the
    /// only bytes that name the failure) survives into the synthetic body, the
    /// text that arrived before it is kept, and the finish is the provider's
    /// own failure declaration, normalized — not a silent drop.
    #[test]
    fn an_error_chunk_is_kept_in_the_body_and_becomes_the_finish() {
        let s = chat(ERROR_REQUEST, ERROR_STREAM);
        assert_eq!(s.stream_terminated, Some(true), "the error ends the stream");
        assert_eq!(s.finish_reasons, Some(vec!["error".to_string()]));
        assert_eq!(s.error_type.as_deref(), Some("server_error"));
        assert_eq!(
            s.usage.output_tokens(),
            None,
            "no usage arrived; none is invented"
        );
        let body = json(s.decoded_response.as_deref());
        assert_eq!(body["error"]["type"], "server_error");
        assert_eq!(
            body["error"]["message"],
            "The server had an error while processing your request."
        );
        let msgs = json(s.output_messages.as_deref().map(str::as_bytes));
        assert_eq!(msgs.as_array().map(Vec::len), Some(1), "{msgs}");
        assert_eq!(msgs[0]["finish_reason"], "error");
        assert_eq!(msgs[0]["parts"][0]["content"], "Hello");
    }

    /// The error object is read where it is, not where one shape puts it: a
    /// numeric `code` (OpenAI-compatible gateways put the HTTP status there),
    /// a `type` alone, or neither — declared, unclassified, still a failure.
    #[test]
    fn an_error_chunk_is_classified_by_code_then_type() {
        let head = r#"data: {"id":"c","model":"m","choices":[{"delta":{"content":"x"}}]}"#;
        for (error, expected) in [
            (
                r#"{"code":"rate_limit_exceeded","type":"requests"}"#,
                "rate_limit_exceeded",
            ),
            (r#"{"code":400,"type":"BadRequestError"}"#, "400"),
            (r#"{"code":null,"type":"server_error"}"#, "server_error"),
            (r#"{"message":"unclassified"}"#, ""),
            (r#""a bare string""#, ""),
        ] {
            let sse = format!("{head}\n\ndata: {{\"error\":{error}}}\n\n");
            let s = chat(br#"{"model":"m"}"#, sse.as_bytes());
            assert_eq!(s.error_type.as_deref(), Some(expected), "{error}");
            assert_eq!(s.finish_reasons, Some(vec!["error".to_string()]), "{error}");
        }
    }

    /// `"error": null` on an ordinary chunk is not a failure, and the stream
    /// still finishes the way it said.
    #[test]
    fn a_null_error_field_is_not_a_failure() {
        let sse = b"data: {\"id\":\"c\",\"model\":\"m\",\"error\":null,\"choices\":[{\"delta\":{\"content\":\"x\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        let s = chat(br#"{"model":"m"}"#, sse);
        assert_eq!(s.error_type, None);
        assert_eq!(s.finish_reasons, Some(vec!["stop".to_string()]));
        let body = json(s.decoded_response.as_deref());
        assert!(body.get("error").is_none(), "{body}");
    }

    /// An `error` chunk as the stream's ONLY event, before anything named an
    /// id or a model: still the provider's failure declaration. No response
    /// field names the model, so the seam keeps this call on the request's
    /// model and this declaration together; none is invented here.
    #[test]
    fn an_error_chunk_alone_still_declares_the_failure() {
        let sse = b"data: {\"error\":{\"message\":\"x\",\"type\":\"server_error\",\
                    \"param\":null,\"code\":null}}\n\n";
        let s = chat(br#"{"model":"gpt-4o","stream":true}"#, sse);
        assert_eq!(s.error_type.as_deref(), Some("server_error"));
        assert_eq!(s.finish_reasons, Some(vec!["error".to_string()]));
        assert_eq!(s.stream_terminated, Some(true));
        assert_eq!(s.request_model.as_deref(), Some("gpt-4o"));
        assert_eq!(s.response_model, None);
        assert_eq!(s.response_id, None);
    }

    /// The HTTP error envelope every 4xx/5xx carries is NOT a response
    /// object: no finish, no output message, no declared error type. That
    /// span keeps taking its error from the HTTP status.
    #[test]
    fn a_bare_error_envelope_still_claims_nothing_about_the_response() {
        let s = chat(
            br#"{"model":"m"}"#,
            br#"{"error":{"message":"slow down","type":"requests","code":"rate_limit_exceeded"}}"#,
        );
        assert_eq!(s.finish_reasons, None);
        assert_eq!(s.output_messages, None);
        assert_eq!(s.error_type, None);
    }
}
